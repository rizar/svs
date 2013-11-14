#include "components/baseapp.h"
#include "components/analysis.h"
#include "components/fastsvm.h"
#include "components/trainset.h"
#include "components/searcher.h"
#include "components/visualization.h"
#include "components/check.h"
#include "components/svs.h"

#include "pcl/common/time.h"
#include "pcl/common/distances.h"

#include "boost/program_options.hpp"

#include <iostream>

namespace po = boost::program_options;

class App : public BaseApp {
public:
    App(int argc, char* argv [])
        : Seed_(1)
        , MaxAlpha_(32)
        , KernelWidth_(5)
        , KernelThreshold_(1e-3)
        , TerminateEps_(1)
        , BorderWidth_(6)
        , TakeProb_(1 / 12.0)
        , NumFP_(10)
        , DoCheck_(false)
        , DoShowFP_(false)
        , DoShowGradientMap_(false)
        , DoShowRelLocGradMap_(false)
        , SaveScreenshot_(false)
    {
        ParseArgs(argc, argv);
    }

    void ParseArgs(int argc, char* argv []);

    int Run();

private:
    void Learn();
    void Search();

    void PrintParameters();
    void PrintStatistics();
    void PrintReport();
    void PrintGradientNorms();
    void ExportForLibSVM();

    void ShowFeaturePoints();
    void ShowGradientMap();
    void ShowRelativeLocalGradientMap();

    void SetSVMParams(BaseSVMParams * svmParams);

    void CalcGradientNormsAndCheck();
    void CalcRelativeLocalGradientNorms();

private:
    int Seed_;

    float MaxAlpha_;
    float KernelWidth_;
    float KernelThreshold_;
    float TerminateEps_;
    int BorderWidth_;
    float TakeProb_;
    int NumFP_;

    bool DoCheck_;
    bool DoShowFP_;
    bool DoShowGradientMap_;
    bool DoShowRelLocGradMap_;
    bool SaveScreenshot_;

    std::string FPReportOutputPath_;
    std::string GradientNormsOutputPath_;
    std::string SaveScreenshotPath_;
    std::string LibSVMExportPath_;

    std::string CameraDescription_;

    std::string OutputPath_;

    cv::Mat Objects_;
    cv::Mat Responses_;

    FastSVM SVM_;
    FeaturePointSearcher Searcher_;
    ModelChecker Checker_;

    float MinGradientNorm_;
    float MaxGradientNorm_;
    std::vector<float> GradientNorms_;
    std::vector<float> RelLocGradNorms_;
};

void App::ParseArgs(int argc, char* argv []) {
    po::options_description desc;
    desc.add_options()
        ("help", "show help")
        ("seed", po::value<int>(&Seed_))
        ("tprob", po::value<float>(&TakeProb_))
        ("bwidth", po::value<int>(&BorderWidth_))
        ("malpha", po::value<float>(&MaxAlpha_))
        ("kwidth", po::value<float>(&KernelWidth_))
        ("kthr", po::value<float>(&KernelThreshold_))
        ("teps", po::value<float>(&TerminateEps_))
        ("numfp", po::value<int>(&NumFP_))
        ("input-path", po::value<std::string>(&InputPath_))
        ("output-path", po::value<std::string>(&OutputPath_))
        ("docheck", po::value<bool>(&DoCheck_)->zero_tokens())
        ("showfp", po::value<bool>(&DoShowFP_)->zero_tokens())
        ("showgm", po::value<bool>(&DoShowGradientMap_)->zero_tokens())
        ("showrlgm", po::value<bool>(&DoShowRelLocGradMap_)->zero_tokens())
        ("fpreport", po::value<std::string>(&FPReportOutputPath_))
        ("gnorms", po::value<std::string>(&GradientNormsOutputPath_))
        ("camera", po::value<std::string>(&CameraDescription_))
        ("savesc", po::value<std::string>(&SaveScreenshotPath_))
        ("libsvm", po::value<std::string>(&LibSVMExportPath_));

    po::positional_options_description pos;
    pos.add("input-path", 1);
    pos.add("output-path", 2);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
            .options(desc).positional(pos).run(), vm);
    vm.notify();

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        exit(0);
    }
    if (InputPath_.empty()) {
        std::cerr << "Nothing to work with" << std::endl;
        exit(1);
    }
}

int App::Run() {
    srand(Seed_);

    Load();
    PrintParameters();
    Learn();
    Search();

    if (FPReportOutputPath_.size()) {
        PrintReport();
    }
    if (GradientNormsOutputPath_.size()) {
        PrintGradientNorms();
    }

    if (DoShowFP_) {
        ShowFeaturePoints();
    }
    if (DoShowGradientMap_) {
        ShowGradientMap();
    }
    if (DoShowRelLocGradMap_) {
        ShowRelativeLocalGradientMap();
    }

    if (OutputPath_.size()) {
        std::ofstream ofstr(OutputPath_);
        SupportVectorShape(Searcher_.FeaturePoints).SaveAsText(ofstr);
    }
    if (LibSVMExportPath_.size()) {
        ExportForLibSVM();
    }

    PrintStatistics();
    return 0;
}

void App::Learn() {
    CalcDistanceToNN();

    TrainingSetGenerator tsg(BorderWidth_, TakeProb_);
    tsg.generate(*InputNoNan_, DistToNN_, &Objects_, &Responses_);

    BaseSVMParams svmParams;
    SetSVMParams(&svmParams);

    {
        pcl::ScopeTime st("SVM");
        SVM_.train(Objects_, Responses_, svmParams);
    }
}

void App::Search() {
    CalcGradientNormsAndCheck();
    CalcRelativeLocalGradientNorms();

    pcl::ScopeTime st("Search");
    Searcher_.NumSeeds = NumFP_;
    Searcher_.MinSpaceSeeds = 10 * Resolution_;
    Searcher_.MinSpaceFP = 5 * Resolution_;
    Searcher_.OneStageSearch(SVM_, *InputNoNan_, RelLocGradNorms_);
    // Searcher_.ChooseSeeds(*InputNoNan_, GradientNorms_);
    // Searcher_.ChooseSeeds(*InputNoNan_, RelLocGradNorms_);
    // Searcher_.Search(SVM_);
}

void App::SetSVMParams(BaseSVMParams * params) {
    CvTermCriteria termCriteria;
    termCriteria.type = CV_TERMCRIT_EPS;
    termCriteria.epsilon = TerminateEps_;
    params->term_crit = termCriteria;

    params->svm_type = FastSVM::C_SVC;
    params->C = MaxAlpha_;
    params->gamma = 1 / sqr(KernelWidth_ * Resolution_);

#ifdef USE_MY_SVM
    My::kernelThreshold = KernelThreshold_;
#endif
}

void App::ExportForLibSVM() {
    std::ofstream ofstr(LibSVMExportPath_);
    for (int i = 0; i < Objects_.rows; ++i) {
        ofstr << Responses_.at<float>(i) << ' '
            << "1:" << Objects_.at<float>(i, 0) << ' '
            << "2:" << Objects_.at<float>(i, 1) << ' '
            << "3:" << Objects_.at<float>(i, 2) << '\n';
    }
}

void App::PrintParameters() {
    std::cout << "Seed is " << Seed_ << std::endl;
    std::cout << "Input cloud size is " << InputNoNan_->size() << " points" << std::endl;
    std::cout << "Resolution is " << Resolution_ << std::endl;
    std::cout << "Kernel width is " << KernelWidth_ << " resolutions" << std::endl;
    std::cout << "Kernel threshold is " << KernelThreshold_ << std::endl;
    std::cout << "Gamma is " << 1 / sqr(KernelWidth_ * Resolution_) << std::endl;
    std::cout << "Max alpha constraint is " << MaxAlpha_ << std::endl;
    std::cout << "Terminate epsilon is " << TerminateEps_ << std::endl;
    std::cout << "Border width is " << BorderWidth_ << std::endl;
    std::cout << "Take probability is " << TakeProb_ << std::endl;
    std::cout << "Number of feature points is " << NumFP_ << std::endl;
}

void App::PrintStatistics() {
    if (DoCheck_) {
        std::cout << "Accuracy is " << Checker_.Accuracy()  << std::endl;
        std::cout << Checker_.LearntRatio() << " of the shape is learnt" << std::endl;
    }
}

void App::PrintReport() {
    std::ofstream repOut(FPReportOutputPath_);
    for (int i = 0; i < Searcher_.Seeds->size(); ++i) {
        PointType seed = Searcher_.Seeds->at(i);
        PointType fp = Searcher_.FeaturePoints->at(i);
        Printer::printStateAtPoint(SVM_, seed, repOut);
        repOut << "distance in resolutions: "
               << sqrt(squaredEuclideanDistance(seed, fp)) / Resolution_ << std::endl;
        Printer::printStateAtPoint(SVM_, fp, repOut);
        repOut << "---------------" << std::endl;
    }
}

void App::PrintGradientNorms() {
    CalcGradientNormsAndCheck();
    std::ofstream ofstr(GradientNormsOutputPath_);
    for (int i = 0; i < InputNoNan_->size(); ++i) {
        ofstr << GradientNorms_[i] << std::endl;
    }
}

void App::ShowFeaturePoints() {
    TUMDataSetVisualizer viewer(CameraDescription_);
    viewer.EasyAdd(InputNoNan_, "input");
    // viewer.EasyAdd(Searcher_.Seeds, "seeds", 0, 0, 255, 5);
    viewer.EasyAdd(Searcher_.FeaturePoints, "fp", 255, 0, 0, 5);
    viewer.Run(SaveScreenshotPath_);
}

void App::ShowGradientMap() {
    CalcGradientNormsAndCheck();
    PointCloud::Ptr clone(new PointCloud(*InputNoNan_));
    for (int i = 0; i < clone->size(); ++i) {
        clone->at(i) = addTemperature(
                clone->at(i), GradientNorms_[i],
                (MinGradientNorm_ + MaxGradientNorm_) / 2, MaxGradientNorm_);
    }
    TUMDataSetVisualizer viewer(CameraDescription_);
    viewer.EasyAdd(clone, "gmap", 1);
    viewer.EasyAdd(Searcher_.FeaturePoints, "fp", 0, 255, 255, 5);
    viewer.Run(SaveScreenshotPath_);
}

void App::ShowRelativeLocalGradientMap() {
    CalcRelativeLocalGradientNorms();
    PointCloud::Ptr clone(new PointCloud(*InputNoNan_));
    for (int i = 0; i < clone->size(); ++i) {
        clone->at(i) = addTemperature(clone->at(i), RelLocGradNorms_[i], 0.5, 1);
    }
    TUMDataSetVisualizer viewer(CameraDescription_);
    viewer.EasyAdd(clone, "gmap", 1);
    viewer.EasyAdd(Searcher_.FeaturePoints, "fp", 0, 255, 255, 5);
    viewer.Run(SaveScreenshotPath_);
}

void App::CalcGradientNormsAndCheck() {
    if (GradientNorms_.size()) {
        return;
    }
    CalcDistanceToNN();
    pcl::ScopeTime st("CalcGradientNormsAndCheck");
    GradientNorms_.resize(InputNoNan_->size());
    for (int i = 0; i < InputNoNan_->size(); ++i) {
        PointType point = InputNoNan_->at(i);

        DecisionFunction df;
        SVM_.buildDecisionFunctionEstimate(point, &df);
        GradientNorms_[i] = df.gradient(point).getVector3fMap().norm();
        if (DoCheck_) {
            Checker_.Check(df, point, DistToNN_[i]);
        }
    }
    MinGradientNorm_ = quantile(GradientNorms_, 0.01);
    MaxGradientNorm_ = quantile(GradientNorms_, 0.99);
    std::cout << "Minimum gradient norm " << MinGradientNorm_ << std::endl;
    std::cout << "Maximum gradient norm " << MaxGradientNorm_ << std::endl;
}

void App::CalcRelativeLocalGradientNorms() {
    if (RelLocGradNorms_.size()) {
        return;
    }
    pcl::ScopeTime st("CalcRelativeLocalGradientNorms");

    RelLocGradNorms_.resize(InputNoNan_->size());

    CalcGradientNormsAndCheck();
    CalcDistanceToNN();

    std::vector<int> indices;
    std::vector<float> dist2;
    for (int i = 0; i < InputNoNan_->size(); ++i) {
        InputOctTree_->radiusSearch(InputNoNan_->at(i), 15 * DistToNN_[i], indices, dist2);

        int less = 0, more = 0;
        for (int j = 0; j < indices.size(); ++j) {
            int const idx = indices[j];
            if (GradientNorms_[idx] < GradientNorms_[i]) {
                less++;
            } else {
                more++;
            }
        }

        RelLocGradNorms_[i] = less / static_cast<float>(less + more);
    }
}

int main(int argc, char* argv []) {
    return App(argc, argv).Run();
}
