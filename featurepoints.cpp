#include "analysis.h"
#include "fastsvm.h"
#include "trainset.h"
#include "searcher.h"
#include "visualization.h"
#include "check.h"
#include "svs.h"

#include "pcl/io/pcd_io.h"
#include "pcl/common/time.h"
#include "pcl/common/distances.h"
#include "pcl/filters/filter.h"

#include "boost/program_options.hpp"

#include <iostream>

namespace po = boost::program_options;

class App {
private:
    typedef pcl::octree::OctreePointCloudSearch<PointType> OctTreeType;

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
        , DoShowFP_(false)
        , DoShowGradientMap_(false)
    {
        ParseArgs(argc, argv);
    }

    void ParseArgs(int argc, char* argv []);

    int Run();

private:
    void Load();
    void Learn();
    void Search();

    void PrintParameters();
    void PrintStatistics();
    void PrintReport();
    void PrintGradientNorms();

    void ShowFeaturePoints();
    void ShowGradientMap();
    void ShowRelativeLocalGradientMap();

    void SetSVMParams(BaseSVMParams * svmParams);

    void CalcGradientNormsAndCheck();
    void CalcRelativeLocalGradientNorms();
    void BuildOctTree();
    void CalcDistanceToNN();

private:
    int Seed_;

    float MaxAlpha_;
    float KernelWidth_;
    float KernelThreshold_;
    float TerminateEps_;
    int BorderWidth_;
    float TakeProb_;
    int NumFP_;

    bool DoShowFP_;
    bool DoShowGradientMap_;
    bool DoShowRelLocGradMap_;
    bool SaveScreenshot_;

    std::string FPReportOutputPath_;
    std::string GradientNormsOutputPath_;
    std::string SaveScreenshotPath_;

    std::string CameraDescription_;

    std::string InputPath_;
    std::string OutputPath_;

    float Resolution_;

    PointCloud::Ptr Input_;
    PointCloud::Ptr InputNoNan_;
    pcl::search::KdTree<PointType>::Ptr InputKDTree_;
    OctTreeType::Ptr InputOctTree_;

    FastSVM SVM_;
    FeaturePointSearcher Searcher_;
    ModelChecker Checker_;

    float MinGradientNorm_;
    float MaxGradientNorm_;
    std::vector<float> GradientNorms_;
    std::vector<float> RelLocGradNorms_;

    std::vector<float> DistToNN_;
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
        ("showfp", po::value<bool>(&DoShowFP_)->zero_tokens())
        ("showgm", po::value<bool>(&DoShowGradientMap_)->zero_tokens())
        ("showrlgm", po::value<bool>(&DoShowRelLocGradMap_)->zero_tokens())
        ("fpreport", po::value<std::string>(&FPReportOutputPath_))
        ("gnorms", po::value<std::string>(&GradientNormsOutputPath_))
        ("camera", po::value<std::string>(&CameraDescription_))
        ("savesc", po::value<std::string>(&SaveScreenshotPath_));

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

    PrintStatistics();
    return 0;
}

void App::Load() {
    Input_.reset(new PointCloud);
    InputNoNan_.reset(new PointCloud);
    pcl::io::loadPCDFile(InputPath_, *Input_);
    std::vector<int> tmp;
    pcl::removeNaNFromPointCloud(*Input_, *InputNoNan_, tmp);

    InputKDTree_.reset(new pcl::search::KdTree<PointType>);
    InputKDTree_->setInputCloud(InputNoNan_);
    Resolution_ = computeCloudResolution(InputNoNan_, *InputKDTree_);
}

void App::Learn() {
    TrainingSetGenerator tsg(Resolution_, BorderWidth_, TakeProb_);
    cv::Mat objects;
    cv::Mat responses;
    tsg.generate(*Input_, &objects, &responses);

    BaseSVMParams svmParams;
    SetSVMParams(&svmParams);

    {
        pcl::ScopeTime st("SVM");
        SVM_.train(objects, responses, svmParams);
    }
}

void App::Search() {
    CalcGradientNormsAndCheck();
    CalcRelativeLocalGradientNorms();

    Searcher_.NumSeeds = NumFP_;
    Searcher_.MinSpace = 10 * Resolution_;
    // Searcher_.ChooseSeeds(*InputNoNan_, GradientNorms_);
    Searcher_.ChooseSeeds(*InputNoNan_,RelLocGradNorms_);
    Searcher_.Search(SVM_);
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
    std::cout << "Accuracy is " << Checker_.Accuracy()  << std::endl;
    std::cout << Checker_.LearntRatio() << " of the shape is learnt" << std::endl;
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
    viewer.EasyAdd(Searcher_.Seeds, "seeds", 0, 0, 255, 5);
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
        Checker_.Check(df, point, DistToNN_[i]);
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
    BuildOctTree();

    std::vector<int> indices;
    std::vector<float> dist2;
    for (int i = 0; i < InputNoNan_->size(); ++i) {
        InputOctTree_->radiusSearch(InputNoNan_->at(i), 15 * Resolution_, indices, dist2);

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

void App::CalcDistanceToNN() {
    if (DistToNN_.size()) {
        return;
    }
    pcl::ScopeTime st("CalcDistanceToNN");

    std::vector<int> indices;
    std::vector<float> dist2;
    DistToNN_.resize(InputNoNan_->size());
    for (int i = 0; i < InputNoNan_->size(); ++i) {
        PointType const& point = InputNoNan_->at(i);
        InputKDTree_->nearestKSearch(i, 2, indices, dist2);
        DistToNN_[i] = sqrt(dist2[1]);
    }
}

void App::BuildOctTree() {
    if (InputOctTree_.get()) {
        return;
    }
    InputOctTree_.reset(new OctTreeType(10 * Resolution_));
    InputOctTree_->setInputCloud(InputNoNan_);
    InputOctTree_->addPointsFromInputCloud();
}

int main(int argc, char* argv []) {
    return App(argc, argv).Run();
}
