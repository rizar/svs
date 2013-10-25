#include "analysis.h"
#include "fastsvm.h"
#include "trainset.h"
#include "searcher.h"
#include "visualization.h"

#include "pcl/io/pcd_io.h"
#include "pcl/common/time.h"
#include "pcl/common/distances.h"
#include "pcl/filters/filter.h"

#include "boost/program_options.hpp"

#include <iostream>

namespace po = boost::program_options;
using My::CvSVMParams;

class App {
public:
    App(int argc, char* argv [])
        : Seed_(1)
        , MaxAlpha_(32)
        , KernelWidth_(5)
        , KernelThreshold_(1e-4)
        , TerminateEps_(1)
        , BorderWidth_(6)
        , TakeProb_(1 / 12.0)
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
    void PrintReport();
    void PrintGradientNorms();

    void ShowFeaturePoints();
    void ShowGradientMap();

    void SetSVMParams(CvSVMParams * svmParams);

    void CalcGradientNorms();

private:
    int Seed_;

    float MaxAlpha_;
    float KernelWidth_;
    float KernelThreshold_;
    float TerminateEps_;
    int BorderWidth_;
    float TakeProb_;

    bool DoShowFP_;
    bool DoShowGradientMap_;
    bool SaveScreenshot_;

    std::string FPReportOutputPath_;
    std::string GradientNormsOutputPath_;
    std::string SaveScreenshotPath_;

    std::string CameraDescription_;

    std::string InputPath_;

    float Resolution_;

    PointCloud::Ptr Input_;
    PointCloud::Ptr InputNoNan_;
    pcl::search::KdTree<PointType>::Ptr InputKDTree_;

    FastSVM SVM_;
    FeaturePointSearcher Searcher_;

    float MinGradientNorm_;
    float MaxGradientNorm_;
    std::vector<float> GradientNorms_;
};

void App::ParseArgs(int argc, char* argv []) {
    po::options_description desc;
    desc.add_options()
        ("seed", po::value<int>(&Seed_))
        ("tprob", po::value<float>(&TakeProb_))
        ("bwidth", po::value<int>(&BorderWidth_))
        ("malpha", po::value<float>(&MaxAlpha_))
        ("kwidth", po::value<float>(&KernelWidth_))
        ("kthr", po::value<float>(&KernelThreshold_))
        ("teps", po::value<float>(&TerminateEps_))
        ("path", po::value<std::string>(&InputPath_)->required())
        ("showfp", po::value<bool>(&DoShowFP_)->zero_tokens())
        ("showgm", po::value<bool>(&DoShowGradientMap_)->zero_tokens())
        ("fpreport", po::value<std::string>(&FPReportOutputPath_))
        ("gnorms", po::value<std::string>(&GradientNormsOutputPath_))
        ("camera", po::value<std::string>(&CameraDescription_))
        ("savesc", po::value<std::string>(&SaveScreenshotPath_));

    po::positional_options_description pos;
    pos.add("path", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
            .options(desc).positional(pos).run(), vm);
    vm.notify();
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

    return 0;
}

void App::Load() {
    Input_.reset(new PointCloud);
    InputNoNan_.reset(new PointCloud);
    pcl::io::loadPCDFile(InputPath_, *Input_);
    std::vector<int> tmp;
    pcl::removeNaNFromPointCloud(*Input_, *InputNoNan_, tmp);

    InputKDTree_.reset(new pcl::search::KdTree<PointType>);
    InputKDTree_->setInputCloud(Input_);
    Resolution_ = computeCloudResolution(Input_, *InputKDTree_);
}

void App::Learn() {
    TrainingSetGenerator tsg(Resolution_, BorderWidth_, TakeProb_);
    cv::Mat objects;
    cv::Mat responses;
    tsg.generate(*Input_, &objects, &responses);

    CvSVMParams svmParams;
    SetSVMParams(&svmParams);

    {
        pcl::ScopeTime st("SVM");
        SVM_.train(objects, responses, svmParams);
    }
}

void App::Search() {
    Searcher_.NumSeeds = 100;
    Searcher_.Search(SVM_, *InputNoNan_);
}

void App::SetSVMParams(CvSVMParams * params) {
    CvTermCriteria termCriteria;
    termCriteria.type = CV_TERMCRIT_EPS;
    termCriteria.epsilon = TerminateEps_;
    params->term_crit = termCriteria;

    params->svm_type = FastSVM::C_SVC;
    params->C = MaxAlpha_;
    params->gamma = 1 / sqr(KernelWidth_ * Resolution_);

    // VERY UGLY
    My::kernelThreshold = KernelThreshold_;
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
    if (GradientNorms_.empty()) {
        CalcGradientNorms();
    }
    std::ofstream ofstr(GradientNormsOutputPath_);
    for (int i = 0; i < InputNoNan_->size(); ++i) {
        ofstr << GradientNorms_[i] << std::endl;
    }
}

void App::ShowFeaturePoints() {
    TUMDataSetVisualizer viewer(CameraDescription_);
    viewer.EasyAdd(InputNoNan_, "input");
    viewer.EasyAdd(Searcher_.Seeds, "seeds", 0, 0, 255, 3);
    viewer.EasyAdd(Searcher_.FeaturePoints, "fp", 255, 0, 0, 3);
    viewer.Run(SaveScreenshotPath_);
}

void App::ShowGradientMap() {
    if (GradientNorms_.empty()) {
        CalcGradientNorms();
    }
    PointCloud::Ptr clone(new PointCloud(*InputNoNan_));
    for (int i = 0; i < clone->size(); ++i) {
        clone->at(i) = addTemperature(
                clone->at(i), GradientNorms_[i],
                (MinGradientNorm_ + MaxGradientNorm_) / 2, MaxGradientNorm_);
    }
    TUMDataSetVisualizer viewer(CameraDescription_);
    viewer.EasyAdd(clone, "gmap", 3);
    viewer.Run(SaveScreenshotPath_);
}

void App::CalcGradientNorms() {
    MinGradientNorm_ = std::numeric_limits<float>::max();
    MaxGradientNorm_ = 0.0;
    GradientNorms_.resize(InputNoNan_->size());
    for (int i = 0; i < InputNoNan_->size(); ++i) {
        PointType point = InputNoNan_->at(i);
        DecisionFunction df;
        SVM_.buildDecisionFunctionEstimate(point, &df);
        GradientNorms_[i] = df.gradient(point).getVector3fMap().norm();
        MinGradientNorm_ = std::min(MinGradientNorm_, GradientNorms_[i]);
        MaxGradientNorm_ = std::max(MaxGradientNorm_, GradientNorms_[i]);
    }
    std::cout << "Minimum gradient norm " << MinGradientNorm_ << std::endl;
    std::cout << "Maximum gradient norm " << MaxGradientNorm_ << std::endl;
}

int main(int argc, char* argv []) {
    return App(argc, argv).Run();
}
