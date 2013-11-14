#include "components/baseapp.h"
#include "components/analysis.h"
#include "components/fastsvm.h"
#include "components/newsvm.h"
#include "components/trainset.h"
#include "components/searcher.h"
#include "components/visualization.h"
#include "components/check.h"
#include "components/svs.h"

#include "utilities/prettyprint.hpp"

#include "pcl/common/time.h"
#include "pcl/common/distances.h"

#include "boost/program_options.hpp"

#include <iostream>
#include <memory>

namespace po = boost::program_options;

class App : public BaseApp {
public:
    App(int argc, char* argv [])
        : Seed_(1)
        , MaxAlpha_(32)
        , KernelWidth_(5)
        , KernelThreshold_(1e-5)
        , TerminateEps_(1e-2)
        , BorderWidth_(1)
        , TakeProb_(1.0)
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
    void GenerateTrainingSet();
    void Learn();
    void LearnOld();
    void Search();

    void PrintParameters();
    void PrintStatistics();
    void PrintReport();
    void PrintGradientNorms();

    void ExportForLibSVM();

    void ShowFeaturePoints();
    void ShowGradientMap();
    void ShowRelativeLocalGradientMap();

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
    bool DoLearnOld_ = false;
    bool SaveScreenshot_;
    bool UseGrid_ = true;

    std::string CameraDescription_;

    std::string FPReportOutputPath_;
    std::string GradientNormsOutputPath_;
    std::string SaveScreenshotPath_;
    std::string OutputPath_;
    std::string LibSVMExportPath_;

    PointCloud::Ptr Objects_;
    std::vector<float> Labels_;
    GridNeighbourModificationStrategy::Grid2Num Grid2Num_;
    GridNeighbourModificationStrategy::Num2Grid Num2Grid_;
    std::shared_ptr<GridNeighbourModificationStrategy> Strategy_;
    FastSVM OldSVM_;
    SVM3D SVM_;
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
        ("docheck", po::value<bool>(&DoCheck_)->zero_tokens())
        ("showfp", po::value<bool>(&DoShowFP_)->zero_tokens())
        ("showgm", po::value<bool>(&DoShowGradientMap_)->zero_tokens())
        ("showrlgm", po::value<bool>(&DoShowRelLocGradMap_)->zero_tokens())
        ("usegrid", po::value<bool>(&UseGrid_))
        ("learnold", po::value<bool>(&DoLearnOld_)->zero_tokens())
        ("fpreport", po::value<std::string>(&FPReportOutputPath_))
        ("gnorms", po::value<std::string>(&GradientNormsOutputPath_))
        ("camera", po::value<std::string>(&CameraDescription_))
        ("savesc", po::value<std::string>(&SaveScreenshotPath_))
        ("input-path", po::value<std::string>(&InputPath_))
        ("output-path", po::value<std::string>(&OutputPath_))
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
    GenerateTrainingSet();
    Learn();
    LearnOld();
    PrintStatistics();

    if (OutputPath_.size()) {
        std::ofstream ofstr(OutputPath_);
        SupportVectorShape(Searcher_.FeaturePoints).SaveAsText(ofstr);
    }

    ExportForLibSVM();

    return 0;
}

void App::GenerateTrainingSet() {
    CalcDistanceToNN();

    TrainingSetGenerator tsg(BorderWidth_, TakeProb_);
    tsg.Generate(*Input_, DistToNN_);

    Objects_.reset(new PointCloud(tsg.Objects));
    Labels_ = tsg.Labels;
    Num2Grid_ = tsg.Num2Grid;
    Grid2Num_ = tsg.Grid2Num;
}

void App::Learn() {
    if (UseGrid_) {
        Strategy_.reset(new GridNeighbourModificationStrategy(
                    Input_->height, Input_->width,
                    Grid2Num_, Num2Grid_,
                    KernelWidth_, KernelThreshold_, Resolution_,
                    1 << 30));
        SVM_.SetStrategy(Strategy_);
    }
    SVM_.SetParams(MaxAlpha_, 1 / sqr(KernelWidth_ * Resolution_), TerminateEps_);

    {
        pcl::ScopeTime st("SVM");
        SVM_.Train(*Objects_, Labels_);
    }
}

void App::LearnOld() {
    if (! DoLearnOld_) {
        return;
    }

    BaseSVMParams params;
    params.C = MaxAlpha_;
    params.gamma = 1 / sqr(KernelWidth_ * Resolution_);
    params.term_crit.type = CV_TERMCRIT_EPS;
    params.term_crit.epsilon = TerminateEps_;
    My::kernelThreshold = KernelThreshold_;
    {
        pcl::ScopeTime st("Old SVM");
        OldSVM_.train(*Objects_, Labels_, params);
    }
}

void App::ExportForLibSVM() {
    std::ofstream ofstr(LibSVMExportPath_);
    for (int i = 0; i < Objects_->size(); ++i) {
        ofstr << Labels_[i] << ' '
              << "1:" << Objects_->at(i).x << ' '
              << "2:" << Objects_->at(i).y << ' '
              << "3:" << Objects_->at(i).z << '\n';
    }
}

void App::PrintParameters() {
    std::cout << "-------------------- PARAMETERS" << std::endl;
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
    std::cout << "-------------------- STATISTICS" << std::endl;
    std::cout << "New SVM converged in " << SVM_.Iteration << " iterations" << std::endl;
    std::cout << Objects_->size() << " input vectors" << std::endl;
    std::cout << SVM_.SVCount << " support vectors" << std::endl;
    std::cout << SVM_.TargetFunction << " target function" << std::endl;
    if (UseGrid_) {
        Strategy_->PrintStatistics(std::cout);
    }
}

int main(int argc, char* argv []) {
    return App(argc, argv).Run();
}
