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
#include "pcl/features/integral_image_normal.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "boost/program_options.hpp"

#include <iostream>
#include <memory>

namespace po = boost::program_options;

class App : public BaseApp {
public:
    App(int argc, char* argv [])
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
    void ExportAlphaMap();

    void Visualize();
    void ShowFeaturePoints();
    void ShowGradientMap();
    void ShowRelativeLocalGradientMap();

    void CalcNormals();
    void CalcGradientNormsAndCheck();
    void CalcRelativeLocalGradientNorms();

private:
    int Seed_ = 1;

// algorithm params
    float MaxAlpha_ = 32;
    float KernelWidth_ = 5;
    float KernelThreshold_ = 1e-3;
    float TerminateEps_ = 1e-2;
    int BorderWidth_ = 1;
    float StepWidth_ = 1;
    float TakeProb_ = 1.0;
    int NumFP_ = 100;

// visualization flags
    bool DoVisualize_ = false;
    bool DoShowNormals_ = false;
    bool DoShowFP_ = false;
    bool DoShowGradientMap_ = false;
    bool DoShowRelLocGradMap_ = false;

// other flags
    bool DoCheck_ = false;
    bool DoLearnOld_ = false;
    bool SaveScreenshot_ = false;
    bool UseGrid_ = true;
    bool UseNormals_ = false;

// visualization parameters
    std::string CameraDescription_;

// source pathes
    std::string AlphasPath_;

// destination pathes
    std::string FPReportOutputPath_;
    std::string GradientNormsOutputPath_;
    std::string SaveScreenshotPath_;
    std::string OutputPath_;
    std::string LibSVMExportPath_;
    std::string AlphaMapPath_;

// SVM related data
    PointCloud::Ptr Objects_;
    std::vector<float> Labels_;
    GridNeighbourModificationStrategy::Grid2Num Grid2Num_;
    GridNeighbourModificationStrategy::Num2Grid Num2Grid_;
    std::vector<int> Pixel2Num_;
    std::shared_ptr<GridNeighbourModificationStrategy> Strategy_;
    FastSVM OldSVM_;
    SVM3D SVM_;
    FeaturePointSearcher Searcher_;
    ModelChecker Checker_;

// useful data
    NormalCloud::Ptr Normals_;

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
        ("stepwidth", po::value<float>(&StepWidth_))
        ("malpha", po::value<float>(&MaxAlpha_))
        ("kwidth", po::value<float>(&KernelWidth_))
        ("kthr", po::value<float>(&KernelThreshold_))
        ("teps", po::value<float>(&TerminateEps_))
        ("numfp", po::value<int>(&NumFP_))

        ("docheck", po::value<bool>(&DoCheck_)->zero_tokens())
        ("dovis", po::value<bool>(&DoVisualize_)->zero_tokens())
        ("shownr", po::value<bool>(&DoShowNormals_)->zero_tokens())
        ("showfp", po::value<bool>(&DoShowFP_)->zero_tokens())
        ("showgm", po::value<bool>(&DoShowGradientMap_)->zero_tokens())
        ("showrlgm", po::value<bool>(&DoShowRelLocGradMap_)->zero_tokens())

        ("usegrid", po::value<bool>(&UseGrid_))
        ("usenr", po::value<bool>(&UseNormals_)->zero_tokens())
        ("learnold", po::value<bool>(&DoLearnOld_)->zero_tokens())

        ("fpreport", po::value<std::string>(&FPReportOutputPath_))
        ("gnorms", po::value<std::string>(&GradientNormsOutputPath_))
        ("camera", po::value<std::string>(&CameraDescription_))
        ("savesc", po::value<std::string>(&SaveScreenshotPath_))
        ("input", po::value<std::string>(&InputPath_))
        ("alphas", po::value<std::string>(&AlphasPath_))
        ("output", po::value<std::string>(&OutputPath_))
        ("libsvm", po::value<std::string>(&LibSVMExportPath_))
        ("alphamap", po::value<std::string>(&AlphaMapPath_));

    po::positional_options_description pos;
    pos.add("input", 1);
    pos.add("output", 2);

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
    ExportForLibSVM();

    Learn();
    LearnOld();
    ExportAlphaMap();

    if (OutputPath_.size()) {
        std::ofstream ofstr(OutputPath_);
        SupportVectorShape(Searcher_.FeaturePoints).SaveAsText(ofstr);
    }

    PrintStatistics();
    Visualize();
    return 0;
}

void App::GenerateTrainingSet() {
    CalcDistanceToNN();

    TrainingSetGenerator tsg(BorderWidth_, TakeProb_, StepWidth_);
    if (UseNormals_) {
        CalcNormals();
        tsg.GenerateUsingNormals(*Input_, *Normals_, DistToNN_);
    } else {
        tsg.GenerateFromSensor(*Input_, DistToNN_);
    }

    Objects_.reset(new PointCloud(tsg.Objects));
    Labels_ = tsg.Labels;
    Num2Grid_ = tsg.Num2Grid;
    Grid2Num_ = tsg.Grid2Num;
    Pixel2Num_ = tsg.Pixel2Num;
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

    std::ifstream alphaStr(AlphasPath_.c_str());
    if (alphaStr.good()) {
        SVM_.Init(*Objects_, Labels_, alphaStr);
    } else {
        pcl::ScopeTime st("SVM");
        SVM_.Train(*Objects_, Labels_);

        alphaStr.close();
        if (AlphasPath_.size()) {
            std::ofstream writeAlphaStr(AlphasPath_.c_str());
            SVM_.SaveAlphas(writeAlphaStr);
        }
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

void App::CalcNormals() {
    if (Normals_.get()) {
        return;
    }
    Normals_.reset(new NormalCloud);

    pcl::IntegralImageNormalEstimation<PointType, NormalType> ne;
    ne.setNormalSmoothingSize(5.0f);
    ne.setInputCloud(Input_);
    ne.compute(*Normals_);
}

void App::Visualize() {
    if (! DoVisualize_) {
        return;
    }
    TUMDataSetVisualizer viewer(CameraDescription_);
    viewer.EasyAdd(Input_, "input", [this] (int i) {
            return Color(Input_->at(i));
            });
    if (DoShowNormals_) {
        CalcNormals();
        viewer.addPointCloudNormals<PointType, NormalType>(Input_, Normals_);
        viewer.EasyAdd(Input_, "input", [this] (int i) {
                    return pcl_isnan(Normals_->at(i).normal_x) ? Color({0, 0, 255}) : Color({255, 0, 0});
                });
    }
    viewer.EasyAdd(Input_, "input", [this] (int i) {
                int num = Pixel2Num_[i];
                if (num == -1) {
                    return Color();
                }
                float alpha = SVM_.Alphas()[num];
                return Color({static_cast<int>(alpha * 255), 0, static_cast<int>((1 - alpha) * 255)});
            });
    viewer.Run(SaveScreenshotPath_);
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

void App::ExportAlphaMap() {
    if (AlphaMapPath_.empty()) {
        return;
    }

    // works only for non-randomly generated training set
    cv::Mat image(Input_->height, Input_->width, CV_8UC3);
    for (int i = 0; i < Input_->height; ++i) {
        for (int j = 0; j < Input_->width; ++j) {
            std::vector<int> const& nums = Grid2Num_[i][j];
            if (nums.empty()) {
                image.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
                continue;
            }

            float const relAlpha = SVM_.Alphas()[nums[0]] / MaxAlpha_;
            image.at<cv::Vec3b>(i, j) = cv::Vec3b(255 * (1 - relAlpha), 0, 255 * relAlpha);
        }
    }

    imwrite(AlphaMapPath_, image);
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
    std::cout << "Step width is " << StepWidth_ << std::endl;
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
