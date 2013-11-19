#include "components/visualization.h"
#include "components/svs.h"

#include <pcl/io/pcd_io.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "boost/program_options.hpp"

#include <iostream>
#include <memory>

namespace po = boost::program_options;

class App {
public:
    App(int argc, char* argv [])
    {
        ParseArgs(argc, argv);
    }

    void ParseArgs(int argc, char* argv []);

    int Run();

private:
    void Load();

    void PrintParameters();
    void PrintStatistics();
    void PrintReport();
    void PrintGradientNorms();

    void ExportForLibSVM();
    void ExportAlphaMap();

    void Visualize();

private:
    PointCloud::Ptr Input_;

    SVSParams Params_;
    SVSBuilder Builder_;

// visualization flags
    bool DoVisualize_ = false;
    bool DoShowNormals_ = false;
    bool DoShowGradientNorm_ = false;

// other flags
    bool DoLearnOld_ = false;
    bool SaveScreenshot_ = false;

// visualization parameters
    std::string CameraDescription_;

// pathes
    std::string InputPath_;
    std::string OutputPath_;
    std::string FPReportOutputPath_;
    std::string GradientNormsOutputPath_;
    std::string SaveScreenshotPath_;
    std::string LibSVMExportPath_;
    std::string AlphaMapPath_;
};

void App::ParseArgs(int argc, char* argv []) {
    po::options_description desc;
    desc.add_options()
        ("help", "show help")

        ("seed", po::value<int>(&Params_.Seed))
        ("tprob", po::value<float>(&Params_.TakeProb))
        ("bwidth", po::value<int>(&Params_.BorderWidth))
        ("stepwidth", po::value<float>(&Params_.StepWidth))
        ("malpha", po::value<float>(&Params_.MaxAlpha))
        ("kwidth", po::value<float>(&Params_.KernelWidth))
        ("kthr", po::value<float>(&Params_.KernelThreshold))
        ("teps", po::value<float>(&Params_.TerminateEps))
        ("numfp", po::value<int>(&Params_.NumFP))
        ("usegrid", po::value<bool>(&Params_.UseGrid))
        ("usenr", po::value<bool>(&Params_.UseNormals)->zero_tokens())
        ("alphas", po::value<std::string>(&Params_.AlphasPath))

        ("learnold", po::value<bool>(&DoLearnOld_)->zero_tokens())

        ("dovis", po::value<bool>(&DoVisualize_)->zero_tokens())
        ("shownr", po::value<bool>(&DoShowNormals_)->zero_tokens())

        ("fpreport", po::value<std::string>(&FPReportOutputPath_))
        ("gnorms", po::value<std::string>(&GradientNormsOutputPath_))
        ("camera", po::value<std::string>(&CameraDescription_))
        ("savesc", po::value<std::string>(&SaveScreenshotPath_))
        ("input", po::value<std::string>(&InputPath_))
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
    Load();
    Builder_.SetParams(Params_);
    Builder_.SetInputCloud(Input_);
    PrintParameters();

    Builder_.GenerateTrainingSet();
    ExportForLibSVM();

    Builder_.Learn();
    if (DoLearnOld_) {
        Builder_.LearnOld();
    }
    ExportAlphaMap();

    Builder_.CalcGradientNorms();

    PrintStatistics();
    Visualize();
    return 0;
}

void App::Load() {
    Input_.reset(new PointCloud);
    pcl::io::loadPCDFile(InputPath_, *Input_);
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
        Builder_.CalcNormals();
        NormalCloud::ConstPtr normals = Builder_.Normals;
        viewer.addPointCloudNormals<PointType, NormalType>(Input_, normals);
        viewer.EasyAdd(Input_, "input", [this, &normals] (int i) {
                    return pcl_isnan(normals->at(i).normal_x)
                        ? Color({0, 0, 255})
                        : Color({255, 0, 0});
                });
    }
    viewer.EasyAdd(Input_, "input", [this] (int i) {
                int const num = Builder_.Pixel2TrainNum[i];
                if (num == -1) {
                    return Color();
                }
                float const alpha = Builder_.SVM().Alphas()[num];
                return Color({static_cast<int>(alpha * 255), 0, static_cast<int>((1 - alpha) * 255)});
            });
    viewer.EasyAdd(Input_, "input", [this] (int i) {
                int const num = Builder_.Pixel2RowIndex[i];
                if (num == -1) {
                    return Color();
                }
                float const gn = Builder_.GradientNorm[num];
                float const relGN = gn / Builder_.MaxGradientNorm;
                return Color({static_cast<int>(relGN * 255), 0, static_cast<int>((1 - relGN) * 255)});
            });
    viewer.Run(SaveScreenshotPath_);
}

// ALL KINDS OF OUTPUT

void App::ExportForLibSVM() {
    std::ofstream ofstr(LibSVMExportPath_);
    for (int i = 0; i < Builder_.Objects->size(); ++i) {
        PointType const& point = Builder_.Objects->at(i);
        ofstr << Builder_.Labels[i] << ' '
              << "1:" << point.x << ' '
              << "2:" << point.y << ' '
              << "3:" << point.z << '\n';
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
            int num = Builder_.Pixel2TrainNum[i * Input_->width + j];
            if (num == -1) {
                image.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
                continue;
            }

            float const relAlpha = Builder_.SVM().Alphas()[num] / Params_.MaxAlpha;
            image.at<cv::Vec3b>(i, j) = cv::Vec3b(255 * (1 - relAlpha), 0, 255 * relAlpha);
        }
    }

    imwrite(AlphaMapPath_, image);
}

void App::PrintParameters() {
    std::cout << "-------------------- PARAMETERS" << std::endl;
    std::cout << "Seed is " << Params_.Seed << std::endl;
    std::cout << "Input cloud size is " << Builder_.InputNoNan->size() << " points" << std::endl;
    std::cout << "Resolution is " << Builder_.Resolution << std::endl;
    std::cout << "Kernel width is " << Params_.KernelWidth << " resolutions" << std::endl;
    std::cout << "Kernel threshold is " << Params_.KernelThreshold << std::endl;
    std::cout << "Gamma is " << Builder_.Gamma << std::endl;
    std::cout << "Max alpha constraint is " << Params_.MaxAlpha << std::endl;
    std::cout << "Terminate epsilon is " << Params_.TerminateEps << std::endl;
    std::cout << "Border width is " << Params_.BorderWidth << std::endl;
    std::cout << "Step width is " << Params_.StepWidth << std::endl;
    std::cout << "Take probability is " << Params_.TakeProb << std::endl;
    std::cout << "Number of feature points is " << Params_.NumFP << std::endl;
}

void App::PrintStatistics() {
    GridNeighbourModificationStrategy const& strat = *Builder_.Strategy;

    std::cout << "-------------------- STATISTICS" << std::endl;
    std::cout << "New SVM converged in " << Builder_.SVM().Iteration << " iterations" << std::endl;
    std::cout << "SVM3D: " << Builder_.Objects->size() << " input vectors" << std::endl;
    std::cout << "SVM3D: " << Builder_.SVM().SVCount << " support vectors" << std::endl;
    std::cout << "SVM3D: " << Builder_.SVM().TargetFunction << " target function" << std::endl;
    std::cout << "GridStrategy: Number of cache misses: " << strat.NumCacheMisses << std::endl;
    std::cout << "GridStrategy: Number of optimization failures: " << strat.NumOptimizeFailures << std::endl;
    std::cout << "GridStrategy: Average number of neighbors: " <<
        static_cast<float>(strat.TotalNeighborsProcessed) / strat.NumNeighborsCalculations << std::endl;
    std::cout << "SVSBuilder: Minimum gradient norm " << Builder_.MinGradientNorm << std::endl;
    std::cout << "SVSBuilder: Maximum gradient norm " << Builder_.MaxGradientNorm << std::endl;
}

int main(int argc, char* argv []) {
    return App(argc, argv).Run();
}
