#include "components/visualization.h"
#include "components/svs.h"

#include "utilities/prettyprint.hpp"

#include <pcl/io/pcd_io.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "boost/program_options.hpp"

#include <iostream>
#include <memory>
#include <iterator>

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
    bool DoShowGradients_ = false;

// other flags
    bool DoLearnOld_ = false;
    bool SaveScreenshot_ = false;

// visualization parameters
    std::string CameraDescription_;

// source pathes
    std::string InputPath_;
    std::string AlphaInputPath_;
    std::string ParamsInputPath_;

// dest pathes
    std::string OutputPath_;
    std::string AlphaOutputPath_;
    std::string ParamsOutputPath_;
    std::string FPReportOutputPath_;
    std::string GradientNormsOutputPath_;
    std::string SaveScreenshotPath_;
    std::string LibSVMExportPath_;
    std::string AlphaMapPath_;

// status
    bool TrainedModel_ = false;
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
        ("smrange", po::value<float>(&Params_.SmoothingRange))
        ("numfp", po::value<int>(&Params_.NumFP))
        ("usegrid", po::value<bool>(&Params_.UseGrid))
        ("usenr", po::value<bool>(&Params_.UseNormals)->zero_tokens())

        ("dovis", po::value<bool>(&DoVisualize_)->zero_tokens())
        ("shownr", po::value<bool>(&DoShowNormals_)->zero_tokens())
        ("showgr", po::value<bool>(&DoShowGradients_)->zero_tokens())

        ("camera", po::value<std::string>(&CameraDescription_))
        ("savesc", po::value<std::string>(&SaveScreenshotPath_))
        ("input", po::value<std::string>(&InputPath_))
        ("output", po::value<std::string>(&OutputPath_))
        ("usealphas", po::value<std::string>(&AlphaInputPath_))
        ("useparams", po::value<std::string>(&ParamsInputPath_))
        ("savealphas", po::value<std::string>(&AlphaOutputPath_))
        ("saveparams", po::value<std::string>(&ParamsOutputPath_))
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
    if (ParamsInputPath_.size()) {
        Params_.Load(ParamsInputPath_.c_str());
    }

    Builder_.SetParams(Params_);
    Builder_.SetInputCloud(Input_);
    PrintParameters();

    Builder_.GenerateTrainingSet();
    ExportForLibSVM();

    if (AlphaInputPath_.size()) {
        std::ifstream alphaInput(AlphaInputPath_.c_str());
        std::vector<SVMFloat> alphas;
        std::copy(std::istream_iterator<SVMFloat>(alphaInput), std::istream_iterator<SVMFloat>(),
                std::back_inserter(alphas));

        Builder_.InitSVM(alphas);
    } else {
        Builder_.Learn();
        TrainedModel_ = true;
    }
    if (AlphaOutputPath_.size()) {
        std::ofstream alphaOutput(AlphaOutputPath_.c_str());
        std::copy(Builder_.SVM().Alphas(), Builder_.SVM().Alphas() + Builder_.Objects->size(),
                std::ostream_iterator<SVMFloat>(alphaOutput, "\n"));
    }
    if (ParamsOutputPath_.size()) {
        Params_.Save(ParamsOutputPath_.c_str());
    }
    ExportAlphaMap();

    Builder_.CalcGradients();

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
    viewer.EasyAdd(Input_, "input", [this] (int i) { // #1
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
    if (DoShowGradients_) {
        NormalCloud::ConstPtr grads = Builder_.Gradients;
        viewer.addPointCloudNormals<PointType, NormalType>(Input_, grads, 100, 0.002);
    }

    viewer.EasyAdd(Input_, "input", [this] (int i) { // #2
                int const num = Builder_.Pixel2TrainNum[i];
                float const alpha = Builder_.SVM().Alphas()[num];
                float const relAlpha = alpha / Params_.MaxAlpha;
                return Color({static_cast<int>(relAlpha * 255), 0, static_cast<int>((1 - relAlpha) * 255)});
            });

    float const maxGN = *max_element(Builder_.GradientNorms.begin(), Builder_.GradientNorms.end());
    viewer.EasyAdd(Input_, "input", [this, maxGN] (int i) { // #3
                int const num = Builder_.Pixel2RawIndex.at(i);
                float const gn = Builder_.GradientNorms[num];
                float const relGN = gn / maxGN;
                return Color({static_cast<int>(relGN * 255), 0, static_cast<int>((1 - relGN) * 255)});
            });

    float const maxAGN = *max_element(
            Builder_.AdjustedGradientNorms.begin(),
            Builder_.AdjustedGradientNorms.end());
    viewer.EasyAdd(Input_, "input", [this, maxAGN] (int i) { // #4
                int const num = Builder_.Pixel2RawIndex.at(i);
                float const agn = Builder_.AdjustedGradientNorms[num];
                float const relAGN = agn / maxAGN;
                return Color({static_cast<int>(relAGN * 255), 0, static_cast<int>((1 - relAGN) * 255)});
            });

    float const maxDTN = quantile(Builder_.DistToNN, 0.97);
    viewer.EasyAdd(Input_, "input", [this, maxDTN] (int i) { // #5
                int const num = Builder_.Pixel2RawIndex.at(i);
                float const dtn = Builder_.DistToNN.at(num);
                float const relDTN = std::min(1.0f, dtn / maxDTN);
                return Color({static_cast<int>(relDTN * 255), 0, static_cast<int>((1 - relDTN) * 255)});
            });

    float const maxZ = max_element(Input_->begin(), Input_->end(),
            [] (PointType const& lft, PointType const& rgh) {
                if (pcl_isfinite(lft.z) && pcl_isfinite(rgh.z)) {
                    return lft.z < rgh.z;
                } else {
                    return ! pcl_isfinite(lft.z);
                }
            })->z;
    viewer.EasyAdd(Input_, "input", [this, &maxZ] (int i) { // #6
                float const relZ = Input_->at(i).z / maxZ;
                return Color({static_cast<int>(relZ * 255), 0, static_cast<int>((1 - relZ) * 255)});
            });

    float const maxSVCount = *max_element(Builder_.NumCloseSV.begin(), Builder_.NumCloseSV.end());
    viewer.EasyAdd(Input_, "input", [this, &maxSVCount] (int i) { // #7
                int const num = Builder_.Pixel2RawIndex.at(i);
                float const relSVCount = Builder_.NumCloseSV[num] / static_cast<float>(maxSVCount);
                return Color({static_cast<int>(relSVCount * 255), 0, static_cast<int>((1 - relSVCount) * 255)});
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
    if (! TrainedModel_) {
        return;
    }
    GridNeighbourModificationStrategy const& strat = *Builder_.Strategy;

    std::cout << "-------------------- STATISTICS" << std::endl;
    std::cout << "New SVM converged in " << Builder_.SVM().Iteration << " iterations" << std::endl;
    std::cout << "SVM3D: " << Builder_.Objects->size() << " input vectors" << std::endl;
    std::cout << "SVM3D: " << Builder_.SVM().SVCount << " support vectors" << std::endl;
    std::cout << "SVM3D: " << Builder_.SVM().TargetFunction << " target function" << std::endl;
    std::cout << "SVM3D: " << Builder_.SVM().TouchedCount << " touched count" << std::endl;
    std::cout << "GridStrategy: Number of cache misses: " << strat.NumCacheMisses << std::endl;
    std::cout << "GridStrategy: Number of optimization failures: " << strat.NumOptimizeFailures << std::endl;
    std::cout << "GridStrategy: Average number of neighbors: " <<
        static_cast<float>(strat.TotalNeighborsProcessed) / strat.NumNeighborsCalculations << std::endl;
}

int main(int argc, char* argv []) {
    return App(argc, argv).Run();
}
