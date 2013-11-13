#include "components/common.h"
#include "components/fastsvm.h"
#include "components/newsvm.h"
#include "components/gridstrategy.h"

#include "pcl/common/time.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "boost/program_options.hpp"

#include <iostream>

namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////

class App {
public:
    App();

    void ParseArgs(int argc, char* argv []);
    void Run();

private:
    void Generate();
    void RunNew();
    void RunOld();
    void SaveAlphaMap();

private:
    bool DoRunNew_;
    bool DoRunOld_;

    std::string AlphaMapDest_;

    int GridSize_;
    int NPoints_;

    float MaxAlpha_;
    float KernelWidth_;
    float KernelThreshold_;
    float TerminateEps_;

    PointCloud Objects_;
    std::vector<float> Labels_;

    SVM3D NewSVM_;
    FastSVM OldSVM_;
};

App::App()
    : DoRunNew_(true)
    , DoRunOld_(false)
    , GridSize_(10)
    , MaxAlpha_(1)
    , KernelWidth_(sqrt(3))
    , KernelThreshold_(1e-20)
    , TerminateEps_(1e-3)
{
}

void App::ParseArgs(int argc, char* argv []) {
    po::options_description desc;
    desc.add_options()
        ("help", "show help")
        ("old", po::value<bool>(&DoRunOld_)->zero_tokens())
        ("size", po::value<int>(&GridSize_))
        ("malpha", po::value<float>(&MaxAlpha_))
        ("kwidth", po::value<float>(&KernelWidth_))
        ("kthr", po::value<float>(&KernelThreshold_))
        ("teps", po::value<float>(&TerminateEps_))
        ("amap", po::value<std::string>(&AlphaMapDest_));

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    vm.notify();

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        exit(0);
    }
}

void App::Generate() {
    NPoints_ = GridSize_ * GridSize_;

    for (int i = 0; i < GridSize_; ++i) {
        for (int j = 0; j < GridSize_; ++j) {
            Objects_.push_back(createPoint<PointType>(i, j, 0));
        }
    }
    for (int i = 0; i < GridSize_; ++i) {
        for (int j = 0; j < GridSize_; ++j) {
            Objects_.push_back(createPoint<PointType>(i, j, 1));
        }
    }
    Labels_.assign(GridSize_ * GridSize_, 1);
    Labels_.resize(2 * GridSize_ * GridSize_, -1);
}

void App::RunNew() {
    if (! DoRunNew_) {
        return;
    }
    pcl::ScopeTime st("New SVM");

    GridNeighbourModificationStrategy::Grid2Num grid2num;
    GridNeighbourModificationStrategy::Num2Grid num2grid;
    grid2num.resize(GridSize_);
    for (int i = 0; i < GridSize_; ++i) {
        grid2num[i].resize(GridSize_);
        for (int j = 0; j < GridSize_; ++j) {
            std::vector<int> & target = grid2num[i][j];

            target.resize(2);
            target[0] = i * GridSize_ + j;
            target[1] = NPoints_ + target[0];
        }
    }
    num2grid.resize(2 * NPoints_);
    for (int k = 0; k < NPoints_; ++k) {
        num2grid[k] = std::make_pair(k / GridSize_, k % GridSize_);
        num2grid[NPoints_ + k] = num2grid[k];
    }

    NewSVM_.SetParams(MaxAlpha_, 1 / sqr(KernelWidth_), TerminateEps_);
    GridNeighbourModificationStrategy * strategy =
            new GridNeighbourModificationStrategy(
                GridSize_, GridSize_,
                grid2num, num2grid,
                KernelWidth_, KernelThreshold_, 1,
                1 << 30);
    std::shared_ptr<IGradientModificationStrategy> strPtr(strategy);
    NewSVM_.SetStrategy(strPtr);
    NewSVM_.Train(Objects_, Labels_);

    std::cout << "New SVM converged in " << NewSVM_.Iteration << " iterations" << std::endl;
    std::cout << NewSVM_.SVCount << " support vectors" << std::endl;
    std::cout << NewSVM_.TargetFunction << " target function" << std::endl;
    strategy->PrintStatistics(std::cout);
}

void App::RunOld() {
    if (! DoRunOld_) {
        return;
    }
    pcl::ScopeTime st("Old SVM");

    BaseSVMParams params;
    params.svm_type = BaseSVM::C_SVC;
    params.kernel_type = BaseSVM::RBF;
    params.C = MaxAlpha_;
    params.gamma = 1 / sqr(KernelWidth_);
    params.term_crit.epsilon = TerminateEps_;
    params.term_crit.type = CV_TERMCRIT_EPS;

    My::kernelThreshold = SVM_EPS;
    OldSVM_.train(Objects_, Labels_, params);
}

void App::Run() {
    Generate();
    RunNew();
    RunOld();
    SaveAlphaMap();
}

void App::SaveAlphaMap() {
    if (AlphaMapDest_.empty()) {
        return;
    }

    cv::Mat image(GridSize_, GridSize_, CV_8UC3);
    for (int i = 0; i < GridSize_; ++i) {
        for (int j = 0; j < GridSize_; ++j) {
            int const idx = i * GridSize_ + j;
            float const relAlpha = NewSVM_.Alphas()[idx] / MaxAlpha_;

            image.at<cv::Vec3b>(i, j) = cv::Vec3b(255 * (1 - relAlpha), 0, 255 * relAlpha);
        }
    }

    imwrite(AlphaMapDest_, image);
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv []) {
    App app;
    app.ParseArgs(argc, argv);
    app.Run();
    return 0;
}
