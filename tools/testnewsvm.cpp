#include "components/common.h"
#include "components/fastsvm.h"
#include "components/newsvm.h"

#include "boost/program_options.hpp"

#include <iostream>

namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////

class App {
public:
    void ParseArgs(int argc, char* argv []);
    void Run();

private:
    void Generate();
    void RunNew();
    void RunOld();

private:
    int GridSize_;
    int NPoints_;

    PointCloud Objects_;
    std::vector<int> Labels_;

    SVM3D NewSVM_;
    FastSVM OldSVM_;
};

void App::ParseArgs(int argc, char* argv []) {
    po::options_description desc;
    desc.add_options()
        ("help", "show help")
        ("size", po::value<int>(&GridSize_)->required());

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
    NewSVM_.SetParams(1, 1 / 3.0, 1e-3);
    NewSVM_.Train(Objects_, Labels_);
    std::cout << "New SVM converged in " << NewSVM_.Iteration << " iterations" << std::endl;
    std::cout << NewSVM_.SVCount << " support vectors" << std::endl;
    std::cout << NewSVM_.TargetFunction << " target function" << std::endl;
}

void App::RunOld() {
    BaseSVMParams params;
    params.svm_type = BaseSVM::C_SVC;
    params.kernel_type = BaseSVM::RBF;
    params.C = 1;
    params.gamma = 1 / 3.0;
    params.term_crit.epsilon = 1e-3;
    params.term_crit.type = CV_TERMCRIT_EPS;

    My::kernelThreshold = SVM_EPS;
    OldSVM_.train(Objects_, Labels_, params);
}

void App::Run() {
    Generate();
    RunNew();
    RunOld();
}

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv []) {
    App app;
    app.ParseArgs(argc, argv);
    app.Run();
    return 0;
}
