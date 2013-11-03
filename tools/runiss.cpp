#include "components/baseapp.h"
#include "components/svs.h"

#include "boost/program_options.hpp"

#include "pcl/keypoints/iss_3d.h"
#include "pcl/common/time.h"

#include <iostream>
#include <fstream>

namespace po = boost::program_options;

class App : public BaseApp {
public:
    App(int argc, char* argv [])
        : Seed_(1)
        , TakeProb_(0.5)
    {
        ParseArgs(argc, argv);
    }

    void ParseArgs(int argc, char* argv []);

    int Run();
    void CalcISS3D();

private:
    void PrintFeaturePoints();
    void PrintParameters();

private:
    std::string OutputPath_;
    float Seed_;
    float TakeProb_;

    PointCloud::Ptr Subsampled_;
    KDTreeType::Ptr SubsampledTree_;
    PointCloud::Ptr FeaturePoints_;
};

void App::ParseArgs(int argc, char* argv []) {
    po::options_description desc;
    desc.add_options()
        ("help", "show help")
        ("seed", po::value<float>(&Seed_))
        ("ptake", po::value<float>(&TakeProb_))
        ("cloud-path", po::value<std::string>(&InputPath_))
        ("fp-path", po::value<std::string>(&OutputPath_));

    po::positional_options_description pos;
    pos.add("cloud-path", 1);
    pos.add("fp-path", 2);

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
    CalcISS3D();
    PrintFeaturePoints();
    return 0;
}

void App::CalcISS3D() {
    Subsampled_.reset(new PointCloud);
    for (int i = 0; i < InputNoNan_->size(); ++i) {
        if (((rand() % 10000) / 10000.0) < TakeProb_) {
            Subsampled_->push_back(InputNoNan_->at(i));
        }
    }

    SubsampledTree_.reset(new KDTreeType);
    SubsampledTree_->setInputCloud(Subsampled_);

    pcl::ScopeTime st("ISS3D");
    FeaturePoints_.reset(new PointCloud);

    pcl::ISSKeypoint3D<PointType, PointType> iss;
    iss.setSearchMethod(SubsampledTree_);
    iss.setSalientRadius(6 * Resolution_);
    iss.setNonMaxRadius(4 * Resolution_);
    iss.setThreshold21(0.975);
    iss.setThreshold32(0.975);
    iss.setMinNeighbors(5);
    iss.setNumberOfThreads(4);
    iss.setInputCloud(Subsampled_);
    iss.compute(*FeaturePoints_);
}

void App::PrintFeaturePoints() {
    std::ofstream ofstr(OutputPath_);
    SupportVectorShape(FeaturePoints_).SaveAsText(ofstr);
}

void App::PrintParameters() {
    std::cout << "Seed " << Seed_ << std::endl;
}

int main(int argc, char* argv []) {
    return App(argc, argv).Run();
}
