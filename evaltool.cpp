#include "baseapp.h"
#include "svs.h"

#include "boost/program_options.hpp"

#include <iostream>
#include <fstream>

namespace po = boost::program_options;

class App : public BaseApp {
public:
    App(int argc, char* argv [])
        : CloseThreshold_(3)
    {
        ParseArgs(argc, argv);
    }

    void ParseArgs(int argc, char* argv []);

    int Run();

private:
    float CloseThreshold_;

private:
    void Load();

    void PrintParameters();
    void PrintStatistics();

    int ClosestInTreeIndex(PointType const& fp, KDTreeType const& tree);
    PointType ClosestInTreePoint(PointType const& fp, KDTreeType const& tree);
    bool HasClosePointsInAllSVS(PointType const& fp, float threshold);
    void CalcNumStableFP();

private:
    std::vector<std::string> SVSPathes_;
    std::vector<SupportVectorShape> SVS_;
    std::vector<KDTreeType::Ptr> SVSTrees_;

    int NumStableFP_;
};

void App::ParseArgs(int argc, char* argv []) {
    po::options_description desc;
    desc.add_options()
        ("help", "show help")
        ("clthr", po::value<float>(&CloseThreshold_))
        ("cloud-path", po::value<std::string>(&InputPath_))
        ("svs-pathes", po::value< std::vector<std::string> >(&SVSPathes_));

    po::positional_options_description pos;
    pos.add("cloud-path", 1);
    pos.add("svs-pathes", -1);

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
    PrintParameters();
    CalcNumStableFP();
    PrintStatistics();
    return 0;
}

void App::Load() {
    BaseApp::Load();

    SVS_.resize(SVSPathes_.size());
    for (int i = 0; i < SVSPathes_.size(); ++i) {
        std::ifstream ifstr(SVSPathes_[i]);
        SVS_[i].LoadAsText(ifstr);
    }

    SVSTrees_.resize(SVS_.size());
    for (int i = 0; i < SVS_.size(); ++i) {
        SVSTrees_[i].reset(new KDTreeType);
        SVSTrees_[i]->setInputCloud(SVS_[i].FeaturePoints());
    }
}

void App::PrintParameters() {
    std::cout << InputNoNan_->size() << " points in input cloud" << std::endl;
    for (int i = 0; i < SVS_.size(); ++i) {
        std::cout << SVS_[i].FeaturePoints()->size() << ' ';
    }
    std::cout << " points in feature point clouds" << std::endl;
}

void App::PrintStatistics() {
    std::cout << NumStableFP_ << " are stable" << std::endl;
}

int App::ClosestInTreeIndex(PointType const& center, KDTreeType const& tree) {
    std::vector<int> indices;
    std::vector<float> dist2;
    tree.nearestKSearch(center, 1, indices, dist2);
    return indices[0];
}

PointType App::ClosestInTreePoint(PointType const& center, KDTreeType const& tree) {
    return tree.getInputCloud()->at(ClosestInTreeIndex(center, tree));
}

bool App::HasClosePointsInAllSVS(PointType const& fp, float threshold) {
    for (int i = 0; i < SVS_.size(); ++i) {
        float const minDist = sqrt(squaredEuclideanDistance(fp, ClosestInTreePoint(fp, *SVSTrees_[i])));
        if (minDist > threshold) {
            return false;
        }
    }
    return true;
}

void App::CalcNumStableFP() {
    CalcDistanceToNN();

    NumStableFP_ = 0;
    PointCloud const& fp = *(SVS_[0].FeaturePoints());
    for (int i = 0; i < fp.size(); ++i) {
        int const index = ClosestInTreeIndex(fp[i], *InputKDTree_);
        float const localResolution = DistToNN_[index];
        NumStableFP_ += HasClosePointsInAllSVS(fp[i], localResolution * CloseThreshold_) ? 1 : 0;
    }
}

int main(int argc, char* argv []) {
    return App(argc, argv).Run();
}
