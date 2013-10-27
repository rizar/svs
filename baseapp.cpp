#include "baseapp.h"
#include "analysis.h"

#include "pcl/io/pcd_io.h"
#include "pcl/common/time.h"
#include "pcl/common/distances.h"
#include "pcl/filters/filter.h"

void BaseApp::Load() {
    Input_.reset(new PointCloud);
    InputNoNan_.reset(new PointCloud);
    pcl::io::loadPCDFile(InputPath_, *Input_);
    std::vector<int> tmp;
    pcl::removeNaNFromPointCloud(*Input_, *InputNoNan_, tmp);

    InputKDTree_.reset(new pcl::search::KdTree<PointType>);
    InputKDTree_->setInputCloud(InputNoNan_);
    Resolution_ = computeCloudResolution(InputNoNan_, *InputKDTree_);

    InputOctTree_.reset(new OctTreeType(10 * Resolution_));
    InputOctTree_->setInputCloud(InputNoNan_);
    InputOctTree_->addPointsFromInputCloud();
}

void BaseApp::CalcDistanceToNN() {
    if (DistToNN_.size()) {
        return;
    }
    pcl::ScopeTime st("CalcDistanceToNN");

    std::vector<int> indices;
    std::vector<float> dist2;
    DistToNN_.resize(InputNoNan_->size());
    for (int i = 0; i < InputNoNan_->size(); ++i) {
        InputKDTree_->nearestKSearch(i, 2, indices, dist2);
        DistToNN_[i] = sqrt(dist2[1]);
    }
}
