#include "common.h"

#include "pcl/search/kdtree.h"
#include "pcl/search/octree.h"

class BaseApp {
protected:
    typedef pcl::octree::OctreePointCloudSearch<PointType> OctTreeType;
    typedef pcl::search::KdTree<PointType> KDTreeType;

protected:
    void Load();
    void CalcDistanceToNN();

protected:
    std::string InputPath_;

    float Resolution_;

    PointCloud::Ptr Input_;
    PointCloud::Ptr InputNoNan_;
    KDTreeType::Ptr InputKDTree_;
    OctTreeType::Ptr InputOctTree_;
    std::vector<float> DistToNN_;
};
