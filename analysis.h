#include "common.h"

#include "pcl/search/kdtree.h"

double computeCloudResolution(
        PointCloud::ConstPtr cloud,
        pcl::search::KdTree<PointType> const& tree)
{
    double res = 0.0;
    int n_points = 0;
    int nres;
    std::vector<int> indices (2);
    std::vector<float> sqr_distances (2);

    for (size_t i = 0; i < cloud->size (); ++i)
    {
        if (! pcl_isfinite ((*cloud)[i].x))
        {
            continue;
        }
        //Considering the second neighbor since the first is the point itself.
        nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
        if (nres == 2)
        {
            res += sqrt (sqr_distances[1]);
            ++n_points;
        }
    }
    if (n_points != 0)
    {
        res /= n_points;
    }
    return res;
}

void printKernelValueHistogram(PointCloud::Ptr shape, float kernelWidth) {
    int const MAX_LOG = 20;
    float const LOG2 = log(2);

    std::vector<double> logFreq(MAX_LOG + 1);
    std::vector<double> cumFreq(MAX_LOG + 1);
    for (int i = 0; i < shape->size(); ++i) {
        for (int j = i + 1; j < shape->size(); ++j) {
            float const dist = pcl::squaredEuclideanDistance(shape->at(i), shape->at(j));
            float const kernel = exp(-dist / kernelWidth / kernelWidth);
            float log2kernel = -log(kernel) / LOG2;
            if (isnan(log2kernel) || log2kernel > MAX_LOG) {
                log2kernel = MAX_LOG;
            }
            logFreq[static_cast<int>(log2kernel)] += 1.0;
        }
    }

    float total = shape->size() * (shape->size() - 1) / 2;
    for (int i = 0; i < logFreq.size(); ++i) {
        logFreq[i] /= total;
    }
    logFreq[0] = logFreq[0];
    for (int i = 0; i < cumFreq.size(); ++i) {
        cumFreq[i] = cumFreq[i - 1] + logFreq[i];
    }

    for (int i = 0; i < logFreq.size(); ++i) {
        std::cout.precision(5);
        std::cout << -i << "\t" << logFreq[i] << "\t" << cumFreq[i] << std::endl;
    }
}

