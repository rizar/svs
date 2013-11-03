#pragma once

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

typedef pcl::PointXYZRGB PointType;
typedef pcl::PointCloud<PointType> PointCloud;
typedef pcl::PointCloud<pcl::Normal> NormalCloud;

inline float sqr(float x) {
    return x * x;
}

inline bool pointHasNan(PointType const& p) {
    return pcl_isnan(p.x) || pcl_isnan(p.y) || pcl_isnan(p.z);
}

template <class P>
P createPoint(float x, float y, float z) {
    P p;
    p.x = x;
    p.y = y;
    p.z = z;
    return p;
}

template <class P, class V>
P pointFromVector(V const& v) {
    P p;
    p.getVector3fMap() = v.template cast<float>();
    return p;
}

inline PointType addTemperature(PointType const& p, float t,
                            float bottom, float top) {
    float const length = top - bottom;
    float const scaled = std::min(1.0f, std::max(0.0f, (t - bottom) / length));
    PointType res = p;
    res.r = 255 * scaled;
    res.b = 255 * (1 - scaled);
    res.g = 0;
    return res;
}

inline float quantile(std::vector<float> const& xs, float q) {
    std::vector<float> copy(xs);
    std::sort(copy.begin(), copy.end());
    return copy[static_cast<size_t>(copy.size() * q)];
}

