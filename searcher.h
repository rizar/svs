#pragma once

#include "common.h"
#include "fastsvm.h"

class FeaturePointSearcher {
public:
    FeaturePointSearcher();

    void Search(FastSVM const& model);
    PointType SearchFromSeed(FastSVM const& model, PointType const& seed);

    void ChooseSeeds(PointCloud const& cloud, std::vector<float> const& gradientNorms);

public:
    int NumSeeds;
    float MinSpace;

    PointCloud::Ptr Seeds;
    PointCloud::Ptr FeaturePoints;
};
