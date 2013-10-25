#pragma once

#include "common.h"
#include "fastsvm.h"

class FeaturePointSearcher {
public:
    FeaturePointSearcher();

    void Search(FastSVM const& model, PointCloud const& cloud);
    PointType SearchFromSeed(FastSVM const& model, PointType const& seed);

public:
    int NumSeeds;

    PointCloud::Ptr Seeds;
    PointCloud::Ptr FeaturePoints;
};
