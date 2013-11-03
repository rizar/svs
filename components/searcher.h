#pragma once

#include "common.h"
#include "fastsvm.h"

#include <functional>

class FeaturePointSearcher {
public:
    FeaturePointSearcher();

    void Search(FastSVM const& model);
    PointType SearchFromSeed(FastSVM const& model, PointType const& seed, float * gn2gn = 0);

    void ChooseSeeds(PointCloud const& cloud, std::vector<float> const& gradientNorms);
    void OneStageSearch(
            FastSVM const& model,
            PointCloud const& cloud,
            std::vector<float> const& gradientNorms);

private:
    bool HasClosePoint(PointCloud const& cloud, PointType const& point, float threshold);

    void IterateProspectiveSeeds(PointCloud const& cloud, std::vector<float> const& gradientNorms,
            std::function<bool (PointType const& point)>);

public:
    int NumSeeds;
    float MinSpaceSeeds;
    float MinSpaceFP;

    PointCloud::Ptr Seeds;
    PointCloud::Ptr FeaturePoints;
};
