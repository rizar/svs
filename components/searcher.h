#pragma once

#include "common.h"

class FeaturePointSearcher {
public:
    FeaturePointSearcher(PointCloud::Ptr input, int numFP,
                         std::vector<float> adjustedGradientNorms);

private:

public:
    PointCloud::Ptr FeaturePoints;
};
