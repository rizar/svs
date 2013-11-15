#pragma once

#include "opencv2/core/core.hpp"

#include "rangeimagepoint.h"
#include "gridstrategy.h"

class TrainingSetGenerator {
public:
    TrainingSetGenerator(
            float width,
            float prob)
        : Width_(width)
        , Prob_(prob)
    {
    }

public:
    void GenerateFromSensor(PointCloud const& input, std::vector<float> const& localRes);
    void GenerateUsingNormals(PointCloud const& input, NormalCloud const& normals,
            std::vector<float> const& localRes);

private:
    int AddPoint(int x, int y, PointType const& point, float label);

public:
    PointCloud Objects;
    std::vector<float> Labels;
    std::vector<int> Pixel2Num;
    GridNeighbourModificationStrategy::Num2Grid Num2Grid;
    GridNeighbourModificationStrategy::Grid2Num Grid2Num;

private:
    bool TossCoin() const {
        return rand() % MODULE < Prob_ * MODULE;
    }

private:
    static int const MODULE = 100000;

private:
    float Width_;
    float Prob_;
};
