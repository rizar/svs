#pragma once

#include "common.h"

#include <pcl/common/distances.h>

class DecisionFunction {
public:
    DecisionFunction()
    {
    }

    DecisionFunction(
            float gamma,
            std::vector<PointType> const& sv,
            std::vector<float> const& alpha,
            float rho);

    void Reset(float gamma, float rho);

    void AddSupportVector(PointType const& newSV, float alpha) {
        SV_.push_back(newSV);
        Alpha_.push_back(alpha);
    }

    float KernelValueWithAlpha(int svIndex, PointType const& point) const;

    float Value(PointType const& point) const;
    PointType Gradient(PointType const& point) const;
    void Hessian(PointType const& point, Eigen::MatrixXf * result) const;
    PointType SquaredGradientNormGradient(PointType const& point) const;

    int SVCount() const;

public:
    std::vector<PointType> SV_;
    std::vector<float> Alpha_;
    float Gamma_;
    float Rho_;
};
