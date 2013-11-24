#include "common.h"
#include "df.h"

DecisionFunction::DecisionFunction(
        float gamma,
        std::vector<PointType> const& sv,
        std::vector<float> const& alpha,
        float rho)
    : SV_(sv)
    , Alpha_(alpha)
    , Gamma_(gamma)
      , Rho_(rho)
{
    assert(SV_.size() == Alpha_.size());
}

void DecisionFunction::Reset(float gamma, float rho) {
    Gamma_ = gamma;
    Rho_ = rho;
    SV_.clear();
    Alpha_.clear();
}

float DecisionFunction::Value(PointType const& point) const {
    float dfVal = 0.0;
    for (int i = 0; i < SV_.size(); ++i) {
        dfVal += KernelValueWithAlpha(i, point);
    }
    return dfVal;
}

PointType DecisionFunction::Gradient(PointType const& point) const {
    PointType result;
    for (int i = 0; i < SV_.size(); ++i) {
        PointType add = point;
        add.getVector3fMap() -= SV_[i].getVector3fMap();
        add.getVector3fMap() *= KernelValueWithAlpha(i, point);
        // add = \alpha_i K(x, x_i) (x - x_i)
        result.getVector3fMap() += add.getVector3fMap();
    }

    // result = -\sum\limits_{i=1}^n alpha_i K(x, x_i) (x - x_i)
    result.getVector3fMap() *= -1;
    // differs from real gradient in 2\gamma times
    return result;
}

void DecisionFunction::Hessian(PointType const& point, Eigen::MatrixXf * result) const {
    *result = Eigen::Matrix3f::Zero(3,  3);

    for (int svi = 0; svi < SV_.size(); ++svi) {
        float const coof = KernelValueWithAlpha(svi, point);

        PointType diff = point;
        auto diffMap = diff.getVector3fMap();
        diffMap -= SV_[svi].getVector3fMap();

        for (int x = 0; x < 3; ++x) {
            for (int y = 0; y < 3; ++y) {
                (*result)(x, y) += coof * (2 * Gamma_ * diffMap(x) * diffMap(y) - (x == y ? 1 : 0));
            }
        }
    }
}

PointType DecisionFunction::SquaredGradientNormGradient(PointType const& point) const {
    Eigen::MatrixXf hess;
    Hessian(point, &hess);
    Eigen::Vector3f result = 2 * hess * Gradient(point).getVector3fMap();

    return createPoint<PointType>(result(0), result(1), result(2));
}

float DecisionFunction::KernelValueWithAlpha(int svIndex, PointType const& point) const {
    float const dist2 = pcl::squaredEuclideanDistance(SV_[svIndex], point);
    return Alpha_[svIndex] * exp(-Gamma_ * dist2);
}

int DecisionFunction::SVCount() const {
    return SV_.size();
}
