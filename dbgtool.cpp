#include "fastsvm.h"

#include <iostream>

int main() {
    DecisionFunction df(0.5,
                        std::vector<PointType>({PointType(0.0, 0.0, 0.0)}),
                        std::vector<float>({1}));
    PointType point = createPoint<PointType>(0.5, 0.5, 0.0);
    Eigen::MatrixXf hessian;

    std::cout << df.decisionFunction(point) << std::endl;
    std::cout << df.gradient(point) << std::endl;
    df.hessian(point, &hessian);
    std::cout << hessian << std::endl;
    std::cout << df.squaredGradientNormGradient(point) << std::endl;

    return 0;
}
