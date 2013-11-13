#include "components/fastsvm.h"
#include "components/trainset.h"
#include "components/searcher.h"

#include "utilities/prettyprint.hpp"

#include <iostream>

void testDecisionFunction() {
    DecisionFunction df(0.5,
                        std::vector<PointType>({createPoint<PointType>(0.0, 0.0, 0.0)}),
                        std::vector<float>({1}),
                        0);
    PointType point = createPoint<PointType>(0.5, 0.5, 0.0);
    Eigen::MatrixXf hessian;

    std::cout << df.decisionFunction(point) << std::endl;
    std::cout << df.gradient(point) << std::endl;
    df.hessian(point, &hessian);
    std::cout << hessian << std::endl;
    std::cout << df.squaredGradientNormGradient(point) << std::endl;
}

void testTrainSetGenerator() {
    PointCloud pc;
    std::vector<PointType> ps {
        createPoint<PointType>(-1, 0, 1),
        createPoint<PointType>(0, 0, 1),
        createPoint<PointType>(1, 0, 1),
        createPoint<PointType>(-1, -1, 1),
        createPoint<PointType>(0, -1, 1),
        createPoint<PointType>(1, -1, 1)
    };
    pc.insert(pc.begin(), ps.begin(), ps.end());
    pc.height = 2;
    pc.width = 3;

    TrainingSetGenerator tg(2, 1.0);
    tg.Generate(pc, std::vector<float>(ps.size(), sqrt(3) / 3));

    std::cout << tg.Objects.points << std::endl;
    std::cout << tg.Labels << std::endl;
    std::cout << tg.Grid2Num << std::endl;
    std::cout << tg.Num2Grid << std::endl;
}

int main() {
    testTrainSetGenerator();
    return 0;
}
