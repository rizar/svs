#include "components/fastsvm.h"
#include "components/trainset.h"
#include "components/searcher.h"
#include <components/griditer.h>

#include "utilities/prettyprint.hpp"

#include <iostream>

void testDecisionFunction() {
    DecisionFunction df(0.5,
                        std::vector<PointType>({createPoint<PointType>(0.0, 0.0, 0.0)}),
                        std::vector<float>({1}),
                        0);
    PointType point = createPoint<PointType>(0.5, 0.5, 0.0);
    Eigen::MatrixXf hessian;

    std::cout << df.Value(point) << std::endl;
    std::cout << df.Gradient(point) << std::endl;
    df.Hessian(point, &hessian);
    std::cout << hessian << std::endl;
    std::cout << df.SquaredGradientNormGradient(point) << std::endl;
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

    TrainingSetGenerator tg(2, 1.0, 1.0);
    tg.GenerateFromSensor(pc, std::vector<float>(ps.size(), sqrt(3) / 3));

    std::cout << tg.Objects.points << std::endl;
    std::cout << tg.Labels << std::endl;
    std::cout << tg.Grid2Num << std::endl;
    std::cout << tg.Num2Grid << std::endl;
}

void testGridRadiusTraversal() {
    GridRadiusTraversal grt(10, 10);

    grt.TraverseCircle(3, 3, 3, [] (int x, int y) {
                std::cout << x << " " << y << ' ';
                std::cout << sqr(x - 3) + sqr(y - 3) << '\n';
            });
    grt.TraverseCircle(1, 2, 3, [] (int x, int y) {
                std::cout << x << " " << y << ' ';
                std::cout << sqr(x - 1) + sqr(y - 2) << '\n';
            });
}

int main() {
    testDecisionFunction();
    return 0;
}
