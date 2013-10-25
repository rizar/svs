#include "fastsvm.h"
#include "trainset.h"
#include "searcher.h"

#include <iostream>

void testDecisionFunction() {
    DecisionFunction df(0.5,
                        std::vector<PointType>({createPoint<PointType>(0.0, 0.0, 0.0)}),
                        std::vector<float>({1}));
    PointType point = createPoint<PointType>(0.5, 0.5, 0.0);
    Eigen::MatrixXf hessian;

    std::cout << df.decisionFunction(point) << std::endl;
    std::cout << df.gradient(point) << std::endl;
    df.hessian(point, &hessian);
    std::cout << hessian << std::endl;
    std::cout << df.squaredGradientNormGradient(point) << std::endl;
}

void testTrainSetGenerator() {
    PointCloud cloud;
    cloud.push_back(createPoint<PointType>(1, 1, 1));
    cloud.push_back(createPoint<PointType>(1, -1, 1));
    cloud.push_back(createPoint<PointType>(-1, 1, 1));
    cloud.push_back(createPoint<PointType>(-1, -1, 1));

    TrainingSetGenerator tst(sqrt(3), 3, 0.5);
    cv::Mat objects;
    cv::Mat responses;
    tst.generate(cloud, &objects, &responses);
    std::cout << "OBJECTS\n" << objects << std::endl;
    std::cout << "RESPONSES\n" << responses << std::endl;
}

int main() {
    testTrainSetGenerator();
    return 0;
}
