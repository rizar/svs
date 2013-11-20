#include "components/fastsvm.h"
#include "components/trainset.h"
#include "components/searcher.h"
#include "components/griditer.h"
#include "components/svs.h"

#include "utilities/prettyprint.hpp"

#include "pcl/pcl_macros.h"

#include <iostream>
#include <cmath>
#include <limits>

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

void testSVSBuilderGradientNorms() {
    SVSParams params;
    params.SmoothingRange = 3;
    params.KernelWidth = 2;
    params.KernelThreshold = 1e-1;

    int const height = 15;
    int const width = 20;

    PointCloud::Ptr input(new PointCloud);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            input->push_back(createPoint<PointType>(i, j, 1));
        }
    }
    input->height = height;
    input->width = width;

    float const nan = std::numeric_limits<float>::quiet_NaN();
    assert(! pcl_isfinite(nan));
    input->at(1, 2) = createPoint<PointType>(nan, nan, nan);
    input->at(2, 1) = input->at(1, 2);
    input->is_dense = false;

    auto initBuilder = [params, input] (SVSBuilder * builder) {
        builder->SetParams(params);
        builder->SetInputCloud(input);
        builder->GenerateTrainingSet();
    };

    auto findPos = [] (PointCloud const& objects, float x, float y, float z) {
        return std::find_if(objects.begin(), objects.end(), [&x, &y, &z] (PointType const& cur) {
                    return cur.x == x && cur.y == y && cur.z == z;
                }) - objects.begin();
    };

    auto rowIndex = [&input] (SVSBuilder const& builder, int y, int x) {
        return builder.Pixel2RowIndex.at(y * input->width + x);
    };

    {
        SVSBuilder builder;
        initBuilder(&builder);

        std::vector<SVMFloat> alphas(builder.Objects->size());
        alphas.at(findPos(*builder.Objects, 3, 4, 1)) = 1.0;
        builder.InitSVM(alphas);
        builder.CalcGradientNorms();

        assert(builder.Gamma == 0.25);
        assert(builder.PixelRadius == 3);
        assert(fabs(builder.Radius2 - 9.2) < 0.1);
        assert(fabs(builder.GradientNorm[rowIndex(builder, 2, 2)] - exp(-1.25) * sqrt(5)) < 1e-7);
        assert(builder.GradientNorm[rowIndex(builder, 4, 7)] == 0.0);
        assert(builder.GradientNorm[rowIndex(builder, 3, 4)] == 0.0);
    }
}

int main() {
    testSVSBuilderGradientNorms();
    return 0;
}
