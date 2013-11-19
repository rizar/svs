#include "trainset.h"

#include "pcl/features/normal_3d.h"

void TrainingSetGenerator::GenerateFromSensor(PointCloud const& input,
                std::vector<float> const& localRes)
{
    Grid2Num.Resize(input.height, input.width);
    Pixel2Num.resize(input.height * input.width);

    int indexNoNan = 0;
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            PointType point = input.at(x, y);
            int & pixelNum = Pixel2Num[y * input.width + x];

            if (pointHasNan(point)) {
                pixelNum = -1;
                continue;
            }

            RangeImagePoint rip(point);
            for (int j = -Width_ + 1; j <= Width_; ++j) {
                if (TossCoin()) {
                    // localRes is indexed by nan-removed number
                    PointType shifted = rip.shift(Step_ * localRes.at(indexNoNan), j);
                    int num = AddPoint(x, y, shifted, j <= 0 ? +1 : -1);
                    if (j == 0) {
                        pixelNum = num;
                    }
                }
            }

            indexNoNan++;
        }
    }
}

void TrainingSetGenerator::GenerateUsingNormals(const PointCloud & input, const NormalCloud & normals,
        const std::vector<float> & localRes)
{
    Grid2Num.Resize(input.height, input.width);
    Pixel2Num.resize(input.height * input.width);

    int indexNoNan = 0;
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            PointType point = input.at(x, y);
            NormalType normal = normals.at(x, y);
            int & num = Pixel2Num[y * input.width + x];

            if (pointHasNan(point) || pcl_isnan(normal.normal_x)) {
                num = -1;
                continue;
            }

            auto nv3f = normal.getNormalVector3fMap();
            pcl::flipNormalTowardsViewpoint(point, 0, 0, 0, nv3f[0], nv3f[1], nv3f[2]);
            nv3f *= -1;
            nv3f /= nv3f.norm();
            nv3f *= Step_ * localRes[indexNoNan];

            num = AddPoint(x, y, point, +1);

            PointType shifted(point);
            shifted.getVector3fMap() += nv3f;
            AddPoint(x, y, shifted, -1);

            indexNoNan++;
        }
    }
}

int TrainingSetGenerator::AddPoint(int x, int y, PointType const& point, float label) {
    Objects.push_back(point);
    Labels.push_back(label);
    Num2Grid.push_back({y, x});
    Grid2Num.at(y, x).push_back(Objects.size() - 1);
    return Objects.size() - 1;
}
