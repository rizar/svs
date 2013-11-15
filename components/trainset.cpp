#include "trainset.h"

#include "pcl/features/normal_3d.h"

void TrainingSetGenerator::GenerateFromSensor(PointCloud const& input,
                std::vector<float> const& localRes)
{
    Grid2Num.resize(input.height, std::vector< std::vector<int> >(input.width));

    int indexNoNan = 0;
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            PointType point = input.at(x, y);
            if (pointHasNan(point)) {
                continue;
            }

            RangeImagePoint rip(point);
            for (int j = -Width_ + 1; j <= Width_; ++j) {
                if (TossCoin()) {
                    // localRes is indexed by nan-removed number
                    PointType shifted = rip.shift(localRes.at(indexNoNan), j);
                    std::pair<int, int> pixel = {y, x};

                    Objects.push_back(shifted);
                    Labels.push_back(j <= 0 ? +1 : -1);
                    Num2Grid.push_back(pixel);
                    Grid2Num[pixel.first][pixel.second].push_back(Objects.size() - 1);
                }
            }

            indexNoNan++;
        }
    }
}

void TrainingSetGenerator::GenerateUsingNormals(const PointCloud & input, const NormalCloud & normals,
        const std::vector<float> & localRes)
{
    Grid2Num.resize(input.height, std::vector< std::vector<int> >(input.width));
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
            nv3f *= localRes[indexNoNan];

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
    Grid2Num[y][x].push_back(Objects.size() - 1);
    return Objects.size() - 1;
}
