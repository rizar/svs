#include "trainset.h"

void TrainingSetGenerator::Generate(PointCloud const& input,
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
