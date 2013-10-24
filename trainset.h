#include "opencv2/core/core.hpp"

#include "rangeimagepoint.h"

class TrainingSetGenerator {
public:
    TrainingSetGenerator(
            float resolution,
            float width,
            float prob)
        : Resolution_(resolution)
        , Width_(width)
        , Prob_(prob)
    {
    }

    void generate(PointCloud const& input, cv::Mat * objects, cv::Mat * responses) {
        PointCloud classes [2];

        for (int i = 0; i < input.size(); ++i) {
            PointType point = input.at(i);
            if (pointHasNan(point)) {
                continue;
            }

            RangeImagePoint rip(point);
            for (int j = -Width_ + 1; j <= Width_; ++j) {
                if (tossCoin()) {
                    classes[j <= 0 ? 0 : 1].push_back(rip.shift(Resolution_, j));
                }
            }
        }

        int const total = classes[0].size() + classes[1].size();
        objects->create(total, 3, CV_32FC1);
        responses->create(total, 1, CV_32FC1);

        for (int i = 0; i < total; ++i) {
            auto p = (i < classes[0].size()
                    ? classes[0].at(i)
                    : classes[1].at(i - classes[0].size()))
                .getVector3fMap();
            for (int j = 0; j < 3; ++j) {
                objects->at<float>(i, j) = p(j);
            }
            responses->at<float>(i) = i < classes[0].size() ? 1 : -1;
        }
    }

private:
    bool tossCoin() const {
        return rand() % MODULE < Prob_ * MODULE;
    }

private:
    static int const MODULE = 100000;

private:
    float Resolution_;
    float Width_;
    float Prob_;
};
