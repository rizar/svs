#pragma once

#include "common.h"

class RangeImagePoint {
public:
    RangeImagePoint(PointType const& point)
        : Point_(point)
    {
    }

    PointType ort() {
        PointType res = Point_;
        res.getVector3fMap() /= res.getVector3fMap().norm();
        return res;
    }

    PointType shift(double resolution, int k = 1) {
        PointType res = Point_;
        res.getVector3fMap() += ort().getVector3fMap() * resolution * k;
        return res;
    }

    bool isLearn(FastSVM const& svm, double resolution) {
        for (int j = -5; j <= -3; ++j) {
            if (svm.fastPredict(shift(resolution, j)) == -1) {
                return false;
            }
        }
        for (int j = 4; j <= 5; ++j) {
            if (svm.fastPredict(shift(resolution, j)) == 1) {
                return false;
            }
        }
        return true;
    }

private:
     PointType Point_;
};

