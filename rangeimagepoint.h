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

private:
     PointType Point_;
};

