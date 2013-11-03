#include "check.h"
#include "rangeimagepoint.h"

void ModelChecker::Check(DecisionFunction const& df, PointType const& point, float step) {
    NumChecks_++;

    bool failed = false;
    for (int i = 0; i < CheckWidth; ++i) {
        RangeImagePoint rip(point);
        if (df.decisionFunction(rip.shift(step, -CheckOffset - i)) < 1e-5) {
            failed = true;
            FailedPoints_++;
        }
        if (df.decisionFunction(rip.shift(step, CheckOffset + 1 + i)) > -1e-5) {
            failed = true;
            FailedPoints_++;
        }
    }
    FailedChecks_ += failed ? 1 : 0;
}

float ModelChecker::Accuracy() const {
    return 1 - FailedPoints_ / static_cast<float>(CheckWidth * 2 * NumChecks_);
}

float ModelChecker::LearntRatio() const {
    return 1 - FailedChecks_ / static_cast<float>(NumChecks_);
}
