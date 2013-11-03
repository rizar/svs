#include "common.h"
#include "fastsvm.h"

class ModelChecker {
public:
    ModelChecker()
        : CheckOffset(3)
        , CheckWidth(2)
        , FailedPoints_(0)
        , FailedChecks_(0)
        , NumChecks_(0)
    {
    }

    void Check(DecisionFunction const& df, PointType const& point, float step);

    float Accuracy() const;
    float LearntRatio() const;

public:
    int CheckOffset;
    int CheckWidth;

private:
    int FailedPoints_;
    int FailedChecks_;
    int NumChecks_;
};
