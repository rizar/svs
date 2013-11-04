#include "common.h"

typedef double SVMFloat;
SVMFloat const SVM_INF = std::numeric_limits<SVMFloat>::max();
SVMFloat const SVM_EPS = std::numeric_limits<SVMFloat>::epsilon();

struct SegmentInfo {
    SVMFloat Up;
    SVMFloat Low;
    int UpIdx;
    int LowIdx;

    SegmentInfo()
        : Up(SVM_INF)
        , Low(-SVM_INF)
    {
    }

    void Init(int idx, int label, SVMFloat alpha, SVMFloat C, SVMFloat grad);
    bool Update(SegmentInfo const& left, SegmentInfo const& right);
};

class Solution {
public:
    void Init(int n, SVMFloat C, int const* labels);

    int UpperOutlier() const {
        return Segs_[1].UpIdx;
    }

    int LowerOutlier() const {
        return Segs_[1].LowIdx;
    }

    SVMFloat UpperValue() const {
        return Segs_[1].Up;
    }

    SVMFloat LowerValue() const {
        return Segs_[1].Low;
    }

    void Update(int idx, int label, float C);

    void DebugPrint(std::ostream & ostr);

public:
    std::vector<SVMFloat> Alphas;
    std::vector<SVMFloat> Grad;
    float Rho;

private:
    int N_;
    int M_;

    std::vector<SegmentInfo> Segs_;
};

class SVM3D {
public:
    SVM3D()
        : Iteration(0)
        , C_(1)
        , Gamma_(1)
        , MinusGamma_(-Gamma_)
        , Eps_(1e-3)
    {
    }

    void SetParams(SVMFloat C, SVMFloat gamma, SVMFloat eps) {
        C_ = C;
        Gamma_ = gamma;
        MinusGamma_ = -gamma;
        Eps_ = eps;
    }

    void Train(PointCloud const& points, std::vector<int> const& labels);

    SVMFloat const* Alphas() {
        return &Sol_.Alphas[0];
    }

private:
    SVMFloat KernelValue(int i, int j) const;
    SVMFloat QValue(int i, int j) const;

    void Init();
    bool Iterate();

    void CalcRho();
    void CalcSVCount();
    void CalcTargetFunction();

public:
    int Iteration;
    int SVCount;
    float TargetFunction;

private:
    SVMFloat C_;
    SVMFloat Gamma_;
    SVMFloat MinusGamma_;
    SVMFloat Eps_;

    int N_;
    PointType const* Points_;
    int const* Labels_;

    Solution Sol_;
};
