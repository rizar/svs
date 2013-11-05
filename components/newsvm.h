#include "common.h"

#include <memory>

typedef double SVMFloat;
SVMFloat const SVM_INF = std::numeric_limits<SVMFloat>::max();
SVMFloat const SVM_EPS = std::numeric_limits<SVMFloat>::epsilon();

class IGradientModificationStrategy;
class SVM3D;

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

    void Init(int idx, SVMFloat grad, int status);
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

    void UpdateStatus(int idx, int label, float C);
    void Update(int idx, int label);

    void DebugPrint(std::ostream & ostr);

public:
    std::vector<SVMFloat> Alphas;
    std::vector<SVMFloat> Grad;
    float Rho;

private:
    int N_;
    int M_;

    std::vector<SegmentInfo> Segs_;
    std::vector<int> Status_;
};

class IGradientModificationStrategy {
public:
    virtual void ReflectAlphaChange(int idx, SVMFloat deltaAlpha) = 0;
    void ModifyGradient(int idx, SVMFloat value);

    virtual void InitializeFor(SVM3D * parent) {
        Parent_ = parent;
    }

    SVM3D * Parent() {
        return Parent_;
    }

private:
    SVM3D * Parent_;
};

class DefaultGradientModificationStrategy : public IGradientModificationStrategy {
public:
    virtual void ReflectAlphaChange(int idx, SVMFloat deltaAlpha);
};

class PrecomputedGradientModificationStrategy : public IGradientModificationStrategy {
public:
    virtual void InitializeFor(SVM3D * parent);
    virtual void ReflectAlphaChange(int idx, SVMFloat deltaAlpha);

private:
    // force float here because of memory
    std::vector< std::vector<float> > QValues_;
};

class SVM3D {
    friend class IGradientModificationStrategy;

public:
    SVM3D()
        : Iteration(0)
        , C_(1)
        , Gamma_(1)
        , MinusGamma_(-Gamma_)
        , Eps_(1e-3)
    {
        Strategy_.reset(new DefaultGradientModificationStrategy);
    }

    void SetStrategy(IGradientModificationStrategy * strategy) {
        Strategy_.reset(strategy);
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

    int PointCount() const {
        return N_;
    }

    PointType const* Points() const {
        return Points_;
    }

    int const* Labels() const {
        return Labels_;
    }

    SVMFloat KernelValue(int i, int j) const;
    SVMFloat QValue(int i, int j) const;

private:
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

    std::shared_ptr<IGradientModificationStrategy> Strategy_;
};