#pragma once

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
    void Init(int n, SVMFloat C, float const* labels);

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

    void UpdateStatus(int idx, float label, SVMFloat C);
    void Update(int idx, float label);

    void DebugPrint(std::ostream & ostr);


public:
    std::vector<SVMFloat> Alphas;
    std::vector<SVMFloat> Grad;
    std::vector<int> Status;
    float Rho;

public:
    int N_;
    int M_;

    std::vector<SegmentInfo> Segs_;
};

class IGradientModificationStrategy {
    friend class SVM3D;

public:
    virtual void OptimizePivots(int * /*i*/, int * /*j*/) {
    }

    virtual void ReflectAlphaChange(int idx, SVMFloat deltaAlpha) = 0;
    void ModifyGradient(int idx, SVMFloat value);
    virtual void InitializeFor(SVM3D * parent);

    SVM3D * Parent() {
        return Parent_;
    }

    virtual float QValue(int i, int j);

private:
    void StartIteration();
    void FinishIteration();

private:
    std::vector<int> Version_;
    std::vector<int> ToUpdate_;

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
    {
        Strategy_.reset(new DefaultGradientModificationStrategy);
    }

    void SetStrategy(IGradientModificationStrategy * strategy) {
        Strategy_.reset(strategy);
    }

    void SetStrategy(std::shared_ptr<IGradientModificationStrategy> strategy) {
        Strategy_ = strategy;
    }

    void SetParams(SVMFloat C, SVMFloat gamma, SVMFloat eps) {
        C_ = C;
        Gamma_ = gamma;
        MinusGamma_ = -gamma;
        Eps_ = eps;
    }

    void Init(PointCloud const& points, std::vector<float> const& labels,
            std::istream & istr);
    void SaveAlphas(std::ostream & ostr);
    void Train(PointCloud const& points, std::vector<float> const& labels);

    SVMFloat const* Alphas() {
        return &Sol_.Alphas[0];
    }

    int PointCount() const {
        return N_;
    }

    PointType const* Points() const {
        return Points_;
    }

    float const* Labels() const {
        return Labels_;
    }

    SVMFloat Dist2(int i, int j) const {
        return sqr(Points_[i].x - Points_[j].x) +
               sqr(Points_[i].y - Points_[j].y) +
               sqr(Points_[i].z - Points_[j].z);
    }

    SVMFloat KernelValue(float dist2) const {
        return exp(MinusGamma_ * dist2);
    }

    SVMFloat KernelValue(int i, int j) const {
        return exp(MinusGamma_ * Dist2(i, j));
    }

    SVMFloat QValue(int i, int j, float dist2) const {
        return Labels_[i] * Labels_[j] * KernelValue(dist2);
    }

    SVMFloat QValue(int i, int j) const {
        return Labels_[i] * Labels_[j] * KernelValue(i, j);
    }

    SVMFloat PivotsOptimality(int i, int j) const;
    SVMFloat PivotsOptimality(int i, int j, float Qij) const;

private:
    void Init();
    bool Iterate();

    void CalcRho();
    void CalcSVCount();
    void CalcTargetFunction();

public:
    int Iteration = 0;
    int SVCount = 0;
    float TargetFunction = 0.0;

private:
    SVMFloat C_ = 1;
    SVMFloat Gamma_ = 1;
    SVMFloat MinusGamma_ = -1;
    SVMFloat Eps_ = 1e-3;

    int N_;
    PointType const* Points_;
    float const* Labels_;

    Solution Sol_;

    std::shared_ptr<IGradientModificationStrategy> Strategy_;
};
