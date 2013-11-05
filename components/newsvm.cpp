#include "newsvm.h"

#include "utilities/prettyprint.hpp"

enum {
    LOW_SUPPORT_FLAG = 1,
    UP_SUPPORT_FLAG = 2
};

bool isLowerSupport(int label, SVMFloat alpha, SVMFloat C) {
    return (alpha < C && label == 1) || (alpha > 0 && label == -1);
}

bool isUpperSupport(int label, SVMFloat alpha, SVMFloat C) {
    return (alpha < C && label == -1) || (alpha > 0 && label == 1);
}

void SegmentInfo::Init(int idx, SVMFloat minusLabelTimesGrad, int status) {
    Up = SVM_INF;
    Low = -SVM_INF;

    if (status & LOW_SUPPORT_FLAG) {
        Low = minusLabelTimesGrad;
        LowIdx = idx;
    }
    if (status & UP_SUPPORT_FLAG) {
        Up = minusLabelTimesGrad;
        UpIdx = idx;
    }
}

bool SegmentInfo::Update(SegmentInfo const& left, SegmentInfo const& right) {
    SegmentInfo const& minUp = left.Up <= right.Up ? left : right;
    SegmentInfo const& maxLow = left.Low >= right.Low ? left : right;

    bool result = false;

    if (Up != minUp.Up || UpIdx != minUp.UpIdx) {
        Up = minUp.Up;
        UpIdx = minUp.UpIdx;
        result = true;
    }
    if (Low != maxLow.Low || LowIdx != maxLow.LowIdx) {
        Low = maxLow.Low;
        LowIdx = maxLow.LowIdx;
        result = true;
    }

    return result;
}

std::ostream & operator<<(std::ostream & ostr, SegmentInfo const& si) {
    ostr << "(" << si.Up << ", "  << si.UpIdx << ", " << si.Low << ", " << si.LowIdx << ")";
    return ostr;
}

void Solution::Init(int n, SVMFloat C, int const* labels) {
    N_ = n;
    Alphas.resize(N_);
    Grad.resize(N_, -1);

    Status_.resize(N_);
    for (int i = 0; i < N_; ++i) {
        UpdateStatus(i, labels[i], C);
    }

    for (M_ = 1; M_ < N_; M_ *= 2);
    Segs_.resize(2 * M_);

    for (int i = 0; i < N_; ++i) {
        Segs_[M_ + i].Init(i, -labels[i] * Grad[i], Status_[i]);
    }
    for (int i = M_ - 1; i >= 1; --i) {
        Segs_[i].Update(Segs_[2 * i], Segs_[2 * i + 1]);
    }
}

void Solution::UpdateStatus(int idx, int label, float C) {
    Status_[idx] = isLowerSupport(label, Alphas[idx], C)
        + (isUpperSupport(label, Alphas[idx], C) << 1);
}

void Solution::Update(int idx, int label) {
    Segs_[M_ + idx].Init(idx, -label * Grad[idx], Status_[idx]);
    for (int i = (M_ + idx) / 2; i > 0; i /= 2) {
        if (! Segs_[i].Update(Segs_[2 * i], Segs_[2 * i + 1])) {
            break;
        }
    }
}

void Solution::DebugPrint(std::ostream & ostr) {
    ostr << Alphas << ' ' << Grad << std::endl;
}

void SVM3D::Train(PointCloud const& points, std::vector<int> const& labels) {
    assert(points.size() == labels.size());
    N_ = points.size();

    // I'm sure that internally these are just continuous arrays
    Points_ = &points[0];
    Labels_ = &labels[0];

    Sol_.Init(N_, C_, Labels_);
    while (Iterate()) {
        Iteration++;
    }

    CalcRho();
    CalcSVCount();
    CalcTargetFunction();
}

bool SVM3D::Iterate() {
#ifndef NDEBUG
    Sol_.DebugPrint(std::cerr);
#endif

    // check for convergence
    if (Sol_.LowerValue() - Sol_.UpperValue() < Eps_) {
        return false;
    }

    int const i = Sol_.UpperOutlier();
    int const j = Sol_.LowerOutlier();

    SVMFloat const Qii = KernelValue(i, i);
    SVMFloat const Qij = KernelValue(i, j);
    SVMFloat const Qjj = KernelValue(j, j);

    SVMFloat & Gi = Sol_.Grad[i];
    SVMFloat & Gj = Sol_.Grad[j];

    SVMFloat & Ai = Sol_.Alphas[i];
    SVMFloat & Aj = Sol_.Alphas[j];
    SVMFloat const oldAi = Ai;
    SVMFloat const oldAj = Aj;

    if (Labels_[i] != Labels_[j])
    {
        SVMFloat const denomFabs = fabs(Qii + Qjj + 2 * Qij);
        SVMFloat const delta = (-Gi - Gj) / std::max(denomFabs, SVM_EPS);
        SVMFloat const diff = Ai - Aj;
        Ai += delta;
        Aj += delta;

        if (diff > 0 && Aj < 0 )
        {
            Aj = 0;
            Ai = diff;
        }
        else if( diff <= 0 && Ai < 0 )
        {
            Ai = 0;
            Aj = -diff;
        }

        if( diff > C_ - C_ && Ai > C_ )
        {
            Ai = C_;
            Aj = C_ - diff;
        }
        else if( diff <= C_ - C_ && Aj > C_ )
        {
            Aj = C_;
            Ai = C_ + diff;
        }
    }
    else
    {
        SVMFloat const denomFabs = fabs(Qii + Qjj - 2*Qij);
        SVMFloat const delta = (Gi - Gj) / std::max(denomFabs, SVM_EPS);
        SVMFloat const sum = Ai + Aj;
        Ai -= delta;
        Aj += delta;

        if (sum > C_ && Ai > C_)
        {
            Ai = C_;
            Aj = sum - C_;
        }
        else if (sum <= C_ && Aj < 0)
        {
            Aj = 0;
            Ai = sum;
        }

        if (sum > C_ && Aj > C_)
        {
            Aj = C_;
            Ai = sum - C_;
        }
        else if (sum <= C_ && Ai < 0)
        {
            Ai = 0;
            Aj = sum;
        }
    }

    Sol_.UpdateStatus(i, Labels_[i], C_);
    Sol_.UpdateStatus(j, Labels_[j], C_);

    Sol_.Update(i, Labels_[i]);
    Sol_.Update(j, Labels_[j]);

    SVMFloat const deltaAi = Ai - oldAi;
    SVMFloat const deltaAj = Aj - oldAj;

    for (int k = 0; k < N_; ++k) {
        Sol_.Grad[k] += QValue(i, k) * deltaAi;
        Sol_.Grad[k] += QValue(j, k) * deltaAj;
        Sol_.Update(k, Labels_[k]);
    }

    return true;
}

void SVM3D::CalcTargetFunction() {
    TargetFunction = 0.0;
    for (int i = 0; i < N_; ++i) {
        TargetFunction += Sol_.Alphas[i] * (Sol_.Grad[i] - 1);
    }
    TargetFunction *= 0.5;
}

void SVM3D::CalcSVCount() {
    SVCount = 0;
    for (int i = 0; i < N_; ++i) {
        if (Sol_.Alphas[i] > 0) {
            SVCount++;
        }
    }
}

void SVM3D::CalcRho() {
    int i, nFree = 0;
    double ub = SVM_INF, lb = -SVM_INF, sumFree = 0;

    for( i = 0; i < N_; i++ )
    {
        double yG = Labels_[i] * Sol_.Grad[i];

        if (Sol_.Alphas[i] == 0)
        {
            if (Labels_[i] > 0 ) {
                ub = std::min(ub, yG);
            } else {
                lb = std::max(lb, yG);
            }
        } else if (Sol_.Alphas[i] == C_)
        {
            if (Labels_[i] < 0) {
                ub = std::min(ub, yG);
            } else {
                lb = std::max(lb, yG);
            }
        } else {
            ++nFree;
            sumFree += yG;
        }
    }

    Sol_.Rho = nFree > 0 ? sumFree / nFree : (ub + lb) * 0.5;
}

SVMFloat SVM3D::KernelValue(int i, int j) const {
    SVMFloat const dist2 =
        sqr(Points_[i].x - Points_[j].x) +
        sqr(Points_[i].y - Points_[j].y) +
        sqr(Points_[i].z - Points_[j].z);
    return  exp(MinusGamma_ * dist2);
}

SVMFloat SVM3D::QValue(int i, int j) const {
    return Labels_[i] * Labels_[j] * KernelValue(i, j);
}

