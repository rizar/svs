#include "newsvm.h"

#include "utilities/prettyprint.hpp"

enum {
    LOW_SUPPORT_FLAG = 1,
    UP_SUPPORT_FLAG = 2
};

bool isLowerSupport(float label, SVMFloat alpha, SVMFloat C) {
    return (alpha < C && label == 1) || (alpha > 0 && label == -1);
}

bool isUpperSupport(float label, SVMFloat alpha, SVMFloat C) {
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

void Solution::Init(int n, SVMFloat C, float const* labels) {
    N_ = n;
    Alphas.resize(N_);
    Grad.resize(N_, -1);

    Status.resize(N_);
    for (int i = 0; i < N_; ++i) {
        UpdateStatus(i, labels[i], C);
    }

    for (M_ = 1; M_ < N_; M_ *= 2);
    Segs_.resize(2 * M_);

    for (int i = 0; i < N_; ++i) {
        Segs_[M_ + i].Init(i, -labels[i] * Grad[i], Status[i]);
    }
    for (int i = M_ - 1; i >= 1; --i) {
        Segs_[i].Update(Segs_[2 * i], Segs_[2 * i + 1]);
    }
}

void Solution::UpdateStatus(int idx, float label, SVMFloat C) {
    Status[idx] = isLowerSupport(label, Alphas[idx], C)
        + (isUpperSupport(label, Alphas[idx], C) << 1);
}

void Solution::Update(int idx, float label) {
    Segs_[M_ + idx].Init(idx, -label * Grad[idx], Status[idx]);
    for (int i = (M_ + idx) / 2; i > 0; i /= 2) {
        if (! Segs_[i].Update(Segs_[2 * i], Segs_[2 * i + 1])) {
            break;
        }
    }
}

void Solution::DebugPrint(std::ostream & ostr) {
    ostr << Alphas << ' ' << Grad << std::endl;
}

void IGradientModificationStrategy::InitializeFor(SVM3D * parent) {
    Parent_ = parent;
    Version_.resize(Parent_->PointCount(), -1);
}

void IGradientModificationStrategy::ModifyGradient(int idx, SVMFloat value) {
    Parent()->Sol_.Grad[idx] += value;
    if (Version_[idx] < Parent()->Iteration) {
        Version_[idx] = Parent()->Iteration;
        ToUpdate_.push_back(idx);
    }
}

void IGradientModificationStrategy::StartIteration() {
    ToUpdate_.clear();
}

void IGradientModificationStrategy::FinishIteration() {
    for (int i = 0; i < ToUpdate_.size(); ++i) {
        int const idx = ToUpdate_[i];
        Parent()->Sol_.Update(idx, Parent()->Labels_[idx]);
    }
}

void DefaultGradientModificationStrategy::ReflectAlphaChange(int idx, SVMFloat deltaAlpha) {
    for (int k = 0; k < Parent()->PointCount(); ++k) {
        ModifyGradient(k, Parent()->QValue(idx, k) * deltaAlpha);
    }
}

void PrecomputedGradientModificationStrategy::InitializeFor(SVM3D * parent) {
    IGradientModificationStrategy::InitializeFor(parent);

    QValues_.resize(Parent()->PointCount());
    for (int i = 0; i < Parent()->PointCount(); ++i) {
        QValues_[i].resize(Parent()->PointCount());
        for (int j = 0; j < Parent()->PointCount(); ++j) {
            QValues_[i][j] = Parent()->QValue(i, j);
        }
    }
}

void PrecomputedGradientModificationStrategy::ReflectAlphaChange(int idx, SVMFloat deltaAlpha) {
    for (int k = 0; k < Parent()->PointCount(); ++k) {
        ModifyGradient(k, QValues_[idx][k] * deltaAlpha);
    }
}

void SVM3D::Train(PointCloud const& points, std::vector<float> const& labels) {
    assert(points.size() == labels.size());
    N_ = points.size();

    // I'm sure that internally these are just continuous arrays
    Points_ = &points[0];
    Labels_ = &labels[0];

    Strategy_->InitializeFor(this);
    Sol_.Init(N_, C_, Labels_);
    while (Iterate()) {
        Iteration++;
    }

    CalcRho();
    CalcSVCount();
    CalcTargetFunction();
}

bool SVM3D::Iterate() {
    // check for convergence
    if (Sol_.LowerValue() - Sol_.UpperValue() < Eps_) {
        return false;
    }

    Strategy_->StartIteration();

    int i = Sol_.LowerOutlier();
    int j = Sol_.UpperOutlier();
    Strategy_->OptimizePivots(&i, &j);

    SVMFloat const Qii = 1;
    SVMFloat const Qjj = 1;
    SVMFloat const Qij = QValue(i, j);

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

    Strategy_->ReflectAlphaChange(i, deltaAi);
    Strategy_->ReflectAlphaChange(j, deltaAj);
    Strategy_->FinishIteration();

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

SVMFloat SVM3D::PivotsOptimality(int i, int j) const {
    bool can = (Sol_.Status[i] & LOW_SUPPORT_FLAG) && (Sol_.Status[j] & UP_SUPPORT_FLAG);
    if (! can) {
        return 0.0;
    }

    SVMFloat const ygi = -Labels_[i] * Sol_.Grad[i];
    SVMFloat const ygj = Labels_[j] * Sol_.Grad[j];
    SVMFloat const bij = ygi + ygj;
    if (bij <= SVM_EPS) {
        return 0.0;
    }

    SVMFloat const aij = std::max(2 - 2 * KernelValue(i, j), 1e-12);
    return sqr(bij) / aij;
}
