#include "gridstrategy.h"

void GridNeighbourModificationStrategy::InitializeFor(SVM3D * parent) {
    IGradientModificationStrategy::InitializeFor(parent);

    QValues_.resize(parent->PointCount());
    Neighbors_.resize(parent->PointCount());
    LastAccess_.resize(parent->PointCount(), -1);
}

void GridNeighbourModificationStrategy::OptimizePivots(int * i, int * j) {
    float best = Parent()->PivotsOptimality(*i, *j);
    int const startJ = *j;

    InitializeNeighbors(*i);
    std::vector<int> const& nbh = Neighbors_[*i];
    std::vector<float> const& qvls = QValues_[*i];

    for (int k = 0; k < nbh.size(); ++k) {
        float const current = Parent()->PivotsOptimality(*i, nbh[k], qvls[k]);
        if (current > best) {
            best = current;
            *j = nbh[k];
        }
    }

    NumOptimizeFailures_ += startJ == *j;
}

void GridNeighbourModificationStrategy::ReflectAlphaChange(int idx, SVMFloat delta) {
    InitializeNeighbors(idx);
    std::vector<int> const& nbh = Neighbors_[idx];
    std::vector<float> const& qv = QValues_[idx];

    for (int k = 0; k < nbh.size(); ++k) {
        ModifyGradient(nbh[k], qv[k] * delta);
    }
}

void GridNeighbourModificationStrategy::PrintStatistics(std::ostream & ostr) const {
    ostr << "GridStrategy: Radius: " << Radius_ << std::endl;
    ostr << "GridStrategy: Number of cache misses: " << NumCacheMisses_ << std::endl;
    ostr << "GridStrategy: Number of optimization failures: " << NumOptimizeFailures_ << std::endl;
    ostr << "GridStrategy: Average number of neighbors: " <<
        static_cast<float>(TotalNeighborsProcessed_) / NumNeighborsCalculations_ << std::endl;
}

void GridNeighbourModificationStrategy::InitializeNeighbors(int idx) {
    bool firstTime = LastAccess_[idx] == -1;
    LogAccess(idx);

    std::vector<int> & nbh = Neighbors_[idx];
    std::vector<float> & qvls = QValues_[idx];

    if (nbh.size()) {
        return;
    }
    if (! firstTime) {
        NumCacheMisses_++;
    }

    int const y = Num2Grid_[idx].first;
    int const x = Num2Grid_[idx].second;

    for (int j = -std::min(Radius_, y); j <= std::min(GridHeight_ - 1 - y, Radius_); ++j) {
        for (int i = -std::min(Radius_, x); i <= std::min(GridWidth_ - 1 - x, Radius_); ++i) {
            std::vector<int> const& els = Grid2Num_[y + j][x + i];
            for (int k = 0; k < els.size(); ++k) {
                int const nbhIdx = els[k];

                float const dist2 = Parent()->Dist2(idx, nbhIdx);
                if (dist2 <= Radius2Scaled_) {
                    nbh.push_back(nbhIdx);
                    qvls.push_back(Parent()->QValue(idx, nbhIdx, dist2));
                }
            }
        }
    }

    NumNeighborsCalculations_++;
    TotalNeighborsProcessed_ += nbh.size();

    RepackNeighbors(idx);
    RegisterNewNeighbors(nbh.size());
}

void GridNeighbourModificationStrategy::LogAccess(int idx) {
    History_.push_back(idx);
    LastAccess_[idx] = History_.size() - 1;
}

void GridNeighbourModificationStrategy::RegisterNewNeighbors(int num) {
    TotalNeighbours_ += num;
    while (TotalNeighbours_ > MaxTotalNeighbors_) {
        int const pointIdx = History_[HistoryIndex_];
        if (LastAccess_[pointIdx] == HistoryIndex_) {
            TotalNeighbours_ -= Neighbors_[pointIdx].size();
            FreeNeighbors(pointIdx);
        }
        HistoryIndex_++;
    }
}

void GridNeighbourModificationStrategy::RepackNeighbors(int idx) {
    std::vector<int> tmpInts(Neighbors_[idx]);
    std::vector<float> tmpFloats(QValues_[idx]);
    Neighbors_[idx].swap(tmpInts);
    QValues_[idx].swap(tmpFloats);
}

void GridNeighbourModificationStrategy::FreeNeighbors(int idx) {
    std::vector<int> tmpInts;
    std::vector<float> tmpFloats;
    Neighbors_[idx].swap(tmpInts);
    QValues_[idx].swap(tmpFloats);
}