#pragma once

#include "newsvm.h"
#include "griditer.h"

class Grid2Numbers {
private:
    typedef std::vector< std::vector< std::vector<int> > > Impl;

public:
    Grid2Numbers() {
    }

    Grid2Numbers(int nRows, int nCols)
    {
        Resize(nRows, nCols);
    }

    void Resize(int nRows, int nCols) {
        Impl_.resize(nRows, std::vector< std::vector<int> >(nCols));
    }

    int NRows() const {
        return Impl_.size();
    }

    int NCols() const {
        return Impl_[0].size();
    }

    template <class Action>
    void TraverseRectangle(int x0, int y0, int r, Action action) const {
        GridRadiusTraversal(NRows(), NCols())
            .TraverseRectangle(x0, y0, r,
                    [this, &action] (int x, int y) {
                        for (int idx : Impl_[x][y]) {
                            action(x, y, idx);
                        }
                    });
    }

    std::vector<int> & at(int i, int j) {
        return Impl_.at(i).at(j);
    }

private:
    Impl Impl_;
};

typedef std::vector< std::pair<int, int> > Number2Grid;

class GridNeighbourModificationStrategy : public IGradientModificationStrategy {
public:
    GridNeighbourModificationStrategy(
            int gridHeight,
            int gridWidth,
            Grid2Numbers const& grid2num,
            Number2Grid const& num2grid,
            float kernelWidth,
            float kernelThreshold,
            float resolution,
            int cacheSize)
        : GridHeight_(gridHeight)
        , GridWidth_(gridWidth)
        , Grid2Num_(grid2num)
        , Num2Grid_(num2grid)
        , MaxTotalNeighbors_(cacheSize / 8)
    {
        Radius_ = static_cast<int>(ceil(sqrt(-log(kernelThreshold)) * kernelWidth));
        Radius2Scaled_ = -log(kernelThreshold) * sqr(resolution * kernelWidth);
    }

    virtual void InitializeFor(SVM3D * parent);
    virtual void OptimizePivots(int * i, int * j);
    virtual void ReflectAlphaChange(int idx, SVMFloat delta);

    virtual float QValue(int i, int j);

private:
    void InitializeNeighbors(int idx);
    void LogAccess(int idx);
    void RegisterNewNeighbors(int num);
    void RepackNeighbors(int idx);
    void FreeNeighbors(int idx);

public:
    int NumCacheMisses = 0;
    int NumOptimizeFailures = 0;
    int NumNeighborsCalculations = 0;
    int TotalNeighborsProcessed = 0;

private:
    int GridHeight_;
    int GridWidth_;
    int Radius_;
    float Radius2Scaled_;

    Grid2Numbers const& Grid2Num_;
    Number2Grid const& Num2Grid_;

    int MaxTotalNeighbors_;
    int TotalNeighbours_ = 0;
    int HistoryIndex_ = 0;
    std::vector<int> History_;
    std::vector<int> LastAccess_;

    std::vector< std::vector<int> > Neighbors_;
    std::vector< std::vector<float> > QValues_;
};
