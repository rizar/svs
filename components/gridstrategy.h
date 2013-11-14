#pragma once

#include "newsvm.h"

class GridNeighbourModificationStrategy : public IGradientModificationStrategy {
public:
    typedef std::vector< std::vector< std::vector<int> > > Grid2Num;
    typedef std::vector< std::pair<int, int> > Num2Grid;

public:
    GridNeighbourModificationStrategy(
            int gridHeight,
            int gridWidth,
            Grid2Num const& grid2num,
            Num2Grid const& num2grid,
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
        Radius2Scaled_ = sqr(Radius_ * resolution);
    }

    virtual void InitializeFor(SVM3D * parent);
    virtual void OptimizePivots(int * i, int * j);
    virtual void ReflectAlphaChange(int idx, SVMFloat delta);

    void PrintStatistics(std::ostream & ostr) const;

private:
    void InitializeNeighbors(int idx);
    void LogAccess(int idx);
    void RegisterNewNeighbors(int num);
    void RepackNeighbors(int idx);
    void FreeNeighbors(int idx);

private:
    int GridHeight_;
    int GridWidth_;
    int Radius_;
    float Radius2Scaled_;

    Grid2Num const& Grid2Num_;
    Num2Grid const& Num2Grid_;

    int MaxTotalNeighbors_;
    int TotalNeighbours_ = 0;
    int HistoryIndex_ = 0;
    std::vector<int> History_;
    std::vector<int> LastAccess_;

    std::vector< std::vector<int> > Neighbors_;
    std::vector< std::vector<float> > QValues_;

    int NumCacheMisses_ = 0;
    int NumOptimizeFailures_ = 0;
    int NumNeighborsCalculations_ = 0;
    int TotalNeighborsProcessed_ = 0;
};
