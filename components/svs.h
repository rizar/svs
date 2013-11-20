#include "common.h"
#include "gridstrategy.h"
#include "df.h"

#include "pcl/search/kdtree.h"

class SupportVectorShape {
public:
    SupportVectorShape()
        : FeaturePoints_(new PointCloud)
    {
    }

    SupportVectorShape(PointCloud::Ptr featurePoints)
        : FeaturePoints_(featurePoints)
    {
    }

    PointCloud::Ptr FeaturePoints() {
        return FeaturePoints_;
    }

    PointCloud::ConstPtr FeaturePoints() const {
        return FeaturePoints_;
    }

    void LoadAsText(std::istream & istr);
    void SaveAsText(std::ostream & ostr);

private:
    PointCloud::Ptr FeaturePoints_;
};

struct SVSParams {
    int Seed = 1;

    float MaxAlpha = 32;
    float KernelWidth = 5;
    float KernelThreshold = 1e-3;
    float TerminateEps = 1e-2;

    float SmoothingRange = 5;
    int BorderWidth = 1;
    float StepWidth = 1;
    float TakeProb = 1.0;

    int NumFP = 100;

    size_t CacheSize = 1 << 30;

    bool UseGrid = true;
    bool UseNormals = true;
};

class BaseBuilder {
protected:
    typedef pcl::search::KdTree<PointType> KDTreeType;

public:
    void SetInputCloud(PointCloud::ConstPtr input);
    void CalcIndicesInOriginal();
    void CalcDistanceToNN();

    KDTreeType::Ptr InputKDTree_;

public:
    float Resolution;

    int Width_;
    int Height_;

    PointCloud::ConstPtr Input;
    PointCloud::Ptr InputNoNan;
    std::vector<float> DistToNN;
    std::vector<int> RowIndex2Pixel;
    std::vector<int> Pixel2RowIndex;
};

class SVSBuilder : public BaseBuilder {
public:
    void SetParams(SVSParams const& params);

    void SetInputCloud(PointCloud::ConstPtr input);
    void GenerateTrainingSet();
    void Learn();
    void InitSVM(std::vector<SVMFloat> const& alphas);

    void CalcGradients();
    void CalcNormals();

    SVM3D const& SVM() {
        return SVM_;
    }

private:
    void BuildDF(int y, int x, DecisionFunction * df);
    void BuildGrid2SV();

public:
    float Gamma;
    float Radius2;
    float PixelRadius;

    PointCloud::Ptr Objects;
    std::vector<float> Labels;
    std::vector<int> Pixel2TrainNum; // pixel indices

    float MinGradientNorm;
    float MaxGradientNorm;
    std::vector<float> GradientNorms; // row indices
    NormalCloud::Ptr Gradients; // pixel indices

    NormalCloud::Ptr Normals; // pixel indices

    std::shared_ptr<GridNeighbourModificationStrategy> Strategy;

private:
// basic params
    SVSParams Params_;

// auxillary data
    Grid2Numbers Grid2Num_;
    Grid2Numbers Grid2SV_;
    Number2Grid Num2Grid_;

// workhorse
    SVM3D SVM_;
};
