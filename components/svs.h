#include "common.h"
#include "gridstrategy.h"
#include "df.h"
#include "svsparams.h"

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
    std::vector<float> DistToNN; // raw indices
    std::vector<int> RawIndex2Pixel; // raw indices
    std::vector<int> Pixel2RawIndex; // pixel indices
};

class SVSBuilder : public BaseBuilder {
public:
    void SetParams(SVSParams const& params);

    void SetInputCloud(PointCloud::ConstPtr input);
    void GenerateTrainingSet();

    void Learn();
    void InitSVM(std::vector<SVMFloat> const& alphas);

    void FeaturePointSearch();

    void CalcGradients();
    void CalcNormals();

    SVM3D const& SVM() {
        return SVM_;
    }

    void ToImageLayout(std::vector<float> const& data, std::vector<float> * res);

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

    std::vector<float> GradientNorms; // row indices
    std::vector<float> NumCloseSV; // row indices
    std::vector<float> AdjustedGradientNorms; // row indices
    NormalCloud::Ptr Gradients; // pixel indices

    NormalCloud::Ptr Normals; // pixel indices

    PointCloud::Ptr FeaturePoints;

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
