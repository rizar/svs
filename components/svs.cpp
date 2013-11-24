#include "svs.h"
#include "trainset.h"
#include "analysis.h"

#include "pcl/features/integral_image_normal.h"
#include "pcl/common/time.h"
#include "pcl/filters/filter.h"

#include <fstream>

void SupportVectorShape::LoadAsText(std::istream & istr) {
    float x, y, z;
    while (istr >> x >> y >> z) {
        FeaturePoints_->push_back(createPoint<PointType>(x, y, z));
    }
}

void SupportVectorShape::SaveAsText(std::ostream & ostr) {
    for (int i = 0; i < FeaturePoints_->size(); ++i) {
        PointType const& p = FeaturePoints_->at(i);
        ostr << p.x << ' ' << p.y << ' ' << p.z << '\n';
    }
    ostr.flush();
}

void BaseBuilder::SetInputCloud(PointCloud::ConstPtr input) {
    Input = input;

    Height_ = Input->height;
    Width_ = Input->width;

    std::vector<int> tmp;
    InputNoNan.reset(new PointCloud);
    pcl::removeNaNFromPointCloud(*Input, *InputNoNan, tmp);
    CalcIndicesInOriginal();

    InputKDTree_.reset(new pcl::search::KdTree<PointType>);
    InputKDTree_->setInputCloud(InputNoNan);
    Resolution = computeCloudResolution(InputNoNan, *InputKDTree_);
}

void BaseBuilder::CalcIndicesInOriginal() {
    assert(RawIndex2Pixel.empty());
    Pixel2RawIndex.resize(Input->size(), -1);
    for (int i = 0; i < Input->size(); ++i) {
        if (pointHasNan(Input->at(i))) {
            continue;
        }
        Pixel2RawIndex[i] = RawIndex2Pixel.size();
        RawIndex2Pixel.push_back(i);
    }
}

void BaseBuilder::CalcDistanceToNN() {
    if (DistToNN.size()) {
        return;
    }

    std::vector<int> indices;
    std::vector<float> dist2;
    DistToNN.resize(InputNoNan->size());
    for (int i = 0; i < InputNoNan->size(); ++i) {
        InputKDTree_->nearestKSearch(i, 2, indices, dist2);
        DistToNN[i] = sqrt(dist2[1]);
    }
}

void SVSBuilder::SetParams(const SVSParams &params) {
    Params_ = params;
}

void SVSBuilder::GenerateTrainingSet() {
    CalcDistanceToNN();

    srand(Params_.Seed);
    TrainingSetGenerator tsg(
            Params_.BorderWidth,
            Params_.TakeProb,
            Params_.StepWidth);
    if (Params_.UseNormals) {
        CalcNormals();
        tsg.GenerateUsingNormals(*Input, *Normals, DistToNN);
    } else {
        tsg.GenerateFromSensor(*Input, DistToNN);
    }

    Objects.reset(new PointCloud(tsg.Objects));
    Labels = tsg.Labels;
    Num2Grid_ = tsg.Num2Grid;
    Grid2Num_ = tsg.Grid2Num;
    Pixel2TrainNum = tsg.Pixel2Num;
}

void SVSBuilder::CalcNormals() {
    if (Normals.get()) {
        return;
    }
    Normals.reset(new NormalCloud);

    pcl::IntegralImageNormalEstimation<PointType, NormalType> ne;
    ne.setNormalSmoothingSize(Params_.SmoothingRange);
    ne.setInputCloud(Input);
    ne.compute(*Normals);
}

void SVSBuilder::SetInputCloud(PointCloud::ConstPtr input) {
    BaseBuilder::SetInputCloud(input);
    Gamma = 1 / sqr(Params_.KernelWidth * Resolution);
    PixelRadius = static_cast<int>(sqrt(-log(Params_.KernelThreshold))
        * Params_.KernelWidth);
    Radius2 = -log(Params_.KernelThreshold)
        * sqr(Params_.KernelWidth * Resolution);
}

void SVSBuilder::InitSVM(std::vector<SVMFloat> const& alphas) {
    assert(Objects->size() == alphas.size());
    SVM_.Init(*Objects, Labels, alphas);
    BuildGrid2SV();
}

void SVSBuilder::Learn() {
    if (Params_.UseGrid) {
        Strategy.reset(new GridNeighbourModificationStrategy(
                    Input->height, Input->width,
                    Grid2Num_, Num2Grid_,
                    Params_.KernelWidth,
                    Params_.KernelThreshold,
                    Resolution,
                    Params_.CacheSize));
        SVM_.SetStrategy(Strategy);
    }
    SVM_.SetParams(Params_.MaxAlpha, Gamma, Params_.TerminateEps);

    {
        pcl::ScopeTime st("SVM");
        SVM_.Train(*Objects, Labels);
    }

    BuildGrid2SV();
}

void SVSBuilder::CalcGradients() {
    if (GradientNorms.size()) {
        return;
    }
    pcl::ScopeTime st("CalcGradientNorms");

    Gradients.reset(new NormalCloud);
    Gradients->points.resize(Input->size(), toNormal(nanPoint()));
    NumCloseSV.resize(InputNoNan->size());
    GradientNorms.resize(InputNoNan->size());
    AdjustedGradientNorms.resize(InputNoNan->size());

    DecisionFunction df;
    for (int i = 0; i < InputNoNan->size(); ++i) {
        PointType const& point = InputNoNan->at(i);
        int const pixel = RawIndex2Pixel.at(i);
        int const y = pixel / Width_;
        int const x = pixel % Width_;
        BuildDF(y, x, &df);
        auto const grad = df.Gradient(point);
        Gradients->at(RawIndex2Pixel[i]) = toNormal(grad);
        NumCloseSV[i] = df.SVCount();
        GradientNorms[i] = grad.getVector3fMap().norm();
        AdjustedGradientNorms[i] = NumCloseSV[i] ? GradientNorms[i] / NumCloseSV[i] : 0;
    }
}

void SVSBuilder::BuildGrid2SV() {
    Grid2SV_.Resize(Height_, Width_);
    for (int i = 0; i < Objects->size(); ++i) {
        if (SVM().Alphas()[i] > 0) {
            auto pos = Num2Grid_[i];
            Grid2SV_.at(pos.first, pos.second).push_back(i);
        }
    }
}

void SVSBuilder::BuildDF(int y, int x, DecisionFunction * df) {
    df->Reset(Gamma, SVM().Rho);
    PointType const& point = Input->at(x, y);
    Grid2SV_.TraverseRectangle(y, x, PixelRadius, [this, &point, &df] (int /*y*/, int /*x*/, int idx) {
                PointType const& sv = Objects->operator[](idx);
                if (pcl::squaredEuclideanDistance(point, sv) <= Radius2) {
                    df->AddSupportVector(sv, SVM().Alphas()[idx]);
                }
            });
}

void SVSBuilder::ToImageLayout(std::vector<float> const& data, std::vector<float> * res) {
    res->resize(Input->size(), -1);
    for (int i = 0; i < data.size(); ++i) {
        res->at(RawIndex2Pixel[i]) = data[i];
    }
}
