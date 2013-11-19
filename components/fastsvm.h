#pragma once

#include "common.h"
#include "df.h"

#include "pcl/common/distances.h"
#include "pcl/search/octree.h"
#include "pcl/registration/bfgs.h"

#define USE_MY_SVM

#ifdef USE_MY_SVM
#include "mysvm.h"
typedef My::CvSVMParams BaseSVMParams;
typedef My::CvSVM BaseSVM;
#else
#include "opencv2/ml/ml.hpp"
typedef CvSVMParams BaseSVMParams;
typedef CvSVM BaseSVM;
#endif

class GradientSquaredNormFunctor : public BFGSDummyFunctor<double, 3> {
public:
    GradientSquaredNormFunctor(DecisionFunction & df)
        : FCalled(0)
        , DFCalled(0)
        , DF_(df)
    {
    }

    virtual Scalar operator()(VectorType const& x) {
        FCalled++;
        PointType grad = DF_.Gradient(createPoint<PointType>(x(0), x(1), x(2)));
        return -sqr(grad.getVector3fMap().norm());
    }

    virtual void df(VectorType const& x, VectorType & res) {
        DFCalled++;
        PointType grad = DF_.SquaredGradientNormGradient(createPoint<PointType>(x(0), x(1), x(2)));
        grad.getVector3fMap() *= -1;
        res = grad.getVector3fMap().cast<double>();
    }

    virtual void fdf(VectorType const& x, Scalar & fres, VectorType & dfres) {
        fres = (*this)(x);
        df(x, dfres);
    }

public:
    int FCalled;
    int DFCalled;

private:
    DecisionFunction & DF_;
};

class FastSVM : public BaseSVM {
public:
    float getKernelWidth() const {
        return 1 / sqrt(get_params().gamma);
    }

    float get_rho() const {
        return decision_func->rho;
    }

    float get_alpha(int i) const {
        return decision_func->alpha[i];
    }

    void train(cv::Mat objects, cv::Mat responses, BaseSVMParams const& params) {
        BaseSVM::train(objects, responses, cv::Mat(), cv::Mat(), params);
        initFastPredict();

        int nSV = get_support_vector_count();
        std::cerr << nSV << " support vectors" << std::endl;
    }

    void train(PointCloud const& cloud, std::vector<float> labels, BaseSVMParams const& params) {
        cv::Mat objects(cloud.size(), 3, CV_32FC1);
        cv::Mat responses(cloud.size(), 1, CV_32FC1);
        for (int i = 0; i < objects.rows; ++i) {
            for (int j = 0; j < 3; ++j) {
                objects.at<float>(i, j) = cloud[i].getVector3fMap()(j);
            }
            responses.at<float>(i) = labels[i];
        }
        train(objects, responses, params);
    }

    float predict(PointType const& point, bool retDFVal = false) const {
        cv::Mat query(1, 3, CV_32FC1);
        query.at<cv::Vec3f>(0) = cv::Vec3f(point.x, point.y, point.z);
        return CvSVM::predict(query, retDFVal);
    }

    PointCloud::Ptr support_vector_point_cloud() {
        PointCloud::Ptr result(new PointCloud);
        for (int i = 0; i < get_support_vector_count(); ++i) {
            float const* sv = get_support_vector(i);
            result->push_back(createPoint<PointType>(sv[0], sv[1], sv[2]));
        }
        return result;
    }

    void buildDecisionFunctionEstimate(PointType const& point, DecisionFunction * df) const {
        float const kernelWidth = getKernelWidth();
        Indices_.clear();
        Distances_.clear();
        // hardcoded constant is todo
        SVTree->radiusSearch(point, 3 * kernelWidth, Indices_, Distances_);

        df->Reset(get_params().gamma, decision_func->rho);
        for (int i = 0; i < Indices_.size(); ++i) {
            df->AddSupportVector(svAsPoint(Indices_[i]), -get_alpha(Indices_[i]));
        }
    }

private:
    PointType svAsPoint(int svIndex) const {
        float const* sv = get_support_vector(svIndex);
        return createPoint<PointType>(sv[0], sv[1], sv[2]);
    }

    void initFastPredict() {
        float const kernelWidth = getKernelWidth();
        SVTree.reset(new pcl::octree::OctreePointCloudSearch<PointType>(2 * kernelWidth));
        SVCloud_ = support_vector_point_cloud();
        SVTree->setInputCloud(SVCloud_);
        SVTree->addPointsFromInputCloud();
    }

private:
    pcl::octree::OctreePointCloudSearch<PointType>::Ptr SVTree;
    PointCloud::Ptr SVCloud_;
    mutable std::vector<int> Indices_;
    mutable std::vector<float> Distances_;
};

class Printer {
public:
    Printer(DecisionFunction const& df)
        : DF_(df)
    {
    }

    void printStateAtPoint(PointType const& point, std::ostream & ostr) {
        ostr << "STATE AT " << point << std::endl;
        ostr << "value: " << DF_.Value(point) << std::endl;
        ostr << "gradient norm: " << DF_.Gradient(point).getVector3fMap().norm() << std::endl;
        ostr << "squred gradient norm gradient: " << DF_.SquaredGradientNormGradient(point) << std::endl;
    }

    static void printStateAtPoint(FastSVM const& model, PointType const& point, std::ostream & ostr) {
        DecisionFunction df;
        model.buildDecisionFunctionEstimate(point, &df);
        Printer pr(df);
        pr.printStateAtPoint(point, ostr);
    }

private:
    DecisionFunction const& DF_;
};

