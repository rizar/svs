#pragma once

#include "common.h"

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

class DecisionFunction {
public:
    DecisionFunction()
    {
    }

    DecisionFunction(
            float gamma,
            std::vector<PointType> const& sv,
            std::vector<float> const& alpha,
            float rho)
        : SV_(sv)
        , Alpha_(alpha)
        , Gamma_(gamma)
        , Rho_(rho)
    {
        assert(SV_.size() == Alpha_.size());
    }

    void reset(float gamma, float rho) {
        Gamma_ = gamma;
        Rho_ = rho;
        SV_.clear();
        Alpha_.clear();
    }

    void addSupportVector(PointType const& newSV, float alpha) {
        SV_.push_back(newSV);
        Alpha_.push_back(alpha);
    }

    float kernelValueWithAlpha(int svIndex, PointType const& point) const {
        float const dist2 = pcl::squaredEuclideanDistance(SV_[svIndex], point);
        return Alpha_[svIndex] * exp(-Gamma_ * dist2);
    }

    float decisionFunction(PointType const& point) const {
        float dfVal = 0.0;
        for (int i = 0; i < SV_.size(); ++i) {
            dfVal += kernelValueWithAlpha(i, point);
        }
        return dfVal;
    }

    PointType gradient(PointType const& point) const {
        PointType result;
        for (int i = 0; i < SV_.size(); ++i) {
            PointType add = point;
            add.getVector3fMap() -= SV_[i].getVector3fMap();
            add.getVector3fMap() *= kernelValueWithAlpha(i, point);
            // add = \alpha_i K(x, x_i) (x - x_i)
            result.getVector3fMap() += add.getVector3fMap();
        }

        // result = -\sum\limits_{i=1}^n alpha_i K(x, x_i) (x - x_i)
        result.getVector3fMap() *= -1;
        return result;
    }

    void hessian(PointType const& point, Eigen::MatrixXf * result) const {
        *result = Eigen::Matrix3f::Zero(3,  3);

        for (int svi = 0; svi < SV_.size(); ++svi) {
            float const coof = kernelValueWithAlpha(svi, point);

            PointType diff = point;
            auto diffMap = diff.getVector3fMap();
            diffMap -= SV_[svi].getVector3fMap();

            for (int x = 0; x < 3; ++x) {
                for (int y = 0; y < 3; ++y) {
                    (*result)(x, y) += coof * (2 * Gamma_ * diffMap(x) * diffMap(y) - (x == y ? 1 : 0));
                }
            }
        }
    }

    PointType squaredGradientNormGradient(PointType const& point) const {
        Eigen::MatrixXf hess;
        hessian(point, &hess);
        Eigen::Vector3f result = 2 * hess * gradient(point).getVector3fMap();

        return createPoint<PointType>(result(0), result(1), result(2));
    }


public:
    std::vector<PointType> SV_;
    std::vector<float> Alpha_;
    float Gamma_;
    float Rho_;
};

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
        PointType grad = DF_.gradient(createPoint<PointType>(x(0), x(1), x(2)));
        return -sqr(grad.getVector3fMap().norm());
    }

    virtual void df(VectorType const& x, VectorType & res) {
        DFCalled++;
        PointType grad = DF_.squaredGradientNormGradient(createPoint<PointType>(x(0), x(1), x(2)));
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

        int maxAlphas = 0;
        for (int i = 0; i < nSV; ++i) {
            if (get_alpha(i) > params.C - 1e-3) {
                maxAlphas++;
            }
        }
        std::cerr << maxAlphas << " support vectors classified wrongly" << std::endl;
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

        df->reset(get_params().gamma, decision_func->rho);
        for (int i = 0; i < Indices_.size(); ++i) {
            df->addSupportVector(svAsPoint(Indices_[i]), -get_alpha(Indices_[i]));
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
        ostr << "value: " << DF_.decisionFunction(point) << std::endl;
        ostr << "gradient norm: " << DF_.gradient(point).getVector3fMap().norm() << std::endl;
        ostr << "squred gradient norm gradient: " << DF_.squaredGradientNormGradient(point) << std::endl;
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

