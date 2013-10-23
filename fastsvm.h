#include "common.h"
#include "mysvm.h"

#include "pcl/common/distances.h"
#include "pcl/search/octree.h"

class DecisionFunction {
public:
    DecisionFunction()
    {
    }

    DecisionFunction(float gamma,
            std::vector<PointType> const& sv,
            std::vector<float> const& alpha)
        : Gamma_(gamma)
        , SV_(sv)
        , Alpha_(alpha)
    {
        assert(SV_.size() == Alpha_.size());
    }

    void reset(float gamma) {
        Gamma_ = gamma;
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
};

class FastSVM : public My::CvSVM {
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

    void initFastPredict() {
        float const kernelWidth = getKernelWidth();
        SVTree.reset(new pcl::octree::OctreePointCloudSearch<PointType>(2 * kernelWidth));
        SVCloud_ = support_vector_point_cloud();
        SVTree->setInputCloud(SVCloud_);
        SVTree->addPointsFromInputCloud();
    }

    float fastPredict(PointType const& point) const {
        DecisionFunction df;
        buildDecisionFunctionEstimate(point, &df);
        return df.decisionFunction(point) > 0 ? 1 : -1;
    }

    void buildDecisionFunctionEstimate(PointType const& point, DecisionFunction * df) const {
        float const kernelWidth = getKernelWidth();
        Indices_.clear();
        Distances_.clear();
        SVTree->radiusSearch(point, 5 * kernelWidth, Indices_, Distances_);

        df->reset(get_params().gamma);
        for (int i = 0; i < Indices_.size(); ++i) {
            df->addSupportVector(svAsPoint(Indices_[i]), -get_alpha(Indices_[i]));
        }
    }

private:
    PointType svAsPoint(int svIndex) const {
        float const* sv = get_support_vector(svIndex);
        return createPoint<PointType>(sv[0], sv[1], sv[2]);
    }

private:
    pcl::octree::OctreePointCloudSearch<PointType>::Ptr SVTree;
    PointCloud::Ptr SVCloud_;
    mutable std::vector<int> Indices_;
    mutable std::vector<float> Distances_;
};


