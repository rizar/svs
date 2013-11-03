#include "common.h"

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
