#include "common.h"

class SupportVectorShape {
public:
    SupportVectorShape(PointCloud::Ptr featurePoints)
        : FeaturePoints(featurePoints)
    {
    }

    void LoadAsText(std::istream & istr);
    void SaveAsText(std::ostream & ostr);

private:
    PointCloud::Ptr FeaturePoints;
};
