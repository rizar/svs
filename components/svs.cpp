#include "svs.h"

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
