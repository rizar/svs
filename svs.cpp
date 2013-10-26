#include "svs.h"

void SupportVectorShape::LoadAsText(std::istream & istr) {
    float x, y, z;
    while (istr >> x >> y >> z) {
        FeaturePoints->push_back(createPoint<PointType>(x, y, z));
    }
}

void SupportVectorShape::SaveAsText(std::ostream & ostr) {
    for (int i = 0; i < FeaturePoints->size(); ++i) {
        PointType const& p = FeaturePoints->at(i);
        ostr << p.x << ' ' << p.y << ' ' << p.z << '\n';
    }
    ostr.flush();
}
