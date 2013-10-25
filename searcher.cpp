#include "searcher.h"

#include "pcl/filters/random_sample.h"
#include "pcl/registration/bfgs.h"

FeaturePointSearcher::FeaturePointSearcher()
    : Seeds(new PointCloud)
    , FeaturePoints(new PointCloud)
{
}

void FeaturePointSearcher::Search(FastSVM const& model, PointCloud const& cloud) {
    *Seeds = cloud;
    pcl::RandomSample<PointType> rs;
    rs.setSample(NumSeeds);
    rs.setInputCloud(Seeds);
    rs.filter(*Seeds);

    FeaturePoints->clear();
    for (int i = 0; i < Seeds->size(); ++i) {
        FeaturePoints->push_back(SearchFromSeed(model, Seeds->at(i)));
    }
}

PointType FeaturePointSearcher::SearchFromSeed(FastSVM const& model, PointType const& seed) {
    DecisionFunction df;
    model.buildDecisionFunctionEstimate(seed, &df);

    GradientSquaredNormFunctor gsnf(df);
    GradientSquaredNormFunctor::VectorType current(seed.getVector3fMap().cast<double>());
    BFGS<GradientSquaredNormFunctor> bfgs(gsnf);
    bfgs.minimize(current);

    return pointFromVector<PointType>(current);
}
