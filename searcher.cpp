#include "searcher.h"

#include "pcl/filters/random_sample.h"
#include "pcl/registration/bfgs.h"
#include "pcl/common/distances.h"

FeaturePointSearcher::FeaturePointSearcher()
    : Seeds(new PointCloud)
    , FeaturePoints(new PointCloud)
{
}

void FeaturePointSearcher::Search(FastSVM const& model) {
    assert(Seeds->size());

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

void FeaturePointSearcher::ChooseSeeds(PointCloud const& cloud, std::vector<float> const& gradientNorms) {
    std::vector< std::pair<float, int> > pairs(cloud.size());
    for (int i = 0; i < pairs.size(); ++i) {
        pairs[i].first = gradientNorms[i];
        pairs[i].second = i;
    }
    std::sort(pairs.begin(), pairs.end(), std::greater< std::pair<float, int> >());

    Seeds->clear();
    for (int i = 0; Seeds->size() < NumSeeds && i < pairs.size(); ++i) {
        int const idx = pairs[i].second;

        bool alreadyClose = false;
        for (int j = 0; j < Seeds->size(); ++j) {
            if (sqrt(squaredEuclideanDistance(cloud[idx], Seeds->at(j))) < MinSpace) {
                alreadyClose = true;
            }
        }

        if (! alreadyClose) {
            Seeds->push_back(cloud[idx]);
        }
    }
}
