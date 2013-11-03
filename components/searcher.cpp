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

PointType FeaturePointSearcher::SearchFromSeed(FastSVM const& model, PointType const& seed, float * gn2gn) {
    DecisionFunction df;
    model.buildDecisionFunctionEstimate(seed, &df);

    GradientSquaredNormFunctor gsnf(df);
    GradientSquaredNormFunctor::VectorType current(seed.getVector3fMap().cast<double>());
    BFGS<GradientSquaredNormFunctor> bfgs(gsnf);
    bfgs.minimize(current);
    if (gn2gn) {
        *gn2gn = df.squaredGradientNormGradient(
                pointFromVector<PointType>(current)).getVector3fMap().norm();
    }

    return pointFromVector<PointType>(current);
}

void FeaturePointSearcher::ChooseSeeds(PointCloud const& cloud, std::vector<float> const& gradientNorms) {
    IterateProspectiveSeeds(cloud, gradientNorms, [this] (PointType const& point) {
                Seeds->push_back(point);
                return Seeds->size() < NumSeeds;
            });
}

void FeaturePointSearcher::OneStageSearch(
        FastSVM const& model,
        PointCloud const& cloud,
        std::vector<float> const& gradientNorms)
{
    IterateProspectiveSeeds(cloud, gradientNorms, [this, &model] (PointType const& point) {
                float gn2gn;
                PointType fp = SearchFromSeed(model, point, &gn2gn);
                if (gn2gn > 1e-2) {
                    return true;
                }
                if (HasClosePoint(*FeaturePoints, fp, MinSpaceFP)) {
                    return true;
                }
                Seeds->push_back(point);
                FeaturePoints->push_back(fp);
                return Seeds->size() < NumSeeds;
            });
}

void FeaturePointSearcher::IterateProspectiveSeeds(
        PointCloud const& cloud,
        std::vector<float> const& gradientNorms,
        std::function<bool (PointType const& point)> handler)
{
    Seeds->clear();

    std::vector< std::pair<float, int> > pairs(cloud.size());
    for (int i = 0; i < pairs.size(); ++i) {
        pairs[i].first = gradientNorms[i];
        pairs[i].second = i;
    }
    std::sort(pairs.begin(), pairs.end(), std::greater< std::pair<float, int> >());

    for (int i = 0; i < pairs.size(); ++i) {
        int const idx = pairs[i].second;

        if (HasClosePoint(*Seeds, cloud[idx], MinSpaceSeeds)) {
            continue;
        }
        if (! handler(cloud[idx])) {
            break;
        }
    }
}

bool FeaturePointSearcher::HasClosePoint(PointCloud const& cloud, PointType const& point, float threshold) {
    for (int i = 0; i < cloud.size(); ++i) {
        if (sqrt(squaredEuclideanDistance(cloud[i], point)) < threshold) {
            return true;
        }
    }
    return false;
}
