#include "opencv2/ml/ml.hpp"

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/visualization/pcl_visualizer.h"
#include "pcl/io/pcd_io.h"
#include "pcl/io/ply_io.h"
#include "pcl/keypoints/susan.h"
#include "pcl/keypoints/iss_3d.h"
#include "pcl/search/kdtree.h"
#include "pcl/common/time.h"
#include "pcl/common/distances.h"
#include "pcl/features/normal_3d.h"
#include "pcl/filters/filter.h"

class MySVM : public CvSVM {
public:
    float get_rho() {
        return decision_func->rho;
    }

    float get_alpha(int i) {
        return decision_func->alpha[i];
    }
};

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::Normal> NormalCloud;

double computeCloudResolution(
        PointCloud::ConstPtr cloud,
        pcl::search::KdTree<PointCloud::PointType> const& tree)
{
    double res = 0.0;
    int n_points = 0;
    int nres;
    std::vector<int> indices (2);
    std::vector<float> sqr_distances (2);

    for (size_t i = 0; i < cloud->size (); ++i)
    {
        if (! pcl_isfinite ((*cloud)[i].x))
        {
            continue;
        }
        //Considering the second neighbor since the first is the point itself.
        nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
        if (nres == 2)
        {
            res += sqrt (sqr_distances[1]);
            ++n_points;
        }
    }
    if (n_points != 0)
    {
        res /= n_points;
    }
    return res;
}

cv::Mat createMatFromPointCloud(PointCloud::Ptr pc) {
    cv::Mat result(pc->size(), 3, CV_32FC1);
    for (int i = 0; i < pc->size(); ++i) {
        result.at<float>(i, 0) = pc->at(i).x;
        result.at<float>(i, 1) = pc->at(i).y;
        result.at<float>(i, 2) = pc->at(i).z;
    }
    return result;
}

void generateCube(PointCloud::Ptr pc) {
    for (float i = -1; i <= 1; i += 0.25) {
        for (float j = -1; j <= 1; j += 0.25) {
            for (float k = -1; k <= 1; k += 0.25) {
                pc->push_back(PointCloud::PointType(i, j, k));
            }
        }
    }
}

void generateSineGrid(PointCloud::Ptr pc) {
    for (float i = -5; i <= 5; i += 0.1) {
        for (float j = -5; j <= 5; j += 0.1) {
            pc->push_back(PointCloud::PointType(i, j, sin(1 * (i + j))));
        }
    }
}

void printKernelValueHistogram(PointCloud::Ptr pc, float kernelWidth) {
    int const MAX_LOG = 20;
    float const LOG2 = log(2);

    std::vector<double> logFreq(MAX_LOG);
    for (int i = 0; i < pc->size() / 100; ++i) {
        for (int j = i + 1; j < pc->size() / 100; ++j) {
            float const dist = pcl::squaredEuclideanDistance(pc->at(i), pc->at(j));
            float const kernel = exp(-dist / kernelWidth / kernelWidth);
            float log2kernel = -log(kernel) / LOG2;
            if (isnan(log2kernel) || log2kernel > MAX_LOG) {
                log2kernel = MAX_LOG;
            }
            logFreq[static_cast<int>(log2kernel)] += 1.0;
        }
    }

    float total = 0.0;
    for (int i = 0; i < logFreq.size(); ++i) {
        total += logFreq[i];
    }
    for (int i = 0; i < logFreq.size(); ++i) {
        logFreq[i] /= total;
    }

    for (int i = 0; i < logFreq.size(); ++i) {
        std::cout.precision(9);
        std::cout << -i << "\t" << logFreq[i] << std::endl;
    }
}

PointCloud::Ptr randomSlice(PointCloud::Ptr cloud, int k) {
    PointCloud::Ptr result(new PointCloud);
    for (int i = 0; i < cloud->size(); ++i) {
        if (rand() % k == 0) {
            result->push_back(cloud->at(i));
        }
    }
    return result;
}

int main(int argc, char * argv []) {
    PointCloud::Ptr pc(new PointCloud);
    PointCloud::Ptr keypoints(new PointCloud);
    PointCloud::Ptr support(new PointCloud);
    NormalCloud::Ptr normals(new NormalCloud);

    pcl::search::KdTree<PointCloud::PointType>::Ptr tree(
            new pcl::search::KdTree<PointCloud::PointType>);

    double resolution;
    double kernelWidth;

    pcl::visualization::PCLVisualizer viewer("Visualization");

    if (argc > 1) {
        pcl::ScopeTime st("Load cloud");
        pcl::io::loadPCDFile(argv[1], *pc);
        std::vector<int> tmp;
        pcl::removeNaNFromPointCloud(*pc, *pc, tmp);
    }
    else {
        // generateCube(pc);
        generateSineGrid(pc);
    }

    {
        pcl::ScopeTime st("KD tree creation");
        tree->setInputCloud(pc);
    }

    {
        pcl::ScopeTime st("Resolution computation");
        resolution = computeCloudResolution(pc, *tree);
        std::cout << "resolution: " << resolution << std::endl;
    }

    PointCloud::Ptr plus(new PointCloud);
    PointCloud::Ptr minus(new PointCloud);
    int const depth = 5;

    for (int i = 0; i < pc->width; ++i) {
        PointCloud::PointType ort = pc->at(i);
        float norm = ort.getVector3fMap().norm();
        ort.getVector3fMap() /= norm;

        for (int j = 0; j <= depth; ++j) {
            PointCloud::PointType point = pc->at(i);
            point.getVector3fMap() -= ort.getVector3fMap() * resolution * j;
            plus->push_back(point);
        }
        for (int j = 1; j <= depth; ++j) {
            PointCloud::PointType point = pc->at(i);
            point.getVector3fMap() += ort.getVector3fMap() * resolution * j;
            minus->push_back(point);
        }
    }

    {
        kernelWidth = 5 * resolution;

        pcl::ScopeTime st("SVM building");
        CvSVMParams params;
        CvTermCriteria termCriteria;
        termCriteria.type = /*CV_TERMCRIT_ITER | */CV_TERMCRIT_EPS;
        termCriteria.epsilon = 1e-3;
        termCriteria.max_iter = 10000;
        params.svm_type = CvSVM::C_SVC;
        params.nu = 0.01;
        params.C = 1024;
        params.gamma = 1 / kernelWidth / kernelWidth;
        params.term_crit = termCriteria;

        PointCloud::Ptr plusSlice = randomSlice(plus, 10);
        PointCloud::Ptr minusSlice = randomSlice(minus, 10);

        cv::Mat dataMat = createMatFromPointCloud(plusSlice);
        dataMat.push_back(createMatFromPointCloud(minusSlice));
        cv::Mat responseMat(dataMat.rows, 1, CV_32FC1);
        for (int i = 0; i < dataMat.rows; ++i) {
            responseMat.at<float>(i, 0) = i < plusSlice->size() ? 1 : -1;
        }

        MySVM svm;
        std::cout << dataMat.rows << " cases for SVM" << std::endl;
        svm.train(dataMat, responseMat, cv::Mat(), cv::Mat(), params);
        std::cout << svm.get_support_vector_count() << " support vectors" << std::endl;
        for (int i = 0; i < svm.get_support_vector_count(); ++i) {
            float const* sv = svm.get_support_vector(i);
            support->push_back(PointCloud::PointType(sv[0], sv[1], sv[2]));
        }

        float ratio [2] = {0.0, 0.0};
        float total = plus->size() + minus->size();
        cv::Mat query(1, 3, CV_32FC1);
        for (int i = 0; i < plus->size(); ++i) {
            query.at<cv::Vec3f>(0) = cv::Vec3f(plus->at(i).x, plus->at(i).y, plus->at(i).z);
            ratio[svm.predict(query, false) == 1.0] += 1.0;
        }
        for (int i = 0; i < minus->size(); ++i) {
            query.at<cv::Vec3f>(0) = cv::Vec3f(minus->at(i).x, minus->at(i).y, minus->at(i).z);
            ratio[svm.predict(query, false) == -1.0] += 1.0;
        }
        for (int i = 0; i < 2; ++i) {
            std::cout << "class " << i << ": " << ratio[i] / total << std::endl;
        }
    }

    {
        pcl::ScopeTime("Visualization");

        pcl::visualization::PointCloudColorHandlerCustom<PointCloud::PointType>
            plusCH(plus, 255, 0, 0);
        viewer.addPointCloud<PointCloud::PointType>(plus, plusCH, "plus");

        pcl::visualization::PointCloudColorHandlerCustom<PointCloud::PointType>
            minusCH(minus, 0, 0, 255);
        viewer.addPointCloud<PointCloud::PointType>(minus, minusCH, "minus");

        /*pcl::visualization::PointCloudColorHandlerCustom<PointCloud::PointType>
            cloudCH(pc, 200, 200, 200);
        viewer.addPointCloud<PointCloud::PointType>(pc, cloudCH, "cloud");*/

        pcl::visualization::PointCloudColorHandlerCustom<PointCloud::PointType>
            supportCH(support, 0, 255, 0);
        viewer.addPointCloud<PointCloud::PointType>(support, supportCH, "support");
        viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "support");
    }

    while (! viewer.wasStopped()) {
        viewer.spinOnce();
        pcl_sleep(0.01);
    }

    return 0;
}
