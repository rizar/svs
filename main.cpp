#include "mysvm.h"
using namespace My;

//#include "opencv2/ml/ml.hpp"

#include "boost/program_options.hpp"

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/visualization/pcl_visualizer.h"
#include "pcl/io/pcd_io.h"
#include "pcl/io/ply_io.h"
#include "pcl/keypoints/susan.h"
#include "pcl/keypoints/iss_3d.h"
#include "pcl/search/kdtree.h"
#include "pcl/search/octree.h"
#include "pcl/common/time.h"
#include "pcl/common/distances.h"
#include "pcl/features/normal_3d.h"
#include "pcl/filters/filter.h"

namespace po = boost::program_options;

typedef pcl::PointXYZRGB PointType;
typedef pcl::PointCloud<PointType> PointCloud;
typedef pcl::PointCloud<pcl::Normal> NormalCloud;

inline float sqr(float x) {
    return x * x;
}

template <class P>
P createPoint(float x, float y, float z) {
    P p;
    p.x = x;
    p.y = y;
    p.z = z;
    return p;
}

bool printHistogram = false;
bool skipVisualization = false;
bool skipCheck = false;

class MySVM : public CvSVM {
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
        OctTree_.reset(new pcl::octree::OctreePointCloudSearch<PointType>(2 * kernelWidth));
        SVCloud_ = support_vector_point_cloud();
        OctTree_->setInputCloud(SVCloud_);
        OctTree_->addPointsFromInputCloud();
    }

    float fastPredict(PointType const& point, bool retDFVal = false) const {
        assert(OctTree_.get());
        initNeighbors(point);

        float dfVal = 0.0;
        for (int i = 0; i < Indices_.size(); ++i) {
            dfVal += kernelValue(Indices_[i], point);
        }

        if (retDFVal) {
            return dfVal;
        } else {
            return dfVal > 0 ? 1 : -1;
        }
    }

private:
    float kernelValue(int svIndex, PointType const& point) const {
        float const* sv = get_support_vector(svIndex);
        float const gamma = get_params().gamma;
        float const dist2 = sqr(point.x - sv[0]) + sqr(point.y - sv[1]) + sqr(point.z - sv[2]);
        return -get_alpha(svIndex) * exp(-gamma * dist2);
    }

    void initNeighbors(PointType const& point) const {
        float const kernelWidth = getKernelWidth();
        Indices_.clear();
        Distances_.clear();
        OctTree_->radiusSearch(point, 5 * kernelWidth, Indices_, Distances_);
    }

private:
    pcl::octree::OctreePointCloudSearch<PointType>::Ptr OctTree_;
    PointCloud::Ptr SVCloud_;
    mutable std::vector<int> Indices_;
    mutable std::vector<float> Distances_;
};

double computeCloudResolution(
        PointCloud::ConstPtr cloud,
        pcl::search::KdTree<PointType> const& tree)
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

cv::Mat createMatFromPointCloud(PointCloud::Ptr shape) {
    cv::Mat result(shape->size(), 3, CV_32FC1);
    for (int i = 0; i < shape->size(); ++i) {
        result.at<float>(i, 0) = shape->at(i).x;
        result.at<float>(i, 1) = shape->at(i).y;
        result.at<float>(i, 2) = shape->at(i).z;
    }
    return result;
}

void generateCube(PointCloud::Ptr shape) {
    for (float i = -1; i <= 1; i += 0.25) {
        for (float j = -1; j <= 1; j += 0.25) {
            for (float k = -1; k <= 1; k += 0.25) {
                shape->push_back(createPoint<PointType>(i, j, k));
            }
        }
    }
}

void generateSineGrid(PointCloud::Ptr shape) {
    for (float i = -5; i <= 5; i += 0.1) {
        for (float j = -5; j <= 5; j += 0.1) {
            shape->push_back(createPoint<PointType>(i, j, sin(1 * (i + j))));
        }
    }
}

void printKernelValueHistogram(PointCloud::Ptr shape, float kernelWidth) {
    int const MAX_LOG = 20;
    float const LOG2 = log(2);

    std::vector<double> logFreq(MAX_LOG + 1);
    std::vector<double> cumFreq(MAX_LOG + 1);
    for (int i = 0; i < shape->size(); ++i) {
        for (int j = i + 1; j < shape->size(); ++j) {
            float const dist = pcl::squaredEuclideanDistance(shape->at(i), shape->at(j));
            float const kernel = exp(-dist / kernelWidth / kernelWidth);
            float log2kernel = -log(kernel) / LOG2;
            if (isnan(log2kernel) || log2kernel > MAX_LOG) {
                log2kernel = MAX_LOG;
            }
            logFreq[static_cast<int>(log2kernel)] += 1.0;
        }
    }

    float total = shape->size() * (shape->size() - 1) / 2;
    for (int i = 0; i < logFreq.size(); ++i) {
        logFreq[i] /= total;
    }
    logFreq[0] = logFreq[0];
    for (int i = 0; i < cumFreq.size(); ++i) {
        cumFreq[i] = cumFreq[i - 1] + logFreq[i];
    }

    for (int i = 0; i < logFreq.size(); ++i) {
        std::cout.precision(5);
        std::cout << -i << "\t" << logFreq[i] << "\t" << cumFreq[i] << std::endl;
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

class RangeImagePoint {
public:
    RangeImagePoint(PointType const& point)
        : Point_(point)
    {
    }

    PointType ort() {
        PointType res = Point_;
        res.getVector3fMap() /= res.getVector3fMap().norm();
        return res;
    }

    PointType shift(double resolution, int k = 1) {
        PointType res = Point_;
        res.getVector3fMap() += ort().getVector3fMap() * resolution * k;
        return res;
    }

    bool isLearn(MySVM const& svm, double resolution) {
        for (int j = -5; j <= -3; ++j) {
            if (svm.fastPredict(shift(resolution, j)) == -1) {
                return false;
            }
        }
        for (int j = 4; j <= 5; ++j) {
            if (svm.fastPredict(shift(resolution, j)) == 1) {
                return false;
            }
        }
        return true;
    }

private:
     PointType Point_;
};

void doSVM(MySVM & svm,
           float C, float k, float eps, float resolution, int sample,
           PointCloud::Ptr shape, PointCloud::Ptr plus, PointCloud::Ptr minus)
{
    float kernelWidth = k * resolution;

    CvSVMParams params;
    CvTermCriteria termCriteria;
    termCriteria.type = /*CV_TERMCRIT_ITER | */CV_TERMCRIT_EPS;
    termCriteria.epsilon = eps; //1e-3;
    termCriteria.max_iter = 10000;
    params.svm_type = MySVM::C_SVC;
    params.nu = 0.01;
    params.C = C;
    params.gamma = 1 / kernelWidth / kernelWidth;
    params.term_crit = termCriteria;

    PointCloud::Ptr plusSlice = randomSlice(plus, sample);
    PointCloud::Ptr minusSlice = randomSlice(minus, sample);

    cv::Mat dataMat = createMatFromPointCloud(plusSlice);
    dataMat.push_back(createMatFromPointCloud(minusSlice));
    cv::Mat responseMat(dataMat.rows, 1, CV_32FC1);
    for (int i = 0; i < dataMat.rows; ++i) {
        responseMat.at<float>(i, 0) = i < plusSlice->size() ? 1 : -1;
    }

    float trainingTime;
    {
        pcl::ScopeTime st("SVM");
        svm.train(dataMat, responseMat, cv::Mat(), cv::Mat(), params);
        trainingTime = st.getTimeSeconds();
    }

    svm.initFastPredict();

    if (! skipCheck) {
        float ratio [2] = {0.0, 0.0};
        {
            pcl::ScopeTime st("Accuracy");
            float total = plus->size() + minus->size();
            for (int i = 0; i < plus->size(); ++i) {
                ratio[svm.fastPredict(plus->at(i), false) == 1.0] += 1.0;
            }
            for (int i = 0; i < minus->size(); ++i) {
                ratio[svm.fastPredict(minus->at(i), false) == -1.0] += 1.0;
            }
            for (int i = 0; i < 2; ++i) {
                ratio[i] /= total;
            }
        }

        float learnFailed = 0.0;
        {
            pcl::ScopeTime st("Shape learnt");
            for (int i = 0; i < shape->size(); ++i) {
                if (! RangeImagePoint(shape->at(i)).isLearn(svm, resolution)) {
                    learnFailed += 1.0;
                }
            }
            learnFailed /= shape->size();
        }

        std::cout << C << '\t' << k
            << '\t' << dataMat.rows << '\t' << svm.get_support_vector_count() / static_cast<float>(dataMat.rows)
            << '\t' << ratio[0] << '\t' << ratio[1] << "\t"
            << learnFailed << "\t"
            << trainingTime << std::endl;
    }
}

int main(int argc, char * argv []) {
    float givenC = -1;
    float givenK = -1;
    float eps = 1e-3;
    int depth = 5;
    int sample = 2 * depth + 2;
    std::string path;

    po::options_description desc;
    desc.add_options()
        ("C", po::value<float>(&givenC))
        ("k", po::value<float>(&givenK))
        ("eps", po::value<float>(&eps))
        ("d", po::value<int>(&depth))
        ("s", po::value<int>(&sample))
        ("hist", po::value<bool>(&printHistogram))
        ("novis", po::value<bool>(&skipVisualization)->zero_tokens())
        ("nocheck", po::value<bool>(&skipCheck)->zero_tokens())
        ("path", po::value<std::string>(&path)->required());

    po::positional_options_description pos;
    pos.add("path", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
              .options(desc).positional(pos).run(), vm);
    vm.notify();

    PointCloud::Ptr shape(new PointCloud);
    PointCloud::Ptr keypoints(new PointCloud);
    PointCloud::Ptr support;
    PointCloud::Ptr errors;
    NormalCloud::Ptr normals(new NormalCloud);

    pcl::search::KdTree<PointType>::Ptr tree(
            new pcl::search::KdTree<PointType>);

    double resolution;
    double kernelWidth;

    if (argc > 1) {
        pcl::ScopeTime st("Load cloud");
        pcl::io::loadPCDFile(path, *shape);
        std::vector<int> tmp;
        pcl::removeNaNFromPointCloud(*shape, *shape, tmp);
    }
    else {
        // generateCube(shape);
        generateSineGrid(shape);
    }

    {
        pcl::ScopeTime st("KD tree creation");
        tree->setInputCloud(shape);
    }

    {
        pcl::ScopeTime st("Resolution computation");
        resolution = computeCloudResolution(shape, *tree);
        std::cerr << "resolution: " << resolution << std::endl;
    }

    PointCloud::Ptr plus(new PointCloud);
    PointCloud::Ptr minus(new PointCloud);

    for (int i = 0; i < shape->width; ++i) {
        RangeImagePoint rip(shape->at(i));
        for (int j = -depth; j <= 0; ++j) {
            plus->push_back(rip.shift(resolution, j));
        }
        for (int j = 1; j <= depth + 1; ++j) {
            minus->push_back(rip.shift(resolution, j));
        }
    }

    {
        std::cout.setf(std::ios::fixed);
        std::cout.precision(3);
        std::cout << "C" << "\t" << "k" << "\t"
                  << "N" << "\t" << "SV" << "\t"
                  << "ERR" << "\t" << "OK" << "\t"
                  << "LF" << "\t"
                  << "TIME" << std::endl;

        if (givenC < 0 || givenK < 0) {
            for (float k = 2; k <= 10; k += 2) {
                for (float C = pow(2, 0); C <= pow(2, 12); C *= 2) {
                    MySVM svm;
                    doSVM(svm, C, k, eps, resolution, sample,
                            shape, plus, minus);
                }
            }
            return 0;
        } else {
            MySVM svm;
            doSVM(svm, givenC, givenK, eps, resolution, sample,
                    shape, plus, minus);

            support = svm.support_vector_point_cloud();

            errors.reset(new PointCloud);
            for (int i = 0; i < shape->size(); ++i) {
                if (! RangeImagePoint(shape->at(i)).isLearn(svm, resolution)) {
                    errors->push_back(shape->at(i));
                }
            }

            if (printHistogram) {
                printKernelValueHistogram(shape, givenK * resolution);
            }
        }
    }

    if (skipVisualization) {
        return 0;
    }

    int viewerArgc = 3;
    char * viewerArgv [3];
    for (int i = 0; i < 3; ++i) {
        viewerArgv[i] = new char [255];
    }
    strcpy(viewerArgv[0], "aaa");
    strcpy(viewerArgv[1], "-cam");
    strcpy(viewerArgv[2], "0.00552702,5.52702/-0.00980252,-0.249744,1.9633/0,0,0/-0.476705,-0.87054,-0.122115/0.8575/840,525/66,52");
    pcl::visualization::PCLVisualizer viewer(viewerArgc, viewerArgv, "Visualization");

    {
        pcl::ScopeTime("Visualization");

        /*pcl::visualization::PointCloudColorHandlerCustom<PointType>
            plusCH(plus, 100, 100, 100);
        viewer.addPointCloud<PointType>(plus, plusCH, "plus");

        pcl::visualization::PointCloudColorHandlerCustom<PointType>
            minusCH(minus, 100, 100, 100);
        viewer.addPointCloud<PointType>(minus, minusCH, "minus");*/

        pcl::visualization::PointCloudColorHandlerCustom<PointType>
            errorsCH(errors, 0, 0, 255);
        viewer.addPointCloud<PointType>(errors, errorsCH, "errors");

        pcl::visualization::PointCloudColorHandlerCustom<PointType>
            supportCH(support, 0, 255, 0);
        viewer.addPointCloud<PointType>(support, supportCH, "support");

        pcl::visualization::PointCloudColorHandlerRGBField<PointType>
            shapeCH(shape);
        viewer.addPointCloud<PointType>(shape, shapeCH, "shape");

        viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "support");
        viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "errors");
    }

    while (! viewer.wasStopped()) {
        viewer.spinOnce();
        pcl_sleep(0.01);
    }

    return 0;
}
