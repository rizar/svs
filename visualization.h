#pragma once

#include "pcl/visualization/pcl_visualizer.h"

class TUMDataSetVisualizer : public pcl::visualization::PCLVisualizer {
public:
    TUMDataSetVisualizer(std::string const& cam = "")
        : pcl::visualization::PCLVisualizer(*CreateNumParams(), CreateParams(cam), "Visualization")
    {
    }

public:
    void EasyAdd(PointCloud::Ptr cloud, std::string const& name, int size = 1) {
        pcl::visualization::PointCloudColorHandlerRGBField<PointType> ch(cloud);
        addPointCloud<PointType>(cloud, ch, name.c_str());
        setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, size, name.c_str());
    }

    void EasyAdd(PointCloud::Ptr cloud, std::string const& name, int r, int g, int b, int size = 1) {
        pcl::visualization::PointCloudColorHandlerCustom<PointType> ch(cloud, r, g, b);
        addPointCloud<PointType>(cloud, ch, name.c_str());
        setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, size, name.c_str());
    }

    void Run(std::string const& screenshotPath) {
        if (screenshotPath.size()) {
            spinOnce();
            saveScreenshot(screenshotPath);
            return;
        }
        while (! wasStopped()) {
            spinOnce();
            pcl_sleep(0.01);
        }
    }

private:
    // Don't care about memory leaks!

    static int* CreateNumParams() {
        int * numParams = new int;
        *numParams = 3;
        return numParams;
    }

    static char** CreateParams(std::string const& cam) {
        char const* defaultCam = "0.00552702,5.52702/-0.00980252,-0.249744,1.9633/0,0,0/-0.476705,-0.87054,-0.122115/0.8575/840,525/66,52";

        char ** params = new char* [3];
        for (int i = 0; i < 3; ++i) {
            params[i] = new char [255];
        }
        strcpy(params[0], "aaa");
        strcpy(params[1], "-cam");
        strcpy(params[2], cam.size() ? cam.c_str() : defaultCam);
        return params;
    }
};
