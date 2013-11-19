#pragma once

#include "common.h"

#include "pcl/visualization/pcl_visualizer.h"

#include <functional>

struct Color {
    int r = 0;
    int g = 0;
    int b = 0;

    Color() {
    }

    Color(PointType const& point)
        : r(point.r)
        , g(point.g)
        , b(point.b)
    {
    }

    Color(std::initializer_list<int> const& list) {
        auto it = list.begin();
        r = *(it++);
        g = *(it++);
        b = *it;
    }
};

typedef std::function<Color (int)> ColorMap;

class PointCloudLambdaColorHandler : public pcl::visualization::PCLVisualizer::ColorHandler {
    typedef pcl::visualization::PCLVisualizer::ColorHandler Base;

public:
    PointCloudLambdaColorHandler(::PointCloud::ConstPtr cloud, ColorMap colorMap)
        : Base(pcl::PCLPointCloud2::ConstPtr())
        , Cloud_(cloud)
        , ColorMap_(colorMap)
    {
        capable_ = true;
    }

    virtual std::string
    getName () const { return ("PointCloudLambdaColorHandler"); }

    virtual std::string
    getFieldName () const { return ("[lambda]"); }

    virtual bool getColor (vtkSmartPointer<vtkDataArray> &scalars) const
    {
        if (!scalars)
            scalars = vtkSmartPointer<vtkUnsignedCharArray>::New ();
        scalars->SetNumberOfComponents (3);

        vtkIdType nr_points = Cloud_->points.size();
        reinterpret_cast<vtkUnsignedCharArray*>(&(*scalars))->SetNumberOfTuples (nr_points);
        unsigned char* colors = reinterpret_cast<vtkUnsignedCharArray*>(&(*scalars))->GetPointer (0);

        int j = 0;
        for (vtkIdType cp = 0; cp < nr_points; ++cp)
        {
            if (pcl_isnan(Cloud_->at(cp).x)) {
                continue;
            }
            Color col = ColorMap_(cp);

            colors[j * 3 + 0] = static_cast<unsigned char> (col.r);
            colors[j * 3 + 1] = static_cast<unsigned char> (col.g);
            colors[j * 3 + 2] = static_cast<unsigned char> (col.b);
            j++;
        }
        return (true);
    }

private:
    ::PointCloud::ConstPtr Cloud_;
    ColorMap ColorMap_;
};

class TUMDataSetVisualizer : public pcl::visualization::PCLVisualizer {
public:
    TUMDataSetVisualizer(std::string const& cam = "")
        : pcl::visualization::PCLVisualizer(*CreateNumParams(), CreateParams(cam), "Visualization")
    {
    }

public:
    void EasyAdd(PointCloud::Ptr cloud, std::string const& name, ColorMap const& colorMap) {
        addPointCloud<PointType>(cloud, ColorHandlerConstPtr(new PointCloudLambdaColorHandler(cloud, colorMap)), name);
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
