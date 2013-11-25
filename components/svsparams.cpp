#include "svsparams.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

using namespace boost::property_tree;

#define PUT(pt, f) pt.put(#f, f)
#define GET(pt, t, f) f = pt.get<t>(#f)

void SVSParams::Load(const char* path) {
    ptree pt;
    read_xml(path, pt);
    GET(pt, int, Seed);
    GET(pt, float, MaxAlpha);
    GET(pt, float, KernelWidth);
    GET(pt, float, KernelThreshold);
    GET(pt, float, TerminateEps);
    GET(pt, float, SmoothingRange);
    GET(pt, int, BorderWidth);
    GET(pt, float,  StepWidth);
    GET(pt, float, TakeProb);
    GET(pt, int, NumFP);
    GET(pt, size_t, CacheSize);
    GET(pt, bool, UseGrid);
    GET(pt, bool, UseNormals);
}

void SVSParams::Save(const char* path) {
    ptree pt;
    PUT(pt, Seed);
    PUT(pt, MaxAlpha);
    PUT(pt, KernelWidth);
    PUT(pt, KernelThreshold);
    PUT(pt, TerminateEps);
    PUT(pt, SmoothingRange);
    PUT(pt, BorderWidth);
    PUT(pt, StepWidth);
    PUT(pt, TakeProb);
    PUT(pt, NumFP);
    PUT(pt, CacheSize);
    PUT(pt, UseGrid);
    PUT(pt, UseNormals);
    xml_writer_settings<char> settings(' ', 4);
    write_xml(path, pt, std::locale(), settings);
}

