#pragma once

#include <cstdlib>

struct SVSParams {
    int Seed = 1;

    float MaxAlpha = 32;
    float KernelWidth = 5;
    float KernelThreshold = 1e-3;
    float TerminateEps = 1e-2;

    float SmoothingRange = 5;
    int BorderWidth = 1;
    float StepWidth = 1;
    float TakeProb = 1.0;

    int NumFP = 100;

    size_t CacheSize = 1 << 30;

    bool UseGrid = true;
    bool UseNormals = true;

    void Save(const char* path);
    void Load(const char* path);
};

