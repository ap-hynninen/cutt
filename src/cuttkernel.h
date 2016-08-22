#ifndef CUTTKERNEL_H
#define CUTTKERNEL_H
#include "cuttplan.h"

void cuttKernelSetSharedMemConfig();

int cuttKernelLaunchConfiguration(int sizeofType, TensorSplit& ts, cudaDeviceProp& prop,
  LaunchConfig& lc);

bool cuttKernel(cuttPlan_t& plan, void* dataIn, void* dataOut);

#endif // CUTTKERNEL_H
