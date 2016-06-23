#ifndef CUTTKERNEL_H
#define CUTTKERNEL_H
#include "cuttplan.h"

void cuttKernelSetSharedMemConfig();
int cuttKernelLaunchConfiguration(cuttPlan_t& plan, cudaDeviceProp& prop);
bool cuttKernel(cuttPlan_t& plan, void* dataIn, void* dataOut);

#endif // CUTTKERNEL_H
