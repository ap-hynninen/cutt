#ifndef CUTTKERNEL_H
#define CUTTKERNEL_H
#include "cuttplan.h"

void cuttKernelSetSharedMemConfig();

int cuttKernelLaunchConfiguration(int sizeofType, TensorSplit& ts, cudaDeviceProp& prop,
  LaunchConfig& lc);

void cuttKernelNumMemAccess(TensorSplit& ts, cudaDeviceProp& prop, LaunchConfig& lc,
  const int rank, const int* dim, const int* permutation, const size_t sizeofType,
  unsigned long long int& numRead, unsigned long long int& numWrite);

bool cuttKernel(cuttPlan_t& plan, void* dataIn, void* dataOut);

#endif // CUTTKERNEL_H
