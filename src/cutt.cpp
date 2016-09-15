/******************************************************************************
MIT License

Copyright (c) 2016 Antti-Pekka Hynninen
Copyright (c) 2016 Oak Ridge National Laboratory (UT-Batelle)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************/
#include <cuda.h>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include "CudaUtils.h"
#include "cuttplan.h"
#include "cuttkernel.h"
#include "cuttTimer.h"
#include "cutt.h"

// Hash table to store the plans
static std::unordered_map< cuttHandle, cuttPlan_t* > planStorage;

// Current handle
static cuttHandle curHandle = 0;

// Table of devices that have been initialized
static std::unordered_set<int> devicesReady;

cuttResult cuttPlanCheckInput(int rank, int* dim, int* permutation, size_t sizeofType) {
  // Check sizeofType
  if (sizeofType != 4 && sizeofType != 8) return CUTT_INVALID_PARAMETER;
  // Check rank
  if (rank <= 1) return CUTT_INVALID_PARAMETER;
  // Check dim[]
  for (int i=0;i < rank;i++) {
    if (dim[i] <= 1) return CUTT_INVALID_PARAMETER;
  }
  // Check permutation
  bool permutation_fail = false;
  int* check = new int[rank];
  for (int i=0;i < rank;i++) check[i] = 0;
  for (int i=0;i < rank;i++) {
    if (permutation[i] < 0 || permutation[i] >= rank || check[permutation[i]]++) {
      permutation_fail = true;
      break;
    }
  }
  delete [] check;
  if (permutation_fail) return CUTT_INVALID_PARAMETER;  

  return CUTT_SUCCESS;
}

cuttResult cuttPlan(cuttHandle* handle, int rank, int* dim, int* permutation, size_t sizeofType,
  cudaStream_t stream) {

  // Check that input parameters are valid
  cuttResult inpCheck = cuttPlanCheckInput(rank, dim, permutation, sizeofType);
  if (inpCheck != CUTT_SUCCESS) return inpCheck;

  // Create new handle
  *handle = curHandle;
  curHandle++;

  // Check that the current handle is available (it better be!)
  if (planStorage.count(*handle) != 0) return CUTT_INTERNAL_ERROR;

  // Get all possible ways tensor can be transposed
  int deviceID;
  cudaCheck(cudaGetDevice(&deviceID));
  cudaDeviceProp prop;
  cudaCheck(cudaGetDeviceProperties(&prop, deviceID));

  // Reduce ranks
  std::vector<int> redDim;
  std::vector<int> redPermutation;
  reduceRanks(rank, dim, permutation, redDim, redPermutation);

  // Create plans from reduced ranks
  std::list<cuttPlan_t> plans;
  // if (rank != redDim.size()) {
  //   if (!createPlans(redDim.size(), redDim.data(), redPermutation.data(), sizeofType, prop, plans)) return CUTT_INTERNAL_ERROR;
  // }

  // // Create plans from non-reduced ranks
  // if (!createPlans(rank, dim, permutation, sizeofType, prop, plans)) return CUTT_INTERNAL_ERROR;

  if (!createPlans(rank, dim, permutation, redDim.size(), redDim.data(), redPermutation.data(), 
    sizeofType, prop, plans)) return CUTT_INTERNAL_ERROR;

  // Count cycles
  for (auto it=plans.begin();it != plans.end();it++) {
    if (!it->countCycles(sizeofType, prop, 10)) return CUTT_INTERNAL_ERROR;
  }

  // Choose the plan
  std::list<cuttPlan_t>::iterator bestPlan = choosePlanHeuristic(plans);
  if (bestPlan == plans.end()) return CUTT_INTERNAL_ERROR;

  // bestPlan->print();

  // Create copy of the plan outside the list
  cuttPlan_t* plan = new cuttPlan_t();
  // NOTE: No deep copy needed here since device memory hasn't been allocated yet
  *plan = *bestPlan;
  // Set device pointers to NULL in the old copy of the plan so
  // that they won't be deallocated later when the object is destroyed
  bestPlan->nullDevicePointers();

  // Set stream
  plan->setStream(stream);

  // Activate plan
  plan->activate();

  // Insert plan into storage
  planStorage.insert( {*handle, plan} );

  return CUTT_SUCCESS;
}

cuttResult cuttPlanMeasure(cuttHandle* handle, int rank, int* dim, int* permutation, size_t sizeofType,
  cudaStream_t stream, void* idata, void* odata) {

  // Check that input parameters are valid
  cuttResult inpCheck = cuttPlanCheckInput(rank, dim, permutation, sizeofType);
  if (inpCheck != CUTT_SUCCESS) return inpCheck;

  if (idata == odata) return CUTT_INVALID_PARAMETER;

  // Create new handle
  *handle = curHandle;
  curHandle++;

  // Check that the current handle is available (it better be!)
  if (planStorage.count(*handle) != 0) return CUTT_INTERNAL_ERROR;

  // Get all possible ways tensor can be transposed
  int deviceID;
  cudaCheck(cudaGetDevice(&deviceID));
  cudaDeviceProp prop;
  cudaCheck(cudaGetDeviceProperties(&prop, deviceID));

  // Set shared memory configuration if necessary
  if (!devicesReady.count(deviceID)) {
    cuttKernelSetSharedMemConfig();
    devicesReady.insert(deviceID);
  }

  // Reduce ranks
  std::vector<int> redDim;
  std::vector<int> redPermutation;
  reduceRanks(rank, dim, permutation, redDim, redPermutation);

  // Create plans from reduced ranks
  std::list<cuttPlan_t> plans;
#if 0
  // if (rank != redDim.size()) {
    if (!createPlans(redDim.size(), redDim.data(), redPermutation.data(), sizeofType, prop, plans)) return CUTT_INTERNAL_ERROR;
  // }

  // Create plans from non-reduced ranks
  // if (!createPlans(rank, dim, permutation, sizeofType, prop, plans)) return CUTT_INTERNAL_ERROR;
#else
  if (!createPlans(rank, dim, permutation, redDim.size(), redDim.data(), redPermutation.data(), 
    sizeofType, prop, plans)) return CUTT_INTERNAL_ERROR;
#endif

  // // Count cycles
  // for (auto it=plans.begin();it != plans.end();it++) {
  //   if (!it->countCycles(sizeofType, prop, 1)) return CUTT_INTERNAL_ERROR;
  // }

  // // Count the number of elements
  size_t numBytes = sizeofType;
  for (int i=0;i < rank;i++) numBytes *= dim[i];

  // Choose the plan
  double bestTime = 1.0e40;
  auto bestPlan = plans.end();
  Timer timer;
  std::vector<double> times;
  for (auto it=plans.begin();it != plans.end();it++) {
    // Activate plan
    it->activate();
    // Clear output data to invalidate caches
    set_device_array<char>((char *)odata, -1, numBytes);
    cudaCheck(cudaDeviceSynchronize());
    timer.start();
    // Execute plan
    if (!cuttKernel(*it, idata, odata)) return CUTT_INTERNAL_ERROR;
    timer.stop();
    double curTime = timer.seconds();
    // it->print();
    // printf("curTime %1.2lf\n", curTime*1000.0);
    times.push_back(curTime);
    if (curTime < bestTime) {
      bestTime = curTime;
      bestPlan = it;
    }
  }
  if (bestPlan == plans.end()) return CUTT_INTERNAL_ERROR;

  // bestPlan = plans.begin();

  // printMatlab(prop, plans, times);
  // findMispredictionBest(plans, times, bestPlan, bestTime);
  // bestPlan->print();

  // Create copy of the plan outside the list
  cuttPlan_t* plan = new cuttPlan_t();
  *plan = *bestPlan;
  // Set device pointers to NULL in the old copy of the plan so
  // that they won't be deallocated later when the object is destroyed
  bestPlan->nullDevicePointers();

  // Set stream
  plan->setStream(stream);

  // Activate plan
  plan->activate();

  // Insert plan into storage
  planStorage.insert( {*handle, plan} );

  return CUTT_SUCCESS;
}

cuttResult cuttDestroy(cuttHandle handle) {
  auto it = planStorage.find(handle);
  if (it == planStorage.end()) return CUTT_INVALID_PLAN;
  // Delete instance of cuttPlan_t
  delete it->second;
  // Delete entry from plan storage
  planStorage.erase(it);
  return CUTT_SUCCESS;
}

cuttResult cuttExecute(cuttHandle handle, void* idata, void* odata) {
  auto it = planStorage.find(handle);
  if (it == planStorage.end()) return CUTT_INVALID_PLAN;

  if (idata == odata) return CUTT_INVALID_PARAMETER;

  cuttPlan_t& plan = *(it->second);

  int deviceID;
  cudaCheck(cudaGetDevice(&deviceID));
  if (deviceID != plan.deviceID) return CUTT_INVALID_DEVICE;

  // Set shared memory configuration if necessary
  if (!devicesReady.count(deviceID)) {
    cuttKernelSetSharedMemConfig();
    devicesReady.insert(deviceID);
  }

  if (!cuttKernel(plan, idata, odata)) return CUTT_INTERNAL_ERROR;
  return CUTT_SUCCESS;
}
