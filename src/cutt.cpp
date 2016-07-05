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
#include <unordered_map>
#include <unordered_set>
#include "CudaUtils.h"
#include "cuttplan.h"
#include "cuttkernel.h"
#include "cuttTimer.h"
#include "cutt.h"


// Hash table to store the plans
static std::unordered_map< cuttHandle, cuttPlan_t* > plans;

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

cuttResult cuttPlan(cuttHandle* handle, int rank, int* dim, int* permutation, size_t sizeofType) {

  // Check that input parameters are valid
  cuttResult inpCheck = cuttPlanCheckInput(rank, dim, permutation, sizeofType);
  if (inpCheck != CUTT_SUCCESS) return inpCheck;

  // Create new handle
  *handle = curHandle;
  curHandle++;

  // Check that the current handle is available (it better be!)
  if (plans.count(*handle) != 0) return CUTT_INTERNAL_ERROR;

  // Get all possible ways tensor can be transposed
  int deviceID;
  cudaCheck(cudaGetDevice(&deviceID));
  cudaDeviceProp prop;
  cudaCheck(cudaGetDeviceProperties(&prop, deviceID));
  std::vector<TensorSplit> tensorSplits;
  getTensorSplits(rank, dim, permutation, sizeofType, prop, tensorSplits);

  // Choose the way
  int index = chooseTensorSplitHeuristic(tensorSplits);
  if (index == -1) return CUTT_INTERNAL_ERROR;

  // Create new plan
  cuttPlan_t* plan = new cuttPlan_t();
  if (!plan->setup(rank, dim, permutation, sizeofType, prop, tensorSplits[index])) return CUTT_INTERNAL_ERROR;

  // Insert plan into storage
  plans.insert( {*handle, plan} );

  return CUTT_SUCCESS;
}

cuttResult cuttPlanMeasure(cuttHandle* handle, int rank, int* dim, int* permutation, size_t sizeofType,
  void* idata, void* odata) {

  // Check that input parameters are valid
  cuttResult inpCheck = cuttPlanCheckInput(rank, dim, permutation, sizeofType);
  if (inpCheck != CUTT_SUCCESS) return inpCheck;

  if (idata == odata) return CUTT_INVALID_PARAMETER;

  // Create new handle
  *handle = curHandle;
  curHandle++;

  // Check that the current handle is available (it better be!)
  if (plans.count(*handle) != 0) return CUTT_INTERNAL_ERROR;

  // Get all possible ways tensor can be transposed
  int deviceID;
  cudaCheck(cudaGetDevice(&deviceID));
  cudaDeviceProp prop;
  cudaCheck(cudaGetDeviceProperties(&prop, deviceID));
  std::vector<TensorSplit> tensorSplits;
  getTensorSplits(rank, dim, permutation, sizeofType, prop, tensorSplits);

  // Set shared memory configuration if necessary
  if (!devicesReady.count(deviceID)) {
    cuttKernelSetSharedMemConfig();
    devicesReady.insert(deviceID);
  }

  // Choose the plan
  double bestTime = 1.0e40;
  cuttPlan_t* bestPlan = NULL;
  Timer timer;
  for (int i=0;i < tensorSplits.size();i++) {
    // Create new plan    
    cuttPlan_t* plan = new cuttPlan_t();
    if (!plan->setup(rank, dim, permutation, sizeofType, prop, tensorSplits[i])) {
      return CUTT_INTERNAL_ERROR;
    }
    cudaCheck(cudaDeviceSynchronize());
    timer.start();
    // Execute plan
    if (!cuttKernel(*plan, idata, odata)) return CUTT_INTERNAL_ERROR;
    cudaCheck(cudaDeviceSynchronize());
    timer.stop();
    double curTime = timer.seconds();
    if (curTime < bestTime) {
      if (bestPlan != NULL) delete bestPlan;
      bestTime = curTime;
      bestPlan = plan;
    } else {
      delete plan;
    }
  }

  if (bestPlan == NULL) return CUTT_INTERNAL_ERROR;

  // Insert plan into storage
  plans.insert( {*handle, bestPlan} );

  return CUTT_SUCCESS;
}

cuttResult cuttDestroy(cuttHandle handle) {
  auto it = plans.find(handle);
  if (it == plans.end()) return CUTT_INVALID_PLAN;
  // Delete instance of cuttPlan_t
  delete it->second;
  // Delete entry from plan storage
  plans.erase(it);
  return CUTT_SUCCESS;
}

cuttResult cuttSetStream(cuttHandle handle, cudaStream_t stream) {
  auto it = plans.find(handle);
  if (it == plans.end()) return CUTT_INVALID_PLAN;
  it->second->setStream(stream);
}

cuttResult cuttExecute(cuttHandle handle, void* idata, void* odata) {
  auto it = plans.find(handle);
  if (it == plans.end()) return CUTT_INVALID_PLAN;

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
