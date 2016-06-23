#include <cuda.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <unordered_set>
#include "CudaUtils.h"
#include "cuttplan.h"
#include "cuttkernel.h"
#include "cutt.h"


// Hash table to store the plans
static std::unordered_map< cuttHandle, cuttPlan_t* > plans;

// Current handle
static cuttHandle curHandle = 0;

// Table of devices that have been initialized
static std::unordered_set<int> devicesReady;

cuttResult cuttPlan(cuttHandle* handle, int rank, int* dim, int* permutation, size_t sizeofType) {
  // Check sizeofType
  if (sizeofType != 4 && sizeofType != 8) return CUTT_INVALID_PARAMETER;
  // Check rank
  if (rank <= 1) return CUTT_INVALID_PARAMETER;
  // Check dim[]
  for (int i=0;i < rank;i++) {
    if (dim[i] <= 1) return CUTT_INVALID_PARAMETER;
  }
  // Check permutation
  for (int i=0;i < rank;i++) {
    if (permutation[i] < 0 || permutation[i] >= rank) return CUTT_INVALID_PARAMETER;
  }

  // Create new handle
  *handle = curHandle;
  curHandle++;

  // Check that the current handle is available (it better be!)
  if (plans.count(*handle) != 0) return CUTT_INTERNAL_ERROR;

  // Create new plan
  cuttPlan_t* plan = new cuttPlan_t();
  plan->setup(rank, dim, permutation, sizeofType);

  // Insert plan into storage
  plans.insert( {*handle, plan} );

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
