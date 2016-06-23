#ifndef CUTT_H
#define CUTT_H

// Handle type that is used to store and access cutt plans
typedef unsigned int cuttHandle;

// Return value
typedef enum cuttResult_t {
  CUTT_SUCCESS,            // Success
  CUTT_INVALID_PLAN,       // Invalid plan handle
  CUTT_INVALID_PARAMETER,  // Invalid input parameter
  CUTT_INVALID_DEVICE,     // Execution tried on device different than where plan was created
  CUTT_INTERNAL_ERROR,     // Internal error
  CUTT_UNDEFINED_ERROR,    // Undefined error
} cuttResult;

// Create a plan
cuttResult cuttPlan(cuttHandle* handle, int rank, int* dim, int* permutation, size_t sizeofType);

// Destroy plan
cuttResult cuttDestroy(cuttHandle handle);

// Set CUDA stream with the plan
cuttResult cuttSetStream(cuttHandle handle, cudaStream_t stream);

// Execute plan
cuttResult cuttExecute(cuttHandle handle, void* idata, void* odata);

#endif // CUTT_H
