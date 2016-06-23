#ifndef CUTTPLAN_H
#define CUTTPLAN_H

#include <vector>
#include <cuda.h>
#include "cuttTypes.h"

// TILEDIM = warpSize
const int TILEDIM = 32;
const int TILEROWS = 8;

// Structure that stores the plan data
class cuttPlan_t {
public:
  // Device for which this plan was made
  int deviceID;

  // // Maximum number of threads per block
  // int maxThreadsPerBlock;

  // // Maximum number of threads per SM
  // int maxThreadsPerMultiProcessor;

  // CUDA stream associated with the plan
  cudaStream_t stream;

  // Kernel launch configuration
  dim3 numthread;
  dim3 numblock;
  size_t shmemsize;

  // For the General method, number of registers to use for storage
  int numRegStorage;
  
  // Rank of the tensor
  int rank;

  // Size of elements in tensor
  size_t sizeofType;

  // Input volume
  int sizeMm;
  int volMm;

  // Output volume
  int sizeMk;
  int volMk;

  // {Input} U {Output}
  int sizeMmk;
  int volMmk;

  // Remaining volume
  int sizeMbar;
  int volMbar;

  int cuDimMk;
  int cuDimMm;

  // Transposing method
  enum {Unknown, General, TiledSingleRank, TiledLeadVolSame};
  int method;

  int2 readVol;
  // int2 writeVol;

  // sizeMbar
  TensorConvInOut* Mbar;

  // sizeMmk
  TensorConvInOut* Mmk;

  // sizeMmk
  TensorConv* Msh;

  cuttPlan_t();
  ~cuttPlan_t();
  void setStream(cudaStream_t stream_in);
  bool setup(const int rank_in, const int* dim, const int* permutation, const size_t sizeofType_in);
private:
  void setupMm(std::vector<bool>& isMm, const int* dim);
  void setupMk(std::vector<bool>& isMk, const int* dim);
  void setupMmk(std::vector<bool>& isMm, std::vector<bool>& isMk, const int* dim);
  void setupMbar(std::vector<bool>& isMm, std::vector<bool>& isMk, const int* dim);
};

#endif // CUTTPLAN_H
