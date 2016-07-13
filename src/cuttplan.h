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
#ifndef CUTTPLAN_H
#define CUTTPLAN_H

#include <list>
#include <vector>
#include <cuda.h>
#include "cuttTypes.h"

// TILEDIM = warpSize
const int TILEDIM = 32;
const int TILEROWS = 8;

// Transposing methods
enum {Unknown, General,
  GeneralSplitInRank, GeneralSplitOutRank,
  TiledSingleInRank, TiledSingleOutRank,
  TiledSingleRank, TiledLeadVolSame,
  NumTransposeMethods};

// Tells how tensor is split into Mm and Mk and what method is used
// NOTE: sizeMm and sizeMk fully define the split
class TensorSplit {
public:
  // Transposing method
  int method;

  // Input volume
  int sizeMm;
  int volMm;

  // Output volume
  int sizeMk;
  int volMk;

  // {Input} U {Output}
  int sizeMmk;
  int volMmk;

  // {Input} CUT {Output} = Mk which is not in Mm
  int sizeMkBar;
  int volMkBar;

  // Remaining volume
  int sizeMbar;
  int volMbar;

  // Number of active thread blocks, for General method
  int numActiveBlock;

  // Number of splits, for GeneralSplitInRank and GeneralSplitOutRank methods
  int numSplit;

  TensorSplit();

  void print();

  void update(const int sizeMm_in, const int sizeMk_in, const int rank,
    const int* dim, const int* permutation);

  // Number of elements in shared memory space
  size_t shmem() const;

  // Number of elements in Mmk that are used effectively
  size_t volMmkUsed() const;

  // Bytes the shared memory space that needs to be allocated
  // (can be larger than volShmem() due to padding)
  size_t shmemAlloc(int sizeofType) const;

};

class LaunchConfig {
public:
 // Kernel launch configuration
  dim3 numthread;
  dim3 numblock;
  size_t shmemsize;

  // For the General method, number of registers to use for storage
  int numRegStorage;

  void print();

 };

// Class that stores the plan data
class cuttPlan_t {
public:
  // Device for which this plan was made
  int deviceID;

  // CUDA stream associated with the plan
  cudaStream_t stream;

  // Kernel launch configuration
  LaunchConfig launchConfig;
  
  // Rank of the tensor
  int rank;

  // Size of elements in tensor
  size_t sizeofType;

  TensorSplit tensorSplit;

  int cuDimMk;
  int cuDimMm;

  int2 tiledVol;

  // sizeMbar
  TensorConvInOut* Mbar;

  // sizeMmk
  TensorConvInOut* Mmk;

  // sizeMmk
  TensorConv* Msh;

  // For TiledSingleInRank
  TensorConv* Mk;

  // For TiledSingleOutRank
  TensorConv* Mm;

  // For GeneralSplitInRank
  int* posMk;

  // For GeneralSplitOutRank
  int* posMm;

  cuttPlan_t();
  ~cuttPlan_t();
  void print();
  void setStream(cudaStream_t stream_in);
  bool setup(const int rank_in, const int* dim, const int* permutation,
    const size_t sizeofType_in, cudaDeviceProp& prop, TensorSplit& tensorSplit_in);
private:
};

void getTensorSplits(const int rank, const int* dim, const int* permutation, const size_t sizeofType,
  cudaDeviceProp& prop, std::list<TensorSplit>& tensorSplits);
void reduceTensorSplits(std::list<TensorSplit>& tensorSplits);
void reduceMbar(const int rank, const int* dim, const int* permutation,
  std::list<TensorSplit>& tensorSplits,
  int& smallRank, std::vector<int>& smallDim, std::vector<int>& smallPermutation,
  std::list<TensorSplit>& smallTensorSplits);
std::list<TensorSplit>::iterator chooseTensorSplitHeuristic(std::list<TensorSplit>& tensorSplits);

#endif // CUTTPLAN_H
