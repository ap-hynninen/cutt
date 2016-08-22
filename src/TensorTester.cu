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

//
// Testing utilities
//
#include <cuda.h>
#include "CudaUtils.h"
#include "TensorTester.h"

__global__ void setTensorCheckPatternKernel(unsigned int* data, unsigned int ndata) {
  for (unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;i < ndata;i += blockDim.x*gridDim.x) {
    data[i] = i;
  }
}

template<typename T>
__global__ void checkTransposeKernel(T* data, unsigned int ndata, int rank, TensorConv* glTensorConv,
  TensorError_t* glError, int* glFail) {

  extern __shared__ unsigned int shPos[];

  const int warpLane = threadIdx.x & (warpSize - 1);
  TensorConv tc;
  if (warpLane < rank) {
    tc = glTensorConv[warpLane];
  }

  TensorError_t error;
  error.pos = 0xffffffff;
  error.refVal = 0;
  error.dataVal = 0;

  for (int base = blockIdx.x*blockDim.x;base < ndata;base += blockDim.x*gridDim.x) {
    int i = base + threadIdx.x;
    T dataValT = (i < ndata) ? data[i] : -1;
    int refVal = 0;
    for (int j=0;j < rank;j++) {
      refVal += ((i/__shfl(tc.c,j)) % __shfl(tc.d,j))*__shfl(tc.ct,j);
    }

    int dataVal = (dataValT & 0xffffffff)/(sizeof(T)/4);

    if (i < ndata && refVal != dataVal && i < error.pos) {
      error.pos = i;
      error.refVal = refVal;
      error.dataVal = dataVal;
    }
  }

  // Set FAIL flag
  if (error.pos != 0xffffffff) {
    // printf("error %d %d %d\n", error.pos, error.refVal, error.dataVal);
    *glFail = 1;
  }

  shPos[threadIdx.x] = error.pos;
  __syncthreads();
  for (int d=1;d < blockDim.x;d *= 2) {
    int t = threadIdx.x + d;
    unsigned int posval = (t < blockDim.x) ? shPos[t] : 0xffffffff;
    __syncthreads();
    shPos[threadIdx.x] = min(posval, shPos[threadIdx.x]);
  __syncthreads();
  }
  // Minimum error.pos is in shPos[0] (or 0xffffffff in case of no error)

  if (shPos[0] != 0xffffffff && shPos[0] == error.pos) {
    // Error has occured and this thread has the minimum error.pos
    // printf("BOO error %d %d %d | %d\n", error.pos, error.refVal, error.dataVal, blockIdx.x);
    glError[blockIdx.x] = error;
  }

}

// ################################################################################
// ################################################################################
// ################################################################################

//
// Class constructor
//
TensorTester::TensorTester() : maxRank(32), maxNumblock(256) {
  h_tensorConv = new TensorConv[maxRank];
  h_error      = new TensorError_t[maxNumblock];
  allocate_device<TensorConv>(&d_tensorConv, maxRank);
  allocate_device<TensorError_t>(&d_error, maxNumblock);
  allocate_device<int>(&d_fail, 1);
}

//
// Class destructor
//
TensorTester::~TensorTester() {
  delete [] h_tensorConv;
  delete [] h_error;
  deallocate_device<TensorConv>(&d_tensorConv);
  deallocate_device<TensorError_t>(&d_error);
  deallocate_device<int>(&d_fail);
}

void TensorTester::setTensorCheckPattern(unsigned int* data, unsigned int ndata) {
  int numthread = 512;
  int numblock = min(65535, (ndata - 1)/numthread + 1 );
  setTensorCheckPatternKernel<<< numblock, numthread >>>(data, ndata);
  cudaCheck(cudaGetLastError());
}

// void calcTensorConv(const int rank, const int* dim, const int* permutation,
//   TensorConv* tensorConv) {

//   tensorConv[0].c = 1;
//   tensorConv[0].d = dim[0];
//   tensorConv[permutation[0]].ct = 1;
//   int ct_prev = 1;
//   for (int i=1;i < rank;i++) {
//     tensorConv[i].c = tensorConv[i-1].c*dim[i-1];
//     tensorConv[i].d = dim[i];
//     int ct = ct_prev*dim[permutation[i-1]];
//     tensorConv[permutation[i]].ct = ct;
//     ct_prev = ct;
//   }

// }

//
// Calculates tensor conversion constants. Returns total volume of tensor
//
int TensorTester::calcTensorConv(const int rank, const int* dim, const int* permutation,
  TensorConv* tensorConv) {

  int vol = dim[0];

  tensorConv[permutation[0]].c  = 1;
  tensorConv[0].ct = 1;
  tensorConv[0].d  = dim[0];
  for (int i=1;i < rank;i++) {
    vol *= dim[i];

    tensorConv[permutation[i]].c = tensorConv[permutation[i-1]].c*dim[permutation[i-1]];

    tensorConv[i].d  = dim[i];
    tensorConv[i].ct = tensorConv[i-1].ct*dim[i-1];

  }

  return vol;
}

template<typename T> bool TensorTester::checkTranspose(int rank, int* dim, int* permutation, T* data) {

  if (rank > 32) {
    return false;
  }

  int ndata = calcTensorConv(rank, dim, permutation, h_tensorConv);
  copy_HtoD<TensorConv>(h_tensorConv, d_tensorConv, rank);

  // printf("tensorConv\n");
  // for (int i=0;i < rank;i++) {
  //   printf("%d %d %d\n", h_tensorConv[i].c, h_tensorConv[i].d, h_tensorConv[i].ct);
  // }

  set_device_array<TensorError_t>(d_error, 0, maxNumblock);
  set_device_array<int>(d_fail, 0, 1);

  int numthread = 512;
  int numblock = min(maxNumblock, (ndata - 1)/numthread + 1 );
  int shmemsize = numthread*sizeof(unsigned int);
  checkTransposeKernel<<< numblock, numthread, shmemsize >>>(data, ndata, rank, d_tensorConv, d_error, d_fail);
  cudaCheck(cudaGetLastError());

  int h_fail;
  copy_DtoH<int>(d_fail, &h_fail, 1);
  cudaCheck(cudaDeviceSynchronize());

  if (h_fail) {
    copy_DtoH_sync<TensorError_t>(d_error, h_error, maxNumblock);
    TensorError_t error;
    error.pos = 0x0fffffff;
    for (int i=0;i < numblock;i++) {
      // printf("%d %d %d\n", error.pos, error.refVal, error.dataVal);
      if (h_error[i].refVal != h_error[i].dataVal && error.pos > h_error[i].pos) {
        error = h_error[i];
      }
    }
    printf("TensorTester::checkTranspose FAIL at %d ref %d data %d\n", error.pos, error.refVal, error.dataVal);
    return false;
  }

  return true;
}

// Explicit instances
template bool TensorTester::checkTranspose<int>(int rank, int* dim, int* permutation, int* data);
template bool TensorTester::checkTranspose<long long int>(int rank, int* dim, int* permutation, long long int* data);
