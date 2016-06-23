#include <cuda.h>
#include "CudaUtils.h"
#include "cuttkernel.h"

//
// Returns scalar tensor position. Each lane has the same p
// NOTE: c and d on inactive warps must be 1 !!
//
__device__ __forceinline__
int tensorPos(
  const int p, const int rank, const int c, const int d, const int ct,
  const int numLane=warpSize
  ) {

  int r = ((p/c) % d)*ct;
#pragma unroll
  for (int i=numLane/2;i >= 1;i/=2) {
    r += __shfl_xor(r, i);
  }
  return r;

}

//__constant__ int args[2];

//
// Transpose when Mm and Mk don't overlap and contain only single rank
//
//  dim3 numthread(TILEDIM, TILEROWS, 1);
//  dim3 numblock((plan.volMm-1)/TILEDIM+1, (plan.volMk-1)/TILEDIM+1, plan.volMbar);
//
template <typename T>
__global__ void transposeTiledSingleRank(
  const int volMbar, const int sizeMbar,
  const int2 readVol, const int cuDimMk, const int cuDimMm,
  const TensorConvInOut* __restrict__ glMbar,
  const T* __restrict__ dataIn, T* __restrict__ dataOut) {

  // Shared memory
  __shared__ T shTile[TILEDIM][TILEDIM+1];

  const int warpLane = threadIdx.x & (warpSize - 1);
  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = glMbar[warpLane];
  }

  const int xin = blockIdx.x * TILEDIM + threadIdx.x;
  const int yin = blockIdx.y * TILEDIM + threadIdx.y;

  const int xout = blockIdx.x * TILEDIM + threadIdx.y;
  const int yout = blockIdx.y * TILEDIM + threadIdx.x;

  const unsigned int maskIny = __ballot((yin + warpLane < readVol.y))*(xin < readVol.x);
  const unsigned int maskOutx = __ballot((xout + warpLane < readVol.x))*(yout < readVol.y);

  const int posMinorIn = xin + yin*cuDimMk;
  const int posMinorOut = yout + xout*cuDimMm;
  const int posInAdd = TILEROWS*cuDimMk;
  const int posOutAdd = TILEROWS*cuDimMm;

  for (int posMbar=blockIdx.z;posMbar < volMbar;posMbar += gridDim.z)
  {

    // Compute global memory positions
    int posMajorIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
    int posMajorOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      posMajorIn += __shfl_xor(posMajorIn, i);
      posMajorOut += __shfl_xor(posMajorOut, i);
    }
    int posIn = posMajorIn + posMinorIn;
    int posOut = posMajorOut + posMinorOut;

//     int posMajorIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
// #pragma unroll
//     for (int i=16;i >= 1;i/=2) {
//       posMajorIn += __shfl_xor(posMajorIn, i);
//     }
//     int posIn = posMajorIn + posMinorIn;

//     int posMajorOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
// #pragma unroll
//     for (int i=16;i >= 1;i/=2) {
//       posMajorOut += __shfl_xor(posMajorOut, i);
//     }
//     int posOut = posMajorOut + posMinorOut;

    // Read from global memory
    __syncthreads();

    // Read data into shared memory tile
#pragma unroll
    for (int j=0;j < TILEDIM;j += TILEROWS) {
      // int pos = posIn + j*cuDimMk;
      // if (xin < readVol.x && yin + j < readVol.y) {
      if ((maskIny & (1 << j)) != 0) {
        shTile[threadIdx.y + j][threadIdx.x] = dataIn[posIn];
      }
      posIn += posInAdd;
    }

    // Write to global memory
    __syncthreads();

#pragma unroll
    for (int j=0;j < TILEDIM;j += TILEROWS) {
      // int pos = posOut + j*cuDimMm;
      // if (xout + j < readVol.x && yout < readVol.y) {
      if ((maskOutx & (1 << j)) != 0 ) {
        dataOut[posOut] = shTile[threadIdx.x][threadIdx.y + j];
      }
      posOut += posOutAdd;
    }

  }
  
}

//
// General transpose. Thread block loads plan.volMmk number of elements
//
template <typename T, int numRegStorage>
__global__ void transposeGeneral(
  const int volMm, const int volMk, const int volMmk, const int volMbar,
  const int sizeMmk, const int sizeMbar,
  const TensorConvInOut* __restrict__ gl_Mmk,
  const TensorConvInOut* __restrict__ gl_Mbar,
  const TensorConv* __restrict__ gl_Msh,
  const T* __restrict__ dataIn, T* __restrict__ dataOut) {

  // Shared memory. volMmk elements
  extern __shared__ char shBuffer_char[];
  T* shBuffer = (T *)shBuffer_char;

  const int warpLane = threadIdx.x & (warpSize - 1);

  TensorConvInOut Mmk;
  Mmk.c_in = 1;
  Mmk.d_in = 1;
  Mmk.c_out = 1;
  Mmk.d_out = 1;
  if (warpLane < sizeMmk) {
    Mmk = gl_Mmk[warpLane];
  }
  TensorConv Msh;
  Msh.c = 1;
  Msh.d = 1;
  if (warpLane < sizeMmk) {
    Msh = gl_Msh[warpLane];
  }

  // Pre-compute tensor positions in Mmk
  // 3*numRegStorage registers
  int posMmkIn[numRegStorage];
  int posMmkOut[numRegStorage];
  int posSh[numRegStorage];
#pragma unroll
  for (int j=0;j < numRegStorage;j++) {
    posMmkIn[j] = 0;
    posMmkOut[j] = 0;
    posSh[j] = 0;
  }
  for (int i=0;i < sizeMmk;i++) {
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      posMmkIn[j]  += (((threadIdx.x + j*blockDim.x)/__shfl(Mmk.c_in,i)) % __shfl(Mmk.d_in,i))*__shfl(Mmk.ct_in,i);
      posMmkOut[j] += (((threadIdx.x + j*blockDim.x)/__shfl(Mmk.c_out,i)) % __shfl(Mmk.d_out,i))*__shfl(Mmk.ct_out,i);
      posSh[j] += (((threadIdx.x + j*blockDim.x)/__shfl(Msh.c,i)) % __shfl(Msh.d,i))*__shfl(Msh.ct,i);
    }
  }

  // 6 registers
  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = gl_Mbar[warpLane];
  }

  for (int posMbar=blockIdx.x;posMbar < volMbar;posMbar += gridDim.x)
  {

    int posMbarOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      posMbarOut += __shfl_xor(posMbarOut, i);
    }

    // Read from global memory
    // int posMbarIn = tensorPos(posMbar, sizeMbar, Mbar.c_in, Mbar.d_in, Mbar.ct_in);
    int posMbarIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      posMbarIn += __shfl_xor(posMbarIn, i);
    }

    __syncthreads();

#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk = threadIdx.x + j*blockDim.x;
      int posIn = posMbarIn + posMmkIn[j];
      if (posMmk < volMmk) shBuffer[posMmk] = dataIn[posIn];
    }

    // Write to global memory
      // int posMbarOut = tensorPos(posMbar, sizeMbar, Mbar.c_out, Mbar.d_out, Mbar.ct_out);
//       int posMbarOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
// #pragma unroll
//       for (int i=16;i >= 1;i/=2) {
//         posMbarOut += __shfl_xor(posMbarOut, i);
//       }

    __syncthreads();

#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk = threadIdx.x + j*blockDim.x;
      int posOut = posMbarOut + posMmkOut[j];
      if (posMmk < volMmk) dataOut[posOut] = shBuffer[posSh[j]];
    }


  }
  
}

//
// Transpose when the lead dimension is the same, e.g. (1, 2, 3) -> (1, 3, 2)
//
//  dim3 numthread(TILEDIM, TILEROWS, 1);
//  dim3 numblock((plan.volMm-1)/TILEDIM+1, (plan.volMk-1)/TILEDIM+1, plan.volMbar);
//
template <typename T>
__global__ void transposeTiledLeadVolSame(
  const int volMbar, const int sizeMbar,
  const int cuDimMk, const int cuDimMm,
  const int2 readVol,
  const TensorConvInOut* __restrict__ gl_Mbar,
  const T* __restrict__ dataIn, T* __restrict__ dataOut) {

  const int warpLane = threadIdx.x & (warpSize - 1);
  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = gl_Mbar[warpLane];
  }

  // int* dimMmkIn  = &transposeArg[0];

  const int x = blockIdx.x * TILEDIM + threadIdx.x;
  const int y = blockIdx.y * TILEDIM + threadIdx.y;

  for (int posMbar=blockIdx.z;posMbar < volMbar;posMbar += gridDim.z)
  {

    // Variables where values are stored
    T val[TILEDIM/TILEROWS];

    // Read global memory
    {
      int pos0 = tensorPos(posMbar, sizeMbar, Mbar.c_in, Mbar.d_in, Mbar.ct_in);
      pos0 += x + y*cuDimMk;

#pragma unroll
      for (int j=0;j < TILEDIM;j += TILEROWS) {
        int pos  = pos0  + j*cuDimMk;
        if ((x < readVol.x) && (y + j < readVol.y)) {
        // if ((x < dimMmkIn[0]) && (y + j < dimMmkIn[1])) {
          val[j/TILEROWS] = dataIn[pos];
        }
      }
    }

    // Write global memory
    {
      int pos0 = tensorPos(posMbar, sizeMbar, Mbar.c_out, Mbar.d_out, Mbar.ct_out);
      pos0 += x + y*cuDimMm;

#pragma unroll
      for (int j=0;j < TILEDIM;j += TILEROWS) {
        int pos = pos0 + j*cuDimMm;
        if ((x < readVol.x) && (y + j < readVol.y)) {
        // if ((x < dimMmkIn[0]) && (y + j < dimMmkIn[1])) {
          dataOut[pos] = val[j/TILEROWS];
        }
      }
    }

  }
  
}

//######################################################################################
//######################################################################################
//######################################################################################

//
// Sets shared memory bank configuration for all kernels. Needs to be called once per device.
//
void cuttKernelSetSharedMemConfig() {  
#define CALL(NREG) cudaCheck(cudaFuncSetSharedMemConfig(transposeGeneral<float, NREG>, cudaSharedMemBankSizeFourByte ))
    CALL(1);
    CALL(2);
    CALL(3);
    CALL(4);
    CALL(5);
    CALL(6);
    CALL(7);
    CALL(8);
#undef CALL
#define CALL(NREG) cudaCheck(cudaFuncSetSharedMemConfig(transposeGeneral<double, NREG>, cudaSharedMemBankSizeEightByte ))
    CALL(1);
    CALL(2);
    CALL(3);
    CALL(4);
    CALL(5);
    CALL(6);
    CALL(7);
    CALL(8);
#undef CALL

  cudaCheck(cudaFuncSetSharedMemConfig(transposeTiledSingleRank<float>, cudaSharedMemBankSizeFourByte));
  cudaCheck(cudaFuncSetSharedMemConfig(transposeTiledLeadVolSame<float>, cudaSharedMemBankSizeFourByte));

  cudaCheck(cudaFuncSetSharedMemConfig(transposeTiledSingleRank<double>, cudaSharedMemBankSizeEightByte));
  cudaCheck(cudaFuncSetSharedMemConfig(transposeTiledLeadVolSame<double>, cudaSharedMemBankSizeEightByte));
}

//
// Returns the maximum number of active blocks per SM
//
int getNumActiveBlock(cuttPlan_t& plan) {
  int numActiveBlock;
  int numthread = plan.numthread.x * plan.numthread.y * plan.numthread.z;
  switch(plan.method) {
    case cuttPlan_t::General:
    {
    #define CALL(TYPE, NREG) \
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock, \
        transposeGeneral<TYPE, NREG>, numthread, plan.shmemsize)
      switch(plan.numRegStorage) {
        case 1:
        if (plan.sizeofType == 4) CALL(float,  1);
        if (plan.sizeofType == 8) CALL(double, 1);
        break;
        case 2:
        if (plan.sizeofType == 4) CALL(float,  2);
        if (plan.sizeofType == 8) CALL(double, 2);
        break;
        case 3:
        if (plan.sizeofType == 4) CALL(float,  3);
        if (plan.sizeofType == 8) CALL(double, 3);
        break;
        case 4:
        if (plan.sizeofType == 4) CALL(float,  4);
        if (plan.sizeofType == 8) CALL(double, 4);
        break;
        case 5:
        if (plan.sizeofType == 4) CALL(float,  5);
        if (plan.sizeofType == 8) CALL(double, 5);
        break;
        case 6:
        if (plan.sizeofType == 4) CALL(float,  6);
        if (plan.sizeofType == 8) CALL(double, 6);
        break;
        case 7:
        if (plan.sizeofType == 4) CALL(float,  7);
        if (plan.sizeofType == 8) CALL(double, 7);
        break;
        case 8:
        if (plan.sizeofType == 4) CALL(float,  8);
        if (plan.sizeofType == 8) CALL(double, 8);
        break;
      }
    #undef CALL
    }
    break;
    case cuttPlan_t::TiledSingleRank:
    {
      if (plan.sizeofType == 4) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiledSingleRank<float>, numthread, plan.shmemsize);
      } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiledSingleRank<double>, numthread, plan.shmemsize);
      }
    }
    break;
    case cuttPlan_t::TiledLeadVolSame:
    {
      if (plan.sizeofType == 4) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiledLeadVolSame<float>, numthread, plan.shmemsize);
      } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiledLeadVolSame<double>, numthread, plan.shmemsize);
      }
    }
    break;
  }

  return numActiveBlock;
}

//
// Sets up kernel launch configuration
//
// Returns the number of blocks per SM that can be achieved on the General kernel
// NOTE: Returns 0 when kernel execution is not possible
//
// Sets:
// plan.numthread
// plan.numblock
// plan.shmemsize
// plan.numRegStorage  (for General method)
//
int cuttKernelLaunchConfiguration(cuttPlan_t& plan, cudaDeviceProp& prop) {
  switch(plan.method) {
    case cuttPlan_t::General:
    {
      // Amount of shared memory required
      plan.shmemsize = plan.volMmk*plan.sizeofType;

      // Check that we're not using too much shared memory per block
      if (plan.shmemsize > prop.sharedMemPerBlock) return 0;

      plan.numthread.x = ((plan.volMmk - 1)/prop.warpSize + 1)*prop.warpSize;
      plan.numthread.y = 1;
      plan.numthread.z = 1;
      plan.numblock.x = max(1, plan.volMbar);
      plan.numblock.x = min(256, plan.numblock.x);
      plan.numblock.y = 1;
      plan.numblock.z = 1;

      plan.numRegStorage = (plan.volMmk - 1)/plan.numthread.x + 1;
      if (plan.numRegStorage > maxNumRegStorage) {
        // Find number of threads that works
        plan.numthread.x = (( (plan.volMmk - 1)/maxNumRegStorage)/prop.warpSize + 1)*prop.warpSize;
        plan.numRegStorage = (plan.volMmk - 1)/plan.numthread.x + 1;
      }

      // Check that we're not using too many threads or templated register storage
      if (plan.numthread.x > prop.maxThreadsPerBlock || plan.numRegStorage > maxNumRegStorage) return 0;

    }
    break;
    case cuttPlan_t::TiledSingleRank:
    {
      plan.numthread.x = TILEDIM;
      plan.numthread.y = TILEROWS;
      plan.numthread.z = 1;
      plan.numblock.x = (plan.volMm - 1)/TILEDIM + 1;
      plan.numblock.y = (plan.volMk - 1)/TILEDIM + 1;
      plan.numblock.z = max(1, plan.volMbar);
      plan.numblock.z = min(128, plan.numblock.z);
      plan.shmemsize = 0;
      plan.numRegStorage = 0;
    }
    break;
    case cuttPlan_t::TiledLeadVolSame:
    {
      plan.numthread.x = TILEDIM;
      plan.numthread.y = TILEROWS;
      plan.numthread.z = 1;
      plan.numblock.x = (plan.volMmk - 1)/TILEDIM + 1;
      plan.numblock.y = (plan.volMmk - 1)/TILEDIM + 1;
      plan.numblock.z = max(1, plan.volMbar);
      plan.numblock.z = min(256, plan.numblock.z);
      plan.shmemsize = 0;
      plan.numRegStorage = 0;
    }
    break;
  }

  plan.numblock.x = min(65535, plan.numblock.x);
  plan.numblock.y = min(65535, plan.numblock.y);
  plan.numblock.z = min(65535, plan.numblock.z);

  // Return the number of active blocks with these settings
  return getNumActiveBlock(plan);
}

bool cuttKernel(cuttPlan_t& plan, void* dataIn, void* dataOut) {

  printf("numthread %d %d %d numblock %d %d %d shmemsize %d numRegStorage %d\n",
    plan.numthread.x, plan.numthread.y, plan.numthread.z,
    plan.numblock.x, plan.numblock.y, plan.numblock.z,
    plan.shmemsize, plan.numRegStorage);

  switch(plan.method) {
    case cuttPlan_t::General:
    {
      switch(plan.numRegStorage) {
#define CALL(TYPE, NREG) \
    transposeGeneral<TYPE, NREG> <<< plan.numblock, plan.numthread, plan.shmemsize, plan.stream >>> \
      (plan.volMm, plan.volMk, plan.volMmk, plan.volMbar, \
      plan.sizeMmk, plan.sizeMbar, \
      plan.Mmk, plan.Mbar, plan.Msh, (TYPE *)dataIn, (TYPE *)dataOut)
        case 1:
        if (plan.sizeofType == 4) CALL(float,  1);
        if (plan.sizeofType == 8) CALL(double, 1);
        break;
        case 2:
        if (plan.sizeofType == 4) CALL(float,  2);
        if (plan.sizeofType == 8) CALL(double, 2);
        break;
        case 3:
        if (plan.sizeofType == 4) CALL(float,  3);
        if (plan.sizeofType == 8) CALL(double, 3);
        break;
        case 4:
        if (plan.sizeofType == 4) CALL(float,  4);
        if (plan.sizeofType == 8) CALL(double, 4);
        break;
        case 5:
        if (plan.sizeofType == 4) CALL(float,  5);
        if (plan.sizeofType == 8) CALL(double, 5);
        break;
        case 6:
        if (plan.sizeofType == 4) CALL(float,  6);
        if (plan.sizeofType == 8) CALL(double, 6);
        break;
        case 7:
        if (plan.sizeofType == 4) CALL(float,  7);
        if (plan.sizeofType == 8) CALL(double, 7);
        break;
        case 8:
        if (plan.sizeofType == 4) CALL(float,  8);
        if (plan.sizeofType == 8) CALL(double, 8);
        break;
        default:
        printf("cuttKernel no template implemented for numRegStorage %d\n", plan.numRegStorage);
        return false;
#undef CALL
      }

    }
    break;

    case cuttPlan_t::TiledSingleRank:
    {
#define CALL(TYPE) \
      transposeTiledSingleRank<TYPE> <<< plan.numblock, plan.numthread, 0, plan.stream >>> \
      (plan.volMbar, plan.sizeMbar, plan.readVol, plan.cuDimMk, plan.cuDimMm, \
        plan.Mbar, (TYPE *)dataIn, (TYPE *)dataOut)

      // dim3 numthread(TILEDIM, TILEROWS, 1);
      // dim3 numblock((plan.volMm-1)/TILEDIM+1, (plan.volMk-1)/TILEDIM+1, plan.volMbar);
      // numblock.z = min(256, plan.volMbar);
      // numblock.z = min(65535, numblock.z);

      // printf("numthread %d %d %d numblock %d %d %d\n", numthread.x, numthread.y, numthread.z,
      //   numblock.x, numblock.y, numblock.z);

      if (plan.sizeofType == 4) CALL(float);
      if (plan.sizeofType == 8) CALL(double);
#undef CALL
    }
    break;

    case cuttPlan_t::TiledLeadVolSame:
    {
#define CALL(TYPE) \
      transposeTiledLeadVolSame<TYPE> <<< plan.numblock, plan.numthread, 0, plan.stream >>> \
      (plan.volMbar, plan.sizeMbar, plan.cuDimMk, plan.cuDimMm, plan.readVol, \
        plan.Mbar, (TYPE *)dataIn, (TYPE *)dataOut)

      // dim3 numthread(TILEDIM, TILEROWS, 1);
      // dim3 numblock((plan.readVol.x-1)/TILEDIM+1, (plan.readVol.y-1)/TILEDIM+1, plan.volMbar);
      // numblock.z = min(256, plan.volMbar);
      // numblock.z = min(65535, numblock.z);

      // printf("numthread %d %d %d numblock %d %d %d\n", numthread.x, numthread.y, numthread.z,
      //   numblock.x, numblock.y, numblock.z);

      if (plan.sizeofType == 4) CALL(float);
      if (plan.sizeofType == 8) CALL(double);
#undef CALL
    }
    break;

  }

  cudaCheck(cudaGetLastError());
  return true;
}
