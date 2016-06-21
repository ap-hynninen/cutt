#ifndef CUDATRANSPOSE_H
#define CUDATRANSPOSE_H

#include "CudaUtils.h"
#include "TensorConv.h"

// struct MbarRecord {
//   int posDimIn;
//   int dimIn;
//   int cuDimIn;
//   int posDimOut;
//   int dimOut;
//   int cuDimOut;
// };

struct MmkRecord {
  int posDim;
  int dim;
  int cuDim;
};

struct TensorConvInOut {
  int c_in;
  int d_in;
  int ct_in;
  int c_out;
  int d_out;
  int ct_out;
};

// TILEDIM = warpSize
const int TILEDIM = 32;
const int TILEROWS = 8;

// Arguments for transposeTensorKernel
// Enough to support tensors of rank 200 or so.
const int transposeArgSize = 2048 - 6 - 2*3;
__constant__ int transposeArg[transposeArgSize];

//
// Transpose when Mm and Mk don't overlap and contain only single rank
//
//  dim3 numthread(TILEDIM, TILEROWS, 1);
//  dim3 numblock((plan.volMm-1)/TILEDIM+1, (plan.volMk-1)/TILEDIM+1, plan.volMbar);
//
template <typename T>
__global__ void transposeTensorKernelArg(
  const int volMbar, const int sizeMbar,
  const int2 readVol, const int2 writeVol,
  const int cuDimMk, const int cuDimMm,
  const TensorConvInOut* __restrict__ gl_Mbar,
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
    Mbar = gl_Mbar[warpLane];
  }

  const int xin = blockIdx.x * TILEDIM + threadIdx.x;
  const int yin = blockIdx.y * TILEDIM + threadIdx.y;

  const int xout = blockIdx.x * TILEDIM + threadIdx.y;
  const int yout = blockIdx.y * TILEDIM + threadIdx.x;

  const unsigned int maskIny = __ballot((yin + warpLane < readVol.y))*(xin < readVol.x);
  const unsigned int maskOutx = __ballot((xout + warpLane < writeVol.y))*(yout < writeVol.x);

  const int posMinorIn = xin + yin*cuDimMk;
  const int posMinorOut = yout + xout*cuDimMm;

  const int posInAdd = TILEROWS*cuDimMk;
  const int posOutAdd = TILEROWS*cuDimMm;

  for (int blockz=blockIdx.z;blockz < volMbar;blockz += blockDim.z*gridDim.z)
  {

    // Read from global memory
    {
      int posMajorIn = tensorPos(blockz, sizeMbar, Mbar.c_in, Mbar.d_in, Mbar.ct_in);
      int pos = posMajorIn + posMinorIn;

      __syncthreads();

      // Read data into shared memory tile
#pragma unroll
      for (int j=0;j < TILEDIM;j += TILEROWS) {
        if ((maskIny & (1 << j)) != 0) {
          shTile[threadIdx.y + j][threadIdx.x] = dataIn[pos];
        }
        pos += posInAdd;
      }
    }

    // Write to global memory
    {
      int posMajorOut = tensorPos(blockz, sizeMbar, Mbar.c_out, Mbar.d_out, Mbar.ct_out);
      int pos = posMajorOut + posMinorOut;

      __syncthreads();

#pragma unroll
      for (int j=0;j < TILEDIM;j += TILEROWS) {
        if ((maskOutx & (1 << j)) != 0 ) {
          dataOut[pos] = shTile[threadIdx.x][threadIdx.y + j];
        }
        pos += posOutAdd;
      }
    }

  }
  
}

//
// General transpose. Thread block loads plan.volMmk number of elements
//
// numthread.x = warpSize
// numblock(plan.volMbar);
//
template <typename T, int nloopVolMmk>
__global__ void transposeTensorKernelArg_general(
  const int volMm, const int volMk, const int volMmk, const int volMbar,
  const int sizeMmk, const int sizeMbar,
  const TensorConvInOut* __restrict__ gl_Mmk,
  const TensorConvInOut* __restrict__ gl_Mbar,
  const TensorConv* __restrict__ gl_Msh,
  const T* __restrict__ dataIn, T* __restrict__ dataOut) {

  // Shared memory. volMmk elements
  extern __shared__ T shBuffer[];

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
  // 3*nloopVolMmk registers
#if 0
  int posMmkIn[nloopVolMmk];
  int posMmkOut[nloopVolMmk];
#pragma unroll
  for (int j=0;j < nloopVolMmk;j++) {
    posMmkIn[j] = 0;
    posMmkOut[j] = 0;
  }
  for (int i=0;i < sizeMmk;i++) {
#pragma unroll
    for (int j=0;j < nloopVolMmk;j++) {
      posMmkIn[j] += (((threadIdx.x + j*blockDim.x)/__shfl(Mmk.c_in,i)) % __shfl(Mmk.d_in,i))*__shfl(Mmk.ct_in,i);
      posMmkOut[j] += (((threadIdx.x + j*blockDim.x)/__shfl(Mmk.c_out,i)) % __shfl(Mmk.d_out,i))*__shfl(Mmk.ct_out,i);
    }
  }

  const int posTableSize = (nloopVolMmk - 1)/2 + 1;
  int posSh[posTableSize];
#pragma unroll
  for (int j=0;j < posTableSize;j++) {
    posSh[j] = 0;
  }
  for (int i=0;i < sizeMmk;i++) {
#pragma unroll
    for (int j=0;j < posTableSize;j++) {
      posSh[j] += (((threadIdx.x + j*2*blockDim.x)/__shfl(Msh.c,i)) % __shfl(Msh.d,i))*__shfl(Msh.ct,i);
      posSh[j] += ((((threadIdx.x + (j*2+1)*blockDim.x)/__shfl(Msh.c,i)) % __shfl(Msh.d,i))*__shfl(Msh.ct,i)) << 16;
    }
  }

//   const int posTableSize = (nloopVolMmk - 1)/2 + 1;
//   int posMmkIn[nloopVolMmk];
//   int posMmkOut[nloopVolMmk];
//   int posSh[posTableSize];
// #pragma unroll
//   for (int j=0;j < nloopVolMmk;j++) {
//     posMmkIn[j] = 0;
//     posMmkOut[j] = 0;
//   }
// #pragma unroll
//   for (int j=0;j < posTableSize;j++) {
//     posSh[j] = 0;
//   }
//   for (int i=0;i < sizeMmk;i++) {
// #pragma unroll
//     for (int j=0;j < nloopVolMmk;j++) {
//       posMmkIn[j] += (((threadIdx.x + j*blockDim.x)/__shfl(Mmk.c_in,i)) % __shfl(Mmk.d_in,i))*__shfl(Mmk.ct_in,i);
//       posMmkOut[j] += (((threadIdx.x + j*blockDim.x)/__shfl(Mmk.c_out,i)) % __shfl(Mmk.d_out,i))*__shfl(Mmk.ct_out,i);
//     }
// #pragma unroll
//     for (int j=0;j < posTableSize;j++) {
//       posSh[j] += (((threadIdx.x + j*2*blockDim.x)/__shfl(Msh.c,i)) % __shfl(Msh.d,i))*__shfl(Msh.ct,i);
//       posSh[j] += ((((threadIdx.x + (j*2+1)*blockDim.x)/__shfl(Msh.c,i)) % __shfl(Msh.d,i))*__shfl(Msh.ct,i)) << 16;
//     }
//   }

#else
  int posMmkIn[nloopVolMmk];
  int posMmkOut[nloopVolMmk];
  int posSh[nloopVolMmk];
#pragma unroll
  for (int j=0;j < nloopVolMmk;j++) {
    posMmkIn[j] = 0;
    posMmkOut[j] = 0;
    posSh[j] = 0;
  }
  for (int i=0;i < sizeMmk;i++) {
#pragma unroll
    for (int j=0;j < nloopVolMmk;j++) {
      posMmkIn[j]  += (((threadIdx.x + j*blockDim.x)/__shfl(Mmk.c_in,i)) % __shfl(Mmk.d_in,i))*__shfl(Mmk.ct_in,i);
      posMmkOut[j] += (((threadIdx.x + j*blockDim.x)/__shfl(Mmk.c_out,i)) % __shfl(Mmk.d_out,i))*__shfl(Mmk.ct_out,i);
      posSh[j] += (((threadIdx.x + j*blockDim.x)/__shfl(Msh.c,i)) % __shfl(Msh.d,i))*__shfl(Msh.ct,i);
    }
  }
#endif

  // 6 registers
  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = gl_Mbar[warpLane];
  }

  for (int posMbar=blockIdx.z;posMbar < volMbar;posMbar += blockDim.z*gridDim.z)
  {

    // Read from global memory
    {
      // int posMbarIn = tensorPos(posMbar, sizeMbar, Mbar.c_in, Mbar.d_in, Mbar.ct_in);
      int posMbarIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
#pragma unroll
      for (int i=16;i >= 1;i/=2) {
        posMbarIn += __shfl_xor(posMbarIn, i);
      }

      __syncthreads();

#pragma unroll
      for (int j=0;j < nloopVolMmk;j++) {
        int posMmk = threadIdx.x + j*blockDim.x;
        int posIn = posMbarIn + posMmkIn[j];
        if (posMmk < volMmk) shBuffer[posMmk] = dataIn[posIn];
      }

    }

    // Write to global memory
    {
      // int posMbarOut = tensorPos(posMbar, sizeMbar, Mbar.c_out, Mbar.d_out, Mbar.ct_out);
      int posMbarOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
#pragma unroll
      for (int i=16;i >= 1;i/=2) {
        posMbarOut += __shfl_xor(posMbarOut, i);
      }

      __syncthreads();

#if 0
#pragma unroll
      for (int j=0;j < posTableSize;j++) {
        int posMmk = threadIdx.x + j*2*blockDim.x;
        int posOut = posMbarOut + posMmkOut[j*2];
        if (posMmk < volMmk) dataOut[posOut] = shBuffer[posSh[j] & 65535];
        if (j*2+1 < nloopVolMmk) {
          posMmk += blockDim.x;
          posOut = posMbarOut + posMmkOut[j*2+1];
          if (posMmk < volMmk) dataOut[posOut] = shBuffer[posSh[j] >> 16];
        }
      }
// #pragma unroll
//       for (int j=0;j < nloopVolMmk;j++) {
//         int posMmk = threadIdx.x + j*blockDim.x;
//         int posOut = posMbarOut + posMmkOut[j];
//         if ((j % 2) == 0) {
//           if (posMmk < volMmk) dataOut[posOut] = shBuffer[posSh[j/2].val[0]];
//         } else {
//           if (posMmk < volMmk) dataOut[posOut] = shBuffer[posSh[j/2].val[1]];
//         }
//       }
#else
#pragma unroll
      for (int j=0;j < nloopVolMmk;j++) {
        int posMmk = threadIdx.x + j*blockDim.x;
        int posOut = posMbarOut + posMmkOut[j];
        if (posMmk < volMmk) dataOut[posOut] = shBuffer[posSh[j]];
      }
#endif

    }

  }
  
}

#if 0
//
// General transpose. Thread block loads plan.volMmk number of elements
//
// numthread.x = warpSize
// numthread.y = volMmk/numthread.x
// numblock(plan.volMbar);
//
#define USE_LOOP
#define ILP_UNROLL 4
template <typename T>
__global__ void transposeTensorKernelArg_general(
  const int volMm, const int volMk, const int volMmk, const int volMbar,
  const int sizeMmk, const int sizeMbar,
  const TensorConvInOut* __restrict__ gl_Mmk,
  const TensorConvInOut* __restrict__ gl_Mbar,
  const TensorConv* __restrict__ gl_Msh,
  const T* __restrict__ dataIn, T* __restrict__ dataOut) {

  // Shared memory. volMmk elements
  extern __shared__ T shBuffer[];

  const int warpLane = threadIdx.x & (warpSize - 1);

  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = gl_Mbar[warpLane];
  }
  TensorConvInOut* shMmk = (TensorConvInOut *)&shBuffer[volMmk];
  if (warpLane < sizeMmk) {
    shMmk[warpLane] = gl_Mmk[warpLane];
  }
  // TensorConvInOut Mmk;
  // Mmk.c_in = 1;
  // Mmk.d_in = 1;
  // Mmk.c_out = 1;
  // Mmk.d_out = 1;
  // if (warpLane < sizeMmk) {
  //   Mmk = gl_Mmk[warpLane];
  // }
  // TensorConv* shMsh = (TensorConv *)&shMmk[sizeMmk];
  // if (warpLane < sizeMmk) {
  //   shMsh[warpLane] = gl_Msh[warpLane];
  // }
  TensorConv Msh;
  Msh.c = 1;
  Msh.d = 1;
  if (warpLane < sizeMmk) {
    Msh = gl_Msh[warpLane];
  }

  for (int posMbar=blockIdx.z;posMbar < volMbar;posMbar += blockDim.z*gridDim.z)
  {

    // Read from global memory
    {
      // int posMbarIn = tensorPos(posMbar, sizeMbar, Mbar.c_in, Mbar.d_in, Mbar.ct_in);
      int posMbarIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
#pragma unroll
      for (int i=16;i >= 1;i/=2) {
        posMbarIn += __shfl_xor(posMbarIn, i);
      }

      int posMmkIn[ILP_UNROLL];
#pragma unroll
      for (int j=0;j < ILP_UNROLL;j++) {
        posMmkIn[j] = 0;
      }
      for (int i=0;i < sizeMmk;i++) {
        int c_in = shMmk[i].c_in;
        int d_in = shMmk[i].d_in;
        int ct_in = shMmk[i].ct_in;
#pragma unroll
        for (int j=0;j < ILP_UNROLL;j++) {
          posMmkIn[j] += (((threadIdx.x + j*blockDim.x)/c_in) % d_in)*ct_in;
        }
      }

      __syncthreads();

      int posMmk0 = 0;
      for (;posMmk0 < volMmk-blockDim.x*ILP_UNROLL;posMmk0 += blockDim.x*ILP_UNROLL) {
#pragma unroll
        for (int j=0;j < ILP_UNROLL;j++) {
          shBuffer[posMmk0 + threadIdx.x + j*blockDim.x] = dataIn[posMbarIn + posMmkIn[j]];
        }
      }

      for (;posMmk0 < volMmk;posMmk0 += blockDim.x) {
        int posMmk = posMmk0 + threadIdx.x;
        int posMmkIn = 0;
        for (int i=0;i < sizeMmk;i++) {
          posMmkIn += ((posMmk/shMmk[i].c_in) % shMmk[i].d_in)*shMmk[i].ct_in;
        }
        if (posMmk < volMmk) shBuffer[posMmk] = dataIn[posMbarIn + posMmkIn];
      }

/*
      __syncthreads();

#ifdef USE_LOOP
      for (int posMmk0=0;posMmk0 < volMmk;posMmk0 += blockDim.x) {
        int posMmk = posMmk0 + threadIdx.x;
        // int posMmkIn = tensorPosLoop(posMmk, sizeMmk, Mmk.c_in, Mmk.d_in, Mmk.ct_in);
        int posMmkIn = 0;
        for (int i=0;i < sizeMmk;i++) {
          posMmkIn += ((posMmk/__shfl(Mmk.c_in,i)) % __shfl(Mmk.d_in,i))*__shfl(Mmk.ct_in,i);
          // posMmkIn += ((posMmk/shMmk[i].c_in) % shMmk[i].d_in)*shMmk[i].ct_in;
        }
        int posIn    = posMbarIn + posMmkIn;
        if (posMmk < volMmk) shBuffer[posMmk] = dataIn[posIn];
      }
#else
      for (int posMmk0=threadIdx.y*volMm;posMmk0 < volMmk;posMmk0 += blockDim.y*volMm) {
        for (int posMm0=0;posMm0 < volMm;posMm0 += blockDim.x) {
          int posMm    = posMm0 + threadIdx.x;
          int posMmkIn = tensorPos(posMmk0, sizeMmk, Mmk.c_in, Mmk.d_in, Mmk.ct_in);
          int posIn    = posMbarIn + posMmkIn + posMm;
          int posMmk   = posMmk0 + posMm;
          if (posMm < volMm) shBuffer[posMmk] = dataIn[posIn];
        }
      }
#endif
*/

    }

    // Write to global memory
    {
      // int posMbarOut = tensorPos(posMbar, sizeMbar, Mbar.c_out, Mbar.d_out, Mbar.ct_out);
      int posMbarOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
#pragma unroll
      for (int i=16;i >= 1;i/=2) {
        posMbarOut += __shfl_xor(posMbarOut, i);
      }

      int posMmkOut[ILP_UNROLL];
      int posSh[ILP_UNROLL];
#pragma unroll
      for (int j=0;j < ILP_UNROLL;j++) {
        posMmkOut[j] = 0;
        posSh[j] = 0;
      }
      for (int i=0;i < sizeMmk;i++) {
        int c_out = shMmk[i].c_out;
        int d_out = shMmk[i].d_out;
        int ct_out = shMmk[i].ct_out;
#pragma unroll
        for (int j=0;j < ILP_UNROLL;j++) {
          posMmkOut[j] += (((threadIdx.x + j*blockDim.x)/c_out) % d_out)*ct_out;
        }
        int c = __shfl(Msh.c,i);
        int d = __shfl(Msh.d,i);
        int ct = __shfl(Msh.ct,i);
#pragma unroll
        for (int j=0;j < ILP_UNROLL;j++) {
          posSh[j] += (((threadIdx.x + j*blockDim.x)/c) % d)*ct;
        }
      }

      __syncthreads();

      int posMmk0 = 0;
      for (;posMmk0 < volMmk-blockDim.x*ILP_UNROLL;posMmk0 += blockDim.x*ILP_UNROLL) {
#pragma unroll
        for (int j=0;j < ILP_UNROLL;j++) {
          dataOut[posMbarOut + posMmkOut[j]] = shBuffer[posSh[j]];
        }
      }

      for (;posMmk0 < volMmk;posMmk0 += blockDim.x) {
        int posMmk = posMmk0 + threadIdx.x;
        int posMmkOut = 0;
        int posSh = 0;
        for (int i=0;i < sizeMmk;i++) {
          posMmkOut += ((posMmk/shMmk[i].c_out) % shMmk[i].d_out)*shMmk[i].ct_out;
          posSh += ((posMmk/__shfl(Msh.c,i)) % __shfl(Msh.d,i))*__shfl(Msh.ct,i);
        }
        int posOut = posMbarOut + posMmkOut;
        if (posMmk < volMmk) dataOut[posOut] = shBuffer[posSh];
      }

/*
#ifdef USE_LOOP
      for (int posMmk0=0;posMmk0 < volMmk;posMmk0 += blockDim.x) {
        int posMmk = posMmk0 + threadIdx.x;
        // int posOut = posMbarOut + tensorPosLoop(posMmk, sizeMmk, Mmk.c_out, Mmk.d_out, Mmk.ct_out);
        // int posSh  = tensorPosLoop(posMmk, sizeMmk, Msh.c_out, Msh.d_out, Msh.ct_out);

        int posMmkOut = 0;
        for (int i=0;i < sizeMmk;i++) {
          // posMmkOut += ((posMmk/__shfl(Mmk.c_out,i)) % __shfl(Mmk.d_out,i))*__shfl(Mmk.ct_out,i);
          posMmkOut += ((posMmk/shMmk[i].c_out) % shMmk[i].d_out)*shMmk[i].ct_out;
        }
        int posOut = posMbarOut + posMmkOut;
        int posSh = 0;
        for (int i=0;i < sizeMmk;i++) {
          posSh += ((posMmk/__shfl(Msh.c,i)) % __shfl(Msh.d,i))*__shfl(Msh.ct,i);
          // posSh += ((posMmk/shMsh[i].c) % shMsh[i].d)*shMsh[i].ct;
        }

        // int posMinorIn;
        // int posSh;
        // tensorPosLoop2(posMmk, sizeMmk, Mmk.c_out, Mmk.d_out, Mmk.ct_out, Msh.c_out, Msh.d_out, Msh.ct_out,
        //   posMinorIn, posSh);
        // int posOut = posMbarOut + posMinorIn;
        if (posMmk < volMmk) dataOut[posOut] = shBuffer[posSh];
      }
#else
      for (int posMmk0=threadIdx.y*volMk;posMmk0 < volMmk;posMmk0 += blockDim.y*volMk) {
        for (int posMk0=0;posMk0 < volMk;posMk0 += blockDim.x) {
          int posMk  = posMk0 + threadIdx.x;
          int posMmk = posMmk0 + posMk;
          int posOut = posMbarOut + tensorPosLoop(posMmk, sizeMmk, Mmk.c_out, Mmk.d_out, Mmk.ct_out);
          int posSh  = tensorPosLoop(posMmk, sizeMmk, Msh.c_out, Msh.d_out, Msh.ct_out);
          if (posMk < volMk) dataOut[posOut] = shBuffer[posSh];
        }
      }
#endif
*/

    }

  }
  
}
#endif

#if 0
//
// Transpose when Mm and Mk are the same
// (transpose only done in Mbar -volume)
//
//  dim3 numthread(nx, 1, 1);
//  dim3 numblock((plan.volMmk-1)/nx+1, plan.volMbar);
//
// Number of elements each thread loads and stores
const int ELEMENTS_PER_THREAD = 4;
template <typename T>
__global__ void transposeTensorKernelArg_leadVolSame(
  const int volMmk, const int volMbar, const int sizeMbar,
  const TensorConvInOut* __restrict__ gl_Mbar,
  const T* __restrict__ dataIn, T* __restrict__ dataOut) {

  const int warpLane = threadIdx.x & (warpSize - 1);
  TensorConvInOut Mbar;
  if (warpLane < sizeMbar) {
    Mbar = gl_Mbar[warpLane];
  }

  const int x = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

  for (int posMbar=blockIdx.y;posMbar < volMbar;posMbar += blockDim.y*gridDim.y)
  {
    int posMbarIn = tensorPos(warpLane, posMbar, sizeMbar, Mbar.c_in, Mbar.d_in, Mbar.ct_in);
    int posIn = posMbarIn + x;
    T val[ELEMENTS_PER_THREAD];
#pragma unroll
    for (int j=0;j < ELEMENTS_PER_THREAD;j++) {
      int jb = j*blockDim.x;
      if (x + jb < volMmk) val[j] = dataIn[posIn + jb];
    }

    int posMbarOut = tensorPos(warpLane, posMbar, sizeMbar, Mbar.c_out, Mbar.d_out, Mbar.ct_out);
    int posOut = posMbarOut + x;
#pragma unroll
    for (int j=0;j < ELEMENTS_PER_THREAD;j++) {
      int jb = j*blockDim.x;
      if (x + jb < volMmk) dataOut[posOut + jb] = val[j];
    }
  }
  
}
#endif

#if 1
//
// Transpose when the lead dimension is the same, e.g. (1, 2, 3) -> (1, 3, 2)
//
//  dim3 numthread(TILEDIM, TILEROWS, 1);
//  dim3 numblock((plan.volMm-1)/TILEDIM+1, (plan.volMk-1)/TILEDIM+1, plan.volMbar);
//
template <typename T>
__global__ void transposeTensorKernelArg_leadDimSame(
  const int volMbar, const int sizeMbar,
  const int cuDimMk, const int cuDimMm,
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

  int* dimMmkIn  = &transposeArg[0];

  const int x = blockIdx.x * TILEDIM + threadIdx.x;
  const int y = blockIdx.y * TILEDIM + threadIdx.y;

  for (int blockz=blockIdx.z;blockz < volMbar;blockz += blockDim.z*gridDim.z)
  {

    // Variables where values are stored
    T val[TILEDIM/TILEROWS];

    // Read global memory
    {
      int pos0 = tensorPos(blockz, sizeMbar, Mbar.c_in, Mbar.d_in, Mbar.ct_in);
      pos0 += x + y*cuDimMk;

#pragma unroll
      for (int j=0;j < TILEDIM;j += TILEROWS) {
        int pos  = pos0  + j*cuDimMk;
        if ((x < dimMmkIn[0]) && (y + j < dimMmkIn[1])) {
          val[j/TILEROWS] = dataIn[pos];
        }
      }
    }

    // Write global memory
    {
      int pos0 = tensorPos(blockz, sizeMbar, Mbar.c_out, Mbar.d_out, Mbar.ct_out);
      pos0 += x + y*cuDimMm;

#pragma unroll
      for (int j=0;j < TILEDIM;j += TILEROWS) {
        int pos = pos0 + j*cuDimMm;
        if ((x < dimMmkIn[0]) && (y + j < dimMmkIn[1])) {
          dataOut[pos] = val[j/TILEROWS];
        }
      }
    }

  }
  
}
#endif

class TensorTransposePlan {
public:
  // Device for which this plan was made
  int deviceID;

  // Rank of the tensor
  const int rank;

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
  int2 writeVol;

  // Parameters for TiledLeadVolSame
  int vol0, vol1;

  // sizeMbar
  TensorConvInOut* Mbar;

  // sizeMmk
  TensorConvInOut* Mmk;

  // sizeMmk
  TensorConv* Msh;

  static int getDataSize(const int rank) {
    return (2 + 4*rank + 6*(rank - 2));
  }

  TensorTransposePlan(const int rank, const int* dim, const int* permutation, const size_t sizeofType) : rank(rank) {

    if (rank <= 1) {
      printf("TensorTransposePlan::TensorTransposePlan(), tensor rank must be > 1\n");
      exit(1);
    }

    // Read device properties to determine how much shared memory we can afford to use
    cudaCheck(cudaGetDevice(&deviceID));
    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, deviceID));

    // Absolute maximum number of elements shared memory can hold
    const int maxVolMmk = (prop.sharedMemPerMultiprocessor/sizeofType);
    // Try to use maximum of 1/3 of shared memory
    const int useVolMmk = (prop.sharedMemPerMultiprocessor/sizeofType)/3;
    printf("maxVolMmk %d useVolMmk %d sharedMemPerMultiprocessor %d\n",
      maxVolMmk, useVolMmk, prop.sharedMemPerMultiprocessor);

    bool* isMm = new bool[rank];
    bool* isMk = new bool[rank];
    for (int i=0;i < rank;i++) {
      isMm[i] = false;
      isMk[i] = false;
    }

    // Minimum allowed dimension that is dealt with
    // using the tiled algorithm
    const int MIN_TILED_DIM = TILEDIM;

    // Setup Mm
    {
      int r = 0;
      sizeMm = 0;
      volMm = 1;
      while (r < rank && volMm < MIN_TILED_DIM) {
        isMm[r] = true;
        volMm *= dim[r];
        sizeMm++;
        r++;
      }
    }

    // Setup Mk
    {
      int r = 0;
      sizeMk = 0;
      volMk = 1;
      while (r < rank && volMk < MIN_TILED_DIM) {
        int pr = permutation[r];
        isMk[pr] = true;
        volMk *= dim[pr];
        sizeMk++;
        r++;
      }
    }

    // Setup Mmk
    setupMmk(isMm, isMk, dim);

    // Setup method
    method = Unknown;
    if (sizeMm > 1 || sizeMk > 1) {
      // General case: Mm or Mk are > 1
      bool Mm_Mk_same = (sizeMm == sizeMk);
      if (Mm_Mk_same) {
        for (int i=0;i < sizeMm;i++) {
          if (permutation[i] != i) {
            Mm_Mk_same = false;
            break;
          }
        }
      }

      // APH DEBUG: REMOVE THIS AFTER TiledLeadVolSame WORKS
      Mm_Mk_same = false;

      if (Mm_Mk_same) {
        method = TiledLeadVolSame;
      } else {
        method = General;
        while (volMmk > useVolMmk) {
          int r = sizeMm - 1;
          int pr = permutation[sizeMk - 1];
          if ((dim[r] <= dim[pr] || sizeMm == 1) && sizeMk > 1) {
            // Remove one from Mk
            isMk[pr] = false;
          } else if (sizeMm > 1) {
            // Remove one from Mm
            isMm[r] = false;
          } else if (volMmk > maxVolMmk) {
            // Unable to remove and exceeds shared memory size
            printf("TensorTransposePlan::TensorTransposePlan(), unable to reduce shared memory usage\n");
            exit(1);
          } else {
            // Unable to remove but does not exceed shared memory size => bail out
            break;
          }
          setupMm(isMm, dim);
          setupMk(isMk, dim);
          setupMmk(isMm, isMk, dim);
        }
        if (volMmk > maxVolMmk) {
          printf("TensorTransposePlan::TensorTransposePlan(), volMmk exceeds shared memory size\n");
          exit(1);
        }
      }
    } else {
      // Tiled case: Mm and Mk are size 1

      // Check if Mm and Mk are the same
      if (permutation[0] == 0) {
        method = TiledLeadVolSame;
        // isMm[1] = true;
        // isMk[1] = true;
        // setupMm(isMm, dim);
        // setupMk(isMk, dim);

        // Choose next rank as Mk
        // isMk[0] = false;
        // isMk[1] = true;
        // volMk = dim[1];
      } else {
        method = TiledSingleRank;
      }
    }

    if (method == Unknown) {
      printf("TensorTransposePlan::TensorTransposePlan(), method not determined\n");
      exit(1);
    }

/*
    // Check for overlap between Mm and Mk
    {
      int firstOverlapRank = -1;
      int numOverlap = 0;
      for (int i=0;i < rank;i++) {
        if (isMm[i] && isMk[i]) {
          if (numOverlap == 0) firstOverlapRank = i;
          numOverlap++;
        }
      }
      // Single overlap on the first rank and dimension large enough => 
      // Use tiled method with non-transposing leading dimension
      if (rank > 1 && numOverlap == 1 && firstOverlapRank == 0 &&
          dim[0] >= MIN_TILED_DIM && dim[1] >= MIN_TILED_DIM) {

        // Re-do Mk for this special case
        leadDimSame = true;
        for (int i=0;i < rank;i++) isMk[i] = false;
        isMk[1] = true;
        volMk = dim[1];
        sizeMk = 1;

      } else if (numOverlap >= 1) {
        general = true;
      }
    }
*/
    printf("method ");
    switch(method) {
      case General:
      printf("General\n");
      break;
      case TiledSingleRank:
      printf("TiledSingleRank\n");
      break;
      case TiledLeadVolSame:
      printf("TiledLeadVolSame\n");
      break;
    };

    // Setup Mmk
    setupMmk(isMm, isMk, dim);

    // Setup Mbar
    setupMbar(isMm, isMk, dim);

    // Build cI
    int* I = new int[rank];
    for (int i=0;i < rank;i++) {
      I[i] = i;
    }
    TensorC cI(rank, rank, I, dim);
    delete [] I;

    // Build cO
    TensorC cO(rank, rank, permutation, dim);

    if (method == TiledSingleRank) {
      // cuDimMk = cI.get(MkI[0]);
      // cuDimMm = cO.get(MmI[0]);
      cuDimMk = cI.get(permutation[0]);
      cuDimMm = cO.get(0);
    } else if (method == TiledLeadVolSame) {
      vol0 = volMm;
      // Mm and Mk are the same => try including one more rank into Mmk from input
      if (sizeMmk < rank) {
        isMm[sizeMmk] = true;
        isMk[sizeMmk] = true;
        cuDimMk = cI.get(sizeMmk);
        cuDimMm = cO.get(sizeMmk);
        vol1 = dim[sizeMmk];
        setupMm(isMm, dim);
        setupMk(isMk, dim);
        setupMmk(isMm, isMk, dim);
        setupMbar(isMm, isMk, dim);
      } else {
        cuDimMk = 1;
        cuDimMm = 1;
        vol1 = 1;
      }
    }

    // Build MmI and MkI
    int* MmI = new int[sizeMm];
    int* MkI = new int[sizeMk];
    {
      int iMm = 0;
      int iMk = 0;
      for (int i=0;i < rank;i++) {
        if (isMm[i]) {
          MmI[iMm++] = i;
        }
        if (isMk[i]) {
          MkI[iMk++] = i;
        }
      }
    }

    // if (method == TiledSingleRank) {
    // } else if (method == TiledLeadVolSame) {
    //   cuDimMk = cI.get(MmkI[MmkSplit]);
    //   cuDimMm = cO.get(MkI[0]);
    // }

    // if (method != General) {
    //   cuDimMk = cI.get(MkI[0]);
    //   if (method == TiledLeadVolSame) {
    //     cuDimMm = cO.get(MkI[0]);
    //   } else {
    //     cuDimMm = cO.get(MmI[0]);
    //   }
    // }

    TensorConvInOut* hostMbar = NULL;
    if (sizeMbar > 0) {
      // Build MbarI = {s_1, ...., s_h}, indices in input order
      int* MbarI = new int[sizeMbar];
      int j = 0;
      for (int i=0;i < rank;i++) {
        if (!(isMm[i] || isMk[i])) {
          MbarI[j] = i;
          j++;
        }
      }
      TensorC cMbarI(rank, sizeMbar, MbarI, dim);

      // Build MbarO = {s_l1, ...., s_lh}, indices in output (permuted) order
      int* MbarO = new int[sizeMbar];
      j = 0;
      for (int i=0;i < rank;i++) {
        int pi = permutation[i];
        if (!(isMm[pi] || isMk[pi])) {
          MbarO[j] = pi;
          j++;
        }
      }

      hostMbar = new TensorConvInOut[sizeMbar];
      for (int i=0;i < sizeMbar;i++) {
        int si = MbarI[i];
        hostMbar[i].c_in  = cMbarI.get(si);
        hostMbar[i].d_in  = dim[si];
        hostMbar[i].ct_in = cI.get(si);
        int sli = MbarO[i];
        hostMbar[i].c_out  = cMbarI.get(sli);
        hostMbar[i].d_out  = dim[sli];
        hostMbar[i].ct_out = cO.get(sli);
      }

#if 1
      printf("MbarI");
      for (int i=0;i < sizeMbar;i++) printf(" %d", MbarI[i]+1);
      printf("\n");

      printf("MbarO");
      for (int i=0;i < sizeMbar;i++) printf(" %d", MbarO[i]+1);
      printf("\n");
#endif

      delete [] MbarI;
      delete [] MbarO;
    }

    TensorConvInOut* hostMmk = NULL;
    TensorConv* hostMsh = NULL;
    if (method == General) {
      // Build MmkI = {q_1, ..., q_a}
      int* MmkI = new int[sizeMmk];
      int j = 0;
      for (int i=0;i < rank;i++) {
        if (isMm[i] || isMk[i]) {
          MmkI[j] = i;
          j++;
        }
      }
      TensorC cMmkI(rank, sizeMmk, MmkI, dim);
      // Build MmkO = {q_t1, ..., q_ta}
      int* MmkO = new int[sizeMmk];
      j = 0;
      for (int i=0;i < rank;i++) {
        int pi = permutation[i];
        if (isMm[pi] || isMk[pi]) {
          MmkO[j] = pi;
          j++;
        }
      }
      TensorC cMmkO(rank, sizeMmk, MmkO, dim);

      hostMmk = new TensorConvInOut[sizeMmk];
      for (int i=0;i < sizeMmk;i++) {
        // Minor reading position
        int qi = MmkI[i];
        hostMmk[i].c_in  = cMmkI.get(qi);
        hostMmk[i].d_in  = dim[qi];
        hostMmk[i].ct_in = cI.get(qi);
        // Shared memory reading position
        // int qti = MmkO[i];
        // hostMmk[i].c_out  = cMmkO.get(qti);
        // hostMmk[i].d_out  = dim[qti];
        // hostMmk[i].ct_out = cMmkI.get(qti);
        // Minor writing position
        int qti = MmkO[i];
        hostMmk[i].c_out  = cMmkO.get(qti);
        hostMmk[i].d_out  = dim[qti];
        hostMmk[i].ct_out = cO.get(qti);
      }

      hostMsh = new TensorConv[sizeMmk];
      for (int i=0;i < sizeMmk;i++) {
        // Shared memory reading position
        int qti = MmkO[i];
        hostMsh[i].c  = cMmkO.get(qti);
        hostMsh[i].d  = dim[qti];
        hostMsh[i].ct = cMmkI.get(qti);
      }
/*
      {
        int i = 0;
        // MkO
        for (;i < sizeMk;i++) {
          int qti = MmkO[i];
          hostMmk[i].c_out  = cMmkO.get(qti);
          hostMmk[i].d_out  = dim[qti];
          hostMmk[i].ct_out = cMmkI.get(qti);
        }
        // Rm0
        for (;i < sizeMmk;i++) {
          int qti = MmkO[i];
          hostMmk[i].c_out  = cMmkO.get(qti);
          hostMmk[i].d_out  = dim[qti];
          hostMmk[i].ct_out = cO.get(qti);
        }
      }
*/

      delete [] MmkI;
      delete [] MmkO;
    }

    if (method != General) {
      int* tmp_dimMmkIn = new int[sizeMmk];
      tmp_dimMmkIn[0] = dim[MmI[0]];
      tmp_dimMmkIn[1] = dim[MkI[0]];

      readVol.x = 1;
      readVol.y = 1;
      for (int i=0;i < sizeMm;i++) {
        readVol.x *= dim[MmI[i]];
      }
      for (int i=0;i < sizeMk;i++) {
        readVol.y *= dim[MkI[i]];
      }

      int* tmp_dimMmkOut = new int[sizeMmk];
      int j = 0;
      for (int i=0;i < rank;i++) {
        int pi = permutation[i];
        if (isMm[pi] || isMk[pi]) {
          tmp_dimMmkOut[j] = dim[pi];
          j++;
        }
      }

      writeVol.x = 1;
      writeVol.y = 1;
      for (int i=0;i < rank;i++) {
        int pi = permutation[i];
        if (isMm[pi]) {
          writeVol.x *= dim[pi];
        }
        if (isMk[pi]) {
          writeVol.y *= dim[pi];
        }
      }

      int* h_transposeArg = new int[transposeArgSize];
      int iarg = 0;
      for (int j=0;j < sizeMmk;j++) h_transposeArg[iarg++] = tmp_dimMmkIn[j];
      for (int j=0;j < sizeMmk;j++) h_transposeArg[iarg++] = tmp_dimMmkOut[j];

      cudaCheck(cudaMemcpyToSymbol(transposeArg, h_transposeArg,
        transposeArgSize*sizeof(int), 0, cudaMemcpyHostToDevice));

      delete [] tmp_dimMmkIn;
      delete [] tmp_dimMmkOut;
      delete [] h_transposeArg;
    }

#if 1
    printf("MmI");
    for (int i = 0; i < sizeMm; ++i) printf(" %d", MmI[i]+1);
    printf(" volMm %d\n", volMm);

    printf("MkI");
    for (int i = 0; i < sizeMk; ++i) printf(" %d", MkI[i]+1);
    printf(" volMk %d\n", volMk);

    printf("Mmk");
    for (int i = 0; i < rank; ++i) if (isMm[i] || isMk[i]) printf(" %d", i+1);
    printf(" volMmk %d\n", volMmk);

    if (sizeMbar > 0) {
      printf("Mbar");
      for (int i = 0; i < rank; ++i) if (!(isMm[i] || isMk[i])) printf(" %d", i+1);
      printf(" volMbar %d\n", volMbar);
    }

    if (sizeMbar > 0) {
      printf("MbarIn\n");
      for (int i=0;i < sizeMbar;i++) printf("%d %d %d\n",
        hostMbar[i].c_in, hostMbar[i].d_in, hostMbar[i].ct_in);

      printf("MbarOut\n");
      for (int i=0;i < sizeMbar;i++) printf("%d %d %d\n",
        hostMbar[i].c_out, hostMbar[i].d_out, hostMbar[i].ct_out);
    }

    if (method == General) {
      printf("MmkIn\n");
      for (int i=0;i < sizeMmk;i++) printf("%d %d %d\n",
        hostMmk[i].c_in, hostMmk[i].d_in, hostMmk[i].ct_in);

      printf("MmkOut\n");
      for (int i=0;i < sizeMmk;i++) printf("%d %d %d\n",
        hostMmk[i].c_out, hostMmk[i].d_out, hostMmk[i].ct_out);

      printf("Msh\n");
      for (int i=0;i < sizeMmk;i++) printf("%d %d %d\n",
        hostMsh[i].c, hostMsh[i].d, hostMsh[i].ct);
    }

    if (method != General) printf("cuDimMk %d cuDimMm %d\n", cuDimMk, cuDimMm);

    printf("readVol %d %d writeVol %d %d\n", readVol.x, readVol.y, writeVol.x, writeVol.y);
#endif

    delete [] isMm;
    delete [] isMk;

    delete [] MmI;
    delete [] MkI;

    if (sizeMbar > 0) {
      allocate_device<TensorConvInOut>(&Mbar, sizeMbar);
      copy_HtoD_sync<TensorConvInOut>(hostMbar, Mbar, sizeMbar);
      delete [] hostMbar;
    }

    if (method == General) {
      allocate_device<TensorConvInOut>(&Mmk, sizeMmk);
      copy_HtoD_sync<TensorConvInOut>(hostMmk, Mmk, sizeMmk);
      delete [] hostMmk;
      allocate_device<TensorConv>(&Msh, sizeMmk);
      copy_HtoD_sync<TensorConv>(hostMsh, Msh, sizeMmk);
      delete [] hostMsh;
    }

    cudaCheck(cudaDeviceSynchronize());
  }

  ~TensorTransposePlan() {
    if (sizeMbar > 0) deallocate_device<TensorConvInOut>(&Mbar);
    if (method == General) deallocate_device<TensorConvInOut>(&Mmk);
    if (method == General) deallocate_device<TensorConv>(&Msh);
  }

private:

  void setupMm(const bool* isMm, const int* dim) {
    sizeMm = 0;
    volMm = 1;
    for (int i=0;i < rank;i++) {
      if (isMm[i]) {
        volMm *= dim[i];
        sizeMm++;
      }
    }
  }

  void setupMk(const bool* isMk, const int* dim) {
    sizeMk = 0;
    volMk = 1;
    for (int i=0;i < rank;i++) {
      if (isMk[i]) {
        volMk *= dim[i];
        sizeMk++;
      }
    }
  }

  void setupMmk(const bool* isMm, const bool* isMk, const int* dim) {
    sizeMmk = 0;
    volMmk = 1;
    for (int i=0;i < rank;i++) {
      if (isMm[i] || isMk[i]) {
        volMmk *= dim[i];
        sizeMmk++;
      }
    }
  }

  void setupMbar(const bool* isMm, const bool* isMk, const int* dim) {
    sizeMbar = 0;
    volMbar = 1;
    for (int i=0;i < rank;i++) {
      if (!(isMm[i] || isMk[i])) {
        volMbar *= dim[i];
        sizeMbar++;
      }
    }
  }

};

template <typename T>
void transposeTensorArg(TensorTransposePlan& plan,
  const T* dataIn, T* dataOut, cudaStream_t stream) {

  int deviceID;
  cudaCheck(cudaGetDevice(&deviceID));
  if (deviceID != plan.deviceID) {
    printf("transposeTensorArg plan device and current device different\n");
    exit(1);
  }

  switch(plan.method) {

    case TensorTransposePlan::General:
    {
      dim3 numthread(256, 1);
      dim3 numblock(1, 1, plan.volMbar);
      numblock.z = min(256, plan.volMbar);
      numblock.z = min(65535, numblock.z);
      int shmemsize = plan.volMmk*sizeof(T);
      int nloopVolMmk = (plan.volMmk - 1)/numthread.x + 1;

      printf("numthread %d %d %d numblock %d %d %d shmemsize %d nloopVolMmk %d\n",
        numthread.x, numthread.y, numthread.z,
        numblock.x, numblock.y, numblock.z, shmemsize, nloopVolMmk);

      switch(nloopVolMmk) {
        // case 1:
        // transposeTensorKernelArg_general<T, 1> <<< numblock, numthread, shmemsize, stream >>>
        // (plan.volMm, plan.volMk, plan.volMmk, plan.volMbar,
        //   plan.sizeMmk, plan.sizeMbar,
        //   plan.Mmk, plan.Mbar, plan.Msh, dataIn, dataOut);
        // break;
        // case 2:
        // transposeTensorKernelArg_general<T, 2> <<< numblock, numthread, shmemsize, stream >>>
        // (plan.volMm, plan.volMk, plan.volMmk, plan.volMbar,
        //   plan.sizeMmk, plan.sizeMbar,
        //   plan.Mmk, plan.Mbar, plan.Msh, dataIn, dataOut);
        // break;
        // case 3:
        // transposeTensorKernelArg_general<T, 3> <<< numblock, numthread, shmemsize, stream >>>
        // (plan.volMm, plan.volMk, plan.volMmk, plan.volMbar,
        //   plan.sizeMmk, plan.sizeMbar,
        //   plan.Mmk, plan.Mbar, plan.Msh, dataIn, dataOut);
        // break;
        case 6:
        transposeTensorKernelArg_general<T, 6> <<< numblock, numthread, shmemsize, stream >>>
        (plan.volMm, plan.volMk, plan.volMmk, plan.volMbar,
          plan.sizeMmk, plan.sizeMbar,
          plan.Mmk, plan.Mbar, plan.Msh, dataIn, dataOut);
        break;
        default:
        printf("transposeTensorArg unsupported nloopVolMmk %d requested\n", nloopVolMmk);
        exit(1);
        break;
      }

#if 0
      dim3 numthread(256, 1);
      dim3 numblock(1, 1, plan.volMbar);
      numblock.z = min(256, plan.volMbar);
      numblock.z = min(65535, numblock.z);
      int shmemsize = plan.volMmk*sizeof(T) + plan.sizeMmk*(sizeof(TensorConvInOut) + 0*sizeof(TensorConv));
      printf("numthread %d %d %d numblock %d %d %d shmemsize %d\n",
        numthread.x, numthread.y, numthread.z,
        numblock.x, numblock.y, numblock.z, shmemsize);
      transposeTensorKernelArg_general<T> <<< numblock, numthread, shmemsize, stream >>>
      (plan.volMm, plan.volMk, plan.volMmk, plan.volMbar,
        plan.sizeMmk, plan.sizeMbar,
        plan.Mmk, plan.Mbar, plan.Msh, dataIn, dataOut);
#endif
    }
    break;

    case TensorTransposePlan::TiledSingleRank:
    {
      dim3 numthread(TILEDIM, TILEROWS, 1);
      dim3 numblock((plan.volMm-1)/TILEDIM+1, (plan.volMk-1)/TILEDIM+1, plan.volMbar);
      numblock.z = min(256, plan.volMbar);
      numblock.z = min(65535, numblock.z);
      printf("numthread %d %d %d numblock %d %d %d\n", numthread.x, numthread.y, numthread.z,
        numblock.x, numblock.y, numblock.z);
      transposeTensorKernelArg<T> <<< numblock, numthread, 0, stream >>>
      (plan.volMbar, plan.sizeMbar, plan.readVol, plan.writeVol, plan.cuDimMk, plan.cuDimMm,
        plan.Mbar, dataIn, dataOut);
    }
    break;

    case TensorTransposePlan::TiledLeadVolSame:
    {
      // dim3 numthread(min(512, ((plan.volMmk-1)/(TILEDIM*ELEMENTS_PER_THREAD)+1)*TILEDIM ), 1, 1);
      // dim3 numblock((plan.volMmk-1)/(numthread.x*ELEMENTS_PER_THREAD)+1, plan.volMbar);
      // numblock.y = min(256, plan.volMbar);
      // numblock.y = min(65535, numblock.y);
      // if (numblock.x > 65535) {
      //   printf("transposeTensorArg too many thread blocks requested\n");
      //   exit(1);
      // }
      // printf("numthread %d %d %d numblock %d %d %d\n", numthread.x, numthread.y, numthread.z,
      //   numblock.x, numblock.y, numblock.z);
      // transposeTensorKernelArg_leadVolSame<T> <<< numblock, numthread, 0, stream >>>
      // (plan.volMmk, plan.volMbar, plan.sizeMbar,
      //   plan.Mbar, dataIn, dataOut);

      dim3 numthread(TILEDIM, TILEROWS, 1);
      dim3 numblock((plan.vol0-1)/TILEDIM+1, (plan.vol1-1)/TILEDIM+1, plan.volMbar);
      numblock.z = min(256, plan.volMbar);
      numblock.z = min(65535, numblock.z);
      printf("numthread %d %d %d numblock %d %d %d\n", numthread.x, numthread.y, numthread.z,
        numblock.x, numblock.y, numblock.z);
      transposeTensorKernelArg_leadDimSame<T> <<< numblock, numthread, 0, stream >>>
      (plan.volMbar, plan.sizeMbar, plan.cuDimMk, plan.cuDimMm,
        plan.Mbar, dataIn, dataOut);
    }
    break;

  }

/*
  if (plan.method == General) {
    dim3 numthread(TILEDIM, 1, 1);
    dim3 numblock(1, 1, plan.volMbar);
    numblock.z = min(65535, numblock.z);
    int shmemsize = plan.volMmk*sizeof(T);
    transposeTensorKernelArg_general<T> <<< numblock, numthread, shmemsize, stream >>>
    (plan.volMm, plan.volMk, plan.volMmk, plan.volMbar,
      plan.sizeMk, plan.sizeRm, plan.sizeMmk, plan.sizeMbar,
      plan.Mmk, plan.Mbar, dataIn, dataOut);

  } else if (plan.method == TiledSingleRank) {
    dim3 numthread(TILEDIM, TILEROWS, 1);
    dim3 numblock((plan.volMm-1)/TILEDIM+1, (plan.volMk-1)/TILEDIM+1, plan.volMbar);
    numblock.z = min(256, plan.volMbar);
    numblock.z = min(65535, numblock.z);
  } else if (plan.method == TiledLeadVolSame)
    if (plan.leadDimSame) {
      transposeTensorKernelArg_leadDimSame<T> <<< numblock, numthread, 0, stream >>>
      (plan.volMbar, plan.sizeMbar, plan.cuDimMk, plan.cuDimMm,
        plan.Mbar, dataIn, dataOut);
    } else {
      // if (plan.subTransp) {
      //   transposeTensorKernelArg_subTransp<T> <<< numblock, numthread, 0, stream >>>
      //   (plan.volMbar, plan.sizeMmk, plan.sizeMbar, plan.readVol, plan.writeVol, plan.cuDimMk, plan.cuDimMm,
      //     plan.Mbar, plan.Mmk, dataIn, dataOut);
      // } else {
        transposeTensorKernelArg<T> <<< numblock, numthread, 0, stream >>>
        (plan.volMbar, plan.sizeMmk, plan.sizeMbar, plan.readVol, plan.writeVol, plan.cuDimMk, plan.cuDimMm,
          plan.Mbar, dataIn, dataOut);
      // }
    }
  }
*/

  cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
//
// Transposes a 3d matrix out-of-place: data_in(x, y, z) -> data_out(y, z, x)
//
template <typename T>
__device__ __forceinline__
void transpose_xyz_yzx_device(
  const int x_in, const int y_in, const int z_in,
  const int x_out, const int y_out,
  const int nx, const int ny, const int nz,
  const int xsize_in, const int ysize_in,
  const int ysize_out, const int zsize_out,
  const T* data_in, T* data_out) {

  // Shared memory
  __shared__ T tile[TILEDIM][TILEDIM+1];

  // Read (x,y) data_in into tile (shared memory)
  for (int j=0;j < TILEDIM;j += TILEROWS)
    if ((x_in < nx) && (y_in + j < ny) && (z_in < nz))
      tile[threadIdx.y + j][threadIdx.x] = data_in[x_in + (y_in + j + z_in*ysize_in)*xsize_in];

  __syncthreads();

  // Write (y,x) tile into data_out
  const int z_out = z_in;
  for (int j=0;j < TILEDIM;j += TILEROWS)
    if ((x_out + j < nx) && (y_out < ny) && (z_out < nz))
      data_out[y_out + (z_out + (x_out+j)*zsize_out)*ysize_out] = tile[threadIdx.x][threadIdx.y + j];
}

template <typename T>
__global__ void transpose_xyz_yzx_kernel(
  const int nx, const int ny, const int nz,
  const int xsize_in, const int ysize_in,
  const int ysize_out, const int zsize_out,
  const T* data_in, T* data_out) {

  int x_in = blockIdx.x * TILEDIM + threadIdx.x;
  int y_in = blockIdx.y * TILEDIM + threadIdx.y;
  int z_in = blockIdx.z           + threadIdx.z;

  int x_out = blockIdx.x * TILEDIM + threadIdx.y;
  int y_out = blockIdx.y * TILEDIM + threadIdx.x;

  transpose_xyz_yzx_device<T>(
    x_in, y_in, z_in,
    x_out, y_out,
    nx, ny, nz,
    xsize_in, ysize_in,
    ysize_out, zsize_out,
    data_in, data_out);

}
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
//
// Transposes a 3d matrix out-of-place: data_in(x, y, z) -> data_out(z, x, y)
//
template <typename T>
__device__ __forceinline__
void transpose_xyz_zxy_device(
  const int x_in, const int y_in, const int z_in,
  const int x_out, const int z_out,
  const int nx, const int ny, const int nz,
  const int xsize_in, const int ysize_in,
  const int zsize_out, const int xsize_out,
  const T* data_in, T* data_out) {

  // Shared memory
  __shared__ T tile[TILEDIM][TILEDIM+1];

  // Read (x,z) data_in into tile (shared memory)
  for (int k=0;k < TILEDIM;k += TILEROWS)
    if ((x_in < nx) && (y_in < ny) && (z_in + k < nz))
      tile[threadIdx.y + k][threadIdx.x] = data_in[x_in + (y_in + (z_in + k)*ysize_in)*xsize_in];

  __syncthreads();

  // Write (z,x) tile into data_out
  const int y_out = y_in;
  for (int k=0;k < TILEDIM;k += TILEROWS)
    if ((x_out + k < nx) && (y_out < ny) && (z_out < nz))
      data_out[z_out + (x_out + k + y_out*xsize_out)*zsize_out] = tile[threadIdx.x][threadIdx.y + k];
}

template <typename T>
__global__ void transpose_xyz_zxy_kernel(
  const int nx, const int ny, const int nz,
  const int xsize_in, const int ysize_in,
  const int zsize_out, const int xsize_out,
  const T* data_in, T* data_out) {

  int x_in = blockIdx.x * TILEDIM + threadIdx.x;
  int y_in = blockIdx.z           + threadIdx.z;
  int z_in = blockIdx.y * TILEDIM + threadIdx.y;

  int x_out = blockIdx.x * TILEDIM + threadIdx.y;
  int z_out = blockIdx.y * TILEDIM + threadIdx.x;

  transpose_xyz_zxy_device<T>(
    x_in, y_in, z_in, x_out, z_out,
    nx, ny, nz,
    xsize_in, ysize_in,
    zsize_out, xsize_out,
    data_in, data_out);

}
// ----------------------------------------------------------------------------

#if 0
// ----------------------------------------------------------------------------
//
// Transposes a 3d matrix out-of-place: data_in(x, y, z) -> data_out(z, y, x)
//
template <typename T>
__device__ __forceinline__
void transpose_xyz_zyx_device(
  const int x_in, const int y_in, const int z_in,
  const int y_out, const int z_out,
  const int nx, const int ny, const int nz,
  const int xsize_in, const int ysize_in,
  const int zsize_out, const int ysize_out,
  const T* data_in, T* data_out) {

  // Shared memory
  __shared__ T tile[TILEDIM][TILEDIM+1];

  // Read (x,z) data_in into tile (shared memory)
  for (int k=0;k < TILEDIM;k += TILEROWS)
    if ((x_in < nx) && (y_in < ny) && (z_in + k < nz))
      tile[threadIdx.y + k][threadIdx.x] = data_in[x_in + (y_in + (z_in + k)*ysize_in)*xsize_in];

  __syncthreads();

  // Write (z,x) tile into data_out
  const int y_out = y_in;
  for (int k=0;k < TILEDIM;k += TILEROWS)
    if ((x_out + k < nx) && (y_out < ny) && (z_out < nz))
      data_out[z_out + (x_out + k + y_out*xsize_out)*zsize_out] = tile[threadIdx.x][threadIdx.y + k];
}

template <typename T>
__global__ void transpose_xyz_zyx_kernel(
  const int nx, const int ny, const int nz,
  const int xsize_in, const int ysize_in,
  const int zsize_out, const int ysize_out,
  const T* data_in, T* data_out) {

  int x_in = blockIdx.x * TILEDIM + threadIdx.x;
  int y_in = blockIdx.z           + threadIdx.z;
  int z_in = blockIdx.y * TILEDIM + threadIdx.y;

  int x_out = blockIdx.x * TILEDIM + threadIdx.y;
  int z_out = blockIdx.y * TILEDIM + threadIdx.x;

  transpose_xyz_zyx_device<T>(
    x_in, y_in, z_in, y_out, z_out,
    nx, ny, nz,
    xsize_in, ysize_in,
    zsize_out, ysize_out,
    data_in, data_out);

}
// ----------------------------------------------------------------------------
#endif

//
// Transpose
//
template <typename T>
void transpose_xyz_yzx(
  const int nx, const int ny, const int nz,
  const int xsize_in, const int ysize_in,
  const int ysize_out, const int zsize_out,
  const T* data_in, T* data_out, cudaStream_t stream) {

  dim3 numthread(TILEDIM, TILEROWS, 1);
  dim3 numblock((nx-1)/TILEDIM+1, (ny-1)/TILEDIM+1, nz);

  transpose_xyz_yzx_kernel<T> <<< numblock, numthread, 0, stream >>>
  (nx, ny, nz, xsize_in, ysize_in,
    ysize_out, zsize_out,
    data_in, data_out);

  cudaCheck(cudaGetLastError());
}

//
// Transpose
//
template <typename T>
void transpose_xyz_zxy(
  const int nx, const int ny, const int nz,
  const int xsize_in, const int ysize_in,
  const int zsize_out, const int xsize_out,
  const T* data_in, T* data_out, cudaStream_t stream) {

  dim3 numthread(TILEDIM, TILEROWS, 1);
  dim3 numblock((nx-1)/TILEDIM+1, (nz-1)/TILEDIM+1, ny);

  transpose_xyz_zxy_kernel<T> <<< numblock, numthread, 0, stream >>>
  (nx, ny, nz, xsize_in, ysize_in,
    zsize_out, xsize_out,
    data_in, data_out);

  cudaCheck(cudaGetLastError());
}


template <typename T>
__global__ void copy_kernel(
  const int nx_ny_nz,
  const T* data_in, T* data_out) {

  for (int i = threadIdx.x + blockIdx.x*blockDim.x;i < nx_ny_nz;i += blockDim.x*gridDim.x) {
    data_out[i] = data_in[i];
  }

}

//
// Copy using scalar loads and stores
//
template <typename T>
void copy_xyz(
  const int nx_ny_nz,
  const T* data_in, T* data_out, cudaStream_t stream) {

  int numthread = TILEDIM*TILEROWS;
  int numblock = min(65535, (nx_ny_nz - 1)/numthread + 1);

  copy_kernel<T> <<< numblock, numthread, 0, stream >>>
  (nx_ny_nz, data_in, data_out);

  cudaCheck(cudaGetLastError());
}

template <typename T>
__global__ void copy_vector_kernel(const int n, T* data_in, T* data_out) {

  // Maximum vector load is 128 bits = 16 bytes
  const int vectorLength = 16/sizeof(T);

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  // Vector elements
  for (int i = idx;i < n/vectorLength;i += blockDim.x*gridDim.x) {
    reinterpret_cast<int4*>(data_out)[i] = reinterpret_cast<int4*>(data_in)[i];
  }

  // Remaining elements
  for (int i = idx + (n/vectorLength)*vectorLength;i < n;i += blockDim.x*gridDim.x + threadIdx.x) {
    data_out[i] = data_in[i];
  }

}

//
// Copy using vectorized loads and stores
//
template <typename T>
void copy_vector_xyz(
  const int nx_ny_nz,
  T* data_in, T* data_out, cudaStream_t stream) {

  const int vectorLength = 16/sizeof(T);

  int numthread = TILEDIM*TILEROWS;
  int numblock = min(65535, (nx_ny_nz/vectorLength - 1)/numthread + 1);
  int shmemsize = 0;

  copy_vector_kernel<T> <<< numblock, numthread, shmemsize, stream >>>
  (nx_ny_nz, data_in, data_out);

  cudaCheck(cudaGetLastError());
}

template <int numElem>
__global__ void memcpy_kernel(const int n, float4 *data_in, float4* data_out) {
  int index = threadIdx.x + numElem*blockIdx.x*blockDim.x;
  float4 a[numElem];
  for (int i=0;i < numElem;i++) {
    if (index + i*blockDim.x < n) a[i] = data_in[index + i*blockDim.x];
  }
  for (int i=0;i < numElem;i++) {
    if (index + i*blockDim.x < n) data_out[index + i*blockDim.x] = a[i];
  }
}

#define NUM_ELEM 2
//
// Copy using vectorized loads and stores
//
void memcpy_float(const int n,
  float* data_in, float* data_out, cudaStream_t stream) {

  int numthread = 256;
  int numblock = min(65535, (n/(4*NUM_ELEM) - 1)/numthread + 1);
  int shmemsize = 0;
  printf("numblock %d\n", numblock);

  memcpy_kernel<NUM_ELEM> <<< numblock, numthread, shmemsize, stream >>>
  (n/4, (float4 *)data_in, (float4 *)data_out);

  cudaCheck(cudaGetLastError());
}

#endif // CUDATRANSPOSE_H
