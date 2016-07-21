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

#define RESTRICT __restrict__

//
// Transpose when Mm and Mk don't overlap and contain only single rank
//
//  dim3 numthread(TILEDIM, TILEROWS, 1);
//  dim3 numblock((plan.volMm-1)/TILEDIM+1, (plan.volMk-1)/TILEDIM+1, plan.volMbar);
//
template <typename T>
__global__ void transposeTiledSingleRank(
  const int volMbar, const int sizeMbar,
  const int2 tiledVol, const int cuDimMk, const int cuDimMm,
  const TensorConvInOut* __restrict__ glMbar,
  const T* RESTRICT dataIn, T* RESTRICT dataOut) {

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

  const unsigned int maskIny = __ballot((yin + warpLane < tiledVol.y))*(xin < tiledVol.x);
  const unsigned int maskOutx = __ballot((xout + warpLane < tiledVol.x))*(yout < tiledVol.y);

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
// Transpose when Mm and Mk don't overlap, and Mm has single rank
//
template <typename T>
__global__ void transposeTiledSingleInRank(
  const int volMm, const int volMk, const int volMbar,
  const int sizeMk, const int sizeMbar,
  const int cMm,
  const TensorConvInOut* __restrict__ glMbar,
  const TensorConv* __restrict__ glMk,
  const T* RESTRICT dataIn, T* RESTRICT dataOut) {

  // Shared memory. (TILEDIM + 1)*volMk elements
  extern __shared__ char shBuffer_char[];
  T* shBuffer = (T *)shBuffer_char;

  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (threadIdx.x < sizeMbar) {
    Mbar = glMbar[threadIdx.x];
  }

  TensorConv Mk;
  Mk.c = 1;
  Mk.d = 1;
  if (threadIdx.x < sizeMk) {
    Mk = glMk[threadIdx.x];
  }

  // Single register stores all minor positions per warp
  int posMinorIn = 0;
  {
    int j = threadIdx.x*blockDim.y + threadIdx.y;
    for (int i=0;i < sizeMk;i++) {
      posMinorIn += ((j / __shfl(Mk.c,i)) % __shfl(Mk.d,i) ) * __shfl(Mk.ct,i);
    }
  }

  const int xi = threadIdx.x + blockIdx.x*TILEDIM;

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
    posMajorIn += xi;

    // Read from global memory
    __syncthreads();

    int jj = 0;
    for (int j=threadIdx.y;j < volMk;) {
      int posIn1 = posMajorIn + __shfl(posMinorIn, jj);
      if (xi < volMm) {
        int posShj = threadIdx.x + j*(TILEDIM+1);
        shBuffer[posShj] = dataIn[posIn1];
      }
      jj++;
      j += blockDim.y;
      int posIn2 = posMajorIn + __shfl(posMinorIn, jj);
      if (j < volMk && xi < volMm) {
        int posShj = threadIdx.x + j*(TILEDIM+1);
        shBuffer[posShj] = dataIn[posIn2];
      }
      jj++;
      j += blockDim.y;
    }

    // Write to global memory
    __syncthreads();

    for (int k=threadIdx.y;k < TILEDIM;k += blockDim.y) {
      int i = k + blockIdx.x*blockDim.y;
      if (i < volMm) {
        int posOut = posMajorOut + i*cMm + threadIdx.x;
        for (int j=threadIdx.x;j < volMk;) {
          dataOut[posOut] = shBuffer[j*(TILEDIM + 1) + k];
          j += TILEDIM;
          posOut += TILEDIM;
          if (j < volMk) {
            dataOut[posOut] = shBuffer[j*(TILEDIM + 1) + k];
          }
          j += TILEDIM;
          posOut += TILEDIM;
        }
      }
    }

  }
  
}

//
// Transpose when Mm and Mk don't overlap, and Mk has single rank
//
template <typename T>
__global__ void transposeTiledSingleOutRank(
  const int volMm, const int volMk, const int volMbar,
  const int sizeMm, const int sizeMbar,
  const int cMk,
  const TensorConvInOut* __restrict__ glMbar,
  const TensorConv* __restrict__ glMm,
  const T* RESTRICT dataIn, T* RESTRICT dataOut) {

  // Shared memory. TILEDIM*volMm elements
  extern __shared__ char shBuffer_char[];
  T* shBuffer = (T *)shBuffer_char;

  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (threadIdx.x < sizeMbar) {
    Mbar = glMbar[threadIdx.x];
  }

  TensorConv Mm;
  Mm.c = 1;
  Mm.d = 1;
  if (threadIdx.x < sizeMm) {
    Mm = glMm[threadIdx.x];
  }

  // Single register stores all minor positions per warp
  int posMinorOut = 0;
  {
    int j = threadIdx.x*blockDim.y + threadIdx.y;
    for (int i=0;i < sizeMm;i++) {
      posMinorOut += ((j / __shfl(Mm.c,i)) % __shfl(Mm.d,i) ) * __shfl(Mm.ct,i);
    }
  }

  // Position in Mk. blockIdx.x = tile index
  const int xi = threadIdx.x + blockIdx.x*TILEDIM;

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
    posMajorOut += xi;

    // Read from global memory
    __syncthreads();

    for (int k=threadIdx.y;k < TILEDIM;k+=blockDim.y) {
      int i = k + blockIdx.x*blockDim.y;
      if (i < volMk) {
        for (int j=threadIdx.x;j < volMm;j+=TILEDIM) {
          int posIn = posMajorIn + i*cMk + j;
          int posShj = j + k*volMm;
          shBuffer[posShj] = dataIn[posIn];
        }
      }
    }

    // Write to global memory
    __syncthreads();

    int jj = 0;
    for (int j=threadIdx.y;j < volMm;) {
      // j = position in MmO
      // xi = position in MkO
      int posOut = posMajorOut + __shfl(posMinorOut, jj);
      if (xi < volMk) {
        int posShj = j + threadIdx.x*volMm;
        dataOut[posOut] = shBuffer[posShj];
      }
      jj++;
      j += blockDim.y;
    }

  }
  
}

//
// General transpose. Thread block loads plan.volMmk number of elements
//
template <typename T, int numRegStorage>
__global__ void transposeGeneral(
  const int volMmk, const int volMbar,
  const int sizeMmk, const int sizeMbar,
  const TensorConvInOut* __restrict__ gl_Mmk,
  const TensorConvInOut* __restrict__ gl_Mbar,
  const TensorConv* __restrict__ gl_Msh,
  const T* RESTRICT dataIn, T* RESTRICT dataOut) {

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
// General method with split in-rank. Mm and Mk must not overlap. Mm must have single rank
//
template <typename T>
__global__ void transposeGeneralSplitInRank(
  const int volMm, const int volMk, const int volMbar,
  const int sizeMbar, const int cMm,
  const int* __restrict__ posMk,
  const TensorConvInOut* __restrict__ glMbar,
  const T* RESTRICT dataIn, T* RESTRICT dataOut) {

  // Shared memory. max(volMmSplit)*volMk T elements + volMk int elements
  extern __shared__ char shBuffer_char[];
  T* shBuffer = (T *)shBuffer_char;

  const int warpLane = threadIdx.x & (warpSize - 1);

  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = glMbar[warpLane];
  }

  // gridDim.x = number of splits
  // blockIdx.x = {0 ... gridDim.x - 1} is the split-index
  // Volume of this split
  const int volMmSplit = (volMm/gridDim.x) + (blockIdx.x < (volMm % gridDim.x));
  // Start position in this split
  const int xi = (volMm/gridDim.x)*blockIdx.x + min(blockIdx.x, (volMm % gridDim.x));
  const int volMmkSplit = volMmSplit*volMk;

  int maxVolMmSplit = (volMm/gridDim.x) + ((volMm % gridDim.x) > 0);
  int* shPosMk = (int *)&shBuffer[maxVolMmSplit*volMk];
  for (int j=threadIdx.x;j < volMk;j+=blockDim.x) {
    shPosMk[j] = posMk[j];
  }

  for (int posMbar=blockIdx.y;posMbar < volMbar;posMbar+=gridDim.y)
  {

    int posMbarOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      posMbarOut += __shfl_xor(posMbarOut, i);
    }
    posMbarOut += xi*cMm;

    int posMbarIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      posMbarIn += __shfl_xor(posMbarIn, i);
    }
    posMbarIn += xi;

    // Read from global memory
    __syncthreads();

    for (int j=threadIdx.x;j < volMmkSplit;j+=blockDim.x) {
      // (j % volMmSplit) = position in MmI
      // (j / volMmSplit) = {0 ... volMk - 1} = position in MkI
      // posMk[] translates to MkI -> MkO, so that we don't need
      // to compute shared memory positions
      int posIn = posMbarIn + (j % volMmSplit) + shPosMk[(j / volMmSplit)];
      shBuffer[j] = dataIn[posIn];
    }

    // Write to global memory
    __syncthreads();

    for (int j=threadIdx.x;j < volMmkSplit;j+=blockDim.x) {
      // (j % volMk) = position in MkO
      // (j / volMk) = position in MmO
      int posMkO = (j % volMk);
      int posMmO = (j / volMk);
      int posOut = posMbarOut + posMkO + posMmO*cMm;
      int posShj = posMkO*volMmSplit + posMmO;
      dataOut[posOut] = shBuffer[posShj];
    }

  }

}

//
// General method with split out-rank. Mm and Mk must not overlap. Mk must have single rank.
//
template <typename T>
__global__ void transposeGeneralSplitOutRank(
  const int volMm, const int volMk, const int volMbar,
  const int sizeMbar, const int cMk,
  const int* __restrict__ posMm,
  const TensorConvInOut* __restrict__ glMbar,
  const T* RESTRICT dataIn, T* RESTRICT dataOut) {

  // Shared memory. max(volMkSplit)*volMm T elements + volMm int elements
  extern __shared__ char shBuffer_char[];
  T* shBuffer = (T *)shBuffer_char;

  const int warpLane = threadIdx.x & (warpSize - 1);

  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = glMbar[warpLane];
  }

  // gridDim.x = number of splits
  // blockIdx.x = {0 ... gridDim.x - 1} is the split-index
  // Volume of this split
  const int volMkSplit = (volMk/gridDim.x) + (blockIdx.x < (volMk % gridDim.x));
  // Start position in this split
  const int xi = (volMk/gridDim.x)*blockIdx.x + min(blockIdx.x, (volMk % gridDim.x));
  const int volMmkSplit = volMkSplit*volMm;

  int maxVolMkSplit = (volMk/gridDim.x) + ((volMk % gridDim.x) > 0);
  int* shPosMm = (int *)&shBuffer[maxVolMkSplit*volMm];
  for (int j=threadIdx.x;j < volMm;j+=blockDim.x) {
    shPosMm[j] = posMm[j];
  }

  for (int posMbar=blockIdx.y;posMbar < volMbar;posMbar+=gridDim.y)
  {

    int posMbarOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      posMbarOut += __shfl_xor(posMbarOut, i);
    }
    posMbarOut += xi;

    int posMbarIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      posMbarIn += __shfl_xor(posMbarIn, i);
    }
    posMbarIn += xi*cMk;

    // Read from global memory
    __syncthreads();

    for (int j=threadIdx.x;j < volMmkSplit;j+=blockDim.x) {
      // (j % volMm) = position in MmI
      // (j / volMm) = {0 ... volMkSplit - 1} = position in MkI_{Split}
      int posIn = posMbarIn + (j / volMm)*cMk + (j % volMm);
      shBuffer[j] = dataIn[posIn];
    }

    // Write to global memory
    __syncthreads();

    for (int j=threadIdx.x;j < volMmkSplit;j+=blockDim.x) {
      // (j % volMkSplit) = position in MkO_{Split}
      // (j / volMkSplit) = position in MmO
      int posMkO = (j % volMkSplit);
      int posMmO = (j / volMkSplit);
      int posOut = posMbarOut + posMkO + shPosMm[posMmO];
      int posShj = posMkO*volMm + posMmO;
      dataOut[posOut] = shBuffer[posShj];
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
  const int2 tiledVol,
  const TensorConvInOut* __restrict__ gl_Mbar,
  const T* RESTRICT dataIn, T* RESTRICT dataOut) {

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
        if ((x < tiledVol.x) && (y + j < tiledVol.y)) {
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
        if ((x < tiledVol.x) && (y + j < tiledVol.y)) {
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
#include "calls.h"
#undef CALL

#define CALL(NREG) cudaCheck(cudaFuncSetSharedMemConfig(transposeGeneral<double, NREG>, cudaSharedMemBankSizeEightByte ))
#include "calls.h"
#undef CALL

  cudaCheck(cudaFuncSetSharedMemConfig(transposeGeneralSplitInRank<float>, cudaSharedMemBankSizeFourByte));
  cudaCheck(cudaFuncSetSharedMemConfig(transposeGeneralSplitInRank<double>, cudaSharedMemBankSizeEightByte));

  cudaCheck(cudaFuncSetSharedMemConfig(transposeGeneralSplitOutRank<float>, cudaSharedMemBankSizeFourByte));
  cudaCheck(cudaFuncSetSharedMemConfig(transposeGeneralSplitOutRank<double>, cudaSharedMemBankSizeEightByte));

  cudaCheck(cudaFuncSetSharedMemConfig(transposeTiledSingleInRank<float>, cudaSharedMemBankSizeFourByte));
  cudaCheck(cudaFuncSetSharedMemConfig(transposeTiledSingleInRank<double>, cudaSharedMemBankSizeEightByte));

  cudaCheck(cudaFuncSetSharedMemConfig(transposeTiledSingleOutRank<float>, cudaSharedMemBankSizeFourByte));
  cudaCheck(cudaFuncSetSharedMemConfig(transposeTiledSingleOutRank<double>, cudaSharedMemBankSizeEightByte));

  cudaCheck(cudaFuncSetSharedMemConfig(transposeTiledSingleRank<float>, cudaSharedMemBankSizeFourByte));
  cudaCheck(cudaFuncSetSharedMemConfig(transposeTiledLeadVolSame<float>, cudaSharedMemBankSizeFourByte));

  cudaCheck(cudaFuncSetSharedMemConfig(transposeTiledSingleRank<double>, cudaSharedMemBankSizeEightByte));
  cudaCheck(cudaFuncSetSharedMemConfig(transposeTiledLeadVolSame<double>, cudaSharedMemBankSizeEightByte));
}

//
// Returns the maximum number of active blocks per SM
//
int getNumActiveBlock(int method, int sizeofType, LaunchConfig& lc) {
  int numActiveBlock;
  int numthread = lc.numthread.x * lc.numthread.y * lc.numthread.z;
  switch(method) {
    case General:
    {
#define CALL0(TYPE, NREG) \
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock, \
    transposeGeneral<TYPE, NREG>, numthread, lc.shmemsize)
      switch(lc.numRegStorage) {
#define CALL(ICASE) case ICASE: if (sizeofType == 4) CALL0(float,  ICASE); if (sizeofType == 8) CALL0(double, ICASE); break
#include "calls.h"
      }
#undef CALL
#undef CALL0
    }
    break;

    case GeneralSplitInRank:
    {
      if (sizeofType == 4) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeGeneralSplitInRank<float>, numthread, lc.shmemsize);
      } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeGeneralSplitInRank<double>, numthread, lc.shmemsize);
      }
    }
    break;

    case GeneralSplitOutRank:
    {
      if (sizeofType == 4) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeGeneralSplitOutRank<float>, numthread, lc.shmemsize);
      } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeGeneralSplitOutRank<double>, numthread, lc.shmemsize);
      }
    }
    break;

    case TiledSingleInRank:
    {
      if (sizeofType == 4) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiledSingleInRank<float>, numthread, lc.shmemsize);
      } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiledSingleInRank<double>, numthread, lc.shmemsize);
      }
    }
    break;

    case TiledSingleOutRank:
    {
      if (sizeofType == 4) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiledSingleOutRank<float>, numthread, lc.shmemsize);
      } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiledSingleOutRank<double>, numthread, lc.shmemsize);
      }
    }
    break;

    case TiledSingleRank:
    {
      if (sizeofType == 4) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiledSingleRank<float>, numthread, lc.shmemsize);
      } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiledSingleRank<double>, numthread, lc.shmemsize);
      }
    }
    break;

    case TiledLeadVolSame:
    {
      if (sizeofType == 4) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiledLeadVolSame<float>, numthread, lc.shmemsize);
      } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiledLeadVolSame<double>, numthread, lc.shmemsize);
      }
    }
    break;
  }

  return numActiveBlock;
}

//
// Sets up kernel launch configuration
//
// Returns the number of active blocks per SM that can be achieved on the General kernel
// NOTE: Returns 0 when kernel execution is not possible
//
// Sets:
// lc.numthread
// lc.numblock
// lc.shmemsize
// lc.numRegStorage  (for General method)
//
int cuttKernelLaunchConfiguration(int sizeofType, TensorSplit& ts, cudaDeviceProp& prop,
  LaunchConfig& lc) {

  switch(ts.method) {
    case General:
    {
      // Amount of shared memory required
      lc.shmemsize = ts.shmemAlloc(sizeofType); //ts.volMmk*sizeofType;

      // Check that we're not using too much shared memory per block
      if (lc.shmemsize > prop.sharedMemPerBlock) {
        // printf("lc.shmemsize %d prop.sharedMemPerBlock %d\n", lc.shmemsize, prop.sharedMemPerBlock);
        return 0;
      }

      // Min and max number of threads we can use
      int minNumthread = ((ts.volMmk - 1)/(prop.warpSize*MAX_REG_STORAGE) + 1)*prop.warpSize;
      int maxNumthread = ((ts.volMmk - 1)/(prop.warpSize) + 1)*prop.warpSize;      
      if (minNumthread > prop.maxThreadsPerBlock) return 0;
      maxNumthread = min(prop.maxThreadsPerBlock, maxNumthread);
      // printf("minNumthread %d maxNumthread %d\n", minNumthread, maxNumthread);

      // Min and max number of register storage we can use
      int minNumRegStorage = (ts.volMmk - 1)/maxNumthread + 1;
      int maxNumRegStorage = (ts.volMmk - 1)/minNumthread + 1;
      // printf("minNumRegStorage %d maxNumRegStorage %d\n", minNumRegStorage, maxNumRegStorage);

      int bestVal = 0;
      int bestNumRegStorage = 0;

      lc.numthread.y = 1;
      lc.numthread.z = 1;
      lc.numblock.x = max(1, ts.volMbar);
      lc.numblock.x = min(256, lc.numblock.x);
      lc.numblock.y = 1;
      lc.numblock.z = 1;

      for (lc.numRegStorage=minNumRegStorage;lc.numRegStorage <= maxNumRegStorage;lc.numRegStorage++) {
        lc.numthread.x = ((ts.volMmk - 1)/(prop.warpSize*lc.numRegStorage) + 1)*prop.warpSize;

        int numActiveBlock = getNumActiveBlock(ts.method, sizeofType, lc);
        int val = numActiveBlock*lc.numthread.x;
        if (val > bestVal) {
          bestVal = val;
          bestNumRegStorage = lc.numRegStorage;
        }
      }

      if (bestNumRegStorage == 0) return 0;

      lc.numRegStorage = bestNumRegStorage;
      lc.numthread.x = ((ts.volMmk - 1)/(prop.warpSize*lc.numRegStorage) + 1)*prop.warpSize;
    }
    break;

    case GeneralSplitInRank:
    {
      int maxVolMmSplit = (ts.volMm/ts.numSplit) + ((ts.volMm % ts.numSplit) > 0);
      int maxVolMmkSplit = maxVolMmSplit*ts.volMk;
      // lc.shmemsize = maxVolMmSplit*ts.volMk*sizeofType + ts.volMk*sizeof(int);
      lc.shmemsize = ts.shmemAlloc(sizeofType);
      if (lc.shmemsize > prop.sharedMemPerBlock) return 0;
      lc.numthread.x = min(1024, ((maxVolMmkSplit - 1)/prop.warpSize + 1)*prop.warpSize );
      lc.numthread.y = 1;
      lc.numthread.z = 1;
      lc.numblock.x = ts.numSplit;
      lc.numblock.y = ts.volMbar;
      lc.numblock.y = min(64/lc.numblock.x, lc.numblock.y);
      lc.numblock.y = max(1, lc.numblock.y);
      lc.numblock.z = 1;
      lc.numRegStorage = 0;
    }
    break;

    case GeneralSplitOutRank:
    {
      int maxVolMkSplit = (ts.volMk/ts.numSplit) + ((ts.volMk % ts.numSplit) > 0);
      int maxVolMmkSplit = maxVolMkSplit*ts.volMm;
      // lc.shmemsize = maxVolMkSplit*ts.volMm*sizeofType + ts.volMm*sizeof(int);
      lc.shmemsize = ts.shmemAlloc(sizeofType);
      if (lc.shmemsize > prop.sharedMemPerBlock) return 0;
      lc.numthread.x = min(1024, ((maxVolMmkSplit - 1)/prop.warpSize + 1)*prop.warpSize );
      lc.numthread.y = 1;
      lc.numthread.z = 1;
      lc.numblock.x = ts.numSplit;
      lc.numblock.y = ts.volMbar;
      lc.numblock.y = min(64/lc.numblock.x, lc.numblock.y);
      lc.numblock.y = max(1, lc.numblock.y);
      lc.numblock.z = 1;
      lc.numRegStorage = 0;
    }
    break;

    case TiledSingleInRank:
    {
      // lc.shmemsize = (TILEDIM+1)*ts.volMk*sizeofType;
      lc.shmemsize = ts.shmemAlloc(sizeofType);
      if (lc.shmemsize > prop.sharedMemPerBlock) return 0;
      lc.numthread.x = TILEDIM;
      lc.numthread.y = 32;//min(32, ts.volMk);
      lc.numthread.z = 1;
      lc.numblock.x = (ts.volMm - 1)/TILEDIM + 1;
      lc.numblock.y = 1;
      lc.numblock.z = ts.volMbar;
      lc.numblock.z = min(64/lc.numblock.x, lc.numblock.z);
      lc.numblock.z = max(1, lc.numblock.z);
      lc.numRegStorage = 0;
    }
    break;

    case TiledSingleOutRank:
    {
      // lc.shmemsize = TILEDIM*ts.volMm*sizeofType;
      lc.shmemsize = ts.shmemAlloc(sizeofType);
      if (lc.shmemsize > prop.sharedMemPerBlock) return 0;
      lc.numthread.x = TILEDIM;
      lc.numthread.y = 32;//min(32, ts.volMm);
      lc.numthread.z = 1;
      lc.numblock.x = (ts.volMk - 1)/TILEDIM + 1;
      lc.numblock.y = 1;
      lc.numblock.z = ts.volMbar;
      lc.numblock.z = min(64/lc.numblock.x, lc.numblock.z);
      lc.numblock.z = max(1, lc.numblock.z);
      lc.numRegStorage = 0;
    }
    break;

    case TiledSingleRank:
    {
      lc.numthread.x = TILEDIM;
      lc.numthread.y = TILEROWS;
      lc.numthread.z = 1;
      lc.numblock.x = (ts.volMm - 1)/TILEDIM + 1;
      lc.numblock.y = (ts.volMk - 1)/TILEDIM + 1;
      lc.numblock.z = ts.volMbar;
      lc.numblock.z = min(64/(lc.numblock.x*lc.numblock.y), lc.numblock.z);
      lc.numblock.z = max(1, lc.numblock.z);
      lc.shmemsize = 0;
      lc.numRegStorage = 0;
    }
    break;

    case TiledLeadVolSame:
    {
      lc.numthread.x = TILEDIM;
      lc.numthread.y = TILEROWS;
      lc.numthread.z = 1;
      lc.numblock.x = (ts.volMm - 1)/TILEDIM + 1;
      lc.numblock.y = (ts.volMkBar - 1)/TILEDIM + 1;
      lc.numblock.z = ts.volMbar;
      lc.numblock.z = min(256/(lc.numblock.x*lc.numblock.y), lc.numblock.z);
      lc.numblock.z = max(1, lc.numblock.z);
      lc.shmemsize = 0;
      lc.numRegStorage = 0;
    }
    break;
  }

  if (lc.numblock.x > prop.maxGridSize[0] ||
    lc.numblock.y > prop.maxGridSize[1] ||
    lc.numblock.z > prop.maxGridSize[2]) return 0;

  // Return the number of active blocks with these settings
  return getNumActiveBlock(ts.method, sizeofType, lc);
}

#if 1
//
// Returns estimate of the number of memory reads and writes at warp level
//
void cuttKernelNumMemAccess(TensorSplit& ts, cudaDeviceProp& prop, LaunchConfig& lc,
  const int rank, const int* dim, const int* permutation, const size_t sizeofType,
  unsigned long long int& numRead, unsigned long long int& numWrite) {

  // Number of elements that are loaded per memory transaction:
  // 128 bytes per transaction
  unsigned int loadWidth = 128/sizeofType;

  std::vector<bool> isMmk(rank, false);
  for (int i=0;i < ts.sizeMm;i++) {
    isMmk[i] = true;
  }
  for (int i=0;i < ts.sizeMk;i++) {
    isMmk[permutation[i]] = true;
  }
  // Determine contigious read volume
  unsigned int volRead = 1;
  {
    for (int i=0;i < rank;i++) {
      if (!isMmk[i]) break;
      volRead *= dim[i];
    }
  }
  // Determine contigious write volume
  unsigned int volWrite = 1;
  {
    for (int i=0;i < rank;i++) {
      if (!isMmk[permutation[i]]) break;
      volWrite *= dim[permutation[i]];
    }
  }
  // Total volume
  unsigned long long int vol = ts.volMmk*ts.volMbar;

  // unsigned int numTileMm = ((ts.volMm - 1)/prop.warpSize + 1);
  // unsigned int numTileMk = ((ts.volMk - 1)/prop.warpSize + 1);

  switch(ts.method) {
    case General:
    {
      numRead = ((volRead - 1)/loadWidth + 1)*vol/volRead;
      numWrite = ((volWrite - 1)/loadWidth + 1)*vol/volWrite;
    }
    break;

    case GeneralSplitInRank:
    {
      numRead = 0;
      for (int i=0;i < ts.numSplit;i++) {
        int volMmSplit = (ts.volMm/ts.numSplit) + (i < (ts.volMm % ts.numSplit));
        numRead += ((volMmSplit - 1)/loadWidth + 1);
      }
      numRead = numRead*vol/ts.volMm;
      numWrite = ((volWrite - 1)/loadWidth + 1)*vol/volWrite;
    }
    break;

    case GeneralSplitOutRank:
    {
      numWrite = 0;
      for (int i=0;i < ts.numSplit;i++) {
        int volMkSplit = (ts.volMk/ts.numSplit) + (i < (ts.volMk % ts.numSplit));
        numWrite += ((volMkSplit - 1)/loadWidth + 1);
      }
      numWrite = numWrite*vol/ts.volMk;
      numRead = ((volRead - 1)/loadWidth + 1)*vol/volRead;
    }
    break;

    case TiledSingleInRank:
    {
      numRead = ((ts.volMm - 1)/loadWidth + 1)*vol/ts.volMm;
      numWrite = ((ts.volMk - 1)/loadWidth + 1)*vol/ts.volMk;
    }
    break;

    case TiledSingleOutRank:
    {
      numRead = ((ts.volMm - 1)/loadWidth + 1)*vol/ts.volMm;
      numWrite = ((ts.volMk - 1)/loadWidth + 1)*vol/ts.volMk;
    }
    break;

    case TiledSingleRank:
    {
      numRead = ((ts.volMm - 1)/loadWidth + 1)*vol/ts.volMm;
      numWrite = ((ts.volMk - 1)/loadWidth + 1)*vol/ts.volMk;
    }
    break;

    case TiledLeadVolSame:
    {
      numRead = ((ts.volMm - 1)/loadWidth + 1)*vol/ts.volMm;
      numWrite = ((ts.volMk - 1)/loadWidth + 1)*vol/ts.volMk;
    }
    break;
  }

}
#endif

bool cuttKernel(cuttPlan_t& plan, void* dataIn, void* dataOut) {

  LaunchConfig& lc = plan.launchConfig;
  TensorSplit& ts = plan.tensorSplit;

  switch(ts.method) {
    case General:
    {
      switch(lc.numRegStorage) {
#define CALL0(TYPE, NREG) \
    transposeGeneral<TYPE, NREG> <<< lc.numblock, lc.numthread, lc.shmemsize, plan.stream >>> \
      (ts.volMmk, ts.volMbar, ts.sizeMmk, ts.sizeMbar, \
      plan.Mmk, plan.Mbar, plan.Msh, (TYPE *)dataIn, (TYPE *)dataOut)
#define CALL(ICASE) case ICASE: if (plan.sizeofType == 4) CALL0(float,  ICASE); if (plan.sizeofType == 8) CALL0(double, ICASE); break
#include "calls.h"
        default:
        printf("cuttKernel no template implemented for numRegStorage %d\n", lc.numRegStorage);
        return false;
#undef CALL
#undef CALL0
      }

    }
    break;

    case GeneralSplitInRank:
    {
#define CALL(TYPE) \
      transposeGeneralSplitInRank<TYPE> \
      <<< lc.numblock, lc.numthread, lc.shmemsize, plan.stream >>> \
      (ts.volMm, ts.volMk, ts.volMbar, ts.sizeMbar, plan.cuDimMm, \
        plan.posMk, plan.Mbar, (TYPE *)dataIn, (TYPE *)dataOut)
      if (plan.sizeofType == 4) CALL(float);
      if (plan.sizeofType == 8) CALL(double);
#undef CALL      
    }
    break;

    case GeneralSplitOutRank:
    {
#define CALL(TYPE) \
      transposeGeneralSplitOutRank<TYPE> \
      <<< lc.numblock, lc.numthread, lc.shmemsize, plan.stream >>> \
      (ts.volMm, ts.volMk, ts.volMbar, ts.sizeMbar, plan.cuDimMk, \
        plan.posMm, plan.Mbar, (TYPE *)dataIn, (TYPE *)dataOut)
      if (plan.sizeofType == 4) CALL(float);
      if (plan.sizeofType == 8) CALL(double);
#undef CALL      
    }
    break;

    case TiledSingleInRank:
    {
#define CALL(TYPE) \
      transposeTiledSingleInRank<TYPE> \
      <<< lc.numblock, lc.numthread, lc.shmemsize, plan.stream >>> \
      (ts.volMm, ts.volMk, ts.volMbar, ts.sizeMk, ts.sizeMbar, plan.cuDimMm, \
        plan.Mbar, plan.Mk, (TYPE *)dataIn, (TYPE *)dataOut)
      if (plan.sizeofType == 4) CALL(float);
      if (plan.sizeofType == 8) CALL(double);
#undef CALL
    }
    break;

    case TiledSingleOutRank:
    {
#define CALL(TYPE) \
      transposeTiledSingleOutRank<TYPE> \
      <<< lc.numblock, lc.numthread, lc.shmemsize, plan.stream >>> \
      (ts.volMm, ts.volMk, ts.volMbar, ts.sizeMm, ts.sizeMbar, plan.cuDimMk, \
        plan.Mbar, plan.Mm, (TYPE *)dataIn, (TYPE *)dataOut)
      if (plan.sizeofType == 4) CALL(float);
      if (plan.sizeofType == 8) CALL(double);
#undef CALL
    }
    break;

    case TiledSingleRank:
    {
#define CALL(TYPE) \
      transposeTiledSingleRank<TYPE> <<< lc.numblock, lc.numthread, 0, plan.stream >>> \
      (ts.volMbar, ts.sizeMbar, plan.tiledVol, plan.cuDimMk, plan.cuDimMm, \
        plan.Mbar, (TYPE *)dataIn, (TYPE *)dataOut)
      if (plan.sizeofType == 4) CALL(float);
      if (plan.sizeofType == 8) CALL(double);
#undef CALL
    }
    break;

    case TiledLeadVolSame:
    {
#define CALL(TYPE) \
      transposeTiledLeadVolSame<TYPE> <<< lc.numblock, lc.numthread, 0, plan.stream >>> \
      (ts.volMbar, ts.sizeMbar, plan.cuDimMk, plan.cuDimMm, plan.tiledVol, \
        plan.Mbar, (TYPE *)dataIn, (TYPE *)dataOut)
      if (plan.sizeofType == 4) CALL(float);
      if (plan.sizeofType == 8) CALL(double);
#undef CALL
    }
    break;

  }

  cudaCheck(cudaGetLastError());
  return true;
}
