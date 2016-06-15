#ifndef CUDATRANSPOSE_H
#define CUDATRANSPOSE_H

#include "CudaUtils.h"

struct MbarRecord {
  int posDimIn;
  int dimIn;
  int cuDimIn;
  int posDimOut;
  int dimOut;
  int cuDimOut;
};

struct MmkRecord {
  int posDim;
  int dim;
  int cuDim;
};

const int TILEDIM = 32;
const int TILEROWS = 8;

// Arguments for transposeTensorKernel
// Enough to support tensors of rank 200 or so.
const int transposeArgSize = 2048 - 6 - 2*3;
__constant__ int transposeArg[transposeArgSize];

//
// Transpose when the lead dimension is different e.g. (1, 2, 3) -> (2, 1, 3)
//
//  dim3 numthread(TILEDIM, TILEROWS, 1);
//  dim3 numblock((plan.volMm-1)/TILEDIM+1, (plan.volMk-1)/TILEDIM+1, plan.volMbar);
//
template <typename T>
__global__ void transposeTensorKernelArg(
  const int volMbar,
  // const int sizeMmk, 
  const int sizeMbar,
  const int2 readVol, const int2 writeVol,
  const int cuDimMk, const int cuDimMm,
  const MbarRecord* __restrict__ gl_Mbar,
  const T* __restrict__ dataIn, T* __restrict__ dataOut) {

  // Shared memory
  __shared__ T shTile[TILEDIM][TILEDIM+1];

  const int warpLane = threadIdx.x & (warpSize - 1);
  MbarRecord Mbar;
  if (warpLane < sizeMbar) {
    Mbar = gl_Mbar[warpLane];
  }

  // int* dimMmkIn  = &transposeArg[0];
  // int* dimMmkOut = &transposeArg[sizeMmk];

  const int xin = blockIdx.x * TILEDIM + threadIdx.x;
  const int yin = blockIdx.y * TILEDIM + threadIdx.y;

  const int xout = blockIdx.x * TILEDIM + threadIdx.y;
  const int yout = blockIdx.y * TILEDIM + threadIdx.x;

  // if (xin == 0 && yin == 0 && blockIdx.z == 0) {
  //   printf("dimMmkIn %d %d dimMmkOut %d %d\n", dimMmkIn[0], dimMmkIn[1], dimMmkOut[0], dimMmkOut[1]);
  //   printf("cuDimMk %d cuDimMm %d MbarIn %d %d %d MbarOut %d %d %d\n",
  //     cuDimMk, cuDimMm, Mbar.posDimIn, Mbar.dimIn, Mbar.cuDimIn,
  //     Mbar.posDimOut, Mbar.dimOut, Mbar.cuDimOut);
  // }

  for (int blockz=blockIdx.z;blockz < volMbar;blockz += blockDim.z*gridDim.z)
  {

    // Read from global memory
    {
      int pos0 = 0;
      if (warpLane < sizeMbar) {
        int z = blockz/Mbar.posDimIn;
        pos0 = (z % Mbar.dimIn)*Mbar.cuDimIn;
      }
#pragma unroll
      for (int i=16;i >= 1;i/=2) {
        pos0 += __shfl_xor(pos0, i);
      }
      pos0 += xin + yin*cuDimMk;

      __syncthreads();

      // Read data into shared memory tile
      for (int j=0;j < TILEDIM;j += TILEROWS) {
        int pos = pos0 + j*cuDimMk;
        // if ((xin < dimMmkIn[0]) && (yin + j < dimMmkIn[1])) {
        if ((xin < readVol.x) && (yin + j < readVol.y)) {
          shTile[threadIdx.y + j][threadIdx.x] = dataIn[pos];
        }
      }
    }

    // Write to global memory
    {
      int pos0 = 0;
      if (warpLane < sizeMbar) {
        int z = blockz/Mbar.posDimOut;
        pos0 = (z % Mbar.dimOut)*Mbar.cuDimOut;
      }
#pragma unroll
      for (int i=16;i >= 1;i/=2) {
        pos0 += __shfl_xor(pos0, i);
      }
      pos0 += yout + xout*cuDimMm;

      __syncthreads();

      for (int j=0;j < TILEDIM;j += TILEROWS) {
        int pos = pos0 + j*cuDimMm;
        // if ((yout < dimMmkOut[0]) && (xout + j < dimMmkOut[1])) {
        if ((yout < writeVol.x) && (xout + j < writeVol.y)) {
          dataOut[pos] = shTile[threadIdx.x][threadIdx.y + j];
        }
      }
    }

  }
  
}

//
// Transpose when dimensions are combined
//
//  dim3 numthread(TILEDIM, TILEROWS, 1);
//  dim3 numblock((plan.volMm-1)/TILEDIM+1, (plan.volMk-1)/TILEDIM+1, plan.volMbar);
//
template <typename T>
__global__ void transposeTensorKernelArg_subTransp(
  const int volMbar, const int sizeMmk, const int sizeMbar,
  const int2 readVol, const int2 writeVol,
  const int cuDimMk, const int cuDimMm,
  const MbarRecord* __restrict__ gl_Mbar,
  const MmkRecord* __restrict__ gl_Mmk,
  const T* __restrict__ dataIn, T* __restrict__ dataOut) {

  // Shared memory
  __shared__ T shTile[TILEDIM][TILEDIM+1];
  __shared__ T shTile2[TILEDIM][TILEDIM+1];

  const int warpLane = threadIdx.x & (warpSize - 1);
  MbarRecord Mbar;
  if (warpLane < sizeMbar) {
    Mbar = gl_Mbar[warpLane];
  }

  MmkRecord Mmk;
  if (warpLane < sizeMmk) {
    Mmk = gl_Mmk[warpLane];
  }

  // int* dimMmkIn  = &transposeArg[0];
  // int* dimMmkOut = &transposeArg[sizeMmk];

  const int xin = blockIdx.x * TILEDIM + threadIdx.x;
  const int yin = blockIdx.y * TILEDIM + threadIdx.y;

  const int xout = blockIdx.x * TILEDIM + threadIdx.y;
  const int yout = blockIdx.y * TILEDIM + threadIdx.x;

  // if (xin == 0 && yin == 0 && blockIdx.z == 0) {
  //   printf("dimMmkIn %d %d dimMmkOut %d %d\n", dimMmkIn[0], dimMmkIn[1], dimMmkOut[0], dimMmkOut[1]);
  //   printf("cuDimMk %d cuDimMm %d MbarIn %d %d %d MbarOut %d %d %d\n",
  //     cuDimMk, cuDimMm, Mbar.posDimIn, Mbar.dimIn, Mbar.cuDimIn,
  //     Mbar.posDimOut, Mbar.dimOut, Mbar.cuDimOut);
  // }

  for (int blockz=blockIdx.z;blockz < volMbar;blockz += blockDim.z*gridDim.z)
  {

    // Read from global memory
    {
      int pos0 = 0;
      if (warpLane < sizeMbar) {
        int z = blockz/Mbar.posDimIn;
        pos0 = (z % Mbar.dimIn)*Mbar.cuDimIn;
      }
#pragma unroll
      for (int i=16;i >= 1;i/=2) {
        pos0 += __shfl_xor(pos0, i);
      }
      pos0 += xin + yin*cuDimMk;

      __syncthreads();

      // Read data into shared memory tile
      for (int j=0;j < TILEDIM;j += TILEROWS) {
        int pos = pos0 + j*cuDimMk;
        if ((xin < readVol.x) && (yin + j < readVol.y)) {
          shTile[threadIdx.y + j][threadIdx.x] = dataIn[pos];
        }
      }
    }

    // Transpose within tile
    int* shTile2p = (int *)shTile2;
    for (int j=0;j < TILEDIM;j += TILEROWS) {
      // Read from (threadIdx.x, threadIdx.y + j)
      T val = shTile[threadIdx.y + j][threadIdx.x];
      int xy = threadIdx.x + (threadIdx.y + j)*TILEDIM;
      // Break down xy to calculate output address
      int pos = 0;
      if (warpLane < sizeMmk) {
        int p = xy/Mmk.posDim;
        pos = (p % Mmk.dim)*Mmk.cuDim;
      }
#pragma unroll
      for (int i=16;i >= 1;i/=2) {
        pos += __shfl_xor(pos, i);
      }
      // Write to shared memory
      shTile2p[pos] = val;
    }

    // Write to global memory
    {
      int pos0 = 0;
      if (warpLane < sizeMbar) {
        int z = blockz/Mbar.posDimOut;
        pos0 = (z % Mbar.dimOut)*Mbar.cuDimOut;
      }
#pragma unroll
      for (int i=16;i >= 1;i/=2) {
        pos0 += __shfl_xor(pos0, i);
      }
      pos0 += yout + xout*cuDimMm;

      __syncthreads();

      for (int j=0;j < TILEDIM;j += TILEROWS) {
        int pos = pos0 + j*cuDimMm;
        if ((yout < writeVol.x) && (xout + j < writeVol.y)) {
          dataOut[pos] = shTile2[threadIdx.x][threadIdx.y + j];
        }
      }
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
__global__ void transposeTensorKernelArg_leadDimSame(
  const int volMbar, const int sizeMbar,
  const int cuDimMk, const int cuDimMm,
  const MbarRecord* __restrict__ gl_Mbar,
  const T* __restrict__ dataIn, T* __restrict__ dataOut) {

  const int warpLane = threadIdx.x & (warpSize - 1);
  MbarRecord Mbar;
  if (warpLane < sizeMbar) {
    Mbar = gl_Mbar[warpLane];
  }

  int* dimMmkIn  = &transposeArg[0];

  const int x = blockIdx.x * TILEDIM + threadIdx.x;
  const int y = blockIdx.y * TILEDIM + threadIdx.y;

  // if (x == 0 && y == 0 && blockIdx.z == 0) {
  //   printf("cuDimMk %d cuDimMm %d volMmk %d MbarOut %d %d %d\n",
  //     cuDimMk, cuDimMm, volMmk, MbarOut.x, MbarOut.y, MbarOut.z);
  // }

  for (int blockz=blockIdx.z;blockz < volMbar;blockz += blockDim.z*gridDim.z)
  {

    // Variable where values are stored
    T val[TILEDIM/TILEROWS];

    // Read global memory
    {
    // int posIn0 = x + y*cuDimMk + blockz*volMmk;
      int pos0 = 0;
      if (warpLane < sizeMbar) {
        int z = blockz/Mbar.posDimIn;
        pos0 = (z % Mbar.dimIn)*Mbar.cuDimIn;
      }
#pragma unroll
      for (int i=16;i >= 1;i/=2) {
        pos0 += __shfl_xor(pos0, i);
      }
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
      int pos0 = 0;
      if (warpLane < sizeMbar) {
        int z = blockz/Mbar.posDimOut;
        pos0 = (z % Mbar.dimOut)*Mbar.cuDimOut;
      }
#pragma unroll
      for (int i=16;i >= 1;i/=2) {
        pos0 += __shfl_xor(pos0, i);
      }
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

class TensorTransposePlan {
public:
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

  bool leadDimSame;

  bool subTransp;

  int2 readVol;
  int2 writeVol;

  // sizeMbar
  MbarRecord* Mbar;

  // sizeMmk
  MmkRecord* Mmk;

  static int getDataSize(const int rank) {
    return (2 + 4*rank + 6*(rank - 2));
  }

  TensorTransposePlan(const int rank, const int* dim, const int* permutation) : rank(rank) {

    leadDimSame = false;
    subTransp = false;

    bool* isMm = new bool[rank];
    bool* isMk = new bool[rank];
    int* inv_permutation = new int[rank];
    for (int i=0;i < rank;i++) {
      isMm[i] = false;
      isMk[i] = false;
      inv_permutation[permutation[i]] = i;
    }

    // Setup Mm
    sizeMm = 0;
    volMm = 1;
    while (sizeMm < rank && volMm < TILEDIM) {
    // while (sizeMm < rank && volMm < 2) {
      isMm[sizeMm] = true;
      volMm *= dim[sizeMm++];
    }

    // Setup Mk
    int r = 0;
    sizeMk = 0;
    volMk = 1;
    while (r < rank && volMk < TILEDIM) {
    // while (r < rank && volMk < 2) {
      if (leadDimSame) {
        isMk[r] = true;
        volMk *= dim[r];
        sizeMk++;
      } else {
        int pr = permutation[r];
        if (isMm[pr]) {
          leadDimSame = true;
        } else {
          isMk[pr] = true;
          volMk *= dim[pr];
          sizeMk++;
        }
      }
      r++;
    }

    int* tmp_Mm = new int[sizeMm];
    int* tmp_Mk = new int[sizeMk];
    int iMm = 0;
    int iMk = 0;
    for (int i=0;i < rank;i++) {
      if (isMm[i]) {
        tmp_Mm[iMm++] = i;
      }
      if (isMk[i]) {
        tmp_Mk[iMk++] = i;
      }
    }

    // Setup Mmk
    sizeMmk = 0;
    sizeMbar = 0;
    volMbar = 1;
    volMmk = 1;
    for (int i=0;i < rank;i++) {
      if (isMm[i] || isMk[i]) {
        volMmk *= dim[i];
        sizeMmk++;
      } else {
        volMbar *= dim[i];
        sizeMbar++;
      }
    }

    int* tmp_cuDimIn = new int[rank];
    int* tmp_cuDimOut = new int[rank];
    tmp_cuDimIn[0] = 1;
    for (int i=1;i < rank;i++) {
      tmp_cuDimIn[i] = tmp_cuDimIn[i-1]*dim[i-1];
    }
    tmp_cuDimOut[0] = 1;
    for (int i=1;i < rank;i++) {
      tmp_cuDimOut[i] = tmp_cuDimOut[i-1]*dim[permutation[i-1]];
    }

    cuDimMk = tmp_cuDimIn[tmp_Mk[0]];
    if (leadDimSame) {
      cuDimMm = tmp_cuDimOut[inv_permutation[tmp_Mk[0]]];
    } else {
      cuDimMm = tmp_cuDimOut[inv_permutation[tmp_Mm[0]]];
    }

    MbarRecord* hostMbar = NULL;
    if (sizeMbar > 0) {
      hostMbar = new MbarRecord[sizeMbar];
      int* tmp = new int[rank];
      int prev_posDimMbarIn  = 1;
      int iMbar = 0;
      for (int i=0;i < rank;i++) {
        if (!(isMm[i] || isMk[i])) {
          tmp[i] = prev_posDimMbarIn;
          hostMbar[iMbar].posDimIn = prev_posDimMbarIn;
          hostMbar[iMbar].dimIn    = dim[i];
          hostMbar[iMbar].cuDimIn  = tmp_cuDimIn[i];
          prev_posDimMbarIn *= hostMbar[iMbar].dimIn;
          iMbar++;
        }
      }

      iMbar = 0;
      for (int i=0;i < rank;i++) {
        int pi = permutation[i];
        if (!(isMm[pi] || isMk[pi])) {
          hostMbar[iMbar].posDimOut = tmp[pi];
          hostMbar[iMbar].dimOut    = dim[pi];
          hostMbar[iMbar].cuDimOut  = tmp_cuDimOut[i];
          iMbar++;
        }
      }

      delete [] tmp;
    }

    int* tmp_dimMmkIn = new int[sizeMmk];
    tmp_dimMmkIn[0] = dim[tmp_Mm[0]];
    tmp_dimMmkIn[1] = dim[tmp_Mk[0]];

    readVol.x = 1;
    readVol.y = 1;
    for (int i=0;i < sizeMm;i++) {
      readVol.x *= dim[tmp_Mm[i]];
    }
    for (int i=0;i < sizeMk;i++) {
      readVol.y *= dim[tmp_Mk[i]];
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

#if 1
    printf("Mm");
    for (int i = 0; i < sizeMm; ++i) printf(" %d", tmp_Mm[i]+1);
    printf(" volMm %d\n", volMm);

    printf("Mk");
    for (int i = 0; i < sizeMk; ++i) printf(" %d", tmp_Mk[i]+1);
    printf(" volMk %d\n", volMk);

    printf("Mmk");
    for (int i = 0; i < rank; ++i) if (isMm[i] || isMk[i]) printf(" %d", i+1);
    printf(" volMmk %d\n", volMmk);

    if (sizeMbar > 0) {
      printf("Mbar");
      for (int i = 0; i < rank; ++i) if (!(isMm[i] || isMk[i])) printf(" %d", i+1);
      printf(" volMbar %d\n", volMbar);
    }

    printf("cuDimIn ");
    for (int i=0;i < rank;i++) printf("%d ", tmp_cuDimIn[i]);
    printf("\n");

    printf("cuDimOut ");
    for (int i=0;i < rank;i++) printf("%d ", tmp_cuDimOut[i]);
    printf("\n");

    printf("cuDimMk %d cuDimMm %d\n", cuDimMk, cuDimMm);

    if (sizeMbar > 0) {
      printf("MbarIn\n");
      for (int i=0;i < sizeMbar;i++) printf("%d %d %d\n",
        hostMbar[i].posDimIn, hostMbar[i].dimIn, hostMbar[i].cuDimIn);

      printf("MbarOut\n");
      for (int i=0;i < sizeMbar;i++) printf("%d %d %d\n",
        hostMbar[i].posDimOut, hostMbar[i].dimOut, hostMbar[i].cuDimOut);
    }

    printf("readVol %d %d writeVol %d %d\n", readVol.x, readVol.y, writeVol.x, writeVol.y);
#endif

    int* h_transposeArg = new int[transposeArgSize];
    int iarg = 0;
    for (int j=0;j < sizeMmk;j++) h_transposeArg[iarg++] = tmp_dimMmkIn[j];
    for (int j=0;j < sizeMmk;j++) h_transposeArg[iarg++] = tmp_dimMmkOut[j];

    cudaCheck(cudaMemcpyToSymbol(transposeArg, h_transposeArg,
      transposeArgSize*sizeof(int), 0, cudaMemcpyHostToDevice));

    // Check for sub-transpose
    MmkRecord* hostMmk = NULL;
    if (sizeMm > 1 || sizeMk > 1) {
      // We need sub-transpose if the order within Mm and Mk changes
      int* tmp_MmOut = new int[sizeMm];
      int jm = 0;
      for (int i=0;i < rank;i++) {
        int pi = permutation[i];
        if (isMm[pi]) {
          tmp_MmOut[jm++] = pi;
        }
      }
      bool Mm_order_changes = false;
      for (int i=0;i < sizeMm;i++) {
        if (tmp_Mm[i] != tmp_MmOut[i]) {
          Mm_order_changes = true;
          break;
        }
      }

      int* tmp_MkOut = new int[sizeMk];
      int jk = 0;
      for (int i=0;i < rank;i++) {
        int pi = permutation[i];
        if (isMk[pi]) {
          tmp_MkOut[jk++] = pi;
        }
      }
      bool Mk_order_changes = false;
      for (int i=0;i < sizeMk;i++) {
        if (tmp_Mk[i] != tmp_MkOut[i]) {
          Mk_order_changes = true;
          break;
        }
      }

      subTransp = (Mm_order_changes || Mk_order_changes);
      
      if (subTransp) {
        hostMmk = new MmkRecord[sizeMmk];
        int* tmp = new int[rank];
        int prev_posDim = 1;
        int prev_cuDim = 1;
        int j = 0;
        // Mm
        for (int i=0;i < sizeMm;i++) {
          tmp[tmp_Mm[i]] = prev_posDim;
          hostMmk[j].dim   = dim[tmp_MmOut[i]];
          hostMmk[j].cuDim = prev_cuDim;
          prev_cuDim *= hostMmk[j].dim;
          prev_posDim *= dim[tmp_Mm[i]];
          j++;
        }
        // Mk
        for (int i=0;i < sizeMk;i++) {
          tmp[tmp_Mk[i]] = prev_posDim;
          hostMmk[j].dim   = dim[tmp_MkOut[i]];
          hostMmk[j].cuDim = prev_cuDim;
          prev_cuDim *= hostMmk[j].dim;
          prev_posDim *= dim[tmp_Mk[i]];
          j++;
        }
        j = 0;
        // Mm
        for (int i=0;i < sizeMm;i++) {
          hostMmk[j].posDim = tmp[tmp_MmOut[i]];
          j++;
        }
        // Mk
        for (int i=0;i < sizeMk;i++) {
          hostMmk[j].posDim = tmp[tmp_MkOut[i]];
          j++;
        }
        delete [] tmp;
      }

      delete [] tmp_MmOut;
      delete [] tmp_MkOut;
    }

    delete [] tmp_dimMmkIn;
    delete [] tmp_dimMmkOut;
    delete [] h_transposeArg;

    delete [] isMm;
    delete [] isMk;
    delete [] inv_permutation;

    delete [] tmp_Mm;
    delete [] tmp_Mk;
    delete [] tmp_cuDimIn;
    delete [] tmp_cuDimOut;
    // delete [] tmp_dimMbarIn;
    // delete [] tmp_dimMbarOut;
    // delete [] tmp_cuDimMbarIn;
    // delete [] tmp_cuDimMbarOut;
    // delete [] tmp_posDimMbarIn;
    // delete [] tmp_posDimMbarOut;

    if (sizeMbar > 0) {
      allocate_device<MbarRecord>(&Mbar, sizeMbar);
      copy_HtoD_sync<MbarRecord>(hostMbar, Mbar, sizeMbar);
      delete [] hostMbar;
    }

    if (subTransp) {
      allocate_device<MmkRecord>(&Mmk, sizeMmk);
      copy_HtoD_sync<MmkRecord>(hostMmk, Mmk, sizeMmk);
      delete [] hostMmk;
    }

    cudaCheck(cudaDeviceSynchronize());
  }

  ~TensorTransposePlan() {
    if (sizeMbar > 0) deallocate_device<MbarRecord>(&Mbar);
    if (subTransp) deallocate_device<MmkRecord>(&Mmk);
  }
};

template <typename T>
void transposeTensorArg(TensorTransposePlan& plan,
  const T* dataIn, T* dataOut, cudaStream_t stream) {

  dim3 numthread(TILEDIM, TILEROWS, 1);
  dim3 numblock((plan.volMm-1)/TILEDIM+1, (plan.volMk-1)/TILEDIM+1, plan.volMbar);
  numblock.z = min(256, plan.volMbar);
  numblock.z = min(65535, numblock.z);

  if (plan.leadDimSame) {
    transposeTensorKernelArg_leadDimSame<T> <<< numblock, numthread, 0, stream >>>
    (plan.volMbar, plan.sizeMbar, plan.cuDimMk, plan.cuDimMm,
      plan.Mbar, dataIn, dataOut);
  } else {
    if (plan.subTransp) {
      transposeTensorKernelArg_subTransp<T> <<< numblock, numthread, 0, stream >>>
      (plan.volMbar, plan.sizeMmk, plan.sizeMbar, plan.readVol, plan.writeVol, plan.cuDimMk, plan.cuDimMm,
        plan.Mbar, plan.Mmk, dataIn, dataOut);
    } else {
      transposeTensorKernelArg<T> <<< numblock, numthread, 0, stream >>>
      (plan.volMbar, plan.sizeMbar, plan.readVol, plan.writeVol, plan.cuDimMk, plan.cuDimMm,
        plan.Mbar, dataIn, dataOut);
    }
  }

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
