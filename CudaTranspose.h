#ifndef CUDATRANSPOSE_H
#define CUDATRANSPOSE_H

#include "CudaUtils.h"

const int TILEDIM = 32;
const int TILEROWS = 8;

// Arguments for transposeTensorKernel
// Enough to support tensors of rank 200 or so.
const int transposeArgSize = 2048 - 4 - 2*2;
__constant__ int transposeArg[transposeArgSize];

//
//
template <typename T>
__global__ void transposeTensorKernelArg(
  const int sizeMmk, const int sizeMbar,
  const int cuDimMk, const int cuDimMm,
  const T* __restrict__ dataIn, T* __restrict__ dataOut) {
  
  // Shared memory
  __shared__ T shTile[TILEDIM][TILEDIM+1];

  // const int warpLane = threadIdx.x & (warpSize - 1);

  int iarg = 0;
  int* dimMmkIn     = &transposeArg[iarg];
  int* dimMmkOut    = &transposeArg[iarg+=sizeMmk];
  int* dimMbarIn    = &transposeArg[iarg+=sizeMmk];
  int* dimMbarOut   = &transposeArg[iarg+=sizeMbar];
  int* cuDimMbarIn  = &transposeArg[iarg+=sizeMbar];
  int* cuDimMbarOut = &transposeArg[iarg+=sizeMbar];

  {
    // Read position
    const int x = blockIdx.x * TILEDIM + threadIdx.x;
    const int y = blockIdx.y * TILEDIM + threadIdx.y;

#if 1
    int pos0 = x + y*cuDimMk;
    int z = blockIdx.z;
    for (int i=0;i < sizeMbar;i++) {
      int dimMbarVal = dimMbarIn[i];
      pos0 += (z % dimMbarVal)*cuDimMbarIn[i];
      z /= dimMbarVal;
    }
#else
    int cuDimMk = cuDimIn[Mk[0]];
    int pos0 = 0;
    if (warpLane < sizeMbar) {
      int z = blockIdx.z/posDimMbarIn[warpLane];
      pos0 = (z % dimMbarIn[warpLane])*cuDimMbarIn[warpLane];
    }
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      pos0 += __shfl_xor(pos0, i, 32);
    }
    pos0 += x + y*cuDimMk;
#endif

    // Read data into shared memory tile
    for (int j=0;j < TILEDIM;j += TILEROWS) {
      int pos = pos0 + j*cuDimMk;
      if ((x < dimMmkIn[0]) && (y + j < dimMmkIn[1]))
        shTile[threadIdx.y + j][threadIdx.x] = dataIn[pos];
    }
  }

  {
    // Write position
    const int x = blockIdx.x * TILEDIM + threadIdx.y;
    const int y = blockIdx.y * TILEDIM + threadIdx.x;

#if 1
    int pos0 = y + x*cuDimMm;
    int z = blockIdx.z;
    for (int i=0;i < sizeMbar;i++) {
      int dimMbarVal = dimMbarOut[i];
      pos0 += (z % dimMbarVal)*cuDimMbarOut[i];
      z /= dimMbarVal;
    }
#else
    int cuDimMm = cuDimOut[Mm[0]];
    int pos0 = 0;
    if (warpLane < sizeMbar) {
      int z = blockIdx.z/posDimMbarOut[warpLane];
      pos0 = (z % dimMbarOut[warpLane])*cuDimMbarOut[warpLane];
    }
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      pos0 += __shfl_xor(pos0, i, 32);
    }
    pos0 += y + x*cuDimMm;
#endif

    __syncthreads();

    // Write data into global memory
    for (int j=0;j < TILEDIM;j += TILEROWS) {
      int pos = pos0 + j*cuDimMm;
      if ((y < dimMmkOut[0]) && (x + j < dimMmkOut[1]))
        dataOut[pos] = shTile[threadIdx.x][threadIdx.y + j];
    }
  }

}

#if 0
//
//
template <typename T>
__global__ void transposeTensorKernel(
  const int rank, const int sizeMbar,
  const int* __restrict__ permutation,
  const int* __restrict__ Mm, const int* __restrict__ Mk,
  const int* __restrict__ dim,
  const int* __restrict__ dimMbarIn, const int* __restrict__ dimMbarOut,
  const int* __restrict__ cuDimIn, const int* __restrict__ cuDimOut,
  const int* __restrict__ cuDimMbarIn, const int* __restrict__ cuDimMbarOut,
  const int* __restrict__ posDimMbarIn, const int* __restrict__ posDimMbarOut,
  const T* __restrict__ dataIn, T* __restrict__ dataOut) {
  
  // Shared memory
  __shared__ T shTile[TILEDIM][TILEDIM+1];
  // __shared__ int shDimMbarIn[1];
  // __shared__ int shCuDimMbarIn[1];
  // __shared__ int shDimMbarOut[1];
  // __shared__ int shCuDimMbarOut[1]; 

  // const int warpLane = threadIdx.x & (warpSize - 1);

  int MmVal = __ldg(&Mm[0]);
  int MkVal = __ldg(&Mk[0]);

  {
    // Read position
    const int x = blockIdx.x * TILEDIM + threadIdx.x;
    const int y = blockIdx.y * TILEDIM + threadIdx.y;

    int dimMmk[2];
    dimMmk[0] = __ldg(&dim[MmVal]);
    dimMmk[1] = __ldg(&dim[MkVal]);

#if 1
    int cuDimMk = __ldg(&cuDimIn[MkVal]);
    int pos0 = x + y*cuDimMk;
    int z = blockIdx.z;
    for (int i=0;i < sizeMbar;i++) {
      int dimMbarVal = __ldg(&dimMbarIn[i]);
      pos0 += (z % dimMbarVal)*__ldg(&cuDimMbarIn[i]);
      z /= dimMbarVal;
    }
#else
    int cuDimMk = cuDimIn[Mk[0]];
    int pos0 = 0;
    if (warpLane < sizeMbar) {
      int z = blockIdx.z/posDimMbarIn[warpLane];
      pos0 = (z % dimMbarIn[warpLane])*cuDimMbarIn[warpLane];
    }
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      pos0 += __shfl_xor(pos0, i, 32);
    }
    pos0 += x + y*cuDimMk;
#endif

    // Read data into shared memory tile
    if (x < dimMmk[0]) {
      for (int j=0;j < TILEDIM;j += TILEROWS) {
        int pos = pos0 + j*cuDimMk;
        // if ((x < dimMmk[0]) && (y + j < dimMmk[1]))
        if (y + j < dimMmk[1])
          shTile[threadIdx.y + j][threadIdx.x] = dataIn[pos];
      }
    }
  }

  {
    // Write position
    const int x = blockIdx.x * TILEDIM + threadIdx.y;
    const int y = blockIdx.y * TILEDIM + threadIdx.x;

    int dimMmk[2];
    dimMmk[0] = dim[permutation[MkVal]];
    dimMmk[1] = dim[permutation[MmVal]];

#if 1
    int cuDimMm = cuDimOut[MmVal];
    int pos0 = y + x*cuDimMm;
    int z = blockIdx.z;
    for (int i=0;i < sizeMbar;i++) {
      int dimMbarVal = dimMbarOut[i];
      pos0 += (z % dimMbarVal)*cuDimMbarOut[i];
      z /= dimMbarVal;
    }
#else
    int cuDimMm = cuDimOut[Mm[0]];
    int pos0 = 0;
    if (warpLane < sizeMbar) {
      int z = blockIdx.z/posDimMbarOut[warpLane];
      pos0 = (z % dimMbarOut[warpLane])*cuDimMbarOut[warpLane];
    }
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      pos0 += __shfl_xor(pos0, i, 32);
    }
    pos0 += y + x*cuDimMm;
#endif

    __syncthreads();

    // Write data into global memory
    if (y < dimMmk[0]) {
      for (int j=0;j < TILEDIM;j += TILEROWS) {
        int pos = pos0 + j*cuDimMm;
        // if ((y < dimMmk[0]) && (x + j < dimMmk[1]))
        if (x + j < dimMmk[1])
          dataOut[pos] = shTile[threadIdx.x][threadIdx.y + j];
      }
    }
  }

}
#endif

class TensorTransposePlan {
private:
  int* data;

public:
  // Rank of the tensor
  const int rank;
  // Input volume
  int volMm;
  // Output volume
  int volMk;
  // Remaining volume
  int sizeMbar;
  int volMbar;

  int sizeMmk;
  int cuDimMk;
  int cuDimMm;

  // Sizes:
  // Mm           [1]
  // Mk           [1]
  // permutation  [rank]
  // dim          [rank]
  // cuDimIn      [rank]
  // cuDimOut     [rank]
  // dimMbarIn    [rank - 2]
  // dimMbarOut   [rank - 2]
  // cuDimMbarIn  [rank - 2]
  // cuDimMbarOut [rank - 2]
  // posDimMbarIn  [rank - 2]
  // posDimMbarOut [rank - 2]

  // Pointers to data
  int* Mm;
  int* Mk;
  int* permutation;
  int* dim;
  int* cuDimIn;
  int* cuDimOut;
  int* dimMbarIn;
  int* dimMbarOut;
  int* cuDimMbarIn;
  int* cuDimMbarOut;
  int* posDimMbarIn;
  int* posDimMbarOut;

  static int getDataSize(const int rank) {
    return (2 + 4*rank + 6*(rank - 2));
  }

  TensorTransposePlan(const int rank, const int* in_dim, const int* in_permutation) : rank(rank) {

    allocate_device<int>(&data, getDataSize(rank));

    // Calculate pointers
    Mm           = data;
    Mk           = Mm + 1;
    permutation  = Mk + 1;
    dim          = permutation  + rank;
    cuDimIn      = dim          + rank;
    cuDimOut     = cuDimIn      + rank;
    dimMbarIn    = cuDimOut     + rank;
    dimMbarOut   = dimMbarIn    + rank - 2;
    cuDimMbarIn  = dimMbarOut   + rank - 2;
    cuDimMbarOut = cuDimMbarIn  + rank - 2;
    posDimMbarIn = cuDimMbarOut + rank - 2;
    posDimMbarOut = posDimMbarIn+ rank - 2;

    bool* isMmk = new bool[rank];
    int* inv_permutation = new int[rank];
    for (int i=0;i < rank;i++) {
      isMmk[i] = false;
      inv_permutation[in_permutation[i]] = i;
    }

    int* tmp_Mm = new int[1];
    int* tmp_Mk = new int[1];
    int* tmp_cuDimIn = new int[rank];
    int* tmp_cuDimOut = new int[rank];
    int* tmp_dimMbarIn = new int[rank - 2];
    int* tmp_dimMbarOut = new int[rank - 2];
    int* tmp_cuDimMbarIn = new int[rank - 2];
    int* tmp_cuDimMbarOut = new int[rank - 2];
    int* tmp_posDimMbarIn = new int[rank - 2];
    int* tmp_posDimMbarOut = new int[rank - 2];

    tmp_Mm[0] = 0;
    volMm = in_dim[tmp_Mm[0]];
    isMmk[tmp_Mm[0]] = true;

    if (in_permutation[0] == 0) {
      tmp_Mk[0] = in_permutation[1];
    } else {
      tmp_Mk[0] = in_permutation[0];
    }
    volMk = in_dim[tmp_Mk[0]];
    isMmk[tmp_Mk[0]] = true;

    tmp_cuDimIn[0] = 1;
    for (int i=1;i < rank;i++) {
      tmp_cuDimIn[i] = tmp_cuDimIn[i-1]*in_dim[i-1];
    }
    for (int i=0;i < rank;i++) {
      tmp_cuDimOut[i] = tmp_cuDimIn[inv_permutation[i]];
    }

    volMbar = 1;
    sizeMbar = 0;
    int prev_posDimMbarIn  = 1;
    int prev_posDimMbarOut = 1;
    for (int i=0;i < rank;i++) {
      if (!isMmk[i]) {
        volMbar *= in_dim[i];
        tmp_dimMbarIn[sizeMbar]    = in_dim[i];
        tmp_dimMbarOut[sizeMbar]   = in_dim[in_permutation[i]];
        tmp_cuDimMbarIn[sizeMbar]  = tmp_cuDimIn[i];
        tmp_cuDimMbarOut[sizeMbar] = tmp_cuDimOut[i];
        tmp_posDimMbarIn[sizeMbar]  = prev_posDimMbarIn;
        tmp_posDimMbarOut[sizeMbar] = prev_posDimMbarOut;
        prev_posDimMbarIn  *= tmp_dimMbarIn[sizeMbar];
        prev_posDimMbarOut *= tmp_dimMbarOut[sizeMbar];
        sizeMbar++;
      }
    }

    // ----------------------------------------------------------------------
    // For arg -version
    sizeMmk = 2;
    cuDimMk = tmp_cuDimIn[tmp_Mk[0]];
    cuDimMm = tmp_cuDimOut[tmp_Mm[0]];

    int* tmp_dimMmkIn = new int[sizeMmk];
    tmp_dimMmkIn[0] = in_dim[tmp_Mm[0]];
    tmp_dimMmkIn[1] = in_dim[tmp_Mk[0]];    

    int* tmp_dimMmkOut = new int[sizeMmk];
    tmp_dimMmkOut[0] = in_dim[in_permutation[tmp_Mk[0]]];
    tmp_dimMmkOut[1] = in_dim[in_permutation[tmp_Mm[1]]];

    int* h_transposeArg = new int[transposeArgSize];
    int iarg = 0;
    for (int j=0;j < sizeMmk;j++) h_transposeArg[iarg++] = tmp_dimMmkIn[j];
    for (int j=0;j < sizeMmk;j++) h_transposeArg[iarg++] = tmp_dimMmkOut[j];
    for (int j=0;j < sizeMbar;j++) h_transposeArg[iarg++] = tmp_dimMbarIn[j];
    for (int j=0;j < sizeMbar;j++) h_transposeArg[iarg++] = tmp_dimMbarOut[j];
    for (int j=0;j < sizeMbar;j++) h_transposeArg[iarg++] = tmp_cuDimMbarIn[j];
    for (int j=0;j < sizeMbar;j++) h_transposeArg[iarg++] = tmp_cuDimMbarOut[j];

    cudaCheck(cudaMemcpyToSymbol(transposeArg, h_transposeArg,
      transposeArgSize*sizeof(int), 0, cudaMemcpyHostToDevice));

    cudaCheck(cudaDeviceSynchronize());

    delete [] tmp_dimMmkIn;
    delete [] tmp_dimMmkOut;
    delete [] h_transposeArg;
    // ----------------------------------------------------------------------

#if 0
    printf("tmp_Mm %d\n", tmp_Mm[0]);
    printf("tmp_Mk %d\n", tmp_Mk[0]);

    printf("tmp_cuDimIn\n");
    for (int r=0;r < rank;r++) {
      printf("%d ", tmp_cuDimIn[r]);
    }
    printf("\n");

    printf("tmp_cuDimOut\n");
    for (int r=0;r < rank;r++) {
      printf("%d ", tmp_cuDimOut[r]);
    }
    printf("\n");

    printf("tmp_cuDimMbarIn\n");
    for (int r=0;r < sizeMbar;r++) {
      printf("%d ", tmp_cuDimMbarIn[r]);
    }
    printf("\n");

    printf("tmp_cuDimMbarOut\n");
    for (int r=0;r < sizeMbar;r++) {
      printf("%d ", tmp_cuDimMbarOut[r]);
    }
    printf("\n");
#endif

    delete [] isMmk;
    delete [] inv_permutation;

    int* hostData = new int[getDataSize(rank)];
    int i = 0;
    hostData[i++] = tmp_Mm[0];
    hostData[i++] = tmp_Mk[0];
    for (int j=0;j < rank;j++) hostData[i++] = in_permutation[j];
    for (int j=0;j < rank;j++) hostData[i++] = in_dim[j];
    for (int j=0;j < rank;j++) hostData[i++] = tmp_cuDimIn[j];
    for (int j=0;j < rank;j++) hostData[i++] = tmp_cuDimOut[j];
    for (int j=0;j < rank - 2;j++) hostData[i++] = tmp_dimMbarIn[j];
    for (int j=0;j < rank - 2;j++) hostData[i++] = tmp_dimMbarOut[j];
    for (int j=0;j < rank - 2;j++) hostData[i++] = tmp_cuDimMbarIn[j];
    for (int j=0;j < rank - 2;j++) hostData[i++] = tmp_cuDimMbarOut[j];
    for (int j=0;j < rank - 2;j++) hostData[i++] = tmp_posDimMbarIn[j];
    for (int j=0;j < rank - 2;j++) hostData[i++] = tmp_posDimMbarOut[j];

    delete [] tmp_Mm;
    delete [] tmp_Mk;
    delete [] tmp_cuDimIn;
    delete [] tmp_cuDimOut;
    delete [] tmp_dimMbarIn;
    delete [] tmp_dimMbarOut;
    delete [] tmp_cuDimMbarIn;
    delete [] tmp_cuDimMbarOut;
    delete [] tmp_posDimMbarIn;
    delete [] tmp_posDimMbarOut;

    copy_HtoD_sync<int>(hostData, data, getDataSize(rank));

    delete [] hostData;
  }

  ~TensorTransposePlan() {
    deallocate_device<int>(&data);
  }
};

template <typename T>
void transposeTensorArg(TensorTransposePlan& plan,
  const T* dataIn, T* dataOut, cudaStream_t stream) {

  dim3 numthread(TILEDIM, TILEROWS, 1);
  dim3 numblock((plan.volMm-1)/TILEDIM+1, (plan.volMk-1)/TILEDIM+1, plan.volMbar);

  transposeTensorKernelArg<T> <<< numblock, numthread, 0, stream >>>
  (plan.sizeMmk, plan.sizeMbar, plan.cuDimMk, plan.cuDimMm,
    dataIn, dataOut);

  cudaCheck(cudaGetLastError());
}

#if 0
template <typename T>
void transposeTensor(TensorTransposePlan& plan,
  const T* dataIn, T* dataOut, cudaStream_t stream) {

  dim3 numthread(TILEDIM, TILEROWS, 1);
  dim3 numblock((plan.volMm-1)/TILEDIM+1, (plan.volMk-1)/TILEDIM+1, plan.volMbar);

  transposeTensorKernel<T> <<< numblock, numthread, 0, stream >>>
  (plan.rank, plan.sizeMbar,
    plan.permutation,
    plan.Mm, plan.Mk,
    plan.dim, plan.dimMbarIn, plan.dimMbarOut,
    plan.cuDimIn, plan.cuDimOut,
    plan.cuDimMbarIn, plan.cuDimMbarOut,
    plan.posDimMbarIn, plan.posDimMbarOut,
    dataIn, dataOut);

  cudaCheck(cudaGetLastError());
}
#endif

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
  const int nx, const int ny, const int nz,
  const T* data_in, T* data_out, cudaStream_t stream) {

  int nx_ny_nz = nx*ny*nz;

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
  const int nx, const int ny, const int nz,
  T* data_in, T* data_out, cudaStream_t stream) {

  const int vectorLength = 16/sizeof(T);

  int nx_ny_nz = nx*ny*nz;

  int numthread = TILEDIM*TILEROWS;
  int numblock = min(65535, (nx_ny_nz/vectorLength - 1)/numthread + 1);
  int shmemsize = 16384*2;

  copy_vector_kernel<T> <<< numblock, numthread, shmemsize, stream >>>
  (nx_ny_nz, data_in, data_out);

  cudaCheck(cudaGetLastError());
}

#endif // CUDATRANSPOSE_H
