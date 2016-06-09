#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARPSIZE 32

//
// Error checking wrapper for CUDA
//
#define cudaCheck(stmt) do {                                 \
	cudaError_t err = stmt;                            \
  if (err != cudaSuccess) {                          \
	  fprintf(stderr, "%s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
	  exit(1); \
  }                                                  \
} while(0)

void clear_device_array_async_T(void *data, const int ndata, cudaStream_t stream, const size_t sizeofT);
void clear_device_array_T(void *data, const int ndata, const size_t sizeofT);

template <class T>
void clear_device_array(T *data, const int ndata, cudaStream_t stream=0) {
  clear_device_array_async_T(data, ndata, stream, sizeof(T));
}

template <class T>
void clear_device_array_sync(T *data, const int ndata) {
  clear_device_array_T(data, ndata, sizeof(T));
}

void allocate_host_T(void **pp, const int len, const size_t sizeofT);
//----------------------------------------------------------------------------------------
//
// Allocate page-locked host memory
// pp = memory pointer
// len = length of the array
//
template <class T>
void allocate_host(T **pp, const int len) {
  allocate_host_T((void **)pp, len, sizeof(T));
}


void allocate_device_T(void **pp, const int len, const size_t sizeofT);
//----------------------------------------------------------------------------------------
//
// Allocate gpu memory
// pp = memory pointer
// len = length of the array
//
template <class T>
void allocate_device(T **pp, const int len) {
  allocate_device_T((void **)pp, len, sizeof(T));
}


void deallocate_device_T(void **pp);
//----------------------------------------------------------------------------------------
//
// Deallocate gpu memory
// pp = memory pointer
//
template <class T>
void deallocate_device(T **pp) {
  deallocate_device_T((void **)pp);
}
//----------------------------------------------------------------------------------------

bool resize_device_T(void **pp, int *curlen, const int cur_size, const int new_size,
        const float fac, const size_t sizeofT);

//----------------------------------------------------------------------------------------
//
// Allocate & re-allocate GPU memory, preserves content
// pp = memory pointer
// curlen = current length of the array
// cur_size = current size of the data content
// new_size = new size
// fac = extra space allocation factor: in case of re-allocation new length will be fac*new_size
//
template <class T>
bool resize_device(T **pp, int *curlen, const int cur_size, const int new_size, const float fac) {
  return resize_device_T((void **)pp, curlen, cur_size, new_size, fac, sizeof(T));
}

bool reallocate_device_T(void **pp, int *curlen, const int newlen, const float fac, const size_t sizeofT);
//----------------------------------------------------------------------------------------
//
// Allocate & re-allocate device memory
// pp = memory pointer
// curlen = current length of the array
// newlen = new required length of the array
// fac = extra space allocation factor: in case of re-allocation new length will be fac*newlen
//
// returns true if reallocation happened
//
template <class T>
bool reallocate_device(T **pp, int *curlen, const int newlen, const float fac=1.0f) {
  return reallocate_device_T((void **)pp, curlen, newlen, fac, sizeof(T));
}
//----------------------------------------------------------------------------------------
bool reallocate_host_T(void **pp, int *curlen, const int newlen, const float fac, 
		       const unsigned int flag, const size_t sizeofT);
//----------------------------------------------------------------------------------------
//
// Allocate & re-allocate pinned host memory
// pp = memory pointer
// curlen = current length of the array
// newlen = new required length of the array
// fac = extra space allocation factor: in case of re-allocation new length will be fac*newlen
// flag = allocation type:
//        cudaHostAllocDefault = default type, emulates cudaMallocHost
//        cudaHostAllocMapped  = maps allocation into CUDA address space
//
// returns true if reallocation happened
//
template <class T>
bool reallocate_host(T **pp, int *curlen, const int newlen,
		     const float fac=1.0f, const unsigned int flag=cudaHostAllocDefault) {
  return reallocate_host_T((void **)pp, curlen, newlen, fac, flag, sizeof(T));
}

bool resize_host_T(void **pp, int *curlen, const int cur_size, const int new_size,
       const float fac, const unsigned int flag, const size_t sizeofT);

//----------------------------------------------------------------------------------------
//
// Allocate & re-allocate host memory, preserves content
// pp = memory pointer
// curlen = current length of the array
// cur_size = current size of the data content
// new_size = new size
// fac = extra space allocation factor: in case of re-allocation new length will be fac*new_size
// flag = allocation type:
//        cudaHostAllocDefault = default type, emulates cudaMallocHost
//        cudaHostAllocMapped  = maps allocation into CUDA address space
//
template <class T>
bool resize_host(T **pp, int *curlen, const int cur_size, const int new_size,
  const float fac=1.0f, const unsigned int flag=cudaHostAllocDefault) {
  return resize_host_T((void **)pp, curlen, cur_size, new_size, fac, flag, sizeof(T));
}

void deallocate_host_T(void **pp);
//----------------------------------------------------------------------------------------
//
// Deallocate page-locked host memory
// pp = memory pointer
//
template <class T>
void deallocate_host(T **pp) {
  deallocate_host_T((void **)pp);
}
//----------------------------------------------------------------------------------------

void copy_HtoD_async_T(const void *h_array, void *d_array, int array_len, cudaStream_t stream,
           const size_t sizeofT);
void copy_HtoD_T(const void *h_array, void *d_array, int array_len,
     const size_t sizeofT);
void copy_DtoH_async_T(const void *d_array, void *h_array, const int array_len, cudaStream_t stream,
           const size_t sizeofT);
void copy_DtoH_T(const void *d_array, void *h_array, const int array_len, const size_t sizeofT);

void copy_DtoD_async_T(const void *d_src, void *d_dst, const int array_len, cudaStream_t stream,
           const size_t sizeofT);
void copy_DtoD_T(const void *d_src, void *d_dst, const int array_len, const size_t sizeofT);

//----------------------------------------------------------------------------------------
//
// Copies memory Host -> Device
//
template <class T>
void copy_HtoD(const T *h_array, T *d_array, int array_len, cudaStream_t stream=0) {
  copy_HtoD_async_T(h_array, d_array, array_len, stream, sizeof(T));
}

//----------------------------------------------------------------------------------------
//
// Copies memory Host -> Device using synchronous calls
//
template <class T>
void copy_HtoD_sync(const T *h_array, T *d_array, int array_len) {
  copy_HtoD_T(h_array, d_array, array_len, sizeof(T));
}

//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Host
//
template <class T>
void copy_DtoH(const T *d_array, T *h_array, const int array_len, cudaStream_t stream=0) {
  copy_DtoH_async_T(d_array, h_array, array_len, stream, sizeof(T));
}
//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Host using synchronous calls
//
template <class T>
void copy_DtoH_sync(const T *d_array, T *h_array, const int array_len) {
  copy_DtoH_T(d_array, h_array, array_len, sizeof(T));
}
//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Device
//
template <class T>
void copy_DtoD(const T *d_src, T *h_dst, const int array_len, cudaStream_t stream=0) {
  copy_DtoD_async_T(d_src, h_dst, array_len, stream, sizeof(T));
}
//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Device using synchronous calls
//
template <class T>
void copy_DtoD_sync(const T *d_src, T *h_dst, const int array_len) {
  copy_DtoD_T(d_src, h_dst, array_len, sizeof(T));
}

//----------------------------------------------------------------------------------------
//
// Copies memory between two peer devices Device -> Device
//
void copy_PeerDtoD_async_T(const int src_dev, const int dst_dev,
  const void *d_src, void *d_dst, const int array_len, cudaStream_t stream,
  const size_t sizeofT);

template <class T>
void copy_PeerDtoD(const int src_dev, const int dst_dev,
  const T *d_src, T *d_dst, const int array_len, cudaStream_t stream=0) {
  copy_PeerDtoD_async_T(src_dev, dst_dev, d_src, d_dst, array_len, stream, sizeof(T));
}

//----------------------------------------------------------------------------------------
//
// Copies 3D memory block Host -> Device
//
void copy3D_HtoD_T(void* src_data, void* dst_data,
  int src_x0, int src_y0, int src_z0,
  size_t src_xsize, size_t src_ysize,
  int dst_x0, int dst_y0, int dst_z0,
  size_t dst_xsize, size_t dst_ysize,
  size_t width, size_t height, size_t depth,
  size_t sizeofT, cudaStream_t stream);

template <class T>
void copy3D_HtoD(T* src_data, T* dst_data,
  int src_x0, int src_y0, int src_z0,
  size_t src_xsize, size_t src_ysize,
  int dst_x0, int dst_y0, int dst_z0,
  size_t dst_xsize, size_t dst_ysize,
  size_t width, size_t height, size_t depth,
  cudaStream_t stream=0) {
  copy3D_HtoD_T(src_data, dst_data,
    src_x0, src_y0, src_z0,
    src_xsize, src_ysize,
    dst_x0, dst_y0, dst_z0,
    dst_xsize, dst_ysize,
    width, height, depth,
    sizeof(T), stream);
}

//----------------------------------------------------------------------------------------
//
// Copies 3D memory block Device -> Host
//
void copy3D_DtoH_T(void* src_data, void* dst_data,
  int src_x0, int src_y0, int src_z0,
  size_t src_xsize, size_t src_ysize,
  int dst_x0, int dst_y0, int dst_z0,
  size_t dst_xsize, size_t dst_ysize,
  size_t width, size_t height, size_t depth,
  size_t sizeofT, cudaStream_t stream);

template <class T>
void copy3D_DtoH(T* src_data, T* dst_data,
  int src_x0, int src_y0, int src_z0,
  size_t src_xsize, size_t src_ysize,
  int dst_x0, int dst_y0, int dst_z0,
  size_t dst_xsize, size_t dst_ysize,
  size_t width, size_t height, size_t depth,
  cudaStream_t stream=0) {
  copy3D_DtoH_T(src_data, dst_data,
    src_x0, src_y0, src_z0,
    src_xsize, src_ysize,
    dst_x0, dst_y0, dst_z0,
    dst_xsize, dst_ysize,
    width, height, depth,
    sizeof(T), stream);
}

//----------------------------------------------------------------------------------------
//
// Copies 3D memory block Device -> Device
//
void copy3D_DtoD_T(void* src_data, void* dst_data,
  int src_x0, int src_y0, int src_z0,
  size_t src_xsize, size_t src_ysize,
  int dst_x0, int dst_y0, int dst_z0,
  size_t dst_xsize, size_t dst_ysize,
  size_t width, size_t height, size_t depth,
  size_t sizeofT, cudaStream_t stream);

template <class T>
void copy3D_DtoD(T* src_data, T* dst_data,
  int src_x0, int src_y0, int src_z0,
  size_t src_xsize, size_t src_ysize,
  int dst_x0, int dst_y0, int dst_z0,
  size_t dst_xsize, size_t dst_ysize,
  size_t width, size_t height, size_t depth,
  cudaStream_t stream=0) {
  copy3D_DtoD_T(src_data, dst_data,
    src_x0, src_y0, src_z0,
    src_xsize, src_ysize,
    dst_x0, dst_y0, dst_z0,
    dst_xsize, dst_ysize,
    width, height, depth,
    sizeof(T), stream);
}

//----------------------------------------------------------------------------------------
//
// Copies 3D memory block between two peer devices Device -> Device
//
void copy3D_PeerDtoD_T(int src_dev, int dst_dev,
  void* src_data, void* dst_data,
  int src_x0, int src_y0, int src_z0,
  size_t src_xsize, size_t src_ysize,
  int dst_x0, int dst_y0, int dst_z0,
  size_t dst_xsize, size_t dst_ysize,
  size_t width, size_t height, size_t depth,
  size_t sizeofT, cudaStream_t stream);

template <class T>
void copy3D_PeerDtoD(int src_dev, int dst_dev,
  T* src_data, T* dst_data,
  int src_x0, int src_y0, int src_z0,
  size_t src_xsize, size_t src_ysize,
  int dst_x0, int dst_y0, int dst_z0,
  size_t dst_xsize, size_t dst_ysize,
  size_t width, size_t height, size_t depth,
  cudaStream_t stream=0) {
  copy3D_PeerDtoD_T(src_dev, dst_dev,
    src_data, dst_data,
    src_x0, src_y0, src_z0,
    src_xsize, src_ysize,
    dst_x0, dst_y0, dst_z0,
    dst_xsize, dst_ysize,
    width, height, depth,
    sizeof(T), stream);
}

#endif // CUDAUTILS_H