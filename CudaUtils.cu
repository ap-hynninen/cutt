
#include <stdio.h>
#include "CudaUtils.h"

//----------------------------------------------------------------------------------------

void clear_device_array_async_T(void *data, const int ndata, cudaStream_t stream, const size_t sizeofT) {
  cudaCheck(cudaMemsetAsync(data, 0, sizeofT*ndata, stream));
}

void clear_device_array_T(void *data, const int ndata, const size_t sizeofT) {
  cudaCheck(cudaMemset(data, 0, sizeofT*ndata));
}

//----------------------------------------------------------------------------------------
//
// Allocate page-locked host memory
// pp = memory pointer
// len = length of the array
//
void allocate_host_T(void **pp, const int len, const size_t sizeofT) {
  cudaCheck(cudaMallocHost(pp, sizeofT*len));
}

//----------------------------------------------------------------------------------------
//
// Allocate gpu memory
// pp = memory pointer
// len = length of the array
//
void allocate_device_T(void **pp, const int len, const size_t sizeofT) {
  cudaCheck(cudaMalloc(pp, sizeofT*len));
}

//----------------------------------------------------------------------------------------
//
// Deallocate gpu memory
// pp = memory pointer
//
void deallocate_device_T(void **pp) {
  
  if (*pp != NULL) {
    cudaCheck(cudaFree((void *)(*pp)));
    *pp = NULL;
  }

}

//----------------------------------------------------------------------------------------
//
// Deallocate page-locked host memory
// pp = memory pointer
//
void deallocate_host_T(void **pp) {
  
  if (*pp != NULL) {
    cudaCheck(cudaFreeHost((void *)(*pp)));
    *pp = NULL;
  }

}

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
bool reallocate_device_T(void **pp, int *curlen, const int newlen, const float fac, const size_t sizeofT) {

  if (*pp != NULL && *curlen < newlen) {
    cudaCheck(cudaFree((void *)(*pp)));
    *pp = NULL;
  }

  if (*pp == NULL) {
    if (fac > 1.0f) {
      *curlen = (int)(((double)(newlen))*(double)fac);
    } else {
      *curlen = newlen;
    }
    cudaCheck(cudaMalloc(pp, sizeofT*(*curlen)));
    return true;
  }

  return false;
}

//----------------------------------------------------------------------------------------
//
// Allocate & re-allocate page-locked host memory, preserves content
//
bool resize_device_T(void **pp, int *curlen, const int cur_size, const int new_size,
        const float fac, const size_t sizeofT) {

  void *old = NULL;  

  if (*pp != NULL && *curlen < new_size) {
    allocate_device_T(&old, cur_size, sizeofT);
    copy_DtoD_T(*pp, old, cur_size, sizeofT);
    cudaCheck(cudaDeviceSynchronize());       //Make sure D-D copy is done
    cudaCheck(cudaFree((void *)(*pp)));
    *pp = NULL;
  }

  if (*pp == NULL) {
    if (fac > 1.0f) {
      *curlen = (int)(((double)(new_size))*(double)fac);
    } else {
      *curlen = new_size;
    }
    allocate_device_T(pp, *curlen, sizeofT);
    if (old != NULL) {
      copy_DtoD_T(old, *pp, cur_size, sizeofT);
      cudaCheck(cudaDeviceSynchronize());       //Make sure D-D copy is done
      deallocate_device_T(&old);
    }
    return true;
  }

  return false;
}

//----------------------------------------------------------------------------------------
//
// Allocate & re-allocate page-locked host memory
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
bool reallocate_host_T(void **pp, int *curlen, const int newlen, 
		       const float fac, const unsigned int flag, const size_t sizeofT) {

  if (*pp != NULL && *curlen < newlen) {
    cudaCheck(cudaFreeHost((void *)(*pp)));
    *pp = NULL;
  }

  if (*pp == NULL) {
    if (fac > 1.0f) {
      *curlen = (int)(((double)(newlen))*(double)fac);
    } else {
      *curlen = newlen;
    }
    cudaCheck(cudaHostAlloc(pp, sizeofT*(*curlen), flag));
    return true;
  }

  return false;
}

//----------------------------------------------------------------------------------------
//
// Allocate & re-allocate page-locked host memory, preserves content
//
bool resize_host_T(void **pp, int *curlen, const int cur_size, const int new_size,
       const float fac, const unsigned int flag, const size_t sizeofT) {

  char *old = NULL;

  if (*pp != NULL && *curlen < new_size) {
    old = new char[cur_size*sizeofT];
    memcpy(old, *pp, cur_size*sizeofT);
    cudaCheck(cudaFreeHost((void *)(*pp)));
    *pp = NULL;
  }

  if (*pp == NULL) {
    if (fac > 1.0f) {
      *curlen = (int)(((double)(new_size))*(double)fac);
    } else {
      *curlen = new_size;
    }
    cudaCheck(cudaHostAlloc(pp, sizeofT*(*curlen), flag));
    if (old != NULL) {
      memcpy(*pp, old, cur_size*sizeofT);
      delete [] old;
    }
    return true;
  }

  return false;
}

//----------------------------------------------------------------------------------------
//
// Copies memory Host -> Device
//
void copy_HtoD_async_T(const void *h_array, void *d_array, int array_len, cudaStream_t stream,
           const size_t sizeofT) {
  cudaCheck(cudaMemcpyAsync(d_array, h_array, sizeofT*array_len, cudaMemcpyHostToDevice, stream));
}

void copy_HtoD_T(const void *h_array, void *d_array, int array_len,
     const size_t sizeofT) {
  cudaCheck(cudaMemcpy(d_array, h_array, sizeofT*array_len, cudaMemcpyHostToDevice));
}

//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Host
//
void copy_DtoH_async_T(const void *d_array, void *h_array, const int array_len, cudaStream_t stream,
           const size_t sizeofT) {
  cudaCheck(cudaMemcpyAsync(h_array, d_array, sizeofT*array_len, cudaMemcpyDeviceToHost, stream));
}

void copy_DtoH_T(const void *d_array, void *h_array, const int array_len, const size_t sizeofT) {
  cudaCheck(cudaMemcpy(h_array, d_array, sizeofT*array_len, cudaMemcpyDeviceToHost));
}

//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Device
//
void copy_DtoD_async_T(const void *d_src, void *d_dst, const int array_len, cudaStream_t stream,
           const size_t sizeofT) {
  cudaCheck(cudaMemcpyAsync(d_dst, d_src, sizeofT*array_len, cudaMemcpyDeviceToDevice, stream));
}

void copy_DtoD_T(const void *d_src, void *d_dst, const int array_len, const size_t sizeofT) {
  cudaCheck(cudaMemcpy(d_dst, d_src, sizeofT*array_len, cudaMemcpyDeviceToDevice));
}

//----------------------------------------------------------------------------------------
//
// Copies memory between two devices Device -> Device
//
void copy_PeerDtoD_async_T(const int src_dev, const int dst_dev,
  const void *d_src, void *d_dst, const int array_len, cudaStream_t stream,
  const size_t sizeofT) {
  cudaCheck(cudaMemcpyPeerAsync(d_dst, dst_dev, d_src, src_dev, sizeofT*array_len, stream));
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
  size_t sizeofT, cudaStream_t stream) {
  cudaMemcpy3DParms parms = {0};

  parms.srcPos = make_cudaPos(sizeofT*src_x0, src_y0, src_z0);
  parms.srcPtr = make_cudaPitchedPtr(src_data, sizeofT*src_xsize, src_xsize, src_ysize);

  parms.dstPos = make_cudaPos(sizeofT*dst_x0, dst_y0, dst_z0);
  parms.dstPtr = make_cudaPitchedPtr(dst_data, sizeofT*dst_xsize, dst_xsize, dst_ysize);

  parms.extent = make_cudaExtent(sizeofT*width, height, depth);
  parms.kind = cudaMemcpyHostToDevice;

  cudaCheck(cudaMemcpy3DAsync(&parms, stream));
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
  size_t sizeofT, cudaStream_t stream) {
  cudaMemcpy3DParms parms = {0};

  parms.srcPos = make_cudaPos(sizeofT*src_x0, src_y0, src_z0);
  parms.srcPtr = make_cudaPitchedPtr(src_data, sizeofT*src_xsize, src_xsize, src_ysize);

  parms.dstPos = make_cudaPos(sizeofT*dst_x0, dst_y0, dst_z0);
  parms.dstPtr = make_cudaPitchedPtr(dst_data, sizeofT*dst_xsize, dst_xsize, dst_ysize);

  parms.extent = make_cudaExtent(sizeofT*width, height, depth);
  parms.kind = cudaMemcpyDeviceToHost;

  cudaCheck(cudaMemcpy3DAsync(&parms, stream));
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
  size_t sizeofT, cudaStream_t stream) {
  cudaMemcpy3DParms parms = {0};

  parms.srcPos = make_cudaPos(sizeofT*src_x0, src_y0, src_z0);
  parms.srcPtr = make_cudaPitchedPtr(src_data, sizeofT*src_xsize, src_xsize, src_ysize);

  parms.dstPos = make_cudaPos(sizeofT*dst_x0, dst_y0, dst_z0);
  parms.dstPtr = make_cudaPitchedPtr(dst_data, sizeofT*dst_xsize, dst_xsize, dst_ysize);

  parms.extent = make_cudaExtent(sizeofT*width, height, depth);
  parms.kind = cudaMemcpyDeviceToDevice;

  cudaCheck(cudaMemcpy3DAsync(&parms, stream));
}

//----------------------------------------------------------------------------------------
//
// Copies 3D memory block between devices Device -> Device
//
void copy3D_PeerDtoD_T(int src_dev, int dst_dev,
  void* src_data, void* dst_data,
  int src_x0, int src_y0, int src_z0,
  size_t src_xsize, size_t src_ysize,
  int dst_x0, int dst_y0, int dst_z0,
  size_t dst_xsize, size_t dst_ysize,
  size_t width, size_t height, size_t depth,
  size_t sizeofT, cudaStream_t stream) {
  cudaMemcpy3DPeerParms parms = {0};

  parms.srcDevice = src_dev;
  parms.dstDevice = dst_dev;

  parms.srcPos = make_cudaPos(sizeofT*src_x0, src_y0, src_z0);
  parms.srcPtr = make_cudaPitchedPtr(src_data, sizeofT*src_xsize, src_xsize, src_ysize);

  parms.dstPos = make_cudaPos(sizeofT*dst_x0, dst_y0, dst_z0);
  parms.dstPtr = make_cudaPitchedPtr(dst_data, sizeofT*dst_xsize, dst_xsize, dst_ysize);

  parms.extent = make_cudaExtent(sizeofT*width, height, depth);

  cudaCheck(cudaMemcpy3DPeerAsync(&parms, stream));
}
