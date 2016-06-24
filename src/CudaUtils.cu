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

