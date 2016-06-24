cuTT - CUDA Tensor Transpose

cuTT is a high performance tensor transpose library for NVIDIA GPUs. It works with Kepler (SM 3.0) and above GPUs.

version 1.0

(c) Antti-Pekka Hynninen, 2016

----------------
# Installation #
----------------

Software requirements:
-C++ compiler with C++11 compitability
-CUDA compiler

Hardware requirements:
-Kepler (SM 3.0) or above NVIDIA GPU

To compile cuTT library as well as test cases and benchmarks, simply do

make

This will create the library itself:

include/cutt.h
lib/libcutt.a

as well as the test and benchmarks

bin/cutt_test
bin/cutt_bench

---------
# Usage #
---------

cuTT uses a "plan structure" similar to FFTW and cuFFT libraries, where the
user first creates a plan for the transpose and then executes that plan.
Here is an example code.

-----------------------------------------------------------------------------------
#include <cutt.h>

//
// Error checking wrapper for cutt
//
#define cuttCheck(stmt) do {                                 \
  cuttResult err = stmt;                            \
  if (err != CUTT_SUCCESS) {                          \
    fprintf(stderr, "%s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
    exit(1); \
  }                                                  \
} while(0)

int main() {

  // Four dimensional tensor
  int dim[4] = {31, 549, 2, 3};
  int permutation[4] = {3, 0, 2, 1};

  .... input and output data is setup here ...
  // double* idata : size product(dim)
  // double* odata : size product(dim)

  // Create plan
  cuttHandle plan;
  cuttCheck(cuttPlan(&plan, 4, dim, permutation, 8));

  // Execute plan
  cuttCheck(cuttExecute(plan, idata, odata));

  ... do stuff with your output and deallocate data ...

  // Destroy plan
  cuttCheck(cuttDestroy(plan));

  return 0;
}
-----------------------------------------------------------------------------------

Input (idata) and output (odata) data are both in GPU memory and must point to different
memory areas for correct operation. That is, cuTT only currently supports out-of-place
transposes.

------------
# cuTT API #
------------

//
// Create plan
//
// Parameters
// handle            = Returned handle to cuTT plan
// rank              = Rank of the tensor
// dim[rank]         = Dimensions of the tensor
// permutation[rank] = Transpose permutation
// sizeofType        = Size of the elements of the tensor in bytes (=4 or 8)
//
// Returns
// Success/unsuccess code
// 
cuttResult cuttPlan(cuttHandle* handle, int rank, int* dim, int* permutation, size_t sizeofType);

//
// Destroy plan
//
// Parameters
// handle            = Handle to the cuTT plan
// 
// Returns
// Success/unsuccess code
//
cuttResult cuttDestroy(cuttHandle handle);

//
// Associate CUDA stream with plan
//
// Parameters
// handle            = Handle to the cuTT plan
// stream            = CUDA stream
// 
// Returns
// Success/unsuccess code
//
cuttResult cuttSetStream(cuttHandle handle, cudaStream_t stream);

//
// Execute plan out-of-place
//
// Parameters
// handle            = Returned handle to cuTT plan
// idata             = Input data size product(dim)
// odata             = Output data size product(dim)
// 
// Returns
// Success/unsuccess code
//
cuttResult cuttExecute(cuttHandle handle, void* idata, void* odata);

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
