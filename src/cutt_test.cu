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
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include "cutt.h"
#include "CudaUtils.h"
#include "TensorTester.h"

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

long long int* dataIn  = NULL;
long long int* dataOut = NULL;
int dataSize  = 100000000;
TensorTester* tester = NULL;

bool test1();
bool test2();
bool test3();
template <typename T> bool test_tensor(std::vector<int>& dim, std::vector<int>& permutation);

int main(int argc, char *argv[]) {

  int gpuid = -1;
  bool arg_ok = true;
  if (argc >= 3) {
    if (strcmp(argv[1], "-device") == 0) {
      sscanf(argv[2], "%d", &gpuid);
    } else {
      arg_ok = false;
    }
  } else if (argc > 1) {
    arg_ok = false;
  }

  if (!arg_ok) {
    printf("cutt_test [options]\n");
    printf("Options:\n");
    printf("-device gpuid : use GPU with ID gpuid\n");
    return 1;
  }

  if (gpuid >= 0) {
    cudaCheck(cudaSetDevice(gpuid));
  }

  cudaCheck(cudaDeviceReset());
  cudaCheck(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

  // Allocate device data, 100M elements
  allocate_device<long long int>(&dataIn, dataSize);
  allocate_device<long long int>(&dataOut, dataSize);

  // Create tester
  tester = new TensorTester();
  tester->setTensorCheckPattern((unsigned int *)dataIn, dataSize*2);

  // if (!test1()) goto fail;
  // if (!test2()) goto fail;
  if (!test3()) goto fail;

  printf("test OK\n");
  goto end;
fail:
  printf("test FAIL\n");
end:
  deallocate_device<long long int>(&dataIn);
  deallocate_device<long long int>(&dataOut);
  delete tester;

  cudaCheck(cudaDeviceReset());
  return 0;
}

//
// Test 1: Test all permutations up to rank 7 on smallish tensors
//
bool test1() {
  const int minDim = 2;
  const int maxDim = 16;
  for (int rank = 2;rank <= 7;rank++) {

    std::vector<int> dim(rank);
    std::vector<int> permutation(rank);
    for (int r=0;r < rank;r++) {
      permutation[r] = r;
      dim[r] = minDim + r*(maxDim - minDim)/rank;
    }

    do {
      if (!test_tensor<long long int>(dim, permutation)) return false;
      if (!test_tensor<int>(dim, permutation)) return false;
    } while (std::next_permutation(permutation.begin(), permutation.begin() + rank));

  }

  return true;
}

//
// Test 2: Test ranks 2-15, random size, random permutation, random dimensions
//         100 samples each rank
//
bool test2() {
  const int minDim = 2;
  const int maxDim = 256;

  std::srand(unsigned (std::time(0)));

  for (int rank = 2;rank <= 15;rank++) {

    for (int isample=0;isample < 100;isample++) {
      long long int volmin = minDim << rank;
      long long int volmax = dataSize;
      // int vol = (int)(volmin + (volmax - volmin)*((long long int)rand())/((long long int)RAND_MAX) );

      std::vector<int> dim(rank);
      std::vector<int> permutation(rank);
      for (int r=0;r < rank;r++) permutation[r] = r;
      double vol = 1.0;
      double curvol = 1.0;
      int iter = 0;
      do {
        vol = (volmin + (volmax - volmin)*((double)rand())/((double)RAND_MAX) );

        int subiter = 0;
        do {
          for (int r=0;r < rank;r++) {
            dim[r] = minDim + ((long long int)(maxDim - minDim))*((long long int)rand())/((long long int)RAND_MAX);
            curvol *= (double)dim[r];
          }

          double vol_scale = pow(vol/curvol, 1.0/(double)rank);
          curvol = 1.0;
          for (int r=0;r < rank;r++) {
            dim[r] = max(2, (int)(dim[r]*vol_scale));
            curvol *= dim[r];
          }
        } while (subiter < 50 && (curvol > volmax || fabs(curvol-vol)/(double)vol > 0.3));

        // printf("vol %d curvol %lf\n", vol, curvol);
        // printf("dim");
        // for (int r=0;r < rank;r++) printf(" %d", dim[r]);
        // printf("\n");

        iter++;
        if (iter == 1000) {
          printf("vol %lf\n", vol);
          printf("Unable to determine dimensions in 1000 iterations\n");
          return false;
        }
      } while (curvol > volmax || fabs(curvol-vol)/(double)vol > 0.3);

      std::random_shuffle(permutation.begin(), permutation.end());

      if (!test_tensor<long long int>(dim, permutation)) return false;
      if (!test_tensor<int>(dim, permutation)) return false;
    }

  }

  return true;
}

//
// Test 3: hand picked examples
//
bool test3() {

  {
    int rank = 2;
    std::vector<int> dim(rank);
    std::vector<int> permutation(rank);
    dim[0] = 43;
    dim[1] = 67;
    permutation[0] = 1;
    permutation[1] = 0;
    if (!test_tensor<long long int>(dim, permutation)) return false;
    if (!test_tensor<int>(dim, permutation)) return false;
    dim[0] = 65536*32;
    dim[1] = 2;
    permutation[0] = 1;
    permutation[1] = 0;
    if (!test_tensor<long long int>(dim, permutation)) return false;
    if (!test_tensor<int>(dim, permutation)) return false;
  }

  {
    int rank = 3;
    std::vector<int> dim(rank);
    std::vector<int> permutation(rank);
    dim[0] = 1305;
    dim[1] = 599;
    dim[2] = 88;
    permutation[0] = 0;
    permutation[1] = 2;
    permutation[2] = 1;
    if (!test_tensor<long long int>(dim, permutation)) return false;
    if (!test_tensor<int>(dim, permutation)) return false;
  }

  {
    int rank = 4;
    std::vector<int> dim(rank);
    std::vector<int> permutation(rank);
    dim[0] = 24;
    dim[1] = 330;
    dim[2] = 64;
    dim[3] = 147;
    permutation[0] = 1;
    permutation[1] = 0;
    permutation[2] = 2;
    permutation[3] = 3;
    if (!test_tensor<long long int>(dim, permutation)) return false;
    if (!test_tensor<int>(dim, permutation)) return false;
  }

  {
    int rank = 4;
    std::vector<int> dim(rank);
    std::vector<int> permutation(rank);
    dim[0] = 2;
    dim[1] = 5;
    dim[2] = 9;
    dim[3] = 12;
    permutation[0] = 0;
    permutation[1] = 1;
    permutation[2] = 2;
    permutation[3] = 3;
    if (!test_tensor<long long int>(dim, permutation)) return false;
    if (!test_tensor<int>(dim, permutation)) return false;    
  }

  {
    int rank = 6;
    std::vector<int> dim(rank);
    std::vector<int> permutation(rank);
    dim[0] = 2;
    dim[1] = 4;
    dim[2] = 6;
    dim[3] = 9;
    dim[4] = 11;
    dim[5] = 13;
    permutation[0] = 0;
    permutation[1] = 1;
    permutation[2] = 2;
    permutation[3] = 3;
    permutation[4] = 4;
    permutation[5] = 5;
    if (!test_tensor<long long int>(dim, permutation)) return false;
    if (!test_tensor<int>(dim, permutation)) return false;    
  }

  return true;
}

template <typename T>
bool test_tensor(std::vector<int>& dim, std::vector<int>& permutation) {

  int rank = dim.size();

  int vol = 1;
  for (int r=0;r < rank;r++) {
    vol *= dim[r];
  }

  size_t volmem = vol*sizeof(T);
  size_t datamem = dataSize*sizeof(long long int);
  if (volmem > datamem) {
    printf("test_tensor, data size exceeded\n");
    return false;
  }

  printf("number of elements %d\n", vol);
  printf("dimensions\n");
  for (int r=0;r < rank;r++) {
    printf("%d ", dim[r]);
  }
  printf("\n");
  printf("permutation\n");
  for (int r=0;r < rank;r++) {
    printf("%d%c",permutation[r]+1, (r==rank-1) ? ' ' : '-');
  }
  printf("\n");

  cuttHandle plan;
  cuttCheck(cuttPlan(&plan, rank, dim.data(), permutation.data(), sizeof(T)));
  clear_device_array<T>((T *)dataOut, vol);
  cuttCheck(cuttExecute(plan, dataIn, dataOut));
  cuttCheck(cuttDestroy(plan));

  return tester->checkTranspose<T>(rank, dim.data(), permutation.data(), (T *)dataOut);
}
