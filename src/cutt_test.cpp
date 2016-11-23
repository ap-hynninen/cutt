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
#include <ctime>           // std::time
#include <cstring>         // strcmp
#include <cmath>
#include "cutt.h"
#include "CudaUtils.h"
#include "TensorTester.h"
#include "cuttTimer.h"
#include "cuttGpuModel.h"  // testCounters

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

cuttTimer* timerFloat;
cuttTimer* timerDouble;

long long int* dataIn  = NULL;
long long int* dataOut = NULL;
int dataSize  = 200000000;
TensorTester* tester = NULL;

bool test1();
bool test2();
bool test3();
bool test4();
bool test5();
template <typename T> bool test_tensor(std::vector<int>& dim, std::vector<int>& permutation);
void printVec(std::vector<int>& vec);

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

  timerFloat = new cuttTimer(4);
  timerDouble = new cuttTimer(8);

  // Allocate device data, 100M elements
  allocate_device<long long int>(&dataIn, dataSize);
  allocate_device<long long int>(&dataOut, dataSize);

  // Create tester
  tester = new TensorTester();
  tester->setTensorCheckPattern((unsigned int *)dataIn, dataSize*2);

  if (!test1()) goto fail;
  if (!test2()) goto fail;
  if (!test3()) goto fail;
  if (!test4()) goto fail;
  if (!test5()) goto fail;

  {
    std::vector<int> worstDim;
    std::vector<int> worstPermutation;
    double worstBW = timerDouble->getWorst(worstDim, worstPermutation);
    printf("worstBW %4.2lf GB/s\n", worstBW);
    printf("dim\n");
    printVec(worstDim);
    printf("permutation\n");
    printVec(worstPermutation);
  }

  printf("test OK\n");
  goto end;
fail:
  printf("test FAIL\n");
end:
  deallocate_device<long long int>(&dataIn);
  deallocate_device<long long int>(&dataOut);
  delete tester;

  delete timerFloat;
  delete timerDouble;

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
// Test 2: Test ranks 2-15, random volume, random permutation, random dimensions
//         100 samples each rank
//
bool test2() {
  double minDim = 2.0;

  std::srand(unsigned (std::time(0)));

  for (int rank = 2;rank <= 15;rank++) {
    double volmin = pow(minDim+1, rank);
    double volmax = (double)dataSize;

    for (int isample=0;isample < 100;isample++) {

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
            double vol_left = vol/(curvol*pow(minDim, (double)(rank-r)));
            double aveDim = pow(vol, 1.0/(double)rank);
            double dimSpread = (aveDim - minDim);
            // rn = -1 ... 1
            double rn = 2.0*(((double)rand())/((double)RAND_MAX) - 0.5);
            dim[r] = (int)(aveDim + dimSpread*rn);
            curvol *= (double)dim[r];
          }

          // printf("vol %lf curvol %lf\n", vol, curvol);
          // printf("dim");
          // for (int r=0;r < rank;r++) printf(" %d", dim[r]);
          // printf("\n");

          double vol_scale = pow(vol/curvol, 1.0/(double)rank);
          // printf("vol_scale %lf\n", vol_scale);
          curvol = 1.0;
          for (int r=0;r < rank;r++) {
            dim[r] = std::max(2, (int)round((double)dim[r]*vol_scale));
            curvol *= dim[r];
          }

          // printf("vol %lf curvol %lf\n", vol, curvol);
          // printf("dim");
          // for (int r=0;r < rank;r++) printf(" %d", dim[r]);
          // printf("\n");
          // return false;

          subiter++;
        } while (subiter < 50 && (curvol > volmax || fabs(curvol-vol)/(double)vol > 2.3));

        // printf("vol %lf curvol %lf volmin %lf volmax %lf\n", vol, curvol, volmin, volmax);
        // printf("dim");
        // for (int r=0;r < rank;r++) printf(" %d", dim[r]);
        // printf("\n");

        iter++;
        if (iter == 1000) {
          printf("vol %lf\n", vol);
          printf("Unable to determine dimensions in 1000 iterations\n");
          return false;
        }
      } while (curvol > volmax || fabs(curvol-vol)/(double)vol > 2.3);

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

  {
    std::vector<int> dim(5);
    std::vector<int> permutation(5);
    dim[0] = 5;
    dim[1] = 42;
    dim[2] = 75;
    dim[3] = 86;
    dim[4] = 57;
    permutation[0] = 2 - 1;
    permutation[1] = 4 - 1;
    permutation[2] = 5 - 1;
    permutation[3] = 3 - 1;
    permutation[4] = 1 - 1;
    if (!test_tensor<long long int>(dim, permutation)) return false;
    if (!test_tensor<int>(dim, permutation)) return false;        
  }

  {
    std::vector<int> dim(5);
    std::vector<int> permutation(5);
    dim[0] = 5;
    dim[1] = 3;
    dim[2] = 2;
    dim[3] = 9;
    dim[4] = 14;
    permutation[0] = 0;
    permutation[1] = 1;
    permutation[2] = 3;
    permutation[3] = 2;
    permutation[4] = 4;
    if (!test_tensor<long long int>(dim, permutation)) return false;
    if (!test_tensor<int>(dim, permutation)) return false;        
  }

  return true;
}

//
// Test 4: streaming
//
bool test4() {

  std::vector<int> dim = {24, 32, 16, 36, 43, 9};
  std::vector<int> permutation = {5, 1, 4, 2, 3, 0};

  const int numStream = 10;

  cudaStream_t streams[numStream];
  for (int i=0;i < numStream;i++) {
    cudaCheck(cudaStreamCreate(&streams[i]));
  }

  cudaCheck(cudaDeviceSynchronize());

  cuttHandle plans[numStream];

  for (int i=0;i < numStream;i++) {
    cuttCheck(cuttPlan(&plans[i], dim.size(), dim.data(), permutation.data(), sizeof(double), streams[i]));
    cuttCheck(cuttExecute(plans[i], dataIn, dataOut));
  }

  cudaCheck(cudaDeviceSynchronize());

  bool run_ok = tester->checkTranspose(dim.size(), dim.data(), permutation.data(), (long long int *)dataOut);

  cudaCheck(cudaDeviceSynchronize());

  for (int i=0;i < numStream;i++) {
    cuttCheck(cuttDestroy(plans[i]));
    cudaCheck(cudaStreamDestroy(streams[i]));
  }

  return run_ok;
}


//
// Test 5: Transaction and cache line counters
//
bool test5() {

  {
    // Number of elements that are loaded per memory transaction:
    // 128 bytes per transaction
    const  int accWidth = 128/sizeof(double);
    // L2 cache line width is 32 bytes
    const int cacheWidth = 32/sizeof(double);
    if (!testCounters(32, accWidth, cacheWidth)) return false;
  }

  {
    // Number of elements that are loaded per memory transaction:
    // 128 bytes per transaction
    const  int accWidth = 128/sizeof(float);
    // L2 cache line width is 32 bytes
    const int cacheWidth = 32/sizeof(float);
    if (!testCounters(32, accWidth, cacheWidth)) return false;
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
  printVec(dim);
  printf("permutation\n");
  printVec(permutation);

  cuttTimer* timer;
  if (sizeof(T) == 4) {
    timer = timerFloat;
  } else {
    timer = timerDouble;
  }

  cuttHandle plan;
  cuttCheck(cuttPlan(&plan, rank, dim.data(), permutation.data(), sizeof(T), 0));
  set_device_array<T>((T *)dataOut, -1, vol);
  cudaCheck(cudaDeviceSynchronize());

  if (vol > 1000000) timer->start(dim, permutation);
  cuttCheck(cuttExecute(plan, dataIn, dataOut));
  if (vol > 1000000) timer->stop();

  cuttCheck(cuttDestroy(plan));

  return tester->checkTranspose<T>(rank, dim.data(), permutation.data(), (T *)dataOut);
}

void printVec(std::vector<int>& vec) {
  for (int i=0;i < vec.size();i++) {
    printf("%d ", vec[i]);
  }
  printf("\n");
}

