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
#include <cstring>         // strcmp
#include <cmath>
#include "cutt.h"
#include "CudaUtils.h"
#include "TensorTester.h"
#include "cuttTimer.h"
#include "CudaMemcpy.h"

#define MILLION 1000000
#define BILLION 1000000000

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
int dataSize  = 250*MILLION;
TensorTester* tester = NULL;

cuttTimer timerFloat(4);
cuttTimer timerDouble(8);

bool bench1(int numElem);
bool bench2(int numElem);
bool bench3(int numElem);
bool bench4();
bool bench_memcpy(int numElem);

void getRandomDim(double vol, std::vector<int>& dim);
template <typename T> bool bench_tensor(std::vector<int>& dim, std::vector<int>& permutation);
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
    printf("cutt_bench [options]\n");
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

  std::vector<int> worstDim;
  std::vector<int> worstPermutation;

  unsigned seed = unsigned (std::time(0));
  std::srand(seed);

  // if (!bench1(40*MILLION, bandwidths)) goto fail;
  // printf("bench1:\n");
  // for (int i=0;i < bandwidths.size();i++) {
  //   printf("%lf\n", bandwidths[i]);
  // }

  // if (!bench2(40*MILLION, bandwidths)) goto fail;
  // printf("bench2:\n");
  // for (int i=0;i < bandwidths.size();i++) {
  //   printf("%lf\n", bandwidths[i]);
  // }

  if (bench3(200*MILLION)) {
    printf("bench3:\n");
    printf("rank <BW>  worst  best\n");
    for (auto it=timerDouble.ranksBegin();it != timerDouble.ranksEnd();it++) {
      double worstBW = timerDouble.getWorst(*it);
      double bestBW = timerDouble.getBest(*it);
      double aveBW = timerDouble.getAverage(*it);
      printf("%2d %6.2lf %6.2lf %6.2lf\n", *it, aveBW, worstBW, bestBW);
    }
    double worstBW = timerDouble.getWorst(worstDim, worstPermutation);
    printf("worst of all %4.2lf rank %d\n", worstBW, worstDim.size());
    printf("dimensions\n");
    printVec(worstDim);
    printf("permutation\n");
    printVec(worstPermutation);
  } else {
    goto fail;
  }

  // if (!bench4()) goto fail;

  if (!bench_memcpy(200*MILLION)) goto fail;

  printf("bench OK\n");

  goto end;
fail:
  printf("bench FAIL\n");
end:
  deallocate_device<long long int>(&dataIn);
  deallocate_device<long long int>(&dataOut);
  delete tester;

  printf("seed %u\n", seed);

  cudaCheck(cudaDeviceReset());
  return 0;
}

//
// Benchmark 1: ranks 2-8,15 in inverse permutation. 32 start and end dimension
//
bool bench1(int numElem) {
  int ranks[8] = {2, 3, 4, 5, 6, 7, 8, 15};
  for (int i=0;i <= 7;i++) {
    std::vector<int> dim(ranks[i]);
    std::vector<int> permutation(ranks[i]);
    int dimave = (int)pow(numElem, 1.0/(double)ranks[i]);

    if (dimave < 100.0) {
      dim[0]            = 32;
      dim[ranks[i] - 1] = 32;
    } else {
      dim[0]            = dimave;
      dim[ranks[i] - 1] = dimave;
    }
    // Distribute remaining volume to the middle ranks
    int ranks_left = ranks[i] - 2;
    double numElem_left = numElem/(double)(dim[0]*dim[ranks[i] - 1]);
    for (int r=1;r < ranks[i] - 1;r++) {
      dim[r] = (int)pow(numElem_left, 1.0/(double)ranks_left);
      numElem_left /= (double)dim[r];
      ranks_left--;
    }

    // Inverse order
    for (int r=0;r < ranks[i];r++) {
      permutation[r] = ranks[i] - 1 - r;
    }

    if (!bench_tensor<long long int>(dim, permutation)) return false;
  }

  return true;
}

//
// Benchmark 2: ranks 2-8,15 in inverse permutation. Even spread of dimensions.
//
bool bench2(int numElem) {
  int ranks[8] = {2, 3, 4, 5, 6, 7, 8, 15};
  for (int i=0;i <= 7;i++) {
    std::vector<int> dim(ranks[i]);
    std::vector<int> permutation(ranks[i]);
    int dimave = (int)pow(numElem, 1.0/(double)ranks[i]);

    double numElem_left = numElem;
    for (int r=0;r < ranks[i];r++) {
      dim[r] = (int)pow(numElem_left, 1.0/(double)(ranks[i] - r));
      numElem_left /= (double)dim[r];
    }

    // Inverse order
    for (int r=0;r < ranks[i];r++) {
      permutation[r] = ranks[i] - 1 - r;
    }

    if (!bench_tensor<long long int>(dim, permutation)) return false;
  }

  return true;
}

//
// Benchmark 3: ranks 2-8,15 in random permutation and dimensions.
//
bool bench3(int numElem) {

  int ranks[8] = {2, 3, 4, 5, 6, 7, 8, 15};

  for (int i=0;i <= 7;i++) {
    std::vector<int> dim(ranks[i]);
    std::vector<int> permutation(ranks[i]);
    for (int r=0;r < ranks[i];r++) permutation[r] = r;
    for (int nsample=0;nsample < 10;nsample++) {
      std::random_shuffle(permutation.begin(), permutation.end());
      getRandomDim((double)numElem, dim);
      if (!bench_tensor<long long int>(dim, permutation)) return false;
    }
  }

  return true;
}

//
// Benchmark 4: specific examples
//
bool bench4() {
  {
    // rank 7 bandwidth 38.99
    // dimensions
    // 2 33 25 13 25 10 28 
    // permutation
    // 3-4-2-6-1-5-7 
    std::vector<int> dim(7);
    dim[0] = 2;
    dim[1] = 33;
    dim[2] = 25;
    dim[3] = 13;
    dim[4] = 25;
    dim[5] = 10;
    dim[6] = 28;
    std::vector<int> permutation(7);
    permutation[0] = 3 - 1;
    permutation[1] = 4 - 1;
    permutation[2] = 2 - 1;
    permutation[3] = 6 - 1;
    permutation[4] = 1 - 1;
    permutation[5] = 5 - 1;
    permutation[6] = 7 - 1;
    if (!bench_tensor<long long int>(dim, permutation)) return false;
    printf("dimensions\n");
    printVec(dim);
    printf("permutation\n");
    printVec(permutation);
    printf("bandwidth %4.2lf GB/s\n", timerDouble.GBs());
  }

  {
    std::vector<int> dim(2);
    std::vector<int> permutation(2);
    dim[0] = 65536*32;
    dim[1] = 2;
    permutation[0] = 1;
    permutation[1] = 0;
    if (!bench_tensor<long long int>(dim, permutation)) return false;
    printf("dimensions\n");
    printVec(dim);
    printf("permutation\n");
    printVec(permutation);
    printf("bandwidth %4.2lf GB/s\n", timerDouble.GBs());
    
    permutation[0] = 0;
    permutation[1] = 1;
    if (!bench_tensor<long long int>(dim, permutation)) return false;
    printf("dimensions\n");
    printVec(dim);
    printf("permutation\n");
    printVec(permutation);
    printf("bandwidth %4.2lf GB/s\n", timerDouble.GBs());
  }

  {
    // rank 7 bandwidth 17.47
    // dimensions
    // 8 17 24 6 18 20 20 
    // permutation
    // 1-4-2-7-3-5-6 
    std::vector<int> dim(7);
    dim[0] = 8;
    dim[1] = 17;
    dim[2] = 24;
    dim[3] = 6;
    dim[4] = 18;
    dim[5] = 20;
    dim[6] = 20;
    std::vector<int> permutation(7);
    permutation[0] = 1 - 1;
    permutation[1] = 4 - 1;
    permutation[2] = 2 - 1;
    permutation[3] = 7 - 1;
    permutation[4] = 3 - 1;
    permutation[5] = 5 - 1;
    permutation[6] = 6 - 1;
    if (!bench_tensor<long long int>(dim, permutation)) return false;
    printf("dimensions\n");
    printVec(dim);
    printf("permutation\n");
    printVec(permutation);
    printf("bandwidth %4.2lf GB/s\n", timerDouble.GBs());
  }

  {
    // rank 6 bandwidth 20.89
    // dimensions
    // 4 20 33 40 47 36 
    // permutation
    // 1-6-2-5-3-4 
    std::vector<int> dim(6);
    dim[0] = 4;
    dim[1] = 20;
    dim[2] = 33;
    dim[3] = 40;
    dim[4] = 47;
    dim[5] = 36;
    std::vector<int> permutation(6);
    permutation[0] = 1 - 1;
    permutation[1] = 6 - 1;
    permutation[2] = 2 - 1;
    permutation[3] = 5 - 1;
    permutation[4] = 3 - 1;
    permutation[5] = 4 - 1;
    if (!bench_tensor<long long int>(dim, permutation)) return false;
    printf("dimensions\n");
    printVec(dim);
    printf("permutation\n");
    printVec(permutation);
    printf("bandwidth %4.2lf GB/s\n", timerDouble.GBs());
  }

  {
    // rank 8 bandwidth 20.10
    // dimensions
    // 9 21 7 19 7 4 23 9 
    // permutation
    // 5-1-2-4-6-3-8-7 
    std::vector<int> dim(8);
    dim[0] = 9;
    dim[1] = 21;
    dim[2] = 7;
    dim[3] = 19;
    dim[4] = 7;
    dim[5] = 4;
    dim[6] = 23;
    dim[7] = 9;
    std::vector<int> permutation(8);
    permutation[0] = 5 - 1;
    permutation[1] = 1 - 1;
    permutation[2] = 2 - 1;
    permutation[3] = 4 - 1;
    permutation[4] = 6 - 1;
    permutation[5] = 3 - 1;
    permutation[6] = 8 - 1;
    permutation[7] = 7 - 1;
    if (!bench_tensor<long long int>(dim, permutation)) return false;
    printf("dimensions\n");
    printVec(dim);
    printf("permutation\n");
    printVec(permutation);
    printf("bandwidth %4.2lf GB/s\n", timerDouble.GBs());
  }

  {
    // rank 6 bandwidth 37.19
    // dimensions
    // 60 36 33 4 31 19 
    // permutation
    // 4-2-5-6-1-3 
    std::vector<int> dim(6);
    dim[0] = 60;
    dim[1] = 36;
    dim[2] = 33;
    dim[3] = 4;
    dim[4] = 31;
    dim[5] = 19;
    std::vector<int> permutation(6);
    permutation[0] = 4 - 1;
    permutation[1] = 2 - 1;
    permutation[2] = 5 - 1;
    permutation[3] = 6 - 1;
    permutation[4] = 1 - 1;
    permutation[5] = 3 - 1;
    if (!bench_tensor<long long int>(dim, permutation)) return false;
    printf("dimensions\n");
    printVec(dim);
    printf("permutation\n");
    printVec(permutation);
    printf("bandwidth %4.2lf GB/s\n", timerDouble.GBs());
  }

  {
    // rank 6 bandwidth 46.02
    // dimensions
    // 36 45 8 46 7 45 
    // permutation
    // 5-6-2-1-4-3 
    std::vector<int> dim(6);
    dim[0] = 36;
    dim[1] = 45;
    dim[2] = 8;
    dim[3] = 46;
    dim[4] = 7;
    dim[5] = 45;
    std::vector<int> permutation(6);
    permutation[0] = 5 - 1;
    permutation[1] = 6 - 1;
    permutation[2] = 2 - 1;
    permutation[3] = 1 - 1;
    permutation[4] = 4 - 1;
    permutation[5] = 3 - 1;
    if (!bench_tensor<long long int>(dim, permutation)) return false;
    printf("dimensions\n");
    printVec(dim);
    printf("permutation\n");
    printVec(permutation);
    printf("bandwidth %4.2lf GB/s\n", timerDouble.GBs());
  }

  {
    // rank 4 bandwidth 31.51
    // dimensions
    // 35 52 320 329 
    // permutation
    // 4-3-2-1 
    std::vector<int> dim(4);
    dim[0] = 35;
    dim[1] = 52;
    dim[2] = 320;
    dim[3] = 329;
    std::vector<int> permutation(4);
    permutation[0] = 4 - 1;
    permutation[1] = 3 - 1;
    permutation[2] = 2 - 1;
    permutation[3] = 1 - 1;
    if (!bench_tensor<long long int>(dim, permutation)) return false;
    printf("dimensions\n");
    printVec(dim);
    printf("permutation\n");
    printVec(permutation);
    printf("bandwidth %4.2lf GB/s\n", timerDouble.GBs());
  }

  {
    // rank 7 bandwidth 34.16
    // dimensions
    // 25 19 15 19 16 23 3 
    // permutation
    // 7-5-3-6-2-1-4 
    std::vector<int> dim(7);
    dim[0] = 25;
    dim[1] = 19;
    dim[2] = 15;
    dim[3] = 19;
    dim[4] = 16;
    dim[5] = 23;
    dim[6] = 3;
    std::vector<int> permutation(7);
    permutation[0] = 7 - 1;
    permutation[1] = 5 - 1;
    permutation[2] = 3 - 1;
    permutation[3] = 6 - 1;
    permutation[4] = 2 - 1;
    permutation[5] = 1 - 1;
    permutation[6] = 4 - 1;
    if (!bench_tensor<long long int>(dim, permutation)) return false;
    printf("dimensions\n");
    printVec(dim);
    printf("permutation\n");
    printVec(permutation);
    printf("bandwidth %4.2lf GB/s\n", timerDouble.GBs());
  }

  {
    // rank 7 bandwidth 33.98
    // dimensions
    // 3 17 24 24 10 22 22 
    // permutation
    // 1-7-4-6-3-2-5 
    std::vector<int> dim(7);
    dim[0] = 3;
    dim[1] = 17;
    dim[2] = 24;
    dim[3] = 24;
    dim[4] = 10;
    dim[5] = 22;
    dim[6] = 22;
    std::vector<int> permutation(7);
    permutation[0] = 1 - 1;
    permutation[1] = 7 - 1;
    permutation[2] = 4 - 1;
    permutation[3] = 6 - 1;
    permutation[4] = 3 - 1;
    permutation[5] = 2 - 1;
    permutation[6] = 5 - 1;
    if (!bench_tensor<long long int>(dim, permutation)) return false;
    printf("dimensions\n");
    printVec(dim);
    printf("permutation\n");
    printVec(permutation);
    printf("bandwidth %4.2lf GB/s\n", timerDouble.GBs());
  }

// rank 15 bandwidth 27.19
// dimensions
// 4 4 4 3 3 2 4 4 4 3 4 3 4 4 4 
// permutation
// 6-7-9-13-5-4-11-10-14-3-1-15-2-8-12 

// worst of all 59.62 rank 6
// dimensions
// 21 30 44 25 19 13 
// permutation
// 5 3 4 2 1 0 

  return true;
}

//
// Get random dimensions for a fixed volume tensor
//
void getRandomDim(double vol, std::vector<int>& dim) {
  double dimave = floor(pow(vol, 1.0/(double)dim.size()));
  double curvol = 1.0;
  int iter = 0;
  do {
    curvol = 1.0;
    for (int r=0;r < dim.size();r++) {
      // p is -1 ... 1
      double p = (((double)rand()/(double)RAND_MAX) - 0.5)*2.0;
      dim[r] = round(dimave + p*(dimave - 2.0));
      curvol *= (double)dim[r];
    }

    double vol_scale = pow(vol/curvol, 1.0/(double)dim.size());
    curvol = 1.0;
    for (int r=0;r < dim.size();r++) {
      dim[r] = std::max(2, (int)(dim[r]*vol_scale));
      curvol *= dim[r];
    }
    // printf("curvol %lf\n", curvol/MILLION);
    iter++;
  } while (iter < 5000 && (fabs(curvol-vol)/(double)vol > 0.3));

  if (iter == 5000) {
    printf("getRandomDim: Unable to determine dimensions in 5000 iterations\n");
    exit(1);
  }
}

template <typename T>
bool bench_tensor(std::vector<int>& dim, std::vector<int>& permutation) {

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

  cuttHandle plan;
  cuttCheck(cuttPlan(&plan, rank, dim.data(), permutation.data(), sizeof(T)));

  cuttTimer* timer;
  if (sizeof(T) == 4) {
    timer = &timerFloat;
  } else {
    timer = &timerDouble;
  }

  for (int i=0;i < 4;i++) {
    clear_device_array<T>((T *)dataOut, vol);
    cudaCheck(cudaDeviceSynchronize());

    timer->start(dim, permutation);

    cuttCheck(cuttExecute(plan, dataIn, dataOut));
    cudaCheck(cudaDeviceSynchronize());
  
    timer->stop();

    printf("wall time %lfms %lf GB/s\n", timer->seconds()*1000.0, timer->GBs());
  }

  cuttCheck(cuttDestroy(plan));
  return tester->checkTranspose<T>(rank, dim.data(), permutation.data(), (T *)dataOut);
}

void printVec(std::vector<int>& vec) {
  for (int i=0;i < vec.size();i++) {
    printf("%d ", vec[i]);
  }
  printf("\n");
}

//
// Benchmarks memory copy. Returns bandwidth in GB/s
//
bool bench_memcpy(int numElem) {

  std::vector<int> dim(1, numElem);
  std::vector<int> permutation(1, 0);

  {
    cuttTimer timer(8);
    for (int i=0;i < 4;i++) {
      clear_device_array<double>((double *)dataOut, numElem);
      cudaCheck(cudaDeviceSynchronize());
      timer.start(dim, permutation);
      scalarCopy<double>(numElem, (double *)dataIn, (double *)dataOut, 0);
      cudaCheck(cudaDeviceSynchronize());
      timer.stop();
      // printf("%4.2lf GB/s\n", timer.GBs());
    }
    if (!tester->checkTranspose<long long int>(1, dim.data(), permutation.data(), dataOut)) return false;
    printf("scalarCopy %lf GB/s\n", timer.getAverage(1));
  }

  {
    cuttTimer timer(8);
    for (int i=0;i < 4;i++) {
      clear_device_array<double>((double *)dataOut, numElem);
      cudaCheck(cudaDeviceSynchronize());
      timer.start(dim, permutation);
      vectorCopy<double>(numElem, (double *)dataIn, (double *)dataOut, 0);
      cudaCheck(cudaDeviceSynchronize());
      timer.stop();
      // printf("%4.2lf GB/s\n", timer.GBs());
    }
    if (!tester->checkTranspose<long long int>(1, dim.data(), permutation.data(), dataOut)) return false;
    printf("vectorCopy %lf GB/s\n", timer.getAverage(1));
  }

  {
    cuttTimer timer(8);
    for (int i=0;i < 4;i++) {
      clear_device_array<double>((double *)dataOut, numElem);
      cudaCheck(cudaDeviceSynchronize());
      timer.start(dim, permutation);
      memcpyFloat(numElem*2, (float *)dataIn, (float *)dataOut, 0);
      cudaCheck(cudaDeviceSynchronize());
      timer.stop();
      // printf("%4.2lf GB/s\n", timer.GBs());
    }
    if (!tester->checkTranspose<long long int>(1, dim.data(), permutation.data(), dataOut)) return false;
    printf("memcpyFloat %lf GB/s\n", timer.getAverage(1));
  }

  return true;
}

