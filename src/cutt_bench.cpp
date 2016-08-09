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
#include <ctime>           // std::time
#include <cmath>
#include <cctype>
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
bool use_cuttPlanMeasure;

bool bench1(int numElem);
bool bench2(int numElem);
bool bench3(int numElem);
bool bench4();
bool bench5(int numElem, int ratio);
bool bench6();
bool bench_input(std::vector<int>& dim, std::vector<int>& permutation);
bool bench_memcpy(int numElem);

void getRandomDim(double vol, std::vector<int>& dim);
template <typename T> bool bench_tensor(std::vector<int>& dim, std::vector<int>& permutation);
void printVec(std::vector<int>& vec);

int main(int argc, char *argv[]) {

  int gpuid = -1;
  unsigned seed = unsigned (std::time(0));
  bool arg_ok = true;
  use_cuttPlanMeasure = false;
  std::vector<int> dimIn;
  std::vector<int> permutationIn;
  if (argc >= 2) {
    int i = 1;
    while (i < argc) {
      if (strcmp(argv[i], "-device") == 0) {
        sscanf(argv[i+1], "%d", &gpuid);
        i += 2;
      } else if (strcmp(argv[i], "-measure") == 0) {
        use_cuttPlanMeasure = true;
        i++;
      } else if (strcmp(argv[i], "-seed") == 0) {
        sscanf(argv[i+1], "%u", &seed);
        i += 2;
      } else if (strcmp(argv[i], "-dim") == 0) {
        i++;
        while (i < argc && isdigit(*argv[i])) {
          int val;
          sscanf(argv[i++], "%d", &val);
          dimIn.push_back(val);
        }
      } else if (strcmp(argv[i], "-permutation") == 0) {
        i++;
        while (i < argc && isdigit(*argv[i])) {
          int val;
          sscanf(argv[i++], "%d", &val);
          permutationIn.push_back(val);
        }
      } else {
        arg_ok = false;
        break;
      }
    }
  } else if (argc > 1) {
    arg_ok = false;
  }

  if (!arg_ok) {
    printf("cutt_bench [options]\n");
    printf("Options:\n");
    printf("-device gpuid    : use GPU with ID gpuid (default is 0)\n");
    printf("-measure         : use cuttPlanMeasure (default is cuttPlan)\n");
    printf("-seed seed       : seed value for random number generator (default is system timer)\n");
    printf("-dim ...         : space-separated list of dimensions\n");
    printf("-permutation ... : space-separated list of permutations\n");
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

  if (dimIn.size() > 0) {
    if (!bench_input(dimIn, permutationIn)) goto fail;
    goto benchOK;
  }

#if 0
  if (bench3(200*MILLION)) {
    printf("bench3:\n");
    printf("rank best worst average\n");
    for (auto it=timerDouble.ranksBegin();it != timerDouble.ranksEnd();it++) {
      double worstBW = timerDouble.getWorst(*it);
      double bestBW = timerDouble.getBest(*it);
      double aveBW = timerDouble.getAverage(*it);
      printf("%d %6.2lf %6.2lf %6.2lf\n", *it, bestBW, worstBW, aveBW);
    }
    for (auto it=timerDouble.ranksBegin();it != timerDouble.ranksEnd();it++) {
      std::vector<int> dim;
      std::vector<int> permutation;
      double worstBW = timerDouble.getWorst(*it, dim, permutation);
      printf("rank %d BW %4.2lf\n", *it, worstBW);
      printf("dimensions\n");
      printVec(dim);
      printf("permutation\n");
      printVec(permutation);
    }
  } else {
    goto fail;
  }
#endif

  if (bench5(200*MILLION, 15)) {
    printf("bench5:\n");
    printf("rank best worst average median\n");
    for (auto it=timerDouble.ranksBegin();it != timerDouble.ranksEnd();it++) {
      double worstBW = timerDouble.getWorst(*it);
      double bestBW = timerDouble.getBest(*it);
      double aveBW = timerDouble.getAverage(*it);
      double medBW = timerDouble.getMedian(*it);
      printf("%d %6.2lf %6.2lf %6.2lf %6.2lf\n", *it, bestBW, worstBW, aveBW, medBW);
    }
    for (auto it=timerDouble.ranksBegin();it != timerDouble.ranksEnd();it++) {
      std::vector<int> dim;
      std::vector<int> permutation;
      double worstBW = timerDouble.getWorst(*it, dim, permutation);
      printf("rank %d BW %4.2lf\n", *it, worstBW);
      printf("dimensions\n");
      printVec(dim);
      printf("permutation\n");
      printVec(permutation);
    }
  } else {
    goto fail;
  }

#if 0
  if (bench6()) {
    printf("bench6:\n");
    printf("rank best worst average\n");
    for (auto it=timerDouble.ranksBegin();it != timerDouble.ranksEnd();it++) {
      double worstBW = timerDouble.getWorst(*it);
      double bestBW = timerDouble.getBest(*it);
      double aveBW = timerDouble.getAverage(*it);
      printf("%d %6.2lf %6.2lf %6.2lf\n", *it, bestBW, worstBW, aveBW);
    }
    for (auto it=timerDouble.ranksBegin();it != timerDouble.ranksEnd();it++) {
      std::vector<int> dim;
      std::vector<int> permutation;
      double worstBW = timerDouble.getWorst(*it, dim, permutation);
      printf("rank %d BW %4.2lf\n", *it, worstBW);
      printf("dimensions\n");
      printVec(dim);
      printf("permutation\n");
      printVec(permutation);
    }
  } else {
    goto fail;
  }
#endif

  // if (!bench_memcpy(200*MILLION)) goto fail;

benchOK:
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
    for (int nsample=0;nsample < 50;nsample++) {
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
}

bool bench_input(std::vector<int>& dim, std::vector<int>& permutation) {
  if (!bench_tensor<long long int>(dim, permutation)) return false;
  printf("dimensions\n");
  printVec(dim);
  printf("permutation\n");
  printVec(permutation);
  printf("bandwidth %4.2lf GB/s\n", timerDouble.GBs());
  return true;  
}

//
// Benchmark 5: All permutations for ranks 2-7
//
bool bench5(int numElem, int ratio) {

  const int minDim = 2;
  const int maxDim = 16;
  for (int rank = 2;rank <= 7;rank++) {

    std::vector<int> dim(rank);
    std::vector<int> permutation(rank);
    std::vector<double> dimf(rank);
    double volf = 1.0;
    for (int r=0;r < rank;r++) {
      permutation[r] = r;
      dimf[r] = 1.0 + (double)r*(ratio - 1.0)/(double)(rank - 1);
      volf *= dimf[r];
    }
    double scale = pow((double)numElem/volf, 1.0/(double)rank);
    int vol = 1;
    for (int r=0;r < rank;r++) {
      if (r == rank - 1) {
        dim[r] = ratio*dim[0];
      } else {
        dim[r] = std::max(2, (int)round(dimf[r]*scale));
      }
      vol *= dim[r];
    }
    double cur_ratio = (double)dim[rank-1]/(double)dim[0];
    double vol_re = fabs((double)(vol - numElem)/(double)numElem);
    // Fix dimensions if volume is off by more than 5%
    if (vol_re > 0.05) {
      int d = (vol < numElem) ? 1 : -1;
      int r = 1;
      while (vol_re > 0.05) {
        vol = (vol/dim[r])*(dim[r] + d);
        dim[r] += d;
        vol_re = fabs((double)(vol - numElem)/(double)numElem);
        r++;
      }
    }
    int minDim = *(std::min_element(dim.begin(), dim.end()));
    int maxDim = *(std::max_element(dim.begin(), dim.end()));
    cur_ratio = (double)maxDim/(double)minDim;
    printf("vol %d cur_ratio %lf | %lf\n", vol, cur_ratio, vol_re);
    printVec(dim);

    do {
      if (!bench_tensor<long long int>(dim, permutation)) return false;
    } while (std::next_permutation(permutation.begin(), permutation.begin() + rank));

  }

  return true;
}

//
// Benchmark 6: from "TTC: A Tensor Transposition Compiler for Multiple Architectures"
//
bool bench6() {

  std::vector< std::vector<int> > dims = {
    std::vector<int>{7248,7248},
    std::vector<int>{43408,1216},
    std::vector<int>{1216,43408},
    std::vector<int>{368,384,384},
    std::vector<int>{2144,64,384},
    std::vector<int>{368,64,2307},
    std::vector<int>{384,384,355},
    std::vector<int>{2320,384,59},
    std::vector<int>{384,2320,59},
    std::vector<int>{384,355,384},
    std::vector<int>{2320,59,384},
    std::vector<int>{384,59,2320},
    std::vector<int>{80,96,75,96},
    std::vector<int>{464,16,75,96},
    std::vector<int>{80,16,75,582},
    std::vector<int>{96,75,96,75},
    std::vector<int>{608,12,96,75},
    std::vector<int>{96,12,608,75},
    std::vector<int>{96,75,96,75},
    std::vector<int>{608,12,96,75},
    std::vector<int>{96,12,608,75},
    std::vector<int>{96,96,75,75},
    std::vector<int>{608,96,12,75},
    std::vector<int>{96,608,12,75},
    std::vector<int>{96,75,75,96},
    std::vector<int>{608,12,75,96},
    std::vector<int>{96,12,75,608},
    std::vector<int>{32,48,28,28,48},
    std::vector<int>{176,8,28,28,48},
    std::vector<int>{32,8,28,28,298},
    std::vector<int>{48,28,28,48,28},
    std::vector<int>{352,4,28,48,28},
    std::vector<int>{48,4,28,352,28},
    std::vector<int>{48,28,48,28,28},
    std::vector<int>{352,4,48,28,28},
    std::vector<int>{48,4,352,28,28},
    std::vector<int>{48,48,28,28,28},
    std::vector<int>{352,48,4,28,28},
    std::vector<int>{48,352,4,28,28},
    std::vector<int>{48,28,28,28,48},
    std::vector<int>{352,4,28,28,48},
    std::vector<int>{48,4,28,28,352},
    std::vector<int>{16,32,15,32,15,15},
    std::vector<int>{48,10,15,32,15,15},
    std::vector<int>{16,10,15,103,15,15},
    std::vector<int>{32,15,15,32,15,15},
    std::vector<int>{112,5,15,32,15,15},
    std::vector<int>{32,5,15,112,15,15},
    std::vector<int>{32,15,32,15,15,15},
    std::vector<int>{112,5,32,15,15,15},
    std::vector<int>{32,5,112,15,15,15},
    std::vector<int>{32,15,15,32,15,15},
    std::vector<int>{112,5,15,32,15,15},
    std::vector<int>{32,5,15,112,15,15},
    std::vector<int>{32,15,15,15,15,32},
    std::vector<int>{112,5,15,15,15,32},
    std::vector<int>{32,5,15,15,15,112}
  };

  std::vector< std::vector<int> > permutations = {
    std::vector<int>{1,0},
    std::vector<int>{1,0},
    std::vector<int>{1,0},
    std::vector<int>{0,2,1},
    std::vector<int>{0,2,1},
    std::vector<int>{0,2,1},
    std::vector<int>{1,0,2},
    std::vector<int>{1,0,2},
    std::vector<int>{1,0,2},
    std::vector<int>{2,1,0},
    std::vector<int>{2,1,0},
    std::vector<int>{2,1,0},
    std::vector<int>{0,3,2,1},
    std::vector<int>{0,3,2,1},
    std::vector<int>{0,3,2,1},
    std::vector<int>{2,1,3,0},
    std::vector<int>{2,1,3,0},
    std::vector<int>{2,1,3,0},
    std::vector<int>{2,0,3,1},
    std::vector<int>{2,0,3,1},
    std::vector<int>{2,0,3,1},
    std::vector<int>{1,0,3,2},
    std::vector<int>{1,0,3,2},
    std::vector<int>{1,0,3,2},
    std::vector<int>{3,2,1,0},
    std::vector<int>{3,2,1,0},
    std::vector<int>{3,2,1,0},
    std::vector<int>{0,4,2,1,3},
    std::vector<int>{0,4,2,1,3},
    std::vector<int>{0,4,2,1,3},
    std::vector<int>{3,2,1,4,0},
    std::vector<int>{3,2,1,4,0},
    std::vector<int>{3,2,1,4,0},
    std::vector<int>{2,0,4,1,3},
    std::vector<int>{2,0,4,1,3},
    std::vector<int>{2,0,4,1,3},
    std::vector<int>{1,3,0,4,2},
    std::vector<int>{1,3,0,4,2},
    std::vector<int>{1,3,0,4,2},
    std::vector<int>{4,3,2,1,0},
    std::vector<int>{4,3,2,1,0},
    std::vector<int>{4,3,2,1,0},
    std::vector<int>{0,3,2,5,4,1},
    std::vector<int>{0,3,2,5,4,1},
    std::vector<int>{0,3,2,5,4,1},
    std::vector<int>{3,2,0,5,1,4},
    std::vector<int>{3,2,0,5,1,4},
    std::vector<int>{3,2,0,5,1,4},
    std::vector<int>{2,0,4,1,5,3},
    std::vector<int>{2,0,4,1,5,3},
    std::vector<int>{2,0,4,1,5,3},
    std::vector<int>{3,2,5,1,0,4},
    std::vector<int>{3,2,5,1,0,4},
    std::vector<int>{3,2,5,1,0,4},
    std::vector<int>{5,4,3,2,1,0},
    std::vector<int>{5,4,3,2,1,0},
    std::vector<int>{5,4,3,2,1,0}
  };

  for (int i=0;i < dims.size();i++) {
    if (!bench_tensor<long long int>(dims[i], permutations[i])) return false;
    printf("dimensions\n");
    printVec(dims[i]);
    printf("permutation\n");
    printVec(permutations[i]);
    printf("bandwidth %4.2lf GiB/s\n", timerDouble.GiBs());
  }

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
  if (use_cuttPlanMeasure) {
    cuttCheck(cuttPlanMeasure(&plan, rank, dim.data(), permutation.data(), sizeof(T), 0, dataIn, dataOut));
  } else {
    cuttCheck(cuttPlan(&plan, rank, dim.data(), permutation.data(), sizeof(T), 0));
  }

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
      timer.stop();
      // printf("%4.2lf GB/s\n", timer.GBs());
    }
    if (!tester->checkTranspose<long long int>(1, dim.data(), permutation.data(), dataOut)) return false;
    printf("memcpyFloat %lf GB/s\n", timer.getAverage(1));
  }

  return true;
}

