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

bool bench1(std::vector<double>& bandwidths);
template <typename T> bool bench_tensor(std::vector<int>& dim, std::vector<int>& permutation, double& bandwidth_ave);

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

  std::vector<double> bandwidths;

  if (!bench1(bandwidths)) goto fail;
  printf("bench1:\n");
  for (int i=0;i < bandwidths.size();i++) {
    printf("%lf\n", bandwidths[i]);
  }


  printf("bench OK\n");
  goto end;
fail:
  printf("bench FAIL\n");
end:
  deallocate_device<long long int>(&dataIn);
  deallocate_device<long long int>(&dataOut);
  delete tester;

  cudaCheck(cudaDeviceReset());
  return 0;
}

//
// Benchmark 1: ranks 2-8,15 with 40M doubles
//
bool bench1(std::vector<double>& bandwidths) {
  bandwidths.resize(8, 0.0);
  int numElem = 40*1000000;
  int ranks[8] = {2, 3, 4, 5, 6, 7, 8, 15};
  for (int i=7;i < 8;i++) {
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

    if (!bench_tensor<long long int>(dim, permutation, bandwidths[i])) return false;
  }

  return true;
}

template <typename T>
bool bench_tensor(std::vector<int>& dim, std::vector<int>& permutation, double& bandwidth_ave) {

  const double GB = 1000000000.0;

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

  bandwidth_ave = 0.0;
  for (int i=0;i < 4;i++) {
    clear_device_array<T>((T *)dataOut, vol);
    cudaCheck(cudaDeviceSynchronize());

    struct timespec now, tmstart;
    clock_gettime(CLOCK_REALTIME, &tmstart);

    cuttCheck(cuttExecute(plan, dataIn, dataOut));
    cudaCheck(cudaDeviceSynchronize());
  
    clock_gettime(CLOCK_REALTIME, &now);
    double seconds = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
    double bandwidth = (double)(vol*sizeof(T)*2)/GB/seconds;
    printf("wall time %lfms %lf GB/s\n", seconds*1000.0, bandwidth);
    bandwidth_ave += bandwidth;
  }
  bandwidth_ave /= 4.0;

  cuttCheck(cuttDestroy(plan));
  return tester->checkTranspose<T>(rank, dim.data(), permutation.data(), (T *)dataOut);
}
