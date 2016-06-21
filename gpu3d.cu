#include <vector>
#include <math.h>
#include <time.h>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include "CudaUtils.h"
#include "CudaTranspose.h"
#include "CpuTranspose.h"

double test_memcpy(int size);
template <typename T> double test_tensor(std::vector<int>& dim, std::vector<int>& permutation,
  T* h_data_in, T* h_data_ref, T* h_data_out, T* d_data_in, T* d_data_out);
template <typename T> void test(int size);

std::vector<int> decode_permutation(const int rank, const int ind) {
  std::vector<int> numbers(rank);
  for (int r=0;r < rank;r++) {
    numbers[r] = r;
  }

  std::vector<int> permutation(rank);
  int i = ind;
  for (int r=0;r < rank;r++) {
    // Number of choices
    int n = rank - r;
    int j = (i % n);
    permutation[r] = numbers[j];
    numbers.erase(numbers.begin() + j);
    i /= n;
  }

  return permutation;
}

int factorial(int n) {
  int ans = 1;
  while (n > 1) {
    ans *= n--;
  }
  return ans;
}

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
    printf("gpu3d [options]\n");
    printf("Options:\n");
    printf("-device gpuid : use GPU with ID gpuid\n");
    return 1;
  }

  if (gpuid >= 0) {
    cudaCheck(cudaSetDevice(gpuid));
  }

  cudaCheck(cudaDeviceReset());

  cudaCheck(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
  // cudaCheck(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));

  // cudaCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual));

  // test<double>(256);

  // test_memcpy(67108864);

#if 0
  int dim2dIn[2] = {2, 4};
  int dim2dOut[2] = {4, 2};
  int permutation2d[2] = {1, 0};
  int data2dIn[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int data2dOut[8];
  cpuTransposeTensor(2, dim2dIn, permutation2d, data2dIn, data2dOut);
  printTensor2D(dim2dIn, data2dIn);
  printTensor2D(dim2dOut, data2dOut);
#endif

#if 0
  for (int rank = 4;rank <= 4;rank++) {
    std::vector<int> dim(rank);
    for (int r=0;r < rank;r++) dim[r] = 16;
    // for (int r=0;r < rank;r++) dim[r] = 2 + r;
    int rank_factorial = factorial(rank);
    std::vector<double> bandwidths(rank_factorial);

    int vol = 1;
    for (int r=0;r < rank;r++) {
      vol *= dim[r];
    }
    double* h_data_in =  new double[vol];
    double* h_data_ref = new double[vol];
    double* h_data_out = new double[vol];
    for (int i=0;i < vol;i++) {
      h_data_in[i] = (double)i;
      h_data_ref[i] = -1;
      h_data_out[i] = -1;
    }
    double* d_data_in  = NULL;
    double* d_data_out = NULL;
    allocate_device<double>(&d_data_in, vol);
    allocate_device<double>(&d_data_out, vol);
    copy_HtoD<double>(h_data_in, d_data_in, vol);

    // printf("rank_factorial %d\n", rank_factorial);
    for (int p=0;p < rank_factorial;p++) {

      std::vector<int> permutation = decode_permutation(rank, p);

      if (permutation[0] == 3 && permutation[1] == 2 &&
        permutation[2] == 1 && permutation[3] == 0) {
    
        bandwidths[p] = test_tensor<double>(dim, permutation,
          h_data_in, h_data_ref, h_data_out, d_data_in, d_data_out);

        if (bandwidths[p] < 0.0) goto break_loop;
  
      }

    }

    for (int p=0;p < rank_factorial;p++) {
      printf("%1.3lf\n", bandwidths[p]);
    }    

    delete [] h_data_in;
    delete [] h_data_ref;
    delete [] h_data_out;
    deallocate_device<double>(&d_data_in);
    deallocate_device<double>(&d_data_out);

  }
break_loop:
#endif

#if 0
  std::vector<int> dim(4);
  std::vector<int> permutation(4);
  dim[0] = 16;
  dim[1] = 2;
  dim[2] = 8;
  dim[3] = 4;
  for (int r=0;r < 4;r++) {
    permutation[r] = 4 - 1 - r;
  }
  // permutation[0] = 2;
  // permutation[1] = 3;
  // permutation[2] = 0;
  // permutation[3] = 1;

  int vol = 1;
  for (int r=0;r < 4;r++) {
    vol *= dim[r];
  }
  double* h_data_in =  new double[vol];
  double* h_data_ref = new double[vol];
  double* h_data_out = new double[vol];
  for (int j=0;j < vol;j++) {
    h_data_in[j] = (double)j;
    h_data_ref[j] = -1;
    h_data_out[j] = -1;
  }
  double* d_data_in  = NULL;
  double* d_data_out = NULL;
  allocate_device<double>(&d_data_in, vol);
  allocate_device<double>(&d_data_out, vol);
  copy_HtoD<double>(h_data_in, d_data_in, vol);

  double bandwidth = test_tensor<double>(dim, permutation,
    h_data_in, h_data_ref, h_data_out, d_data_in, d_data_out);

  delete [] h_data_in;
  delete [] h_data_ref;
  delete [] h_data_out;
  deallocate_device<double>(&d_data_in);
  deallocate_device<double>(&d_data_out);
#endif

#if 1
  int numElem = 40*1000000;
  int ranks[8] = {2, 3, 4, 5, 6, 7, 8, 15};
  double bandwidths[8];
  for (int i=7;i < 8;i++) {
    std::vector<int> dim(ranks[i]);
    std::vector<int> permutation(ranks[i]);
    int dimave = (int)pow(numElem, 1.0/(double)ranks[i]);

#if 0
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
#else
    double numElem_left = numElem;
    for (int r=0;r < ranks[i];r++) {
      dim[r] = (int)pow(numElem_left, 1.0/(double)(ranks[i] - r));
      numElem_left /= (double)dim[r];
    }
#endif

    // Inverse order
    for (int r=0;r < ranks[i];r++) {
      permutation[r] = ranks[i] - 1 - r;
    }

    // // Random order
    // for (int r=0;r < ranks[i];r++) {
    //   permutation[r] = r;
    // }
    // std::srand(unsigned (std::time(0)));
    // std::random_shuffle(permutation.begin(), permutation.end());

    int vol = 1;
    for (int r=0;r < ranks[i];r++) {
      vol *= dim[r];
    }
    double* h_data_in =  new double[vol];
    double* h_data_ref = new double[vol];
    double* h_data_out = new double[vol];
    for (int j=0;j < vol;j++) {
      h_data_in[j] = (double)j;
      h_data_ref[j] = -1;
      h_data_out[j] = -1;
    }
    double* d_data_in  = NULL;
    double* d_data_out = NULL;
    allocate_device<double>(&d_data_in, vol);
    allocate_device<double>(&d_data_out, vol);
    copy_HtoD<double>(h_data_in, d_data_in, vol);

    bandwidths[i] = test_tensor<double>(dim, permutation,
      h_data_in, h_data_ref, h_data_out, d_data_in, d_data_out);

    delete [] h_data_in;
    delete [] h_data_ref;
    delete [] h_data_out;
    deallocate_device<double>(&d_data_in);
    deallocate_device<double>(&d_data_out);
  }

  for (int i=0;i < 8;i++) {
    printf("%lf\n", bandwidths[i]);
  }
#endif

  cudaCheck(cudaDeviceReset());

  return 0;
}

template <typename T>
bool checkResult(const int vol,
  const T* h_data_ref, const T* h_data_out) {

  for (int i=0;i < vol;i++) {
    if (h_data_ref[i] != h_data_out[i] || h_data_ref[i] < 0.0 || h_data_out[i] < 0.0) {
      printf("checkResult i %d | %lf %lf\n", i, h_data_ref[i], h_data_out[i]);
      return false;
    }
  }

  return true;
}

template <typename T>
bool checkResult(const int vol, CpuTensorTranspose& tensorRef, const T* h_data_out) {

  for (int i=0;i < vol;i++) {
    T refval = (T)tensorRef.pos(i);
    if (refval != h_data_out[i] || h_data_out[i] < 0.0) {
      printf("checkResult i %d | %lf %lf\n", i, refval, h_data_out[i]);
      return false;
    }
  }

  return true;
}

#define GB 1000000000.0

double test_memcpy(int size) {
  float* h_data_in =  new float[size];
  float* h_data_out = new float[size];
  for (int i=0;i < size;i++) {
    h_data_in[i] = (float)i;
  }
  float* d_data_in  = NULL;
  float* d_data_out = NULL;
  allocate_device<float>(&d_data_in, size);
  allocate_device<float>(&d_data_out, size);
  copy_HtoD<float>(h_data_in, d_data_in, size);

  printf("Memcpy size %d bytes\n", size*sizeof(float));

  double bandwidth_ave = 0.0;
  for (int i=0;i < 4;i++) {
    clear_device_array<float>(d_data_out, size);
    cudaCheck(cudaDeviceSynchronize());

    struct timespec now, tmstart;
    clock_gettime(CLOCK_REALTIME, &tmstart);

    memcpy_float(size, d_data_in, d_data_out, 0);

    cudaCheck(cudaDeviceSynchronize());

    clock_gettime(CLOCK_REALTIME, &now);
    double seconds = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
    double bandwidth = (double)(size*sizeof(float)*2)/GB/seconds;
    printf("memcpy_float wall time %lfms %lfGB/s\n", seconds*1000.0, bandwidth);
    bandwidth_ave += bandwidth;
  }
  bandwidth_ave /= 4.0;

  copy_DtoH<float>(d_data_out, h_data_out, size);
  if (!checkResult<float>(size, h_data_in, h_data_out)) {
    printf("memcpy_float FAILED\n");
  } else {
    printf("memcpy_float OK\n");
  }

  delete [] h_data_in;
  delete [] h_data_out;
  deallocate_device<float>(&d_data_in);
  deallocate_device<float>(&d_data_out);

  return bandwidth_ave;
}

template <typename T>
double test_tensor(std::vector<int>& dim, std::vector<int>& permutation,
  T* h_data_in, T* h_data_ref, T* h_data_out, T* d_data_in, T* d_data_out) {

  int rank = dim.size();

  int vol = 1;
  for (int r=0;r < rank;r++) {
    vol *= dim[r];
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

  TensorTransposePlan* plan;
  {
    struct timespec now, tmstart;
    clock_gettime(CLOCK_REALTIME, &tmstart);
    
    plan = new TensorTransposePlan(rank, dim.data(), permutation.data(), sizeof(T));
    
    clock_gettime(CLOCK_REALTIME, &now);
    double seconds = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
    printf("plan took %lfs\n", seconds);
  }

  {
    struct timespec now, tmstart;
    clock_gettime(CLOCK_REALTIME, &tmstart);

    cpuTransposeTensor<T>(rank, dim.data(), permutation.data(), h_data_in, h_data_ref);
    
    clock_gettime(CLOCK_REALTIME, &now);
    double seconds = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
    printf("cpu transpose took %lfs\n", seconds);
  }

  for (int i=0;i < 4;i++) {
    struct timespec now, tmstart;
    clock_gettime(CLOCK_REALTIME, &tmstart);

    copy_xyz(vol, d_data_in, d_data_out, 0);

    cudaCheck(cudaDeviceSynchronize());

    clock_gettime(CLOCK_REALTIME, &now);
    double seconds = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
    printf("copy_xyz wall time %lfms %lfGB/s\n", seconds*1000.0, (double)(vol*sizeof(T)*2)/GB/seconds);
  }

  double bandwidth_ave = 0.0;
  printf("transposeTensorArg\n");
  for (int i=0;i < 4;i++) {
    clear_device_array<T>(d_data_out, vol);
    cudaCheck(cudaDeviceSynchronize());

    struct timespec now, tmstart;
    clock_gettime(CLOCK_REALTIME, &tmstart);
  
    transposeTensorArg<T>(*plan, d_data_in, d_data_out, 0);
    cudaCheck(cudaDeviceSynchronize());
  
    clock_gettime(CLOCK_REALTIME, &now);
    double seconds = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
    double bandwidth = (double)(vol*sizeof(T)*2)/GB/seconds;
    printf("wall time %lfms %lfGB/s\n", seconds*1000.0, bandwidth);
    bandwidth_ave += bandwidth;
  }
  bandwidth_ave /= 4.0;

  bool ok;
  copy_DtoH<T>(d_data_out, h_data_out, vol);
  if (!checkResult<T>(vol, h_data_ref, h_data_out)) {
    printf("FAILED\n");
    ok = false;
  } else {
    printf("OK\n");
    ok = true;
  }

  delete plan;

  return (ok) ? bandwidth_ave : -1.0;
}

template <typename T>
void test(int size) {
  int nx = size;
  int ny = size;
  int nz = size;
  T* h_data_in =  new T[nx*ny*nz];
  T* h_data_out = new T[nx*ny*nz];
  for (int i=0;i < nx*ny*nz;i++) {
    h_data_in[i] = (T)i;
  }
  T* d_data_in  = NULL;
  T* d_data_out = NULL;
  allocate_device<T>(&d_data_in, nx*ny*nz);
  allocate_device<T>(&d_data_out, nx*ny*nz);
  copy_HtoD<T>(h_data_in, d_data_in, nx*ny*nz);

  printf("Transpose size %d bytes\n", nx*ny*nz*sizeof(T));

  for (int i=0;i < 4;i++) {
    struct timespec now, tmstart;
    clock_gettime(CLOCK_REALTIME, &tmstart);

    copy_xyz(nx*ny*nz, d_data_in, d_data_out, 0);

    cudaCheck(cudaDeviceSynchronize());

    clock_gettime(CLOCK_REALTIME, &now);
    double seconds = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
    printf("copy_xyz wall time %lfms %lfGB/s\n", seconds*1000.0, (double)(nx*ny*nz*sizeof(T)*2)/GB/seconds);
  }

  copy_DtoH<T>(d_data_out, h_data_out, nx*ny*nz);
  if (!checkResult<T>(nx*ny*nz, h_data_in, h_data_out)) {
    printf("copy_xyz FAILED\n");
  } else {
    printf("copy_xyz OK\n");
  }

  for (int i=0;i < 4;i++) {
    struct timespec now, tmstart;
    clock_gettime(CLOCK_REALTIME, &tmstart);

    copy_vector_xyz(nx*ny*nz, d_data_in, d_data_out, 0);

    cudaCheck(cudaDeviceSynchronize());

    clock_gettime(CLOCK_REALTIME, &now);
    double seconds = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
    printf("copy_vector_xyz wall time %lfms %lfGB/s\n", seconds*1000.0, (double)(nx*ny*nz*sizeof(T)*2)/GB/seconds);
  }

  copy_DtoH<T>(d_data_out, h_data_out, nx*ny*nz);
  if (!checkResult<T>(nx*ny*nz, h_data_in, h_data_out)) {
    printf("copy_vector_xyz FAILED\n");
  } else {
    printf("copy_vector_xyz OK\n");
  }

  // 2-3-1
  for (int i=0;i < 4;i++) {
    
    struct timespec now, tmstart;
    clock_gettime(CLOCK_REALTIME, &tmstart);
    
    transpose_xyz_yzx(nx, ny, nz, 
      nx, ny, ny, nz,
      d_data_in, d_data_out, 0);
    
    cudaCheck(cudaDeviceSynchronize());

    clock_gettime(CLOCK_REALTIME, &now);
    double seconds = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
    printf("2-3-1 wall time %lfms %lfGB/s\n", seconds*1000.0, (double)(nx*ny*nz*sizeof(T)*2)/GB/seconds);
  }

  // 3-1-2
  for (int i=0;i < 4;i++) {
    struct timespec now, tmstart;
    clock_gettime(CLOCK_REALTIME, &tmstart);

    transpose_xyz_zxy(nx, ny, nz, 
      nx, ny, ny, nz,
      d_data_in, d_data_out, 0);

    cudaCheck(cudaDeviceSynchronize());

    clock_gettime(CLOCK_REALTIME, &now);
    double seconds = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
    printf("3-1-2 wall time %lfms %lfGB/s\n", seconds*1000.0, (double)(nx*ny*nz*sizeof(T)*2)/GB/seconds);
  }

  copy_DtoH<T>(d_data_out, h_data_out, nx*ny*nz);

  delete [] h_data_in;
  delete [] h_data_out;
  deallocate_device<T>(&d_data_in);
  deallocate_device<T>(&d_data_out);
}