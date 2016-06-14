#include <vector>
#include <math.h>
#include <time.h>
#include "CudaUtils.h"
#include "CudaTranspose.h"
#include "CpuTranspose.h"

double test_memcpy(int size);
template <typename T> double test_tensor(std::vector<int>& dim);
template <typename T> void test(int size);

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

  int vol = 40*1000000;
  int ranks[7] = {3, 4, 5, 6, 7, 8, 15};
  double bandwidths[7];
  for (int i=0;i < 7;i++) {
    std::vector<int> dim(ranks[i]);
    int dimave = (int)pow(vol, 1.0/(double)ranks[i]);
    if (dimave < 100.0) {
      dim[0]            = 32;
      dim[ranks[i] - 1] = 32;
    } else {
      dim[0]            = dimave;
      dim[ranks[i] - 1] = dimave;
    }
    // Distribute remaining volume to the middle ranks
    int ranks_left = ranks[i] - 2;
    double vol_left = vol/(double)(dim[0]*dim[ranks[i] - 1]);
    for (int r=1;r < ranks[i] - 1;r++) {
      dim[r] = (int)pow(vol_left, 1.0/(double)ranks_left);
      vol_left /= (double)dim[r];
      ranks_left--;
    }
    bandwidths[i] = test_tensor<double>(dim);
  }

  for (int i=0;i < 7;i++) {
    printf("%lf\n", bandwidths[i]);
  }

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
double test_tensor(std::vector<int>& dim) {

  int rank = dim.size();

  int vol = 1;
  for (int r=0;r < rank;r++) {
    vol *= dim[r];
  }

  int* permutation = new int[rank];
  // Inverse permutation
  for (int r=0;r < rank;r++) {
    permutation[r] = rank - 1 - r;
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

  T* h_data_in =  new T[vol];
  T* h_data_ref = new T[vol];
  T* h_data_out = new T[vol];
  for (int i=0;i < vol;i++) {
    h_data_in[i] = (T)i;
    h_data_ref[i] = -1;
    h_data_out[i] = -1;
  }
  T* d_data_in  = NULL;
  T* d_data_out = NULL;
  allocate_device<T>(&d_data_in, vol);
  allocate_device<T>(&d_data_out, vol);
  copy_HtoD<T>(h_data_in, d_data_in, vol);

  TensorTransposePlan plan(rank, dim.data(), permutation);

  cpuTransposeTensor<T>(rank, dim.data(), permutation, h_data_in, h_data_ref);

  for (int i=0;i < 4;i++) {
    struct timespec now, tmstart;
    clock_gettime(CLOCK_REALTIME, &tmstart);

    copy_xyz(vol, d_data_in, d_data_out, 0);

    cudaCheck(cudaDeviceSynchronize());

    clock_gettime(CLOCK_REALTIME, &now);
    double seconds = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
    printf("copy_xyz wall time %lfms %lfGB/s\n", seconds*1000.0, (double)(vol*sizeof(T)*2)/GB/seconds);
  }

#if 0
  printf("transposeTensor\n");
  for (int i=0;i < 4;i++) {
    clear_device_array<T>(d_data_out, vol);
    cudaCheck(cudaDeviceSynchronize());

    struct timespec now, tmstart;
    clock_gettime(CLOCK_REALTIME, &tmstart);
  
    transposeTensor<T>(plan, d_data_in, d_data_out, 0);
    cudaCheck(cudaDeviceSynchronize());
  
    clock_gettime(CLOCK_REALTIME, &now);
    double seconds = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
    printf("wall time %lfms %lfGB/s\n", seconds*1000.0, (double)(vol*sizeof(T)*2)/GB/seconds);
  }

  copy_DtoH<T>(d_data_out, h_data_out, vol);
  if (!checkResult<T>(vol, h_data_ref, h_data_out)) {
    printf("FAILED\n");
  } else {
    printf("OK\n");
  }
#endif

  double bandwidth_ave = 0.0;
  printf("transposeTensorArg\n");
  for (int i=0;i < 4;i++) {
    clear_device_array<T>(d_data_out, vol);
    cudaCheck(cudaDeviceSynchronize());

    struct timespec now, tmstart;
    clock_gettime(CLOCK_REALTIME, &tmstart);
  
    transposeTensorArg<T>(plan, d_data_in, d_data_out, 0);
    cudaCheck(cudaDeviceSynchronize());
  
    clock_gettime(CLOCK_REALTIME, &now);
    double seconds = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
    double bandwidth = (double)(vol*sizeof(T)*2)/GB/seconds;
    printf("wall time %lfms %lfGB/s\n", seconds*1000.0, bandwidth);
    bandwidth_ave += bandwidth;
  }
  bandwidth_ave /= 4.0;

  copy_DtoH<T>(d_data_out, h_data_out, vol);
  if (!checkResult<T>(vol, h_data_ref, h_data_out)) {
    printf("FAILED\n");
  } else {
    printf("OK\n");
  }

  // printf("h_data_ref h_data_out\n");
  // for (int i=0;i <= 5;i++) {
  //   printf("%lf %lf\n", h_data_ref[i], h_data_out[i]);
  // }

  delete [] permutation;
  delete [] h_data_in;
  delete [] h_data_ref;
  delete [] h_data_out;

  deallocate_device<T>(&d_data_in);
  deallocate_device<T>(&d_data_out);

  return bandwidth_ave;
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