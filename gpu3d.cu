#include <time.h>
#include "CudaUtils.h"
#include "CudaTranspose.h"
#include "CpuTranspose.h"

template <typename T> void test_tensor(int rank, int size);
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

  test<double>(256);
  test_tensor<double>(3, 256);

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

template <typename T>
void test_tensor(int rank, int size) {

  int vol = 1;
  int* dim = new int[rank];
  for (int r=0;r < rank;r++) {
    dim[r] = size;
    vol *= size;
  }

  int* permutation = new int[rank];
  // Inverse permutation
  for (int r=0;r < rank;r++) {
    permutation[r] = rank - 1 - r;
  }
  permutation[0] = 1;
  permutation[1] = 2;
  permutation[2] = 0;

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

  TensorTransposePlan plan(rank, dim, permutation);

  cpuTransposeTensor<T>(rank, dim, permutation, h_data_in, h_data_ref);

  for (int i=0;i < 4;i++) {
    struct timespec now, tmstart;
    clock_gettime(CLOCK_REALTIME, &tmstart);
  
    transposeTensorArg<T>(plan, d_data_in, d_data_out, 0);
    cudaCheck(cudaDeviceSynchronize());
  
    clock_gettime(CLOCK_REALTIME, &now);
    double seconds = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
    for (int r=0;r < rank;r++)
      printf("%d%c",permutation[r]+1, (r==rank-1) ? ' ' : '-');
    printf("wall time %lfms %lfGB/s\n", seconds*1000.0, (double)(vol*sizeof(T)*2)/GB/seconds);
  }

  // transpose_xyz_yzx(size, size, size, 
  //   size, size, size, size,
  //   d_data_in, d_data_out, 0);

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

  delete [] dim;
  delete [] permutation;
  delete [] h_data_in;
  delete [] h_data_ref;
  delete [] h_data_out;

  deallocate_device<T>(&d_data_in);
  deallocate_device<T>(&d_data_out);
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

    copy_xyz(nx, ny, nz, d_data_in, d_data_out, 0);

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

    copy_vector_xyz(nx, ny, nz, d_data_in, d_data_out, 0);

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