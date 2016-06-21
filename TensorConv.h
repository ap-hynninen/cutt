#ifndef TENSORCONV_H
#define TENSORCONV_H

//
// Stores tensor c object
//
class TensorC {
private:
  const int rank;
  int* c;
  // map[i] tells where to find rank i in c[]
  int* map;
public:
  TensorC(const int rank, const int n, const int* rankInd, const int* dim) : rank(rank) {
    if (rank < 1 || n < 1 || n > rank) {
      printf("TensorC::TensorC, Invalid rank or n\n");
      exit(1);
    }
    map = new int[rank];
    for (int i=0;i < rank;i++) map[i] = -1;
    for (int i=0;i < n;i++) {
      map[rankInd[i]] = i;
    }
    c = new int[n];
    c[0] = 1;
    for (int i=1;i < n;i++) {
      c[i] = c[i-1]*dim[rankInd[i-1]];
    }
  }

  ~TensorC() {
    delete [] c;
    delete [] map;
  }

  int get(const int i) {
    int mapi;
    if (i < 0 || i >= rank || (mapi = map[i]) == -1) {
      printf("TensorC::get(), index out of range\n");
      exit(1);
    }
    return c[mapi];
  }

};

struct TensorConv {
  int c;
  int d;
  int ct;
};

void calcTensorConv(const int rank, const int* dim, const int* permutation,
  TensorConv* tensorConv) {

  tensorConv[0].c = 1;
  tensorConv[0].d = dim[0];
  tensorConv[permutation[0]].ct = 1;
  int ct_prev = 1;
  for (int i=1;i < rank;i++) {
    tensorConv[i].c = tensorConv[i-1].c*dim[i-1];
    tensorConv[i].d = dim[i];
    int ct = ct_prev*dim[permutation[i-1]];
    tensorConv[permutation[i]].ct = ct;
    ct_prev = ct;
  }

}

//
// Determines permutation that produces
// out[i] = inp[permutation[i]], i=0,...,n-1
// inp[i] and out[i] are within the range 0,...,rank-1
//
void calcPermutation(const int rank, const int n, const int* inp, const int* out, int* permutation) {
  // tmp[i] tells where to find rank i in inp[]
  int* tmp = new int[rank];
  for (int i=0;i < rank;i++) tmp[i] = -1;
  for (int i=0;i < n;i++) {
    tmp[inp[i]] = i;
  }
  for (int i=0;i < n;i++) {
    int a = tmp[out[i]];
    permutation[i] = a;
    if (a == -1) {
      printf("calcPermutation, unable to determine permutation\n");
      exit(1);
    }
  }
  delete [] tmp;
}


//
// Calculates the cumulative product of tensor dimensions for a set of rank indices
//
// Input:
// n = number of rank indices
// rankInd = {w1, ..., wn}
// dim(wi) = dimension of rank wi
//
// Output:
// c[0 ... n-1] = cumulative product in the order of the rank indices
//
void calc_c(const int n, const int* rankInd, const int* dim, int* c) {
  c[0] = 1;
  for (int i=1;i < n;i++) {
    c[i] = c[i-1]*dim[rankInd[i-1]];
  }
}


__host__ inline int tensorPos(const int p, const int rank, const TensorConv* tensorConv) {
  int r = 0;
  for (int i=0;i < rank;i++) {
    TensorConv tc = tensorConv[i];
    r += ((p/tc.c) % tc.d)*tc.ct;
  }
  return r;
}

#ifdef __CUDACC__
//
// Returns scalar tensor position. Each lane has different p
//
__device__ __forceinline__
int tensorPosLoop(const int p, const int rank, const int c, const int d, const int ct) {

  int r = 0;
  for (int i=0;i < rank;i++) {
    r += ((p/__shfl(c,i)) % __shfl(d,i))*__shfl(ct,i);
  }

  return r;
}

//
// Returns scalar tensor position. Each lane has different p
//
__device__ __forceinline__
void tensorPosLoop2(const int p, const int rank, const int c1, const int d1, const int ct1,
  const int c2, const int d2, const int ct2, int& r1, int& r2) {

  r1 = 0;
  r2 = 0;
  for (int i=0;i < rank;i++) {
    r1 += ((p/__shfl(c1,i)) % __shfl(d1,i))*__shfl(ct1,i);
    r2 += ((p/__shfl(c2,i)) % __shfl(d2,i))*__shfl(ct2,i);
  }

}

//
// Returns scalar tensor position. Each lane has the same p
// NOTE: c and d on inactive warps must be 1 !!
//
__device__ __forceinline__
int tensorPos(
  const int p, const int rank, const int c, const int d, const int ct,
  const int numLane=warpSize
  ) {

  int r = ((p/c) % d)*ct;
#pragma unroll
  for (int i=numLane/2;i >= 1;i/=2) {
    r += __shfl_xor(r, i);
  }
  return r;

}
#endif

#endif