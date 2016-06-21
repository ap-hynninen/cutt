#include <iostream>

#include "TensorConv.h"

class CpuTensorTranspose {
public:
  // Rank of the tensor
  const int rank;
  
  int* dim;
  int* cuDimOut;  

  CpuTensorTranspose(const int rank, const int* in_dim, const int* permutation) : rank(rank) {
    dim = new int[rank];
    for (int r=0;r < rank;r++) {
      dim[r] = in_dim[r];
    }

    int* tmp = new int[rank];
    int* inv_permutation = new int[rank];
    for (int i=0;i < rank;i++) {
      inv_permutation[permutation[i]] = i;
    }

    tmp[0] = 1;
    for (int r=1;r < rank;r++) {
      tmp[r] = tmp[r-1]*dim[permutation[r-1]];
    }

    cuDimOut = new int[rank];
    for (int r=0;r < rank;r++) {
      cuDimOut[r] = tmp[inv_permutation[r]];
    }

    delete [] inv_permutation;
    delete [] tmp;
  }

  ~CpuTensorTranspose() {
    delete [] dim;
    delete [] cuDimOut;
  }

  inline int pos(int i) {
    int posval = 0;
    for (int r=0;r < rank;r++) {
      int dimVal = dim[r];
      posval += (i % dimVal)*cuDimOut[r];
      i /= dimVal;
    }
    return posval;
  }
};

//
// Slow version, for result correctness testing only
//
template <typename T>
void cpuTransposeTensor(const int rank, const int* dim, int* permutation,
  const T* dataIn, T* dataOut) {

  // int* cuDimOut = new int[rank];
  // int* inv_permutation = new int[rank];
  // for (int i=0;i < rank;i++) {
  //   inv_permutation[permutation[i]] = i;
  // }

  // int* tmp = new int[rank];
  // tmp[0] = 1;
  // for (int r=1;r < rank;r++) {
  //   tmp[r] = tmp[r-1]*dim[permutation[r-1]];
  // }

  // for (int r=0;r < rank;r++) {
  //   cuDimOut[r] = tmp[inv_permutation[r]];
  // }

  TensorConv* tensorConv = new TensorConv[rank];
  calcTensorConv(rank, dim, permutation, tensorConv);

  int vol = 1;
  for (int r=0;r < rank;r++) vol *= dim[r];

  for (int i=0;i < vol;i++) {
    // Read data
    int dataInVal = dataIn[i];
    // Calculate position in transposed tensor
    // int j = i;
    // int pos = 0;
    // for (int r=0;r < rank;r++) {
    //   int dimVal = dim[r];
    //   pos += (j % dimVal)*cuDimOut[r];
    //   j /= dimVal;
    // }
    // for (int r=0;r < rank;r++) {
    //   pos += ((j/tensorConv[r].c) % tensorConv[r].d)*tensorConv[r].ct;
    // }

    int pos = tensorPos(i, rank, tensorConv);

    // Write data
    dataOut[pos] = dataInVal;
  }

  // delete [] tmp;
  // delete [] cuDimOut;
  // delete [] inv_permutation;
}

template <typename T>
void printTensor2D(const int* dim, const T* data) {
  int p = 0;
  for (int j=0;j < dim[1];j++) {
    for (int i=0;i < dim[0];i++) {
      std::cout << data[p] << " ";
      p++;
    }
    std::cout << std::endl;
  }
}