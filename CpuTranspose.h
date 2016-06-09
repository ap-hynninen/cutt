
//
// Slow version, for result correctness testing only
//
template <typename T>
void cpuTransposeTensor(const int rank, const int* dim, int* permutation,
  const T* dataIn, T* dataOut) {

  int* dimOut = new int[rank];
  int* cuDimIn = new int[rank+1];
  int* cuDimOut = new int[rank+1];
  int* inv_permutation = new int[rank];

  for (int i=0;i < rank;i++) {
    inv_permutation[permutation[i]] = i;
  }

  for (int r=0;r < rank;r++) {
    dimOut[r] = dim[permutation[r]];
  }

  cuDimIn[0] = 1;
  for (int r=1;r <= rank;r++) {
    cuDimIn[r] = cuDimIn[r-1]*dim[r-1];
  }

  for (int r=0;r < rank;r++) {
    cuDimOut[r] = cuDimIn[inv_permutation[r]];
  }

  // printf("permutation\n");
  // for (int r=0;r < rank;r++) {
  //   printf("%d ", permutation[r]);
  // }
  // printf("\n");

  // printf("cuDimOut\n");
  // for (int r=0;r < rank;r++) {
  //   printf("%d ", cuDimOut[r]);
  // }
  // printf("\n");

  for (int i=0;i < cuDimIn[rank];i++) {
    // Read data
    int dataInVal = dataIn[i];
    // Calculate position in transposed tensor
    int j = i;
    int pos = 0;
    for (int r=0;r < rank;r++) {
      int dimVal = dimOut[r];
      pos += (j % dimVal)*cuDimOut[r];
      j /= dimVal;
    }
    // Write data
    dataOut[pos] = dataInVal;
  }

  delete [] dimOut;
  delete [] cuDimIn;
  delete [] cuDimOut;
  delete [] inv_permutation;

}
