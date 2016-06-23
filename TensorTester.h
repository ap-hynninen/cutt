#ifndef TENSORTESTER_H
#define TENSORTESTER_H
#include "cuttTypes.h"

//
// Simple tensor transpose tester class
//

struct error_t {
  int refVal;
  int dataVal;
  unsigned int pos;
};

class TensorTester {
private:
  static int calcTensorConv(const int rank, const int* dim, const int* permutation, TensorConv* tensorConv);

  const int maxRank;
  const int maxNumblock;

public:
  TensorConv* h_tensorConv;
  TensorConv* d_tensorConv;
  error_t* h_error;
  error_t* d_error;
  int* d_fail;

  TensorTester();
  ~TensorTester();

  void setTensorCheckPattern(unsigned int* data, unsigned int ndata);
  
  template<typename T> bool checkTranspose(int rank, int* dim, int* permutation, T* data);

};

#endif // TENSORTESTER_H
