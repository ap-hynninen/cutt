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
#ifndef TENSORTESTER_H
#define TENSORTESTER_H
#include "cuttTypes.h"

//
// Simple tensor transpose tester class
//

struct TensorError_t {
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
  TensorError_t* h_error;
  TensorError_t* d_error;
  int* d_fail;

  TensorTester();
  ~TensorTester();

  void setTensorCheckPattern(unsigned int* data, unsigned int ndata);
  
  template<typename T> bool checkTranspose(int rank, int* dim, int* permutation, T* data);

};

#endif // TENSORTESTER_H
