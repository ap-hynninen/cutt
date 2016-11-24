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
#ifndef CUTTTYPES_H
#define CUTTTYPES_H
#include "int_fastdiv.h"

#define MAX_REG_STORAGE 8

struct TensorConv {
  int c;
  int d;
  int ct;
};

struct TensorConvFast {
  int_fastdiv c;
  int_fastdiv d;
  int ct;
};

struct TensorConvInOut {
  int c_in;
  int d_in;
  int ct_in;
  int c_out;
  int d_out;
  int ct_out;
  
  static TensorConvInOut make_TensorConvInOut(const int c_in, const int d_in, const int ct_in,
    const int c_out, const int d_out, const int ct_out) {
    TensorConvInOut res;
    res.c_in   = c_in;
    res.d_in   = d_in;
    res.ct_in  = ct_in;
    res.c_out  = c_out;
    res.d_out  = d_out;
    res.ct_out = ct_out;
    return res;
  }
};


struct TensorConvInOutFast {
  int_fastdiv c_in;
  int_fastdiv d_in;
  int ct_in;
  int_fastdiv c_out;
  int_fastdiv d_out;
  int ct_out;

  static TensorConvInOutFast make_TensorConvInOutFast(const TensorConvInOut& conv) {
    TensorConvInOutFast res;
    res.c_in   = conv.c_in;
    res.d_in   = conv.d_in;
    res.ct_in  = conv.ct_in;
    res.c_out  = conv.c_out;
    res.d_out  = conv.d_out;
    res.ct_out = conv.ct_out;
    return res;
  }
};


#endif // CUTTTYPES_H
