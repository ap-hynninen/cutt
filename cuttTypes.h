#ifndef CUTTTYPES_H
#define CUTTTYPES_H

const int maxNumRegStorage = 8;

struct TensorConv {
  int c;
  int d;
  int ct;
};

struct TensorConvInOut {
  int c_in;
  int d_in;
  int ct_in;
  int c_out;
  int d_out;
  int ct_out;
};

#endif // CUTTTYPES_H
