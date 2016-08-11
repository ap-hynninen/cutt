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

#ifndef CUTTTIMER_H
#define CUTTTIMER_H

#include <vector>
#include <chrono>
#include <cstdlib>
#include <unordered_map>
#include <set>
#include <queue>          // std::priority_queue

//
// Simple raw timer
//
class Timer {
private:
  std::chrono::high_resolution_clock::time_point tmstart, tmend;
public:
  void start();
  void stop();
  double seconds();
};

//
// Records timings for cuTT and gives out bandwidths and other data
//
class cuttTimer {
private:
  // Size of the type we're measuring
  const int sizeofType;

  // Dimension and permutation of the current run
  std::vector<int> curDim;
  std::vector<int> curPermutation;

  // Bytes transposed in the current run
  size_t curBytes;

  // Timer for current run
  Timer timer;

  struct Stat {
    int numSample;
    double totBW;
    double minBW;
    double maxBW;
    std::priority_queue<double> maxHeap;
    std::priority_queue<double, std::vector<double>, std::greater<double> > minHeap;
    std::vector<int> worstDim;
    std::vector<int> worstPermutation;
    Stat() {
      numSample = 0;
      totBW = 0.0;
      minBW = 1.0e20;
      maxBW = -1.0;
    }
  };

  // List of ranks that have been recorded
  std::set<int> ranks;

  // Statistics for every rank
  std::unordered_map<int, Stat> stats;

public:
  cuttTimer(int sizeofType);
  ~cuttTimer();
  void start(std::vector<int>& dim, std::vector<int>& permutation);
  void stop();
  double seconds();
  double GBs();
  double GiBs();
  double getBest(int rank);
  double getWorst(int rank);
  double getWorst(int rank, std::vector<int>& dim, std::vector<int>& permutation);
  double getMedian(int rank);
  double getAverage(int rank);

  double getWorst(std::vector<int>& dim, std::vector<int>& permutation);

  std::set<int>::const_iterator ranksBegin() {
    return ranks.begin();
  }

  std::set<int>::const_iterator ranksEnd() {
    return ranks.end();
  }
};

#endif // CUTTTIMER_H
