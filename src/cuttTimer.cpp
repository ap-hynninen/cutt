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

#include "cuttTimer.h"
#include "CudaUtils.h"
#include <limits>       // std::numeric_limits

void Timer::start() {
  tmstart = std::chrono::high_resolution_clock::now();
}

void Timer::stop() {
  cudaCheck(cudaDeviceSynchronize());
  tmend = std::chrono::high_resolution_clock::now();
}

//
// Returns the duration of the last run in seconds
//
double Timer::seconds() {
  return std::chrono::duration_cast< std::chrono::duration<double> >(tmend - tmstart).count();
}

//
// Class constructor
//
cuttTimer::cuttTimer(int sizeofType) : sizeofType(sizeofType) {}

//
// Class destructor
//
cuttTimer::~cuttTimer() {}

//
// Start timer
//
void cuttTimer::start(std::vector<int>& dim, std::vector<int>& permutation) {
  curDim = dim;
  curPermutation = permutation;
  curBytes = sizeofType*2;   // "2x" because every element is read and also written out
  for (int i=0;i < curDim.size();i++) {
    curBytes *= dim[i];
  }
  ranks.insert(curDim.size());
  timer.start();
}

//
// Stop timer and record statistics
//
void cuttTimer::stop() {
  timer.stop();
  double bandwidth = GBs();
  auto it = stats.find(curDim.size());
  if (it == stats.end()) {
    Stat new_stat;
    std::pair<int, Stat> new_elem(curDim.size(), new_stat);
    auto retval = stats.insert(new_elem);
    it = retval.first;
  }
  Stat& stat = it->second;
  stat.numSample++;
  stat.totBW += bandwidth;
  if (bandwidth < stat.minBW) {
    stat.minBW = bandwidth;
    stat.worstDim = curDim;
    stat.worstPermutation = curPermutation;
  }
  stat.maxBW = std::max(stat.maxBW, bandwidth);
  // maxHeap <= minHeap
  if (stat.maxHeap.size() == 0) {
    stat.maxHeap.push(bandwidth);    
  } else if (stat.minHeap.size() == 0) {
    if (stat.maxHeap.top() <= bandwidth) {
      stat.minHeap.push(bandwidth);
    } else {
      stat.minHeap.push(stat.maxHeap.top());
      stat.maxHeap.pop();
      stat.maxHeap.push(bandwidth);
    }
  } else {
    if (bandwidth <= stat.maxHeap.top()) {
      stat.maxHeap.push(bandwidth);
    } else {
      stat.minHeap.push(bandwidth);
    }
  }
  // Balance
  if (stat.maxHeap.size() > stat.minHeap.size() + 1) {
    stat.minHeap.push(stat.maxHeap.top());
    stat.maxHeap.pop();
  } else if (stat.minHeap.size() > stat.maxHeap.size() + 1) {
    stat.maxHeap.push(stat.minHeap.top());
    stat.minHeap.pop();
  }
}

//
// Returns the duration of the last run in seconds
//
double cuttTimer::seconds() {
  return timer.seconds();
}

//
// Returns the bandwidth of the last run in GB/s
//
double cuttTimer::GBs() {
  const double BILLION = 1000000000.0;
  double sec = seconds();
  return (sec == 0.0) ? 0.0 : (double)(curBytes)/(BILLION*sec);
}

//
// Returns the bandwidth of the last run in GiB/s
//
double cuttTimer::GiBs() {
  const double iBILLION = 1073741824.0;
  double sec = seconds();
  return (sec == 0.0) ? 0.0 : (double)(curBytes)/(iBILLION*sec);
}

//
// Returns the best performing tensor transpose for rank
//
double cuttTimer::getBest(int rank) {
  auto it = stats.find(rank);
  if (it == stats.end()) return 0.0;
  Stat& stat = it->second;
  return stat.maxBW;  
}

//
// Returns the worst performing tensor transpose for rank
//
double cuttTimer::getWorst(int rank) {
  auto it = stats.find(rank);
  if (it == stats.end()) return 0.0;
  Stat& stat = it->second;
  return stat.minBW;
}

//
// Returns the worst performing tensor transpose for rank
//
double cuttTimer::getWorst(int rank, std::vector<int>& dim, std::vector<int>& permutation) {
  auto it = stats.find(rank);
  if (it == stats.end()) return 0.0;
  Stat& stat = it->second;
  dim = stat.worstDim;
  permutation = stat.worstPermutation;
  return stat.minBW;
}

//
// Returns the median bandwidth for rank
//
double cuttTimer::getMedian(int rank) {
  auto it = stats.find(rank);
  if (it == stats.end()) return 0.0;
  Stat& stat = it->second;
  if (stat.minHeap.size() > stat.maxHeap.size()) {
    return stat.minHeap.top();
  } else if (stat.maxHeap.size() > stat.minHeap.size()) {
    return stat.maxHeap.top();
  } else {
    if (stat.minHeap.size() == 0) return 0.0;
    return 0.5*(stat.minHeap.top() + stat.maxHeap.top());
  }
}

//
// Returns the average bandwidth for rank
//
double cuttTimer::getAverage(int rank) {
  auto it = stats.find(rank);
  if (it == stats.end()) return 0.0;
  Stat& stat = it->second;
  return stat.totBW/(double)stat.numSample;
}

//
// Returns the worst performing tensor transpose of all
//
double cuttTimer::getWorst(std::vector<int>& dim, std::vector<int>& permutation) {
  double worstBW = 1.0e20;
  int worstRank = 0;
  for (auto it=ranks.begin(); it != ranks.end(); it++) {
    double bw = stats.find(*it)->second.minBW;
    if (worstBW > bw) {
      worstRank = *it;
      worstBW = bw;
    }
  }
  if (worstRank == 0) {
    dim.resize(0);
    permutation.resize(0);
    return 0.0;
  }
  dim = stats.find(worstRank)->second.worstDim;
  permutation = stats.find(worstRank)->second.worstPermutation;
  return worstBW;
}
