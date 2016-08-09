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
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <cmath>
#include "CudaUtils.h"
#include "cuttplan.h"
#include "cuttkernel.h"

void printMethod(int method) {
  switch(method) {
    case Trivial:
    printf("Trivial");
    break;
    case General:
    printf("General");
    break;
    case GeneralSplitInRank:
    printf("GeneralSplitInRank");
    break;
    case GeneralSplitOutRank:
    printf("GeneralSplitOutRank");
    break;
    case TiledSingleInRank:
    printf("TiledSingleInRank");
    break;
    case TiledSingleOutRank:
    printf("TiledSingleOutRank");
    break;
    case TiledSingleRank:
    printf("TiledSingleRank");
    break;
    case TiledLeadVolSame:
    printf("TiledLeadVolSame");
    break;
    case Unknown:
    printf("Unknown");
    return;
    break;
  };  
}

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

TensorSplit::TensorSplit() {
  method = Unknown;
  sizeMm = 0;
  volMm = 0;
  sizeMk = 0;
  volMk = 0;
  sizeMmk = 0;
  volMmk = 0;
  sizeMkBar = 0;
  volMkBar = 0;
  sizeMbar = 0;
  volMbar = 0;
  numSplit = 0;
}

void TensorSplit::print() {
  printf("sizeMm %d sizeMk %d sizeMmk %d sizeMbar %d sizeMkBar %d\n",
    sizeMm, sizeMk, sizeMmk, sizeMbar, sizeMkBar);
  printf("volMm %d volMk %d volMmk %d volMbar %d volMkBar %d\n",
    volMm, volMk, volMmk, volMbar, volMkBar);
  printf("numSplit %d\n", numSplit);
}

void TensorSplit::update(const int sizeMm_in, const int sizeMk_in, const int rank,
  const int* dim, const int* permutation) {

  sizeMm = sizeMm_in;
  sizeMk = sizeMk_in;

  // First sizeMm are in Mm
  volMm = 1;
  for (int i=0;i < sizeMm;i++) {
    volMm *= dim[i];
  }
  // First sizeMk in permuted order are in Mk
  volMk = 1;
  for (int i=0;i < sizeMk;i++) {
    volMk *= dim[permutation[i]];
  }

  int vol = 1;
  volMmk = 1;
  sizeMmk = 0;
  volMkBar = 1;
  sizeMkBar = 0;
  for (int i=0;i < rank;i++) {
    int pi = permutation[i];
    if (i < sizeMm) {
      volMmk *= dim[i];
      sizeMmk++;
    }
    if (i < sizeMk && pi >= sizeMm) {
      volMmk *= dim[pi];
      sizeMmk++;
      volMkBar *= dim[pi];
      sizeMkBar++;
    }
    vol *= dim[i];
  }

  sizeMbar = rank - sizeMmk;
  volMbar = vol/volMmk;
}

bool operator==(const TensorSplit& lhs, const TensorSplit& rhs) {
  if (lhs.method != rhs.method) return false;

  if (lhs.method == General) {
    return
    (lhs.sizeMmk == rhs.sizeMmk) &&
    (lhs.volMmk == rhs.volMmk) &&
    (lhs.sizeMbar == rhs.sizeMbar) &&
    (lhs.volMbar == rhs.volMbar) &&
    // (lhs.numActiveBlock == rhs.numActiveBlock) &&
    (lhs.numSplit == rhs.numSplit);
  } else {
    return
    (lhs.sizeMm == rhs.sizeMm) &&
    (lhs.volMm == rhs.volMm) &&
    (lhs.sizeMk == rhs.sizeMk) &&
    (lhs.volMk == rhs.volMk) &&
    (lhs.sizeMmk == rhs.sizeMmk) &&
    (lhs.volMmk == rhs.volMmk) &&
    (lhs.sizeMkBar == rhs.sizeMkBar) &&
    (lhs.volMkBar == rhs.volMkBar) &&
    (lhs.sizeMbar == rhs.sizeMbar) &&
    (lhs.volMbar == rhs.volMbar) &&
    // (lhs.numActiveBlock == rhs.numActiveBlock) &&
    (lhs.numSplit == rhs.numSplit);    
  }
}

//
// Number of elements in shared memory space
//
size_t TensorSplit::shmem() const {

  size_t vol = 0;

  switch(method) {

    case Trivial:
    {
      vol = 0;
    }
    break;

    case General:
    {
      vol = volMmk;
    }
    break;

    case GeneralSplitInRank:
    {
      size_t maxVolMmSplit = (volMm/numSplit) + ((volMm % numSplit) > 0);
      vol = maxVolMmSplit*(size_t)volMk;
    }
    break;

    case GeneralSplitOutRank:
    {
      size_t maxVolMkSplit = (volMk/numSplit) + ((volMk % numSplit) > 0);
      vol = maxVolMkSplit*(size_t)volMm;
    }
    break;

    case TiledSingleInRank:
    {
      vol = TILEDIM*volMk;
    }
    break;

    case TiledSingleOutRank:
    {
      vol = TILEDIM*volMm;
    }
    break;

    case TiledSingleRank:
    {
      vol = TILEDIM*TILEDIM;
    }
    break;

    case TiledLeadVolSame:
    {
      vol = 0;
    }
    break;

  }

  return vol;
}

//
// Number of elements in Mmk that are used effectively
//
size_t TensorSplit::volMmkUsed() const {
  size_t vol = 0;

  switch(method) {

    case Trivial:
    {
      vol = volMmk;
    }
    break;

    case General:
    {
      vol = volMmk;
    }
    break;

    case GeneralSplitInRank:
    {
      size_t volMmSplit = (volMm/numSplit);// + ((volMm % numSplit) > 0);
      vol = volMmSplit*(size_t)volMk;
    }
    break;

    case GeneralSplitOutRank:
    {
      size_t volMkSplit = (volMk/numSplit);// + ((volMk % numSplit) > 0);
      vol = volMkSplit*(size_t)volMm;
    }
    break;

    case TiledSingleInRank:
    {
      vol = std::min(TILEDIM, volMm)*volMk;
    }
    break;

    case TiledSingleOutRank:
    {
      vol = std::min(TILEDIM, volMk)*volMm;
    }
    break;

    case TiledSingleRank:
    {
      vol = std::min(TILEDIM, volMm)*std::min(TILEDIM, volMk);
    }
    break;

    case TiledLeadVolSame:
    {
      vol = std::min(TILEDIM, volMm)*std::min(TILEDIM, volMk);
    }
    break;

  }

  return vol;
}

//
// Bytes the shared memory space that needs to be allocated
// (can be larger than shmem() due to padding)
//
size_t TensorSplit::shmemAlloc(int sizeofType) const {
  size_t vol = 0;

  switch(method) {

    case Trivial:
    {
      vol = 0;
    }
    break;

    case General:
    {
      vol = (size_t)volMmk*sizeofType;
    }
    break;

    case GeneralSplitInRank:
    {
      size_t maxVolMmSplit = (volMm/numSplit) + ((volMm % numSplit) > 0);
      vol = maxVolMmSplit*(size_t)volMk*sizeofType + volMk*sizeof(int);
    }
    break;

    case GeneralSplitOutRank:
    {
      size_t maxVolMkSplit = (volMk/numSplit) + ((volMk % numSplit) > 0);
      vol = maxVolMkSplit*(size_t)volMm*sizeofType + volMm*sizeof(int);
    }
    break;

    case TiledSingleInRank:
    {
      vol = (TILEDIM+1)*(size_t)volMk*sizeofType;
    }
    break;

    case TiledSingleOutRank:
    {
      vol = TILEDIM*(size_t)volMm*sizeofType;
    }
    break;

    case TiledSingleRank:
    {
      vol = (TILEDIM+1)*TILEDIM*sizeofType;
    }
    break;

    case TiledLeadVolSame:
    {
      vol = 0;
    }
    break;

  }

  return vol;
}

//
// Turns out that combining ranks does not lead to better performance in general
//
#if 0
//
// Reduce ranks by combining groups of ranks are in consequtive order
//
void reduceRanks(const int rank, const int* dim, const int* permutation,
  std::vector<int>& redDim, std::vector<int>& redPermutation) {

  // Previous permutation value,
  // start with impossible value so that we always do push_back(permutation[0])
  int prev = -2;
  for (int i=0;i < rank;i++) {
    int cur = permutation[i];
    if (cur == prev + 1) {
      // Skip over ranks that are in consequtive order and
      // combine dimensions
      redDim.back() *= dim[cur];
    } else {
      // Include ranks that start the consequtive sequence
      redPermutation.push_back(cur);
      // NOTE: redDim will be in permuted order, re-order after dust settles
      redDim.push_back(dim[cur]);
    }
    prev = cur;
  }

  // Re-number redPermutation
  std::vector<int> tmp(rank, -1);
  for (int i=0;i < redPermutation.size();i++) {
    tmp[redPermutation[i]] = i;
  }
  int j = 0;
  for (int i=0;i < rank;i++) {
    if (tmp[i] != -1) {
      tmp[j++] = tmp[i];
    }
  }
  for (int i=0;i < redPermutation.size();i++) {
    redPermutation[tmp[i]] = i;
  }

  // Re-order redDim
  for (int i=0;i < redDim.size();i++) {
    tmp[redPermutation[i]] = redDim[i];
  }  
  for (int i=0;i < redDim.size();i++) {
    redDim[i] = tmp[i];
  }

  for (int i=0;i < rank;i++) {
    printf("%d ", dim[i]);
  }
  printf("| ");
  for (int i=0;i < rank;i++) {
    printf("%d ", permutation[i]);
  }
  printf("\n");

  for (int i=0;i < redPermutation.size();i++) {
    printf("%d ", redDim[i]);
  }
  printf("| ");
  for (int i=0;i < redPermutation.size();i++) {
    printf("%d ", redPermutation[i]);
  }
  printf("\n");

}
#endif

bool createTrivialPlans(const int rank, const int* dim, const int* permutation,
  const size_t sizeofType, cudaDeviceProp& prop, std::list<cuttPlan_t>& plans) {

  if (rank == 1) {
    TensorSplit ts;
    ts.method = Trivial;
    ts.update(1, 1, rank, dim, permutation);    
    LaunchConfig lc;
    int numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, ts, prop, lc);
    if (numActiveBlock > 0) {
      cuttPlan_t plan;
      if (!plan.setup(rank, dim, permutation, sizeofType, prop, ts)) return false;
      plans.push_back(plan);
    }
  }

  return true;
}

bool createTiledSingleInRankPlans(const int rank, const int* dim, const int* permutation,
  const size_t sizeofType, cudaDeviceProp& prop, std::list<cuttPlan_t>& plans) {

  for (int numMk=2;numMk < rank;numMk++) {
    TensorSplit ts;
    ts.method = TiledSingleInRank;
    ts.update(1, numMk, rank, dim, permutation);
    // If Mm and Mk overlap, break out of the loop
    if (ts.sizeMmk != numMk + 1) break;
    LaunchConfig lc;
    int numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, ts, prop, lc);
    if (numActiveBlock > 0) {
      cuttPlan_t plan;
      if (!plan.setup(rank, dim, permutation, sizeofType, prop, ts)) return false;
      plans.push_back(plan);
    }
  }

  return true;
}

bool createTiledSingleOutRankPlans(const int rank, const int* dim, const int* permutation,
  const size_t sizeofType, cudaDeviceProp& prop, std::list<cuttPlan_t>& plans) {

  for (int numMm=2;numMm < rank;numMm++) {
    TensorSplit ts;
    ts.method = TiledSingleOutRank;
    ts.update(numMm, 1, rank, dim, permutation);
    // If Mm and Mk overlap, break out of the loop
    if (ts.sizeMmk != numMm + 1) break;
    LaunchConfig lc;
    int numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, ts, prop, lc);
    if (numActiveBlock > 0) {
      cuttPlan_t plan;
      if (!plan.setup(rank, dim, permutation, sizeofType, prop, ts)) return false;
      plans.push_back(plan);
    }
  }

  return true;
}

bool createTiledSingleRankPlans(const int rank, const int* dim, const int* permutation,
  const size_t sizeofType, cudaDeviceProp& prop, std::list<cuttPlan_t>& plans) {

  if (permutation[0] != 0 && rank > 1) {
    TensorSplit ts;
    ts.method = TiledSingleRank;
    ts.update(1, 1, rank, dim, permutation);    
    LaunchConfig lc;
    int numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, ts, prop, lc);
    if (numActiveBlock > 0) {
      cuttPlan_t plan;
      if (!plan.setup(rank, dim, permutation, sizeofType, prop, ts)) return false;
      plans.push_back(plan);
    }
  }

  return true;
}

bool createTiledLeadVolSamePlans(const int rank, const int* dim, const int* permutation,
  const size_t sizeofType, cudaDeviceProp& prop, std::list<cuttPlan_t>& plans) {

  // Count number of Mm and Mk which are the same
  int numMmMkSame = 0;
  while (numMmMkSame < rank && permutation[numMmMkSame] == numMmMkSame) {
    numMmMkSame++;
  }
  if (numMmMkSame >= 1) {
    TensorSplit ts;
    ts.method = TiledLeadVolSame;
    if (numMmMkSame < rank) {
      ts.update(numMmMkSame, numMmMkSame + 1, rank, dim, permutation);      
    } else {
      ts.update(numMmMkSame - 1, numMmMkSame, rank, dim, permutation);      
    }
    LaunchConfig lc;
    int numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, ts, prop, lc);
    if (numActiveBlock > 0) {
      cuttPlan_t plan;
      if (!plan.setup(rank, dim, permutation, sizeofType, prop, ts)) return false;
      plans.push_back(plan);
    }
  }

  return true;
}

bool createGeneralPlans(const int rank, const int* dim, const int* permutation,
  const size_t sizeofType, cudaDeviceProp& prop, std::list<cuttPlan_t>& plans) {

  // Stores TensorSplits that have already been added (in order to avoid duplicates)
  std::vector<TensorSplit> tsAdded;

  LaunchConfig lc;
  for (int numMm=1;numMm < rank;numMm++) {
    for (int numMk=1;numMk < rank;numMk++) {
      TensorSplit ts;
      ts.method = General;
      ts.update(numMm, numMk, rank, dim, permutation);
      int numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, ts, prop, lc);
      // If we can't fit to device, break out from inner loop
      if (numActiveBlock == 0) break;
      // Do not add multiple copies of the same plan
      bool multiple = false;
      for (int i=0;i < tsAdded.size();i++) {
        if (tsAdded[i] == ts) {
          multiple = true;
          break;
        }
      }
      if (multiple) continue;
      tsAdded.push_back(ts);
      cuttPlan_t plan;
      if (!plan.setup(rank, dim, permutation, sizeofType, prop, ts)) return false;
      plans.push_back(plan);
    }
  }

  return true;
}

bool createGeneralSplitInRankPlans(const int rank, const int* dim, const int* permutation,
  const size_t sizeofType, cudaDeviceProp& prop, std::list<cuttPlan_t>& plans) {

  LaunchConfig lc;
  for (int numMk=2;numMk < rank;numMk++) {
    TensorSplit ts;
    ts.method = GeneralSplitInRank;
    ts.update(1, numMk, rank, dim, permutation);
    // If Mm and Mk overlap, break out of the loop
    if (ts.sizeMmk != numMk + 1) break;
    // Determine number of splits
    ts.numSplit = 1;
    int shmemsize = 0;
    do {
      ts.numSplit++;
      int maxVolMmSplit = (ts.volMm/ts.numSplit) + ((ts.volMm % ts.numSplit) > 0);
      shmemsize = maxVolMmSplit*ts.volMk*sizeofType + ts.volMk*sizeof(int);
    } while (shmemsize > prop.sharedMemPerBlock && ts.numSplit < 8);
    if (shmemsize > prop.sharedMemPerBlock) continue;
    int numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, ts, prop, lc);
    if (numActiveBlock > 0) {
      cuttPlan_t plan;
      if (!plan.setup(rank, dim, permutation, sizeofType, prop, ts)) return false;
      plans.push_back(plan);
    }
  }

  return true;
}

bool createGeneralSplitOutRankPlans(const int rank, const int* dim, const int* permutation,
  const size_t sizeofType, cudaDeviceProp& prop, std::list<cuttPlan_t>& plans) {

  LaunchConfig lc;
  for (int numMm=2;numMm < rank;numMm++) {
    TensorSplit ts;
    ts.method = GeneralSplitOutRank;
    ts.update(numMm, 1, rank, dim, permutation);
    // If Mm and Mk overlap, break out of the loop
    if (ts.sizeMmk != numMm + 1) break;
    // Determine number of splits
    ts.numSplit = 1;
    int shmemsize = 0;
    do {
      ts.numSplit++;
      int maxVolMkSplit = (ts.volMk/ts.numSplit) + ((ts.volMk % ts.numSplit) > 0);
      shmemsize = maxVolMkSplit*ts.volMm*sizeofType + ts.volMm*sizeof(int);
    } while (shmemsize > prop.sharedMemPerBlock && ts.numSplit < 8);
    if (shmemsize > prop.sharedMemPerBlock) continue;
    int numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, ts, prop, lc);
    if (numActiveBlock > 0) {
      cuttPlan_t plan;
      if (!plan.setup(rank, dim, permutation, sizeofType, prop, ts)) return false;
      plans.push_back(plan);
    }
  }

  return true;
}

//
// Create all possible plans
//
bool createPlans(const int rank, const int* dim, const int* permutation, const size_t sizeofType,
  cudaDeviceProp& prop, std::list<cuttPlan_t>& plans) {

  if (!createTrivialPlans(rank, dim, permutation, sizeofType, prop, plans)) return false;
  if (!createTiledLeadVolSamePlans(rank, dim, permutation, sizeofType, prop, plans)) return false;
  if (!createTiledSingleInRankPlans(rank, dim, permutation, sizeofType, prop, plans)) return false;
  if (!createTiledSingleOutRankPlans(rank, dim, permutation, sizeofType, prop, plans)) return false;
  if (!createTiledSingleRankPlans(rank, dim, permutation, sizeofType, prop, plans)) return false;
  if (!createGeneralPlans(rank, dim, permutation, sizeofType, prop, plans)) return false;
  if (!createGeneralSplitInRankPlans(rank, dim, permutation, sizeofType, prop, plans)) return false;
  if (!createGeneralSplitOutRankPlans(rank, dim, permutation, sizeofType, prop, plans)) return false;

  return true;
}

bool comp_memaccess(const cuttPlan_t& lhs, const cuttPlan_t& rhs) {
  const TensorSplit& lts = lhs.tensorSplit;
  const TensorSplit& rts = rhs.tensorSplit;

  double dp = fabs(((double)lhs.numMemAccess - (double)rhs.numMemAccess)/(double)std::min(lhs.numMemAccess, rhs.numMemAccess));
  // printf("dp %lf\n", dp);
  if (dp < 0.20) {
    // If number of mem accesses are close, choose the one with largest Mmk
    size_t lhs_nab = (lts.method == General) ? std::min(20, lhs.numActiveBlock) : lhs.numActiveBlock;
    size_t rhs_nab = (rts.method == General) ? std::min(20, rhs.numActiveBlock) : rhs.numActiveBlock;
    lhs_nab = std::min((size_t)4, lhs_nab);
    rhs_nab = std::min((size_t)4, rhs_nab);
    // printf("%d %d\n", lts.volMmkUsed()*lhs_nab, rts.volMmkUsed()*rhs_nab);
    return (lts.volMmkUsed()*lhs_nab > rts.volMmkUsed()*rhs_nab);
  } else {
    return (lhs.numMemAccess < rhs.numMemAccess);
  }

}

bool operator>(const cuttPlan_t& lhs, const cuttPlan_t& rhs) {
  // return (lhs.numMemAccess < rhs.numMemAccess);

  const TensorSplit& lts = lhs.tensorSplit;
  const TensorSplit& rts = rhs.tensorSplit;

  // Trivial method always wins
  if (lts.method == Trivial) return true;
  if (rts.method == Trivial) return false;

#if 1
  // General method wins over TiledLeadVolSame when volMmk is small
  // (empirical observation on Tesla K20x)
  if (lts.method == General && rts.method == TiledLeadVolSame && rts.volMmk <= 8000) {
    return true;
  }
  if (lts.method == TiledLeadVolSame && lts.volMmk <= 8000 && rts.method == General) {
    return false;
  }

  // if (lts.method == General && rts.method == TiledSingleOutRank) {
  //   return true;
  // }
  // if (lts.method == TiledSingleOutRank && rts.method == General) {
  //   return false;
  // }

  // if (lts.method == General && rts.method == TiledSingleInRank) {
  //   return true;
  // }
  // if (lts.method == TiledSingleInRank && rts.method == General) {
  //   return false;
  // }

  // if (lts.method == General && rts.method == GeneralSplitOutRank) {
  //   return true;
  // }
  // if (lts.method == GeneralSplitOutRank && rts.method == General) {
  //   return false;
  // }

  // if (lts.method == General && rts.method == GeneralSplitInRank) {
  //   return true;
  // }
  // if (lts.method == GeneralSplitInRank && rts.method == General) {
  //   return false;
  // }

  return comp_memaccess(lhs, rhs);
#else
  if (
    // * All self comparisons (7)
    lts.method == rts.method ||
    // * TiledSingleInRank vs. TiledSingleOutRank (2)
    (lts.method == TiledSingleInRank && rts.method == TiledSingleOutRank) ||
    (lts.method == TiledSingleOutRank && rts.method == TiledSingleInRank) ||
    // * GeneralSplitInRank vs. GeneralSplitOutRank (2)
    (lts.method == GeneralSplitInRank && rts.method == GeneralSplitOutRank) ||
    (lts.method == GeneralSplitOutRank && rts.method == GeneralSplitInRank) ||
    // * General vs. GeneralSplitInRank (2)
    (lts.method == General && rts.method == GeneralSplitInRank) ||
    (lts.method == GeneralSplitInRank && rts.method == General) ||
    // * General vs. GeneralSplitOutRank (2)
    (lts.method == General && rts.method == GeneralSplitOutRank) ||
    (lts.method == GeneralSplitOutRank && rts.method == General)
    ) {

    return comp_memaccess(lhs, rhs);

    // double dp = fabs(((double)lhs.numMemAccess - (double)rhs.numMemAccess)/(double)std::min(lhs.numMemAccess, rhs.numMemAccess));
    // // printf("dp %lf\n", dp);
    // if (dp < 0.15) {
    //   // If number of mem accesses are close, choose the one with largest Mmk
    //   size_t lhs_nab = (lts.method == General) ? std::min(20, lhs.numActiveBlock) : lhs.numActiveBlock;
    //   size_t rhs_nab = (rts.method == General) ? std::min(20, rhs.numActiveBlock) : rhs.numActiveBlock;
    //   lhs_nab = std::min((size_t)6, lhs_nab);
    //   rhs_nab = std::min((size_t)6, rhs_nab);
    //   // printf("%d %d\n", lts.volMmkUsed()*lhs_nab, rts.volMmkUsed()*rhs_nab);
    //   return (lts.volMmkUsed()*lhs_nab > rts.volMmkUsed()*rhs_nab);
    //   // return (lts.volMmkUsed() > rts.volMmkUsed());
    // } else {
    //   return (lhs.numMemAccess < rhs.numMemAccess);
    // }

    // // if (lts.method == General && rts.method == General) {
    //   double dp = fabs(((double)lhs.numMemAccess - (double)rhs.numMemAccess)/(double)rhs.numMemAccess);
    //   if (dp < 0.15) {
    //     // If number of mem accesses are close, choose the one with largest Mmk
    //     // return (lts.volMmkUsed() > rts.volMmkUsed());
    //     size_t lhs_nab = (lts.method == General) ? std::min(2, lts.numActiveBlock) : lts.numActiveBlock;
    //     size_t rts_nab = (rts.method == General) ? std::min(2, rts.numActiveBlock) : rts.numActiveBlock;
    //     return (lts.volMmkUsed()*lhs_nab > rts.volMmkUsed()*rts_nab);
    //   } else {
    //     return (lhs.numMemAccess < rhs.numMemAccess);
    //   }
    // // }

/*
    if (lhs.numMemAccess == rhs.numMemAccess) {
      if (lts.method == General && rts.method == General) {
      //   return (lhs.numTransPerAccess < rhs.numTransPerAccess);
      } else {
        return (lts.volMmkUsed() > rts.volMmkUsed());
        // size_t lhs_nab = (lts.method == General) ? std::min(2, lts.numActiveBlock) : lts.numActiveBlock;
        // size_t rts_nab = (rts.method == General) ? std::min(2, rts.numActiveBlock) : rts.numActiveBlock;
        // return (lts.volMmkUsed()*lhs_nab > rts.volMmkUsed()*rts_nab);
      }
    } else {
      return (lhs.numMemAccess < rhs.numMemAccess);
    }
*/

  } else {
    const int MIN_TILED_DIM = 20; //TILEDIM/2;
    // * TiledLeadVolSame vs. TiledSingleRank || TiledSingleInRank || TiledSingleOutRank (6)
    //   TiledLeadVolSame always wins
    if (lts.method == TiledLeadVolSame && 
      (rts.method == TiledSingleRank || rts.method == TiledSingleInRank || rts.method == TiledSingleOutRank)) {
      return true;
    } else if (rts.method == TiledLeadVolSame && 
      (lts.method == TiledSingleRank || lts.method == TiledSingleInRank || lts.method == TiledSingleOutRank)) {
      return !(rhs > lhs);
    }
    // * TiledLeadVolSame vs. General || GeneralSplitInRank || GeneralSplitOutRank (6)
    if (lts.method == TiledLeadVolSame &&
      (rts.method == General || rts.method == GeneralSplitInRank || rts.method == GeneralSplitOutRank)) {
      return (lts.volMm >= MIN_TILED_DIM && lts.volMk >= MIN_TILED_DIM) || comp_memaccess(lhs, rhs);
      // return (
      // (lts.volMm >= MIN_TILED_DIM && lts.volMk >= MIN_TILED_DIM) ||
      // ((lts.volMm >= MIN_TILED_DIM || lts.volMk >= MIN_TILED_DIM) && lts.volMmkUsed() >= rts.volMmkUsed())
      // );
    } else if (rts.method == TiledLeadVolSame &&
      (lts.method == General || lts.method == GeneralSplitInRank || lts.method == GeneralSplitOutRank)) {
      return !(rhs > lhs);
    }
    // * TiledSingleRank vs. TiledSingleInRank (2)
    //   TiledSingleInRank wins if its sizeMk > 1
    if (lts.method == TiledSingleRank && rts.method == TiledSingleInRank) {
      return (rts.sizeMk == 1);
    } else if (rts.method == TiledSingleRank && lts.method == TiledSingleInRank) {
      return !(rhs > lhs);
    }
    // * TiledSingleRank vs. TiledSingleOutRank (2)
    //   TiledSingleOutRank wins if its sizeMm > 1
    if (lts.method == TiledSingleRank && rts.method == TiledSingleOutRank) {
      return (rts.sizeMm == 1);
    } else if (rts.method == TiledSingleRank && lts.method == TiledSingleOutRank) {
      return !(rhs > lhs);
    }
    // * TiledSingleRank vs. General || GeneralSplitInRank || GeneralSplitOutRank (6)
    if (lts.method == TiledSingleRank &&
      (rts.method == General || rts.method == GeneralSplitInRank || rts.method == GeneralSplitOutRank)) {
      return (lts.volMm >= MIN_TILED_DIM && lts.volMk >= MIN_TILED_DIM) || comp_memaccess(lhs, rhs);
      // return (
      // (lts.volMm >= MIN_TILED_DIM && lts.volMk >= MIN_TILED_DIM) ||
      // ((lts.volMm >= MIN_TILED_DIM || lts.volMk >= MIN_TILED_DIM) && lts.volMmkUsed() >= rts.volMmkUsed())
      // );
      // return (lhs.numMemAccess < rhs.numMemAccess);
    } else if (rts.method == TiledSingleRank &&
      (lts.method == General || lts.method == GeneralSplitInRank || lts.method == GeneralSplitOutRank)) {
      return !(rhs > lhs);
    }
    // * TiledSingleInRank vs. General || GeneralSplitInRank || GeneralSplitOutRank (6)
    if (lts.method == TiledSingleInRank &&
      (rts.method == General || rts.method == GeneralSplitInRank || rts.method == GeneralSplitOutRank)) {
      return (lts.volMm >= MIN_TILED_DIM && lts.volMk >= MIN_TILED_DIM) || comp_memaccess(lhs, rhs);
      // return (
      // (lts.volMm >= MIN_TILED_DIM && lts.volMk >= MIN_TILED_DIM) ||
      // ((lts.volMm >= MIN_TILED_DIM || lts.volMk >= MIN_TILED_DIM) && lts.volMmkUsed() >= rts.volMmkUsed())
      // );
      // return (lhs.numMemAccess < rhs.numMemAccess);
    } else if (rts.method == TiledSingleInRank &&
      (lts.method == General || lts.method == GeneralSplitInRank || lts.method == GeneralSplitOutRank)) {
      return !(rhs > lhs);
    }
    // * TiledSingleOutRank vs. General || GeneralSplitInRank || GeneralSplitOutRank (6)
    if (lts.method == TiledSingleOutRank &&
      (rts.method == General || rts.method == GeneralSplitInRank || rts.method == GeneralSplitOutRank)) {
      return (lts.volMm >= MIN_TILED_DIM && lts.volMk >= MIN_TILED_DIM) || comp_memaccess(lhs, rhs);
      // return (
      // (lts.volMm >= MIN_TILED_DIM && lts.volMk >= MIN_TILED_DIM) ||
      // ((lts.volMm >= MIN_TILED_DIM || lts.volMk >= MIN_TILED_DIM) && lts.volMmkUsed() >= rts.volMmkUsed())
      // );
      // return (lhs.numMemAccess < rhs.numMemAccess);
    } else if (rts.method == TiledSingleOutRank &&
      (lts.method == General || lts.method == GeneralSplitInRank || lts.method == GeneralSplitOutRank)) {
      return !(rhs > lhs);
    }
  }
#endif
  // We should not end up here
  printf("bool operator>(const cuttPlan_t& lhs, const cuttPlan_t& rhs): FATAL implementation bug with:\n");
  printf("lts.method ");
  printMethod(lts.method);
  printf(" rts.method ");
  printMethod(rts.method);
  printf("\n");
  exit(1);

}

bool operator<(const cuttPlan_t& lhs, const cuttPlan_t& rhs) {
  return !(lhs > rhs);
}

//
// Choose best plan among the same method
//
void reducePlans(std::list<cuttPlan_t>& plans, int method) {
  // Find the best
  bool foundBest = false;
  auto bestIt = plans.end();
  for (auto it = plans.begin();it != plans.end();it++) {
    if (it->tensorSplit.method == method) {
      if (foundBest == false || *it > *bestIt) {
        foundBest = true;
        bestIt = it;
      }
    }
  }
  if (!foundBest) return;
  // Remove all but the best
  for (auto it=plans.begin();it != plans.end();) {
    if (it->tensorSplit.method == method && it != bestIt) {
      it = plans.erase(it++);
    } else {
      it++;
    }
  }
}

//
// Returns best plan according to heuristic criteria
// Returns plans.end() on invalid input or when nothing can be chosen
//
std::list<cuttPlan_t>::iterator choosePlanHeuristic(std::list<cuttPlan_t>& plans) {

  for (int method = Trivial;method < NumTransposeMethods;method++) {
    reducePlans(plans, method);
  }

  // Choose the "largest" plan
  auto bestIt = plans.end();
  for (auto it=plans.begin();it != plans.end();it++) {
    // it->print();
    if (bestIt == plans.end() || *bestIt < *it) {
      bestIt = it;
      // bestIt->print();
    }
  }
  // bestIt = plans.begin();

  return bestIt;
}

void LaunchConfig::print() {
  printf("numthread %d %d %d numblock %d %d %d shmemsize %d numRegStorage %d\n",
    numthread.x, numthread.y, numthread.z,
    numblock.x, numblock.y, numblock.z,
    (int)shmemsize, numRegStorage);
}

//
// Output contents of the plan
//
void cuttPlan_t::print() {
  printf("method ");
  printMethod(tensorSplit.method);
  printf("\n");
  tensorSplit.print();
  launchConfig.print();
  printf("numActiveBlock %d\n", numActiveBlock);
  printf("numMemAccess %llu %llu %llu numTransPerAccess %1.2f\n", numRead, numWrite, numMemAccess, numTransPerAccess);
}

//
// Setup plan
//
bool cuttPlan_t::setup(const int rank_in, const int* dim, const int* permutation,
  const size_t sizeofType_in, cudaDeviceProp& prop, TensorSplit& tensorSplit_in) {
  
  rank = rank_in;
  sizeofType = sizeofType_in;
  tensorSplit = tensorSplit_in;

  std::vector<bool> isMm(rank, false);
  std::vector<bool> isMk(rank, false);
  for (int i=0;i < tensorSplit.sizeMm;i++) {
    isMm[i] = true;
  }
  for (int i=0;i < tensorSplit.sizeMk;i++) {
    isMk[permutation[i]] = true;
  }

  // Setup launch configuration
  numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, tensorSplit, prop, launchConfig);
  if (numActiveBlock == 0) return false;

  // Build cI
  int* I = new int[rank];
  for (int i=0;i < rank;i++) {
    I[i] = i;
  }
  TensorC cI(rank, rank, I, dim);
  delete [] I;

  // Build cO
  TensorC cO(rank, rank, permutation, dim);

  if (tensorSplit.method == TiledSingleRank) {
    cuDimMk = cI.get(permutation[0]);
    cuDimMm = cO.get(0);
    tiledVol.x = dim[0];
    tiledVol.y = dim[permutation[0]];
  } else if (tensorSplit.method == TiledLeadVolSame) {
    int rankMk = permutation[tensorSplit.sizeMk - 1];
    cuDimMk = cI.get(rankMk);
    cuDimMm = cO.get(rankMk);
    tiledVol.x = tensorSplit.volMm;
    tiledVol.y = dim[rankMk];
  }

  // Build MmI
  std::vector<int> MmI(tensorSplit.sizeMm);
  {
    int iMm = 0;
    int iMk = 0;
    for (int i=0;i < rank;i++) {
      if (isMm[i]) {
        MmI[iMm++] = i;
      }
    }
  }

  if (tensorSplit.sizeMbar > 0) {
    // Build MbarI = {s_1, ...., s_h}, indices in input order
    int* MbarI = new int[tensorSplit.sizeMbar];
    int j = 0;
    for (int i=0;i < rank;i++) {
      if (!(isMm[i] || isMk[i])) {
        MbarI[j] = i;
        j++;
      }
    }
    TensorC cMbarI(rank, tensorSplit.sizeMbar, MbarI, dim);

    // Build MbarO = {s_l1, ...., s_lh}, indices in output (permuted) order
    int* MbarO = new int[tensorSplit.sizeMbar];
    j = 0;
    for (int i=0;i < rank;i++) {
      int pi = permutation[i];
      if (!(isMm[pi] || isMk[pi])) {
        MbarO[j] = pi;
        j++;
      }
    }

    hostMbar.resize(tensorSplit.sizeMbar);
    for (int i=0;i < tensorSplit.sizeMbar;i++) {
      int si = MbarI[i];
      hostMbar[i].c_in  = cMbarI.get(si);
      hostMbar[i].d_in  = dim[si];
      hostMbar[i].ct_in = cI.get(si);
      int sli = MbarO[i];
      hostMbar[i].c_out  = cMbarI.get(sli);
      hostMbar[i].d_out  = dim[sli];
      hostMbar[i].ct_out = cO.get(sli);
    }

    delete [] MbarI;
    delete [] MbarO;
  }

  numTransPerAccess = 1.0f;

  // TensorConv* hostMsh1 = NULL;
  // TensorConv* hostMsh2 = NULL;
  if (tensorSplit.method == General) {
    // Build MmkI = {q_1, ..., q_a}
    std::vector<int> MmkI(tensorSplit.sizeMmk);
    int j = 0;
    for (int i=0;i < rank;i++) {
      if (isMm[i] || isMk[i]) {
        MmkI[j] = i;
        j++;
      }
    }
    TensorC cMmkI(rank, tensorSplit.sizeMmk, MmkI.data(), dim);
    // Build MmkO = {q_t1, ..., q_ta}
    std::vector<int> MmkO(tensorSplit.sizeMmk);
    j = 0;
    for (int i=0;i < rank;i++) {
      int pi = permutation[i];
      if (isMm[pi] || isMk[pi]) {
        MmkO[j] = pi;
        j++;
      }
    }
    TensorC cMmkO(rank, tensorSplit.sizeMmk, MmkO.data(), dim);

    hostMmk.resize(tensorSplit.sizeMmk);
    for (int i=0;i < tensorSplit.sizeMmk;i++) {
      // Minor reading position
      int qi = MmkI[i];
      hostMmk[i].c_in  = cMmkI.get(qi);
      hostMmk[i].d_in  = dim[qi];
      hostMmk[i].ct_in = cI.get(qi);
      // Minor writing position
      int qti = MmkO[i];
      hostMmk[i].c_out  = cMmkO.get(qti);
      hostMmk[i].d_out  = dim[qti];
      hostMmk[i].ct_out = cO.get(qti);
    }

    hostMsh.resize(tensorSplit.sizeMmk);
    for (int i=0;i < tensorSplit.sizeMmk;i++) {
      // Shared memory reading position
      int qti = MmkO[i];
      hostMsh[i].c  = cMmkO.get(qti);
      hostMsh[i].d  = dim[qti];
      hostMsh[i].ct = cMmkI.get(qti);
    }

    int numTotAccess = 0;
    int numIdealAccess = 0;
    {
      for (int j0=0;j0 < tensorSplit.volMmk;j0+=prop.warpSize)
      {
        // Number of accesses for each bank
        std::vector<int> numAccess(prop.warpSize, 0);
        for (int j1=0;j1 < prop.warpSize;j1++) {
          int j = j0 + j1;
          if (j < tensorSplit.volMmk) {
            int pos = 0;
            for (int i=0;i < tensorSplit.sizeMmk;i++) {
              pos += ((j / hostMsh[i].c) % hostMsh[i].d) * hostMsh[i].ct;
            }
            int bank = pos % prop.warpSize;
            ++numAccess[bank];
          }
        }
        int maxNumAccess = 0;
        for (int i=0;i < prop.warpSize;i++) {
          maxNumAccess = std::max(maxNumAccess, numAccess[i]);
        }
        numTotAccess += maxNumAccess;
        numIdealAccess++;
      }
      // Calculate number of transactions per access
      numTransPerAccess = (float)numTotAccess/(float)numIdealAccess;
    }

  }

  if (tensorSplit.method == TiledSingleInRank) {
    cuDimMm = cO.get(0);

    // Build MkO = {p_t1, ..., p_tb}
    std::vector<int> MkO(tensorSplit.sizeMk);
    {
      int j = 0;
      for (int i=0;i < rank;i++) {
        int pi = permutation[i];
        if (isMk[pi]) {
          MkO[j] = pi;
          j++;
        }
      }
    }
    TensorC cMkO(rank, tensorSplit.sizeMk, MkO.data(), dim);

    hostMk.resize(tensorSplit.sizeMk);
    for (int i=0;i < tensorSplit.sizeMk;i++) {
      int pti = MkO[i];
      // Global memory read position
      hostMk[i].c  = cMkO.get(pti);
      hostMk[i].d  = dim[pti];
      hostMk[i].ct = cI.get(pti);
    }

  }

  if (tensorSplit.method == TiledSingleOutRank) {
    cuDimMk = cI.get(permutation[0]);

    TensorC cMmI(rank, tensorSplit.sizeMm, MmI.data(), dim);

    hostMm.resize(tensorSplit.sizeMm);
    for (int i=0;i < tensorSplit.sizeMm;i++) {
      int pi = MmI[i];
      // Global memory write position
      hostMm[i].c  = cMmI.get(pi);
      hostMm[i].d  = dim[pi];
      hostMm[i].ct = cO.get(pi);
    }

  }

  if (tensorSplit.method == GeneralSplitInRank) {
    cuDimMm = cO.get(0);

    // Build MkO = {p_t1, ..., p_tb}
    std::vector<int> MkO(tensorSplit.sizeMk);
    {
      int j = 0;
      for (int i=0;i < rank;i++) {
        int pi = permutation[i];
        if (isMk[pi]) {
          MkO[j++] = pi;
        }
      }
    }
    TensorC cMkO(rank, tensorSplit.sizeMk, MkO.data(), dim);

    hostMk.resize(tensorSplit.sizeMk);
    for (int i=0;i < tensorSplit.sizeMk;i++) {
      int pti = MkO[i];
      // Global memory read position
      hostMk[i].c  = cMkO.get(pti);
      hostMk[i].d  = dim[pti];
      hostMk[i].ct = cI.get(pti);
    }

    hostPosMk.resize(tensorSplit.volMk);
    for (int j=0;j < tensorSplit.volMk;j++) {
      hostPosMk[j] = 0;
      for (int i=0;i < tensorSplit.sizeMk;i++) {
        hostPosMk[j] += ((j / hostMk[i].c) % hostMk[i].d) * hostMk[i].ct;
      }
    }

  }

  if (tensorSplit.method == GeneralSplitOutRank) {
    cuDimMk = cI.get(permutation[0]);

    TensorC cMmI(rank, tensorSplit.sizeMm, MmI.data(), dim);

    hostMm.resize(tensorSplit.sizeMm);
    for (int i=0;i < tensorSplit.sizeMm;i++) {
      int pi = MmI[i];
      // Global memory write position
      hostMm[i].c  = cMmI.get(pi);
      hostMm[i].d  = dim[pi];
      hostMm[i].ct = cO.get(pi);
    }

    hostPosMm.resize(tensorSplit.volMm);
    for (int j=0;j < tensorSplit.volMm;j++) {
      hostPosMm[j] = 0;
      for (int i=0;i < tensorSplit.sizeMm;i++) {
        hostPosMm[j] += ((j / hostMm[i].c) % hostMm[i].d) * hostMm[i].ct;
      }
    }

  }

  cuttKernelNumMemAccess(tensorSplit, prop,
    launchConfig, rank, dim, permutation, sizeofType,
    numRead, numWrite);

  numMemAccess = numRead + numWrite;

  return true;
}

//
// Activates the plan: Allocates device memory buffers and copies data to them
//
void cuttPlan_t::activate() {

  if (tensorSplit.sizeMbar > 0) {
    allocate_device<TensorConvInOut>(&Mbar, tensorSplit.sizeMbar);
    copy_HtoD<TensorConvInOut>(hostMbar.data(), Mbar, tensorSplit.sizeMbar, stream);
  }

  if (tensorSplit.method == General) {
    allocate_device<TensorConvInOut>(&Mmk, tensorSplit.sizeMmk);
    copy_HtoD<TensorConvInOut>(hostMmk.data(), Mmk, tensorSplit.sizeMmk, stream);
    allocate_device<TensorConv>(&Msh, tensorSplit.sizeMmk);
    copy_HtoD<TensorConv>(hostMsh.data(), Msh, tensorSplit.sizeMmk, stream);
  }

  if (tensorSplit.method == TiledSingleInRank) {
    allocate_device<TensorConv>(&Mk, tensorSplit.sizeMk);
    copy_HtoD<TensorConv>(hostMk.data(), Mk, tensorSplit.sizeMk, stream);
  }

  if (tensorSplit.method == TiledSingleOutRank) {
    allocate_device<TensorConv>(&Mm, tensorSplit.sizeMm);
    copy_HtoD<TensorConv>(hostMm.data(), Mm, tensorSplit.sizeMm, stream);
  }

  if (tensorSplit.method == GeneralSplitInRank) {
    allocate_device<int>(&posMk, tensorSplit.volMk);
    copy_HtoD<int>(hostPosMk.data(), posMk, tensorSplit.volMk, stream);
  }

  if (tensorSplit.method == GeneralSplitOutRank) {
    allocate_device<int>(&posMm, tensorSplit.volMm);
    copy_HtoD<int>(hostPosMm.data(), posMm, tensorSplit.volMm, stream);
  }

}

//
// Set device buffers to NULL
//
void cuttPlan_t::nullDevicePointers() {
  Mbar = NULL;
  Mmk = NULL;
  Msh = NULL;
  Mk = NULL;
  Mm = NULL;
  posMk = NULL;
  posMm = NULL;
}

cuttPlan_t::cuttPlan_t() {
  cudaCheck(cudaGetDevice(&deviceID));
  stream = 0;
  numActiveBlock = 0;
  nullDevicePointers();
}

cuttPlan_t::~cuttPlan_t() {
  // Deallocate device buffers
  if (Mbar != NULL) deallocate_device<TensorConvInOut>(&Mbar);
  if (Mmk != NULL) deallocate_device<TensorConvInOut>(&Mmk);
  if (Msh != NULL) deallocate_device<TensorConv>(&Msh);
  if (Mk != NULL) deallocate_device<TensorConv>(&Mk);
  if (Mm != NULL) deallocate_device<TensorConv>(&Mm);
  if (posMk != NULL) deallocate_device<int>(&posMk);
  if (posMm != NULL) deallocate_device<int>(&posMm);
}

void cuttPlan_t::setStream(cudaStream_t stream_in) {
  stream = stream_in;
}
