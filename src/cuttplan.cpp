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
// #include <cstring>
#include <algorithm>
#include "CudaUtils.h"
#include "cuttplan.h"
#include "cuttkernel.h"

void printMethod(int method) {
  switch(method) {
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
  numActiveBlock = 0;
  numSplit = 0;
}

void TensorSplit::print() {
  printf("sizeMm %d sizeMk %d sizeMmk %d sizeMbar %d sizeMkBar %d\n",
    sizeMm, sizeMk, sizeMmk, sizeMbar, sizeMkBar);
  printf("volMm %d volMk %d volMmk %d volMbar %d volMkBar %d\n",
    volMm, volMk, volMmk, volMbar, volMkBar);
  printf("numActiveBlock %d numSplit %d\n", numActiveBlock, numSplit);
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

//
// Number of elements in shared memory space
//
size_t TensorSplit::shmem() const {

  size_t vol = 0;

  switch(method) {

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

void getTiledSingleInRank(const int rank, const int* dim, const int* permutation,
  const size_t sizeofType, cudaDeviceProp& prop, std::list<TensorSplit>& tensorSplits) {

  LaunchConfig lc;
  for (int numMk=2;numMk < rank;numMk++) {
    TensorSplit ts;
    ts.method = TiledSingleInRank;
    ts.update(1, numMk, rank, dim, permutation);
    // If Mm and Mk overlap, break out of the loop
    if (ts.sizeMmk != numMk + 1) break;
    ts.numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, ts, prop, lc);
    if (ts.numActiveBlock > 0) {
      tensorSplits.push_back(ts);
    }
  }
}

void getTiledSingleOutRank(const int rank, const int* dim, const int* permutation,
  const size_t sizeofType, cudaDeviceProp& prop, std::list<TensorSplit>& tensorSplits) {

  LaunchConfig lc;
  for (int numMm=2;numMm < rank;numMm++) {
    TensorSplit ts;
    ts.method = TiledSingleOutRank;
    ts.update(numMm, 1, rank, dim, permutation);
    // If Mm and Mk overlap, break out of the loop
    if (ts.sizeMmk != numMm + 1) break;
    ts.numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, ts, prop, lc);
    if (ts.numActiveBlock > 0) {
      tensorSplits.push_back(ts);
    }
  }
}

void getTiledSingleRank(const int rank, const int* dim, const int* permutation,
  std::list<TensorSplit>& tensorSplits) {

  if (permutation[0] != 0) {
    TensorSplit ts;
    ts.method = TiledSingleRank;
    ts.update(1, 1, rank, dim, permutation);    
    tensorSplits.push_back(ts);
  }
}

void getTiledLeadVolSame(const int rank, const int* dim, const int* permutation,
  std::list<TensorSplit>& tensorSplits) {

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
    tensorSplits.push_back(ts);
  }
}

void getGeneral(const int rank, const int* dim, const int* permutation, const size_t sizeofType,
  cudaDeviceProp& prop, std::list<TensorSplit>& tensorSplits) {

  LaunchConfig lc;
  for (int numMm=1;numMm < rank;numMm++) {
    for (int numMk=1;numMk < rank;numMk++) {
      TensorSplit ts;
      ts.method = General;
      ts.update(numMm, numMk, rank, dim, permutation);
      ts.numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, ts, prop, lc);
      // If we can't fit to device, break out from inner loop
      if (ts.numActiveBlock == 0) break;
      tensorSplits.push_back(ts);
    }
  }

}

void getGeneralSplitInRank(const int rank, const int* dim, const int* permutation, const size_t sizeofType,
  cudaDeviceProp& prop, std::list<TensorSplit>& tensorSplits) {

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
    ts.numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, ts, prop, lc);
    if (ts.numActiveBlock > 0) {
      tensorSplits.push_back(ts);
    }
  }

}

void getGeneralSplitOutRank(const int rank, const int* dim, const int* permutation, const size_t sizeofType,
  cudaDeviceProp& prop, std::list<TensorSplit>& tensorSplits) {

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
    ts.numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, ts, prop, lc);
    if (ts.numActiveBlock > 0) {
      tensorSplits.push_back(ts);
    }
  }

}

//
// Returns a list of all possible ways to perform tensor transpose
//
void getTensorSplits(const int rank, const int* dim, const int* permutation, const size_t sizeofType,
  cudaDeviceProp& prop, std::list<TensorSplit>& tensorSplits) {

  getTiledLeadVolSame(rank, dim, permutation, tensorSplits);
  getTiledSingleInRank(rank, dim, permutation, sizeofType, prop, tensorSplits);
  getTiledSingleOutRank(rank, dim, permutation, sizeofType, prop, tensorSplits);
  getTiledSingleRank(rank, dim, permutation, tensorSplits);
  getGeneral(rank, dim, permutation, sizeofType, prop, tensorSplits);
  getGeneralSplitInRank(rank, dim, permutation, sizeofType, prop, tensorSplits);
  getGeneralSplitOutRank(rank, dim, permutation, sizeofType, prop, tensorSplits);
}

//
// This operator enables heuristic comparisons between TensorSplit elements
// returns (lhs > rhs)
//
bool operator>(const TensorSplit& lhs, const TensorSplit& rhs) {
  if (
    // * All self comparisons (7)
    lhs.method == rhs.method ||
    // * TiledSingleInRank vs. TiledSingleOutRank (2)
    (lhs.method == TiledSingleInRank && rhs.method == TiledSingleOutRank) ||
    (lhs.method == TiledSingleOutRank && rhs.method == TiledSingleInRank) ||
    // * GeneralSplitInRank vs. GeneralSplitOutRank (2)
    (lhs.method == GeneralSplitInRank && rhs.method == GeneralSplitOutRank) ||
    (lhs.method == GeneralSplitOutRank && rhs.method == GeneralSplitInRank) ||
    // * General vs. GeneralSplitInRank (2)
    (lhs.method == General && rhs.method == GeneralSplitInRank) ||
    (lhs.method == GeneralSplitInRank && rhs.method == General) ||
    // * General vs. GeneralSplitOutRank (2)
    (lhs.method == General && rhs.method == GeneralSplitOutRank) ||
    (lhs.method == GeneralSplitOutRank && rhs.method == General)
    ) {
    size_t lhs_nab = (lhs.method == General) ? std::min(2, lhs.numActiveBlock) : lhs.numActiveBlock;
    size_t rhs_nab = (rhs.method == General) ? std::min(2, rhs.numActiveBlock) : rhs.numActiveBlock;
    return (lhs.volMmkUsed()*lhs_nab > rhs.volMmkUsed()*rhs_nab);
  } else {
    const int MIN_TILED_DIM = TILEDIM/2;
    // * TiledLeadVolSame vs. TiledSingleRank || TiledSingleInRank || TiledSingleOutRank (6)
    //   TiledLeadVolSame always wins
    if (lhs.method == TiledLeadVolSame && 
      (rhs.method == TiledSingleRank || rhs.method == TiledSingleInRank || rhs.method == TiledSingleOutRank)) {
      return true;
    } else if (rhs.method == TiledLeadVolSame && 
      (lhs.method == TiledSingleRank || lhs.method == TiledSingleInRank || lhs.method == TiledSingleOutRank)) {
      return !(rhs > lhs);
    }
    // * TiledLeadVolSame vs. General || GeneralSplitInRank || GeneralSplitOutRank (6)
    if (lhs.method == TiledLeadVolSame &&
      (rhs.method == General || rhs.method == GeneralSplitInRank || rhs.method == GeneralSplitOutRank)) {
      return (
      (lhs.volMm >= MIN_TILED_DIM && lhs.volMk >= MIN_TILED_DIM) ||
      ((lhs.volMm >= MIN_TILED_DIM || lhs.volMk >= MIN_TILED_DIM) && lhs.volMmkUsed() >= rhs.volMmkUsed())
      );
    } else if (rhs.method == TiledLeadVolSame &&
      (lhs.method == General || lhs.method == GeneralSplitInRank || lhs.method == GeneralSplitOutRank)) {
      return !(rhs > lhs);
    }
    // * TiledSingleRank vs. TiledSingleInRank (2)
    //   TiledSingleInRank wins if its sizeMk > 1
    if (lhs.method == TiledSingleRank && rhs.method == TiledSingleInRank) {
      return (rhs.sizeMk == 1);
    } else if (rhs.method == TiledSingleRank && lhs.method == TiledSingleInRank) {
      return !(rhs > lhs);
    }
    // * TiledSingleRank vs. TiledSingleOutRank (2)
    //   TiledSingleOutRank wins if its sizeMm > 1
    if (lhs.method == TiledSingleRank && rhs.method == TiledSingleOutRank) {
      return (rhs.sizeMm == 1);
    } else if (rhs.method == TiledSingleRank && lhs.method == TiledSingleOutRank) {
      return !(rhs > lhs);
    }
    // * TiledSingleRank vs. General || GeneralSplitInRank || GeneralSplitOutRank (6)
    if (lhs.method == TiledSingleRank &&
      (rhs.method == General || rhs.method == GeneralSplitInRank || rhs.method == GeneralSplitOutRank)) {
      return (
      (lhs.volMm >= MIN_TILED_DIM && lhs.volMk >= MIN_TILED_DIM) ||
      ((lhs.volMm >= MIN_TILED_DIM || lhs.volMk >= MIN_TILED_DIM) && lhs.volMmkUsed() >= rhs.volMmkUsed())
      );
    } else if (rhs.method == TiledSingleRank &&
      (lhs.method == General || lhs.method == GeneralSplitInRank || lhs.method == GeneralSplitOutRank)) {
      return !(rhs > lhs);
    }
    // * TiledSingleInRank vs. General || GeneralSplitInRank || GeneralSplitOutRank (6)
    if (lhs.method == TiledSingleInRank &&
      (rhs.method == General || rhs.method == GeneralSplitInRank || rhs.method == GeneralSplitOutRank)) {
      return (
      (lhs.volMm >= MIN_TILED_DIM && lhs.volMk >= MIN_TILED_DIM) ||
      ((lhs.volMm >= MIN_TILED_DIM || lhs.volMk >= MIN_TILED_DIM) && lhs.volMmkUsed() >= rhs.volMmkUsed())
      );
    } else if (rhs.method == TiledSingleInRank &&
      (lhs.method == General || lhs.method == GeneralSplitInRank || lhs.method == GeneralSplitOutRank)) {
      return !(rhs > lhs);
    }
    // * TiledSingleOutRank vs. General || GeneralSplitInRank || GeneralSplitOutRank (6)
    if (lhs.method == TiledSingleOutRank &&
      (rhs.method == General || rhs.method == GeneralSplitInRank || rhs.method == GeneralSplitOutRank)) {
      return (
      (lhs.volMm >= MIN_TILED_DIM && lhs.volMk >= MIN_TILED_DIM) ||
      ((lhs.volMm >= MIN_TILED_DIM || lhs.volMk >= MIN_TILED_DIM) && lhs.volMmkUsed() >= rhs.volMmkUsed())
      );
    } else if (rhs.method == TiledSingleOutRank &&
      (lhs.method == General || lhs.method == GeneralSplitInRank || lhs.method == GeneralSplitOutRank)) {
      return !(rhs > lhs);
    }
  }
  // We should not end up here
  printf("bool operator>(TensorSplit& lhs, TensorSplit& rhs): FATAL implementation bug with:\n");
  printf("lhs.method ");
  printMethod(lhs.method);
  printf(" rhs.method ");
  printMethod(rhs.method);
  printf("\n");
  exit(1);
}

bool operator<(TensorSplit& lhs, TensorSplit& rhs) {
  return !(lhs > rhs);
}

//
// Choose among "method" the one with the largest total volume of shared memory
//
void reduceBasedOnVolume(std::list<TensorSplit>& tensorSplits, int method) {
  // Find the best
  bool foundBest = false;
  auto bestIt = tensorSplits.end();
  for (auto it = tensorSplits.begin();it != tensorSplits.end();it++) {
    if (it->method == method) {
      if (foundBest == false || *it > *bestIt) {
        foundBest = true;
        bestIt = it;
      }
    }
  }
  if (!foundBest) return;
  // Remove all but the best
  for (auto it=tensorSplits.begin();it != tensorSplits.end();) {
    if (it->method == method && it != bestIt) {
      it = tensorSplits.erase(it++);
    } else {
      it++;
    }
  }
}

//
// Reduce the number of options for tensor transpose by choosing the best among each
// category
//
void reduceTensorSplits(std::list<TensorSplit>& tensorSplits) {
  reduceBasedOnVolume(tensorSplits, TiledSingleInRank);
  reduceBasedOnVolume(tensorSplits, TiledSingleOutRank);
  reduceBasedOnVolume(tensorSplits, GeneralSplitInRank);
  reduceBasedOnVolume(tensorSplits, GeneralSplitOutRank);
  // reduceBasedOnVolume(tensorSplits, General);
}

//
// Build small versions of a set of TensorSplit: Only include ranks
// that are in the union of all TensorSplits
//
void reduceMbar(const int rank, const int* dim, const int* permutation,
  std::list<TensorSplit>& tensorSplits,
  int& smallRank, std::vector<int>& smallDim, std::vector<int>& smallPermutation,
  std::list<TensorSplit>& smallTensorSplits) {

  // Union of Mm and Mk across all TensorSplits
  // 0 = removed
  // 1 = Mm or Mk
  // 2 = Mbar
  std::vector<int> smallMmkMbar(rank, 0);
  int minVolMbar = (1 << 31);
  for (auto it = tensorSplits.begin();it != tensorSplits.end();it++) {
    for (int i=0;i < it->sizeMm;i++) {
      smallMmkMbar[i] = 1;
    }
    for (int i=0;i < it->sizeMk;i++) {
      int pi = permutation[i];
      smallMmkMbar[pi] = 1;
    }
    minVolMbar = std::min(minVolMbar, it->volMbar);
  }

  smallRank = 0;
  for (int i=0;i < rank;i++) {
    if (smallMmkMbar[i] == 1) {
      smallRank++;
    }
  }

  const int volMbarLimit = 64;

  int volMbar = 1;
  for (int i=0;i < rank;i++) {
    if (smallMmkMbar[i] == 0) {
      smallRank++;
      smallMmkMbar[i] = 2;
      volMbar *= dim[i];
      if (volMbar >= volMbarLimit) break;
    }
  }

  smallDim.resize(smallRank);
  smallPermutation.resize(smallRank);
  {
    int j = 0;
    for (int i=0;i < rank;i++) {
      if (smallMmkMbar[i] != 0) {
        // Add next Mmk or Mbar rank
        smallPermutation[j] = permutation[i];
        if (volMbar > volMbarLimit && smallMmkMbar[i] == 2) {
          // Reduce Mbar dimension
          smallDim[j] = std::max(2, dim[i]*volMbarLimit/volMbar);
          // Compute new volMbar
          volMbar = volMbar*smallDim[j]/dim[j];
        } else {
          smallDim[j] = dim[i];
        }
        j++;
      }
    }
  }

  for (auto it = tensorSplits.begin();it != tensorSplits.end();it++) {
    TensorSplit ts = *it;
    ts.update(ts.sizeMm, ts.sizeMk, smallRank, smallDim.data(), smallPermutation.data());
    smallTensorSplits.push_back(ts);
  }

}

//
// Returns tensorSplit that is chosen using heuristic criteria
// Returns -1 on invalid input or when nothing can be chosen
//
std::list<TensorSplit>::iterator chooseTensorSplitHeuristic(std::list<TensorSplit>& tensorSplits) {

  auto bestIt = tensorSplits.end();
  for (auto it=tensorSplits.begin();it != tensorSplits.end();it++) {
    if (bestIt == tensorSplits.end() || *bestIt < *it) {
      bestIt = it;
    }
  }

  return bestIt;
}

void LaunchConfig::print() {
  printf("numthread %d %d %d numblock %d %d %d shmemsize %d numRegStorage %d\n",
    numthread.x, numthread.y, numthread.z,
    numblock.x, numblock.y, numblock.z,
    shmemsize, numRegStorage);
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
  cuttKernelLaunchConfiguration(sizeofType, tensorSplit, prop, launchConfig);

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

  // Build MmI and MkI
  int* MmI = new int[tensorSplit.sizeMm];
  int* MkI = new int[tensorSplit.sizeMk];
  {
    int iMm = 0;
    int iMk = 0;
    for (int i=0;i < rank;i++) {
      if (isMm[i]) {
        MmI[iMm++] = i;
      }
      if (isMk[i]) {
        MkI[iMk++] = i;
      }
    }
  }

  TensorConvInOut* hostMbar = NULL;
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

    hostMbar = new TensorConvInOut[tensorSplit.sizeMbar];
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

  TensorConvInOut* hostMmk = NULL;
  TensorConv* hostMsh = NULL;
  TensorConv* hostMsh1 = NULL;
  TensorConv* hostMsh2 = NULL;
  if (tensorSplit.method == General) {
    // Build MmkI = {q_1, ..., q_a}
    int* MmkI = new int[tensorSplit.sizeMmk];
    int j = 0;
    for (int i=0;i < rank;i++) {
      if (isMm[i] || isMk[i]) {
        MmkI[j] = i;
        j++;
      }
    }
    TensorC cMmkI(rank, tensorSplit.sizeMmk, MmkI, dim);
    // Build MmkO = {q_t1, ..., q_ta}
    int* MmkO = new int[tensorSplit.sizeMmk];
    j = 0;
    for (int i=0;i < rank;i++) {
      int pi = permutation[i];
      if (isMm[pi] || isMk[pi]) {
        MmkO[j] = pi;
        j++;
      }
    }
    TensorC cMmkO(rank, tensorSplit.sizeMmk, MmkO, dim);

    hostMmk = new TensorConvInOut[tensorSplit.sizeMmk];
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

    hostMsh = new TensorConv[tensorSplit.sizeMmk];
    for (int i=0;i < tensorSplit.sizeMmk;i++) {
      // Shared memory reading position
      int qti = MmkO[i];
      hostMsh[i].c  = cMmkO.get(qti);
      hostMsh[i].d  = dim[qti];
      hostMsh[i].ct = cMmkI.get(qti);
    }

    delete [] MmkI;
    delete [] MmkO;
  }

  if (tensorSplit.sizeMbar > 0) {
    allocate_device<TensorConvInOut>(&Mbar, tensorSplit.sizeMbar);
    copy_HtoD_sync<TensorConvInOut>(hostMbar, Mbar, tensorSplit.sizeMbar);
    delete [] hostMbar;
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

    std::vector<TensorConv> hostMk(tensorSplit.sizeMk);
    for (int i=0;i < tensorSplit.sizeMk;i++) {
      int pti = MkO[i];
      // Global memory read position
      hostMk[i].c  = cMkO.get(pti);
      hostMk[i].d  = dim[pti];
      hostMk[i].ct = cI.get(pti);
    }

    allocate_device<TensorConv>(&Mk, tensorSplit.sizeMk);
    copy_HtoD_sync<TensorConv>(hostMk.data(), Mk, tensorSplit.sizeMk);
  }

  if (tensorSplit.method == TiledSingleOutRank) {
    cuDimMk = cI.get(permutation[0]);

    TensorC cMmI(rank, tensorSplit.sizeMm, MmI, dim);

    std::vector<TensorConv> hostMm(tensorSplit.sizeMm);
    for (int i=0;i < tensorSplit.sizeMm;i++) {
      int pi = MmI[i];
      // Global memory write position
      hostMm[i].c  = cMmI.get(pi);
      hostMm[i].d  = dim[pi];
      hostMm[i].ct = cO.get(pi);
    }

    allocate_device<TensorConv>(&Mm, tensorSplit.sizeMm);
    copy_HtoD_sync<TensorConv>(hostMm.data(), Mm, tensorSplit.sizeMm);
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

    std::vector<TensorConv> hostMk(tensorSplit.sizeMk);
    for (int i=0;i < tensorSplit.sizeMk;i++) {
      int pti = MkO[i];
      // Global memory read position
      hostMk[i].c  = cMkO.get(pti);
      hostMk[i].d  = dim[pti];
      hostMk[i].ct = cI.get(pti);
    }

    int* h_posMk = new int[tensorSplit.volMk];
    for (int j=0;j < tensorSplit.volMk;j++) {
      h_posMk[j] = 0;
      for (int i=0;i < tensorSplit.sizeMk;i++) {
        h_posMk[j] += ((j / hostMk[i].c) % hostMk[i].d) * hostMk[i].ct;
      }
    }

    allocate_device<int>(&posMk, tensorSplit.volMk);
    copy_HtoD_sync<int>(h_posMk, posMk, tensorSplit.volMk);
    delete [] h_posMk;
  }

  if (tensorSplit.method == GeneralSplitOutRank) {
    cuDimMk = cI.get(permutation[0]);

    TensorC cMmI(rank, tensorSplit.sizeMm, MmI, dim);

    std::vector<TensorConv> hostMm(tensorSplit.sizeMm);
    for (int i=0;i < tensorSplit.sizeMm;i++) {
      int pi = MmI[i];
      // Global memory write position
      hostMm[i].c  = cMmI.get(pi);
      hostMm[i].d  = dim[pi];
      hostMm[i].ct = cO.get(pi);
    }

    int* h_posMm = new int[tensorSplit.volMm];
    for (int j=0;j < tensorSplit.volMm;j++) {
      h_posMm[j] = 0;
      for (int i=0;i < tensorSplit.sizeMm;i++) {
        h_posMm[j] += ((j / hostMm[i].c) % hostMm[i].d) * hostMm[i].ct;
      }
    }

    allocate_device<int>(&posMm, tensorSplit.volMm);
    copy_HtoD_sync<int>(h_posMm, posMm, tensorSplit.volMm);
    delete [] h_posMm;
  }

  if (tensorSplit.method == General) {
    allocate_device<TensorConvInOut>(&Mmk, tensorSplit.sizeMmk);
    copy_HtoD_sync<TensorConvInOut>(hostMmk, Mmk, tensorSplit.sizeMmk);
    delete [] hostMmk;
    allocate_device<TensorConv>(&Msh, tensorSplit.sizeMmk);
    copy_HtoD_sync<TensorConv>(hostMsh, Msh, tensorSplit.sizeMmk);
    delete [] hostMsh;
  }

  cudaCheck(cudaDeviceSynchronize());

  delete [] MmI;
  delete [] MkI;

  return true;
}

cuttPlan_t::cuttPlan_t() {
  cudaCheck(cudaGetDevice(&deviceID));
  stream = 0;
  Mbar = NULL;
  Mmk = NULL;
  Msh = NULL;
  Mk = NULL;
  Mm = NULL;
  posMk = NULL;
  posMm = NULL;
}

cuttPlan_t::~cuttPlan_t() {
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
