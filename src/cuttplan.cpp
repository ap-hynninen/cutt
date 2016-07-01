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
#include "CudaUtils.h"
#include "cuttplan.h"
#include "cuttkernel.h"

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
  method = cuttPlan_t::Unknown;
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
}

void TensorSplit::print() {
  printf("sizeMm %d sizeMk %d sizeMmk %d sizeMbar %d sizeMkBar %d\n",
    sizeMm, sizeMk, sizeMmk, sizeMbar, sizeMkBar);
  printf("volMm %d volMk %d volMmk %d volMbar %d volMkBar %d\n",
    volMm, volMk, volMmk, volMbar, volMkBar);
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

cuttPlan_t::cuttPlan_t() {
  cudaCheck(cudaGetDevice(&deviceID));
  stream = 0;
  Mbar = NULL;
  Mmk = NULL;
  Msh = NULL;
}

cuttPlan_t::~cuttPlan_t() {
  if (Mbar != NULL) deallocate_device<TensorConvInOut>(&Mbar);
  if (Mmk != NULL) deallocate_device<TensorConvInOut>(&Mmk);
  if (Msh != NULL) deallocate_device<TensorConv>(&Msh);
}

void cuttPlan_t::setStream(cudaStream_t stream_in) {
  stream = stream_in;
}

void getTiledSingleRank(const int rank, const int* dim, const int* permutation,
  std::vector<TensorSplit>& tensorSplits) {

  if (permutation[0] != 0) {
    TensorSplit ts;
    ts.method = cuttPlan_t::TiledSingleRank;
    ts.update(1, 1, rank, dim, permutation);    
    tensorSplits.push_back(ts);
  }
}

void getTiledLeadVolSame(const int rank, const int* dim, const int* permutation,
  std::vector<TensorSplit>& tensorSplits) {

  // Count number of Mm and Mk which are the same
  int numMmMkSame = 0;
  while (numMmMkSame < rank && permutation[numMmMkSame] == numMmMkSame) {
    numMmMkSame++;
  }
  if (numMmMkSame >= 1) {
    TensorSplit ts;
    ts.method = cuttPlan_t::TiledLeadVolSame;
    if (numMmMkSame < rank) {
      ts.update(numMmMkSame, numMmMkSame + 1, rank, dim, permutation);      
    } else {
      ts.update(numMmMkSame - 1, numMmMkSame, rank, dim, permutation);      
    }
    tensorSplits.push_back(ts);
  }
}

void getGeneral(const int rank, const int* dim, const int* permutation, const size_t sizeofType,
  cudaDeviceProp& prop, std::vector<TensorSplit>& tensorSplits) {

  LaunchConfig lc;
  for (int numMm=1;numMm < rank;numMm++) {
    for (int numMk=1;numMk < rank;numMk++) {
      TensorSplit ts;
      ts.method = cuttPlan_t::General;
      ts.update(numMm, numMk, rank, dim, permutation);
      ts.numActiveBlock = cuttKernelLaunchConfiguration(sizeofType, ts, prop, lc);
      // printf("numMm %d numMk %d volMmk %d numActiveBlock %d | %d\n",
      //   numMm, numMk, ts.volMmk, numActiveBlock, ts.volMmk*numActiveBlock);
      // If we can't fit to device, break out from inner loop
      if (ts.numActiveBlock == 0) break;
      tensorSplits.push_back(ts);
    }
  }

}

//
// Returns a list of all possible ways to perform tensor transpose
//
void getTensorSplits(const int rank, const int* dim, const int* permutation, const size_t sizeofType,
  cudaDeviceProp& prop, std::vector<TensorSplit>& tensorSplits) {

  getTiledLeadVolSame(rank, dim, permutation, tensorSplits);
  getTiledSingleRank(rank, dim, permutation, tensorSplits);
  getGeneral(rank, dim, permutation, sizeofType, prop, tensorSplits);

}

//
// Returns index to tensorSplits[] that is chosen using heuristic criteria
// Returns -1 on invalid input or when nothing can be chosen
//
int chooseTensorSplitHeuristic(std::vector<TensorSplit>& tensorSplits) {

  int index_tiledLeadVolSameTS = -1;
  int index_tiledSingleRankTS = -1;
  int index_generalTS = -1;
  TensorSplit tiledLeadVolSameTS;
  TensorSplit tiledSingleRankTS;
  TensorSplit generalTS;

  int bestTotVolMmk = 0;

  for (int i=0;i < tensorSplits.size();i++) {
    if (tensorSplits[i].method == cuttPlan_t::TiledLeadVolSame) {
      if (index_tiledLeadVolSameTS != -1) return -1;
      index_tiledLeadVolSameTS = i;
    } else if (tensorSplits[i].method == cuttPlan_t::TiledSingleRank) {
      if (index_tiledSingleRankTS != -1) return -1;
      index_tiledSingleRankTS = i;
    } else if (tensorSplits[i].method == cuttPlan_t::General) {
      int totVolMmk = tensorSplits[i].volMmk * tensorSplits[i].numActiveBlock;
      if (totVolMmk > bestTotVolMmk) {
        bestTotVolMmk = totVolMmk;
        index_generalTS = i;
      }
    } else {
      // Invalid input
      return -1;
    }
  }

  if (index_tiledLeadVolSameTS != -1) tiledLeadVolSameTS = tensorSplits[index_tiledLeadVolSameTS];
  if (index_tiledSingleRankTS != -1) tiledSingleRankTS = tensorSplits[index_tiledSingleRankTS];
  if (index_generalTS != -1) generalTS = tensorSplits[index_generalTS];

  const int MIN_TILED_DIM = TILEDIM/2;
  int index = -1;

  if (tiledLeadVolSameTS.sizeMmk > 0 && 
    ((tiledLeadVolSameTS.volMmk >= generalTS.volMmk && tiledLeadVolSameTS.volMm >= MIN_TILED_DIM &&
      tiledLeadVolSameTS.volMkBar >= MIN_TILED_DIM) || generalTS.sizeMmk == 0)) {
    // Choose TiledLeadVolSame
    index = index_tiledLeadVolSameTS;
  } else if (tiledSingleRankTS.sizeMmk > 0 &&
    ((tiledSingleRankTS.volMmk >= generalTS.volMmk && tiledSingleRankTS.volMm >= MIN_TILED_DIM &&
      tiledSingleRankTS.volMk >= MIN_TILED_DIM) || generalTS.sizeMmk == 0)) {
    // Choose TiledSingleRank
    index = index_tiledSingleRankTS;
  } else if (generalTS.sizeMmk > 0) {
    // Choose General
    index = index_generalTS;
  }

  return index;
}

/*
//
// Returns index to tensorSplits[] that is chosen by measuring the performance of every option
// Returns -1 on invalid input or when nothing can be chosen
//
int chooseTensorSplitMeasure(std::vector<TensorSplit>& tensorSplits, void* idata, void* odata) {
  int index = -1;
  for (int i=0;i < tensorSplits.size();i++) {
  }
  return index;
}

//
// Execute plan
//
bool cuttPlan_t::execute(void* idata, void* odata) {
  int cur_deviceID;
  cudaCheck(cudaGetDevice(&cur_deviceID));
  if (cur_deviceID != deviceID) return CUTT_INVALID_DEVICE;

  // Set shared memory configuration if necessary
  if (!devicesReady.count(deviceID)) {
    cuttKernelSetSharedMemConfig();
    devicesReady.insert(deviceID);
  }

  if (!cuttKernel(plan, idata, odata)) return CUTT_INTERNAL_ERROR;
}
*/

//
// Setup plan
//
bool cuttPlan_t::setup(const int rank_in, const int* dim, const int* permutation,
  const size_t sizeofType_in, cudaDeviceProp& prop, TensorSplit& tensorSplit_in) {
  
  rank = rank_in;
  sizeofType = sizeofType_in;
  tensorSplit = tensorSplit_in;

/*
  // Read device properties
  cudaCheck(cudaGetDevice(&deviceID));
  cudaDeviceProp prop;
  cudaCheck(cudaGetDeviceProperties(&prop, deviceID));

  // Choose method
  TensorSplit tiledLeadVolSameTS;
  setupTiledLeadVolSame(dim, permutation, tiledLeadVolSameTS);

  TensorSplit tiledSingleRankTS;
  setupTiledSingleRank(dim, permutation, tiledSingleRankTS);

  TensorSplit generalTS;
  setupGeneral(dim, permutation, prop, generalTS);

  // printf("tiledLeadVolSameTS.volMmk %d\n", tiledLeadVolSameTS.volMmk);
  // printf("tiledSingleRankTS.volMmk %d\n", tiledSingleRankTS.volMmk);
  // printf("generalTS.volMmk %d\n", generalTS.volMmk);

  const int MIN_TILED_DIM = TILEDIM/2;

  if (tiledLeadVolSameTS.sizeMmk > 0 && 
    ((tiledLeadVolSameTS.volMmk >= generalTS.volMmk && tiledLeadVolSameTS.volMm >= MIN_TILED_DIM &&
      tiledLeadVolSameTS.volMkBar >= MIN_TILED_DIM) || generalTS.sizeMmk == 0)) {
    // Choose TiledLeadVolSame
    tensorSplit = tiledLeadVolSameTS;
  } else if (tiledSingleRankTS.sizeMmk > 0 &&
    ((tiledSingleRankTS.volMmk >= generalTS.volMmk && tiledSingleRankTS.volMm >= MIN_TILED_DIM &&
      tiledSingleRankTS.volMk >= MIN_TILED_DIM) || generalTS.sizeMmk == 0)) {
    // Choose TiledSingleRank
    tensorSplit = tiledSingleRankTS;
  } else if (generalTS.sizeMmk > 0) {
    // Choose General
    tensorSplit = generalTS;
  } else {
    // Unable to choose a method
    tensorSplit.method = Unknown;
    return false;
  }
*/
#if 0
  printf("method ");
  switch(tensorSplit.method) {
    case General:
    printf("General\n");
    break;
    case TiledSingleRank:
    printf("TiledSingleRank\n");
    break;
    case TiledLeadVolSame:
    printf("TiledLeadVolSame\n");
    break;
  };
#endif

  std::vector<bool> isMm(rank, false);
  std::vector<bool> isMk(rank, false);
  for (int i=0;i < tensorSplit.sizeMm;i++) {
    isMm[i] = true;
  }
  for (int i=0;i < tensorSplit.sizeMk;i++) {
    isMk[permutation[i]] = true;
  }

#if 0
  tensorSplit.print();
#endif

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

#if 0
    printf("MbarI");
    for (int i=0;i < sizeMbar;i++) printf(" %d", MbarI[i]+1);
    printf("\n");

    printf("MbarO");
    for (int i=0;i < sizeMbar;i++) printf(" %d", MbarO[i]+1);
    printf("\n");
#endif

    delete [] MbarI;
    delete [] MbarO;
  }

  TensorConvInOut* hostMmk = NULL;
  TensorConv* hostMsh = NULL;
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

#if 0
  printf("MmI");
  for (int i = 0; i < tensorSplit.sizeMm; ++i) printf(" %d", MmI[i]+1);
  printf(" volMm %d\n", tensorSplit.volMm);

  printf("MkI");
  for (int i = 0; i < tensorSplit.sizeMk; ++i) printf(" %d", MkI[i]+1);
  printf(" volMk %d\n", tensorSplit.volMk);

  printf("Mmk");
  for (int i = 0; i < rank; ++i) if (isMm[i] || isMk[i]) printf(" %d", i+1);
  printf(" volMmk %d\n", tensorSplit.volMmk);

  if (tensorSplit.sizeMbar > 0) {
    printf("Mbar");
    for (int i = 0; i < rank; ++i) if (!(isMm[i] || isMk[i])) printf(" %d", i+1);
    printf(" volMbar %d\n", tensorSplit.volMbar);
  }

  if (tensorSplit.sizeMbar > 0) {
    printf("MbarIn %d\n",tensorSplit.sizeMbar);
    for (int i=0;i < tensorSplit.sizeMbar;i++) printf("%d %d %d\n",
      hostMbar[i].c_in, hostMbar[i].d_in, hostMbar[i].ct_in);

    printf("MbarOut\n");
    for (int i=0;i < tensorSplit.sizeMbar;i++) printf("%d %d %d\n",
      hostMbar[i].c_out, hostMbar[i].d_out, hostMbar[i].ct_out);
  }

  if (tensorSplit.method == General) {
    printf("MmkIn\n");
    for (int i=0;i < tensorSplit.sizeMmk;i++) printf("%d %d %d\n",
      hostMmk[i].c_in, hostMmk[i].d_in, hostMmk[i].ct_in);

    printf("MmkOut\n");
    for (int i=0;i < tensorSplit.sizeMmk;i++) printf("%d %d %d\n",
      hostMmk[i].c_out, hostMmk[i].d_out, hostMmk[i].ct_out);

    printf("Msh\n");
    for (int i=0;i < tensorSplit.sizeMmk;i++) printf("%d %d %d\n",
      hostMsh[i].c, hostMsh[i].d, hostMsh[i].ct);
  }

  if (tensorSplit.method != General) {
    printf("cuDimMk %d cuDimMm %d\n", cuDimMk, cuDimMm);
    printf("tiledVol %d %d\n", tiledVol.x, tiledVol.y);
  }
#endif

  delete [] MmI;
  delete [] MkI;

  if (tensorSplit.sizeMbar > 0) {
    allocate_device<TensorConvInOut>(&Mbar, tensorSplit.sizeMbar);
    copy_HtoD_sync<TensorConvInOut>(hostMbar, Mbar, tensorSplit.sizeMbar);
    delete [] hostMbar;
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

  return true;
}

/*
void cuttPlan_t::setupTiledSingleRank(const int* dim, const int* permutation, TensorSplit& ts) {
  ts.method = TiledSingleRank;
  if (permutation[0] == 0) {
    // Lead ranks match => Must use the LeadVolSame version
    ts.update(0, 0, rank, dim, permutation);
  } else {
    ts.update(1, 1, rank, dim, permutation);    
  }
}

void cuttPlan_t::setupTiledLeadVolSame(const int* dim, const int* permutation, TensorSplit& ts) {
  ts.method = TiledLeadVolSame;
  // Count number of Mm and Mk which are the same
  int numMmMkSame = 0;
  while (numMmMkSame < rank && permutation[numMmMkSame] == numMmMkSame) {
    numMmMkSame++;
  }
  if (numMmMkSame >= 1) {
    if (numMmMkSame < rank) {
      ts.update(numMmMkSame, numMmMkSame + 1, rank, dim, permutation);      
    } else {
      ts.update(numMmMkSame - 1, numMmMkSame, rank, dim, permutation);      
    }
  } else {
    ts.update(0, 0, rank, dim, permutation);    
  }
}

void cuttPlan_t::setupGeneral(const int* dim, const int* permutation, cudaDeviceProp& prop, TensorSplit& ts) {
  ts.method = General;
  // Maximize volMmk*numActiveBlock by trying all possibilities
  LaunchConfig lc;
  int bestVolMmk = 0;
  int bestNumMm = 0;
  int bestNumMk = 0;
  for (int numMm=1;numMm < rank;numMm++) {
    for (int numMk=1;numMk < rank;numMk++) {
      ts.update(numMm, numMk, rank, dim, permutation);
      int numActiveBlock = cuttKernelLaunchConfiguration(General, sizeofType, ts, prop, lc);
      // printf("numMm %d numMk %d volMmk %d numActiveBlock %d | %d\n",
      //   numMm, numMk, ts.volMmk, numActiveBlock, ts.volMmk*numActiveBlock);
      // If we can't fit to device, break out from inner loop
      if (numActiveBlock == 0) break;
      if (ts.volMmk*numActiveBlock > bestVolMmk) {
        bestVolMmk = ts.volMmk*numActiveBlock;
        bestNumMm = numMm;
        bestNumMk = numMk;
      }
    }
  }

  ts.update(bestNumMm, bestNumMk, rank, dim, permutation);
}
*/
