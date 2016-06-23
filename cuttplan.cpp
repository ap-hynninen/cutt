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

cuttPlan_t::cuttPlan_t() {
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

bool cuttPlan_t::setup(const int rank_in, const int* dim, const int* permutation, const size_t sizeofType_in) {
  rank = rank_in;
  sizeofType = sizeofType_in;

  // Read device properties to determine how much shared memory we can afford to use
  cudaCheck(cudaGetDevice(&deviceID));
  cudaDeviceProp prop;
  cudaCheck(cudaGetDeviceProperties(&prop, deviceID));
/*
  maxThreadsPerBlock = prop.maxThreadsPerBlock;
  maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;

  // Absolute maximum number of elements in VolMmk = minimum of
  // number of elements shared memory can hold,
  // number of available registers for the general method,
  // 
  const int maxVolMmk = std::min( (int)(prop.sharedMemPerMultiprocessor/sizeofType),
    maxThreadsPerBlock*maxNumRegStorage);

  // Try to use maximum of 1/3 of shared memory
  const int useVolMmk = (prop.sharedMemPerMultiprocessor/sizeofType)/3;

  printf("maxVolMmk %d useVolMmk %d sharedMemPerMultiprocessor %d\n",
    maxVolMmk, useVolMmk, prop.sharedMemPerMultiprocessor);
*/

  std::vector<bool> isMm(rank, false);
  std::vector<bool> isMk(rank, false);

  // Minimum allowed dimension that is dealt with
  // using the tiled algorithm
  const int MIN_TILED_DIM = TILEDIM;

  // Setup Mm
  {
    int r = 0;
    sizeMm = 0;
    volMm = 1;
    while (r < rank && volMm < MIN_TILED_DIM) {
      isMm[r] = true;
      volMm *= dim[r];
      sizeMm++;
      r++;
    }
  }

  // Setup Mk
  {
    int r = 0;
    sizeMk = 0;
    volMk = 1;
    while (r < rank && volMk < MIN_TILED_DIM) {
      int pr = permutation[r];
      isMk[pr] = true;
      volMk *= dim[pr];
      sizeMk++;
      r++;
    }
  }

  // Setup Mmk
  setupMmk(isMm, isMk, dim);

  // Setup Mbar
  setupMbar(isMm, isMk, dim);

  // Setup method
  method = Unknown;
  while (method == Unknown) {
    if (sizeMm > 1 || sizeMk > 1) {
      // General case: Mm or Mk are > 1
      bool Mm_Mk_same = (sizeMm == sizeMk);
      if (Mm_Mk_same) {
        for (int i=0;i < sizeMm;i++) {
          if (permutation[i] != i) {
            Mm_Mk_same = false;
            break;
          }
        }
      }

      // APH DEBUG: REMOVE THIS AFTER TiledLeadVolSame WORKS
      Mm_Mk_same = false;

      if (Mm_Mk_same) {
        method = TiledLeadVolSame;
      } else {
        method = General;
        // We want to try to have at least two active blocks per SM
        int numActiveBlock = 0;
        while ((numActiveBlock = cuttKernelLaunchConfiguration(*this, prop)) < 2) {
          int r = sizeMm - 1;
          int pr = permutation[sizeMk - 1];
          if (sizeMk > 1 && (volMk > volMm)) {
            // Mk has larger volume => Remove from Mk
            isMk[pr] = false;
          } else if (sizeMm > 1) {
            // Remove from Mm
            isMm[r] = false;
          } else {
            // Unable to remove from either Mk or Mm => Break
            break;
          }
          setupMm(isMm, dim);
          setupMk(isMk, dim);
          setupMmk(isMm, isMk, dim);
          setupMbar(isMm, isMk, dim);
        }
        if (numActiveBlock == 0) {
          // Unable to use the General method =>
          // Switch to tiled method, which will always work but might be slow
          for (int i=0;i < rank;i++) {
            isMm[i] = false;
            isMk[i] = false;
          }
          isMm[0] = true;
          isMk[permutation[0]] = true;
          setupMm(isMm, dim);
          setupMk(isMk, dim);
          setupMmk(isMm, isMk, dim);
          setupMbar(isMm, isMk, dim);
          // This will cause another go at the outer while loop, 
          // and then selection of tiled method
          method = Unknown;
        }
      }
    } else {
      // Tiled case: Mm and Mk are size 1

      // Check if Mm and Mk are the same
      if (permutation[0] == 0) {
        method = TiledLeadVolSame;
        // isMm[1] = true;
        // isMk[1] = true;
        // setupMm(isMm, dim);
        // setupMk(isMk, dim);

        // Choose next rank as Mk
        // isMk[0] = false;
        // isMk[1] = true;
        // volMk = dim[1];
      } else {
        method = TiledSingleRank;
      }
    }
  } // while (method == Unkown)

  setupMm(isMm, dim);
  setupMk(isMk, dim);
  setupMmk(isMm, isMk, dim);
  setupMbar(isMm, isMk, dim);

  // Setup final kernel launch configuration and check that kernel execution is possible
  if (cuttKernelLaunchConfiguration(*this, prop) == 0) {
    return false;
  }

#if 0
  printf("method ");
  switch(method) {
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

  // Build cI
  int* I = new int[rank];
  for (int i=0;i < rank;i++) {
    I[i] = i;
  }
  TensorC cI(rank, rank, I, dim);
  delete [] I;

  // Build cO
  TensorC cO(rank, rank, permutation, dim);

  int vol0 = 0;
  int vol1 = 0;

  if (method == TiledSingleRank) {
    // cuDimMk = cI.get(MkI[0]);
    // cuDimMm = cO.get(MmI[0]);
    cuDimMk = cI.get(permutation[0]);
    cuDimMm = cO.get(0);
  } else if (method == TiledLeadVolSame) {
    vol0 = volMm;
    // Mm and Mk are the same => try including one more rank into Mmk from input
    if (sizeMmk < rank) {
      isMm[sizeMmk] = true;
      isMk[sizeMmk] = true;
      cuDimMk = cI.get(sizeMmk);
      cuDimMm = cO.get(sizeMmk);
      vol1 = dim[sizeMmk];
      setupMm(isMm, dim);
      setupMk(isMk, dim);
      setupMmk(isMm, isMk, dim);
      setupMbar(isMm, isMk, dim);
    } else {
      cuDimMk = 1;
      cuDimMm = 1;
      vol1 = 1;
    }
  }

  // Build MmI and MkI
  int* MmI = new int[sizeMm];
  int* MkI = new int[sizeMk];
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

  // if (method == TiledSingleRank) {
  // } else if (method == TiledLeadVolSame) {
  //   cuDimMk = cI.get(MmkI[MmkSplit]);
  //   cuDimMm = cO.get(MkI[0]);
  // }

  // if (method != General) {
  //   cuDimMk = cI.get(MkI[0]);
  //   if (method == TiledLeadVolSame) {
  //     cuDimMm = cO.get(MkI[0]);
  //   } else {
  //     cuDimMm = cO.get(MmI[0]);
  //   }
  // }

  TensorConvInOut* hostMbar = NULL;
  if (sizeMbar > 0) {
    // Build MbarI = {s_1, ...., s_h}, indices in input order
    int* MbarI = new int[sizeMbar];
    int j = 0;
    for (int i=0;i < rank;i++) {
      if (!(isMm[i] || isMk[i])) {
        MbarI[j] = i;
        j++;
      }
    }
    TensorC cMbarI(rank, sizeMbar, MbarI, dim);

    // Build MbarO = {s_l1, ...., s_lh}, indices in output (permuted) order
    int* MbarO = new int[sizeMbar];
    j = 0;
    for (int i=0;i < rank;i++) {
      int pi = permutation[i];
      if (!(isMm[pi] || isMk[pi])) {
        MbarO[j] = pi;
        j++;
      }
    }

    hostMbar = new TensorConvInOut[sizeMbar];
    for (int i=0;i < sizeMbar;i++) {
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
  if (method == General) {
    // Build MmkI = {q_1, ..., q_a}
    int* MmkI = new int[sizeMmk];
    int j = 0;
    for (int i=0;i < rank;i++) {
      if (isMm[i] || isMk[i]) {
        MmkI[j] = i;
        j++;
      }
    }
    TensorC cMmkI(rank, sizeMmk, MmkI, dim);
    // Build MmkO = {q_t1, ..., q_ta}
    int* MmkO = new int[sizeMmk];
    j = 0;
    for (int i=0;i < rank;i++) {
      int pi = permutation[i];
      if (isMm[pi] || isMk[pi]) {
        MmkO[j] = pi;
        j++;
      }
    }
    TensorC cMmkO(rank, sizeMmk, MmkO, dim);

    hostMmk = new TensorConvInOut[sizeMmk];
    for (int i=0;i < sizeMmk;i++) {
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

    hostMsh = new TensorConv[sizeMmk];
    for (int i=0;i < sizeMmk;i++) {
      // Shared memory reading position
      int qti = MmkO[i];
      hostMsh[i].c  = cMmkO.get(qti);
      hostMsh[i].d  = dim[qti];
      hostMsh[i].ct = cMmkI.get(qti);
    }

    delete [] MmkI;
    delete [] MmkO;
  }

  // Setup readVol
  if (method == General) {
    readVol.x = 0;
    readVol.y = 0;
  } else {
    int* tmp_dimMmkIn = new int[sizeMmk];
    tmp_dimMmkIn[0] = dim[MmI[0]];
    tmp_dimMmkIn[1] = dim[MkI[0]];

    if (method == TiledSingleRank) {
      readVol.x = dim[0];
      readVol.y = dim[permutation[0]];
    } else {
      readVol.x = vol0;
      readVol.y = vol1;
    }

    int* tmp_dimMmkOut = new int[sizeMmk];
    int j = 0;
    for (int i=0;i < rank;i++) {
      int pi = permutation[i];
      if (isMm[pi] || isMk[pi]) {
        tmp_dimMmkOut[j] = dim[pi];
        j++;
      }
    }

#if 0
    int* h_transposeArg = new int[transposeArgSize];
    int iarg = 0;
    for (int j=0;j < sizeMmk;j++) h_transposeArg[iarg++] = tmp_dimMmkIn[j];
    for (int j=0;j < sizeMmk;j++) h_transposeArg[iarg++] = tmp_dimMmkOut[j];

    cudaCheck(cudaMemcpyToSymbol(transposeArg, h_transposeArg,
      transposeArgSize*sizeof(int), 0, cudaMemcpyHostToDevice));
    delete [] h_transposeArg;
#endif

    delete [] tmp_dimMmkIn;
    delete [] tmp_dimMmkOut;
  }

#if 0
  printf("MmI");
  for (int i = 0; i < sizeMm; ++i) printf(" %d", MmI[i]+1);
  printf(" volMm %d\n", volMm);

  printf("MkI");
  for (int i = 0; i < sizeMk; ++i) printf(" %d", MkI[i]+1);
  printf(" volMk %d\n", volMk);

  printf("Mmk");
  for (int i = 0; i < rank; ++i) if (isMm[i] || isMk[i]) printf(" %d", i+1);
  printf(" volMmk %d\n", volMmk);

  if (sizeMbar > 0) {
    printf("Mbar");
    for (int i = 0; i < rank; ++i) if (!(isMm[i] || isMk[i])) printf(" %d", i+1);
    printf(" volMbar %d\n", volMbar);
  }

  if (sizeMbar > 0) {
    printf("MbarIn\n");
    for (int i=0;i < sizeMbar;i++) printf("%d %d %d\n",
      hostMbar[i].c_in, hostMbar[i].d_in, hostMbar[i].ct_in);

    printf("MbarOut\n");
    for (int i=0;i < sizeMbar;i++) printf("%d %d %d\n",
      hostMbar[i].c_out, hostMbar[i].d_out, hostMbar[i].ct_out);
  }

  if (method == General) {
    printf("MmkIn\n");
    for (int i=0;i < sizeMmk;i++) printf("%d %d %d\n",
      hostMmk[i].c_in, hostMmk[i].d_in, hostMmk[i].ct_in);

    printf("MmkOut\n");
    for (int i=0;i < sizeMmk;i++) printf("%d %d %d\n",
      hostMmk[i].c_out, hostMmk[i].d_out, hostMmk[i].ct_out);

    printf("Msh\n");
    for (int i=0;i < sizeMmk;i++) printf("%d %d %d\n",
      hostMsh[i].c, hostMsh[i].d, hostMsh[i].ct);
  }

  if (method != General) printf("cuDimMk %d cuDimMm %d\n", cuDimMk, cuDimMm);

  printf("readVol %d %d\n", readVol.x, readVol.y);
#endif

  delete [] MmI;
  delete [] MkI;

  if (sizeMbar > 0) {
    allocate_device<TensorConvInOut>(&Mbar, sizeMbar);
    copy_HtoD_sync<TensorConvInOut>(hostMbar, Mbar, sizeMbar);
    delete [] hostMbar;
  }

  if (method == General) {
    allocate_device<TensorConvInOut>(&Mmk, sizeMmk);
    copy_HtoD_sync<TensorConvInOut>(hostMmk, Mmk, sizeMmk);
    delete [] hostMmk;
    allocate_device<TensorConv>(&Msh, sizeMmk);
    copy_HtoD_sync<TensorConv>(hostMsh, Msh, sizeMmk);
    delete [] hostMsh;
  }

  cudaCheck(cudaDeviceSynchronize());    
}

void cuttPlan_t::setupMm(std::vector<bool>& isMm, const int* dim) {
  sizeMm = 0;
  volMm = 1;
  for (int i=0;i < rank;i++) {
    if (isMm[i]) {
      volMm *= dim[i];
      sizeMm++;
    }
  }
}

void cuttPlan_t::setupMk(std::vector<bool>& isMk, const int* dim) {
  sizeMk = 0;
  volMk = 1;
  for (int i=0;i < rank;i++) {
    if (isMk[i]) {
      volMk *= dim[i];
      sizeMk++;
    }
  }
}

void cuttPlan_t::setupMmk(std::vector<bool>& isMm, std::vector<bool>& isMk, const int* dim) {
  sizeMmk = 0;
  volMmk = 1;
  for (int i=0;i < rank;i++) {
    if (isMm[i] || isMk[i]) {
      volMmk *= dim[i];
      sizeMmk++;
    }
  }
}

void cuttPlan_t::setupMbar(std::vector<bool>& isMm, std::vector<bool>& isMk, const int* dim) {
  sizeMbar = 0;
  volMbar = 1;
  for (int i=0;i < rank;i++) {
    if (!(isMm[i] || isMk[i])) {
      volMbar *= dim[i];
      sizeMbar++;
    }
  }
}

