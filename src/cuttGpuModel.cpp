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
#include <random>
#include <cuda_runtime.h>
#include <cstring>               // memcpy
#include "cuttGpuModel.h"
#include "cuttGpuModelKernel.h"

//
// Count number of global memory transactions per one request for potentially
// scattered accesses to elements listed in pos
// NOTE: Assumes pos is sorted
//
int glTransactions(const int* pos, const int n, const int accWidth) {
  int count = 0;
  int iseg_prev = -1;
  for (int i=0;i < n;i++) {
    int iseg = pos[i]/accWidth;
    count += (iseg != iseg_prev);
    iseg_prev = iseg;
  }
  return count;
}

//
// Count number of global memory transactions per request for contigious memory access
// of n elements starting at pos
//
int glTransactions(const int pos, const int n, const int accWidth) {
  if (n == 0) return 0;
  // Segment at the first memory location
  int seg0 = pos/accWidth;
  // Segment at the last memory location
  int seg1 = (pos + n - 1)/accWidth;
  return (seg1 - seg0 + 1);
}


//
// Count number of full and partial cache-lines accessed for potentially
// scattered accesses to elements listed in pos
//
// cl_full = Number of full cache lines
// cl_part = Number of partial cache lines
//
void countCacheLines(const int* pos, const int n, const int cacheWidth, int& cl_full, int& cl_part) {

  cl_full = 0;
  cl_part = 0;
  
  if (n == 0) return;

  int i = 0;
  while (i < n) {
    int i0 = i;
    int m = std::min(i + cacheWidth, n);
    int seg0 = pos[i++]/cacheWidth;
    while (i < m) {
      int seg = pos[i]/cacheWidth;
      if (seg0 != seg) {
        break;
      }
      i++;
    }
    if (i == i0 + cacheWidth) {
      cl_full++;
    } else {
      cl_part++;
    }
  }
}

//
// Count number of full and partial cache-lines accessed for contigious memory access
// of n elements starting at pos
//
// cl_full = Number of full cache lines
// cl_part = Number of partial cache lines
//
void countCacheLines(const int pos, const int n, const int cacheWidth, int& cl_full, int& cl_part) {
  if (n == 0) {
    cl_full = 0;
    cl_part = 0;
    return;
  }
  if (n < cacheWidth) {
    cl_full = 0;
    cl_part = 1 + ((pos % cacheWidth) + n > cacheWidth);
  } else {
    int start_part = (pos % cacheWidth);
    int end_part = ((pos + n) % cacheWidth);
    //partial:   start full?          end full?
    cl_part = (start_part != 0) + (end_part != 0);
    //full:         number of start partials   number of end partials
    cl_full = (n - (cacheWidth - start_part)*(start_part != 0) - end_part)/cacheWidth;
  }
}

//
// Compute memory element positions
//
void computePos(int vol0, int vol1,
  std::vector<TensorConvInOut>::iterator it0, std::vector<TensorConvInOut>::iterator it1,
  std::vector<int>& posIn, std::vector<int>& posOut) {
  int i=0;
  for (int j=vol0;j <= vol1;j++,i++) {
    int posInVal = 0;
    int posOutVal = 0;
    for (auto it=it0;it != it1;it++) {
      posInVal  += ((j / it->c_in) % it->d_in) * it->ct_in;
      posOutVal += ((j / it->c_out) % it->d_out) * it->ct_out;
    }
    posIn[i] = posInVal;
    posOut[i] = posOutVal;
  }
}

//
// Count number of global memory transactions for Packed -method
//
void countPackedGlTransactions(const int warpSize, const int accWidth, const int cacheWidth,
  const int numthread, const int posMbarIn, const int posMbarOut, const int volMmk, 
  std::vector<int>& posMmkIn, std::vector<int>& posMmkOut,
  int& gld_tran, int& gst_tran, int& gld_req, int& gst_req,
  int& cl_full_l2, int& cl_part_l2, int& cl_full_l1, int& cl_part_l1) {

  std::vector<int> readPos(warpSize);
  std::vector<int> writePos(warpSize);
  std::vector<int> writePosVolMmk(volMmk);

  int m = 0;
  for (int j00=0;j00 < volMmk;j00+=numthread) {
    int n0 = std::min(volMmk, j00 + numthread);
    for (int j0=j00;j0 < n0;j0+=warpSize) {
      int n = std::min(warpSize, volMmk - j0);
      for (int j1=0;j1 < n;j1++) {
        int j = j0 + j1;
        int posIn  = posMbarIn + posMmkIn[j];
        int posOut = posMbarOut + posMmkOut[j];
        readPos[j1] = posIn;
        writePos[j1] = posOut;
        writePosVolMmk[m] = posOut;
        m++;
      }
      // Global memory transactions
      gld_tran += glTransactions(readPos.data(), n, accWidth);
      gst_tran += glTransactions(writePos.data(), n, accWidth);
      gld_req++;
      gst_req++;
    }
  }
  // Global write non-full cache-lines
  int cl_full_tmp, cl_part_tmp;
  countCacheLines(writePosVolMmk.data(), writePosVolMmk.size(), cacheWidth, cl_full_tmp, cl_part_tmp);
  cl_full_l2 += cl_full_tmp;
  cl_part_l2 += cl_part_tmp;

  countCacheLines(writePosVolMmk.data(), writePosVolMmk.size(), accWidth, cl_full_tmp, cl_part_tmp);
  cl_full_l1 += cl_full_tmp;
  cl_part_l1 += cl_part_tmp;

}

//
// Count numnber of shared memory transactions for Packed -method
//
void countPackedShTransactions(const int warpSize, const int bankWidth, const int numthread,
  const int volMmk, std::vector<TensorConv>::iterator Msh_it0, std::vector<TensorConv>::iterator Msh_it1,
  int& sld_tran, int& sst_tran, int& sld_req, int& sst_req) {

  for (int j00=0;j00 < volMmk;j00+=numthread) {
    int n0 = std::min(volMmk, j00 + numthread);
    for (int j0=j00;j0 < n0;j0+=warpSize) {
      // Number of accesses for each bank
      std::vector<int> numAccess(warpSize, 0);
      int maxNumAccess = 0;
      int n = std::min(warpSize, volMmk - j0);
      for (int j1=0;j1 < n;j1++) {
        int j = j0 + j1;
        int pos = 0;
        for (auto it=Msh_it0;it != Msh_it1;it++) {
          pos += ((j / it->c) % it->d) * it->ct;
        }
        int bank = pos % bankWidth;
        maxNumAccess = std::max(maxNumAccess, ++numAccess[bank]);
      }
      sld_tran += maxNumAccess;
      sst_tran++;
      sld_req++;
      sst_req++;
    }
  }
}

//
// Count number of global memory transactions for Tiled method
//
void countTiledGlTransactions(const bool isCopy,
  const int numPosMbarSample, const int volMm, const int volMk, const int volMbar,
  const int cIn, const int cOut, const int accWidth, const int cacheWidth,
  std::vector<TensorConvInOut>& hostMbar, const int sizeMbar,
  int& num_iter, float& mlp, int& gld_tran, int& gst_tran, int& gld_req, int& gst_req, int& cl_full, int& cl_part) {

  int ntile = ((volMm - 1)/TILEDIM + 1)*((volMk - 1)/TILEDIM + 1);
  num_iter = volMbar*ntile;

  gld_tran = 0;
  gst_tran = 0;
  gld_req = 0;
  gst_req = 0;
  cl_full = 0;
  cl_part = 0;

  // Random number generator
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, volMbar - 1);

  // Number of elements inside the horizontally clipped tiles
  int h = volMm % TILEDIM;
  // Number of elements inside the vertically clipped tiles
  int v = volMk % TILEDIM;

  // Number of full tiles
  int ntile_full = (volMm/TILEDIM)*(volMk/TILEDIM);
  // Number of tiles that are clipped in horizontal direction
  int ntile_horz = (h > 0)*(volMk/TILEDIM);
  // Number of tiles that are clipped in vertical direction
  int ntile_vert = (v > 0)*(volMm/TILEDIM);
  // Number of corner tiles (0 or 1)
  int ntile_corn = (h > 0)*(v > 0);

  if (isCopy) {
    // Total number of memory level parallelism
    int mlp_tot = (TILEDIM/TILEROWS)*(ntile_full + ntile_horz) + ((v - 1)/TILEROWS + 1)*(ntile_vert + ntile_corn);
    // Average memory level parallelism per tile
    mlp = (float)mlp_tot/(float)ntile;
  } else {
    // Total number of memory level parallelism
    int mlp_tot = (TILEDIM/TILEROWS)*(2*ntile_full + ntile_horz + ntile_vert) + 
    ((v - 1)/TILEROWS + 1)*(ntile_vert + ntile_corn) + ((h - 1)/TILEROWS + 1)*(ntile_horz + ntile_corn);
    // Average memory level parallelism per tile
    mlp = (float)mlp_tot/(float)(2*ntile);
  }

  int num_iposMbar = (numPosMbarSample == 0) ? volMbar : numPosMbarSample;

  for (int iposMbar=0;iposMbar < num_iposMbar;iposMbar++) {
    int posMbar = (numPosMbarSample == 0) ? iposMbar : distribution(generator);

    std::vector<int> posMbarInV(1);
    std::vector<int> posMbarOutV(1);
    computePos(posMbar, posMbar, hostMbar.begin(), hostMbar.begin() + sizeMbar, posMbarInV, posMbarOutV);
    int posMbarIn = posMbarInV[0];
    int posMbarOut = posMbarOutV[0];

    // Reads happen at {posMbarIn, posMbarIn + cuDimMk, posMbarIn + 2*cuDimMk, ..., posMbarIn + (TILEDIM - 1)*cuDimMk}
    // Each tile has same number of transactions

    if (ntile_full > 0) {
      int gld_tran_tmp = 0;
      int gst_tran_tmp = 0;
      int cl_full_tmp = 0;
      int cl_part_tmp = 0;
      for (int i=0;i < TILEDIM;i++) {
        int posIn  = posMbarIn + i*cIn;
        int posOut = posMbarOut + i*cOut;
        gld_tran_tmp += glTransactions(posIn, TILEDIM, accWidth);
        gst_tran_tmp += glTransactions(posOut, TILEDIM, accWidth);
        int cl_full_tmp2, cl_part_tmp2;
        countCacheLines(posOut, TILEDIM, cacheWidth, cl_full_tmp2, cl_part_tmp2);
        cl_full_tmp += cl_full_tmp2;
        cl_part_tmp += cl_part_tmp2;
      }
      gld_tran += gld_tran_tmp*ntile_full;
      gst_tran += gst_tran_tmp*ntile_full;
      cl_full += cl_full_tmp*ntile_full;
      cl_part += cl_part_tmp*ntile_full;
    }

    if (ntile_horz > 0) {
      int gld_tran_tmp = 0;
      int gst_tran_tmp = 0;
      int cl_full_tmp = 0;
      int cl_part_tmp = 0;
      if (isCopy) {
        for (int i=0;i < TILEDIM;i++) {
          int posIn  = posMbarIn + i*cIn;
          int posOut = posMbarOut + i*cOut;
          gld_tran_tmp += glTransactions(posIn, h, accWidth);
          gst_tran_tmp += glTransactions(posOut, h, accWidth);
          int cl_full_tmp2, cl_part_tmp2;
          countCacheLines(posOut, h, cacheWidth, cl_full_tmp2, cl_part_tmp2);
          cl_full_tmp += cl_full_tmp2;
          cl_part_tmp += cl_part_tmp2;
        }
      } else {
        for (int i=0;i < TILEDIM;i++) {
          int posIn  = posMbarIn + i*cIn;
          gld_tran_tmp += glTransactions(posIn, h, accWidth);
        }
        for (int i=0;i < h;i++) {
          int posOut = posMbarOut + i*cOut;
          gst_tran_tmp += glTransactions(posOut, TILEDIM, accWidth);
          int cl_full_tmp2, cl_part_tmp2;
          countCacheLines(posOut, TILEDIM, cacheWidth, cl_full_tmp2, cl_part_tmp2);
          cl_full_tmp += cl_full_tmp2;
          cl_part_tmp += cl_part_tmp2;
        }
      }
      gld_tran += gld_tran_tmp*ntile_horz;
      gst_tran += gst_tran_tmp*ntile_horz;
      cl_full += cl_full_tmp*ntile_horz;
      cl_part += cl_part_tmp*ntile_horz;
    }

    if (ntile_vert > 0) {
      int gld_tran_tmp = 0;
      int gst_tran_tmp = 0;
      int cl_full_tmp = 0;
      int cl_part_tmp = 0;
      if (isCopy) {
        for (int i=0;i < v;i++) {
          int posIn  = posMbarIn + i*cIn;
          int posOut = posMbarOut + i*cOut;
          gld_tran_tmp += glTransactions(posIn, TILEDIM, accWidth);
          gst_tran_tmp += glTransactions(posOut, TILEDIM, accWidth);
          int cl_full_tmp2, cl_part_tmp2;
          countCacheLines(posOut, TILEDIM, cacheWidth, cl_full_tmp2, cl_part_tmp2);
          cl_full_tmp += cl_full_tmp2;
          cl_part_tmp += cl_part_tmp2;
        }
      } else {
        for (int i=0;i < v;i++) {
          int posIn  = posMbarIn + i*cIn;
          gld_tran_tmp += glTransactions(posIn, TILEDIM, accWidth);
        }
        for (int i=0;i < TILEDIM;i++) {
          int posOut = posMbarOut + i*cOut;
          gst_tran_tmp += glTransactions(posOut, v, accWidth);
          int cl_full_tmp2, cl_part_tmp2;
          countCacheLines(posOut, v, cacheWidth, cl_full_tmp2, cl_part_tmp2);
          cl_full_tmp += cl_full_tmp2;
          cl_part_tmp += cl_part_tmp2;
        }
      }
      gld_tran += gld_tran_tmp*ntile_vert;
      gst_tran += gst_tran_tmp*ntile_vert;
      cl_full += cl_full_tmp*ntile_vert;
      cl_part += cl_part_tmp*ntile_vert;
    }

    if (ntile_corn > 0) {
      int gld_tran_tmp = 0;
      int gst_tran_tmp = 0;
      int cl_full_tmp = 0;
      int cl_part_tmp = 0;
      if (isCopy) {
        for (int i=0;i < v;i++) {
          int posIn  = posMbarIn + i*cIn;
          int posOut = posMbarOut + i*cOut;
          gld_tran_tmp += glTransactions(posIn, h, accWidth);
          gst_tran_tmp += glTransactions(posOut, h, accWidth);
          int cl_full_tmp2, cl_part_tmp2;
          countCacheLines(posOut, h, cacheWidth, cl_full_tmp2, cl_part_tmp2);
          cl_full_tmp += cl_full_tmp2;
          cl_part_tmp += cl_part_tmp2;
        }
      } else {
        for (int i=0;i < v;i++) {
          int posIn  = posMbarIn + i*cIn;
          gld_tran_tmp += glTransactions(posIn, h, accWidth);
        }
        for (int i=0;i < h;i++) {
          int posOut = posMbarOut + i*cOut;
          gst_tran_tmp += glTransactions(posOut, v, accWidth);
          int cl_full_tmp2, cl_part_tmp2;
          countCacheLines(posOut, v, cacheWidth, cl_full_tmp2, cl_part_tmp2);
          cl_full_tmp += cl_full_tmp2;
          cl_part_tmp += cl_part_tmp2;
        }
      }
      gld_tran += gld_tran_tmp*ntile_corn;
      gst_tran += gst_tran_tmp*ntile_corn;
      cl_full += cl_full_tmp*ntile_corn;
      cl_part += cl_part_tmp*ntile_corn;
    }

  }
  // Requests
  if (isCopy) {
    gld_req = num_iposMbar*( TILEDIM*ntile_full + TILEDIM*ntile_horz + v*ntile_vert + v*ntile_corn );
    gst_req = gld_req;
  } else {
    gld_req = num_iposMbar*( TILEDIM*ntile_full + TILEDIM*ntile_horz + v*ntile_vert + v*ntile_corn );
    gst_req = num_iposMbar*( TILEDIM*ntile_full + TILEDIM*ntile_vert + h*ntile_horz + h*ntile_corn );
  }
}

struct GpuModelProp {
  double base_dep_delay;
  double base_mem_latency;
  double sh_mem_latency;
  double iter_cycles;
  double fac;
};

void prepmodel5(cudaDeviceProp& prop, GpuModelProp& gpuModelProp,
  int nthread, int numActiveBlock, float mlp,
  int gld_req, int gst_req, int gld_tran, int gst_tran,
  int sld_req, int sst_req, int sld_tran, int sst_tran,
  int cl_full, int cl_part,
  double& delta_ll, double& mem_cycles, double& sh_mem_cycles, double& MWP) {

  double active_SM = prop.multiProcessorCount;
  // Memory bandwidth in GB/s
  double mem_BW = (double)(prop.memoryClockRate*2*(prop.memoryBusWidth/8))/1.0e6;
  if (prop.ECCEnabled) mem_BW *= (1.0 - 0.125);
  // GPU clock in GHz
  double freq = (double)prop.clockRate/1.0e6;
  int warpSize = prop.warpSize;

  int active_warps_per_SM = nthread*numActiveBlock/warpSize;

  // avg. number of memory transactions per memory request
  // double num_trans_per_request = ((double)gld_tran + (double)gst_tran*(1.0 + part_cl)) / (double)(gld_req + gst_req);
  // double num_trans_per_request = ((double)gld_tran + (double)gst_tran + (double)cl_part) / (double)(gld_req + gst_req);
  double cl = (double)cl_part/(double)(cl_full + cl_part);
  double num_trans_per_request = ((double)gld_tran + ((double)gst_tran)*(1.0 + cl)) / (double)(gld_req + gst_req);
  double shnum_trans_per_request = (double)(sld_tran + sst_tran) / (double)(sld_req + sst_req);

  double mem_l = gpuModelProp.base_mem_latency + (num_trans_per_request - 1.0) * gpuModelProp.base_dep_delay;

  const double hitrate = 0.2;

  // Avg. number of memory cycles per warp per iteration
  mem_cycles = gpuModelProp.fac * mem_l * mlp;
  sh_mem_cycles = 2.0 * shnum_trans_per_request * gpuModelProp.sh_mem_latency * mlp;

  // The final value of departure delay
  double dep_delay = num_trans_per_request * gpuModelProp.base_dep_delay;

  // double bytes_per_request = num_trans_per_request*128;
  double bytes_per_request = (num_trans_per_request*(1.0 - hitrate) + hitrate)*128.0;

  delta_ll = gpuModelProp.base_dep_delay;
  double BW_per_warp = freq*bytes_per_request/mem_l;
  double MWP_peak_BW = mem_BW/(BW_per_warp*active_SM);
  MWP = mem_l / dep_delay;
  MWP = std::min(MWP*mlp, std::min(MWP_peak_BW, (double)active_warps_per_SM));
}

double cyclesPacked(const bool isSplit, const size_t sizeofType, cudaDeviceProp& prop,
  int nthread, int numActiveBlock, float mlp, 
  int gld_req, int gst_req, int gld_tran, int gst_tran,
  int sld_req, int sst_req, int sld_tran, int sst_tran, int num_iter, int cl_full, int cl_part) {

  int warps_per_block = nthread/32;

  GpuModelProp gpuModelProp;
  if (prop.major <= 3) {
    // Kepler
    gpuModelProp.base_dep_delay = 14.0;
    gpuModelProp.base_mem_latency = 358.0;
    gpuModelProp.sh_mem_latency = 11.0;
    gpuModelProp.iter_cycles = 50.0;
    gpuModelProp.fac = 2.0;
  } else if (prop.major <= 5) {
    // Maxwell
    gpuModelProp.base_dep_delay = 2.5;
    gpuModelProp.base_mem_latency = 385.0;
    gpuModelProp.sh_mem_latency = 5.0;
    gpuModelProp.iter_cycles = 50.0;
    gpuModelProp.fac = 2.0;
  } else {
    // Pascal and above
    gpuModelProp.base_dep_delay = 2.5;
    gpuModelProp.base_mem_latency = 385.0;
    gpuModelProp.sh_mem_latency = 5.0;
    gpuModelProp.iter_cycles = 50.0;
    gpuModelProp.fac = 1.0;
  } 

  double delta_ll, mem_cycles, sh_mem_cycles, MWP;
  prepmodel5(prop, gpuModelProp, nthread, numActiveBlock, mlp,
    gld_req, gst_req, gld_tran, gst_tran,
    sld_req, sst_req, sld_tran, sst_tran, cl_full, cl_part,
    delta_ll, mem_cycles, sh_mem_cycles, MWP);
  double ldst_cycles = mem_cycles*warps_per_block/MWP;
  double sync_cycles = 0.0;//2.0*delta_ll*(warps_per_block - 1.0);
  double cycles = (ldst_cycles + sh_mem_cycles + sync_cycles + gpuModelProp.iter_cycles)*num_iter;

  return cycles;
}

double cyclesTiled(const bool isCopy, const size_t sizeofType, cudaDeviceProp& prop,
  int nthread, int numActiveBlock, float mlp, 
  int gld_req, int gst_req, int gld_tran, int gst_tran,
  int sld_req, int sst_req, int sld_tran, int sst_tran, int num_iter, int cl_full, int cl_part) {

  int warps_per_block = nthread/32;

  GpuModelProp gpuModelProp;
  if (prop.major <= 3) {
    // Kepler
    gpuModelProp.base_dep_delay = 14.0;
    gpuModelProp.base_mem_latency = 358.0;
    gpuModelProp.sh_mem_latency = 11.0;
    gpuModelProp.iter_cycles = 50.0;
    gpuModelProp.fac = 2.0;
  } else if (prop.major <= 5) {
    // Maxwell
    gpuModelProp.base_dep_delay = 2.5;
    gpuModelProp.base_mem_latency = 385.0;
    if (sizeofType == 4) {
      gpuModelProp.sh_mem_latency = 20.0;
      gpuModelProp.iter_cycles = 90.0;
      gpuModelProp.fac = 2.0;
    } else {
      gpuModelProp.sh_mem_latency = 40.0;
      gpuModelProp.iter_cycles = 110.0;
      gpuModelProp.fac = 2.0;
    }
  } else {
    // Pascal and above
    gpuModelProp.base_dep_delay = 2.5;
    gpuModelProp.base_mem_latency = 385.0;
    gpuModelProp.sh_mem_latency = 5.0;
    gpuModelProp.iter_cycles = 50.0;
    gpuModelProp.fac = 1.4;
  }

  double delta_ll, mem_cycles, sh_mem_cycles, MWP;
  prepmodel5(prop, gpuModelProp, nthread, numActiveBlock, mlp,
    gld_req, gst_req, gld_tran, gst_tran,
    sld_req, sst_req, sld_tran, sst_tran, cl_full, cl_part,
    delta_ll, mem_cycles, sh_mem_cycles, MWP);
  double ldst_cycles = mem_cycles*warps_per_block/MWP;
  double sync_cycles = 0.0;//2.0*delta_ll*(warps_per_block - 1.0);
  if (isCopy) {
    sh_mem_cycles = 0.0;
    sync_cycles = 0.0;
  }
  double cycles = (ldst_cycles + sh_mem_cycles + sync_cycles + gpuModelProp.iter_cycles)*num_iter;

  return cycles;
}

bool check_results(const int tran, const int cl_full, const int cl_part, const int* results) {
  if (tran != results[0] || cl_full != results[1] || cl_part != results[2] ) return false;
  return true;
}


bool testCounters(const int warpSize, const int accWidth, const int cacheWidth) {

  if (warpSize != 32) return false;

  const int numArray = 10;

  const int posData[numArray][32] =
{{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31},
{0,1,2,4,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{43,44,45,46,47,48,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,-1,-1,-1,-1},
{0,3,6,9,12,15,18,21,24,27,30,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
{0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,-1,-1,-1,-1,-1,-1},
{0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124},
{0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248},
{0,1,2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,102,104,106,108,110,112,114,116,118,120,-1,-1,-1,-1,-1},
{5,6,7,8,9,91,92,93,94,95,96,97,98,99,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,-1,-1,-1}};

  const int arrayResultsDouble[numArray][3] =
{{0, 0, 0},
{2, 8, 0},
{1, 0, 2},
{4, 6, 4},
{2, 0, 8},
{5, 0, 19},
{8, 0, 32},
{16, 0, 32},
{5, 0, 18},
{4, 5, 4}};

  const int arrayResultsFloat[numArray][3] = 
{{0, 0, 0},
{1, 4, 0},
{1, 0, 1},
{2, 2, 4},
{1, 0, 4},
{3, 0, 10},
{4, 0, 16},
{8, 0, 32},
{3, 0, 10},
{4, 1, 5}};

  const int contResultsDouble[16*33][3] =
{{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,1,0},{1,1,1},{1,1,1},{1,1,1},{1,2,0},{1,2,1},{1,2,1},{1,2,1},
{1,3,0},{1,3,1},{1,3,1},{1,3,1},{1,4,0},{2,4,1},{2,4,1},{2,4,1},{2,5,0},{2,5,1},{2,5,1},{2,5,1},
{2,6,0},{2,6,1},{2,6,1},{2,6,1},{2,7,0},{2,7,1},{2,7,1},{2,7,1},{2,8,0},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{1,2,2},{1,2,2},{1,2,2},
{1,3,1},{2,3,2},{2,3,2},{2,3,2},{2,4,1},{2,4,2},{2,4,2},{2,4,2},{2,5,1},{2,5,2},{2,5,2},{2,5,2},
{2,6,1},{2,6,2},{2,6,2},{2,6,2},{2,7,1},{3,7,2},{0,0,0},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{1,2,2},{1,2,2},{1,2,2},{1,3,1},{2,3,2},{2,3,2},{2,3,2},
{2,4,1},{2,4,2},{2,4,2},{2,4,2},{2,5,1},{2,5,2},{2,5,2},{2,5,2},{2,6,1},{2,6,2},{2,6,2},{2,6,2},
{2,7,1},{3,7,2},{3,7,2},{0,0,0},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},
{1,2,1},{1,2,2},{1,2,2},{1,2,2},{1,3,1},{2,3,2},{2,3,2},{2,3,2},{2,4,1},{2,4,2},{2,4,2},{2,4,2},
{2,5,1},{2,5,2},{2,5,2},{2,5,2},{2,6,1},{2,6,2},{2,6,2},{2,6,2},{2,7,1},{3,7,2},{3,7,2},{3,7,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,1,0},{1,1,1},{1,1,1},{1,1,1},{1,2,0},{1,2,1},{1,2,1},{1,2,1},
{1,3,0},{2,3,1},{2,3,1},{2,3,1},{2,4,0},{2,4,1},{2,4,1},{2,4,1},{2,5,0},{2,5,1},{2,5,1},{2,5,1},
{2,6,0},{2,6,1},{2,6,1},{2,6,1},{2,7,0},{3,7,1},{3,7,1},{3,7,1},{3,8,0},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,4,1},{2,4,2},{2,4,2},{2,4,2},{2,5,1},{2,5,2},{2,5,2},{2,5,2},
{2,6,1},{3,6,2},{3,6,2},{3,6,2},{3,7,1},{3,7,2},{0,0,0},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},
{2,4,1},{2,4,2},{2,4,2},{2,4,2},{2,5,1},{2,5,2},{2,5,2},{2,5,2},{2,6,1},{3,6,2},{3,6,2},{3,6,2},
{3,7,1},{3,7,2},{3,7,2},{0,0,0},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},
{1,2,1},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,4,1},{2,4,2},{2,4,2},{2,4,2},
{2,5,1},{2,5,2},{2,5,2},{2,5,2},{2,6,1},{3,6,2},{3,6,2},{3,6,2},{3,7,1},{3,7,2},{3,7,2},{3,7,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,1,0},{1,1,1},{1,1,1},{1,1,1},{1,2,0},{2,2,1},{2,2,1},{2,2,1},
{2,3,0},{2,3,1},{2,3,1},{2,3,1},{2,4,0},{2,4,1},{2,4,1},{2,4,1},{2,5,0},{2,5,1},{2,5,1},{2,5,1},
{2,6,0},{3,6,1},{3,6,1},{3,6,1},{3,7,0},{3,7,1},{3,7,1},{3,7,1},{3,8,0},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,4,1},{2,4,2},{2,4,2},{2,4,2},{2,5,1},{3,5,2},{3,5,2},{3,5,2},
{3,6,1},{3,6,2},{3,6,2},{3,6,2},{3,7,1},{3,7,2},{0,0,0},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},
{2,4,1},{2,4,2},{2,4,2},{2,4,2},{2,5,1},{3,5,2},{3,5,2},{3,5,2},{3,6,1},{3,6,2},{3,6,2},{3,6,2},
{3,7,1},{3,7,2},{3,7,2},{0,0,0},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{2,1,2},{2,1,2},{2,1,2},
{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,4,1},{2,4,2},{2,4,2},{2,4,2},
{2,5,1},{3,5,2},{3,5,2},{3,5,2},{3,6,1},{3,6,2},{3,6,2},{3,6,2},{3,7,1},{3,7,2},{3,7,2},{3,7,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,1,0},{2,1,1},{2,1,1},{2,1,1},{2,2,0},{2,2,1},{2,2,1},{2,2,1},
{2,3,0},{2,3,1},{2,3,1},{2,3,1},{2,4,0},{2,4,1},{2,4,1},{2,4,1},{2,5,0},{3,5,1},{3,5,1},{3,5,1},
{3,6,0},{3,6,1},{3,6,1},{3,6,1},{3,7,0},{3,7,1},{3,7,1},{3,7,1},{3,8,0},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{2,0,2},{2,0,2},{2,0,2},{2,1,1},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,4,1},{3,4,2},{3,4,2},{3,4,2},{3,5,1},{3,5,2},{3,5,2},{3,5,2},
{3,6,1},{3,6,2},{3,6,2},{3,6,2},{3,7,1},{3,7,2},{0,0,0},{1,0,1},{1,0,1},{2,0,2},{2,0,2},{2,0,2},
{2,1,1},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},
{2,4,1},{3,4,2},{3,4,2},{3,4,2},{3,5,1},{3,5,2},{3,5,2},{3,5,2},{3,6,1},{3,6,2},{3,6,2},{3,6,2},
{3,7,1},{3,7,2},{3,7,2},{0,0,0},{1,0,1},{2,0,2},{2,0,2},{2,0,2},{2,1,1},{2,1,2},{2,1,2},{2,1,2},
{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,4,1},{3,4,2},{3,4,2},{3,4,2},
{3,5,1},{3,5,2},{3,5,2},{3,5,2},{3,6,1},{3,6,2},{3,6,2},{3,6,2},{3,7,1},{3,7,2},{3,7,2},{3,7,2}};

  const int contResultsFloat[32*33][3] =
{{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,1,0},{1,1,1},{1,1,1},{1,1,1},
{1,1,1},{1,1,1},{1,1,1},{1,1,1},{1,2,0},{1,2,1},{1,2,1},{1,2,1},{1,2,1},{1,2,1},{1,2,1},{1,2,1},
{1,3,0},{1,3,1},{1,3,1},{1,3,1},{1,3,1},{1,3,1},{1,3,1},{1,3,1},{1,4,0},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{1,2,2},{1,2,2},{1,2,2},
{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,3,1},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},
{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},
{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},
{1,3,1},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},
{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},
{1,2,1},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,3,1},{2,3,2},{2,3,2},{2,3,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{1,2,2},{1,2,2},{1,2,2},
{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},
{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},
{1,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},
{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},
{1,2,1},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,3,1},{2,3,2},{2,3,2},{2,3,2},
{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{1,2,2},{1,2,2},{1,2,2},
{1,2,2},{1,2,2},{1,2,2},{1,2,2},{1,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,1,0},{1,1,1},{1,1,1},{1,1,1},
{1,1,1},{1,1,1},{1,1,1},{1,1,1},{1,2,0},{1,2,1},{1,2,1},{1,2,1},{1,2,1},{1,2,1},{1,2,1},{1,2,1},
{1,3,0},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,4,0},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},
{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},
{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},
{2,3,1},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},
{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},
{1,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},
{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},
{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},
{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},
{1,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},
{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,1,2},{1,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,1,0},{1,1,1},{1,1,1},{1,1,1},
{1,1,1},{1,1,1},{1,1,1},{1,1,1},{1,2,0},{2,2,1},{2,2,1},{2,2,1},{2,2,1},{2,2,1},{2,2,1},{2,2,1},
{2,3,0},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,4,0},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},
{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{2,1,2},{2,1,2},{2,1,2},
{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},
{2,3,1},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},
{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},
{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{2,1,2},{2,1,2},{2,1,2},
{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},
{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,2},{1,0,2},{1,0,2},
{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},
{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},
{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},{1,0,2},
{1,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,1,0},{2,1,1},{2,1,1},{2,1,1},
{2,1,1},{2,1,1},{2,1,1},{2,1,1},{2,2,0},{2,2,1},{2,2,1},{2,2,1},{2,2,1},{2,2,1},{2,2,1},{2,2,1},
{2,3,0},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,3,1},{2,4,0},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},
{2,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},
{1,0,1},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,1,1},{2,1,2},{2,1,2},{2,1,2},
{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},
{2,3,1},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{2,0,2},{2,0,2},{2,0,2},
{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},
{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},
{0,0,0},{1,0,1},{1,0,1},{1,0,1},{1,0,1},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},
{2,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},
{1,0,1},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,1,1},{2,1,2},{2,1,2},{2,1,2},
{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},
{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{1,0,1},{2,0,2},{2,0,2},{2,0,2},
{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},
{2,2,1},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},
{2,3,2},{2,3,2},{2,3,2},{0,0,0},{1,0,1},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},{2,0,2},
{2,1,1},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,1,2},{2,2,1},{2,2,2},{2,2,2},{2,2,2},
{2,2,2},{2,2,2},{2,2,2},{2,2,2},{2,3,1},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2},{2,3,2}
};

  //
  // Test array version
  //
  for (int i=0;i < numArray;i++) {
    int n = 0;
    while (posData[i][n] != -1 && n < warpSize) n++;
    int tran = glTransactions(posData[i], n, accWidth);
    int cl_full;
    int cl_part;
    countCacheLines(posData[i], n, cacheWidth, cl_full, cl_part);
    bool ok = true;
    if (accWidth == 16) {
      ok = check_results(tran, cl_full, cl_part, arrayResultsDouble[i]);
    } else {
      ok = check_results(tran, cl_full, cl_part, arrayResultsFloat[i]);
    }
    if (!ok) {
      for (int j=0;j < warpSize;j++) {
        int pos = posData[i][j];
        if (pos == -1) {
          printf("X ");
        } else {
          printf("%d ", pos);
        }
      }
      printf("\n");
      printf("n %d tran %d cl %d %d\n", n, tran, cl_full, cl_part);
      return false;
    }
  }

  int numCont = accWidth*(warpSize + 1);
  int* gpuPosData = new int[(numArray + numCont)*warpSize];

  //
  // Test contigious version
  //
  {
    int i = 0;
    for (int pos=0;pos < accWidth;pos++) {
      for (int n=0;n <= warpSize;n++,i++) {
        int tran = glTransactions(pos, n, accWidth);
        int cl_full;
        int cl_part;
        countCacheLines(pos, n, cacheWidth, cl_full, cl_part);

        std::vector<int> posvec(warpSize, -1);
        for (int i=0;i <n;i++) posvec[i] = pos + i;

        memcpy(&gpuPosData[(numArray + i)*warpSize], posvec.data(), warpSize*sizeof(int));

        int tran2 = glTransactions(posvec.data(), n, accWidth);
        int cl_full2;
        int cl_part2;
        countCacheLines(posvec.data(), n, cacheWidth, cl_full2, cl_part2);

        bool ok = true;
        if (accWidth == 16) {
          ok = check_results(tran, cl_full, cl_part, contResultsDouble[i]);
        } else {
          ok = check_results(tran, cl_full, cl_part, contResultsFloat[i]);
        }

        if (tran != tran2 || cl_full != cl_full2 || cl_part != cl_part2) ok = false;

        if (!ok) {
          printf("%d:%d\n", pos, pos + n - 1);
          printf("tran %d %d cl_full %d %d cl_part %d %d\n", tran, tran2, cl_full, cl_full2, cl_part, cl_part2);
          return false;        
        }

      }
    }
  }

  //
  // Test GPU version
  //
  {
    for (int i=0;i < numArray;i++) {
      memcpy(&gpuPosData[i*warpSize], posData[i], warpSize*sizeof(int));
    }
    int* tran_data = new int[numArray + numCont];
    int* cl_full_data = new int[numArray + numCont];
    int* cl_part_data = new int[numArray + numCont];
    runCounters(warpSize, gpuPosData, (numArray + numCont)*warpSize, accWidth, cacheWidth, tran_data, cl_full_data, cl_part_data);

    for (int i=0;i < numArray;i++) {
      bool ok = true;
      const int *p = (accWidth == 16) ? arrayResultsDouble[i] : arrayResultsFloat[i];
      if (accWidth == 16) {
        ok = check_results(tran_data[i], cl_full_data[i], cl_part_data[i], p);
      } else {
        ok = check_results(tran_data[i], cl_full_data[i], cl_part_data[i], p);
      }
      if (!ok) {
        printf("Array %d\n", i);
        for (int j=0;j < warpSize;j++) {
          int pos = gpuPosData[i*warpSize + j];
          if (pos == -1) {
            printf("X ");
          } else {
            printf("%d ", pos);
          }
        }
        printf("\n");
        printf("tran %d cl %d %d\n", tran_data[i], cl_full_data[i], cl_part_data[i]);
        printf("tran %d cl %d %d\n", p[0], p[1], p[2]);
        return false;
      }
    }

    for (int i=numArray;i < numArray + numCont;i++) {
      bool ok = true;
      const int *p = (accWidth == 16) ? contResultsDouble[i - numArray] : contResultsFloat[i - numArray];
      if (accWidth == 16) {
        ok = check_results(tran_data[i], cl_full_data[i], cl_part_data[i], p);
      } else {
        ok = check_results(tran_data[i], cl_full_data[i], cl_part_data[i], p);
      }
      if (!ok) {
        printf("Cont %d\n", i - numArray);
        for (int j=0;j < warpSize;j++) {
          int pos = gpuPosData[i*warpSize + j];
          if (pos == -1) {
            printf("X ");
          } else {
            printf("%d ", pos);
          }
        }
        printf("\n");
        printf("tran %d cl %d %d\n", tran_data[i], cl_full_data[i], cl_part_data[i]);
        printf("tran %d cl %d %d\n", p[0], p[1], p[2]);
        return false;
      }
    }

    delete [] tran_data;
    delete [] cl_full_data;
    delete [] cl_part_data;
  }

  delete [] gpuPosData;

  return true;
}
