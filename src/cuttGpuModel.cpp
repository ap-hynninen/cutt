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
#include "cuttGpuModel.h"

//
// Count number of global memory transactions per one request for potentially
// scattered accesses to elements listed in pos
// NOTE: Assumes pos is sorted
//
int glTransactions(std::vector<int>& pos, const int n, const int accWidth) {
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
  int count;
  if (n == 1) {
    count = 1;
  } else if (n <= accWidth) {
    // (pos % accWidth) is the position in segment where access starts
    count = 1 + ((pos % accWidth) + n > accWidth);
  } else {
    // n > accWidth, hence at least two transactions, third one if
    // access does not start aligned
    count = 2 + ((pos % accWidth) != 0);
  }
  return count;
}


//
// Count number of full and partial cache-lines accessed for potentially
// scattered accesses to elements listed in pos
//
// cl_full = Number of full cache lines
// cl_part = Number of partial cache lines
//
void countCacheLines(std::vector<int>& pos, const int cacheWidth, int& cl_full, int& cl_part) {

  cl_full = 0;
  cl_part = 0;
  for (int i=0;i < pos.size();) {
    int i0 = i;
    int n = std::min(i + 3, (int)pos.size() - 1);
    int seg0 = pos[i++]/cacheWidth;
    while (i <= n) {
      int seg = pos[i]/cacheWidth;
      if (seg0 != seg) {
        break;
      }
      i++;
    }
    if (i == i0 + 4) {
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
// Count number of global memory transactions for General -method
//
void countGeneralGlTransactions(const int warpSize, const int accWidth, const int cacheWidth,
  const int numthread, const int posMbarIn, const int posMbarOut, const int volMmk, 
  std::vector<int>& posMmkIn, std::vector<int>& posMmkOut,
  int& gld_tran, int& gst_tran, int& gld_req, int& gst_req, int& cl_full, int& cl_part) {

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
        writePosVolMmk[m++] = posOut;
      }
      // Global memory transactions
      gld_tran += glTransactions(readPos, n, accWidth);
      gst_tran += glTransactions(writePos, n, accWidth);
      gld_req++;
      gst_req++;
    }
  }
  // Global write non-full cache-lines
  int cl_full_tmp, cl_part_tmp;
  countCacheLines(writePosVolMmk, cacheWidth, cl_full_tmp, cl_part_tmp);
  cl_full += cl_full_tmp;
  cl_part += cl_part_tmp;
}

//
// Count numnber of shared memory transactions for General -method
//
void countGeneralShTransactions(const int warpSize, const int bankWidth, const int numthread,
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
void countTiledGlTransactions(const int numPosMbarSample, const int volMm, const int volMk, const int volMbar,
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

  // Total number of memory level parallelism
  int mlp_tot = (TILEDIM/TILEROWS)*(ntile_full + ntile_horz) + ((v - 1)/TILEROWS + 1)*(ntile_vert + ntile_corn);
  // Average memory level parallelism per tile
  mlp = (float)mlp_tot/(float)ntile;

  for (int iposMbar=0;iposMbar < numPosMbarSample;iposMbar++) {
    int posMbar = distribution(generator);

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
      gld_tran += gld_tran_tmp*ntile_corn;
      gst_tran += gst_tran_tmp*ntile_corn;
      cl_full += cl_full_tmp*ntile_corn;
      cl_part += cl_part_tmp*ntile_corn;
    }

  }
  // Requests
  gld_req = numPosMbarSample*( TILEDIM*ntile_full + TILEDIM*ntile_horz + v*ntile_vert + v*ntile_corn );
  gst_req = gld_req;
}

/*
void prepmodel3(cudaDeviceProp& prop, 
  int nthread, int numActiveBlock, float mlp, int insts, int req, int tran,
  int sh_req, int sh_tran, double cl, double& delta_ll, double& mem_cycles, double& MWP) {

  double active_SM = prop.multiProcessorCount;
  // Memory bandwidth in GB/s
  double mem_BW = (double)(prop.memoryClockRate*2*(prop.memoryBusWidth/8))/1.0e6;
  // GPU clock in GHz
  double freq = (double)prop.clockRate/1.0e6;
  int warpSize = prop.warpSize;
  // printf("active_SM %1.1lf mem_BW %1.2lf freq %1.4lf\n", active_SM, mem_BW, freq);
  // Delays & latencies in cycles
  int base_dep_delay = 7;
  int base_mem_latency = 230;
  int sh_mem_latency = 20;  
  int unaligned_dep_delay = 70;
  int mlp_delay = 20;

  int active_warps_per_SM = nthread*numActiveBlock/warpSize;

  // avg. number of memory transactions per memory request
  double num_trans_per_request = (double)tran / (double)req;

  double sh_num_trans_per_request = (double)sh_tran / (double)sh_req;

  double dep_delay = cl*unaligned_dep_delay + (1.0 - cl)*base_dep_delay;

  double mem_l = (1 + 0.2*cl)*base_mem_latency + (num_trans_per_request - 1.0) * dep_delay + 
  (mlp - 1.0f)*mlp_delay + (sh_num_trans_per_request - 1)*sh_mem_latency;

  // Avg. number of memory cycles per warp per iteration
  mem_cycles = mem_l * insts;

  // The final value of departure delay
  double delta_l = num_trans_per_request * dep_delay;

  double bytes_per_request = num_trans_per_request*128;

  delta_ll = dep_delay;
  double BW_per_warp = freq*bytes_per_request/mem_l;
  double MWP_peak_BW = mem_BW/(BW_per_warp*active_SM);
  MWP = mem_l / delta_l;
  MWP = std::min(MWP*mlp, std::min(MWP_peak_BW, (double)active_warps_per_SM));
}
*/

void prepmodel4(cudaDeviceProp& prop, 
  int nthread, int numActiveBlock, float mlp, int insts, int req, int tran,
  int sh_req, int sh_tran, double cl,
  double& delta_ll, double& mem_cycles, double& sh_mem_cycles, double& MWP) {

  double active_SM = prop.multiProcessorCount;
  // Memory bandwidth in GB/s
  double mem_BW = (double)(prop.memoryClockRate*2*(prop.memoryBusWidth/8))/1.0e6;
  if (prop.ECCEnabled) mem_BW *= (1.0 - 0.125);
  // GPU clock in GHz
  double freq = (double)prop.clockRate/1.0e6;
  int warpSize = prop.warpSize;
  // printf("active_SM %1.1lf mem_BW %1.2lf freq %1.4lf\n", active_SM, mem_BW, freq);
  // Delays & latencies in cycles
  double base_dep_delay = 10.0;
  double base_mem_latency = 560.0;
  double sh_mem_latency = 1.0;

  int active_warps_per_SM = nthread*numActiveBlock/warpSize;

  // avg. number of memory transactions per memory request
  double num_trans_per_request = (double)tran / (double)req;

  double sh_num_trans_per_request = (double)sh_tran / (double)sh_req;

  double mem_l = (1.0 + cl)*base_mem_latency + (num_trans_per_request - 1.0) * base_dep_delay;

  // Avg. number of memory cycles per warp per iteration
  mem_cycles = mem_l * insts;
  sh_mem_cycles = sh_num_trans_per_request * sh_mem_latency * insts;

  // The final value of departure delay
  double delta_l = num_trans_per_request * base_dep_delay;

  double bytes_per_request = num_trans_per_request*128;

  delta_ll = base_dep_delay;
  double BW_per_warp = freq*bytes_per_request/mem_l;
  double MWP_peak_BW = mem_BW/(BW_per_warp*active_SM);
  MWP = mem_l / delta_l;
  MWP = std::min(MWP*mlp, std::min(MWP_peak_BW, (double)active_warps_per_SM));
}

double cyclesGeneral(cudaDeviceProp& prop, int nthread, int numActiveBlock, int numRegStorage, 
  int gld_req, int gst_req, int gld_tran, int gst_tran,
  int sld_req, int sst_req, int sld_tran, int sst_tran, int num_iter, double cl) {

  int warps_per_block = nthread/32;

  // double delta_ll, mem_cycles, MWP;
  // prepmodel3(prop, nthread, numActiveBlock, (float)numRegStorage, numRegStorage,
  //   gld_req + gst_req, gld_tran + gst_tran,
  //   sld_req + sst_req, sld_tran + sst_tran, cl,
  //   delta_ll, mem_cycles, MWP);
  // double iter_cycles = 10.0*num_iter;
  // double ldst_cycles = (2.0*mem_cycles*warps_per_block/MWP)*num_iter;
  // double sync_cycles = delta_ll*2.0*(warps_per_block - 1)*num_iter;
  // double cycles = ldst_cycles + sync_cycles + iter_cycles;

  double delta_ll, mem_cycles, sh_mem_cycles, MWP;

  prepmodel4(prop, nthread, numActiveBlock, (float)numRegStorage, numRegStorage,
    gld_req + gst_req, gld_tran + gst_tran,
    sld_req + sst_req, sld_tran + sst_tran, cl,
    delta_ll, mem_cycles, sh_mem_cycles, MWP);
  double ldst_cycles = 2.0*mem_cycles*warps_per_block/MWP;
  double sync_cycles = 2.0*delta_ll*(warps_per_block - 1.0);
  double iter_cycles = 20.0;
  double cycles = (ldst_cycles + sh_mem_cycles + sync_cycles + iter_cycles)*num_iter;

  return cycles;
}

double cyclesTiled(cudaDeviceProp& prop, int nthread, int numActiveBlock, float mlp, 
  int gld_req, int gst_req, int gld_tran, int gst_tran,
  int sld_req, int sst_req, int sld_tran, int sst_tran, int num_iter, double cl) {

  int warps_per_block = nthread/32;

  // double delta_ll, mem_cycles, MWP;
  // prepmodel3(prop, nthread, numActiveBlock, mlp, TILEDIM/TILEROWS,
  //   gld_req + gst_req, gld_tran + gst_tran,
  //   sld_req + sst_req, sld_tran + sst_tran, cl,
  //   delta_ll, mem_cycles, MWP);
  // double iter_cycles = 10.0*num_iter;
  // double ldst_cycles = (2.0*mem_cycles*warps_per_block/MWP)*num_iter;
  // double sync_cycles = delta_ll*2.0*(warps_per_block - 1)*num_iter;
  // double cycles = ldst_cycles + sync_cycles + iter_cycles;

  double delta_ll, mem_cycles, sh_mem_cycles, MWP;
  prepmodel4(prop, nthread, numActiveBlock, mlp, mlp,
    gld_req + gst_req, gld_tran + gst_tran,
    sld_req + sst_req, sld_tran + sst_tran, cl,
    delta_ll, mem_cycles, sh_mem_cycles, MWP);
  double ldst_cycles = 2.0*mem_cycles*warps_per_block/MWP;
  double sync_cycles = 2.0*delta_ll*(warps_per_block - 1.0);
  double iter_cycles = 20.0;
  double cycles = (ldst_cycles + sh_mem_cycles + sync_cycles + iter_cycles)*num_iter;

  return cycles;
}
