#******************************************************************************
#MIT License
#
#Copyright (c) 2016 Antti-Pekka Hynninen
#Copyright (c) 2016 Oak Ridge National Laboratory (UT-Batelle)
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#*******************************************************************************

#################### User Settings ####################

# C++ compiler
CC = g++

# CUDA compiler
CUDAC = nvcc

# SM versions for which code is generated must be sm_30 and above
GENCODE_SM35  := -gencode arch=compute_35,code=sm_35
GENCODE_SM50  := -gencode arch=compute_50,code=sm_50
GENCODE_FLAGS := $(GENCODE_SM35) $(GENCODE_SM50)

#######################################################

# Detect OS
ifeq ($(shell uname -a|grep Linux|wc -l), 1)
OS = linux
endif

ifeq ($(shell uname -a|grep titan|wc -l), 1)
OS = titan
endif

ifeq ($(shell uname -a|grep Darwin|wc -l), 1)
OS = osx
endif

# Set optimization level
OPTLEV = -O2

OBJSLIB = cutt.o cuttplan.o cuttkernel.o CudaUtils.o

OBJSTEST = cutt_test.o cutt.o cuttplan.o cuttkernel.o CudaUtils.o TensorTester.o
OBJSBENCH = cutt_bench.o cutt.o cuttplan.o cuttkernel.o CudaUtils.o TensorTester.o
OBJS = $(OBJSTEST) $(OBJSBENCH)

#CUDAROOT = $(subst /bin/,,$(dir $(shell which nvcc)))
CUDAROOT = $(subst /bin/,,$(dir $(shell which $(CUDAC))))

CFLAGS = -I${CUDAROOT}/include -std=gnu++11

CUDA_CFLAGS = -I${CUDAROOT}/include $(OPTLEV) -lineinfo $(GENCODE_FLAGS) -Xcompiler "$(CUDA_CCFLAGS)"

ifeq ($(OS),linux)
CUDA_LFLAGS = -L$(CUDAROOT)/lib64
else
ifeq ($(OS),titan)
CUDA_LFLAGS = -L$(CUDAROOT)/lib64
else
CUDA_LFLAGS = -L$(CUDAROOT)/lib
endif
endif
CUDA_LFLAGS += -lcudart -Llib

all: lib/libcutt.a bin/cutt_test bin/cutt_bench

lib/libcutt.a: $(OBJSLIB)
	mkdir -p lib
	ar -cvq lib/libcutt.a $(OBJSLIB)
	mkdir -p include

bin/cutt_test : $(OBJSTEST)
	mkdir -p bin
	$(CUDAC) $(CUDA_LFLAGS) -o bin/cutt_test $(OBJSTEST)

bin/cutt_bench : $(OBJSBENCH)
	mkdir -p bin
	$(CUDAC) $(CUDA_LFLAGS) -o bin/cutt_bench $(OBJSBENCH)

clean: 
	rm -f *.o
	rm -f *.d
	rm -f *~
	rm -f lib/libcutt.a
	rm -f bin/cutt_test
	rm -f bin/cutt_bench

# Pull in dependencies that already exist
-include $(OBJS:.o=.d)

%.o : %.cu
	$(CUDAC) -c $(CUDA_CFLAGS) $(DEFS) $<
	$(CUDAC) -M $(CUDA_CFLAGS) $(DEFS) $*.cu > $*.d

%.o : %.cpp
	$(CC) -c $(CFLAGS) $(DEFS) $<
	$(CC) -M $(CFLAGS) $(DEFS) $*.cpp > $*.d
