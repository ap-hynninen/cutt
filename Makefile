
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

YES := $(shell which make | wc -l 2> /dev/null)
NO := $(shell which pikaboo | wc -l 2> /dev/null)

# Set optimization level
#OPTLEV = -g
OPTLEV = -O2

# Detect CUDA

CUDA_COMPILER := $(shell which nvcc | wc -l 2> /dev/null)

#CC = $(CUDA_COMPILER)
#CL = $(CUDA_COMPILER)

CC = g++

OBJS3D = CudaTranspose.o CudaUtils.o gpu3d.o
OBJSTEST = cutt_test.o cutt.o cuttplan.o cuttkernel.o CudaUtils.o TensorTester.o
OBJSBENCH = cutt_bench.o cutt.o cuttplan.o cuttkernel.o CudaUtils.o TensorTester.o
OBJS = $(OBJS3D) $(OBJSTEST) $(OBJSBENCH)

ifeq ($(CUDA_COMPILER), $(YES))
CUDAROOT = $(subst /bin/,,$(dir $(shell which nvcc)))
endif

GENCODE_SM35  := -gencode arch=compute_35,code=sm_35
GENCODE_SM50  := -gencode arch=compute_50,code=sm_50
GENCODE_FLAGS := $(GENCODE_SM35) $(GENCODE_SM50)

# CUDA_CFLAGS = flags for compiling with CUDA
# CUDA_LFLAGS = flags for linking with CUDA

CFLAGS = -I${CUDAROOT}/include -std=gnu++11

CUDA_CFLAGS = -I${CUDAROOT}/include $(OPTLEV) -lineinfo $(GENCODE_FLAGS) --resource-usage -Xcompiler "$(CUDA_CCFLAGS)"

ifeq ($(OS),linux)
CUDA_LFLAGS = -L$(CUDAROOT)/lib64
else
ifeq ($(OS),titan)
CUDA_LFLAGS = -L$(CUDAROOT)/lib64
else
CUDA_LFLAGS = -L$(CUDAROOT)/lib
endif
endif
CUDA_LFLAGS += -lcudart

BINARIES = gpu3d cutt_test cutt_bench

all: $(BINARIES)

gpu3d : $(OBJS3D)
	nvcc $(CUDA_LFLAGS) -o gpu3d $(OBJS3D)

cutt_test : $(OBJSTEST)
	nvcc $(CUDA_LFLAGS) -o cutt_test $(OBJSTEST)

cutt_bench : $(OBJSBENCH)
	nvcc $(CUDA_LFLAGS) -o cutt_bench $(OBJSBENCH)

clean: 
	rm -f *.o
	rm -f *.d
	rm -f *~
	rm -f $(BINARIES)

# Pull in dependencies that already exist
-include $(OBJS:.o=.d)

%.o : %.cu
	nvcc -c $(CUDA_CFLAGS) $(DEFS) $<
	nvcc -M $(CUDA_CFLAGS) $(DEFS) $*.cu > $*.d

%.o : %.cpp
	$(CC) -c $(CFLAGS) $(DEFS) $<
	$(CC) -M $(CFLAGS) $(DEFS) $*.cpp > $*.d
