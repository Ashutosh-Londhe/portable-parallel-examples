# Makefile (link with g++)

FC     = gfortran
FFLAGS   = -O3 -Wall

NVCC = nvcc
NVCCFLAGS = -O3 -std=c++11 -Xcompiler="-fPIC"

CUDA_HOME ?= $(shell dirname $(shell which nvcc))/..
CUDA_LIB = $(CUDA_HOME)/lib64
CUDA_INC = $(CUDA_HOME)/include

GPU_NAME   := $(shell nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
ARCH       := $(shell \
	if echo "$(GPU_NAME)" | grep -q "A100"; then echo "80"; \
	elif echo "$(GPU_NAME)" | grep -q "V100"; then echo "70"; \
	elif echo "$(GPU_NAME)" | grep -q "P100"; then echo "60"; \
	fi)

NVCCFLAGS += -gencode arch=compute_$(ARCH),code=sm_$(ARCH)

CU_SRC = cuda_kernels.cu
CU_OBJ = $(CU_SRC:.cu=.o)

F90_SRC = fortran_main.F90
F90_OBJ =  $(F90_SRC:.F90=.o)

TARGET = vector_add

all: $(TARGET)

$(TARGET): $(CU_OBJ) $(F90_OBJ)
	$(FC) $(CU_OBJ) $(F90_OBJ) -L$(CUDA_LIB) -lcudart -lstdc++ -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.F90
	$(FC) $(FFLAGS) -c $< -o $@

clean:
	rm -f $(CU_OBJ) $(F90_OBJ) $(TARGET)

