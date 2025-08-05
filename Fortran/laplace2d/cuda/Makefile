
FC = pgfortran
FFLAGS = -O3 -fast -gopt

GPU_NAME   := $(shell nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
ARCH       := $(shell \
    if echo "$(GPU_NAME)" | grep -q "A100"; then echo "80"; \
    elif echo "$(GPU_NAME)" | grep -q "V100"; then echo "70"; \
    elif echo "$(GPU_NAME)" | grep -q "P100"; then echo "60"; \
    fi)

CUDA_FFLAGS = -cuda -gpu=cc$(ARCH)

CUF_SRC = cuda_kernels.CUF
CUF_OBJ = $(CUF_SRC:.CUF=.o)

F90_SRC = laplace2d_cuda.F90
F90_OBJ = $(F90_SRC:.F90=.o)

TARGET = laplace2d_cuda

all: $(TARGET)

$(TARGET): $(CUF_OBJ) $(F90_OBJ)
	$(FC) $(FFLAGS) $(CUDA_FFLAGS) $(CUF_OBJ) $(F90_OBJ) -o $@

%.o: %.CUF
	$(FC) $(CUDA_FFLAGS) -c $< -o $@

%.o: %.F90
	$(FC) $(FFLAGS) $(CUDA_FFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) *.o *.mod
