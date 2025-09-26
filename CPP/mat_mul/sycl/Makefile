# ICPX compiler

CC       = icpx
CCFLAGS  = -O3 -march=native -std=c++17 -fPIC -Wall -qopenmp

GPU_VENDOR := $(shell \
    if command -v nvidia-smi >/dev/null 2>&1; then \
        echo NVIDIA; \
    elif command -v rocm-smi >/dev/null 2>&1; then \
        echo AMD; \
    elif command -v intel-smi >/dev/null 2>&1; then \
        echo INTEL; \
    elif command -v lspci >/dev/null 2>&1; then \
        if lspci | grep -i 'vga\|3d\|display' | grep -qi 'nvidia'; then echo NVIDIA; \
        elif lspci | grep -i 'vga\|3d\|display' | grep -qi 'amd'; then echo AMD; \
        elif lspci | grep -i 'vga\|3d\|display' | grep -qi 'intel'; then echo INTEL; \
        else echo UNKNOWN; fi; \
    else \
        echo UNKNOWN; \
    fi)

ifeq ($(GPU_VENDOR), NVIDIA)
SYCL_FLAGS := -fsycl -fsycl-targets=nvptx64-nvidia-cuda
endif

ifeq ($(GPU_VENDOR), AMD)
HIP_ARCH=$(shell rocminfo | grep gfx | sed -e 's/Name://' | head -1 | sed 's/ //g')
SYCL_FLAGS := -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=$(HIP_ARCH)
endif

ifeq ($(GPU_VENDOR), INTEL)
SYCL_FLAGS := -fsycl -fsycl-targets=spir64
endif

BIN =  matrix_mul

all: $(BIN)

$(BIN): matrix_mul_sycl.cpp Makefile
	$(CC) $(CCFLAGS) $(SYCL_FLAGS) -o $@ matrix_mul_sycl.cpp

clean:
	$(RM) $(BIN)

