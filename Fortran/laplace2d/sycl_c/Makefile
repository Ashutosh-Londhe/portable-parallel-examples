# Using Intel ICX compiler

FC     = ifx
FFLAGS   = -O3 -march=native -fPIC -qopenmp

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
#CPU
#SYCL_FLAGS := -fsycl -fsycl-targets=spir64_x86_64
endif

ifeq ($(GPU_VENDOR), INTEL)
SYCL_FLAGS := -fsycl -fsycl-targets=spir64
endif

#SYCL_SRC = sycl_kernels.cpp 
#ifneq ($(GPU_VENDOR), INTEL)
#SYCL_SRC += main.cpp
#endif
SYCL_SRC = $(wildcard *.cpp)
SYCL_OBJ = $(SYCL_SRC:.cpp=.o)

ifeq ($(GPU_VENDOR), INTEL)
F90_SRC = laplace2d_program.F90
else
F90_SRC = laplace2d_subroutine.F90
endif

F90_OBJ =  $(F90_SRC:.F90=.o)

TARGET = laplace2d

all: $(TARGET)

# -lsycl
$(TARGET): $(SYCL_OBJ) $(F90_OBJ)
ifeq ($(GPU_VENDOR), INTEL)
	$(FC) $(FFLAGS) $(SYCL_FLAGS) $(SYCL_OBJ) $(F90_OBJ) -lstdc++ -o $@
else
	$(CC) $(CCFLAGS) $(SYCL_FLAGS) $(SYCL_OBJ) $(F90_OBJ) -lstdc++ -fortlib -o $@
endif

%.o: %.cpp
	$(CC) $(CCFLAGS) $(SYCL_FLAGS) -c $< -o $@

%.o: %.F90
	$(FC) $(FFLAGS) -c $< -o $@

clean:
	rm -f $(SYCL_OBJ) $(F90_OBJ) $(TARGET)

