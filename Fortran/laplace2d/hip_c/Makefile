# Makefile (link with g++)

#FC     = gfortran
FC     = ftn
FFLAGS   = -O3

HIPCC = hipcc
HIPFLAGS = -O3 -g -D__HIP_PLATFORM_AMD__

HIP_ARCH = $(shell rocminfo | grep gfx | sed -e 's/Name://' | head -1 | sed 's/ //g')

HIPFLAGS += --offload-arch=$(HIP_ARCH)

HIP_SRC = hip_kernels.cpp
HIP_OBJ = $(HIP_SRC:.cpp=.o)

F90_SRC = laplace2d.F90
F90_OBJ =  $(F90_SRC:.F90=.o)

TARGET = laplace2d

all: $(TARGET)

$(TARGET): $(HIP_OBJ) $(F90_OBJ)
	$(FC) $(HIP_OBJ) $(F90_OBJ) -L$(ROCM_PATH)/lib -lamdhip64 -lrocm_smi64 -lstdc++ -o $@

%.o: %.cpp
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

%.o: %.F90
	$(FC) $(FFLAGS) -c $< -o $@

clean:
	rm -f $(HIP_OBJ) $(F90_OBJ) $(TARGET)

