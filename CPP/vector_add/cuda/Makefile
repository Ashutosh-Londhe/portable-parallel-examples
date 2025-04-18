
NVCC = nvcc
NVCCFLAGS = -O3 -std=c++11 -Xcompiler="-fPIC"

GPU_NAME   := $(shell nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
ARCH       := $(shell \
    if echo "$(GPU_NAME)" | grep -q "A100"; then echo "80"; \
    elif echo "$(GPU_NAME)" | grep -q "V100"; then echo "70"; \
    elif echo "$(GPU_NAME)" | grep -q "P100"; then echo "60"; \
    fi)

NVCCFLAGS += -gencode arch=compute_$(ARCH),code=sm_$(ARCH)

CU_SRC = vector_add.cu
CU_OBJ = $(CU_SRC:.cu=.o)

TARGET = vector_add_cuda

all: $(TARGET)

$(TARGET): $(CU_OBJ)
	$(NVCC) $(CU_OBJ) -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) *.o




