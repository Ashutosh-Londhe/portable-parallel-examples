
HIPCC = hipcc
HIPFLAGS = -O3 -g -D__HIP_PLATFORM_AMD__ 

HIP_ARCH = $(shell rocminfo | grep gfx | sed -e 's/Name://' | head -1 | sed 's/ //g')

HIPFLAGS += --offload-arch=$(HIP_ARCH)

HIP_SRC = vector_add.cpp
HIP_OBJ = $(HIP_SRC:.cpp=.o)

TARGET = vector_add_hip

all: $(TARGET)

$(TARGET): $(HIP_OBJ)
	$(HIPCC) $(HIP_OBJ) -o $@

%.o: %.cpp
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) *.o
	
