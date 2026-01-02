# CRAY compiler

CC       = CC
CCFLAGS  = -O3 -std=c++17 -g -fPIC -fopenmp

BIN = lattboltz2d

all: $(BIN)

lattboltz2d: lattboltz2d_offload.cpp Makefile
	$(CC) $(CCFLAGS) -o $@ lattboltz2d_offload.cpp

clean:
	$(RM) $(BIN)

