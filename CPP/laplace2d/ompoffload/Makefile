# CRAY compiler

CC       = CC
CCFLAGS  = -O3 -std=c++17 -g -fPIC -fopenmp

BIN =  laplace2d

all: $(BIN)

laplace2d: laplace2d_offload.cpp Makefile
	$(CC) $(CCFLAGS) -o $@ laplace2d_offload.cpp

clean:
	$(RM) $(BIN)

