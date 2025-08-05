# GNU compiler

CC       = g++
CCFLAGS  = -O3 -fPIC -Wall -ffloat-store -g -std=c++11 -fopenmp

BIN =  laplace2d

all: $(BIN)

laplace2d: laplace2d_original.cpp Makefile
	$(CC) $(CCFLAGS) -o $@ laplace2d_original.cpp

clean:
	$(RM) $(BIN)

