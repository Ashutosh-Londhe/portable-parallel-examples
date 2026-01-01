# CRAY compiler compiler

FC       = ftn
FCFLAGS  = -O3 -g -fopenmp -f free

BIN = laplace2d

all: $(BIN)

laplace2d: laplace2d_offload.F90 Makefile
	$(FC) $(FCFLAGS) -o $@ laplace2d_offload.F90

clean:
	$(RM) $(BIN)
