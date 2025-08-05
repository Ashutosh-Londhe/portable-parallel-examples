# NVHPC compiler

FC       = pgfortran
FCFLAGS  = -O3 -fast -gopt -mp

BIN = laplace2d

all: $(BIN)

laplace2d: laplace2d_original.F90 Makefile
	$(FC) $(FCFLAGS) -o $@ laplace2d_original.F90

clean:
	$(RM) $(BIN)
