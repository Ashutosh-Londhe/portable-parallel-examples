# Not checked, use CMAKE to build

KOKKOS_PATH = ${HOME}/Kokkos/kokkos
SRC = $(wildcard *.cpp)
KOKKOS_DEVICES=OpenMP
KOKKOS_ARCH=SKX

default: test

CXX = g++
CXXFLAGS = -O3 -g
LINK = ${CXX}

OBJ = $(SRC:.cpp=.o)

include $(KOKKOS_PATH)/Makefile.kokkos

test: $(OBJ) $(KOKKOS_LINK_DEPENDS)
    $(LINK) $(KOKKOS_LDFLAGS) $(OBJ) $(KOKKOS_LIBS) -o laplace2d

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
    $(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) -c $<
