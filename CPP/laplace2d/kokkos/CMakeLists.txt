cmake_minimum_required(VERSION 3.16)
project(Laplace2d_Kokkos LANGUAGES CXX)

# Find Kokkos (must be installed already, e.g. with HIP backend enabled)
find_package(Kokkos REQUIRED)

add_executable(laplace2d laplace2d_kokkos.cpp)

# Link against Kokkos
target_link_libraries(laplace2d PRIVATE Kokkos::kokkos)

