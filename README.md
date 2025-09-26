# 🚀 Portable Parallel Examples

This repository demonstrates architecture-specific parallel computing examples using CUDA, HIP, and SYCL.
The goal is to showcase portable approaches for running computations efficiently on different GPU architectures.


🌟 Features

✅ Multiple GPU Backends: Implementations using CUDA (NVIDIA), HIP (AMD), and SYCL (cross-platform).
✅ Portable Examples: Code designed to illustrate best practices for porting algorithms across architectures.
✅ Simple Build and Run: Each subdirectory contains its own build setup (Makefile or CMake) for quick experimentation.
✅ Demonstrates Performance Patterns: Covers common parallel programming patterns, memory allocation, and kernel offloading techniques.


⚡ Getting Started
1. Install Required Toolkits:
   -  CUDA: NVIDIA GPU support.
   -  HIP: AMD GPU support.
   - SYCL: Cross-platform accelerator support (e.g., Intel oneAPI, Codeplay ComputeCpp).
2. Build the Examples:
   Navigate to the desired backend directory (cuda, hip, or sycl) and follow the instructions in the included Makefile or CMakeLists.txt.
3. Run the Binaries:
   Execute the compiled binaries on the target GPU. Make sure environment variables like CUDA_HOME or HIP_PATH are set correctly.


💡 Notes
 - Subdirectories marked with _c show Fortran-to-C bindings, allowing Fortran programs to interface with CUDA, HIP, and SYCL implementations.
 - Examples are educational: they illustrate patterns for parallel programming and portability, not full production-ready applications.


📌 Recommended Workflow
 - Choose the target GPU backend (cuda, hip, or sycl).
 - Study the example code to understand memory allocation, kernel launches, and parallel patterns.
 - Modify or extend the examples to experiment with your own algorithms or Fortran–C bindings.


✨ Quick Tips
 - Always check that your GPU architecture flags match your hardware (-arch=sm_XX for CUDA, etc.).
 - For Fortran-to-C experiments, _c subdirectories provide a good starting point for porting legacy Fortran code.
 - Use nvcc, hipcc, or your SYCL compiler for linking — Makefiles already handle CUDA/HIP/SYCL libraries via variables like CUDA_LIB.
