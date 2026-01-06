# 🚀 Porting Laplace2D Fortran to SYCL (Intel, AMD & NVIDIA GPUs)

This repository demonstrates how to port the Laplace2D Fortran program to SYCL and run it on Intel, AMD, or NVIDIA GPUs.


## 🔧 Overview  

✅ Uses Intel oneAPI compilers for building the application.  
✅ Shows how to compile and link mixed Fortran + SYCL/C++ code.  
✅ Works with Intel, AMD, and NVIDIA GPU targets.  


## 🧰 Compiler Usage

1. Intel C++ Compiler (icpx)
   - Compiles SYCL/C++ (.cpp) source files.
2. Intel Fortran Compiler (ifx)
   - Compiles Fortran (.F90) source files.


## 🧩 Linking Strategy
### ⚡ Intel GPU Targets
   - Final linking can be done directly using the Fortran compiler (ifx).  
   - No changes to the original Fortran program are required.  


### ⚠️ AMD & NVIDIA GPU Targets
   - ifx cannot be used for final linking with AMD or NVIDIA GPU object files.  
   - Attempting to link with ifx when building for AMD GPU will results in following error:  
```vbnet
ifx: command line error: target architecture triple setting of 'amdgcn' not supported with ifx; use the Intel oneAPI DPC++/C++ Compiler with option '-fortlib'
```
## 🛠 Workaround (for AMD / NVIDIA)

To address the linking limitation:
   - Convert the original Fortran program into a Fortran subroutine.  
   - Create a small main.cpp that calls this Fortran subroutine.  
   - Use the Intel C++ compiler (icpx) for the final link step.  

This approach enables successful linking for AMD and NVIDIA GPU builds.

## 📌 Build Instructions

See the Makefile for detailed build targets and options.
