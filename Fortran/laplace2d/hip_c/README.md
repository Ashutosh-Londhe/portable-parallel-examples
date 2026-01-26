# Hybrid Fortran–HIP Example: Incremental GPU Offloading with Fortran-to-C Bindings

## Overview

This repository demonstrates a **partial hybrid CPU–GPU execution model** in which:

- Part of the computation runs on the **GPU using HIP-C**
- The remaining part runs on the **CPU using Fortran**

The example is designed to support **incremental porting of existing Fortran applications to AMD GPUs** using HIP with **Fortran-to-C bindings**.  
It allows developers to offload selected kernels to the GPU while keeping the rest of the application unchanged in Fortran.

This approach is particularly useful for large legacy Fortran codes where a full GPU port is not feasible in a single step.

---

## Key Features

- Hybrid execution model: **CPU (Fortran) + GPU (HIP-C)**
- Demonstrates **partial offloading** of computation to GPU
- Uses **Fortran-to-C bindings** to interface with HIP kernels
- Includes helper utilities for:
  - Device memory allocation  
  - Host ↔ Device data transfers  

---

## Implementation Variants

This repository contains two implementations of the 2D Laplace solver illustrating different stages of GPU porting:

- **`laplace2d_partial.F90`**  
  Partial implementation demonstrating a **hybrid execution model**, where:
  - Selected kernels are executed on the GPU using HIP-C  
  - The remaining computation runs on the CPU in Fortran  
  This version is intended as a reference for **incremental porting** of existing Fortran applications.

- **`laplace2d.F90`**  
  Full implementation with the **entire computation offloaded to the GPU using HIP**.  
  This version represents a **complete GPU port** of the Laplace solver.

---

## Motivation

Porting large Fortran applications directly to GPUs can be complex and time-consuming.  
This example illustrates a **step-by-step migration strategy**, enabling:

- Validation of GPU kernels in isolation  
- Gradual performance optimization  
- Minimal disruption to existing Fortran codebases  

---

## Structure

- **Fortran code**  
  - Controls overall program flow  
  - Executes CPU-side computations  
  - Invokes GPU kernels via C bindings  

- **HIP-C code**  
  - Implements selected GPU kernels  
  - Manages device execution  

- **Helper routines**  
  - Allocate and free GPU memory  
  - Transfer data between host and device  

---

## Intended Audience

This project is intended for:

- HPC developers working with **legacy Fortran codes**
- Users targeting **AMD GPUs** with HIP
- Developers exploring **incremental GPU offloading strategies**

---

## Notes

This is a **partial working example** meant for demonstration and experimentation.  
It can be extended to support additional kernels, data structures, and performance tuning.

---
