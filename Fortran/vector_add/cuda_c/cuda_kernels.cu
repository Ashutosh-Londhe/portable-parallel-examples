#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cassert>

#define THREADS_PER_BLOCK 256

inline void checkDeviceError(cudaError_t result, const char *msg) {
    if(result != cudaSuccess) {
        std::cerr<<"CUDA Error: "<<msg<<" : "<<cudaGetErrorString(result)<<std::endl;
        assert(result == cudaSuccess);
    }
}

extern "C"
void device_alloc(int **d_a, int N) {
    checkDeviceError(cudaMalloc((void**)d_a, N*sizeof(int)),"cudaMalloc");
}

extern "C"
void device_memcpy_h2d(int **d_a, int *h_a, int N) {
	/* for(int i = 0; i < 10; i++)
		std::cout<<h_a[i]<<" ";
	std::cout<<"\n"; */
    checkDeviceError(cudaMemcpy(*d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy Host-to-Device");
}

extern "C"
void device_memcpy_d2h(int *h_a, int **d_a, int N) {
    checkDeviceError(cudaMemcpy(h_a, *d_a, N*sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy Device-to-Host");
}

__global__ void vector_add(int *a, int *b, int *c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N)
        c[idx] = a[idx] + b[idx];
}

extern "C"
void device_vector_add(int **a, int **b, int **c, int N) {
    vector_add<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(*a, *b, *c, N);
}

extern "C"
void device_free(int *ptr) {
    checkDeviceError(cudaFree(ptr), "cudaFree");
}
