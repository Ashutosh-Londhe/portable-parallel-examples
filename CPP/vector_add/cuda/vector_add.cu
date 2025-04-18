#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cassert>

#define N 4096
#define THREADS_PER_BLOCK 256

void init_array(int *a) {
    for(int i = 0; i< N; i++)
        a[i] = rand()%100;
}

inline void checkDeviceError(cudaError_t result, const char *msg) {
    if(result != cudaSuccess) {
        std::cerr<<"CUDA Error: "<<msg<<" : "<<cudaGetErrorString(result)<<std::endl;
        assert(result == cudaSuccess);
    }
}

__global__ void vector_add(int *a, int *b, int *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N)
        c[idx] = a[idx] + b[idx];
}

int main() {

    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;

    h_a = (int *) calloc(N, sizeof(int));
    h_b = (int *) calloc(N, sizeof(int));
    h_c = (int *) calloc(N, sizeof(int));

    srand(1);

    init_array(h_a);
    init_array(h_b);

    checkDeviceError(cudaMalloc((void**)&d_a, N*sizeof(int)),"cudaMalloc d_a");
    checkDeviceError(cudaMalloc((void**)&d_b, N*sizeof(int)),"cudaMalloc d_b");
    checkDeviceError(cudaMalloc((void**)&d_c, N*sizeof(int)),"cudaMalloc d_c");

    checkDeviceError(cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy h_a -> d_a");
    checkDeviceError(cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy h_b -> d_b");

    vector_add<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

    checkDeviceError(cudaGetLastError(), "vector_add launch");

    cudaDeviceSynchronize();

    checkDeviceError(cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy d_c -> h_c");

    for(int i = 0; i < N; i++) {
        if(h_c[i] != h_a[i]+h_b[i]) {
            std::cout<<"FAILED !!!"<<std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::cout<<"PASSED !!!"<<std::endl;

    free(h_a);      free(h_b);      free(h_c);

    checkDeviceError(cudaFree(d_a), "cudaFree d_a");
    checkDeviceError(cudaFree(d_b), "cudaFree d_b");
    checkDeviceError(cudaFree(d_c), "cudaFree d_c");

    return 0;
}
