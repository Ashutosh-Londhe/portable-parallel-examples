#include <iostream>
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cassert>

#define N 4096
#define THREADS_PER_BLOCK 256

void init_array(int *a) {
    for(int i = 0; i < N; i++)
        a[i] = rand()%100;
}

inline void checkDeviceError(hipError_t result, char *msg) {
    if(result != hipSuccess) {
        std::cerr<<"HIP Error: "<<msg<<" : "<<hipGetErrorString(result)<<std::endl;
        assert(result == hipSuccess);
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

    checkDeviceError(hipMalloc((void**)&d_a, N*sizeof(int)),"hipMalloc d_a");
    checkDeviceError(hipMalloc((void**)&d_b, N*sizeof(int)),"hipMalloc d_b");
    checkDeviceError(hipMalloc((void**)&d_c, N*sizeof(int)),"hipMalloc d_c");

    checkDeviceError(hipMemcpy(d_a, h_a, N*sizeof(int), hipMemcpyHostToDevice), "hipMemcpy h_a -> d_a");
    checkDeviceError(hipMemcpy(d_b, h_b, N*sizeof(int), hipMemcpyHostToDevice), "hipMemcpy h_b -> d_b");

    vector_add<<<(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

    checkDeviceError(hipGetLastError(), "vector_add launch");

    hipDeviceSynchronize();

    checkDeviceError(hipMemcpy(h_c, d_c, N*sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy d_c -> h_c");

    for(int i = 0; i < N; i++) {
        if(h_c[i] != h_a[i]+h_b[i]) {
            std::cout<<"FAILED !!!"<<std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::cout<<"PASSED !!!"<<std::endl;

    free(h_a);      free(h_b);      free(h_c);

    checkDeviceError(hipFree(d_a), "hipFree d_a");
    checkDeviceError(hipFree(d_b), "hipFree d_b");
    checkDeviceError(hipFree(d_c), "hipFree d_c");

    return 0;
}
