#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <hip/hip_runtime.h>

#define jmax 4094
#define imax 4094
#define pi  2.0 * asin(1.0)

#define INFINITY_double DBL_MAX
#define MAX(a,b) ((a > b) ? (a) : (b))

inline void checkDeviceError(hipError_t result, char *msg) {
    if(result != hipSuccess) {
        fprintf(stdout, "CUDA Error: %s : %s\n",msg,hipGetErrorString(result));
        assert(result == hipSuccess);
    }
}

__global__ void boundary_x(double *A) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j == 0 && i < imax+2)
        A[(0)*(imax+2)+i]   = 0.0;
    if(j == (jmax+1) && i < imax+2)
        A[(jmax+1)*(imax+2)+i] = 0.0;
}

__global__ void boundary_y(double *A) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j < jmax+2 && i == 0)
         A[(j)*(imax+2)+0] = sin(pi * j / (jmax+1));
    if(j < jmax+2 && i == imax+1)
        A[(j)*(imax+2)+imax+1] = sin(pi * j / (jmax+1))*exp(-pi);
}

// Implemented from OPS-DSL hip reduction helper functions
__device__ void hip_reduction_max(double *d_reduction, double value) {
    // allocated based on shared memory size passed during kernel launch
    extern __shared__ volatile double temp[];
    double value_tid;

    __syncthreads(); /* important to finish all previous activity */

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    temp[tid] = value;

    // first, cope with blockDim.x perhaps not being a power of 2

    __syncthreads();

    int d = 1 << (31 - __clz(((int)(blockDim.x * blockDim.y * blockDim.z) - 1)));
    // d = blockDim.x/2 rounded up to nearest power of 2

    if (tid + d < blockDim.x * blockDim.y * blockDim.z) {
        value_tid = temp[tid + d];
        if(value_tid > value)
            value = value_tid;
        temp[tid] = value;
    }

    // second, do reductions involving more than one warp

    for (d >>= 1; d > warpSize; d >>= 1) {
        __syncthreads();

        if (tid < d) {
            value_tid = temp[tid + d];
            if(value_tid > value)
                value = value_tid;
            temp[tid] = value;
        }
    }

    // third, do reductions involving just one warp

    __syncthreads();

    if (tid < warpSize) {
        for (; d > 0; d >>= 1) {
            if (tid < d) {
                value_tid = temp[tid + d];
                if(value_tid > value)
                    value = value_tid;
                temp[tid] = value;
            }
        }

        // finally, update global reduction variable

        if (tid == 0) {
            if(value > *d_reduction)
                *d_reduction = value;
        }
    }
}

__global__ void jacobi_device(double *A, double *Anew, double *d_reduction_ptr) {

    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    double local_error = -INFINITY_double;

    if(j > 0 && j < jmax+1 && i > 0 && i < imax+1) {
        Anew[(j)*(imax+2)+i] = 0.25f * ( A[(j)*(imax+2)+i+1] + A[(j)*(imax+2)+i-1]
                                       + A[(j-1)*(imax+2)+i] + A[(j+1)*(imax+2)+i]);

        local_error = MAX(local_error, fabs(Anew[(j)*(imax+2)+i] - A[(j)*(imax+2)+i]));
    }

    hip_reduction_max(&d_reduction_ptr[blockIdx.x + blockIdx.y*gridDim.x], local_error);
}

void jacobi_host(double *d_A, double *d_Anew, double *error, double *h_reduction_ptr, double *d_reduction_ptr, int nshared, int maxblocks, dim3 tblock, dim3 grid) {

    for(int b = 0; b < maxblocks; b++)
        h_reduction_ptr[b] = -INFINITY_double;

    checkDeviceError(hipMemcpy(d_reduction_ptr, h_reduction_ptr, maxblocks*sizeof(double), hipMemcpyHostToDevice), "reduction host -> device");

    jacobi_device<<<grid,tblock,nshared>>>(d_A,d_Anew,d_reduction_ptr);

    checkDeviceError(hipMemcpy(h_reduction_ptr, d_reduction_ptr, maxblocks*sizeof(double), hipMemcpyDeviceToHost), "reduction device -> host");

    for(int b = 0; b < maxblocks; b++)
        *error = MAX(*error, h_reduction_ptr[b]);
}

__global__ void copy(double *A, double *Anew) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j > 0 && j < jmax+1 && i > 0 && i < imax+1)
        A[(j)*(imax+2)+i] = Anew[(j)*(imax+2)+i];
}

int main(int argc, const char** argv)
{
    int iter_max = 100;

    const double tol = 1.0e-6;
    double error     = 1.0;

    double *d_A, *d_Anew;

    checkDeviceError(hipMalloc((void**)&d_A, (imax+2) * (jmax+2) * sizeof(double)), "hipMalloc d_A");
    checkDeviceError(hipMalloc((void**)&d_Anew, (imax+2) * (jmax+2) * sizeof(double)), "hipMalloc d_Anew");

    checkDeviceError(hipMemset(d_A, 0, (imax+2) * (jmax+2) * sizeof(double)), "hipMemset d_A");
    checkDeviceError(hipMemset(d_Anew, 0, (imax+2) * (jmax+2) * sizeof(double)), "hipMemset d_Anew");

    dim3 tblock(32,16,1);
    dim3 grid((jmax+2+tblock.x-1)/tblock.x, (imax+2+tblock.y-1)/tblock.y, 1);

    int nthreads = tblock.x * tblock.y * tblock.z;
    int maxblocks = grid.x * grid.y * grid.z;

    // size of reduction
    int nshared = nthreads * sizeof(double);
    double *h_reduction_ptr = (double *)malloc(maxblocks * sizeof(double));
    double *d_reduction_ptr;
    checkDeviceError(hipMalloc((void**)&d_reduction_ptr, maxblocks * sizeof(double)), "hipMalloc d_reduction");

    // set boundary conditions
    boundary_x<<<grid,tblock>>>(d_A);
    boundary_y<<<grid,tblock>>>(d_A);
    checkDeviceError(hipDeviceSynchronize(), "boundary kernel launch");

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", imax+2, jmax+2);

    boundary_x<<<grid,tblock>>>(d_Anew);
    boundary_y<<<grid,tblock>>>(d_Anew);
    checkDeviceError(hipDeviceSynchronize(), "boundary_x kernel launch");

    int iter = 0;
    while ( error > tol && iter < iter_max )
    {
        error = -INFINITY_double;

        jacobi_host(d_A, d_Anew, &error, h_reduction_ptr, d_reduction_ptr, nshared, maxblocks, tblock, grid);

        copy<<<grid,tblock>>>(d_A, d_Anew);

        if(iter % 10 == 0) printf("%5d, %0.6f\n", iter, error);        
        iter++;
    }

    printf("%5d, %0.6f\n", iter, error);

    double err_diff = fabs((100.0*(error/2.421354960840227e-03))-100.0);
    printf("Total error is within %3.15E %% of the expected error\n",err_diff);
    if(err_diff < 0.001)
        printf("This run is considered PASSED\n");
    else
        printf("This test is considered FAILED\n");

    free(h_reduction_ptr);

    checkDeviceError(hipFree(d_A), "hipFree d_A");
    checkDeviceError(hipFree(d_Anew), "hipFree d_Anew");
    checkDeviceError(hipFree(d_reduction_ptr), "hipFree d_reduction_ptr");
  
    return 0;
}

