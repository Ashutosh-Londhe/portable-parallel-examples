#include <iostream>
#include <cstdlib>
#include <cassert>
#include <math.h>
#include <string.h>
#include <float.h>
#include <hip/hip_runtime.h>

#define pi  2.0 * asin(1.0)

#define INFINITY_double DBL_MAX
#define MAX(a,b) ((a > b) ? (a) : (b))

dim3 tblock(32,16,1);

int nthreads = tblock.x * tblock.y * tblock.z;

// size of reduction
int nshared = nthreads * sizeof(double);

inline void checkDeviceError(hipError_t result, const char *msg) {
    if(result != hipSuccess) {
        std::cerr<<"HIP Error: "<<msg<<" : "<<hipGetErrorString(result)<<std::endl;
        assert(result == hipSuccess);
    }
}

extern "C"
void device_alloc(double **d_a, int N) {
    checkDeviceError(hipMalloc((void**)d_a, N*sizeof(double)),"hipMalloc");
}

extern "C"
void device_memcpy_h2d(double **d_a, double *h_a, int N) {
	/* for(int i = 0; i < 10; i++)
		std::cout<<h_a[i]<<" ";
	std::cout<<"\n"; */
    checkDeviceError(hipMemcpy(*d_a, h_a, N*sizeof(double), hipMemcpyHostToDevice), "hipMemcpy Host-to-Device");
}

extern "C"
void device_memcpy_d2h(double *h_a, double **d_a, int N) {
    checkDeviceError(hipMemcpy(h_a, *d_a, N*sizeof(double), hipMemcpyDeviceToHost), "hipMemcpy Device-to-Host");
}

__global__ void boundary_x(double *A, int imax, int jmax) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j == 0 && i < imax+2)
        A[(0)*(imax+2)+i]   = 0.0;
    if(j == (jmax+1) && i < imax+2)
        A[(jmax+1)*(imax+2)+i] = 0.0;
}

__global__ void boundary_y(double *A, int imax, int jmax) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j < jmax+2 && i == 0)
         A[(j)*(imax+2)+0] = sin(pi * j / (jmax+1));
    if(j < jmax+2 && i == imax+1)
        A[(j)*(imax+2)+imax+1] = sin(pi * j / (jmax+1))*exp(-pi);
}

extern "C"
void device_boundary_x(double **d_A, int imax, int jmax) {
    dim3 grid((jmax+2+tblock.x-1)/tblock.x, (imax+2+tblock.y-1)/tblock.y, 1);
    boundary_x<<<grid,tblock>>>(*d_A, imax, jmax);
    checkDeviceError(hipDeviceSynchronize(), "boundary kernel launch");
}

extern "C"
void device_boundary_y(double **d_A, int imax, int jmax) {
    dim3 grid((jmax+2+tblock.x-1)/tblock.x, (imax+2+tblock.y-1)/tblock.y, 1);
    boundary_y<<<grid,tblock>>>(*d_A, imax, jmax);
    checkDeviceError(hipDeviceSynchronize(), "boundary kernel launch");
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

__global__ void jacobi_device(double *A, double *Anew, double *d_reduction_ptr, int imax, int jmax) {

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

extern "C"
void jacobi_host(double **d_A, double **d_Anew, int imax, int jmax, double *error) {
    dim3 grid((jmax+2+tblock.x-1)/tblock.x, (imax+2+tblock.y-1)/tblock.y, 1);
    int maxblocks = grid.x * grid.y * grid.z;

    double *h_reduction_ptr = (double *)malloc(maxblocks * sizeof(double));
    double *d_reduction_ptr;
    checkDeviceError(hipMalloc((void**)&d_reduction_ptr, maxblocks * sizeof(double)), "hipMalloc d_reduction");
    for(int b = 0; b < maxblocks; b++)
        h_reduction_ptr[b] = -INFINITY_double;

    checkDeviceError(hipMemcpy(d_reduction_ptr, h_reduction_ptr, maxblocks*sizeof(double), hipMemcpyHostToDevice), "reduction host -> device");

    jacobi_device<<<grid,tblock,nshared>>>(*d_A,*d_Anew,d_reduction_ptr,imax,jmax);
    checkDeviceError(hipDeviceSynchronize(), "Jacobi kernel launch");

    checkDeviceError(hipMemcpy(h_reduction_ptr, d_reduction_ptr, maxblocks*sizeof(double), hipMemcpyDeviceToHost), "reduction device -> host");

    double error_local = -INFINITY_double;
    for(int b = 0; b < maxblocks; b++)
        error_local = MAX(error_local, h_reduction_ptr[b]);
    //std::cout<<"error: "<<error<<std::endl;
    *error = error_local;
}

__global__ void copy(double *A, double *Anew, int imax, int jmax) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j > 0 && j < jmax+1 && i > 0 && i < imax+1)
        A[(j)*(imax+2)+i] = Anew[(j)*(imax+2)+i];
}

extern "C"
void device_copy(double **d_A, double **d_Anew, int imax, int jmax) {
    dim3 grid((jmax+2+tblock.x-1)/tblock.x, (imax+2+tblock.y-1)/tblock.y, 1);
    copy<<<grid,tblock>>>(*d_A, *d_Anew, imax, jmax);
    checkDeviceError(hipDeviceSynchronize(), "copy kernel launch");
}

extern "C"
void device_free(double *ptr) {
    checkDeviceError(hipFree(ptr), "hipFree");
}
