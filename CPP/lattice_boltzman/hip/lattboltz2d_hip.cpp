#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <hip/hip_runtime.h>

#include "timer.h"

#define NX 128
#define NY 128

__device__ int mod(int v, int m) {
    int val = v%m;
    if (val<0) val = m+val;
    return val;
}

inline void checkDeviceError(hipError_t result, char *msg) {
    if(result != hipSuccess) {
        fprintf(stdout, "HIP Error: %s : %s\n",msg,hipGetErrorString(result));
        assert(result == hipSuccess);
    }
}

__global__ void equationA(int *SOLID) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j < NY && i < NX) {
        if( (i>=2 && i<=65 && j>=4 && j<=29) || (i>=45 && i<=123 && j>=41 && j<=65) || (i>=30 && i<=101 && j>=91 && j<=115))
            SOLID[j*NX+i] = 0;
        else
            SOLID[j*NX+i] = 1;
    }
}

__global__ void equationB(double *N, double rho0, double *W) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j < NY && i < NX) {
        for (int f = 0; f < 9; f++) {
            N[(j*NX+i)*9 + f] = rho0 * W[f];
        }
    }
}

__global__ void equationC(double *workArray, double *N) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j < NY && i < NX) {
        for (int f = 0; f < 9; f++) {
            workArray[(j*NX+i)*9 + f] = N[(j*NX+i)*9 + f];
        }
    }
}

__global__ void equationD(double *N, double *workArray, int *cx, int *cy) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j < NY && i < NX) {
        for (int f = 1; f < 9; f++) {
            N[(j*NX+i)*9 + f] = workArray[(mod(j-cy[f],NY)*NX+mod(i-cx[f],NX))*9 + f];
        }
    }
}

__global__ void equationE(double *N_SOLID, double *N, int *SOLID, int *opposite) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j < NY && i < NX) {
        if (SOLID[j*NX+i]==1) {
            for (int f = 0; f < 9; f++) {
                N_SOLID[(j*NX+i)*9 + opposite[f]] = N[(j*NX+i)*9 + f];
            }
        }
    }
}

__global__ void equationF(double *rho, double *N) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j < NY && i < NX) {
        rho[j*NX+i] = 0.0;
        for (int f = 0; f < 9; f++) {
            rho[j*NX+i] += N[(j*NX+i)*9 + f];
        }
    }
}

__global__ void equationG(double *ux, double *N, double *rho, int *cx, double deltaUX) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j < NY && i < NX) {
        ux[j*NX+i] = 0.0;
        for (int f = 0; f < 9; f++) {
            ux[j*NX+i] += N[(j*NX+i)*9 + f] * cx[f];
        }
        ux[j*NX+i] = ux[j*NX+i] / rho[j*NX+i] + deltaUX;
    }
}

__global__ void equationH(double *uy, double *N, double *rho, int *cy) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j < NY && i < NX) {
        uy[j*NX+i] = 0.0;
        for (int f = 0; f < 9; f++) {
            uy[j*NX+i] += N[(j*NX+i)*9 + f] * cy[f];
        }
        uy[j*NX+i] = uy[j*NX+i] / rho[j*NX+i];
    }
}

__global__ void equationI(double* workArray, double *uy, double *ux, int *cx, int *cy) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j < NY && i < NX) {
        for (int f = 0; f < 9; f++) {
            workArray[(j*NX+i)*9 + f] = ux[j*NX+i]*cx[f] + uy[j*NX+i]*cy[f];
        }
    }
}

__global__ void equationJ(double* workArray) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j < NY && i < NX) {
        for (int f = 0; f < 9; f++) {
            workArray[(j*NX+i)*9 + f] = (3.0+4.5*workArray[(j*NX+i)*9 + f])*workArray[(j*NX+i)*9 + f];
        }
    }
}

__global__ void equationK(double* workArray, double *ux, double *uy) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j < NY && i < NX) {
        for (int f = 0; f < 9; f++) {
            workArray[(j*NX+i)*9 + f] = workArray[(j*NX+i)*9 + f] - 1.5 * (ux[j*NX+i]*ux[j*NX+i] + uy[j*NX+i]*uy[j*NX+i]);
        }
    }
}

__global__ void equationL(double* workArray, double *rho, double *W) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j < NY && i < NX) {
        for (int f = 0; f < 9; f++) {
            workArray[(j*NX+i)*9 + f] = (1.0+workArray[(j*NX+i)*9 + f]) * W[f] * rho[j*NX+i];
        }
    }
}

__global__ void equationM(double *N, double *workArray, double OMEGA) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j < NY && i < NX) {
        for (int f = 0; f < 9; f++) {
            N[(j*NX+i)*9 + f] += (workArray[(j*NX+i)*9 + f] - N[(j*NX+i)*9 + f]) * OMEGA;
        }
    }
}

__global__ void equationN(double *N, double *N_SOLID, int *SOLID) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if(j < NY && i < NX) {
        if (SOLID[j*NX+i]==1) {
            for (int f = 0; f < 9; f++) {
                N[(j*NX+i)*9 + f] = N_SOLID[(j*NX+i)*9 + f];
            }
        }
    }
}

// Implemented from OPS-DSL hip reduction helper functions
__device__ void hip_reduction_inc(double *d_reduction, double value) {
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
        value = value + value_tid;
        temp[tid] = value;
    }

    // second, do reductions involving more than one warp

    for (d >>= 1; d > warpSize; d >>= 1) {
        __syncthreads();

        if (tid < d) {
            value_tid = temp[tid + d];
            value = value + value_tid;
            temp[tid] = value;
        }
    }

    // third, do reductions involving just one warp

    __syncthreads();

    if (tid < warpSize) {
        for (; d > 0; d >>= 1) {
            if (tid < d) {
                value_tid = temp[tid + d];
                value = value + value_tid;
                temp[tid] = value;
            }
        }

        // finally, update global reduction variable

        if (tid == 0) {
            *d_reduction = *d_reduction + value;
        }
    }
}

__global__ void equationP(double *ux, double *uy, double *d_reduction_ptr) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    double local_energy = 0.0;

    if(j < NY && i < NX) {
        local_energy += ux[j*NX+i]*ux[j*NX+i]+uy[j*NX+i]*uy[j*NX+i]; // reduction
    }

    hip_reduction_inc(&d_reduction_ptr[blockIdx.x + blockIdx.y*gridDim.x], local_energy);
}

void cal_energy_host(double *d_ux, double *d_uy, double *energy, double *h_reduction_ptr, double *d_reduction_ptr, int nshared, int maxblocks, dim3 tblock, dim3 grid) {
    for(int b = 0; b < maxblocks; b++)
        h_reduction_ptr[b] = 0.0;

    checkDeviceError(hipMemcpy(d_reduction_ptr, h_reduction_ptr, maxblocks*sizeof(double), hipMemcpyHostToDevice), "reduction host -> device");

    equationP<<<grid,tblock,nshared>>>(d_ux, d_uy, d_reduction_ptr);

    checkDeviceError(hipMemcpy(h_reduction_ptr, d_reduction_ptr, maxblocks*sizeof(double), hipMemcpyDeviceToHost), "reduction device -> host");

    for(int b = 0; b < maxblocks; b++)
        *energy = *energy + h_reduction_ptr[b];
}

int main(int argc, char ** argv) {
    const double OMEGA = 1.0;
    const double rho0 = 1.0;
    const double deltaUX=10e-6;

    const double W[] = {4.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/9.0, 1.0/36.0, 1.0/9.0, 1.0/36.0, 1.0/9.0, 1.0/36.0};
    const int cx[] = {0,0,1,1, 1, 0,-1,-1,-1};
    const int cy[] = {0,1,1,0,-1,-1,-1, 0, 1};

    const int opposite[] = {0,5,6,7,8,1,2,3,4};
    
    double energy;

    double ct0,et0,ct1,et1; //timer variables

    dim3 tblock(32,16,1);
    dim3 grid((NY+tblock.x-1)/tblock.x, (NX+tblock.y-1)/tblock.y, 1);

    int nthreads = tblock.x * tblock.y * tblock.z;
    int maxblocks = grid.x * grid.y * grid.z;

    // size of reduction
    int nshared = nthreads * sizeof(double);
    double *h_reduction_ptr = (double *)malloc(maxblocks * sizeof(double));
    double *d_reduction_ptr;
    checkDeviceError(hipMalloc((void**)&d_reduction_ptr, maxblocks * sizeof(double)), "hipMalloc d_reduction");

    double *d_W;
    int *d_cx, *d_cy, *d_opposite;
    checkDeviceError(hipMalloc((void**)&d_W, (9)*sizeof(double)), "hipMalloc d_W");
    checkDeviceError(hipMalloc((void**)&d_cx, (9)*sizeof(int)), "hipMalloc d_cx");
    checkDeviceError(hipMalloc((void**)&d_cy, (9)*sizeof(int)), "hipMalloc d_cy");
    checkDeviceError(hipMalloc((void**)&d_opposite, (9)*sizeof(int)), "hipMalloc d_opposite");

    checkDeviceError(hipMemcpy(d_W, W, (9) * sizeof(double), hipMemcpyHostToDevice), "hipMemcpy h2d");
    checkDeviceError(hipMemcpy(d_cx, cx, (9) * sizeof(int), hipMemcpyHostToDevice), "hipMemcpy h2d");
    checkDeviceError(hipMemcpy(d_cy, cy, (9) * sizeof(int), hipMemcpyHostToDevice), "hipMemcpy h2d");
    checkDeviceError(hipMemcpy(d_opposite, opposite, (9) * sizeof(int), hipMemcpyHostToDevice), "hipMemcpy h2d");

    // Generate obstacles based on grid positions
    int * __restrict__ SOLID = new int[NX*NY];
    int * d_SOLID;
    checkDeviceError(hipMalloc((void**)&d_SOLID, (NX*NY) * sizeof(int)), "hipMalloc d_SOLID");

    equationA<<<grid,tblock>>>(d_SOLID);
    checkDeviceError(hipDeviceSynchronize(), "equationA kernel launch");

    //Initial values
    double * __restrict__ N = new double[NX*NY*9];
    double *d_N;
    checkDeviceError(hipMalloc((void**)&d_N, (NX*NY*9)*sizeof(double)), "hipMalloc d_N");

    equationB<<<grid,tblock>>>(d_N, rho0, d_W);
    checkDeviceError(hipDeviceSynchronize(), "equationB kernel launch");

    //Work arrays
    double * __restrict__ workArray = new double[NX*NY*9];
    double * __restrict__ N_SOLID = new double[NX*NY*9];
    double * __restrict__ rho = new double[NX*NY];
    double * __restrict__ ux = new double[NX*NY];
    double * __restrict__ uy = new double[NX*NY];

    double *d_workArray, *d_N_SOLID, *d_rho, *d_ux, *d_uy;

    checkDeviceError(hipMalloc((void**)&d_workArray, (NX*NY*9)*sizeof(double)), "hipMalloc d_workArray");
    checkDeviceError(hipMalloc((void**)&d_N_SOLID, (NX*NY*9)*sizeof(double)), "hipMalloc d_N_SOLID");

    checkDeviceError(hipMalloc((void**)&d_rho, (NX*NY)*sizeof(double)), "hipMalloc d_rho");
    checkDeviceError(hipMalloc((void**)&d_ux, (NX*NY)*sizeof(double)), "hipMalloc d_ux");
    checkDeviceError(hipMalloc((void**)&d_uy, (NX*NY)*sizeof(double)), "hipMalloc d_uy");

    //Start timer
    timer(&ct0, &et0);

    //Main time loop
    for (int t = 0; t < 4000; t++) {

        //Backup values
        equationC<<<grid,tblock>>>(d_workArray, d_N);

        // Gather neighbour values
        equationD<<<grid,tblock>>>(d_N, d_workArray, d_cx, d_cy);

        //Bounce back from solids, no collision
        equationE<<<grid,tblock>>>(d_N_SOLID, d_N, d_SOLID, d_opposite);

        equationF<<<grid,tblock>>>(d_rho, d_N);

        equationG<<<grid,tblock>>>(d_ux, d_N, d_rho, d_cx, deltaUX);

        equationH<<<grid,tblock>>>(d_uy, d_N, d_rho, d_cy);

        equationI<<<grid,tblock>>>(d_workArray, d_uy, d_ux, d_cx, d_cy);

        equationJ<<<grid,tblock>>>(d_workArray);

        equationK<<<grid,tblock>>>(d_workArray, d_ux, d_uy);

        equationL<<<grid,tblock>>>(d_workArray, d_rho, d_W);

        equationM<<<grid,tblock>>>(d_N, d_workArray, OMEGA);

        equationN<<<grid,tblock>>>(d_N, d_N_SOLID, d_SOLID);
        checkDeviceError(hipDeviceSynchronize(), "equationN kernel launch");

        //Calculate kinetic energy
        energy = 0.0;
        cal_energy_host(d_ux, d_uy, &energy, h_reduction_ptr, d_reduction_ptr, nshared, maxblocks, tblock, grid);

        if (t%100==0) 
            printf(" %d  %10.5e \n", t, energy);            
        if (t==3999 && NX == 128 && NY == 128) {
          double diff = fabs(((energy - 0.0000111849)/0.0000111849));
          if (diff < 0.00001) {
            printf("Energy : %10.5e diff: %10.5e %s\n", energy, diff, "Test PASSED");
          } else {
            printf("Energy : %10.5e diff:  %10.5e %s\n", energy, diff, "Test FAILED");
          }
        }
    } // End of main time loop

    //End timer
    timer(&ct1, &et1);
    printf("\nTotal Wall time %lf seconds\n",et1-et0);
    
    checkDeviceError(hipMemcpy(SOLID, d_SOLID, (NX*NY) * sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy d2h");
    checkDeviceError(hipMemcpy(ux, d_ux, (NX*NY) * sizeof(double), hipMemcpyDeviceToHost), "hipMemcpy d2h");
    checkDeviceError(hipMemcpy(uy, d_uy, (NX*NY) * sizeof(double), hipMemcpyDeviceToHost), "hipMemcpy d2h");

    if (true) {
        std::ofstream myfile;
        myfile.open ("output_velocity.txt");
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                myfile << SOLID[j*NX+i] << " " << ux[j*NX+i] << " " << uy[j*NX+i] << std::endl;
            }
        }
        myfile.close();
    }

    checkDeviceError(hipFree(d_W), "hipFree d_W");
    checkDeviceError(hipFree(d_cx), "hipFree d_cx");
    checkDeviceError(hipFree(d_cy), "hipFree d_cy");

    checkDeviceError(hipFree(d_SOLID), "hipFree d_SOLID");
    delete[] SOLID;      SOLID = nullptr;
    checkDeviceError(hipFree(d_N), "hipFree d_N");
    delete[] N;          N = nullptr;
    checkDeviceError(hipFree(d_workArray), "hipFree d_workArray");
    delete[] workArray;  workArray = nullptr;
    checkDeviceError(hipFree(d_N_SOLID), "hipFree d_N_SOLID");
    delete[] N_SOLID;    N_SOLID = nullptr;
    checkDeviceError(hipFree(d_rho), "hipFree d_rho");
    delete[] rho;        rho = nullptr;
    checkDeviceError(hipFree(d_ux), "hipFree d_ux");
    delete[] ux;         ux = nullptr;
    checkDeviceError(hipFree(d_uy), "hipFree d_uy");
    delete[] uy;         uy = nullptr;

}// End of main function

