/*
                    column
    A[][] = ---------------------threadIdx.y
            |
            |
            |
            |
   row      |
            |
            |
            |
            |
        threadIdx.x
*/


#include <iostream>
#include <cstdlib>
#include <string.h>
#include <assert.h>
#include <hip/hip_runtime.h>

#define TILE_WIDTH 16
#define TILE_WIDTH 16

#define ar 311
#define ac_br 312
#define bc 115

using namespace std;

inline void checkDeviceError(hipError_t result, char *msg) {
    if(result != hipSuccess) {
        fprintf(stdout, "HIP Error: %s : %s\n",msg,hipGetErrorString(result));
        assert(result == hipSuccess);
    }
}

__global__ void mat_mul(int *d_A, int *d_B, int *d_C, int rowA, int colA, int rowB, int colB, int rowC, int colC)
{
    int row, col;
    row = threadIdx.x + blockIdx.x*blockDim.x;      // 0 to rowA or rowC
    col = threadIdx.y + blockIdx.y*blockDim.y;      // 0 to colB or colC

    if(row < rowC && col < colC)
    {
        for(int i = 0; i < colA; i++)               // colA = rowB
            d_C[row*colC + col] += d_A[row*colA + i]*d_B[i*colB + col];
    }
}// End of mat_mul function

__global__ void mat_mul_tiled(int *d_A, int *d_B, int *d_C, int rowA, int colA, int rowB, int colB, int rowC, int colC)
{
    int bx = blockIdx.x,     by = blockIdx.y;
    int tx = threadIdx.x,    ty = threadIdx.y;
    int row = tx + bx*TILE_WIDTH;      // 0 to rowA or rowC
    int col = ty + by*TILE_WIDTH;      // 0 to colB or colC

    __shared__ int s_A[TILE_WIDTH][TILE_WIDTH], s_B[TILE_WIDTH][TILE_WIDTH];
    int cvalue = 0;

    for(int i = 0; i < (colA+TILE_WIDTH-1)/TILE_WIDTH; i++)
    {
        if(row < rowA && i*TILE_WIDTH+ty < colA)
            s_A[tx][ty] = d_A[row*colA + i*TILE_WIDTH+ty];
        else
            s_A[tx][ty] = 0;

        if(i*TILE_WIDTH+tx < rowB && col < colB)
            s_B[tx][ty] = d_B[(i*TILE_WIDTH+tx)*colB + col];
        else
            s_B[tx][ty] = 0;

        __syncthreads();    // making sure all threads has populated the shared memory tile

        for(int k = 0; k < TILE_WIDTH; k++)
            cvalue += s_A[tx][k]*s_B[k][ty];

        __syncthreads();
    }

    if(row < rowC && col < colC)
        d_C[row*colC + col] = cvalue;

}// End of mat_mul_tiled function

int main()
{
    int *A, *B, *C1, *C2, rowA, colA, rowB, colB, rowC, colC;
    int *d_A, *d_B, *d_C;
    dim3 dimg, dimb;
    hipEvent_t start, stop;
    float elapsed_time;

    rowA = ar;      rowC = ar;
    colA = ac_br;   rowB = ac_br;
    colB = bc;      colC = bc;

    A = new int[rowA*colA];     B  = new int[rowB*colB];
    C1 = new int[rowC*colC];    C2 = new int[rowC*colC];

    checkDeviceError(hipMalloc((void**)&d_A, rowA*colA*sizeof(int)), "hipMalloc d_A");
    checkDeviceError(hipMalloc((void**)&d_B, rowB*colB*sizeof(int)), "hipMalloc d_B");
    checkDeviceError(hipMalloc((void**)&d_C, rowC*colC*sizeof(int)), "hipMalloc d_C");

    srand(time(NULL));

    for(int i = 0; i < rowA*colA; i++)
        A[i] = rand()%5;

    for(int i = 0; i < rowB*colB; i++)
        B[i] = rand()%5;

    dimg = dim3((rowC+TILE_WIDTH-1)/TILE_WIDTH, (colC+TILE_WIDTH-1)/TILE_WIDTH);
    dimb = dim3(TILE_WIDTH, TILE_WIDTH);

    hipEventCreate(&start);        hipEventCreate(&stop);

    checkDeviceError(hipMemcpy(d_A, A, rowA*colA*sizeof(int), hipMemcpyHostToDevice), "hipMemcpu Host-> device");
    checkDeviceError(hipMemcpy(d_B, B, rowB*colB*sizeof(int), hipMemcpyHostToDevice), "hipMemcpu Host-> device");
	
	// Without shared memory
    hipMemset(d_C, 0, rowC*colC*sizeof(int));

    hipEventRecord(start, 0);
    mat_mul<<<dimg, dimb>>>(d_A, d_B, d_C, rowA, colA, rowB, colB, rowC, colC);
    hipEventRecord(stop, 0);           hipEventSynchronize(stop);

    checkDeviceError(hipMemcpy(C1, d_C, rowC*colC*sizeof(int), hipMemcpyDeviceToHost), "hipMemcpu device -> Host");

    hipEventElapsedTime(&elapsed_time, start, stop);
    cout<<"\n Matrix mulitplication(without shared memory): "<<elapsed_time<<"  mili-seconds";

    // With Shared memory
    hipMemset(d_C, 0, rowC*colC*sizeof(int));

    hipEventRecord(start, 0);
    mat_mul_tiled<<<dimg, dimb>>>(d_A, d_B, d_C, rowA, colA, rowB, colB, rowC, colC);
    hipEventRecord(stop, 0);           hipEventSynchronize(stop);

    checkDeviceError(hipMemcpy(C2, d_C, rowC*colC*sizeof(int), hipMemcpyDeviceToHost), "hipMemcpu device -> Host");

    hipEventElapsedTime(&elapsed_time, start, stop);
    cout<<"\n Matrix mulitplication(with shared memory)   : "<<elapsed_time<<"  mili-seconds";

    for(int i = 0; i < rowC*colC; i++)
    {
        if(C1[i] != C2[i])
        {
            cerr<<"\n FAILED!!! incorrect result....";
            exit(-2);
        }
    }
    cout<<"\n PASSED !!! ...\n";

    delete(A);      delete(B);
    delete(C1);      delete(C2);

    checkDeviceError(hipFree(d_A), "hipFree d_A");
    checkDeviceError(hipFree(d_B), "hipFree d_B");
    checkDeviceError(hipFree(d_C), "hipFree d_C");

    return 0;
}// End of main
