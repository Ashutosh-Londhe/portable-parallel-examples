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
#include <sycl/sycl.hpp>

#define TILE_WIDTH 16
#define TILE_WIDTH 16

#define ar 311
#define ac_br 312
#define bc 115

using namespace std;

int main()
{
    sycl::queue queue(
                        sycl::gpu_selector_v,
                        sycl::property_list{
                            sycl::property::queue::in_order(),
                            sycl::property::queue::enable_profiling{}
                        }
                     );

    int *A, *B, *C1, *C2, rowA, colA, rowB, colB, rowC, colC;
    int *d_A, *d_B, *d_C;
    double elapsed_time;

    rowA = ar;      rowC = ar;
    colA = ac_br;   rowB = ac_br;
    colB = bc;      colC = bc;

    A = sycl::malloc_host<int>(rowA*colA,queue);
    B = sycl::malloc_host<int>(rowB*colB,queue);
    C1 = sycl::malloc_host<int>(rowC*colC,queue);
    C2 = sycl::malloc_host<int>(rowC*colC,queue);

    d_A = sycl::malloc_device<int>(rowA*colA,queue);
    d_B = sycl::malloc_device<int>(rowB*colB,queue);
    d_C = sycl::malloc_device<int>(rowC*colC,queue);

    srand(time(NULL));

    for(int i = 0; i < rowA*colA; i++)
        A[i] = rand()%5;

    for(int i = 0; i < rowB*colB; i++)
        B[i] = rand()%5;


    queue.memcpy(d_A, A, rowA*colA*sizeof(int));
    queue.memcpy(d_B, B, rowB*colB*sizeof(int));

    queue.fill(d_C, 0, rowC*colC);
    queue.wait();

    auto global_range = sycl::range<2>(
        ((rowA + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH,
        ((colB + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH);

    auto local_range = sycl::range<2>(TILE_WIDTH, TILE_WIDTH);

    auto e1 = queue.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>(global_range, local_range)
        , [=](sycl::nd_item<2> item) {
            int i = item.get_global_id()[0];
            int k = item.get_global_id()[1];

            if(i < rowA && k < colB) {
                for(int j = 0; j < rowB; j++)
                    d_C[i*colC + k] += d_A[i*colA + j] * d_B[j*colB + k];
            }
        });
    });

    e1.wait();
    cl_ulong start_time = e1.get_profiling_info<sycl::info::event_profiling::command_start>();
    cl_ulong end_time   = e1.get_profiling_info<sycl::info::event_profiling::command_end>();

    elapsed_time = (end_time - start_time) * 1e-6; // ns → ms
    cout<<"\n Matrix mulitplication(without shared memory): "<<elapsed_time<<"  mili-seconds";

    queue.memcpy(C1, d_C, rowC*colC*sizeof(int));
    queue.wait();

    queue.fill(d_C, 0, rowC*colC);
    queue.wait();
    
    // Shared memory implementation
    auto e2 = queue.submit([&](sycl::handler& h) {
        // local memory (shared memory equivalent)
        sycl::local_accessor<int, 2> s_A({TILE_WIDTH, TILE_WIDTH}, h);
        sycl::local_accessor<int, 2> s_B({TILE_WIDTH, TILE_WIDTH}, h);

        h.parallel_for(sycl::nd_range<2>(global_range, local_range)
            ,[=](sycl::nd_item<2> item) {
                int bx = item.get_group(0);
                int by = item.get_group(1);
                int tx = item.get_local_id(0);
                int ty = item.get_local_id(1);

                int row = bx * TILE_WIDTH + tx;
                int col = by * TILE_WIDTH + ty;

                int cvalue = 0;

                // Loop over tiles
                for (int i = 0; i < (colA + TILE_WIDTH - 1) / TILE_WIDTH; i++) {
                    // load tile from A
                    if (row < rowA && (i * TILE_WIDTH + ty) < colA)
                        s_A[tx][ty] = d_A[row * colA + i * TILE_WIDTH + ty];
                    else
                        s_A[tx][ty] = 0;

                    // load tile from B
                    if ((i * TILE_WIDTH + tx) < rowB && col < colB)
                        s_B[tx][ty] = d_B[(i * TILE_WIDTH + tx) * colB + col];
                    else
                        s_B[tx][ty] = 0;

                    // synchronize
                    item.barrier(sycl::access::fence_space::local_space);

                    // compute partial product
                    for (int k = 0; k < TILE_WIDTH; k++)
                        cvalue += s_A[tx][k] * s_B[k][ty];

                    // sync before next tile load
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // write result
                if (row < rowA && col < colB)
                    d_C[row * colC + col] = cvalue;
        });
    });

    e2.wait();
    start_time = e2.get_profiling_info<sycl::info::event_profiling::command_start>();
    end_time   = e2.get_profiling_info<sycl::info::event_profiling::command_end>();

    elapsed_time = (end_time - start_time) * 1e-6; // ns → ms
    cout<<"\n Matrix mulitplication(with shared memory)   : "<<elapsed_time<<"  mili-seconds";

    queue.memcpy(C2, d_C, rowC*colC*sizeof(int));

    for(int i = 0; i < rowC*colC; i++)
    {
        if(C1[i] != C2[i])
        {
            cerr<<"\n FAILED!!! incorrect result....";
            exit(-2);
        }
    }
    cout<<"\n PASSED !!! ...\n";

    sycl::free(A, queue);
    sycl::free(B, queue);
    sycl::free(C1, queue);
    sycl::free(C2, queue);

    sycl::free(d_A, queue);
    sycl::free(d_B, queue);
    sycl::free(d_C, queue);

    return 0;
}// End of main
