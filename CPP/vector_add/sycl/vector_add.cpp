#include <sycl/sycl.hpp>
#include <iostream>
#include <cstdlib>
#include <cassert>

#define N 4096

void init_array(int *a) {
    for(int i = 0; i < N; i++)
        a[i] = rand()%100;
}

int main() {

    sycl::queue queue(sycl::gpu_selector_v, sycl::property::queue::in_order());

    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;

    h_a = sycl::malloc_host<int>(N, queue);
    h_b = sycl::malloc_host<int>(N, queue);
    h_c = sycl::malloc_host<int>(N, queue);

    srand(1);

    init_array(h_a);
    init_array(h_b);

    d_a = sycl::malloc_device<int>(N, queue);
    d_b = sycl::malloc_device<int>(N, queue);
    d_c = sycl::malloc_device<int>(N, queue);

    queue.memcpy(h_a, d_a, N*sizeof(int));
    queue.memcpy(h_b, d_b, N*sizeof(int));
    queue.wait();

    queue.parallel_for(N, [=](auto i) { d_c[i] = d_a[i] + d_b[i]; });

    queue.memcpy(d_c, h_c, N*sizeof(int));
    queue.wait();

    for(int i = 0; i < N; i++) {
        if(h_c[i] != h_a[i]+h_b[i]) {
            std::cout<<"FAILED !!!"<<std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::cout<<"PASSED !!!"<<std::endl;

    sycl::free(d_a,queue);
    sycl::free(d_b,queue);
    sycl::free(d_c,queue);
    sycl::free(h_a,queue);
    sycl::free(h_b,queue);
    sycl::free(h_c,queue);

    return 0;
}
