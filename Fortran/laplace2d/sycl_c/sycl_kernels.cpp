#include <sycl/sycl.hpp>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <math.h>
#include <string.h>
#include <float.h>


#define INFINITY_double DBL_MAX
#define MAX(a,b) ((a > b) ? (a) : (b))

int tblock_x=32, tblock_y=16;

sycl::queue *queue;

extern "C"
void device_init() {
    queue = new sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order());
}

extern "C"
void host_alloc(double **d_a, int N) {
    *d_a = sycl::malloc_host<double>(N, *queue);
}

extern "C"
void device_alloc(double **d_a, int N) {
//	printf("Allocating device memory: ");
    *d_a = sycl::malloc_device<double>(N, *queue);
//	printf("done....\n");
}

extern "C"
void device_memcpy_h2d(double **d_a, double *h_a, int N) {
    double *ptr = *d_a;
    queue->memcpy(ptr, h_a, (N)*sizeof(double));
    queue->wait();
}

extern "C"
void device_memcpy_d2h(double *h_a, double **d_a, int N) {
    double *ptr = *d_a;
    queue->memcpy(h_a, ptr, (N)*sizeof(double));
    queue->wait();
}

extern "C"
void device_boundary_x(double **d_A, int imax, int jmax) {
	double *ptr = *d_A;   // ptr now points to the 1-D array

    queue->parallel_for(imax+2, [ptr,imax](auto i)  { ptr[(0)*(imax+2)+i]   = 0.0; });
    queue->parallel_for(imax+2, [ptr,imax,jmax](auto i)  { ptr[(jmax+1)*(imax+2)+i] = 0.0; });
    queue->wait();
}

extern "C"
void device_boundary_y(double **d_A, int imax, int jmax) {
	
	double pi  = 2.0 * asin(1.0);
	double *ptr = *d_A;	

    queue->parallel_for(jmax+2, [ptr,imax,jmax,pi](auto j)  { ptr[(j)*(imax+2)+0] = sycl::sin(pi * j / (jmax+1)); });
    queue->parallel_for(jmax+2, [ptr,imax,jmax,pi](auto j)  { ptr[(j)*(imax+2)+imax+1] = sycl::sin(pi * j / (jmax+1))*sycl::exp(-pi); });
    queue->wait();
}

// Implemented from OPS-DSL sycl reduction helper functions
void sycl_reduction_max(double *d_reduction, double value, sycl::local_accessor<double, 1> temp, sycl::nd_item<2> &item_id, int group_size) {
  double value_tid;
  item_id.barrier(sycl::access::fence_space::local_space); /* important to finish all previous activity */

  size_t linear_id = item_id.get_local_linear_id();
  temp[linear_id] = value;

  item_id.barrier(sycl::access::fence_space::local_space); /* important to finish all previous activity */

  for (size_t d = group_size / 2; d > 0; d >>= 1) {
    if (linear_id < d) {
      value_tid = temp[linear_id + d];
      if(value_tid > value)
        value = value_tid;
      temp[linear_id] = value;
    }
    item_id.barrier(sycl::access::fence_space::local_space);
  }

  if (linear_id == 0) {
    if (value > d_reduction[0])
      d_reduction[0] = value;
  }
}

void jacobi_kernel(double *d_A, double *d_Anew, double *d_reduction_ptr, int imax, int jmax, int tblock_x, int tblock_y, int ngrid_x, int ngrid_y) {

  queue->submit([&] (sycl::handler& h) {
    sycl::local_accessor<double, 1> local_mem(sycl::range<1>(tblock_x * tblock_y), h);

    h.parallel_for(sycl::nd_range<2>(
                                      sycl::range<2>(ngrid_y*tblock_y,ngrid_x*tblock_x),
                                      sycl::range<2>(tblock_y,tblock_x)
                                     )
                  , [=](sycl::nd_item<2> item) [[intel::kernel_args_restrict]] {
      int j = item.get_global_id()[0];
      int i = item.get_global_id()[1];

      double error = -INFINITY_double;

      if (i >= 1 && i < imax+1 && j >= 1 && j < jmax+1) {

        d_Anew[(j)*(imax+2)+i] = 0.25 * ( d_A[(j)*(imax+2)+i+1] + d_A[(j)*(imax+2)+i-1]
                                        + d_A[(j-1)*(imax+2)+i] + d_A[(j+1)*(imax+2)+i]);
        error = sycl::fmax( error, sycl::fabs(d_Anew[(j)*(imax+2)+i] - d_A[(j)*(imax+2)+i]));
      }

      int group_size = item.get_local_range(0) * item.get_local_range(1);

      sycl_reduction_max(&d_reduction_ptr[item.get_group_linear_id()], error, local_mem, item, group_size);
    }); //h.parallel_for
  }); // end of queue->submit
}

extern "C"
void jacobi_host(double **d_A, double **d_Anew, double **host_reduction_ptr, double **device_reduction_ptr, int imax, int jmax, double *error) {

    int start_x = 1, end_x = imax+1;
    int start_y = 1, end_y = jmax+1;
    int tblock_x = 32, tblock_y = 16;
    int ngrid_x = (end_x-start_x-1)/tblock_x + 1;
    int ngrid_y = (end_y-start_y-1)/tblock_y + 1;
    int maxblocks = ngrid_x*ngrid_y;

    double *h_reduction_ptr = *host_reduction_ptr;
    double *d_reduction_ptr = *device_reduction_ptr;

    for(int b = 0; b < maxblocks; b++)
        h_reduction_ptr[b] = -INFINITY_double;

    queue->memcpy(d_reduction_ptr, h_reduction_ptr, (maxblocks)*sizeof(double));
    queue->wait();

    jacobi_kernel(*d_A, *d_Anew, d_reduction_ptr, imax, jmax, tblock_x, tblock_y, ngrid_x, ngrid_y);
    queue->wait();

    queue->memcpy(h_reduction_ptr, d_reduction_ptr, (maxblocks)*sizeof(double));
    queue->wait();

    double error_local = -INFINITY_double;
    for(int b = 0; b < maxblocks; b++)
        error_local = MAX(error_local, h_reduction_ptr[b]);
    *error = error_local;

}

extern "C"
void device_copy(double **d_A, double **d_Anew, int imax, int jmax) {
    int start_x = 1, end_x = imax+1;
    int start_y = 1, end_y = jmax+1;
    int tblock_x = 32, tblock_y = 16;
    int ngrid_x = (end_x-start_x-1)/tblock_x + 1;
    int ngrid_y = (end_y-start_y-1)/tblock_y + 1;

	double *ptr1 = *d_A;
	double *ptr2 = *d_Anew;

    queue->submit([&] (sycl::handler& h) {
    h.parallel_for(sycl::nd_range<2>(
                                      sycl::range<2>(ngrid_y*tblock_y,ngrid_x*tblock_x),
                                      sycl::range<2>(tblock_y,tblock_x)
                                     )
                  , [=](sycl::nd_item<2> item) {
      int j = item.get_global_id()[0];
      int i = item.get_global_id()[1];

      if (i >= 1 && i < imax+1 && j >= 1 && j < jmax+1) {
         ptr1[(j)*(imax+2)+i] = ptr2[(j)*(imax+2)+i];
      }
    }); //h.parallel_for
  }); //queue->submit
}

extern "C"
void device_free(double *ptr) {
    sycl::free(ptr, *queue);
}

