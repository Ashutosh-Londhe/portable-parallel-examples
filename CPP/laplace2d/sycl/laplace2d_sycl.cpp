#include <sycl/sycl.hpp>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#define INFINITY_double DBL_MAX
#define MAX(a,b) ((a > b) ? (a) : (b))

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

void jacobi_kernel(sycl::queue& queue, double *d_A, double *d_Anew, double *d_reduction_ptr, int imax, int jmax, int tblock_x, int tblock_y, int ngrid_x, int ngrid_y) {

  queue.submit([&] (sycl::handler& h) {
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
  }); // end of queue.submit
}

void copy(sycl::queue& queue, double *d_A, double *d_Anew, int imax, int jmax, int tblock_x, int tblock_y, int ngrid_x, int ngrid_y) {

  queue.submit([&] (sycl::handler& h) {
    h.parallel_for(sycl::nd_range<2>(
                                      sycl::range<2>(ngrid_y*tblock_y,ngrid_x*tblock_x),
                                      sycl::range<2>(tblock_y,tblock_x)
                                     )
                  , [=](sycl::nd_item<2> item) {
      int j = item.get_global_id()[0];
      int i = item.get_global_id()[1];

      if (i >= 1 && i < imax+1 && j >= 1 && j < jmax+1) {
         d_A[(j)*(imax+2)+i] = d_Anew[(j)*(imax+2)+i];        
      }
    }); //h.parallel_for
  }); //queue.submit
}

int main(int argc, const char** argv)
{

  //sycl::queue queue(sycl::cpu_selector_v, sycl::property::queue::in_order());
  sycl::queue queue(sycl::gpu_selector_v, sycl::property::queue::in_order());

  //Size along y
  int jmax = 4094;
  //Size along x
  int imax = 4094;
  //Size along x
  int iter_max = 100;

  double pi  = 2.0 * asin(1.0);
  const double tol = 1.0e-6;
  double error     = 1.0;

  double *d_A;
  double *d_Anew;

  d_A    = sycl::malloc_device<double>((imax+2) * (jmax+2), queue);
  d_Anew = sycl::malloc_device<double>((imax+2) * (jmax+2), queue);

  queue.fill(d_A, 0, (imax+2) * (jmax+2));
  queue.wait();

  // set boundary conditions
  queue.parallel_for(imax+2, [d_A,imax](auto i)  { d_A[(0)*(imax+2)+i]   = 0.0; });

  queue.parallel_for(imax+2, [d_A,imax,jmax](auto i)  { d_A[(jmax+1)*(imax+2)+i] = 0.0; });

  queue.parallel_for(jmax+2, [d_A,imax,jmax,pi](auto j)  { d_A[(j)*(imax+2)+0] = sycl::sin(pi * j / (jmax+1)); });

  queue.parallel_for(jmax+2, [d_A,imax,jmax,pi](auto j)  { d_A[(j)*(imax+2)+imax+1] = sycl::sin(pi * j / (jmax+1))*sycl::exp(-pi); });

  queue.parallel_for(imax+2, [d_Anew,imax](auto i)  { d_Anew[(0)*(imax+2)+i]   = 0.0; });

  queue.parallel_for(imax+2, [d_Anew,imax,jmax](auto i)  { d_Anew[(jmax+1)*(imax+2)+i] = 0.0; });

  queue.parallel_for(jmax+2, [d_Anew,imax,jmax,pi](auto j)  { d_Anew[(j)*(imax+2)+0] = sycl::sin(pi * j / (jmax+1)); });

  queue.parallel_for(jmax+2, [d_Anew,imax,jmax,pi](auto j)  { d_Anew[(j)*(imax+2)+imax+1] = sycl::sin(pi * j / (jmax+1))*sycl::exp(-pi); });

  printf("Jacobi relaxation Calculation: %d x %d mesh\n", imax+2, jmax+2);

  int iter = 0;

  int start_x = 1, end_x = imax+1;
  int start_y = 1, end_y = jmax+1;
  int tblock_x = 32, tblock_y = 16;
  int ngrid_x = (end_x-start_x-1)/tblock_x + 1;
  int ngrid_y = (end_y-start_y-1)/tblock_y + 1;
  int maxblocks = ngrid_x*ngrid_y;

  double *h_reduction_ptr = sycl::malloc_host<double>(maxblocks, queue);
  double *d_reduction_ptr = sycl::malloc_device<double>(maxblocks, queue);

  queue.wait();

  while ( error > tol && iter < iter_max )
  {
    error = -INFINITY_double;
    
    for(int b = 0; b < maxblocks; b++)
      h_reduction_ptr[b] = -INFINITY_double;

//  for memcpy to work host memory should be allocated using sycl::malloc_host and device memory using sycl::malloc_device
    queue.memcpy(d_reduction_ptr, h_reduction_ptr, (maxblocks)*sizeof(double));
    queue.wait();

    jacobi_kernel(queue, d_A, d_Anew, d_reduction_ptr, imax, jmax, tblock_x, tblock_y, ngrid_x, ngrid_y);
    queue.wait();

    queue.memcpy(h_reduction_ptr, d_reduction_ptr, (maxblocks)*sizeof(double));
    queue.wait();

    for(int b = 0; b < maxblocks; b++)
      error = MAX(error, h_reduction_ptr[b]);

    copy(queue, d_A, d_Anew, imax, jmax, tblock_x, tblock_y, ngrid_x, ngrid_y);
    queue.wait();

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

  sycl::free(d_reduction_ptr,queue);
  sycl::free(h_reduction_ptr,queue);

  sycl::free(d_A,queue);
  sycl::free(d_Anew,queue);
 
  return 0;
}

