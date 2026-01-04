#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>

#include "timer.h"

static inline int mod(int v, int m) {
    int val = v%m;
    if (val<0) val = m+val;
    return val;
}

// Implemented from OPS-DSL sycl reduction helper functions
void sycl_reduction_inc(double *d_reduction, double value, sycl::local_accessor<double, 1> temp, sycl::nd_item<2> &item_id, int group_size) {
  double value_tid;
  item_id.barrier(sycl::access::fence_space::local_space); /* important to finish all previous activity */

  size_t linear_id = item_id.get_local_linear_id();
  temp[linear_id] = value;

  item_id.barrier(sycl::access::fence_space::local_space); /* important to finish all previous activity */

  for (size_t d = group_size / 2; d > 0; d >>= 1) {
    if (linear_id < d) {
      value_tid = temp[linear_id + d];
      value = value + value_tid;
      temp[linear_id] = value;
    }
    item_id.barrier(sycl::access::fence_space::local_space);
  }

  if (linear_id == 0) {
    d_reduction[0] = d_reduction[0] + value;
  }
}

void cal_energy(sycl::queue& queue, double *d_ux, double *d_uy, double *d_reduction_ptr, int NX, int NY, int tblock_x, int tblock_y, int ngrid_x, int ngrid_y) {

    queue.submit([&] (sycl::handler& h) {
        sycl::local_accessor<double, 1> local_mem(sycl::range<1>(tblock_x * tblock_y), h);

        h.parallel_for(sycl::nd_range<2>(
                                      sycl::range<2>(ngrid_y*tblock_y,ngrid_x*tblock_x),
                                      sycl::range<2>(tblock_y,tblock_x)
                                     )
                  , [=](sycl::nd_item<2> item) [[intel::kernel_args_restrict]] {
            int j = item.get_global_id()[0];
            int i = item.get_global_id()[1];

            double energy = 0.0;

            if (i < NX && j < NY) {
                energy += d_ux[j*NX+i]*d_ux[j*NX+i] + d_uy[j*NX+i]*d_uy[j*NX+i]; // reduction
            }

            int group_size = item.get_local_range(0) * item.get_local_range(1);

            sycl_reduction_inc(&d_reduction_ptr[item.get_group_linear_id()], energy, local_mem, item, group_size);
        }); //h.parallel_for
    }); // end of queue.submit
    queue.wait();
}

int main(int argc, char ** argv) {

    //sycl::queue queue(sycl::cpu_selector_v, sycl::property::queue::in_order());
    sycl::queue queue(sycl::gpu_selector_v, sycl::property::queue::in_order());

    const int NX = 128;
    const int NY = 128;

    const double OMEGA = 1.0;
    const double rho0 = 1.0;
    const double deltaUX=10e-6;

    const double W[] = {4.0/9.0,1.0/9.0,1.0/36.0,1.0/9.0,1.0/36.0,1.0/9.0,1.0/36.0,1.0/9.0,1.0/36.0};
    const int cx[] = {0,0,1,1, 1, 0,-1,-1,-1};
    const int cy[] = {0,1,1,0,-1,-1,-1, 0, 1};

    const int opposite[] = {0,5,6,7,8,1,2,3,4};
    
    double energy;

    double ct0,et0,ct1,et1; //timer variables

    double *h_W = sycl::malloc_host<double>(9, queue);
    int *h_cx = sycl::malloc_host<int>(9, queue);
    int *h_cy = sycl::malloc_host<int>(9, queue);
    int *h_opposite = sycl::malloc_host<int>(9, queue);

    memcpy(h_W, W, 9*sizeof(double));
    memcpy(h_cx, cx, 9*sizeof(int));
    memcpy(h_cy, cy, 9*sizeof(int));
    memcpy(h_opposite, opposite, 9*sizeof(int));

//  Copy from host to device, host memory should be allocated with sycl::malloc_host and device memory with sycl::malloc_device
    double *d_W = sycl::malloc_device<double>(9, queue);
    int *d_cx = sycl::malloc_device<int>(9, queue);
    int *d_cy = sycl::malloc_device<int>(9, queue);
    int *d_opposite = sycl::malloc_device<int>(9, queue);

    queue.memcpy(d_W, h_W, 9*sizeof(double));
    queue.memcpy(d_cx, h_cx, 9*sizeof(int));
    queue.memcpy(d_cy, h_cy, 9*sizeof(int));
    queue.memcpy(d_opposite, h_opposite, 9*sizeof(int));
    queue.wait();

    int start_x = 0, end_x = NX;
    int start_y = 0, end_y = NY;
    int tblock_x = 32, tblock_y = 16;
    int ngrid_x = (end_x-start_x-1)/tblock_x + 1;
    int ngrid_y = (end_y-start_y-1)/tblock_y + 1;
    int maxblocks = ngrid_x*ngrid_y;

    double *h_reduction_ptr = sycl::malloc_host<double>(maxblocks, queue);
    double *d_reduction_ptr = sycl::malloc_device<double>(maxblocks, queue);

    // Generate obstacles based on grid positions
    int *h_SOLID, *d_SOLID;
    h_SOLID = sycl::malloc_host<int>(NX*NY, queue);
    d_SOLID = sycl::malloc_device<int>(NX*NY, queue);

    queue.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>(
                                      sycl::range<2>(ngrid_y*tblock_y,ngrid_x*tblock_x),
                                      sycl::range<2>(tblock_y,tblock_x)
                                     )
                       , [=](sycl::nd_item<2> item) {

            int j = item.get_global_id()[0];
            int i = item.get_global_id()[1];

            if (i < NX && j < NY) {
                if( (i>=2 && i<=65 && j>=4 && j<=29) || (i>=45 && i<=123 && j>=41 && j<=65) || (i>=30 && i<=101 && j>=91 && j<=115))
                    d_SOLID[j*NX+i] = 0;
                else
                    d_SOLID[j*NX+i] = 1;
            }
        });  //h.parallel_for
    }); //queue.submit
    queue.wait();

    //Initial values
    double *d_N;
    d_N = sycl::malloc_device<double>(NX*NY*9, queue);
    queue.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>(
                                      sycl::range<2>(ngrid_y*tblock_y,ngrid_x*tblock_x),
                                      sycl::range<2>(tblock_y,tblock_x)
                                     )
                       , [=](sycl::nd_item<2> item) {

            int j = item.get_global_id()[0];
            int i = item.get_global_id()[1];

            if (i < NX && j < NY) {
                for (int f = 0; f < 9; f++) {
                    d_N[(j*NX+i)*9 + f] = rho0 * d_W[f];
                }
            }
        });  //h.parallel_for
    }); //queue.submit    
    queue.wait();

    //Work arrays
    double *d_workArray = sycl::malloc_device<double>(NX*NY*9, queue);
    double *d_N_SOLID = sycl::malloc_device<double>(NX*NY*9, queue);
    double *d_rho = sycl::malloc_device<double>(NX*NY, queue);
    double *h_ux, *d_ux;
    h_ux = sycl::malloc_host<double>(NX*NY, queue);
    d_ux = sycl::malloc_device<double>(NX*NY, queue);
    double *h_uy, *d_uy;
    h_uy = sycl::malloc_host<double>(NX*NY, queue);
    d_uy = sycl::malloc_device<double>(NX*NY, queue);

    //Start timer
    timer(&ct0, &et0);

    //Main time loop
    for (int t = 0; t < 4000; t++) {

        //Backup values
        queue.submit([&] (sycl::handler& h) {
            h.parallel_for(sycl::nd_range<2>(
                                      sycl::range<2>(ngrid_y*tblock_y,ngrid_x*tblock_x),
                                      sycl::range<2>(tblock_y,tblock_x)
                                     )
                       , [=](sycl::nd_item<2> item) {

                int j = item.get_global_id()[0];
                int i = item.get_global_id()[1];

                if (i < NX && j < NY) {
                    for (int f = 0; f < 9; f++) {
                        d_workArray[(j*NX+i)*9 + f] = d_N[(j*NX+i)*9 + f];
                    }
                }
            });  //h.parallel_for
        }); //queue.submit
        queue.wait();

        // Gather neighbour values
        queue.submit([&] (sycl::handler& h) {
            h.parallel_for(sycl::nd_range<2>(
                                      sycl::range<2>(ngrid_y*tblock_y,ngrid_x*tblock_x),
                                      sycl::range<2>(tblock_y,tblock_x)
                                     )
                       , [=](sycl::nd_item<2> item) {

                int j = item.get_global_id()[0];
                int i = item.get_global_id()[1];

                if (i < NX && j < NY) {
				    for (int f = 1; f < 9; f++) {
					    d_N[(j*NX+i)*9 + f] = d_workArray[(mod(j-d_cy[f],NY)*NX+mod(i-d_cx[f],NX))*9 + f];
				    }
                }
			});  //h.parallel_for
        }); //queue.submit
        queue.wait();

        //Bounce back from solids, no collision
        queue.submit([&] (sycl::handler& h) {
            h.parallel_for(sycl::nd_range<2>(
                                      sycl::range<2>(ngrid_y*tblock_y,ngrid_x*tblock_x),
                                      sycl::range<2>(tblock_y,tblock_x)
                                     )
                       , [=](sycl::nd_item<2> item) {

                int j = item.get_global_id()[0];
                int i = item.get_global_id()[1];

                if (i < NX && j < NY) {
                    if (d_SOLID[j*NX+i]==1) {
                        for (int f = 0; f < 9; f++) {
                            d_N_SOLID[(j*NX+i)*9 + d_opposite[f]] = d_N[(j*NX+i)*9 + f];
                        }
                    }
                }
            });  //h.parallel_for
        }); //queue.submit
        queue.wait();

        queue.submit([&] (sycl::handler& h) {
            h.parallel_for(sycl::nd_range<2>(
                                      sycl::range<2>(ngrid_y*tblock_y,ngrid_x*tblock_x),
                                      sycl::range<2>(tblock_y,tblock_x)
                                     )
                       , [=](sycl::nd_item<2> item) {

                int j = item.get_global_id()[0];
                int i = item.get_global_id()[1];

                if (i < NX && j < NY) {
                    d_rho[j*NX+i] = 0.0;
                    for (int f = 0; f < 9; f++) {
                        d_rho[j*NX+i] += d_N[(j*NX+i)*9 + f];
                    }
                }
            });  //h.parallel_for
        }); //queue.submit
        queue.wait();

        queue.submit([&] (sycl::handler& h) {
            h.parallel_for(sycl::nd_range<2>(
                                      sycl::range<2>(ngrid_y*tblock_y,ngrid_x*tblock_x),
                                      sycl::range<2>(tblock_y,tblock_x)
                                     )
                       , [=](sycl::nd_item<2> item) {

                int j = item.get_global_id()[0];
                int i = item.get_global_id()[1];

                if (i < NX && j < NY) {
                    d_ux[j*NX+i] = 0.0;
                    for (int f = 0; f < 9; f++) {
                        d_ux[j*NX+i] += d_N[(j*NX+i)*9 + f] * d_cx[f];
                    }
                    d_ux[j*NX+i] = d_ux[j*NX+i] / d_rho[j*NX+i] + deltaUX;
                }
            });  //h.parallel_for
        }); //queue.submit
        queue.wait();

        queue.submit([&] (sycl::handler& h) {
            h.parallel_for(sycl::nd_range<2>(
                                      sycl::range<2>(ngrid_y*tblock_y,ngrid_x*tblock_x),
                                      sycl::range<2>(tblock_y,tblock_x)
                                     )
                       , [=](sycl::nd_item<2> item) {

                int j = item.get_global_id()[0];
                int i = item.get_global_id()[1];

                if (i < NX && j < NY) {
                    d_uy[j*NX+i] = 0.0;
                    for (int f = 0; f < 9; f++) {
                        d_uy[j*NX+i] += d_N[(j*NX+i)*9 + f] * d_cy[f];
                    }
                    d_uy[j*NX+i] = d_uy[j*NX+i] / d_rho[j*NX+i];
                }
            });  //h.parallel_for
        }); //queue.submit
        queue.wait();

        queue.submit([&] (sycl::handler& h) {
            h.parallel_for(sycl::nd_range<2>(
                                      sycl::range<2>(ngrid_y*tblock_y,ngrid_x*tblock_x),
                                      sycl::range<2>(tblock_y,tblock_x)
                                     )
                       , [=](sycl::nd_item<2> item) {

                int j = item.get_global_id()[0];
                int i = item.get_global_id()[1];

                if (i < NX && j < NY) {
                    for (int f = 0; f < 9; f++) {
                        d_workArray[(j*NX+i)*9 + f] = d_ux[j*NX+i]*d_cx[f] + d_uy[j*NX+i]*d_cy[f];
                    }
                }
            });  //h.parallel_for
        }); //queue.submit
        queue.wait();

        queue.submit([&] (sycl::handler& h) {
            h.parallel_for(sycl::nd_range<2>(
                                      sycl::range<2>(ngrid_y*tblock_y,ngrid_x*tblock_x),
                                      sycl::range<2>(tblock_y,tblock_x)
                                     )
                       , [=](sycl::nd_item<2> item) {

                int j = item.get_global_id()[0];
                int i = item.get_global_id()[1];

                if (i < NX && j < NY) {
                    for (int f = 0; f < 9; f++) {
                        d_workArray[(j*NX+i)*9 + f] = (3.0+4.5*d_workArray[(j*NX+i)*9 + f])*d_workArray[(j*NX+i)*9 + f];
                    }
                }
            });  //h.parallel_for
        }); //queue.submit
        queue.wait();

        queue.submit([&] (sycl::handler& h) {
            h.parallel_for(sycl::nd_range<2>(
                                      sycl::range<2>(ngrid_y*tblock_y,ngrid_x*tblock_x),
                                      sycl::range<2>(tblock_y,tblock_x)
                                     )
                       , [=](sycl::nd_item<2> item) {

                int j = item.get_global_id()[0];
                int i = item.get_global_id()[1];

                if (i < NX && j < NY) {
                    for (int f = 0; f < 9; f++) {
                        d_workArray[(j*NX+i)*9 + f] = d_workArray[(j*NX+i)*9 + f] - 1.5 * (d_ux[j*NX+i]*d_ux[j*NX+i] + d_uy[j*NX+i]*d_uy[j*NX+i]);
                    }
                }
            });  //h.parallel_for
        }); //queue.submit
        queue.wait();

        queue.submit([&] (sycl::handler& h) {
            h.parallel_for(sycl::nd_range<2>(
                                      sycl::range<2>(ngrid_y*tblock_y,ngrid_x*tblock_x),
                                      sycl::range<2>(tblock_y,tblock_x)
                                     )
                       , [=](sycl::nd_item<2> item) {

                int j = item.get_global_id()[0];
                int i = item.get_global_id()[1];

                if (i < NX && j < NY) {
                    for (int f = 0; f < 9; f++) {
                        d_workArray[(j*NX+i)*9 + f] = (1.0+d_workArray[(j*NX+i)*9 + f]) * d_W[f] * d_rho[j*NX+i];
                    }
                }
            });  //h.parallel_for
        }); //queue.submit
        queue.wait();

        queue.submit([&] (sycl::handler& h) {
            h.parallel_for(sycl::nd_range<2>(
                                      sycl::range<2>(ngrid_y*tblock_y,ngrid_x*tblock_x),
                                      sycl::range<2>(tblock_y,tblock_x)
                                     )
                       , [=](sycl::nd_item<2> item) {

                int j = item.get_global_id()[0];
                int i = item.get_global_id()[1];

                if (i < NX && j < NY) {
                    for (int f = 0; f < 9; f++) {
                        d_N[(j*NX+i)*9 + f] += (d_workArray[(j*NX+i)*9 + f] - d_N[(j*NX+i)*9 + f]) * OMEGA;
                    }
                }
            });  //h.parallel_for
        }); //queue.submit
        queue.wait();

        queue.submit([&] (sycl::handler& h) {
            h.parallel_for(sycl::nd_range<2>(
                                      sycl::range<2>(ngrid_y*tblock_y,ngrid_x*tblock_x),
                                      sycl::range<2>(tblock_y,tblock_x)
                                     )
                       , [=](sycl::nd_item<2> item) {

                int j = item.get_global_id()[0];
                int i = item.get_global_id()[1];

                if (i < NX && j < NY) {
                    if (d_SOLID[j*NX+i]==1) {
                        for (int f = 0; f < 9; f++) {
                            d_N[(j*NX+i)*9 + f] = d_N_SOLID[(j*NX+i)*9 + f];
                        }
                    }
                }
            });  //h.parallel_for
        }); //queue.submit
        queue.wait();

        //Calculate kinetic energy
        energy = 0.0;
/*
        queue.memcpy(h_ux, d_ux, NX*NY*sizeof(double));
        queue.memcpy(h_uy, d_uy, NX*NY*sizeof(double));
        queue.wait();

        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                energy += h_ux[j*NX+i]*h_ux[j*NX+i]+h_uy[j*NX+i]*h_uy[j*NX+i]; // reduction
            }
        }
*/
        for(int b = 0; b < maxblocks; b++)
            h_reduction_ptr[b] = 0.0;

        //  for memcpy to work host memory should be allocated using sycl::malloc_host and device memory using sycl::malloc_device
        queue.memcpy(d_reduction_ptr, h_reduction_ptr, (maxblocks)*sizeof(double));
        queue.wait();

        cal_energy(queue, d_ux, d_uy, d_reduction_ptr, NX, NY, tblock_x, tblock_y, ngrid_x, ngrid_y);

        queue.memcpy(h_reduction_ptr, d_reduction_ptr, (maxblocks)*sizeof(double));
        queue.wait();

        for(int b = 0; b < maxblocks; b++)
            energy += h_reduction_ptr[b];
        queue.wait();

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
    
    queue.memcpy(h_SOLID, d_SOLID, NX*NY*sizeof(int));
    queue.memcpy(h_ux, d_ux, NX*NY*sizeof(double));
    queue.memcpy(h_uy, d_uy, NX*NY*sizeof(double));
    queue.wait();

    if (true) {
        std::ofstream myfile;
        myfile.open ("output_velocity.txt");
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                myfile << h_SOLID[j*NX+i] << " " << h_ux[j*NX+i] << " " << h_uy[j*NX+i] << std::endl;
            }
        }
        myfile.close();
    }

    sycl::free(h_W, queue);    sycl::free(d_W, queue);
    sycl::free(h_cx, queue);    sycl::free(d_cx, queue);
    sycl::free(h_cy, queue);    sycl::free(d_cy, queue);
    sycl::free(h_opposite, queue);    sycl::free(d_opposite, queue);

    sycl::free(h_SOLID, queue);    sycl::free(d_SOLID, queue);
    sycl::free(d_N, queue);
    sycl::free(d_workArray, queue);
    sycl::free(d_N_SOLID, queue);
    sycl::free(d_rho, queue);
    sycl::free(h_ux, queue);    sycl::free(d_ux, queue);
    sycl::free(h_uy, queue);    sycl::free(d_uy, queue);

}// End of main function

