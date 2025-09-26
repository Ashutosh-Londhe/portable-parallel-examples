#include <Kokkos_Core.hpp>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char** argv)
{

  Kokkos::initialize(argc, argv);
  {
    //Size along y
    int jmax = 4094;
    //Size along x
    int imax = 4094;
    //Size along x
    int iter_max = 100;

    double pi  = 2.0 * asin(1.0);
    const double tol = 1.0e-6;
    double error     = 1.0;

    Kokkos::View<double**> A("A", (jmax+2), (imax+2));
    Kokkos::View<double**> Anew("Anew", (jmax+2), (imax+2));

    // KOKKOS_LAMBDA is set to "[=] __host__ __device__" so that code will be compiled for both host and device target
    // set boundary conditions
    Kokkos::parallel_for("Boundary_Y_top", imax+2,
      KOKKOS_LAMBDA(const int i) {
        A(0,i)   = 0.0;
    });

    Kokkos::parallel_for("Boundary_Y_bottom", imax+2,
      KOKKOS_LAMBDA(const int i) {
        A(jmax+1, i) = 0.0;
    });

    Kokkos::parallel_for("Boundary_X_left", jmax+2,
      KOKKOS_LAMBDA(const int j) {
        A(j,0) = sin(pi * j / (jmax+1));
    });

    Kokkos::parallel_for("Boundary_X_right", jmax+2,
      KOKKOS_LAMBDA(const int j) {
        A(j,imax+1) = sin(pi * j / (jmax+1))*exp(-pi);
    });

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", imax+2, jmax+2);

    Kokkos::parallel_for("Boundary_Y_top", imax+2,
      KOKKOS_LAMBDA(const int i) {
        Anew(0,i)   = 0.0;
    });

    Kokkos::parallel_for("Boundary_Y_bottom", imax+2,
      KOKKOS_LAMBDA(const int i) {
        Anew(jmax+1,i) = 0.0;
    });

    Kokkos::parallel_for("Boundary_X_left", jmax+2,
      KOKKOS_LAMBDA(const int j) {
        Anew(j,0) = sin(pi * j / (jmax+1));
    });

    Kokkos::parallel_for("Boundary_X_right", jmax+2,
      KOKKOS_LAMBDA(const int j) {
        Anew(j,imax+1) = sin(pi * j / (jmax+1))*exp(-pi);
    });


    int iter = 0;

    while ( error > tol && iter < iter_max )
    {
      double max_error_host = 0.0;

      // Jacobi iteration with reduction on error
      Kokkos::parallel_reduce(
        "Jacobi",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {jmax+1,imax+1}),
        KOKKOS_LAMBDA(const int j, const int i, double& local_error) {
          Anew(j,i) = 0.25 * ( A(j,i+1) + A(j,i-1)
                             + A(j-1,i) + A(j+1,i) );
          
          local_error = Kokkos::max<double>(local_error, fabs(Anew(j,i) - A(j,i)));  
        },
        Kokkos::Max<double>(max_error_host)
      );

      // Copy back: A = Anew
      Kokkos::parallel_for(
        "copy",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {jmax+1,imax+1}),
        KOKKOS_LAMBDA(const int j, const int i) {
          A(j,i) = Anew(j,i);
        }
      );

      // optional synchronization before host-side prints (useful for debugging)
      Kokkos::fence();

      // update host-side error for loop condition / printing
      error = max_error_host;

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

    //A = Kokkos::View<double*>();   // frees A immediately
    
  }

  Kokkos::finalize();
  return 0;
}
