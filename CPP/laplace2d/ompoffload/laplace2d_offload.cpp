#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <omp.h>
#define INFINITY_double DBL_MAX
#define MAX(a, b) ((a > b) ? (a) : (b))

int main(int argc, const char** argv)
{

  omp_set_default_device(0);
  //Size along y
  int jmax = 4094;
  //Size along x
  int imax = 4094;
  //Size along x
  int iter_max = 100;

  double pi  = 2.0 * asin(1.0);
  const double tol = 1.0e-6;
  double error     = 1.0;

  double *A;
  double *Anew;
  int iter = 0;

  A    = (double *)malloc((imax+2) * (jmax+2) * sizeof(double));
  Anew = (double *)malloc((imax+2) * (jmax+2) * sizeof(double));

  memset(A, 0, (imax+2) * (jmax+2) * sizeof(double));
  memset(Anew, 0, (imax+2) * (jmax+2) * sizeof(double));

  size_t N = (imax+2)*(jmax+2);

// Moving data to device memory
  #pragma omp target enter data map(to: A[0:N])
  #pragma omp target enter data map(to: Anew[0:N])

  // set boundary conditions
  #pragma omp target teams distribute parallel for
  for (int i = 0; i < imax+2; i++)
  {
    A[(0)*(imax+2)+i]   = 0.0;
    A[(jmax+1)*(imax+2)+i] = 0.0;
  }  

  #pragma omp target teams distribute parallel for
  for (int j = 0; j < jmax+2; j++)
  {
    A[(j)*(imax+2)+0] = sin(pi * j / (jmax+1.0));
    A[(j)*(imax+2)+imax+1] = sin(pi * j / (jmax+1.0))*exp(-pi);
  }

  printf("Jacobi relaxation Calculation: %d x %d mesh\n", imax+2, jmax+2);

  #pragma omp target teams distribute parallel for
  for (int i = 1; i < imax+2; i++)
  {
    Anew[(0)*(imax+2)+i]   = 0.0;
    Anew[(jmax+1)*(imax+2)+i] = 0.0;
  }

  #pragma omp target teams distribute parallel for
  for (int j = 1; j < jmax+2; j++)
  {
    Anew[(j)*(imax+2)+0]   = sin(pi * j / (jmax+1.0));
    Anew[(j)*(imax+2)+imax+1] = sin(pi * j / (jmax+1.0))*exp(-pi);
  }

  while ( error > tol && iter < iter_max )
  {
    error = -INFINITY_double;
    #pragma omp target teams distribute parallel for reduction(max: error) 
    for( int j = 1; j < jmax+1; j++ )
    {
      for( int i = 1; i < imax+1; i++)
      {
        Anew[(j)*(imax+2)+i] = 0.25f * ( A[(j)*(imax+2)+i+1] + A[(j)*(imax+2)+i-1]
            + A[(j-1)*(imax+2)+i] + A[(j+1)*(imax+2)+i]);

        double diff = -INFINITY_double;
        diff = fmax(diff, fabs(Anew[(j)*(imax+2)+i]-A[(j)*(imax+2)+i]));
        error = MAX(error, diff);
      }
    }

    #pragma omp target teams distribute parallel for collapse(2)
    for( int j = 1; j < jmax+1; j++ )
    {
      for( int i = 1; i < imax+1; i++)
      {
        A[(j)*(imax+2)+i] = Anew[(j)*(imax+2)+i];    
      }
    }

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

  #pragma omp target exit data map(from: A)
  #pragma omp target exit data map(from: Anew)

  free(A);
  free(Anew);
  return 0;
}

