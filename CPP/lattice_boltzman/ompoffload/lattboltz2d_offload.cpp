#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <omp.h>

#include "timer.h"

static inline int mod(int v, int m) {
    int val = v%m;
    if (val<0) val = m+val;
    return val;
}

int main(int argc, char ** argv) {

//  Set GPU device
    omp_set_default_device(0);

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

    // Generate obstacles based on grid positions
    int * __restrict__ SOLID = new int[NX*NY];
    #pragma omp target enter data map(to: SOLID[0:NX*NY])

    #pragma omp target teams distribute parallel for collapse(2)
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            if( (i>=2 && i<=65 && j>=4 && j<=29) || (i>=45 && i<=123 && j>=41 && j<=65) || (i>=30 && i<=101 && j>=91 && j<=115))
                SOLID[j*NX+i] = 0;
            else
                SOLID[j*NX+i] = 1;
        }
    }

    //Initial values
    double * __restrict__ N = new double[NX*NY*9];
    #pragma omp target enter data map(to: N[0:NX*NY*9])

    #pragma omp target teams distribute parallel for collapse(2)
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            for (int f = 0; f < 9; f++) {
                N[(j*NX+i)*9 + f] = rho0 * W[f];
            }
        }
    }

    //Work arrays
    double * __restrict__ workArray = new double[NX*NY*9];
    double * __restrict__ N_SOLID = new double[NX*NY*9];
    double * __restrict__ rho = new double[NX*NY];
    double * __restrict__ ux = new double[NX*NY];
    double * __restrict__ uy = new double[NX*NY];

    //Start timer
    timer(&ct0, &et0);

//  Moving data to device memory
    #pragma omp target enter data map(to: W[0:9])
    #pragma omp target enter data map(to: cx[0:9])
    #pragma omp target enter data map(to: cy[0:9])
    #pragma omp target enter data map(to: opposite[0:9])

    #pragma omp target enter data map(to: workArray[0:NX*NY*9])
    #pragma omp target enter data map(to: N_SOLID[0:NX*NY*9])
    #pragma omp target enter data map(to: rho[0:NX*NY])
    #pragma omp target enter data map(to: ux[0:NX*NY])
    #pragma omp target enter data map(to: uy[0:NX*NY])
    //Main time loop
    for (int t = 0; t < 4000; t++) {

        //Backup values
        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                for (int f = 0; f < 9; f++) {
                    workArray[(j*NX+i)*9 + f] = N[(j*NX+i)*9 + f];
                }
            }
        }

        // Gather neighbour values
        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = 0; j < NY; j++) {
			for (int i = 0; i < NX; i++) {
				for (int f = 1; f < 9; f++) {
					N[(j*NX+i)*9 + f] = workArray[(mod(j-cy[f],NY)*NX+mod(i-cx[f],NX))*9 + f];
				}
			}
		}

        //Bounce back from solids, no collision
        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                if (SOLID[j*NX+i]==1) {
                    for (int f = 0; f < 9; f++) {
                        N_SOLID[(j*NX+i)*9 + opposite[f]] = N[(j*NX+i)*9 + f];
                    }
                }
            }
        }

        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                rho[j*NX+i] = 0.0;
                for (int f = 0; f < 9; f++) {
                    rho[j*NX+i] += N[(j*NX+i)*9 + f];
                }
            }
        }

        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                ux[j*NX+i] = 0.0;
                for (int f = 0; f < 9; f++) {
                    ux[j*NX+i] += N[(j*NX+i)*9 + f] * cx[f];
                }
                ux[j*NX+i] = ux[j*NX+i] / rho[j*NX+i] + deltaUX;
            }
        }

        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                uy[j*NX+i] = 0.0;
                for (int f = 0; f < 9; f++) {
                    uy[j*NX+i] += N[(j*NX+i)*9 + f] * cy[f];
                }
                uy[j*NX+i] = uy[j*NX+i] / rho[j*NX+i];
            }
        }

        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                for (int f = 0; f < 9; f++) {
                    workArray[(j*NX+i)*9 + f] = ux[j*NX+i]*cx[f] + uy[j*NX+i]*cy[f];
                }
            }
        }

        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                for (int f = 0; f < 9; f++) {
                    workArray[(j*NX+i)*9 + f] = (3.0+4.5*workArray[(j*NX+i)*9 + f])*workArray[(j*NX+i)*9 + f];
                }
            }
        }

        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                for (int f = 0; f < 9; f++) {
                    workArray[(j*NX+i)*9 + f] = workArray[(j*NX+i)*9 + f] - 1.5 * (ux[j*NX+i]*ux[j*NX+i] + uy[j*NX+i]*uy[j*NX+i]);
                }
            }
        }

        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                for (int f = 0; f < 9; f++) {
                    workArray[(j*NX+i)*9 + f] = (1.0+workArray[(j*NX+i)*9 + f]) * W[f] * rho[j*NX+i];
                }
            }
        }

        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                for (int f = 0; f < 9; f++) {
                    N[(j*NX+i)*9 + f] += (workArray[(j*NX+i)*9 + f] - N[(j*NX+i)*9 + f]) * OMEGA;
                }
            }
        }

        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                if (SOLID[j*NX+i]==1) {
                    for (int f = 0; f < 9; f++) {
                        N[(j*NX+i)*9 + f] = N_SOLID[(j*NX+i)*9 + f];
                    }
                }
            }
        }

        //Calculate kinetic energy
        energy = 0.0;
        #pragma omp target teams distribute parallel for reduction(+: energy)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                energy += ux[j*NX+i]*ux[j*NX+i]+uy[j*NX+i]*uy[j*NX+i]; // reduction 
            }
        }

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

//  Update host memory for writing arrays to file    
    #pragma omp target update from(SOLID[0:NX*NY])
    #pragma omp target update from(ux[0:NX*NY])
    #pragma omp target update from(uy[0:NX*NY])

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

//  Delete device copies
    #pragma omp target exit data map(delete: W)
    #pragma omp target exit data map(delete: cx)
    #pragma omp target exit data map(delete: cy)
    #pragma omp target exit data map(delete: opposite)

    #pragma omp target exit data map(delete: SOLID)
    delete[] SOLID;      SOLID = nullptr;
    #pragma omp target exit data map(delete: N)
    delete[] N;          N = nullptr;
    #pragma omp target exit data map(delete: workArray)
    delete[] workArray;  workArray = nullptr;
    #pragma omp target exit data map(delete: N_SOLID)
    delete[] N_SOLID;    N_SOLID = nullptr;
    #pragma omp target exit data map(delete: rho)
    delete[] rho;        rho = nullptr;
    #pragma omp target exit data map(delete: ux)
    delete[] ux;         ux = nullptr;
    #pragma omp target exit data map(delete: uy)
    delete[] uy;         uy = nullptr;

}// End of main function

