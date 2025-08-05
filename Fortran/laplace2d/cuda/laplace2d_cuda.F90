!
!               ^
!               |        top
!               |    000000000000   r
!               | l  000000000000   i
!   y-direction | e  000000000000   g
!    (index-j)  | f  000000000000   h
!               | t  000000000000   t
!               |       bottom
!               o------------------>
!                      x-direction
!                       (index-i)

PROGRAM LAPLACE_2D

    USE CUDAFOR
    USE CUDA_KERNELS

    IMPLICIT NONE

    ! size along x 
    INTEGER(KIND=4), PARAMETER :: imax=4094
    ! size along y
    INTEGER(KIND=4), PARAMETER :: jmax=4094
    ! max iterations
    INTEGER(KIND=4), PARAMETER :: iter_max=100

    INTEGER(KIND=4) :: i, j, iter

    REAL(KIND=8), DEVICE, DIMENSION(:), ALLOCATABLE  :: d_A, d_Anew

    REAL(KIND=8), PARAMETER :: pi=2.0_8*asin(1.0_8)
    REAL(KIND=8), PARAMETER :: tol=1.0e-6_8
    REAL(KIND=8) :: error=1.0_8
    REAL(KIND=8) :: err_diff

    REAL(KIND=8) :: start_time, stop_time
    
    TYPE(dim3) :: grid, tblock
    INTEGER(KIND=4) :: maxblocks
    INTEGER(KIND=4) :: nshared
    INTEGER(KIND=4) :: nthreads
    REAL(KIND=8), DIMENSION(:), ALLOCATABLE :: h_reduction_array
    REAL(KIND=8), DEVICE, DIMENSION(:), ALLOCATABLE :: d_reduction_array

    ALLOCATE ( d_A   ((jmax+2)*(imax+2)) )
    ALLOCATE ( d_Anew((jmax+2)*(imax+2)) )

    ! Initialize
    d_A = 0.0_8

    call cpu_time(start_time)

    ! Set boundary conditions

    tblock = dim3(32,16,1)
    grid   = dim3((jmax+2+tblock%x-1)/tblock%x, (imax+2+tblock%y-1)/tblock%y, 1);

    ! Bottom

    CALL boundary_x<<<grid,tblock>>>(d_A, imax, jmax)    
    CALL boundary_y<<<grid,tblock>>>(d_A, imax, jmax, pi)


    WRITE(*,'(a,i5,a,i5,a)') 'Jacobi relaxation Calculation:', jmax+2, ' x', imax+2, ' mesh'

    CALL boundary_x<<<grid,tblock>>>(d_Anew, imax, jmax)
    CALL boundary_y<<<grid,tblock>>>(d_Anew, imax, jmax, pi)

 
    iter=0

    nthreads = tblock%x * tblock%y * tblock%z
    maxblocks = grid%x * grid%y * grid%z

    nshared = nthreads * sizeof(error)
    ALLOCATE(h_reduction_array(maxblocks))
    ALLOCATE(d_reduction_array(maxblocks))

    DO WHILE ( error .gt. tol .and. iter .lt. iter_max )
        error = 0.0_8

        CALL jacobi_host(d_A, d_Anew, error, imax, jmax, h_reduction_array, d_reduction_array, nshared, maxblocks, grid, tblock)      
 
        CALL copy<<<grid,tblock>>>(d_A, d_Anew, imax, jmax)

        if(mod(iter,10).eq.0 ) write(*,'(i5,a,f16.7)') iter, ', ',error
            iter = iter +1

    END DO  ! End of do while loop

    DEALLOCATE(h_reduction_array)
    DEALLOCATE(d_reduction_array)

    WRITE(*,'(i5,a,f16.7)') iter, ', ',error

    err_diff = abs((100.0*(error/2.421354960840227e-03))-100.0)
       
    WRITE(*,'(a,e18.5,a)') 'Total error is within ', err_diff,' % of the expected error'
     
    IF(err_diff .lt. 0.001_8) THEN
        write(*,'(a)') 'This run is considered PASSED'
    ELSE
        write(*,'(a)') 'This test is considered FAILED'
    END IF

    call cpu_time(stop_time) 
    WRITE(*,'(a,f16.7,a)')  ' completed in ', stop_time-start_time, ' seconds'

    DEALLOCATE (d_A,d_Anew)

END PROGRAM LAPLACE_2D
