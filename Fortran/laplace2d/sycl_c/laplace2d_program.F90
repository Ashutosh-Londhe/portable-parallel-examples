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

program laplace
    use iso_c_binding
    implicit none

!--- interface to C functions
    interface
    
        subroutine device_init_c( ) bind(C, name="device_init")
            use iso_c_binding
        end subroutine device_init_c

        subroutine host_alloc_c(ptr, n) bind(C, name="host_alloc")
            use iso_c_binding
            integer(c_int), value :: n
            type(c_ptr) :: ptr
        end subroutine host_alloc_c

        subroutine device_alloc_c(ptr, n) bind(C, name="device_alloc")
            use iso_c_binding
            integer(c_int), value :: n
            type(c_ptr) :: ptr
        end subroutine device_alloc_c

        subroutine device_memcpy_h2d_c(dst, src, n) bind(C, name="device_memcpy_h2d")
            use iso_c_binding
            integer(c_int), value :: n
            real(c_double) :: src(n)
            type(c_ptr) :: dst
        end subroutine device_memcpy_h2d_c

        subroutine device_memcpy_d2h_c(dst, src, n) bind(C, name="device_memcpy_d2h")
            use iso_c_binding
            integer(c_int), value :: n
            real(c_double) :: dst(n)
            type(c_ptr) :: src
        end subroutine device_memcpy_d2h_c

        subroutine device_boundary_x_c(ptr, imax, jmax) bind(C, name="device_boundary_x")
            use iso_c_binding
            integer(c_int), value :: imax, jmax
            type(c_ptr) :: ptr
        end subroutine device_boundary_x_c

        subroutine device_boundary_y_c(ptr, imax, jmax) bind(C, name="device_boundary_y")
            use iso_c_binding
            integer(c_int), value :: imax, jmax
            type(c_ptr) :: ptr
        end subroutine device_boundary_y_c

        subroutine jacobi_host_c(d_A, d_Anew, h_reduction_ptr, d_reduction_ptr, imax, jmax, error_host) bind(C, name="jacobi_host")
            use iso_c_binding
            type(c_ptr) :: d_A
            type(c_ptr) :: d_Anew
            type(c_ptr) :: h_reduction_ptr
            type(c_ptr) :: d_reduction_ptr
            integer(c_int), value :: imax, jmax
            real(c_double) :: error_host
        end subroutine jacobi_host_c

        subroutine device_copy_c(dst, src, imax, jmax) bind(C, name="device_copy")
            use iso_c_binding
            integer(c_int), value :: imax, jmax
            type(c_ptr) :: dst, src
        end subroutine device_copy_c

        subroutine device_free_c(ptr) bind(C, name="device_free")
            use iso_c_binding
            type(c_ptr), value :: ptr
        end subroutine device_free_c

    end interface

    ! size along x 
    integer, parameter :: imax=4094
    ! size along y
    integer, parameter :: jmax=4094
    ! max iterations
    integer, parameter :: iter_max=100

    integer :: iter
    integer :: start_x,end_x,start_y,end_y,tblock_x,tblock_y,ngrid_x,ngrid_y,maxblocks

    real(8), dimension (:), allocatable :: A

    real(8), parameter :: pi=2.0_8*asin(1.0_8)
    real(8), parameter :: tol=1.0e-6_8
    real(8) :: error=1.0_8
    real(8) :: err_diff

    type(c_ptr) :: d_A, d_Anew    
    type(c_ptr) :: h_reduction_ptr, d_reduction_ptr

    real(8) :: start_time, stop_time

    call device_init_c()

    allocate ( A   (((jmax+2)*(imax+2))) )

    d_A = c_null_ptr
    call device_alloc_c(d_A, (jmax+2)*(imax+2))
    d_Anew = c_null_ptr
    call device_alloc_c(d_Anew, (jmax+2)*(imax+2))

!   Allocating memory for reduction
    start_x = 1; end_x = imax+1
    start_y = 1; end_y = jmax+1
    tblock_x = 32; tblock_y = 16
    ngrid_x = (end_x-start_x-1)/tblock_x + 1
    ngrid_y = (end_y-start_y-1)/tblock_y + 1
    maxblocks = ngrid_x*ngrid_y
    call host_alloc_c(h_reduction_ptr, maxblocks)
    call device_alloc_c(d_reduction_ptr, maxblocks)

    ! Initialize
    A = 0.0_8

    call cpu_time(start_time)

!   Copy from fortran host memory to C side device memory
    call device_memcpy_h2d_c(d_A, A, (jmax+2)*(imax+2))

!   Set boundary conditions
    call device_boundary_x_c(d_A, imax, jmax)
    call device_boundary_y_c(d_A, imax, jmax)

    call device_boundary_x_c(d_Anew, imax, jmax)
    call device_boundary_y_c(d_Anew, imax, jmax)

    write(*,'(a,i5,a,i5,a)') 'Jacobi relaxation Calculation:', jmax+2, ' x', imax+2, ' mesh'
 
    iter=0

    do while ( error .gt. tol .and. iter .lt. iter_max )
        
        call jacobi_host_c(d_A, d_Anew, h_reduction_ptr, d_reduction_ptr, imax, jmax, error)
 
        call device_copy_c(d_A, d_Anew, imax, jmax)

        if(mod(iter,10).eq.0 ) write(*,'(i5,a,f16.7)') iter, ', ',error
            iter = iter +1

    end do  ! End of do while loop

    write(*,'(i5,a,f16.7)') iter, ', ',error

    err_diff = abs((100.0*(error/2.421354960840227e-03))-100.0)
       
    write(*,'(a,e18.5,a)') 'Total error is within ', err_diff,' % of the expected error'
     
    if(err_diff .lt. 0.001_8) then
        write(*,'(a)') 'This run is considered PASSED'
    else
        write(*,'(a)') 'This test is considered FAILED'
    end if

    call cpu_time(stop_time) 
    write(*,'(a,f16.7,a)')  ' completed in ', stop_time-start_time, ' seconds'

    deallocate (A)

    call device_free_c(d_A)
    call device_free_c(d_Anew)
    call device_free_c(h_reduction_ptr)
    call device_free_c(d_reduction_ptr)

end program laplace
