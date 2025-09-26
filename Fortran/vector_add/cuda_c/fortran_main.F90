program vector_add

    use iso_c_binding
    implicit none

!--- interface to C functions
    interface 

        subroutine device_alloc_c(ptr, n) bind(C, name="device_alloc")
            use iso_c_binding
            integer(c_int), value :: n
            type(c_ptr) :: ptr
        end subroutine device_alloc_c

        subroutine device_memcpy_h2d_c(dst, src, n) bind(C, name="device_memcpy_h2d")
            use iso_c_binding
            integer(c_int), value :: n
            integer(c_int) :: src(n)
            type(c_ptr) :: dst
        end subroutine device_memcpy_h2d_c

        subroutine device_memcpy_d2h_c(dst, src, n) bind(C, name="device_memcpy_d2h")
            use iso_c_binding
            integer(c_int), value :: n
            integer(c_int) :: dst(n)
            type(c_ptr) :: src
        end subroutine device_memcpy_d2h_c

        subroutine device_vector_add_c(a, b, c, n) bind(C, name="device_vector_add")
            use iso_c_binding
            integer(c_int), value :: n
            type(c_ptr) :: a, b, c
        end subroutine device_vector_add_c

        subroutine device_free_c(ptr) bind(C, name="device_free")
            use iso_c_binding
            type(c_ptr), value :: ptr
        end subroutine device_free_c

    end interface

!--- variable declarations
    integer, parameter :: N = 4096
!   Fortran side arrays - resides on host memory
    integer(c_int), allocatable :: h_A(:), h_B(:), h_C(:)
    integer :: i
    real :: rnd

!   C-ptr will be allocated on C side and resides on device memory
    type(c_ptr) :: d_A, d_B, d_C

!   allocate fortran side arrays and initialise with some random numbers
    allocate(h_A(N))
    allocate(h_B(N))
    allocate(h_C(N))

    h_C = 0

    call random_seed()   ! initialize RNG

    do i = 1, N
        call random_number(rnd)      ! rnd in [0,1)
        h_A(i) = int(rnd * 100.0)    ! scale to 0..99
        call random_number(rnd)
        h_B(i) = int(rnd * 100.0)
    end do

!   allocate c side arrays on device
    d_A = c_null_ptr
    call device_alloc_c(d_A, N)
    d_B = c_null_ptr
    call device_alloc_c(d_B, N)
    d_C = c_null_ptr
    call device_alloc_c(d_C, N)

!   copy fortran host array content to C device array
    call device_memcpy_h2d_c(d_A, h_A, N)
    call device_memcpy_h2d_c(d_B, h_B, N)

!   call vector_addition on C side
    call device_vector_add_c(d_A, d_B, d_C, N)

!   copy result back to fortran side host array
    call device_memcpy_d2h_c(h_C, d_C, N)

!   verify result
    do i = 1, N
        if(h_C(i) .ne. h_A(i)+h_B(i)) then
            print *, "FAILED !!!"
            stop
        end if
    end do

    print *, "PASSED !!!"

    deallocate(h_A)
    deallocate(h_B)
    deallocate(h_C)
    call device_free_c(d_A)
    call device_free_c(d_B)
    call device_free_c(d_C)

end program vector_add
