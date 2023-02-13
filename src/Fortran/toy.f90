program matrix_multiplication

    integer, parameter :: N = 3
    real, dimension(N, N) :: A, B, C
    integer :: i, j, k

    ! Initialize matrices A and B
    ! NOTE: The following reshape will result in two matrices:
    ! A = [[1, 4, 7],
    !      [2, 5, 8],
    !      [3, 6, 9]]
    A = reshape([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3])
    ! B = [[9, 6, 3],
    !      [8, 5, 2],
    !      [7, 4, 1]]
    B = reshape([9, 8, 7, 6, 5, 4, 3, 2, 1], [3, 3])

    ! Multiply matrices A and B
    do i = 1, N
        do j = 1, N
            C(i, j) = 0.0
            do k = 1, N
                C(i, j) = C(i, j) + A(i, k) * B(k, j)
            end do
        end do
    end do

    ! Output the result
    write(*,'(A)') "The result is:"
    write(*,'(3F6.2)') C(1, 1), C(1, 2), C(1, 3)
    write(*, '(3F6.2)') C(2, 1), C(2, 2), C(2, 3)
    write(*, '(3F6.2)') C(3, 1), C(3, 2), C(3, 3)

end program matrix_multiplication
