program mpi_sum
    use mpi
    implicit none
    integer :: ierr, myid, numprocs, sum_all, partial_sum, root, i
    integer, dimension(100) :: a
    root = 0

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)

   !  Call MPI_debug_vscode(myid) so that we can debug the code
    CALL MPI_debug_vscode(myid)

    if (myid .eq. root) then
        ! initialize the array 'a' on the root process
        do i = 1, 100
            a(i) = i
        end do
    end if

    ! broadcast the array 'a' from the root process to all other processes
    call MPI_BCAST(a, 100, MPI_INTEGER, root, MPI_COMM_WORLD, ierr)

    ! compute the partial sum on each process
    partial_sum = sum(a(myid + 1:100:numprocs))

    ! send the partial sum to the root process
    if (myid .ne. root) then
        call MPI_SEND(partial_sum, 1, MPI_INTEGER, root, 0, MPI_COMM_WORLD, ierr)
    else
        ! root process receives the partial sums from all other processes
        sum_all = partial_sum
        do i = 1, numprocs - 1
            call MPI_RECV(partial_sum, 1, MPI_INTEGER, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
            sum_all = sum_all + partial_sum
        end do
    end if

    ! root process prints the final sum
    if (myid .eq. root) then
        write(*, *) "The final sum is: ", sum_all
    end if

    call MPI_FINALIZE(ierr)

CONTAINS 

   SUBROUTINE fortran_sleep(seconds)

      !! Using the method provide here: https://stackoverflow.com/a/6936205
      !! instead of the C binding of sleep() function, which is not portable across OSes

      INTEGER, INTENT(IN) :: seconds

      ! Argument for the date_and_time subroutine
      INTEGER :: t_arg(8)
      INTEGER :: ms1, ms2

      ! Get the start time
      CALL DATE_AND_TIME(values=t_arg)
      ! Start time in milliseconds
      ms1 = ( (t_arg(5) * 3600 + t_arg(6) * 60 + t_arg(7) ) * 1000 + t_arg(8))

      ! Check time
      DO 
         ! Get the time in the loop
         CALL DATE_AND_TIME(values=t_arg)
         ! The time at this moment in milliseconds
         ms2 = ( (t_arg(5) * 3600 + t_arg(6) * 60 + t_arg(7) ) * 1000 + t_arg(8))
         ! Check whether the difference is larger than the seconds passed in
         IF (ms2 - ms1 >= seconds * 1000) EXIT
      END DO

   END SUBROUTINE fortran_sleep

   SUBROUTINE MPI_debug_vscode(myid)

      !! When debugging with VSCode, you need to manually change the value of the variable ii from 
      !! 0 to 1 in order to continue running the program
      !! We assume that MPI_init has been called

      INTEGER, INTENT(IN) :: myid
      INTEGER :: ii, ierr

      PRINT*, 'RANK: ', myid

      ii = 0
      IF (myid == 0)THEN
         DO WHILE (ii /= 1)
         ! We only execute this on the master process so that we do not need to fire up
         ! vscode instances for all processes to step out the loop
            CALL fortran_sleep(3)
         END DO
      END IF
      ! A barrier is required so that all processes will stay at this point while the 
      ! master process is being attached (or any other slave processes are being 
      ! attached when needed)
      CALL MPI_Barrier(MPI_COMM_WORLD, ierr)

   END SUBROUTINE MPI_debug_vscode


end program mpi_sum
