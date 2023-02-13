# A simple make file used to build the two Fortran toy examples

# The compiler to use
FC = ifort

# The MPI compiler to use
MPI_FC = mpiifort

# The flags to pass to the compiler
FLAGS = -g -traceback -check all -warn all -O0

# The source and build directories
SRC = src/Fortran/
BUILD = build/

# The name of the executable
a.out: $(SRC)toy.f90
	$(FC) $(FLAGS) $(SRC)toy.f90 -o $(BUILD)a.out

b.out: $(SRC)toy_mpi.f90
	$(MPI_FC) $(FLAGS) $(SRC)toy_mpi.f90 -o $(BUILD)b.out
