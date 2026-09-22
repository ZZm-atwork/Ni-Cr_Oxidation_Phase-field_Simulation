MPICXX ?= mpic++
CXX = c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic
MPIEXEC ?= mpirun
NP ?= 4

.PHONY: all test verify-mpi serial-check emulated-check clean
all: nicr2d_mpi

nicr2d_mpi: NiCr2D_grains.cpp
	$(MPICXX) $(CXXFLAGS) $< -o $@

test: nicr2d_mpi
	$(MPIEXEC) -np $(NP) ./nicr2d_mpi --self-test

verify-mpi: nicr2d_mpi
	python3 tests/verify_mpi.py --binary ./nicr2d_mpi --launcher "$(MPIEXEC)" --ranks 1 2 4

# These two targets are developer checks, NOT substitutes for native MPI tests.
serial_verify: NiCr2D_grains.cpp tests/mpi_serial.h
	$(CXX) $(CXXFLAGS) -DNICR_SERIAL_VERIFY $< -o $@
serial-check: serial_verify
	./serial_verify --self-test

emulated_mpi_test: NiCr2D_grains.cpp tests/mpi_threads.h tests/emulated_mpi_driver.cpp
	$(CXX) $(CXXFLAGS) -pthread tests/emulated_mpi_driver.cpp -o $@
emulated-check: emulated_mpi_test
	./emulated_mpi_test 4 --self-test

clean:
	rm -f nicr2d_mpi serial_verify emulated_mpi_test
