MPICXX ?= mpic++
CXX = c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic
MPIEXEC ?= mpirun
NP ?= 8
RANKS ?= 1 2 4 8
GRID ?=

.PHONY: all test verify-mpi serial-check emulated-check clean
all: nicr3d_mpi

# Production: one self-contained source, linked to a real MPI installation.
nicr3d_mpi: NiCr3D_grains.cpp
	$(MPICXX) $(CXXFLAGS) $< -o $@

test: nicr3d_mpi
	$(MPIEXEC) -np $(NP) ./nicr3d_mpi --self-test $(GRID)

# Run inside an allocation with enough ranks for the requested RANKS list.
verify-mpi: nicr3d_mpi
	python3 tests/verify_mpi.py --binary ./nicr3d_mpi --launcher "$(MPIEXEC)" --ranks $(RANKS)

# Verification-only fallbacks; do NOT launch these as native MPI executables.
serial_verify: NiCr3D_grains.cpp tests/mpi_serial.h tests/mpi_threads.h
	$(CXX) $(CXXFLAGS) -DNICR_INDEX_CHECK -DNICR_SERIAL_VERIFY -pthread $< -o $@
serial-check: serial_verify
	./serial_verify --self-test

emulated_mpi_test: NiCr3D_grains.cpp tests/mpi_threads.h tests/emulated_mpi_driver.cpp
	$(CXX) $(CXXFLAGS) -DNICR_INDEX_CHECK -pthread tests/emulated_mpi_driver.cpp -o $@
emulated-check: emulated_mpi_test
	./emulated_mpi_test $(NP) --self-test $(GRID)

clean:
	rm -f nicr3d_mpi serial_verify emulated_mpi_test
