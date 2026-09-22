CXX = mpic++
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS += -lz
RANKS ?= 1 2 4
MPIEXEC ?= mpirun

.PHONY: all clean verify-serial verify-emulated verify-mpi
all: nicr3d_mpi

nicr3d_mpi: NiCr3D_grains.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

verify-serial:
	g++ -O2 -std=c++17 -DNICR_SERIAL_VERIFY NiCr3D_grains.cpp -o tests/serial_check -lz
	./tests/serial_check --self-test
	g++ -O2 -std=c++17 -DNICR_SERIAL_VERIFY tests/optimization_tests.cpp -o tests/optimization_check -lz
	./tests/optimization_check check
	python3 tests/verify.py --backend serial

verify-emulated:
	g++ -O2 -std=c++17 -pthread tests/emulated_mpi_driver.cpp -o tests/emulated_check -lz
	@for n in $(RANKS); do ./tests/emulated_check $$n --self-test || exit 1; done
	python3 tests/verify.py --backend emulated --ranks $(RANKS)

# Run within a suitable MPI allocation. This target invokes native MPI.
verify-mpi: nicr3d_mpi
	@for n in $(RANKS); do $(MPIEXEC) -np $$n ./nicr3d_mpi --self-test || exit 1; done
	python3 tests/verify.py --backend native --cxx "$(CXX)" --launcher "$(MPIEXEC)" --ranks $(RANKS)

clean:
	rm -f nicr3d_mpi tests/serial_check tests/emulated_check tests/optimization_check
