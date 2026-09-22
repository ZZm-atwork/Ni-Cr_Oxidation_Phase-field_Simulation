# Verification — 3D Cartesian MPI solver

## Scope and execution environment

The packaged source is `NiCr3D_grains.cpp`, version `mpi-eta-3d-cartesian-v2`.

Source SHA-256: `bbe92fe1f197e7b6436370a070b1d751bc5bd394e3e6136567599043d8267a9f`

GCC 14.2.0 and Clang 17.0.0 compiled the source through its explicitly separate
verification backends. The multi-rank checks used a thread-based communication
harness, **not a native MPI runtime**. Native MPI headers, compiler wrapper and
launcher were unavailable in this environment. No native MPI build/run,
multi-node run, or Sol scaling benchmark is claimed.

The harness checks independent communicator contexts, Cartesian neighbor
mapping, committed subarray datatype selection, message lengths/order,
collective signatures, and MPI resource lifetimes. Array-index assertions
were enabled. Manufactured global-coordinate fields independently verify
all six faces; unused corner padding is initialized with NaN and remains unused.

## Results

| Check | Result |
|---|---|
| GCC and Clang verification builds | Passed, with `-Wall -Wextra -Werror -pedantic` |
| Built-in numerical suite | Passed |
| Automatic process-grid self-tests | Passed at 1, 2, 4, 8, 16, 27 and 64 emulated ranks |
| All self-test layouts | 14 configurations passed |
| Explicit and partially specified grids | Passed |
| Output/ownership regression | 130 cases passed on 13 layouts |
| Comparison to the supplied slab solver | 10 single-rank physics cases matched |
| Address/undefined-behavior sanitizers | Passed at 8 emulated ranks, with leak detection enabled |
| CLI rejection cases | 6 invalid configurations rejected before output creation |
| More ranks than X cells | Passed: 3 × 5 × 5 mesh on 16 ranks |

The self-tests include 2 × 2 × 2, 4 × 2 × 2, 2 × 2 × 4, 3 × 3 × 3 and
4 × 4 × 4 process grids, along with X/Y/Z slabs, a YZ pencil arrangement,
and a partially fixed grid. Thin test meshes exercise one-cell local
extents, including one owned cell per rank in all three directions.

## Numerical and communication coverage

The built-in suite checks seven-point Laplacians, six-face conservative
transport, O/Cr inventories, all three X boundary modes, periodic Y/Z faces,
the oxygen transport switch, and the retained Ck option. It compares both
owned cells and face halos to an independent single-rank model.

Grain tests cover static Voronoi masks, evolving eta fields, frozen eta,
inlet anchoring, eta/phi free-energy derivatives and grain-energy descent.
Nucleation tests cover exact center spheres, hard spheres crossing internal
block junctions and both periodic seams, ordered Ksp insertion, exclusion,
and eta clearing after scatter. A deliberately invalid trial on a non-root
rank verifies collective rejection and rollback of the inlet/budget.

The 130 output cases use a 31 × 17 × 15 mesh and 30 PDE steps, with ten
physics configurations per layout: fixed-grain mu boundary, evolving-eta mu
boundary, inlet anchoring, dynamic/fixed center nuclei, oxygen off, legacy
Ck, frozen eta, evolving-eta concentration boundary, and single crystal.
These test meshes are intentionally nondivisible by many process grids.

VTK files, initial grain statistics, grain history, and grain diagnostics
match **byte-for-byte** across layouts. Inventory diagnostics are compared
using `abs(a-b) <= 1e-10*(1+abs(a))` to allow reduction-order roundoff.
`mpi_layout.csv` is audited for exact coverage, nonoverlap, global offsets,
local dimensions and preservation of requested process-grid axes.

The prior-solver reference was built from the supplied 3D slab source with
its original single-process verification backend. Its source SHA-256 is
`1835a9aea3c2745c177ad3d3a44d929822efd9c10aaa5ffd9ed32ab15602d6ed`. The same ten configurations reproduce
its physical fields and grain outputs byte-for-byte. The prior source is
not duplicated in this package.

## Reproduce the checks

For native MPI, run inside an appropriate allocation:

```bash
make
mpirun -np 8 ./nicr3d_mpi --self-test --px 2 --py 2 --pz 2
mpirun -np 16 ./nicr3d_mpi --self-test --px 4 --py 2 --pz 2
make verify-mpi RANKS="1 2 4 8 16"
```

The built-in self-test uses small meshes and applies the selected process
grid; it does not run the user's production simulation.

For the test-only communication harness used here:

```bash
make emulated_mpi_test
./emulated_mpi_test 64 --self-test
python3 tests/verify_mpi.py --binary ./emulated_mpi_test \
  --emulated --ranks 1 2 4 8 16
```

The verifier's optional `--reference-binary` argument accepts a previous
solver executable to run directly in single-process mode. Its
`--skip-self-tests` flag runs only the output/ownership comparisons.

Sanitizer build and execution:

```bash
clang++ -O1 -g -std=c++17 -DNICR_INDEX_CHECK -pthread \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  tests/emulated_mpi_driver.cpp -o blocks_sanitized
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 ./blocks_sanitized 8 --self-test
```

Raw self-test/compiler logs, all output-regression invocation logs, field
checksums, and machine-readable summaries are under `tests/logs/`. Large
VTK test datasets and compiled binaries are omitted from the distribution.

## Remaining validation

Native MPI compilation/execution and cluster performance still need
verification on the target MPI installation. The comparisons establish
short-run implementation consistency, not long-time stability, timestep
convergence or new material calibration. Root still gathers a full-domain
snapshot for ordered Ksp scans, output and snapshot-based diagnostics;
those operations retain a root-memory and I/O scaling limit.
