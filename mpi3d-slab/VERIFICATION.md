# Verification and local performance

## Scope

Baseline: the shared seven-field X-slab `NiCr3D_grains.cpp` with weighted `grain_sum`, `GB_mask` and the earlier internal-array cleanup.

Baseline SHA-256:

`12c980043086ea1ee1b11d4bb442a586aa28a3650813318e87fbd3217b47d073`

The optimized source preserves the Physics coefficients, chemical free energies, concentration-weighted mobility functions and local grain-energy derivatives byte-for-byte. The search predicates and accepted update order are preserved. Float32 output is intentionally rounded; internal physical states are compared before any output conversion.

## Numerical regression

48 configurations cover:

- Single crystal, fixed Voronoi grains, evolving eta and frozen eta.
- Closed, chemical-potential and concentration boundary conditions.
- Oxygen transport on/off and legacy Ck on/off.

Each configuration uses a 25 x 15 x 13 mesh, smooth initial fields followed by sphere insertion across X-slab and Y/Z periodic boundaries, 24 PDE steps, scheduled ordered Ksp scans and a deliberate eta edit on one rank with collective cache invalidation.

For all 48 configurations, comparisons passed for:

| Comparison | Result |
|---|---|
| Optimized serial vs exact baseline: initial and final double physical-state dumps | Bit-for-bit identical |
| Optimized four-emulated-rank vs baseline serial physical-state dumps | Bit-for-bit identical |
| Frozen cache on vs forced rebuild | Bit-for-bit identical physical states |
| All seven legacy ASCII array payloads | Identical to baseline |
| VTI arrays vs expected output conversion | Exact Float32 cast or exact original Float64, as selected |
| Dimensions, origin and spacing | Unchanged |
| Serial diagnostic and grain CSV files | Byte-identical to baseline |

Physical-state dumps include O, Cr, phi, both chemical potentials, both mobilities, GB mask, eta_s2 where allocated, and every individual eta field. Across MPI rank counts, floating-point scalar reductions can depend on summation order; the comparison does not require diagnostic CSV bytes to match across layouts.

The portable verification tool was also run through all 48 cases in serial and four-rank-emulated modes. See `tests/logs/verify_serial.json`, `verify_emulated.json` and the corresponding text logs. Test code is supplied.

## Geometry and file-format checks

- 5,016 periodic interval cases: sorted coverage and no duplicate wrapped cells.
- 50,400 exclusion decisions: exact agreement with the original full-plane predicate, including strict cutoff edges, short periodic axes, odd/even axes and neighborhoods spanning the full period.
- 432 sphere insertions: identical voxel updates, eta consumption and mass/event budgets, preserving update order.
- 24 VTI fixtures: Float32/Float64, zlib levels 0/1/6/9, partial compression blocks, exact full blocks and multiple blocks. All loaded with **VTK 9.6.2**, with byte-exact array recovery against the expected typed payloads.
- Float32 overflow rejected; unfinished temporary VTI removed, without publishing a partial final snapshot.
- Patch applied both to the exact baseline and to a copy with deliberately altered W/ko/kc/kp values; those edited values were preserved.

## Builds and runtime checks

GCC and Clang builds passed with `-Wall -Wextra`. The built-in tests passed in the serial verification backend and with 2, 4 and 8 thread-emulated ranks. A Clang AddressSanitizer/UndefinedBehaviorSanitizer build passed the built-in tests with leak detection enabled.

**Native MPI compilation/execution and the ParaView GUI were not tested in this container.** Thread emulation exercises message matching, collective order and data exchange in one address space; it is not a network/cluster benchmark. VTI files were validated with the actual Python VTK reader. Before production, use `make verify-mpi` inside a suitable MPI allocation.

## Measured performance — controlled local tests

All figures below are local single-process tests using the serial verification backend, not timings from Sol. Medians are from three trials. Raw measurements are in `tests/logs/benchmark_*.csv`. Source is in `tests/optimization_tests.cpp`.

### Seven-field snapshot writer

A 151 x 101 x 101 mesh with smooth synthetic fields and frozen eta was written repeatedly. Each format saves the same seven arrays from the same state; no physical timestepping occurs between formats. Timings include the writer's refresh and file close, not initialization. Data were written to the container filesystem, without forcing a physical-disk `fsync`; filesystem caches and machine load affect timings.

| Format | Bytes | Decimal MB | Median seconds |
|---|---:|---:|---:|
| Legacy ASCII Float64 | 214,456,140 | 214.456 | 3.154730 |
| VTI Float64, zlib level 1 | 31,162,450 | 31.162 | 0.973897 |
| VTI Float32, zlib level 1 | 12,631,233 | 12.631 | 0.594769 |

For this test only, default VTI Float32 was about **17.0 times smaller** (94.1% fewer bytes) and the write operation was about **5.30 times faster**. Float64 compressed output preserved the double values and was about 6.88 times smaller. Compression ratios depend on the actual fields and are not a fixed production guarantee.

### Exclusion-search microbenchmark

2,000 identical candidate locations on a 151 x 101 x 101 synthetic oxide distribution, radius 5.5 grids, exclusion gap 6 grids. Both methods accepted 906 candidates.

| Search | Median seconds |
|---|---:|
| Original full-YZ-plane scan | 0.138600 |
| Bounded periodic neighborhood | 0.0164632 |

This is about **8.42 times faster for `valid_nucleus()` in this test**, not an 8.42-times speedup of the complete solver or the entire nucleation event. Different oxide distributions alter early exits and the relative benefit.

### Frozen-eta PDE benchmark

61 x 41 x 31 mesh, 14 eta fields, `eta_L=0`, 100 timesteps per trial. No nucleation or output is included in the timed segment. Cached and uncached states were checked for equality after each segment.

| Mode | Median seconds per 100 steps |
|---|---:|
| Forced eta halo/mask refresh | 0.711561 |
| Frozen eta cache enabled | 0.604872 |

Approximately **15.0% less elapsed time** (1.18 times throughput) in this local PDE test. This saving is specific to frozen eta; moving eta still requires refreshes. Native MPI communication savings require a cluster measurement.

## Reproducing the microbenchmarks

```bash
g++ -O3 -std=c++17 -DNICR_SERIAL_VERIFY tests/optimization_tests.cpp -o tests/optimization_check -lz
./tests/optimization_check check
./tests/optimization_check search
./tests/optimization_check cache
./tests/optimization_check io /path/to/new/benchmark_output
```

For native MPI, run the supplied regression/self-tests and then repeat the user's saved-state timing experiment. No production checkpoint was available in this container, so no end-to-end production speedup is claimed.
