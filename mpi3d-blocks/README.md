# Ni–Cr oxidation: 3D Cartesian MPI solver

The solver evolves oxygen, chromium, oxide fraction and optional metal-grain
order parameters on a three-dimensional mesh. MPI partitions the mesh into
`Px × Py × Pz` rectangular blocks. Each rank stores only its own block and a
one-cell halo. X is nonperiodic; Y and Z are periodic.

The production program is the single source file `NiCr3D_grains.cpp`.
`MODEL.md` describes the equations and numbered source sections.
`VERIFICATION.md` records the checks performed on this package.

## 1. Build

Use a C++17 compiler and an MPI development environment:

```bash
make
```

Equivalent direct compilation:

```bash
mpic++ -O3 -std=c++17 NiCr3D_grains.cpp -o nicr3d_mpi
```

A different MPI compiler wrapper can be selected with `make MPICXX=mpicxx`.
Use the compiler wrapper and launcher from the same MPI installation.
No HDF5 or GSL library is needed by this source.

```bash
./nicr3d_mpi --help
./nicr3d_mpi --version
```

## 2. Choose the process grid

Omit the process-grid options for automatic selection:

```bash
mpirun -np 16 ./nicr3d_mpi \
  --nx 61 --ny 61 --nz 61 \
  --steps 0 --out initialize_auto
```

This is an initialization-only example. Existing physics/run arguments can be
used unchanged; the MPI grid is an additional set of controls.

For an explicit `4 × 2 × 2` arrangement:

```bash
mpirun -np 16 ./nicr3d_mpi \
  --px 4 --py 2 --pz 2 \
  --nx 61 --ny 61 --nz 61 \
  --steps 0 --out initialize_4x2x2
```

The axes are always **X, Y, Z**, in that order. The product must equal the
number of MPI ranks. Blocks need not be literal cubes.

| MPI ranks | Example process grid | Flags |
|---:|---:|---|
| 8 | 2 × 2 × 2 | `--px 2 --py 2 --pz 2` |
| 16 | 4 × 2 × 2 | `--px 4 --py 2 --pz 2` |
| 27 | 3 × 3 × 3 | `--px 3 --py 3 --pz 3` |
| 64 | 4 × 4 × 4 | `--px 4 --py 4 --pz 4` |

An omitted axis or a value of zero is automatic. Partial specification is
supported; for example, `mpirun -np 16 ... --py 2` fixes only Y. Automatic
selection uses `MPI_Dims_create`. If its factorization would create empty
blocks on a thin mesh, the solver searches for a fitting factorization while
preserving the fixed axes. Otherwise, it retains MPI's selected grid.
Automatic selection is a process-count balance, not a performance optimizer
for a highly elongated physical mesh.

Uneven partitions are supported independently in all three directions. Along
an axis, the first `N % P` process coordinates receive `N / P + 1` cells;
the rest receive `N / P`. Each axis must have at least one owned cell per
process coordinate. For `61³` on `4 × 2 × 2`, the local dimensions are
15–16 by 30–31 by 30–31 cells. There is no padded enlargement of the physical
mesh and no duplicated ownership at block boundaries.

At startup the chosen process grid is printed. Every output directory also
contains `mpi_layout.csv`, with each rank's Cartesian coordinates, inclusive
one-based global bounds, local dimensions, and owned-cell count.

An X-slab run is still available explicitly: use `--px N --py 1 --pz 1`
with `-np N`. Y slabs, Z slabs, and two-dimensional pencil arrangements are
also supported.

## 3. Run examples

### Single center nucleus with evolving metal grains

```bash
mpirun -np 16 ./nicr3d_mpi \
  --px 4 --py 2 --pz 2 \
  --nx 61 --ny 61 --nz 61 --dx 1e-8 \
  --grains on --grain-evolution on --eta-output all \
  --grain-diameter-grid 50 --seed 2001 \
  --nucleation center --radius-grid 5.5 \
  --oxygen-diffusion on --bc closed --ck off --bounds legacy \
  --dt 1e-12 --steps 1000 --output-every 1000 \
  --out center_blocks
```

This command specifies one initial sphere and closed boundaries. It is a
short numerical run, not a calibrated oxidation-duration example. `center`
disables subsequent Ksp insertion.

### Oxygen supply with Ksp insertion

```bash
mpirun -np 16 ./nicr3d_mpi \
  --px 4 --py 2 --pz 2 \
  --nx 61 --ny 61 --nz 61 --dx 1e-8 \
  --grains on --grain-evolution on --eta-output all \
  --grain-diameter-grid 50 --seed 2001 \
  --nucleation ksp --radius-grid 5.5 \
  --oxygen-diffusion on --bc mu --ck legacy --bounds legacy \
  --gb-factor-o 28.2 --gb-factor-cr 563 \
  --dt 1e-12 --steps 1000 --output-every 1000 \
  --out oxygen_blocks
```

The default Ksp scan period is `1e-7 s`. At `dt=1e-12`, the first scan is at
step 100,000; the 1,000-step example tests startup and transport before that
scan. A scan inserts oxide only where the model's eligibility criteria hold.
Use a new output directory for each run. A nonempty directory is rejected.

To keep eta fields for visualization while freezing their PDE evolution, add
`--eta-L 0`. Hard oxide insertion still sets eta to zero inside inserted oxide.
To use the static Voronoi/GB model, use `--grain-evolution off`. To disable GB
structure altogether, use `--grains off`.

## 4. Inputs

Run `--help` for the complete option syntax.

| Controls | Default | Meaning |
|---|---|---|
| `--nx`, `--ny`, `--nz` | 151, 101, 101 | Global cell counts, each at least 3 |
| `--px`, `--py`, `--pz` | 0, 0, 0 | MPI process-grid axes; zero is automatic |
| `--dx` | 1e-8 m | Sets dx = dy = dz; cubic cells |
| `--dt` | 2e-10 s | Explicit timestep |
| `--steps`, `--output-every` | 0, 5000 | Accepted steps requested; output interval |
| `--grains` | on | Voronoi grain structure and GB transport |
| `--grain-evolution` | off | Enable eta fields, evolution, and reciprocal phi coupling |
| `--grain-diameter-grid` | 50 | Target initial volume-equivalent grain diameter |
| `--gb-width-grid` | 4 | Fixed-mask full width at half maximum |
| `--gb-factor-o`, `--gb-factor-cr` | 28.2, 563 | GB mobility enhancement factors |
| `--seed` | 20260909 | Voronoi random seed |
| `--left-face-sites` | 0 | Relocate existing sites onto the inlet face |
| `--oxygen-diffusion` | on | O transport and continuing boundary supply |
| `--bc` | concentration | `concentration`, `mu`, or `closed` |
| `--mu-res`, `--surface-c-ref` | 364688, 0.0023 | Left reservoir potential and face-mobility reference |
| `--nucleation` | ksp | `ksp`, `center`, or `off` |
| `--radius-grid` | 5.5 | Oxide sphere radius in grid spacings |
| `--nucleation-period` | 1e-7 s | Physical interval between Ksp scans |
| `--eta-W`, `--eta-kappa`, `--eta-L` | 1e4, 0, 0.2 | Eta potential, gradient, and kinetic coefficients |
| `--eta-oxide-penalty` | -1 | -1 selects eta-W |
| `--eta-output`, `--max-eta-fields` | all, 512 | Individual/summary output and field-count limit |
| `--ck`, `--bounds` | legacy, stop | Concentration adjustment; trial bounds handling |
| `--out` | nicr3d_grains_output | New or empty output directory |

`--left-edge-sites` is an alias for `--left-face-sites`.
`--gb-factor value` sets both GB enhancement factors.
`--eta-kappa 0` selects the mask-width-based gradient coefficient rather than
turning off its gradient energy. `--eta-output summary` retains the summary
fields but omits individual `eta_g` fields from VTK output.

The retained matrix mobility prefactors are `2.21e-16` for O and
`3.5436e-20` for Cr. Oxide prefactors are `2.21e-18` and `3.5436e-21`.
These are the supplied 18 wt.% Cr setup, not an interpolated composition
series and not diffusivities. The local `M_O` and `M_Cr` fields also include
concentration, phase interpolation, and GB enhancement. Material constants
are in source section [1]. Changing decomposition does not change them.

## 5. Boundaries, communication and output

Only blocks touching global X=0 apply the oxygen inlet. The right X face
retains the no-flux condition. Internal rank boundaries are never treated as
physical boundaries. Y and Z periodicity applies across the entire domain.

Each exchange fills six face halos with committed MPI subarray datatypes.
Edge and corner padding is not exchanged or used. Concentrations/phi/eta
are exchanged before computing local derivatives; chemical potentials are
exchanged again before the flux update. This preserves the composed
Cahn–Hilliard operator across rank boundaries.

Outputs retain the single-file, voxel-centered, X-fastest VTK format:

| File | Contents |
|---|---|
| `fields_0.vtk`, `fields_N.vtk` | phi, O/Cr/Ni, grain and GB fields, chemical potentials, local mobilities; eta output when enabled |
| `diagnostics.csv` | Inventories, boundary/seed/Ck/clipping contributions, extrema and oxide volume |
| `grain_diagnostics.csv` | Dynamic-grain extrema, active grain count, integrated GB indicator and grain energy |
| `grain_history.csv` | Per-grain metal volume over time |
| `grains.csv` | Initial sites, voxel counts and volume-equivalent diameters |
| `parameters.txt` | Physical inputs, actual MPI grid and storage estimates |
| `mpi_layout.csv` | Per-rank block coordinates, global bounds and sizes |
| `STOPPED.txt`, `state_at_failure.vtk` | Diagnostic output if a timestep fails |

Grain diagnostic/history files are written when eta fields are active.
The VTK output remains readable as a file series in ParaView. This source
does not implement a checkpoint/restart reader; VTK files are outputs, not
restart checkpoints.

**Root-memory and scaling limit:** stepping is block-distributed, but ordered
Ksp scans, VTK writing, and snapshot-based diagnostics still gather onto
rank zero. Root holds an additional full-domain model and a packed field
buffer for those operations. This preserves global nucleation scan order
and the existing output convention; it does not make I/O or seeding fully
distributed. `parameters.txt` estimates the root's extra field storage.
The inherited int-count gather limit is checked before allocation. Benchmark
large runs on the target machine rather than assuming a particular speedup.

## 6. Verification on your MPI environment

Build and run the built-in tests inside an allocation:

```bash
make
mpirun -np 8 ./nicr3d_mpi --self-test --px 2 --py 2 --pz 2
mpirun -np 16 ./nicr3d_mpi --self-test --px 4 --py 2 --pz 2
```

The self-test uses small test meshes and applies the selected MPI layout.
It checks face values, transport, grain fields, nucleation, thin blocks,
periodic seams, global reconstruction, and collective trial rejection.

The output-regression script compares complete fields and grain outputs
across automatic/manual layouts and audits every rank's cell ownership:

```bash
make verify-mpi RANKS="1 2 4 8 16"
```

This requires Python 3 and a working MPI launcher with enough allocated slots.
The default list is `1 2 4 8`. The launcher is configurable, for example
`make verify-mpi MPIEXEC="mpirun --oversubscribe"` on a suitable local host.
On a scheduler-managed cluster, use the site's MPI launch configuration and
an allocation large enough for the largest requested rank count.

When native MPI is absent, the explicitly separate fallback checks are:

```bash
make serial-check
make emulated-check NP=8 GRID="--px 2 --py 2 --pz 2"
```

These use a single-process/thread test harness, not a native MPI library.
They cannot validate an MPI ABI, network, launcher, or performance. Do not
launch `serial_verify` or `emulated_mpi_test` as production MPI solvers.

## MPI API references

The implementation uses the portable MPI C interface from C++17:

- MPI_Cart_create: https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Cart_create.3.html
- MPI_Dims_create: https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Dims_create.3.html
- MPI_Cart_shift: https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Cart_shift.3.html
- MPI_Type_create_subarray: https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Type_create_subarray.3.html
