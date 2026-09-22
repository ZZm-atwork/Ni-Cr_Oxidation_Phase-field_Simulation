# Ni–Cr oxidation: MPI phase-field solvers

C++17 research solvers for oxygen transport, chromium redistribution, oxide
formation, and grain-boundary-assisted transport in Ni–Cr alloys. The 2D and 3D
implementations couple composition fields to an oxide phase field and optionally
evolve metal-grain order parameters.

## Implementations

| Directory | Domain decomposition | Purpose |
| --- | --- | --- |
| [mpi2d](mpi2d/README.md) | X slabs on a 2D mesh | 2D oxidation and optional grain evolution |
| [mpi3d-slab](mpi3d-slab/README.md) | X slabs on a 3D mesh | Optimized nucleation searches, frozen-eta caching, compressed VTI output |
| [mpi3d-blocks](mpi3d-blocks/README.md) | 3D Cartesian blocks | Configurable process grid and six-face halo exchange |

The two 3D versions are independent implementations. The block version does not
include the latest slab optimizations; this repository makes no claim that one
is faster across all problem sizes. Each directory contains its own complete
source, Makefile, usage guide, and verification record.

## Build and run

Requirements: C++17, an MPI compiler/launcher, and Make. The optimized 3D slab
version also requires zlib headers and libraries. Python 3 is used by the
existing comparison scripts. ParaView can display the saved fields.

From the repository root:

```sh
make -C mpi2d
make -C mpi3d-slab
make -C mpi3d-blocks
mpirun -np 4 ./mpi2d/nicr2d_mpi --self-test
mpirun -np 4 ./mpi3d-slab/nicr3d_mpi --self-test
mpirun -np 4 ./mpi3d-blocks/nicr3d_mpi --self-test
```

Use the MPI compiler and launcher supplied by the same installation. On a
cluster, run inside an allocation with enough CPU cores. See each implementation
README for full inputs, process grids, physical assumptions, and examples.

A small initialization example for the optimized 3D slab version:

```sh
mpirun -np 2 ./mpi3d-slab/nicr3d_mpi \
  --nx 21 --ny 21 --nz 21 --steps 0 --out run_3d_init
```

This writes initial fields only. For a short evolution example, follow the
implementation guide and choose a new output directory for each run.

## Model and output

- Oxygen and chromium transport coupled to oxide phase-field evolution.
- Single-crystal, static Voronoi-grain, and optional evolving-grain modes.
- Center-seed, transport-only, and Ksp-based insertion controls.
- Oxygen boundary/supply controls, mass-budget diagnostics, and grain statistics.
- Legacy VTK output; the optimized slab version also supports compressed VTI.

Model details are in [2D MODEL.md](mpi2d/MODEL.md) and
[3D block MODEL.md](mpi3d-blocks/MODEL.md). Implementation defaults may differ;
use each source and recorded run parameters as the authority.

## Verification and limitations

During repository preparation on 2026-09-21, all three variants compiled and
passed their existing single-process self-tests. Native MPI was unavailable in
that environment. Single-process and emulated checks do not establish real
multi-process MPI correctness or cluster scaling.

Historical checks remain in each implementation's `VERIFICATION.md` and
`tests/logs/`. Physical calibration, mesh/time convergence, and production-scale
performance require separate evaluation. In particular, grain coefficients are
numerical reference values and root gathers remain a scaling constraint.

The optimized slab VTI snapshots are visualization output, not complete restart
checkpoints. See its guide for saved fields and precision controls.

## Source provenance

[IMPORT_NOTES.md](IMPORT_NOTES.md) records the imported snapshots. Solver source
files are byte-identical to those packages. No release history or previous
commits have been reconstructed. No new license has been assigned to this
repository; the imported Ni–Cr packages did not include a license file.
