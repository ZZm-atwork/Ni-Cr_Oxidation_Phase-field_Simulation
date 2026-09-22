# Verification performed for mpi-eta-v1

## Scope

Numerical implementation and short regression checks, not experimental
validation of Ni–Cr grain mobility, interfacial energy or oxidation kinetics.
No long production oxidation run was performed.

**A native MPI runtime/compiler wrapper was not available in the authoring
environment. Native multi-process MPI compilation/execution is not claimed.**
The production source retains the MPI API and uses additional exchanges/gathers
for eta. `make test NP=4` and `make verify-mpi` are the supplied native acceptance
checks to run on the target MPI installation.

Compiler checks actually run:

- GCC 14.2.0, C++17, `-O2` and `-O3 -Wall -Wextra -pedantic`, single-process
  verification backend: passed.
- Clang 17.0.0, C++17, `-O2 -Wall -Wextra -pedantic`, single-process
  verification backend: passed.
- Clang 17.0.0, `-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer`,
  single-process self-tests with leak detection: passed.
- Thread-based communication emulation with 2, 4 and 8 ranks: passed.
  This validates the tested communication logic in one address space; it is not
  a test of a vendor MPI library, launch environment or inter-node communication.

The logs identify their execution backend. The optional test headers are not
included in a normal mpic++ build. No authoring-machine executable is packaged.

## Grain physics / numerical checks

1. One eta field per initial Voronoi site, smooth normalized initialization,
   reproducible geometry, and grain-off override.
2. Finite-difference comparison of the **discrete grain energy** with the
   implemented eta and reciprocal phi forces, including physical X and periodic
   Y boundaries. Largest scaled error: approximately 1.84e-10 in the GCC test.
3. Analytic planar bicrystal profile: maximum force residuals approximately
   0.00512028, 0.00132059, and 0.000331574 at successive dx=1, 0.5, 0.25.
   This approaches second-order spatial convergence.
4. Planar eta-mask integral matches the original Gaussian-mask integral:
   4.257868... in that dimensionless reference test.
5. An isolated circular metal grain actually shrinks. The integral of its eta
   changes from 462.725 to 437.842 over the dimensionless test interval. The added
   grain energy decreases from 49.9466 to 48.5495, with a nonincreasing value at
   every checked step. The dynamic mask changes with the migrating boundary.
6. Residual metal eta decays in a uniform oxide; one metal/oxide interface does
   not create a false substrate GB.
7. Center and ordered hard seeds clear every metal eta in exactly their original
   disk, including disks crossing the periodic seam.
8. Eta trial failure rejects the other PDE trials as well. Eta is not clipped.

These are implementation checks using controlled test states, not inferred
experimental grain sizes or material coefficients.

## Regression against the uploaded MPI source

Baseline source SHA256:

    d4845cbd918de2e7f672f0f39e3b2dd2368b2a77d43d6536d704e7e71c7d5818

The complete `Physics` block, `fmetal`, `foxide`, and `mobility` function text
were checked identical to this baseline. The new reciprocal force is added in
the phi update only when grain evolution is active.

With `--grain-evolution off`, the following short cases produced byte-identical
VTK, initial-grain CSV and original diagnostics CSV to the baseline when both
were compiled with the same single-process verification shim:

- Ksp/mu-boundary startup.
- Inlet-anchored Voronoi startup.
- Center hard seed.
- Oxygen transport off.
- Grain structure off with concentration boundary.

The shim changes only the MPI calls to their one-process behavior; it is not a
separate implementation of the numerical equations.

## Communication-logic checks

At 2, 4 and 8 emulated ranks, built-in tests exercised:

- Uneven X slabs and slabs only one column wide.
- Every eta halo, coupled phi force, and eta-derived transport mask.
- Root snapshots containing evolved eta (not reinitialized values).
- Gather/scatter during ordered Ksp insertion, including simultaneous candidates
  and exclusion across process boundaries and the periodic Y seam.
- All original boundary modes, oxygen switch and collective failure handling.

Additional 1-versus-4 emulated-rank output regressions were run for mu startup,
anchored startup, center oxide and oxygen-off center cases. VTK fields, grain
history and grain diagnostics matched byte-for-byte. Maximum reported difference
in the original reduced diagnostic quantities was approximately 1.29e-22.

## Short oxidation-path checks

The README's 10,000-step Ksp command was run using the single-process backend,
with dt=2e-10, legacy Ck and legacy bounds. It completed, but formed no nuclei in
that short interval. Final max_O was approximately 0.000406977. This is a startup
execution check, not a post-nucleation stability test. The separate built-in
supersaturated-state test exercises actual Ksp insertion and coupled updates.

The included `example_center/` was run for 500 steps at dt=1e-12 with Ck off,
legacy concentration bounds and a center hard oxide. It illustrates output
fields only, not a critical-radius measurement.

The inherited concentration update can still undershoot and legacy clipping is
still nonconservative. This grain extension does not fix that separate issue.
The grain/coupling energy by itself need not decrease during oxidation.
