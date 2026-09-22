# Ni–Cr oxidation with evolving metal grains — MPI

This solver evolves oxygen, chromium, oxide phase `phi`, and optionally one
Allen–Cahn order parameter `eta_i` per initial Voronoi grain. The grain boundaries
can migrate; their evolving overlap supplies the fast-diffusion network. Metal
order is suppressed where oxide forms. There is no separate oxide-grain model.

**Activate the new model with `--grain-evolution on`.** The default is `off`, which
retains the uploaded solver's fixed-grain evolution. Both modes use the same
original oxide thermodynamic coefficients and concentration-update formulas.

## 1. Build and check MPI

Compile on the machine where the program will run, using that machine's MPI:

```sh
make
mpirun -np 4 ./nicr2d_mpi --self-test
```

Equivalent compile command:

```sh
mpic++ -O3 -std=c++17 -Wall -Wextra -pedantic NiCr2D_grains.cpp -o nicr2d_mpi
```

Run the broader native MPI output comparison with `make verify-mpi`. On a cluster,
use its allocated-job launcher. Do not use more ranks than X columns or allocated
cores. This package contains source, not a platform-specific executable.

## 2. Run an oxidation case with evolving grains

```sh
mpirun -np 4 ./nicr2d_mpi \
  --grain-evolution on --eta-output all \
  --grains on --seed 2001 \
  --nx 61 --ny 61 --grain-diameter-grid 50 \
  --nucleation ksp --radius-grid 5.5 \
  --oxygen-diffusion on \
  --bc mu --ck legacy --bounds legacy \
  --gb-factor-o 28.2 --gb-factor-cr 563 \
  --dt 2e-10 --steps 10000 --output-every 1000 \
  --out eta_oxidation
```

Use a new output directory for every run. Change `--steps` to your desired run
length; `--steps 0` writes initialization only. Progress is printed every 1,000
accepted steps. The example above is a startup run, not a validated production
condition or an assurance that an oxide appears within 10,000 steps.

To partition the left oxygen inlet between two grains, additionally use
`--left-edge-sites 2`. This relocates existing Voronoi sites without adding grains.
The original area/diameter grain-count rule and random realization are retained.
Anchoring applies only to initialization; the resulting boundaries are free to
move. With 61 x 61 cells and diameter 50 grids the original rule gives two sites.

## 3. Controls

| Argument | Meaning |
|---|---|
| `--grain-evolution on` | Evolving eta, eta-derived GB transport, reciprocal oxide/grain coupling. |
| `--grain-evolution off` | Original static Voronoi/Gaussian GB model; no eta PDEs. |
| `--grains off` | Original single-crystal model; overrides grain evolution. |
| `--eta-output all` | Summary fields plus `eta_1`, `eta_2`, etc. Default. |
| `--eta-output summary` | Summary grain fields only; smaller VTK files. |
| `--eta-W 10000` | New grain free-energy weight, independent of oxide W. |
| `--eta-kappa 0` | Auto-select grain gradient coefficient to match the original integrated GB width. A positive value overrides it. |
| `--eta-L 0.2` | Grain order-parameter mobility, independent of oxide L. Zero disables the eta PDE update; hard insertion still clears eta. This is NOT the original fixed-grain model. |
| `--eta-oxide-penalty -1` | Auto-use eta-W for the quadratic oxide-suppression coefficient. A nonnegative value overrides it. |
| `--max-eta-fields 512` | Memory guard on the number of grain fields. |
| `--nucleation ksp` | Original automatic hard-seed insertion. |
| `--nucleation center` | One hard oxide at the box center at t=0; no later insertion. |
| `--nucleation off` | No oxide insertion; useful for transport/grain-growth comparisons. |
| `--oxygen-diffusion on/off` | Existing O transport/boundary-supply switch. Cr and phi still evolve. |

The remaining original options are listed by `./nicr2d_mpi --help`.

`--eta-L 0` must not be used as a replacement for `--grain-evolution off`: the
former still uses the eta GB profile and adds a grain contribution to the phi
force. Compare against `--grain-evolution off` for the original baseline.

## 4. Visualize in ParaView

Open the `fields_*.vtk` series. In dynamic mode:

- `eta_sum_sq`: close to 1 in grain interiors, depressed at grain boundaries,
  approaching 0 in oxide. Useful for viewing the complete grain network.
- `grain_id` / `active_grain_id`: current dominant metal grain; 0 where phi >= 0.5
  or no metal order is resolved. This is a categorical label, not a smooth field.
- `eta_1`, `eta_2`, ...: actual evolving metal order parameters (when requested).
- `GB_mask`: current transport mask; `GB_active` also includes the oxide gate.
- `grain_id_initial` and `GB_initial`: the original Voronoi labels/Gaussian mask,
  for comparison. Neither controls transport once eta evolution is on.
- `phi`, `conc_O`, `conc_Cr`, `mu_O`, `M_O`, etc.: existing oxidation fields.

Eta fields are order parameters, not concentration or volume-fraction fields.
They are not renormalized or clipped each timestep. `metal_fraction = 1-h(phi)`
is the phase fraction used for the output area weighting. Finite diffuse eta
tails near the metal/oxide interface are normal; they are not new oxide grains.

## 5. Diagnostics

`diagnostics.csv` retains the original column layout and mass-budget terms.
`grain_diagnostics.csv` adds eta ranges, live-grain counts, unassigned metal cells,
active GB-mask area, and the new grain/coupling energy per out-of-plane depth.
`grain_history.csv` gives each grain's current dominant-cell count and weighted
metal area. `grains.csv` continues to describe the **initial** Voronoi sites.
`parameters.txt` records the actual new coefficients, interface width, transport
normalization and both requested/active grain switches.

In a closed, oxide-free, source-free grain-growth test, the grain energy should
converge to a decreasing history. During coupled oxidation, the grain energy
alone need not decrease: it exchanges energy with phi, and the original model
also has inlet, hard-insertion, Ck and clipping contributions.

## 6. Parameter meaning and checks

All values in the uploaded `Physics` block are retained, including oxide
W=1e4, kappa_phi=1.7e-10, L=0.2, kappa_O=1e-9, kappa_Cr=7.5e-9,
Ck_O=Ck_Cr=2e-10 and the O/Cr mobility prefactors. Default dt=2e-10 s and
Ksp scan interval=1e-7 s are retained as well.

**The new eta coefficients are numerical reference values, not measured Ni–Cr
grain-boundary properties.** Grain energy, migration rate and oxide/grain
interfacial behavior require calibration before quantitative interpretation.
Dynamic eta changes the actual equations: it adds the reciprocal grain-energy
term to phi and changes the diffusion pathways. Keeping the old coefficients
does not guarantee the same total metal/oxide interfacial energy or critical
radius. The full free energy, derivatives and width mapping are in `MODEL.md`.

The new eta trial is checked collectively and is never clipped, including with
`--bounds legacy`. The original oxygen/chromium clipping and Ck rules are retained;
this extension does not repair the earlier concentration-positivity limitation.
A smaller timestep may be needed once transport reaches a moving GB or oxide.

One field per grain increases memory and communication cost. Fields for vanished
grains are retained; there is no grain-remapping optimization. Root still gathers
the full state for output and ordered Ksp insertion.

The included `example_center/` contains a centered hard oxide and two diffuse metal
grains, at initialization and after 500 short steps. It was generated with the
single-process verification backend, not a native MPI run; settings are recorded
in its `parameters.txt`.

## 7. Verification status

See `VERIFICATION.md` for checks actually run. The authoring environment did not
have a usable native MPI runtime. Numerical checks were run with GCC and Clang
in single-process mode, and communication-logic checks with 2, 4 and 8 emulated
ranks. **Emulation is not a native multi-process MPI test.** Run `make test NP=4`
and `make verify-mpi` on your MPI installation before a long simulation.

`make serial-check` and `make emulated-check` are optional developer checks for
hosts without MPI; their executables must not be used as MPI production binaries.
