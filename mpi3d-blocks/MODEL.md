# Three-dimensional Ni–Cr oxidation model

## 1. Fields, geometry and normalization

The evolving fields are the oxygen and chromium mole fractions, cO and cCr, the
oxide order parameter phi, and optionally one metal-grain order parameter eta_g
per initial Voronoi site. Nickel is reported as cNi = 1 - cO - cCr. The phase
interpolation is h(phi) = phi³(10 - 15phi + 6phi²).

The domain is a cell-centered cubic grid. X is the inward oxidation direction;
Y and Z are periodic. Distances are in metres after multiplying grid coordinates
by dx. Geometric grain and seed calculations use cubic-grid units.

Free energies below use the solver normalization with Vm = 1. No conversion of
the supplied mobility prefactors into diffusivities is made.

## 2. Chemical free energy and oxide kinetics

The chemical free energies are

```text
f_m = (6000000*cO² + 583264.303945031*cCr² + 3200000*cO*cCr
       - 298432*cO - 97045.64*cCr - 61732.2974223055) / Vm

f_ox = (12000000*cO² + 7000000*cCr² - 13800000*cO*cCr
        - 8880000*cO + 2680000*cCr + 1860990) / Vm
```

The total functional is

```text
F = integral_V [ (1-h)*f_m + h*f_ox + W*phi²*(1-phi)²
                + kappa_O*|grad cO|² + kappa_Cr*|grad cCr|²
                + kappa_phi*|grad phi|² + f_eta ] dV.
```

In fixed-grain mode f_eta is absent. Concentration chemical potentials follow
from the variational derivatives:

```text
mu_s = (1-h)*df_m/dc_s + h*df_ox/dc_s - 2*kappa_s*laplacian(c_s).
```

The oxide evolves by Allen–Cahn kinetics:

```text
dphi/dt = -L_phi * [ W*(4*phi³ - 6*phi² + 2*phi)
                    + h'(phi)*(f_ox-f_m)
                    - 2*kappa_phi*laplacian(phi)
                    + df_eta_local/dphi ].
```

There is a single oxide order parameter. Oxide–oxide grain boundaries are not
represented by the metal eta fields.

## 3. Concentration-weighted mobilities and transport

For species s = O or Cr,

```text
M_s = c_s * [(1-h)*b_m,s*(1 + (alpha_s-1)*B) + h*b_ox,s]
J_s = -M_s * grad(mu_s)
dc_s/dt = -div(J_s).
```

B is the fixed Gaussian or eta-derived GB indicator. Its contribution is gated
out in oxide by 1-h. The local VTK fields `M_O` and `M_Cr` are these complete
concentration-weighted coefficients.

| Parameter | O | Cr |
|---|---:|---:|
| Matrix mobility prefactor b_m | 2.21e-16 | 3.5436e-20 |
| Oxide mobility prefactor b_ox | 2.21e-18 | 3.5436e-21 |
| Default GB factor alpha | 28.2 | 563 |
| Concentration gradient coefficient | 1e-9 | 7.5e-9 |
| Legacy Ck coefficient | 2e-10 | 2e-10 |

The matrix prefactors and bulk Cr fraction are the uploaded 18 wt.% Cr setup.
A different nominal composition requires consistent material inputs; the solver
does not interpolate a composition-dependent table.

Face mobilities are arithmetic averages of neighboring cell mobilities. For
example,

```text
Jx_(i+1/2,j,k) = -0.5*(M_i,j,k + M_i+1,j,k)
                 * (mu_i+1,j,k - mu_i,j,k) / dx.
```

The rate sums the flux differences through all six faces. The Laplacian uses
three central second differences (seven-point stencil). This construction
conserves the domain O/Cr inventory under closed boundaries when insertion,
Ck and clipping are absent.

## 4. Metal grains and grain-boundary transport

Initial sites are distributed throughout XYZ with a reproducible random seed.
The site count is max(1, round(6V_grid/(pi*d_grid³))). Each site defines a 3D
Voronoi grain with periodic images in Y/Z. The fixed GB mask is

```text
B_initial = exp[-4*ln(2)*d_boundary² / w_FWHM²].
```

Boundary distances are measured to competing Voronoi bisector planes, including
nine Y/Z image combinations. Images of the same physical grain are excluded.
Inlet anchoring relocates existing sites onto a two-dimensional inlet pattern
without changing the grain count.

With grain evolution enabled, define

```text
S = sum_g eta_g²
P = sum_(g<h) eta_g² * eta_h²
H = h(phi)

f_eta = W_eta/4 * [S-(1-H)]² + W_eta*P + A/2*H*S
        + K_eta/2 * sum_g |grad eta_g|².
```

The corresponding forces are

```text
dF/deta_g = eta_g * { W_eta*[S-(1-H) + 2*(S-eta_g²)] + A*H }
            - K_eta*laplacian(eta_g)

df_eta_local/dphi = h'(phi)/2 * {W_eta*[S-(1-H)] + A*S}

deta_g/dt = -L_eta * dF/deta_g.
```

Both reciprocal coupling terms are included. Pure metal has one eta = 1; pure
oxide has all eta = 0. Initial eta fields are smooth nearest-site weights based
on Voronoi bisector distances. No extra grain-growth iterations are performed
silently during initialization.

The evolving mask is

```text
B = scale * min(1, 16*P).
```

It requires overlap of two metal grain fields. A single metal/oxide interface
therefore does not create an artificial substrate GB pathway.

Let w_int = w_FWHM*sqrt(pi)/(2*sqrt(ln 2)). With eta-kappa = 0,
ell = 3*w_int/8 and K_eta = 2*W_eta*ell². For a prescribed positive K_eta,
ell = sqrt(K_eta/(2*W_eta)). In both cases scale = w_int/(8*ell/3), retaining
the integrated planar GB enhancement convention. The eta 10–90% width is
2*ln(9)*ell.

Defaults are W_eta = 1e4, L_eta = 0.2, A = W_eta, and automatic K_eta. Eta-L = 0
skips the eta PDE update but retains its transport profile and phi coupling.
Hard oxide insertion still sets eta to zero in inserted voxels.

## 5. Boundary conditions and oxide insertion

At X boundaries, concentrations, phi and eta use natural zero-normal-gradient
ghost conditions. Chemical potentials normally use zero-flux X conditions.
All fields are periodic in Y/Z.

For `--bc mu`, the left oxygen flux is

```text
J_O,in = 2*M_face*(mu_res-mu_O,first_cell)/dx
M_face = mobility(surface_c_ref, phi_first, B_first, oxygen).
```

The default reservoir potential is 364688 and surface concentration reference
is 0.0023. The reference sets the face mobility; it does not pin the first cell's
concentration. Chromium has no imposed external flux.

For concentration mode, the initial left cell layer is cO = 0.0023 and cCr =
0.182. Only oxygen is reset before subsequent timesteps. Turning O diffusion
off disables both redistribution and ongoing boundary supply, while preserving
initial inventories and the independent hard-insertion/Ck operations.

Ksp mode scans lexicographically in i, j, k on the gathered root model. A site
requires cO*cCr > 1.28e-7, phi <= 0.01, the retained X eligibility margin, and
no existing oxide inside the radius-plus-gap exclusion neighborhood. Default
radius is 5.5 grid spacings, exclusion gap is 6 and scan period is 1e-7 s.
Insertion sets phi = 1, cO = 0.56300203, cCr = 0.35677915 and eta_g = 0 within
an integer-centered sphere. Y/Z minimum-image distances are used for insertion
and exclusion. The introduced O/Cr inventory is recorded explicitly.

Center mode instead initializes one sphere at the exact geometric box center,
including half-integer index centers on even-sized axes, and disables later
insertion. The center mode uses the same hard phase/composition assignment as the Ksp mode.

## 6. Time stepping and optional concentration adjustment

The PDEs use an explicit timestep from the same old state. The legacy Ck option
adds, after the transport and phi trial,

```text
q_s = -Ck_s * [ (c_s,oxide - c_s,transport)*h(phi_trial)
               - (c_s,oxide - c_s,old)*h(phi_old) ] / dt
c_s,trial = c_s,transport + q_s.
```

This is an explicit, timestep-dependent concentration adjustment, not a
conservative flux. Its contribution is recorded in the inventory budget.
The oxide constants are W = 1e4, kappa_phi = 1.7e-10 and L_phi = 0.2. Bulk
concentrations are cO = 0 and cCr = 0.1986. Default dx = 1e-8 m and dt = 2e-10 s.

Stop mode rejects invalid concentration/phi trials; legacy mode independently
clips cO/cCr and snaps phi near its endpoints. Eta updates are never clipped.
An invalid eta trial rejects the coupled PDE step. On rejection, the boundary
reset and source budget are restored. A Ksp insertion performed immediately
before that attempted step remains part of the reported failure state.

## 7. Three-dimensional diagnostics and MPI

The voxel volume is dx*dy*dz. Oxide volume is sum(h(phi))*voxel_volume.
Initial grain volumes and equivalent diameters use pi*d³/6. Dynamic per-grain
metal volume weights are (1-h)*eta_g²/S where S is nonzero. Gradient energy is
summed once per forward face, including periodic Y/Z faces, and integrated over
volume. No per-unit-depth interpretation is used for 3D grain energy.

MPI partitions the global mesh into Px × Py × Pz Cartesian blocks using a
nonperiodic X topology and periodic Y/Z topology. Block sizes may differ by
one cell along each axis. Numerical expressions retain one-based global cell
indices, while arrays store only the local block plus one ghost layer.

Six committed face datatypes select owned tangential cells without transmitting
edge/corner padding. Concentrations, phi and eta are exchanged before their
local derivatives; chemical potentials are exchanged separately before the
conservative flux evaluation. Only blocks on the global X boundary apply the
physical X boundary conditions. Interfaces between blocks use neighbor data.

Owned cells are packed rank-by-rank for MPI_Gatherv and reconstructed at their
global XYZ positions on root. MPI_Scatterv applies the inverse ordering after
insertion. Root scans the global model in the same lexicographic i,j,k order,
so nuclei do not depend on block ordering. VTK data are written X-fastest,
followed by Y and Z, at voxel-center coordinates. Root still holds a full
global snapshot for ordered insertion, output and snapshot-based diagnostics;
these operations are not distributed I/O. No restart reader is present.

The MPI decomposition changes storage and communication, not the free-energy,
mobility, time-integration, or nucleation equations above.

## 8. Source sections

The single production source `NiCr3D_grains.cpp` has numbered section comments:
1 inputs/coefficients; 2 domain/fields/halos; 3 chemical energies/mobilities;
4 Voronoi geometry; 5 eta fields; 6 chemical potentials/transport;
7 gather/scatter and spherical insertion; 8 coupled timestep; 9 VTK output;
10 metadata/diagnostics; 11 input parsing; 12 verification; 13 run control.
