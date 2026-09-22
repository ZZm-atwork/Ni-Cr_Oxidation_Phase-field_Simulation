# Model and implementation

## 1. State and scope

The conserved fields are oxygen cO and chromium cCr; cNi = 1-cO-cCr. The oxide
order parameter is phi. With `--grain-evolution on`, each initial Voronoi site
also has one metal order parameter eta_i. Eta_i is an orientation/grain-order
field, not a concentration and not an independent oxide variable.

The uploaded alloy/oxide free-energy functions, interpolation
h(phi)=phi^3(10-15 phi+6 phi^2), composition-gradient coefficients, bulk and oxide
transport prefactors, hard-seed compositions, and boundary conditions are kept.
The original explicit concentration update and optional Ck/clipping are also
kept. The grain subsystem below is an added model, not a transcription of the
old five-field GB_new2.cpp equations or its dimensionless parameters.

## 2. Common free energy

Write H=h(phi), m=1-H, S=sum_i eta_i^2, and
P=sum_{i<j} eta_i^2 eta_j^2. The added energy density is

    f_eta = W_eta/4 (S-m)^2 + W_eta P
            + A_eta/2 H S + K_eta/2 sum_i |grad eta_i|^2.

All coefficients W_eta, K_eta are positive; A_eta is nonnegative. This energy is
nonnegative for phi in [0,1]. The added homogeneous energy vanishes in both pure
states: one eta=1 in metal, or all eta=0 in oxide. A_eta provides an additional
quadratic suppression of residual metal order inside oxide. Its default equals
W_eta; it is a numerical reference, not an experimental reaction barrier.

For H=0, the local part can be written

    W_eta [sum_i eta_i^4/4 - sum_i eta_i^2/2
           + 3/2 sum_{i<j} eta_i^2 eta_j^2 + 1/4].

This is the symmetric gamma=1.5 multi-order-parameter grain-growth energy. The
standard grain-growth formulation and its Allen–Cahn evolution are described in
MOOSE's primary documentation:
https://mooseframework.inl.gov/modules/phase_field/Grain_Growth_Model.html

The oxide coupling used here is an explicit additional modeling choice. It is
not asserted to be a calibrated Ni–Cr/Cr2O3 model from that reference.

## 3. Evolution and reciprocal coupling

Eta evolves with the current (old timestep) concentrations, phi and eta:

    d eta_i/dt = -L_eta [
        W_eta eta_i (S-m + 2(S-eta_i^2))
        + A_eta H eta_i - K_eta Laplacian(eta_i)
    ].

There is NO per-step normalization of the eta fields. A sum-of-squares condition
is encouraged energetically, not imposed as a phase-fraction constraint. At a
metal boundary, S is generally below 1. In a mixed metal/oxide region it is not
in general equal to m. For one homogeneous nonzero eta, the stationary amplitude
satisfies eta^2 = max(1-(1+A_eta/W_eta)H,0).

The SAME free energy adds the following term to the existing phi driving force:

    d f_eta/d phi = h'(phi)/2 [W_eta (S-m) + A_eta S].

The phi update is therefore

    d phi/dt = -L_phi [
        W_phi g'(phi) + h'(phi)(f_oxide-f_metal)
        - 2 kappa_phi Laplacian(phi)
        + d f_eta/d phi
    ].

Eta is not added to the O/Cr chemical free-energy branches, so there are no
extra chemical-potential derivatives from f_eta. The concentration transport
changes indirectly because its GB indicator now comes from eta.

**The extra phi term cannot be omitted while calling this a reciprocal
free-energy coupling.** In particular, eta gradients and oxide suppression add
to the metal/oxide interfacial structure and energy. Preserving W_phi and
kappa_phi does not preserve the total interfacial energy of the expanded model.
This is why oxide-growth/critical-radius validation must not be inferred from
the earlier fixed-grain model, even though its numerical coefficients are kept.

In the source, `grain_local_energy`, `grain_local_deta`, `grain_local_dphi`,
`eta_force`, and `eta_phi_force` implement these expressions. `eta_trial` is
called before any PDE trial fields are swapped. Collective rejection therefore
rejects the eta, phi and concentration trial updates together.

## 4. Initial eta profiles and hard nuclei

The original Voronoi grain-count formula and seed placement determine N_eta.
At a cell, the winning seed has weight 1; each other seed has weight
exp(-d_i/ell_eta), where d_i is its minimum nonnegative bisector distance relative
to the winning seed, with periodic Y images. Normalizing these weights gives
smooth initial eta profiles. An isolated two-grain planar boundary has the
logistic profile below. Multi-junctions generally relax after startup; the code
does not run hidden grain-growth steps during initialization.

The existing hard oxide insertion still sets phi, cO and cCr in the same disk.
It additionally sets every metal eta_i to zero in those same cells. This is part
of the discrete insertion, not a concentration source added by eta evolution.
There is no temporary nucleation bias or smooth phi-only nucleation operation.
A center seed is inserted once; `--nucleation off` suppresses all insertions.

Metal order can have diffuse tails near an oxide. It is energetically suppressed,
not algebraically overwritten on each timestep. The output active grain label
is 0 for phi >= 0.5, avoiding a false grain label inside oxide.

## 5. Moving GB transport and width normalization

Only overlap between DISTINCT metal eta fields contributes:

    B_eta = min(1,16 P).

Thus a single metal/oxide interface cannot create a substrate GB channel. The
cap at 1 applies to this transport indicator, not to eta. The actual mask used by
the existing mobility function is

    GB_mask = C_eta B_eta

and the mobility remains

    M_s = c_s [(1-H) B_s^metal (1+(F_s-1)GB_mask) + H B_s^oxide].

The existing F_O=28.2 and F_Cr=563 are retained as reference multipliers. To avoid
silently increasing/decreasing integrated GB conductance merely by using a
different diffuse shape, C_eta is determined from a planar interface.

For two metal grains, eta_1=q, eta_2=1-q, and

    ell_eta = sqrt(K_eta/(2 W_eta)),
    q(x) = 1/(1+exp(x/ell_eta)),
    B_eta(x) = sech^4(x/(2 ell_eta)),
    integral B_eta dx = 8 ell_eta/3,
    eta 10–90% width = 2 ln(9) ell_eta.

For the original Gaussian mask with FWHM w=gb_fwhm_grid*dx,

    integral B_old dx = w_G = w sqrt(pi)/(2 sqrt(ln(2))).

We set C_eta=w_G/(8 ell_eta/3), so

    (F_s-1) integral GB_mask dx = (F_s-1) w_G.

This is a PLANAR equilibrium-profile match. It is not a promise of identical
local fluxes, transient profiles, triple-junction transport or oxidation rates.
No global normalization keeps total GB area fixed as grains coarsen.

Default `--eta-kappa 0` chooses ell_eta=3 w_G/8 and K_eta=2 W_eta ell_eta^2.
Then C_eta=1. For dx=1e-8 m, w=4 dx, W_eta=1e4, the numerical values are about
K_eta=5.10e-12 and a 7.02-grid eta 10–90% width. This is not the same definition
as a 4-grid Gaussian FWHM. A positive explicit eta-kappa uses the corresponding
C_eta. In that case the actual peak enhancement can differ from F_s; the
normalization preserves the reference integrated excess, not the peak.

## 6. Calibrating the new grain subsystem

For a planar metal/metal interface in this convention,

    sigma_GB = sqrt(2 K_eta W_eta)/3 = K_eta/(3 ell_eta),
    M_GB(migration) = 3 L_eta ell_eta,
    M_GB sigma_GB = L_eta K_eta.

The last product gives the sharp-interface curvature coefficient. For a
well-resolved isolated circular grain in 2D, the thin-interface limit is

    dR/dt = -L_eta K_eta/R.

Given a physical sigma_GB, migration mobility M_GB and chosen numerical ell_eta,
use W_eta=3 sigma_GB/(2 ell_eta), K_eta=3 sigma_GB ell_eta and
L_eta=M_GB/(3 ell_eta), provided the full solver uses consistent units. A_eta
and the metal/oxide interface must be validated separately. These migration
parameters are NOT the O/Cr GB diffusion multipliers.

When free-energy density is in J/m^3 and length in metres, W_eta and A_eta have
units J/m^3, K_eta has J/m, L_eta has m^3/(J s), sigma has J/m^2 and the geometric
migration mobility has m^4/(J s). These dimensional relations do not by themselves
calibrate the inherited thermodynamic or mobility scales.

The defaults W_eta=1e4, L_eta=0.2, A_eta=W_eta were selected as separate numerical
reference inputs. They are not fitted to a measured Ni–Cr grain-growth rate.
Grain migration, wetting/pinning behavior and interfacial angles must not be
reported as quantitatively validated without appropriate material data/tests.

## 7. Discretization, MPI and output

All eta fields are X-slab distributed with the same one-column halo and periodic
Y as the existing fields. Eta halos are exchanged before evaluating Laplacians;
the dynamic GB mask is rebuilt and exchanged before computing face mobilities.
O/Cr, phi and eta use simultaneous forward-Euler trials from the same old state.
No grain subcycling, asynchronous update or remapping is used.

Ksp scans gather eta together with the original fields, clear eta inside the
ordered hard insertions on root, and scatter them back. Output gathers the
actual evolved eta, not its initialization. Root has an additional full-domain
eta snapshot. The new work/memory scales linearly with the number of initial
grain fields for each local grid cell; a vanished grain's field is not removed.

Eta remains subject to a strict finite/[0,1] trial check even under legacy bounds.
That check does not imply the original O/Cr update is positivity preserving.
There is no universal stable dt for this coupled nonlinear explicit solver.
The new gradient term contributes a scale of order dt <= dx^2/(4 L_eta K_eta)
in two dimensions, before local-potential and other PDE restrictions are included.

`grain_energy_per_depth` integrates ONLY f_eta (including its coupling), with
forward-face gradient differences. It has units J/m in a consistent 2D SI model.
Its discrete derivative is the implemented -K_eta Laplacian, including periodic
faces and physical no-flux X faces. The built-in test checks this derivative for
both eta and phi, in interiors and at boundaries.

Per-grain weighted metal area uses (1-H) eta_i^2/S where S is resolved. This is an
output partition, not a constraint or update of the eta fields. Unresolved metal
cells are explicitly counted rather than divided by zero or silently relabeled.
