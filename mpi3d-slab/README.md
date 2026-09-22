# Ni–Cr 3D oxidation — optimized X-slab MPI solver

The solver advances oxygen, chromium, oxide phase fraction and optional metal-grain eta fields on a 3D Cartesian mesh. MPI partitions X into slabs. Y and Z are periodic; the configured oxygen boundary condition acts at the global left X face.

## Build

A C++17 MPI compiler and the **zlib development headers/library** are required. There is no VTK C++ library dependency.

```bash
make
# Equivalent command:
mpic++ -O3 -std=c++17 NiCr3D_grains.cpp -o nicr3d_mpi -lz
```

When using an existing Makefile, add `-lz` to the link command (after the source/object files). If the compiler cannot find `zlib.h` or the linker cannot find `-lz`, make the zlib headers/library available in the build environment. Avoid `-ffast-math`/`-Ofast` when reproducing the verification results.

Run the native self-test within a suitable allocation:

```bash
mpirun -np 4 ./nicr3d_mpi --self-test
```

Use the existing physical simulation arguments. The default mesh, timestep, material coefficients, mobility prefactors and output cadence are retained. `--help` lists all input options. Choose a new empty `--out` directory for each run.

## Output

The default output is **compressed Float32 VTI**. Files are named `fields_0.vti`, `fields_<step>.vti`, and, on an error, `state_at_failure.vti`. Open the VTI series in ParaView; do not treat it as a legacy `.vtk` text file or merely rename the extension.

Exactly seven arrays are saved:

| Array | Definition |
|---|---|
| `phi` | Oxide phase field |
| `conc_O` | Oxygen concentration |
| `conc_Cr` | Chromium concentration |
| `mu_O` | Oxygen chemical potential |
| `mu_Cr` | Chromium chemical potential |
| `grain_sum` | With eta: sum of `g * eta_g`, starting at g=1, unnormalized and without an extra phi multiplier. Without eta: initial grain ID times `1-h(phi)`. |
| `GB_mask` | The existing transport GB mask: scaled eta overlap or fixed Voronoi Gaussian mask, depending on grain mode. No extra phi multiplier is applied to the exported mask. |

VTI retains the original sample positions, grid spacing, dimensions and X-fastest output ordering. Data are stored as PointData at the existing cell-centre positions, matching the former legacy VTK representation. Use Slice or Clip to inspect interior grains/GBs, rather than relying on an external-surface view.

### Precision and compression controls

```bash
# Default: small visualization snapshots, fast zlib setting
--output-format vti --output-precision float32 --compression-level 1

# Preserve the double-precision field values in the saved arrays
--output-format vti --output-precision float64 --compression-level 1

# Reproduce the former 17-significant-digit ASCII representation
--output-format vtk
```

`--compression-level` accepts integers from 0 through 9. Level 1 favors speed; larger levels trade compression work for potentially smaller files. Level 0 uses uncompressed/stored zlib blocks. Precision and compression settings apply only to VTI; legacy VTK always writes double values as 17-digit text.

**The solver state, physical parameters and all PDE/nucleation arithmetic remain double precision.** Float32 conversion occurs only in the output writer. It rounds the saved values (approximately seven significant decimal digits for ordinary normalized values); sufficiently small tails can round to zero. For analysis sensitive to tiny concentrations, derivatives or small residuals, use Float64 output. Zlib compression itself is lossless relative to the selected output precision.

These seven arrays are visualization snapshots, **not complete restart checkpoints**: they do not contain the individual eta fields, cumulative budgets and complete restart metadata. Selecting Float64 does not turn them into full checkpoints. Existing files are not converted or deleted.

The writer streams 64-KiB data blocks and compression buffers instead of allocating full-volume output copies. It writes a temporary `.vti.tmp` file, checks writes/compression, and renames it after completion. Float32 overflow is reported instead of silently writing infinities; select Float64 when needed.

## Faster ordered nucleation

The candidate scan still visits the same global cells in the same order and applies the same Ksp, phi and exclusion conditions. Only the neighborhood searches in `valid_nucleus()` and `stamp()` are bounded.

The searches use sorted, disjoint periodic intervals in Y and Z, and a clipped X interval. Cells outside the maximum interaction distance are skipped. When a neighborhood spans an entire periodic axis, that axis is visited once, without duplicate wrapped cells. The original distance predicates and the order of accepted updates/budget accumulation are retained, including the strict exclusion inequality and inclusive sphere boundary.

No radius, exclusion gap, nucleation period or thermodynamic driving force has been changed. Nucleation remains ordered on rank zero. This is an algorithmic search optimization, not parallel nucleation.

## Frozen-eta cache

For the coupled eta model with frozen grain motion, use the existing settings:

```bash
--grains on --grain-evolution on --eta-L 0
```

By default, `--frozen-eta-cache on` reuses eta halo values, `eta_s2` and the eta-derived `GB_mask` between eta changes. Their construction depends on eta and fixed coefficients, not on the evolving phi/O/Cr values. The phi-dependent coupling and transport gating are still evaluated normally.

The cache is invalidated after an accepted eta update, initial/hard seeding that changes eta, and replacement of the root snapshot's eta data. Cross-rank insertion invalidates all ranks together. An empty Ksp scan does not invalidate the distributed frozen cache. Evolving eta (`eta_L > 0`) continues to refresh the eta fields each step.

For a reference comparison:

```bash
--frozen-eta-cache off
```

Developers adding an external eta edit or restart loader must call `invalidate_eta_cache()` **on every participating rank** before the next collective refresh, even when only one rank's eta values changed. Changes to the eta-derived mask coefficients also require invalidation. Internal code paths handle this already.

Turning `--grain-evolution off` selects the fixed-Voronoi model; it is not equivalent to freezing the coupled eta model.

## Applying to a locally tuned source

`optimize_xslab.patch` targets the latest seven-field X-slab source (the one with `grain_sum` and `GB_mask`). It leaves material-coefficient lines unchanged. Apply it to a backup:

```bash
cp NiCr3D_grains.cpp NiCr3D_grains.cpp.before_optimization
patch --dry-run -p0 < optimize_xslab.patch
# After the dry run succeeds and the diff has been reviewed:
patch -p0 < optimize_xslab.patch
mpic++ -O3 -std=c++17 NiCr3D_grains.cpp -o nicr3d_mpi -lz
```

The complete replacement CPP contains the coefficients from the shared source. Preserve locally tuned `ko`, `kc`, `kp`, `W` and other settings by using the patch or transferring those edits. Independently changed functions may need manual reconciliation. The block-decomposition source is not part of this update.

## Verification

```bash
make verify-serial
make verify-emulated RANKS="2 4 8"
# Native MPI, within an allocation large enough for the requested ranks:
make verify-mpi RANKS="1 2 4"
```

The Python comparison tool needs only the standard library; `--with-vtk` additionally validates loading through an installed Python VTK reader. `tests/reference/NiCr3D_grains.cpp` is the exact shared pre-optimization source for reproducible comparison, not another production executable.

```bash
python3 tests/verify.py --backend serial --with-vtk
python3 tests/verify.py --backend native --ranks 4 --cxx mpic++ --launcher mpirun
```

See `VERIFICATION.md` and `tests/logs/` for the checks and measured local performance. Native MPI execution was not available in the development container.

## Remaining scaling limits

PDE fields remain distributed, but output, snapshot-based diagnostics and ordered nucleation still gather state onto rank zero. These operations and root memory can limit larger problems. The update reduces output bytes, decimal formatting, unnecessary neighborhood visits and repeated frozen-eta work; it does not eliminate every serial operation or promise a particular end-to-end cluster speedup.

## Format references

The writer follows the VTK XML ImageData/raw appended format and its zlib block-header convention; compression uses zlib `compress2`.

- VTK XML format: https://docs.vtk.org/en/latest/vtk_file_formats/vtkxml_file_format.html
- VTK compression header implementation: https://github.com/Kitware/VTK/blob/master/IO/XML/vtkXMLWriter.cxx
- zlib manual: https://zlib.net/manual.html
