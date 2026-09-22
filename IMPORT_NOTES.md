# Import notes — 2026-09-21

| Repository directory | Source archive | Archive SHA-256 |
| --- | --- | --- |
| `mpi2d` | `NiCr2D_MPI_eta.zip` | `a40e3c0d53990ead70230370b34eed2ddb1ec0ec508b33c06f34c8ed662b0ea8` |
| `mpi3d-slab` | `NiCr3D_Xslab_optimized.zip` | `2a4f664cc7227fa0dc782d7ac177b21fd0f2bdcf9cc2a9eabf0d3504b4b1405a` |
| `mpi3d-blocks` | `NiCr3D_MPI_blocks.zip` | `f5c84ff5e8169c36afdf8b5d98ab04ec2f32abea4cfbab54fad33957e0d818cd` |

All imported solver and test source files are unchanged. Existing logs and the
small 2D example are retained as historical evidence. New repository-level
README, ignore rules, and attributes organize these snapshots for GitHub.

Fresh checks in the preparation environment:

| Implementation | Check | Result |
| --- | --- | --- |
| 2D | `make serial-check` | PASS |
| 3D block | `make serial-check` | PASS |
| 3D optimized slab | GCC C++17, `NICR_SERIAL_VERIFY`, zlib; `--self-test` | PASS |
| All variants | Native MPI | Not run; MPI unavailable |

Only existing single-process self-tests were rerun. Historical benchmark and
emulated-rank results were not rerun as part of repository preparation.
