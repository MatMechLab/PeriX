# PeriX

PeriX is a C++20 framework for multiphysics simulation with the Peridynamic
Differential Operator (PDDO). It builds a cell-centred peridynamic mesh from a
structured, circular-lattice, or imported Gmsh mesh, evaluates nonlocal
derivative operators over each node family, and advances the published scalar,
phase-field, and fracture models through a JSON input file.

The command-line JSON interface is the supported public interface. This
repository does not currently install a C++ SDK or promise API stability for
its internal classes.

## Public release scope

This release contains only the element formulations and supporting systems
needed by the accompanying manuscript:

- `PoissonElement`
- `DiffusionElement`
- `CahnHilliardElement`
- `PDDODynamicFracElement`
- `ExplicitPDDOFracElement`
- `FracStressCahnHilliardElement`

Thermal transport and other unpublished element extensions are not included.
The input schema rejects element, boundary-condition, initial-condition, and
solver options outside the public allow-list.

AMGCL support is an optional public solver extension added for reproducible
CPU-only builds. It is not one of the three linear solvers described in the
current manuscript, which are the in-house profile LDU solver, Intel MKL
PARDISO, and NVIDIA cuDSS.

All complete example decks shipped here are two-dimensional. Core mesh,
operator, and selected element paths contain three-dimensional support, but
this release does not yet provide a validated end-to-end 3D benchmark.

## Numerical models

For a node \(i\), PeriX approximates a derivative of a field \(q\) by a
weighted family sum of differences,

\[
D^\alpha q(\mathbf{x}_i)
\approx \sum_{j\in\mathcal{H}_i}
  \left(q_j-q_i\right)g_{ij}^{\alpha}V_j ,
\]

where \(\mathcal{H}_i\) is the horizon family, \(V_j\) is the neighbour volume,
and \(g_{ij}^{\alpha}\) is obtained from the local PDDO moment system. The
published element kernels require `PDMesh.Order >= 2`.

| JSON element type | Unknowns | Physics and strong form | Time/discrete treatment | Examples |
|---|---|---|---|---|
| `poisson` | \(u\) | \(\sigma\nabla^2u+f=0\) | Static, assembled nonlinear-driver path | `poisson*.json` |
| `diffusion` | \(c\) | \(\dot c-D\nabla^2c-f=0\) | Backward Euler | `diffusion.json` |
| `cahnhilliard` | \(c,\mu\) | \(\dot c=\nabla\cdot(M(c)\nabla\mu)\), \(\mu=f'(c)-\kappa\nabla^2c\), \(f=c\ln c+(1-c)\ln(1-c)+\chi c(1-c)\), \(M(c)=M_0c(1-c)\) | Split second-order system, backward Euler, Picard-frozen mobility, conservative antisymmetric bond flux | `spinodal*.json` |
| `pddo_dynamic_frac` | \(\mathbf{u}\) | \(\rho\ddot{\mathbf{u}}=\nabla\cdot\boldsymbol{\sigma}+\mathbf{b}\), small-strain isotropic elasticity with irreversible critical-stretch bond failure | Implicit backward-Euler velocity form; damage is committed between converged steps | `tensile_plate.json` |
| `explicit_pddo_frac` | \(\mathbf{u}\) | Same elastodynamic strong form and critical-stretch family as the implicit fracture element | Matrix-free central difference; fixed CFL-limited step | `kalthoff_winkler*.json` |
| `frac_stress_cahnhilliard` | \(c,\mu,\mathbf{u}\) | Cahn-Hilliard transport coupled to \(\boldsymbol{\epsilon}^*=\frac{\Omega}{3}(c-c_\mathrm{ref})\mathbf{I}\), stress-dependent chemical potential, small-strain mechanics, and irreversible bond fracture | Backward Euler; quasi-static mechanics when \(\rho=0\), inertial mechanics when \(\rho>0\) | `silicon_particle.json` |

The implicit and explicit fracture elements share their spatial constitutive
model, but their time integration and the instant at which damage is committed
are different. Their transient histories should therefore be compared at the
model level, not expected to be bitwise identical.

In the coupled fracture/Cahn-Hilliard element, broken-bond health gates the
mechanical terms but does not block or accelerate species transport. The
current implementation directly constructs the bulk-only species stencil at a
boundary; it does not apply a separate scalar surface-rescaling factor to that
coupled stencil. A positive `rho`, even a small one, selects the inertial
mechanics branch.

CPU, OpenMP, and CUDA paths solve the same public formulations. Floating-point
reduction order and different solver algorithms may produce small
tolerance-level differences; cross-backend results are not promised to be
bitwise identical.

## Dependencies

PeriX is currently oriented to Linux and other Unix-like build environments.

| Dependency | When needed | How it is supplied |
|---|---|---|
| CMake 3.21 or newer | Always | System package or CMake distribution |
| C++20 compiler | Always | GCC or Clang with C++20 support |
| Eigen | Always | Vendored snapshot in `external/eigen` |
| nlohmann/json 3.12.0 | Always | Vendored headers in `external/nlohmann` |
| OpenMP | Default CPU assembly; AMGCL | Compiler runtime; found by CMake |
| Intel oneAPI MKL | Optional PARDISO solver | `USE_ONEAPI=ON`; set `ONEAPI_DIR` or `MKLROOT` |
| AMGCL | Optional CPU iterative solver | Header-only source/install tree; `USE_AMGCL=ON` and `AMGCL_DIR` |
| CUDA Toolkit | Optional CUDA assembly or cuDSS | `CUDA_ASSEMBLE=ON` or `USE_CUDSS=ON`; set `CUDA_DIR` or `CUDAToolkit_ROOT` |
| NVIDIA cuDSS | Optional GPU direct solver | `USE_CUDSS=ON`; set `CUDSS_DIR` |

AMGCL is used through its compile-time builtin/OpenMP backend with
`AMGCL_NO_BOOST`, so Boost is not required. AMGCL itself is MIT licensed and
is not vendored in this repository.

Gmsh is not required to run the examples: the imported
`examples/impact2D.msh` mesh is committed. Gmsh is only needed to regenerate it
from `examples/impact2D.geo`. Python and NumPy are optional and used only by
`examples/export_cell_ic.py`; they are not build dependencies.

## Build configurations

### Basic CPU build

OpenMP assembly is enabled by default:

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

This always builds the in-house profile LDU solver. Use
`-DPARALLEL_ASSEMBLE=OFF` for a serial-only build. The default solver is useful
for small problems; its skyline/profile storage can grow rapidly for large
nonlocal systems.

### CPU build with Intel MKL PARDISO

Load the oneAPI runtime environment before configuring and again in each new
shell used to run PeriX:

```sh
source /path/to/oneapi/setvars.sh

cmake -S . -B build-oneapi \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DUSE_ONEAPI=ON \
  -DONEAPI_DIR=/path/to/oneapi
cmake --build build-oneapi --parallel
ctest --test-dir build-oneapi --output-on-failure
```

If `MKLROOT` is already set correctly, `ONEAPI_DIR` may be omitted. A binary
linked to `libmkl_rt` still needs the oneAPI library paths at run time.

### CPU build with AMGCL

Point `AMGCL_DIR` to the directory that contains `amgcl/amg.hpp`:

```sh
cmake -S . -B build-amgcl \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DUSE_AMGCL=ON \
  -DAMGCL_DIR=/path/to/amgcl
cmake --build build-amgcl --parallel
ctest --test-dir build-amgcl --output-on-failure
```

The public AMGCL configuration is intentionally narrow:

```json
"LinearSolver": {
  "type": "amgcl",
  "params": {
    "tol": 1.0e-10,
    "maxiter": 200,
    "verbose": false,
    "iluk_level": 2,
    "solver_threads": 0
  }
}
```

These five keys are the only accepted AMGCL parameters. Internally, PeriX uses
BiCGSTAB, detects a node block size from one to five, and reuses the
preconditioner. If the recomputed true residual misses the requested tolerance,
it rebuilds once and then escalates the preconditioner along a fixed ladder:

1. smoothed-aggregation AMG with ILU(0) relaxation (the default),
2. single-level ILU(0),
3. single-level ILU(k), with `iluk_level` (default 2, range 1-6) levels of fill.

The escalation matters because the PDDO stencil is dense, non-symmetric and not
an M-matrix: a zero-fill incomplete factorization of it can stagnate or even be
exactly singular, while ILU(k>=1) converges. Each rung is built only after the
cheaper one has been shown to fail on that very matrix, so well-conditioned
problems pay nothing. A solve succeeds only when
\(\lVert b-Ax\rVert_2/\lVert b\rVert_2\leq\texttt{tol}\).

`solver_threads` sizes the OpenMP team used for the AMGCL solve alone; `0` (the
default) derives it from the row count, allowing one thread per 20000 rows up to
eight. AMGCL's builtin backend opens a fresh parallel region for every sparse
kernel, so on a many-core node the barrier latency swamps the arithmetic long
before the cores are saturated: on a 96-core machine the 200x200 Poisson deck
takes 172 s on the default 96-wide team and 1.6 s once the team is sized to the
work. Set the key explicitly only to override that heuristic; it never changes
the assembly team, which always follows `OMP_NUM_THREADS`.

PARDISO and AMGCL may be enabled in the same CPU build.

### CUDA assembly and NVIDIA cuDSS

CUDA assembly and the cuDSS linear solver are independent options. Enable both
for the complete implicit GPU path:

```sh
cmake -S . -B build-gpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCUDA_ASSEMBLE=ON \
  -DUSE_CUDSS=ON \
  -DCUDA_DIR=/path/to/cuda \
  -DCUDSS_DIR=/path/to/cudss \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-gpu --parallel
ctest --test-dir build-gpu --output-on-failure
```

Replace `89` with the compute capability of the target GPU. CMake validates
the CUDA toolkit, cuBLAS, cuDSS headers, and cuDSS library before enabling this
configuration.

The two JSON selections are also independent:

```json
"JobSystem": {
  "type": "static",
  "assemble": "cuda"
},
"LinearSolver": {
  "type": "cudss"
}
```

`assemble` may be `serial`, `openmp`, or `cuda` when its build option is
available. An explicit fracture deck is matrix-free and must not contain
`LinearSolver` or `NonlinearSolver` blocks.

## Running an example

PeriX writes results next to the input JSON file. To keep the source tree
clean, copy a deck and any relative assets to a separate run directory.

The smallest end-to-end example is the AMGCL Poisson deck:

```sh
mkdir -p /tmp/perix-poisson-amgcl
cp examples/poisson_amgcl.json /tmp/perix-poisson-amgcl/
cd /tmp/perix-poisson-amgcl
/path/to/PeriX/build-amgcl/perix -i poisson_amgcl.json
```

For an imported-mesh example, copy the mesh with the deck:

```sh
mkdir -p /tmp/perix-kalthoff
cp examples/kalthoff_winkler.json examples/impact2D.msh /tmp/perix-kalthoff/
cd /tmp/perix-kalthoff
/path/to/PeriX/build-oneapi/perix -i kalthoff_winkler.json
```

The command-line form is:

```text
perix -i input.json [--read-only]
```

`--read-only` validates the public schema and builds or imports the base mesh;
it does not assemble or solve the numerical problem. It may still write a base
mesh if the deck sets `"savemesh": true`.

The normal VTU output set is:

- `<input>-pdmesh.vtu`
- `<input>-PD-results[-NNNNN].vtu` and its `.pvd` time-series index
- `<input>-Element-results[-NNNNN].vtu` and its `.pvd` time-series index
- `<input>-mesh.vtu` when `Mesh.savemesh` is enabled

`OutputSystem.format` may be `vtu`, `exodus`, or `both`. VTU is the default.
Transient output is written at step zero, at each
`OutputSystem.interval`, and at the final step. The VTU writer currently uses
ASCII arrays, which are easy to inspect but can be large.

## Example catalogue

| Deck | Model | Assembly | Linear solver |
|---|---|---|---|
| `poisson_amgcl.json` | Poisson, small public smoke case | Serial | AMGCL |
| `poisson.json` | Poisson, large CPU case | OpenMP | PARDISO |
| `poisson_gpu.json` | Poisson GPU case | CUDA | cuDSS |
| `diffusion.json` | Transient diffusion | OpenMP | PARDISO |
| `spinodal.json` | Spinodal decomposition | OpenMP | PARDISO |
| `spinodal_m1_kappa0p01.json` | Spinodal parameter study | OpenMP | PARDISO |
| `spinodal_m4_kappa0p001.json` | Spinodal parameter study | OpenMP | PARDISO |
| `tensile_plate.json` | Implicit dynamic fracture | OpenMP | PARDISO |
| `kalthoff_winkler.json` | Explicit impact fracture | OpenMP | Matrix-free |
| `kalthoff_winkler_gpu.json` | Explicit impact fracture | CUDA | Matrix-free |
| `silicon_particle.json` | Coupled species/stress/fracture | OpenMP | PARDISO |

`examples/M4kappa0.001.i` is retained as a legacy comparison input; it is not a
PeriX JSON deck and cannot be passed to `perix -i`.

The test suite always exercises unit tests, schema validation, and read-only
loading of every public JSON deck. With `USE_AMGCL=ON`, it also performs a
numerical solver test and a complete `poisson_amgcl.json` run. Read-only deck
tests alone are not evidence that the large manuscript simulations were
numerically executed.

## Boundary and initial data

The public boundary-condition types are:

- `dirichlet`: prescribed value, with optional direct node pinning;
- `neumann`: homogeneous generic Neumann condition only;
- `pdtraction`: mechanical traction for the PDDO elasticity models;
- `speciesflux`: source-form flux on the concentration field `c`.

The public initial profiles are constant, linear, box, circle, ellipse,
Gaussian, cosine, and seeded random. Explicit dynamics additionally accepts an
initial velocity field. Random initial data must include a seed so manuscript
runs remain reproducible.

## Performance guidance

- Configure `Release`; debug builds are useful for checking but substantially
  slower.
- Use `JobSystem.assemble: "openmp"` for CPU assembly and set
  `OMP_NUM_THREADS` to physical cores. On dedicated nodes,
  `OMP_PLACES=cores` and `OMP_PROC_BIND=close` often improve locality.
- Use PARDISO for robust large CPU implicit runs, cuDSS when the matrix and
  factorization fit GPU memory, or AMGCL when a lower-memory CPU iterative
  solve is suitable. The default profile LDU solver is primarily a
  dependency-free small-problem backend.
- For CUDA builds, compile for the actual GPU architecture. Avoid relying on a
  generic architecture list when benchmarking.
- Increasing `HorizonRadiusFactor` or PDDO `Order` enlarges each family,
  increases matrix density, and raises both operator and solver cost.
- Explicit fracture avoids matrix assembly and factorization, but its time
  step must satisfy the reported CFL-scale stability limit. Reducing output
  frequency is often essential for long impact simulations.
- For implicit transients, adaptive time stepping can cut back after Newton or
  linear-solver failure and grow after easy steps. Set physically meaningful
  `min_dt`, `max_dt`, and output intervals.
- Request only needed derived `OutputSystem.Fields`, and increase
  `OutputSystem.interval` for long runs. ASCII VTU output can otherwise
  dominate disk space and wall time.
- CUDA operator caching chooses a default heuristic. Benchmark overrides with
  `PERIX_CUDA_OPCACHE=0` or `PERIX_CUDA_OPCACHE=1`; do not assume one setting
  is fastest for both 2D and 3D kernels.
- `PERIX_ASSEMBLE_TIMING=1` prints CPU assembly breakdowns, and
  `PERIX_CUDSS_TIMING=1` prints cuDSS transfer/factor/solve timing. Disable
  diagnostic timing for production measurements.

## License

PeriX is licensed under GNU GPLv3; see `LICENSE`. Optional third-party
dependencies retain their own licenses.
