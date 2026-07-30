# PeriX

PeriX is a Peridynamic Differential Operator (PDDO) framework for
multiphysics simulation.

## Publication scope

This public release contains the element formulations used by the accompanying
manuscript:

- `PoissonElement`
- `DiffusionElement`
- `CahnHilliardElement`
- `PDDODynamicFracElement`
- `ExplicitPDDOFracElement`
- `FracStressCahnHilliardElement`

Thermal transport and other element extensions are outside this release. The
input schema rejects element types that are not part of the publication scope.

The reproducible input decks are in `examples/`:

- `poisson.json` and `poisson_gpu.json`
- `diffusion.json`
- `spinodal.json`, `spinodal_m1_kappa0p01.json`, and
  `spinodal_m4_kappa0p001.json`
- `tensile_plate.json`
- `kalthoff_winkler.json` and `kalthoff_winkler_gpu.json`
- `silicon_particle.json`

## CPU and OpenMP build

OpenMP assembly is enabled by default:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Use `-DPARALLEL_ASSEMBLE=OFF` for a serial-only build.

## CUDA and cuDSS build

The manuscript CUDA assembler and NVIDIA cuDSS solver are opt-in:

```sh
cmake -S . -B build-gpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCUDA_ASSEMBLE=ON \
  -DUSE_CUDSS=ON \
  -DCUDA_DIR=/path/to/cuda \
  -DCUDSS_DIR=/path/to/cudss
cmake --build build-gpu --parallel
ctest --test-dir build-gpu --output-on-failure
```

The CMake configuration checks the requested CUDA toolkit and cuDSS
installation before enabling these backends.
