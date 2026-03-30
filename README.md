# ndiffusion

Multigroup neutron diffusion solver for 1-D slab, cylinder, and sphere geometries. Written in C++17; exposed to Python via pybind11.

## Capabilities

- 1-D slab, cylinder, and sphere geometries
- Arbitrary number of energy groups and material regions
- Vacuum, reflective, and albedo boundary conditions
- **k-eigenvalue solver** - matrix-free power iteration; A &phi; = (1/k) B &phi; without assembling the full system matrix
- **Time-dependent solver** - backward-Euler time stepping, unconditionally stable
- Per-group Thomas (TDMA) tridiagonal solver inside a Gauss-Seidel group sweep
- Harmonic-mean diffusion coefficients at material interfaces

## Installation

Requires a C++17 compiler, Python >= 3.9, and `pybind11 >= 2.12`.

```bash
pip install .
```

For development (rebuilds automatically on source changes):

```bash
pip install -e . --config-settings=editable.mode=compat
```

## Quick start

```python
import numpy as np
import ndiffusion as nd

# 1-group sphere, single material
m = nd.Materials()
m.n_mat    = 1
m.n_groups = 1
m.D        = [3.850204978408833]
m.removal  = [0.1532]
m.scatter  = [0.0]
m.chi      = [1.0]
m.nusigf   = [0.1570]

cells = 50
edges = list(np.linspace(0.0, 100.0, cells + 1))

solver = nd.KEigenSolver(
    mats       = m,
    medium_map = [0] * cells,
    edges_x    = edges,
    geom       = nd.Geometry.Sphere,
    bc         = [nd.BoundaryCondition(A=1.0, B=0.0)],
    epsilon    = 1e-8,
    verbose    = False,
)
result = solver.solve()

flux = np.array(result.flux).reshape(cells, m.n_groups)
print(f"keff = {result.keff:.8f}")   # → 1.00001244
```

See `examples/k_eigenvalue.py` and `examples/time_dependent.py` for more complete examples.

## Boundary conditions

| Type | A | B |
|------|---|---|
| Zero-flux (approx. vacuum) | 1.0 | 0.0 |
| Marshak vacuum | (1−α)/(4(1+α)) | D/2 |
| Reflective | 0.0 | 1.0 |

The `ndiffusion.boundary_conditions(Dg, alpha)` helper constructs the coefficient array from an albedo value `alpha` (0 = vacuum, 1 = reflective).

## Standalone C++ driver

To build and run the 10 reference problems without Python:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/ndiffusion_driver
```

## Project structure

```
CMakeLists.txt              top-level CMake
pyproject.toml              build config (scikit-build-core)

cpp/
  CMakeLists.txt
  include/ndiffusion/
    types.hpp               shared types: Geometry, Materials, BoundaryCondition, results
    solver_1d.hpp           1-D solver class declarations
    solver_2d.hpp           placeholder for future 2-D solver
    solver_3d.hpp           placeholder for future 3-D solver
  src/
    solver_1d.cpp           1-D solver implementation
    main.cpp                standalone driver (10 reference problems)
  python/
    bindings.cpp            pybind11 bindings → ndiffusion._core

src/ndiffusion/
  __init__.py               re-exports from _core + create utilities
  create.py                 load cross-section data from JSON / .npz / .csv
  data/
    problem_setup.json      named problem definitions

tests/
  test_k_eigenvalue.py
  test_time_dependent.py

examples/
  k_eigenvalue.py
  time_dependent.py
```

## Running tests

```bash
pytest
```

The source-controlled test suite lives in `tests/` and is run with pytest via the configuration in `pyproject.toml`.

The `Testing/` directory is a generated CMake/CTest artifact directory, not part of the hand-maintained test suite. It can appear after CMake tooling runs, and should not be treated as project source.

## API docs

`Doxyfile` configures Doxygen for the C++ sources under `cpp/include/ndiffusion`, `cpp/src`, and `cpp/python`. To generate the HTML documentation:

```bash
doxygen Doxyfile
```

Or, after configuring CMake, use:

```bash
cmake --build build --target docs
```

The output is written to `docs/doxygen/html/`.

## Future work

- 2-D geometry (r-z, x-y)
- 3-D geometry (x-y-z)
