# ndiffusion

Multigroup neutron diffusion solver for 1-D and 2-D geometries. Written in C++17; exposed to Python via pybind11.

## Capabilities

### 1-D (slab, cylinder, sphere)
- Arbitrary number of energy groups and material regions
- Vacuum, reflective, and albedo boundary conditions
- **k-eigenvalue solver** — matrix-free power iteration; A&phi; = (1/k)B&phi;
- **Fixed-source solver** — direct solve of A&phi; = q for a user-supplied volumetric source
- **Time-dependent solver** — backward-Euler time stepping, unconditionally stable
- Per-group Thomas (TDMA) tridiagonal solver inside a Gauss-Seidel group sweep
- Harmonic-mean diffusion coefficients at material interfaces

### 2-D structured (Cartesian XY or axisymmetric RZ)
- Finite-difference 5-point stencil on an nx × ny Cartesian grid
- Left (x=0) and bottom (y=0) boundaries hardcoded as reflective; right and top boundaries take user-specified Robin BCs per group
- **k-eigenvalue solver** — line-TDMA x-sweeps inside a Gauss-Seidel outer iteration
- **Fixed-source solver** — same spatial sweep; solves A&phi; = q directly
- **Time-dependent solver** — backward-Euler stepping using the same line-TDMA sweep

### 2-D unstructured (triangles and/or quadrilaterals)
- Cell-centred finite-volume method (FVM)
- Arbitrary Robin BCs per boundary tag; harmonic-mean interface diffusion coefficients
- **k-eigenvalue solver** — power iteration with point Gauss-Seidel inner solve
- **Fixed-source solver** — point SOR (successive over-relaxation) inner solve
- **Time-dependent solver** — backward-Euler stepping with point Gauss-Seidel

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

### 1-D k-eigenvalue

```python
import numpy as np
import ndiffusion as nd

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
)
result = solver.solve()
print(f"keff = {result.keff:.8f}")   # → 1.00001244
```

### 2-D structured k-eigenvalue

```python
solver = nd.KEigenSolver2D(
    mats       = m,
    medium_map = [0] * (nx * ny),
    edges_x    = list(np.linspace(0.0, R, nx + 1)),
    edges_y    = list(np.linspace(0.0, R, ny + 1)),
    geom       = nd.Geometry2D.XY,
    bc_x       = [nd.BoundaryCondition(A=1.0, B=0.0)],   # vacuum right
    bc_y       = [nd.BoundaryCondition(A=1.0, B=0.0)],   # vacuum top
)
result = solver.solve()
flux = np.array(result.flux).reshape(nx, ny, m.n_groups)
```

### 2-D unstructured fixed-source

```python
# Build an unstructured mesh (vertices + connectivity + boundary faces)
mesh = nd.UnstructuredMesh2D()
mesh.vx = vx; mesh.vy = vy
mesh.cell_vertices = cell_vertices
mesh.cell_offsets  = cell_offsets
mesh.material_id   = mat_ids
mesh.bface_v0      = bface_v0
mesh.bface_v1      = bface_v1
mesh.bface_bc_tag  = bface_bc_tag   # integer tag per face

bc = [nd.BoundaryCondition(A=1.0, B=0.0)]   # tag 0 → vacuum

solver = nd.FixedSourceSolverUnstructured2D(
    mats      = m,
    mesh      = mesh,
    bc        = bc,
    epsilon   = 1e-10,
    max_inner = 1000,
    omega     = 1.9,    # SOR relaxation factor
)
result = solver.solve([q] * n_cells)   # volumetric source per cell
```

See `examples/k_eigenvalue.py` and `examples/time_dependent.py` for further examples.

## Boundary conditions

| Type | A | B |
|------|---|---|
| Zero-flux (approx. vacuum) | 1.0 | 0.0 |
| Marshak vacuum | (1−&alpha;)/(4(1+&alpha;)) | D/2 |
| Reflective | 0.0 | 1.0 |

The `ndiffusion.boundary_conditions(Dg, alpha)` helper constructs the coefficient array from an albedo value `alpha` (0 = vacuum, 1 = reflective).

## Standalone C++ driver

To build and run the 1-D reference problems without Python:

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
    solver_2d.hpp           2-D structured and unstructured solver declarations
    solver_3d.hpp           3-D solver declarations (placeholder)
  src/
    solver_1d.cpp               1-D solver implementation
    solver_2d_structured.cpp    structured 2-D implementation
    solver_2d_unstructured.cpp  unstructured 2-D implementation
    main.cpp                    standalone driver (1-D reference problems)
  python/
    bindings.cpp            pybind11 bindings → ndiffusion._core

src/ndiffusion/
  __init__.py               re-exports from _core + create utilities
  create.py                 load cross-section data from JSON / .npz / .csv
  mesh.py                   mesh utilities
  data/
    problem_setup.json      named problem definitions

tests/
  test_1d_k_eigenvalue.py
  test_1d_time_dependent.py
  test_1d_fixed_source.py
  test_2d_k_eigenvalue.py
  test_2d_time_dependent.py
  test_2d_fixed_source.py

examples/
  k_eigenvalue.py
  time_dependent.py
```

## Running tests

```bash
pytest
```

The test suite lives in `tests/` and is configured via `pyproject.toml`.

The `Testing/` directory is a generated CMake/CTest artifact and is not part of the source test suite.

## API docs

`Doxyfile` configures Doxygen for the C++ sources under `cpp/include/ndiffusion`, `cpp/src`, and `cpp/python`. To generate the HTML documentation:

```bash
doxygen Doxyfile
```

Or, after configuring CMake:

```bash
cmake --build build --target docs
```

The output is written to `docs/doxygen/html/`.

## Future work

**Geometry**
- 3-D structured geometry (x-y-z) and 3-D unstructured (tetrahedra/hexahedra)
- Gmsh `.msh` file reader for unstructured meshes

**Physics**
- Delayed neutron precursor groups in the time-dependent solver (currently prompt-only)
- Adjoint flux solver for sensitivity and perturbation analysis
- Depletion coupling — Bateman equations for nuclide inventory evolution

**Solvers and performance**
- CMFD (Coarse Mesh Finite Difference) acceleration for unstructured k-eigenvalue convergence
- Preconditioned Krylov solver (CG or GMRES + ILU) as an alternative to SOR for the fixed-source problem
- OpenMP parallelism for the spatial sweep loops
