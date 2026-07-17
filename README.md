# ndiffusion

Multigroup neutron diffusion solver for 1-D and 2-D geometries. Written in C++17; exposed to Python via pybind11.

## Capabilities

### 1-D (slab, cylinder, sphere)
- Arbitrary number of energy groups and material regions
- Vacuum, reflective, and albedo boundary conditions
- **k-eigenvalue solver** - matrix-free power iteration; A&phi; = (1/k)B&phi;
- **Fixed-source solver** - direct solve of A&phi; = q for a user-supplied volumetric source
- **Time-dependent solver** - backward-Euler time stepping, unconditionally stable
- Per-group Thomas (TDMA) tridiagonal solver inside a Gauss-Seidel group sweep
- Harmonic-mean diffusion coefficients at material interfaces

### 2-D structured (Cartesian XY or axisymmetric RZ)
- Finite-difference 5-point stencil on an nx x ny Cartesian grid
- Left (x=0) and bottom (y=0) boundaries hardcoded as reflective; right and top boundaries take user-specified Robin BCs per group
- **k-eigenvalue solver** - line-TDMA x-sweeps inside a Gauss-Seidel outer iteration
- **Fixed-source solver** - same spatial sweep; solves A&phi; = q directly
- **Time-dependent solver** - backward-Euler stepping using the same line-TDMA sweep

### 2-D unstructured (triangles and/or quadrilaterals)
- Cell-centred finite-volume method (FVM)
- Arbitrary Robin BCs per boundary tag; harmonic-mean interface diffusion coefficients
- **k-eigenvalue solver** - power iteration with point Gauss-Seidel inner solve
- **Fixed-source solver** - point SOR (successive over-relaxation) inner solve
- **Time-dependent solver** - backward-Euler stepping with point Gauss-Seidel

## Installation

Requires a C++17 compiler, Python >= 3.9, and `pybind11 >= 2.12`.

```bash
pip install .
```

For development:

```bash
pip install -e .
```

Python edits under `src/ndiffusion/` are picked up immediately; after editing C++
sources, re-run `pip install -e .` to rebuild the extension.

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
    max_outer  = 500,
)
result = solver.solve()
assert result.converged
print(f"keff = {result.keff:.8f}")   # -> 1.00000475
```

Every result carries a `converged` flag; always check it before trusting the
answer (an unconverged run returns the last iterate without raising).

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

bc = [nd.BoundaryCondition(A=1.0, B=0.0)]   # tag 0 -> vacuum

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

## Transport cross sections

Multigroup transport libraries tabulate a total (or absorption) cross section, a
scatter matrix and fission data; the solvers want `D`, a *removal* cross section
and a diagonal-free scatter matrix. `make_materials_from_transport` does the
conversion and returns a ready-to-use `Materials`:

```python
mats = nd.make_materials_from_transport([uo2, mox, moderator], G=7)
```

Each input dict holds `SigT` **or** `Siga` (the other is derived from the scatter
matrix), a `Scat` matrix, `nuSigf`, and optionally `chi`, `SigTr` or `Scat1`:

| | |
|---|---|
| `D[g] = 1 / (3 Sigma_tr[g])` | `Sigma_tr` from a tabulated `SigTr`, else the P1 outflow correction `SigT - sum_g' Scat1[g->g']` (`transport_correction="none"` gives the uncorrected `1/(3 SigT)`) |
| `Sigma_r[g] = SigT[g] - Scat[g->g]` | the outflow correction cancels here, so removal uses the **uncorrected** total |
| `scatter[g_to][g_from]` | input is assumed `Scat[g_from][g_to]` (the transport convention) and is transposed; pass `scatter_orientation="to_from"` for data already in solver order |

`transport_to_diffusion(data, G)` exposes the same transform for a single
material and returns a plain dict, useful for inspecting the derived `D`,
`Removal` and `SigTr` before building a `Materials`.

See `examples/transport_cross_sections.py` for a runnable end-to-end example.

## Boundary conditions

| Type | A | B |
|------|---|---|
| Zero-flux (approx. vacuum) | 1.0 | 0.0 |
| Marshak vacuum | (1-&alpha;)/(4(1+&alpha;)) | D/2 |
| Reflective | 0.0 | 1.0 |

The `ndiffusion.boundary_conditions(Dg, alpha)` helper constructs the coefficient array from an albedo value `alpha` (0 = vacuum, 1 = reflective).

## Adjoint & solution verification

Two Python helpers layer on top of the compiled solvers (they reuse the existing
solver classes, so no rebuild is involved):

- **Adjoint** - `ndiffusion.make_adjoint_materials(mats)` returns the adjoint
  cross sections (group scatter transposed, `chi`/`nusigf` swapped). Running any
  solver on them solves the adjoint (importance) problem; the k-eigenvalue is
  identical to the forward one, and the flux is the neutron importance function.

- **Method of nearby problems (MNP)** - a discretization-error estimator
  (`ndiffusion.nearby_fixed_source`, `ndiffusion.nearby_k_eigenvalue`). It fits a
  smooth curve through the numerical flux, substitutes it into the continuous
  diffusion operator to form a residual source, and re-solves the resulting
  "nearby problem" whose exact solution is the fit - so `nearby - fit` estimates
  the true error. Works in 1-D, 2-D structured, and 2-D unstructured (a
  high-order least-squares reconstruction supplies the Laplacian on the FVM
  mesh). Requires SciPy: `pip install ndiffusion[nearby]`.

```python
solver = nd.FixedSourceSolver(mats, medium_map, edges_x, nd.Geometry.Slab, bc)
result = nd.nearby_fixed_source(solver, mats, source,
                                medium_map=medium_map, edges_x=edges_x,
                                geometry=nd.Geometry.Slab)
# result.error_estimate estimates (numerical - exact) flux
```

## Standalone C++ driver

To build and run the 1-D reference problems without Python:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/cpp/ndiffusion_driver
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
    bindings.cpp            pybind11 bindings -> ndiffusion._core

src/ndiffusion/
  __init__.py               re-exports from _core + create/mesh utilities
  create.py                 make_materials / make_medium_map / boundary_conditions
  transport.py              transport -> diffusion cross-section transform
  adjoint.py                make_adjoint_materials - forward -> adjoint transform
  nearby.py                 method of nearby problems (fixed-source & k-eigenvalue)
  mesh.py                   load_gmsh - Gmsh .msh import for unstructured meshes

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
  transport_cross_sections.py
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
- General boundary conditions on all edges (1-D currently hardcodes symmetry at the
  left/inner edge; 2-D structured hardcodes left and bottom as reflective)
- Non-orthogonal correction for the unstructured FVM two-point flux approximation
  (accuracy degrades on skewed meshes)

**Physics**
- Delayed neutron precursor groups in the time-dependent solver (currently
  prompt-only), and implicit treatment of the fission source (currently explicit)
- Adjoint flux solver for sensitivity and perturbation analysis
- Depletion coupling - Bateman equations for nuclide inventory evolution

**Solvers and performance**
- Flip the default inner solver for the 2-D k-eigenvalue solvers to the
  within-group CG (now a `use_cg` constructor option; default remains
  Gauss-Seidel, overridable via `NDIFFUSION_KEIG_CG=1`); extend CG to the
  fixed-source and time-dependent solvers, replacing hand-tuned SOR
- Power-iteration acceleration (Wielandt shift or Chebyshev extrapolation);
  CMFD (Coarse Mesh Finite Difference) for unstructured k-eigenvalue convergence
- Zero-copy numpy arrays across the pybind11 boundary (fluxes and sources
  currently cross as Python lists)
- OpenMP parallelism for the spatial sweep loops

**Testing**
- Published two-group benchmark regressions are in `tests/test_benchmarks.py`
  (1-D Ringhals-4 slab, 2-D TWIGL, 2-D IAEA PWR on the stepped quarter core);
  still to add: BIBLIS-2D and C5G7 diffusion via `tools/c5g7_fuel_mesh.py`
