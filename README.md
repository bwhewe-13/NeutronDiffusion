# ndiffusion

Multigroup neutron diffusion solver for 1-D and 2-D geometries. Written in C++17; exposed to Python via pybind11.

## Capabilities

### 1-D (slab, cylinder, sphere)
- Arbitrary number of energy groups and material regions
- Vacuum, reflective, and albedo boundary conditions
- **k-eigenvalue solver** - matrix-free power iteration; A&phi; = (1/k)B&phi;
- **Fixed-source solver** - direct solve of A&phi; = q for a user-supplied volumetric source
- **Time-dependent solver** - theta-weighted time stepping, unconditionally stable
- Per-group Thomas (TDMA) tridiagonal solver inside a Gauss-Seidel group sweep
- Harmonic-mean diffusion coefficients at material interfaces

### Reactor kinetics (all three dimensionalities)
- **Delayed neutron precursors** - any number of precursor groups, per-material
  delayed fractions and delayed fission spectra
- **Implicit fission source**, so the scheme stays unconditionally stable
  through a supercritical transient
- **Second-order time differencing** - `theta = 0.5` is Crank-Nicolson; the
  default `theta = 1` is backward Euler
- **Mid-transient perturbation** via `update_materials`, for reactivity insertions

### 2-D structured (Cartesian XY or axisymmetric RZ)
- Finite-difference 5-point stencil on an nx x ny Cartesian grid
- Left (x=0) and bottom (y=0) boundaries hardcoded as reflective; right and top boundaries take user-specified Robin BCs per group
- **k-eigenvalue solver** - line-TDMA x-sweeps inside a Gauss-Seidel outer iteration
- **Fixed-source solver** - same spatial sweep; solves A&phi; = q directly
- **Time-dependent solver** - theta-weighted stepping using the same line-TDMA sweep

### 2-D unstructured (triangles and/or quadrilaterals)
- Cell-centred finite-volume method (FVM)
- Arbitrary Robin BCs per boundary tag; harmonic-mean interface diffusion coefficients
- **k-eigenvalue solver** - power iteration with point Gauss-Seidel inner solve
- **Fixed-source solver** - point SOR (successive over-relaxation) inner solve
- **Time-dependent solver** - theta-weighted stepping with point Gauss-Seidel

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

## Reactor kinetics

The time-dependent solvers model delayed neutron precursors:

```
(1/v_g) dphi_g/dt = -A_g phi_g + scatter
                  + (1-beta) chi_p,g F + sum_i chi_d,i,g lambda_i C_i
        dC_i/dt   = beta_i F - lambda_i C_i,   F = sum_g' nuSigf_g' phi_g'
```

The time discretisation weights the right-hand side between the two time levels
with a factor `theta` (see **Time differencing** below). Whatever the weight,
`C^{n+1}` still eliminates in closed form, which folds the delayed source into a
`dt`-dependent **effective fission spectrum** plus a source known from the old
precursors:

```
chi_eff,g = (1-beta) chi_p,g
          + sum_i chi_d,i,g beta_i lambda_i (theta dt) / (1 + lambda_i theta dt)
Q_d,g     = sum_i chi_d,i,g lambda_i
            [ C_i^n (1 - (1-theta) lambda_i dt) / (1 + theta lambda_i dt)
            + (1-theta) dt beta_i F^n           / (1 + theta lambda_i dt)
            + ((1-theta)/theta) C_i^n ]
```

At the default `theta = 1` those collapse to the backward-Euler forms
`chi_eff,g = (1-beta) chi_p,g + sum_i chi_d,i,g beta_i lambda_i dt / (1 + lambda_i dt)`
and `Q_d,g = sum_i chi_d,i,g lambda_i C_i^n / (1 + lambda_i dt)`.

As `theta*dt -> 0` the effective spectrum tends to `(1-beta) chi_p` (prompt
only); as `theta*dt -> inf` it tends to the total fission spectrum, so a critical
system with equilibrium precursors is an exact fixed point at any step size *and*
any `theta`. Fission is evaluated at the new time level inside the Gauss-Seidel
sweep, so the scheme stays unconditionally stable.

```python
delayed = nd.make_delayed_data(nd.DELAYED_U235_6GROUP, G=2, n_mat=3, chi=mats.chi)

# Start from a genuine steady state: a k-eigenvalue flux is only stationary
# once nusigf is divided by keff.
res = nd.KEigenSolver2D(mats, medium_map, edges_x, edges_y, geom, bc_x, bc_y).solve()
critical = nd.scale_to_critical(mats, res.keff)

solver = nd.TimeDependentSolver2D(
    mats=critical, medium_map=medium_map,
    edges_x=edges_x, edges_y=edges_y, geom=nd.Geometry2D.XY,
    bc_x=bc_x, bc_y=bc_y, initial_flux=res.flux,
    delayed=delayed,          # omit for prompt-only kinetics
)
solver.update_materials(perturbed)   # step insertion at t = 0
out = solver.run(dt=2e-3, n_steps=250)
out.precursors                        # [cells * n_precursor], per unit volume
```

Precursors default to equilibrium with the initial flux, which is what a
transient starting from steady state needs; pass `initial_precursors` to
override. Drive a ramp by calling `update_materials` once per step with
interpolated cross sections.

`DelayedNeutronData` arrays are flat: `lambda_` is `[n_precursor]`, `beta` is
`[n_mat * n_precursor]`, `chi_delayed` is `[n_mat * n_precursor * n_groups]`
(each spectrum must sum to 1), and `chi_prompt` is `[n_mat * n_groups]` or empty.

When `chi_prompt` is omitted it is **derived** rather than defaulted to
`Materials.chi`:

```
chi_p = (chi - sum_i beta_i chi_d,i) / (1 - beta)
```

so the prompt and delayed parts always add back up to the total spectrum, and
`chi_eff` still tends to `chi` as `dt -> inf` for *any* delayed spectrum. With
the usual `chi_delayed = chi` this reduces to `chi_p = chi`. Pass `chi_prompt`
explicitly only if `Materials.chi` is itself the prompt spectrum.

**Fission-matrix mode** is supported. There is no separable spectrum, so the
split is applied to the matrix: the production cross section is the column sum
`P[g'] = sum_g F[g][g']` (total neutrons emitted per fission caused by a
group-`g'` neutron), the delayed yield `beta_i chi_d,i[g] P[g']` is subtracted
from the tabulated matrix, and the part emitted within the step is added back.
The two representations agree exactly when the matrix is separable. Note that
`Materials.chi` is all zeros in this mode, so it cannot serve as the
`ChiDelayed` fallback - supply one.

**Time differencing.** All three time-dependent solvers take a `theta` weight,
as a constructor argument and as a settable property:

```
(1/(v dt)) (phi^{n+1} - phi^n) = theta * R(phi^{n+1}, C^{n+1})
                               + (1-theta) * R(phi^n,   C^n)
```

`theta = 1` (the default) is backward Euler, first order in `dt`; `theta = 0.5`
is Crank-Nicolson, second order. The weighting is applied consistently to the
flux equation *and* the precursor balance, so the whole transient is second
order at `theta = 0.5` - on the infinite-medium point-kinetics problem in
`tests/test_kinetics.py` the observed order is 2.00, and at a fixed `dt` the
error is ~140x smaller than backward Euler's:

```python
solver = nd.TimeDependentSolver(..., delayed=delayed, theta=0.5)
solver.theta = 1.0    # or change it mid-transient
```

Values outside `[0.5, 1]` raise `ValueError`: that is exactly the A-stable
range, and below it the fast spatial modes - the ones quantified below, decaying
at `~1e5` to `1e6` per second - would diverge at any useful step size.

Both ends of the range are unconditionally stable, but only backward Euler
*damps* the stiff modes. `theta = 0.5` is A-stable and not L-stable: a mode with
`zeta = |lambda| dt >> 1` has amplification `(1 - (1-theta) zeta)/(1 + theta
zeta)`, which is `~1/zeta` at `theta = 1` but tends to `-1` at `theta = 0.5`, so
it decays slowly with an alternating sign instead of being killed.

Whether that ever shows up depends on how much stiff content the state actually
contains. The stiff modes here are the mesh-scale spatial harmonics: on the
8-cell slab in `tests/test_kinetics.py` the checkerboard mode decays at
`~6e5 / s`, so `dt = 1e-3` puts it ~560x beyond the resolved range and a
localized flux bump still retains ~80% of its size after 20 steps, reversing sign
on every one, while backward Euler annihilates it in a single step. A smooth,
mode-shaped perturbation excites almost none of that - the TWIGL step insertion
in `tests/test_benchmarks.py` moves the flux shape by only ~1e-5 and shows no
ringing at all. Nothing diverges either way. When you do have a stiff
perturbation, damp it first and then switch:

```python
solver.update_materials(perturbed)   # step insertion at t = 0
solver.theta = 1.0
solver.run(dt, 2)                    # two damped steps
solver.theta = 0.5
solver.run(dt, n_steps)              # second order from here on
```

`update_materials` takes effect at the start of the next step, which is what a
step insertion at `t_n` means: the new cross sections hold across the whole of
`[t_n, t_n + dt]`, the explicitly weighted term included.

**Choosing `dt` and `max_inner`.** An implicit fission source means the inner
Gauss-Seidel sweep resolves the multiplication as well as the scatter coupling,
and only the `1/(v_g*dt)` diagonal term keeps that iteration contracting. When
`1/(v*dt) << Sigma_r` a near-critical problem converges at roughly `k` per
sweep. The solvers apply **Aitken extrapolation** to that fixed point: the
convergence ratio is estimated from successive iterate changes and, once it has
held steady, the iterate jumps to the limit of the geometric series. In the
worst case tested - exactly critical, zero leakage, `1/(v*dt)` at 3% of
`Sigma_a` - this cuts the iterations per step from thousands to a few dozen.
The extrapolation is safeguarded (ratio stability judged against `1 - sigma`,
and the jump capped relative to `||phi||`) and is self-correcting, since
convergence is still measured across the sweep. Any step that nonetheless hits
`max_inner` prints a warning to stderr naming the solver and the residual -
never trust a transient that warned.

See `examples/kinetics.py` for a runnable end-to-end transient.

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
  kinetics.py               delayed neutron data + critical scaling helpers
  nearby.py                 method of nearby problems (fixed-source & k-eigenvalue)
  mesh.py                   load_gmsh - Gmsh .msh import for unstructured meshes

tests/
  test_1d_k_eigenvalue.py
  test_1d_time_dependent.py
  test_1d_fixed_source.py
  test_2d_k_eigenvalue.py
  test_2d_time_dependent.py
  test_2d_fixed_source.py
  test_kinetics.py
  test_benchmarks.py

examples/
  k_eigenvalue.py
  time_dependent.py
  kinetics.py
  transport_cross_sections.py
  c5g7_quarter_core.py
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
- Automatic time-step control, using the difference between the `theta = 1` and
  `theta = 0.5` answers as a local error estimate
- Improved quasi-static or adiabatic kinetics, factoring the flux into a point
  kinetics amplitude and a slowly varying shape
- Thermal-hydraulic feedback (Doppler / moderator density) driving
  `update_materials` from the power distribution
- Sensitivity and perturbation analysis built on the adjoint importance
  function (the adjoint materials transform `make_adjoint_materials` now exists)
- Depletion coupling - Bateman equations for nuclide inventory evolution

**Solvers and performance**
- Flip the default inner solver for the 2-D k-eigenvalue solvers to the
  within-group CG (now a `use_cg` constructor option; default remains
  Gauss-Seidel, overridable via `NDIFFUSION_KEIG_CG=1`); extend CG to the
  fixed-source and time-dependent solvers, replacing hand-tuned SOR
- Power-iteration acceleration (Wielandt shift or Chebyshev extrapolation);
  CMFD (Coarse Mesh Finite Difference) for unstructured k-eigenvalue convergence.
  The transient inner iteration already uses Aitken extrapolation
  (`FissionAccelerator`); the same idea would apply to the k-eigenvalue outer
- Zero-copy numpy arrays across the pybind11 boundary (fluxes and sources
  currently cross as Python lists)
- OpenMP parallelism for the spatial sweep loops

**Testing**
- Published two-group benchmark regressions are in `tests/test_benchmarks.py`
  (1-D Ringhals-4 slab, 2-D TWIGL, 2-D IAEA PWR on the stepped quarter core, and
  2-D BIBLIS full-core PWR); the C5G7 quarter core runs end-to-end in
  `examples/c5g7_quarter_core.py` (mesh + 7-group transport cross sections +
  unstructured solver); still to add: a CI-sized C5G7 diffusion regression
- The TWIGL kinetics transient (`TestTwiglKinetics`) currently validates against
  its own static reactivity rather than the benchmark's published power history;
  digitising that history would turn it into a true published regression
