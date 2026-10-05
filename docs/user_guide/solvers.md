# Solvers

There are nine solver classes - three problem types on three kinds of mesh - and
they are all built and run the same way: the constructor takes the materials,
the mesh and the boundary conditions, and then `solve()` (or `step()` / `run()`
for a transient) does the work.

| | k-eigenvalue | fixed source | time dependent |
|---|---|---|---|
| 1-D slab, cylinder, sphere | `KEigenSolver` | `FixedSourceSolver` | `TimeDependentSolver` |
| 2-D structured XY, RZ | `KEigenSolver2D` | `FixedSourceSolver2D` | `TimeDependentSolver2D` |
| 2-D unstructured polygons | `KEigenSolverUnstructured2D` | `FixedSourceSolverUnstructured2D` | `TimeDependentSolverUnstructured2D` |

Every solver exposes `n_cells` and `n_groups`, which give the shape of its flux.

## k-eigenvalue

The k-eigenvalue solvers find the fundamental mode of

$$
A\,\phi = \frac{1}{k}\,F\,\phi,
$$

with $A$ the leakage, removal and group-transfer operator and $F$ the fission
production, by power iteration. Each outer iteration solves $A\phi = F\phi/k$
for the current fission source and updates $k$ from the ratio of successive
production integrals.

```python
import numpy as np
import ndiffusion as nd

mats = nd.materials.two_group([(1.4, 0.4, 0.010, 0.080, 0.02, 0.006, 0.135)])
edges = np.linspace(0.0, 100.0, 101)
solver = nd.KEigenSolver(mats, [0] * 100, edges, nd.Geometry.Slab,
                         nd.boundary_conditions([1.4, 0.4], 0.0),
                         epsilon=1e-8, max_outer=1000, max_inner=1000)
result = solver.solve()
result.keff, result.iterations, result.converged
```

`epsilon` is the convergence tolerance on the flux change between outer
iterations, `max_outer` caps the power iterations and `max_inner` the inner
sweeps of each one. The result is a `DiffusionResult` with `flux`, `keff`,
`iterations`, `residual` and `converged`.

The inner solve differs by mesh. The structured solvers use tridiagonal (Thomas)
solves - along the mesh in 1-D, along x lines in 2-D - inside a Gauss-Seidel
sweep over energy groups. The unstructured solver uses point Gauss-Seidel over
the cells. The 2-D k-eigenvalue solvers also have a matrix-free, Jacobi
preconditioned conjugate gradient inner solver, chosen with `use_cg=True` or
`set_use_cg(True)`; the `NDIFFUSION_KEIG_CG=1` environment variable makes it the
default.

## Fixed source

The fixed-source solvers solve $A\phi = q$ for a volumetric source $q$ with no
fission:

```python
source = np.zeros((solver.n_cells, 2))
source[40:60, 0] = 1.0
fixed = nd.FixedSourceSolver(mats, [0] * 100, edges, nd.Geometry.Slab,
                             nd.boundary_conditions([1.4, 0.4], 0.0),
                             epsilon=1e-10, max_inner=2000)
flux = fixed.solve(source).flux
```

The source is per unit volume, shaped `(n_cells, n_groups)` or flat. The same
solver can be called with any number of sources. The unstructured fixed-source
solver uses point successive over-relaxation, with the relaxation factor
`omega` (default 1, plain Gauss-Seidel); values around 1.8 to 1.9 usually
converge much faster on fine meshes.

## Time dependent

The time-dependent solvers advance

$$
\frac{1}{v_g}\frac{\partial \phi_g}{\partial t} = -A_g\,\phi_g + \text{scatter}
+ \text{fission} + \text{delayed}
$$

from an initial flux, with `Materials.velocity` giving the neutron speed of each
group. `step(dt)` advances one step, `run(dt, n_steps)` several, and `result()`
returns the current state. Delayed neutrons, the time differencing and how to
perturb a transient are covered in {doc}`kinetics`.

## Convergence warnings

A solve that stops at an iteration cap returns its last iterate with
`converged = False` and issues an `ndiffusion.ConvergenceWarning` naming the
solver and the residual. It is a `UserWarning`, so the standard `warnings`
filters apply:

```python
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("error", nd.ConvergenceWarning)
    result = solver.solve()     # raises instead of returning an unconverged answer
```

The k-eigenvalue solvers warn when either the power iteration or an inner solve
hits its cap. A transient warns once per solver, on the first step that hits
`max_inner`, and its `converged` flag stays `False` from then on.

## Interrupting a solve

Long-running methods release the GIL and check for Ctrl-C once per outer
iteration, sweep or time step, so a `KeyboardInterrupt` stops the solve rather
than waiting for it to finish. All solver state is held in ordinary arrays, so
the solver object is still usable afterwards.
