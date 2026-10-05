# Quickstart

## A 1-D k-eigenvalue problem

A bare uranium sphere of radius 100 cm, one energy group, zero flux at the
surface:

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
edges = np.linspace(0.0, 100.0, cells + 1)

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

`medium_map` gives the material index of each cell, `edges_x` the cell edges, and
`bc` the boundary condition on the outer edge, one per energy group. The center
of a sphere is a symmetry point, so nothing is needed there.

## Arrays in and out

Inputs take any array-like, so lists work too. `result.flux` comes back as a
numpy array of shape `(n_cells, n_groups)`. Per-cell inputs - a fixed source, an
initial flux - take the same `(n_cells, n_groups)` shape or the flat row-major
equivalent, `flux[cell * n_groups + g]`. A transposed array has the right size
but the wrong layout, so it is rejected rather than silently scrambled.

## Convergence

Every result carries a `converged` flag. A solve that stops at its iteration cap
returns the last iterate and issues an `ndiffusion.ConvergenceWarning`, which the
standard `warnings` filters control - for example, to make it an exception:

```python
import warnings
warnings.simplefilter("error", nd.ConvergenceWarning)
```

A long solve can be stopped with Ctrl-C; the solvers check for interrupts once
per outer iteration.

## A 2-D structured problem

A square of the same material, `R` cm on a side, with vacuum on the right and
top edges. The left and bottom edges default to reflective, so this is one
quarter of a `2R x 2R` core:

```python
R, nx, ny = 100.0, 40, 40

solver = nd.KEigenSolver2D(
    mats       = m,
    medium_map = [0] * (nx * ny),
    edges_x    = np.linspace(0.0, R, nx + 1),
    edges_y    = np.linspace(0.0, R, ny + 1),
    geom       = nd.Geometry2D.XY,
    bc_x       = [nd.BoundaryCondition(A=1.0, B=0.0)],   # vacuum right
    bc_y       = [nd.BoundaryCondition(A=1.0, B=0.0)],   # vacuum top
    max_outer  = 500,
)
result = solver.solve()
flux = result.flux.reshape(nx, ny, m.n_groups)   # row i * ny + j is cell (i, j)
```

## An unstructured fixed-source problem

`ndiffusion.layouts` builds common core geometries as an `UnstructuredMesh2D`.
Here a quarter of a 100 cm square core with a uniform source, using a two-group
material set:

```python
from ndiffusion import layouts

mats = nd.materials.two_group([
    # D1,  D2,   Sa1,   Sa2,  S12,  nuSf1, nuSf2
    (1.4, 0.4, 0.010, 0.080, 0.02, 0.000, 0.000),
])
mesh = layouts.cartesian_mesh(width=100.0, h=2.5, orientation="quarter")
bc = layouts.boundary_conditions(mesh, D=[1.4, 0.4], albedo=0.0)

solver = nd.FixedSourceSolverUnstructured2D(mats, mesh, bc, epsilon=1e-10,
                                            max_inner=5000, omega=1.9)
source = np.zeros((solver.n_cells, 2))
source[:, 0] = 1.0                       # fast source, per unit volume
result = solver.solve(source)
```

`omega` is the over-relaxation factor of the point SOR sweep; `omega = 1` is
plain Gauss-Seidel.

## Next steps

- {doc}`user_guide/geometry` covers meshes, material maps and layouts.
- {doc}`user_guide/materials` covers cross sections, including the published
  benchmark tables.
- The `examples/` directory has runnable scripts for each solver family.
