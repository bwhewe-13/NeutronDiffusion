# Boundary conditions

Every boundary condition is a Robin condition

$$
A\,\phi + B\,\frac{\partial \phi}{\partial n} = 0,
$$

with $n$ the outward normal, so the same coefficients mean the same thing on
every edge. A `BoundaryCondition(A, B)` holds one pair, and the solvers take one
per energy group.

| Type | A | B |
|------|---|---|
| Zero flux (approximate vacuum) | 1 | 0 |
| Marshak vacuum, albedo $\alpha$ | $(1-\alpha)/(4(1+\alpha))$ | $D/2$ |
| Reflective | 0 | 1 |

`nd.boundary_conditions(D, alpha)` builds the Marshak list from the diffusion
coefficient of the edge material, one per group, and an albedo (0 for vacuum, 1
for reflective):

```python
import ndiffusion as nd

vacuum = nd.boundary_conditions([1.4, 0.4], 0.0)   # two groups
```

## Structured meshes

The 1-D and 2-D structured solvers take one `BoundaryCondition` per group on
each edge:

| Solver | Edge | Argument | Default |
|---|---|---|---|
| 1-D | right (outer) | `bc` | required |
| 1-D | left (inner) | `bc_left` | reflective |
| 2-D | right, x = x_max | `bc_x` | required |
| 2-D | top, y = y_max | `bc_y` | required |
| 2-D | left, x = x_min | `bc_x_left` | reflective |
| 2-D | bottom, y = y_min | `bc_y_bottom` | reflective |

The optional edges default to reflective, which is what makes a half slab or a
quarter core the natural model. A full core sets all four:

```python
import numpy as np

mats = nd.materials.one_group(D=1.0, sigma_a=0.05, nusigf=0.06)
nx = ny = 20
edges = np.linspace(0.0, 100.0, nx + 1)
vacuum = nd.boundary_conditions([1.0], 0.0)

full_core = nd.KEigenSolver2D(mats, [0] * (nx * ny), edges, edges,
                              nd.Geometry2D.XY,
                              bc_x=vacuum, bc_y=vacuum,
                              bc_x_left=vacuum, bc_y_bottom=vacuum,
                              max_outer=1000)
```

Where the low edge is the r = 0 axis - a 1-D cylinder or sphere starting at the
center, or the bottom of an RZ mesh - the face area is zero, so any other
condition there would be silently ignored. The constructor raises instead; start
the mesh at r > 0 to model an inner surface.

## Unstructured meshes

The unstructured solvers index `bc` by boundary tag (`mesh.bface_bc_tag`):

```text
bc[tag * n_groups + g]      # length = n_bc_types * n_groups
```

so a two-group problem needs two entries even for a single tag. The constructor
rejects a length that is not a multiple of `n_groups`, or that does not cover
every tag the mesh uses - otherwise those boundaries would silently behave as
reflective, and an all-reflective system just reads as k-infinity.

Meshes from `ndiffusion.layouts` tag their faces `layouts.SYMMETRY` (0) or
`layouts.OUTER` (1), and `layouts.boundary_conditions(mesh, D, albedo)` returns
the matching list: reflective on the symmetry cuts, Marshak outside.

Edges joined periodically are interior faces, not boundaries, and take no
boundary condition (see {doc}`geometry`).
