# Geometry and meshes

Each solver family has its own mesh description. In every case a material index
per cell (`medium_map`, or `material_id` on an unstructured mesh) selects a row
of the {doc}`materials` arrays.

## 1-D

`edges_x` holds the `n_cells + 1` cell edges and `medium_map` the material of
each cell. The geometry is `nd.Geometry.Slab`, `Cylinder` or `Sphere`. The mesh
need not be uniform: interface diffusion coefficients are a harmonic mean taken
over the center-to-center distance, so a graded mesh stays second order.

A cylinder or sphere that starts at r = 0 has a symmetry point at the center.
Start the mesh at r > 0 to model an annulus or a hollow sphere, with a boundary
condition on the inner surface (see {doc}`boundary_conditions`).

`make_medium_map` builds `medium_map` from a compact region list. It accepts
`(mat_id, n_cells)` tuples, plain cell counts, widths with `total_cells`, or
physical lengths with `edges`:

```python
import numpy as np
import ndiffusion as nd

nd.make_medium_map([(0, 61), (1, 39)])              # 61 cells of 0, then 39 of 1
nd.make_medium_map([61, 39])                        # same, ids assigned 0, 1

edges = np.linspace(0.0, 100.0, 101)
mmap = nd.make_medium_map([(0, 45.0), (1, 55.0)], edges=edges)   # by cell center
```

The last form assigns each cell by the position of its center, so it is exact on
a non-uniform mesh.

## 2-D structured

A structured mesh is the tensor product of `edges_x` (`nx + 1` values) and
`edges_y` (`ny + 1` values). Cells are numbered row-major with `y` fastest: cell
`(i, j)` is index `i * ny + j`, in `medium_map`, in a source, and in
`result.flux`, so `result.flux.reshape(nx, ny, n_groups)` gives the grid.

`nd.Geometry2D.XY` is Cartesian. `nd.Geometry2D.RZ` is axisymmetric, with `x`
the axial coordinate `z` and `y` the radius `r`; a mesh whose `edges_y` starts at
zero sits on the axis.

## 2-D unstructured

An `UnstructuredMesh2D` is a cell-centered finite volume mesh of simple polygons
with at least three vertices each - triangles, quadrilaterals, hexagons, and the
clipped cells a symmetry cut leaves behind:

| Field | Meaning |
|---|---|
| `vx`, `vy` | vertex coordinates |
| `cell_vertices` | vertex indices of every cell, concatenated, in order around the cell (either winding) |
| `cell_offsets` | `n_cells + 1` offsets; cell `c` owns `cell_vertices[cell_offsets[c]:cell_offsets[c + 1]]` |
| `material_id` | material index per cell |
| `bface_v0`, `bface_v1` | the two vertices of each boundary face |
| `bface_bc_tag` | boundary tag per face, an index into the `bc` list |
| `periodic_a0/a1`, `periodic_b0/b1` | edge pairs joined periodically (below) |

Two unit squares side by side, every boundary face tagged 0:

```python
mesh = nd.UnstructuredMesh2D()
mesh.vx = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
mesh.vy = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
mesh.cell_vertices = [0, 1, 4, 3,   1, 2, 5, 4]
mesh.cell_offsets  = [0, 4, 8]
mesh.material_id   = [0, 0]
mesh.bface_v0      = [0, 1, 2, 5, 4, 3]
mesh.bface_v1      = [1, 2, 5, 4, 3, 0]
mesh.bface_bc_tag  = [0, 0, 0, 0, 0, 0]
nd.validate_mesh(mesh)
```

The solver constructors validate the connectivity themselves. A mesh must be
conforming: an edge shared by three cells, a vertex sitting inside another
cell's edge (a hanging node), coincident vertices that were never merged, and
zero-area cells are all rejected, because each would quietly cut the mesh
interior apart or leave a face with no area.

Supporting geometry queries, for painting materials and for post-processing:

```python
cx, cy = nd.cell_centroids(mesh)   # area centroids, the same ones the solver uses
areas  = nd.cell_areas(mesh)
```

### Gmsh meshes

`load_gmsh` reads a `.msh` file (it needs the `mesh` extra). Physical surface
groups become material ids and physical curve groups become boundary tags, both
numbered in order of their Gmsh tag. Because adding a group to the file can
shift every index after it, the loader also attaches `mesh.region_names` and
`mesh.bc_names`, and the names are what stay stable.

Triangles and quadrangles of any order are read through their corner nodes, so a
curved element becomes the straight-sided polygon through its corners.

## Assigning materials

Nothing requires `material_id` to be decided when the geometry is built.
`assign_materials` paints it on as a separate step, so a single geometry serves
many material layouts. The spec is a dict keyed by region name or id, a callable
`(x, y) -> int` evaluated at cell centroids, or a per-cell sequence:

```python
mesh = nd.load_gmsh("core.msh")          # geometry + region labels

# by Gmsh physical-group name
nd.assign_materials(mesh, {"fuel": 0, "reflector": 1})

# by region id
nd.assign_materials(mesh, {1: 0, 2: 0, 3: 1})

# by position, evaluated at cell centroids
nd.assign_materials(mesh, lambda x, y: 0 if x*x + y*y < R*R else 1)
```

It rewrites `material_id` in place and returns the mesh; pass `copy=True` to keep
the input intact. Repainting between solves is safe, because each solver takes
its own copy of the mesh at construction:

```python
for name, painter in painters.items():
    nd.assign_materials(mesh, painter)
    results[name] = nd.KEigenSolverUnstructured2D(mats[name], mesh, bc).solve()
```

A dict spec must map every region the mesh uses, and names are checked against
the mesh's own `region_names`, so adding a physical group to the `.msh` is an
error rather than a silent shift of every index after it.

On a large mesh, precomputing the assignment with numpy is about twice as quick
as a per-cell callable:

```python
cx, cy = nd.cell_centroids(mesh)
nd.assign_materials(mesh, np.where(cx > x0, 1, 0))
```

## Preset layouts and symmetry orientations

`ndiffusion.layouts` composes three independent pieces: a geometry, an
orientation (the symmetry sector, and which boundaries are cuts), and a layout
(a painter `(x, y) -> material`).

```python
from ndiffusion import layouts

mats = nd.materials.two_group([
    (1.4, 0.4, 0.010, 0.080, 0.02, 0.006, 0.135),   # core
    (1.3, 0.3, 0.001, 0.020, 0.03, 0.000, 0.000),   # reflector
])
mesh = layouts.cartesian_mesh(width=100.0, h=2.0, orientation="quarter")
nd.assign_materials(mesh, layouts.core_reflector(core_radius=35.0))
bc = layouts.boundary_conditions(mesh, D=[1.4, 0.4], albedo=0.0)
keff = nd.KEigenSolverUnstructured2D(mats, mesh, bc, max_outer=1000).solve().keff
```

| Geometry | Orientations |
|---|---|
| `cartesian_mesh` | `full`, `half`, `quarter`, `eighth` (45 degree octant), `infinite` |
| `hex_mesh` | `full`, `half`, `sector120`, `sector60`, `sector30`, `infinite` |

Sector cuts slice cells into smaller polygons, which the finite volume solver
takes directly. Boundary faces are tagged `layouts.SYMMETRY` or `layouts.OUTER`,
and `layouts.boundary_conditions` pairs them - reflective on the cuts, Marshak
with the given albedo outside. `infinite` tags every boundary reflective, giving
k-infinity.

Cuts are reflective by default. `symmetry="rotational"` joins the two cuts
periodically instead, imposing rotational symmetry without a mirror - the right
choice for a spiral or pinwheel loading, where reflecting solves a different
problem. Only whole rotational periods qualify (Cartesian `half` and `quarter`;
hex `sector120` and `sector60`). The 45 degree octant and 30 degree hex wedge
are fundamental domains only by virtue of the mirror, and are rejected.

| Layout | |
|---|---|
| `homogeneous()` | one material; baseline |
| `core_reflector(core_radius=...)` or `(core_half_width=...)` | two regions, circular or square core |
| `checkerboard(pitch)` | alternating materials |
| `annular(radii, materials)` | concentric rings |
| `hex_rings(pitch, ring_materials)` | material per hexagonal ring |
| `with_rods(base, positions, radius, material)` | overlays rods on another layout |

Painters are plain functions of position, so they compose and work on any
geometry. `with_rods` gives the perturbation pair the mesh/material split was
built for - one geometry, two layouts, a rod worth:

```python
unrodded = layouts.core_reflector(core_radius=35.0)
rodded   = layouts.with_rods(unrodded, [(0.0, 0.0)], 5.0, material=2)

nd.assign_materials(mesh, unrodded); k0 = solve(mesh)
nd.assign_materials(mesh, rodded);   k1 = solve(mesh)
rho = (k1 - k0) / (k1 * k0)
```

## Periodic boundaries

Edges listed in `mesh.periodic_a0/a1` are joined to `periodic_b0/b1`, vertex to
corresponding vertex (`a0` to `b0`, `a1` to `b1`), and become interior faces
rather than boundaries, so the flux is continuous across them. That
correspondence fixes the rigid transform, so both a translation (a repeating
lattice) and a rotation (a symmetry sector) can be expressed, and the solver
works out which from the geometry.

A fully periodic homogeneous square has no leakage anywhere, so it reproduces
k-infinity exactly. `layouts` builds the rotational case for you through
`symmetry="rotational"`.
