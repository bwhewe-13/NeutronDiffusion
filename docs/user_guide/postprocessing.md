# Post-processing

A k-eigenvalue flux is a shape with an arbitrary amplitude. The helpers in
`ndiffusion.postprocess`, all exported at the top level, turn a result into the
usual core quantities.

```python
import numpy as np
import ndiffusion as nd

mats = nd.materials.two_group([
    (1.4, 0.4, 0.010, 0.080, 0.02, 0.006, 0.135),   # fuel
    (1.3, 0.3, 0.001, 0.020, 0.03, 0.000, 0.000),   # reflector
])
nx = ny = 30
edges = np.linspace(0.0, 150.0, nx + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
medium_map = [0 if max(x, y) < 100.0 else 1 for x in centers for y in centers]
vacuum = nd.boundary_conditions([1.3, 0.3], 0.0)
res = nd.KEigenSolver2D(mats, medium_map, edges, edges, nd.Geometry2D.XY,
                        vacuum, vacuum, max_outer=1000).solve()

vol = nd.cell_volumes(edges, nd.Geometry2D.XY, edges)        # or cell_volumes(mesh)
flux = nd.normalize_to_power(res.flux, mats, medium_map, vol, total_power=3.0e9)
q = nd.power_density(flux, mats, medium_map)                 # W/cm^3
by_material = nd.region_powers(q, vol, medium_map)
fq = nd.peaking_factors(q, vol)                              # cell max / fueled average
absorption = nd.reaction_rate(flux, mats, medium_map, "absorption")
```

| Function | Returns |
|---|---|
| `cell_volumes(edges_x, geom, edges_y)` | cell volumes in solver order, for 1-D, 2-D structured, or a mesh |
| `reaction_rate(flux, mats, material_map, kind)` | `"nu-fission"`, `"fission"`, `"absorption"` or `"removal"` rate density |
| `power_density(flux, mats, material_map)` | fission power density |
| `normalize_to_power(flux, mats, material_map, volumes, total_power)` | flux scaled to the given total power |
| `region_powers(density, volumes, regions)` | total per region (material, assembly, ...) |
| `peaking_factors(density, volumes, regions=None)` | maximum over the average of the fueled cells, or of the region averages |

`Materials` stores only `nusigf`, so the fission rate and the power use `nu`
(default `NU_U235 = 2.43`) and `kappa` (default `KAPPA_U235`, 200 MeV in
joules); pass your own for other fuels. The absorption rate is the removal
cross section minus the out-scatter, so it works with the solver's own data.

Volumes match the solvers' own: a 1-D slab is per unit area, and a 1-D
cylinder, 2-D XY and unstructured mesh per unit height, so the total power is
too. Pass `regions=` to `peaking_factors` (for example an assembly id per cell)
for region-averaged peaking.

## Saving results

`save_result(path, result, solver=solver, **metadata)` writes any result to an
`.npz` file along with the ndiffusion version and solver name. `load_result`
reads it back with the same attribute names plus a `metadata` dict:

```python
import os, tempfile

path = os.path.join(tempfile.mkdtemp(), "core.npz")
nd.save_result(path, res, case="reference")
back = nd.load_result(path)
back.keff, back.metadata["case"]
```
