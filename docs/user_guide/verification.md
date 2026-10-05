# Solution verification

The method of nearby problems estimates the spatial discretization error of a
solution on a single mesh, without a refined mesh or an analytic reference. It
needs scipy (`pip install "ndiffusion[nearby]"`).

The recipe:

1. Solve the problem numerically.
2. Fit a smooth curve through the numerical flux, separately within each
   material so the fit never differentiates across a cross-section jump.
3. Substitute the fit into the continuous diffusion operator. What is left over
   is a residual source $r = L[\phi_\text{fit}] - S$, and the fit is then the
   exact solution of the "nearby problem" $L[\phi] = S + r$.
4. Solve the nearby problem on the same mesh. Its exact solution is known - it
   is the fit - so $\phi_\text{nearby} - \phi_\text{fit}$ estimates the true
   error $\phi_\text{numerical} - \phi_\text{exact}$.

The second derivatives of the fit enter the leakage term, so they must be more
accurate than the scheme itself. On structured meshes the fit is a quintic
spline; on the unstructured mesh it is a least-squares polynomial of up to
fourth degree over a stencil of same-material neighbors.

## Fixed source

`nearby_fixed_source` takes a constructed fixed-source solver, the materials,
the source, and a description of the geometry, which the solver does not
expose: `medium_map`, `edges_x` and `geometry` in 1-D, plus `edges_y` in 2-D, or
`mesh=` for the unstructured solver.

A manufactured problem with a known answer, $\phi = \cos(\pi x / 2R)$ on a half
slab:

```python
import numpy as np
import ndiffusion as nd

D, sigma_a, R, cells = 1.0, 0.2, 10.0, 40
B = np.pi / (2 * R)
edges = np.linspace(0.0, R, cells + 1)
x = 0.5 * (edges[:-1] + edges[1:])
mmap = [0] * cells
mats = nd.materials.one_group(D=D, sigma_a=sigma_a)
exact = np.cos(B * x)
source = (D * B**2 + sigma_a) * exact

solver = nd.FixedSourceSolver(mats, mmap, edges, nd.Geometry.Slab,
                              [nd.BoundaryCondition(A=1.0, B=0.0)],
                              epsilon=1e-12, max_inner=5000)
result = nd.nearby_fixed_source(solver, mats, source, medium_map=mmap,
                                edges_x=edges, geometry=nd.Geometry.Slab)

true_error = result.numerical.flux[:, 0] - exact
estimate = result.error_estimate[:, 0]
```

Here the largest error is about $1.4\times10^{-5}$, and the estimate matches it
to within 0.1%.

The result holds the `numerical` and `nearby` solver results, the
`curve_fit`, the `residual` and the `error_estimate`, all shaped
`(n_cells, n_groups)`.

## k-eigenvalue

`nearby_k_eigenvalue` takes the k-eigenvalue solver and a fixed-source solver
built on the same materials, mesh and boundary conditions. The nearby problem
is solved by a power iteration that reuses the fixed-source solver for each
outer step. Along with the flux estimate it returns `k_curve_fit` and
`k_nearby`, and `k_nearby - k_curve_fit` estimates the eigenvalue error.

The fits are independent in each material, so they need not agree at a material
interface. The flux error estimate is most reliable away from interfaces, and on
a strongly heterogeneous core the eigenvalue estimate is better read as an
indicator of size than as a correction.

## Adjoint problems

`make_adjoint_materials(mats)` returns the adjoint cross sections (see
{doc}`materials`). The diffusion and removal operator is self-adjoint, so any
solver run on them solves the adjoint problem: the eigenvalue matches the forward
one and the flux is the importance function.
