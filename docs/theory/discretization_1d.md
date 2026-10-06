# Discretization in 1-D

The 1-D solvers use a cell-centered finite difference (equivalently, finite
volume) scheme on the edges `edges_x`. Integrating the diffusion equation over
cell $i$ with volume $V_i$ and face areas $S_{i \pm 1/2}$ gives

$$
- c_{i-1/2}\,\phi_{i-1} + \left(c_{i-1/2} + c_{i+1/2} + \Sigma_{r,i}\right)\phi_i
- c_{i+1/2}\,\phi_{i+1} = \text{in-scatter} + \text{source},
$$

with the face couplings

$$
c_{i+1/2} = \frac{D_{i+1/2}\, S_{i+1/2}}{h_{i+1/2}\, V_i}, \qquad
D_{i+1/2} = \frac{2 D_i D_{i+1}}{D_i + D_{i+1}}, \qquad
h_{i+1/2} = \tfrac{1}{2}(h_i + h_{i+1}).
$$

The interface coefficient is the harmonic mean of the neighboring $D$, and the
gradient is taken over the center-to-center distance $h_{i+1/2}$. Using the
local cell width instead would only agree on a uniform mesh, and on a
non-uniform one leaves the scheme inconsistent and not conservative.

## Geometry

The geometry enters only through the face areas and cell volumes - per unit
area for the slab and per unit height for the cylinder:

| Geometry | Face area $S$ at radius $r$ | Volume of $[r_a, r_b]$ |
|---|---|---|
| slab | $1$ | $r_b - r_a$ |
| cylinder | $2\pi r$ | $\pi(r_b^2 - r_a^2)$ |
| sphere | $4\pi r^2$ | $\tfrac{4}{3}\pi(r_b^3 - r_a^3)$ |

A cylinder or sphere that starts at $r = 0$ has a zero-area inner face, so the
inner boundary condition has no effect there.

## Boundary conditions

The Robin condition at an edge is imposed through a ghost cell: with the ghost
value $\phi_G$ and the edge cell $\phi_N$, the face value is their average and
the normal derivative their difference over the cell width,

$$
A\,\frac{\phi_G + \phi_N}{2} + B\,\frac{\phi_G - \phi_N}{h_N} = 0 .
$$

On the right edge this is kept as an extra row of the tridiagonal system. On the
left edge the ghost is eliminated, $\phi_G = \alpha\,\phi_0$, and folded into
the first cell's diagonal, so a reflective edge ($\alpha = 1$) adds nothing.

## Solving the system

For each group the equations above form a tridiagonal system, solved directly
by the Thomas algorithm. The groups are coupled by scattering, which is handled
by Gauss-Seidel over the groups: each group is solved with the in-scatter from
the latest values of the others, and the sweep repeats until the flux stops
changing. Fission is handled by the outer iteration described in
{doc}`iteration`.
