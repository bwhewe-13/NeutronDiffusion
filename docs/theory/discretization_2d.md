# Discretization on a structured 2-D mesh

The structured 2-D solvers use the same cell-centered scheme as
{doc}`discretization_1d`, on the tensor-product mesh of `edges_x` and `edges_y`.
Integrating over cell $(i, j)$ gives a five-point stencil,

$$
a_P\,\phi_{i,j} - a_W\,\phi_{i-1,j} - a_E\,\phi_{i+1,j}
  - a_S\,\phi_{i,j-1} - a_N\,\phi_{i,j+1} = \text{in-scatter} + \text{source},
$$

where each neighbor coefficient is the face area times the harmonic-mean
diffusion coefficient over the center-to-center distance, divided by the cell
volume, and $a_P$ is their sum plus the removal cross section. Cells are
numbered with $y$ fastest: cell $(i, j)$ is $i\,n_y + j$.

## Geometry

| Geometry | Coordinates | x-face area | y-face area | Volume |
|---|---|---|---|---|
| `XY` | Cartesian $(x, y)$, per unit height | $\Delta y$ | $\Delta x$ | $\Delta x\,\Delta y$ |
| `RZ` | $x = z$ (axial), $y = r$ (radial) | $\pi(r_b^2 - r_a^2)$ | $2\pi r\,\Delta z$ | $\pi(r_b^2 - r_a^2)\,\Delta z$ |

In RZ a mesh whose `edges_y` starts at zero sits on the axis, where the radial
face area vanishes and the bottom boundary condition must stay reflective.

## Boundary conditions

The right and top edges carry their Robin conditions through ghost cells as in
1-D; the left and bottom edges, reflective by default, have their ghost folded
into the diagonal of the first row or column. Each edge takes one condition
per group.

## Solving the system

Each group's system is solved by line relaxation: the cells of each row along
$x$ form a tridiagonal system, solved directly with the Thomas algorithm, with
the $y$ neighbors taken from their latest values. The sweep over rows repeats
inside a Gauss-Seidel iteration over the groups. The k-eigenvalue solver can use
a Jacobi-preconditioned conjugate gradient solve of each group's system instead
(`use_cg=True`); see {doc}`iteration`.
