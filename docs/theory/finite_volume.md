# Finite volumes on an unstructured mesh

The unstructured solvers use a cell-centered finite volume scheme. Each cell is
a simple polygon; its unknown is the flux at the area centroid, and integrating
the diffusion equation over the cell turns the leakage into a sum of face
fluxes,

$$
\int_{V_P} -\nabla \cdot D \nabla\phi \, dV
  = -\sum_f D_f\,(\nabla\phi)_f \cdot \mathbf{S}_f ,
$$

with $\mathbf{S}_f$ the face's outward normal times its length. Removal, scatter
and sources are taken as constant over the cell and multiplied by its area.
Cell areas and centroids come from the shoelace formulae, so any polygon works,
hexagons and the clipped cells of a symmetry sector included.

## Face fluxes and non-orthogonality

A two-point difference $(\phi_N - \phi_P)/d$ along the line joining the two
centroids captures the normal gradient only when that line is parallel to the
face normal. In general it is not, and the surface vector is split, following
the over-relaxed approach of {cite:t}`jasak1996`, into a part along the
centroid line $\mathbf{e}$ and a remainder:

$$
\mathbf{E} = \frac{\mathbf{S}\cdot\mathbf{S}}{\mathbf{e}\cdot\mathbf{S}}\,\mathbf{e},
\qquad
\mathbf{T} = \mathbf{S} - \mathbf{E},
\qquad
(\nabla\phi)_f \cdot \mathbf{S}
  = |\mathbf{E}|\,\frac{\phi_N - \phi_P}{d} + (\nabla\phi)_f \cdot \mathbf{T}.
$$

The first term is treated implicitly; the second is a deferred correction,
evaluated from the cell gradients of the previous iterate and moved to the
right-hand side. Since $\mathbf{T}\cdot\mathbf{S} = 0$, the correction involves
only the gradient along the face. On an orthogonal mesh - rectangles, and
regular triangles and hexagons - $\mathbf{T}$ vanishes, the solver detects it at
construction, and the correction is skipped entirely.

Without the correction the scheme is inconsistent on a skewed mesh: refining
it converges to the wrong answer. With it, the eigenvalue error on a
single-material right-triangle mesh falls at close to second order (about 1.7
at the resolutions in `tests/test_mesh_materials.py`, rising with refinement).

## Cell gradients

The correction needs a cell gradient, taken from a weighted least-squares fit
to the differences to the face neighbors, with weights $1/|\mathbf{d}|^2$. The
fit depends only on the geometry, so its coefficients are computed once per
face at construction and the gradient is a weighted sum at run time. A
Green-Gauss gradient would be cheaper but is only first-order accurate on
triangles, which is not enough. A boundary face contributes the boundary value
implied by its Robin condition.

## Interfaces and boundaries

Interior faces use the harmonic mean of the two cells' diffusion coefficients.
A boundary face applies its Robin condition between the cell centroid and the
face, along the normal distance $d_n$, which eliminates the face value:

$$
\phi_f = \frac{B/d_n}{A + B/d_n}\,\phi_P .
$$

Boundary conditions are given per boundary tag, `bc[tag * n_groups + g]`.

## Periodic faces

A periodic pair of edges becomes an ordinary interior face between the two
cells on either side. The face stores where each neighbor's centroid sits as
seen across it (its image under the periodic transform) and the rotation
between the two frames, so the face flux, the least-squares gradient and the
correction need nothing special. A translation (a repeating lattice) and a
rotation (a symmetry sector without a mirror) are both expressible, and the
transform is worked out from the vertex correspondence.

## Solving the system

Each group's system is relaxed point by point - Gauss-Seidel in the
k-eigenvalue and time-dependent solvers, successive over-relaxation with factor
`omega` in the fixed-source solver - or, in the k-eigenvalue solver, solved by
Jacobi-preconditioned conjugate gradients. With the non-orthogonal correction
active, the sweeps also iterate the correction to consistency. See
{doc}`iteration`.
