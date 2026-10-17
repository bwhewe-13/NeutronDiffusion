# The method of nearby problems

The method of nearby problems {cite:p}`roy2007` estimates the discretization
error of a solution using only the mesh it was computed on. The idea is to
construct a second problem, close to the original, whose exact solution is
known, and measure the scheme's error on that one instead.

## The nearby problem

Let $L\phi = S$ be the continuous problem and $\phi_h$ its numerical solution.

1. Fit a smooth function $\phi_\text{fit}$ through $\phi_h$.
2. Substitute the fit into the continuous operator. It does not satisfy the
   original equation exactly; the remainder
   $r = L[\phi_\text{fit}] - S$ is a known source.
3. The nearby problem $L\phi = S + r$ then has $\phi_\text{fit}$ as its exact
   solution, by construction.
4. Solve the nearby problem with the same scheme on the same mesh, giving
   $\phi_{\text{nearby},h}$.

Its error, $\phi_{\text{nearby},h} - \phi_\text{fit}$, is known exactly. When the
nearby problem is close to the original - when the fit is close to the true
solution and smooth enough - the scheme makes nearly the same error on both, so

$$
\phi_h - \phi_\text{exact} \approx \phi_{\text{nearby},h} - \phi_\text{fit}.
$$

## Fitting

The leakage term needs the second derivatives of the fit, so they must be
accurate to well below the scheme's own error. The fit is made separately in
each material region, since the flux gradient jumps at a material interface:

| Mesh | Fit |
|---|---|
| 1-D | quintic interpolating spline over each material block (cubic or linear on short blocks) |
| 2-D structured | quintic splines along each direction, per material block |
| 2-D unstructured, `bc=` given | one constrained quintic tensor spline per material region, all fit together (below) |
| 2-D unstructured, `fit="local"` | least-squares polynomial of up to fourth degree over a stencil of same-material neighbors |

The residual is the continuous operator evaluated pointwise at the cell
centers. Sources are per unit volume, so one definition serves all three
geometries.

### The constrained spline fit

Independent fits on either side of an interface disagree there by about the
scheme's own error, which is enough to spoil the estimate near interfaces. On
the unstructured mesh the fit therefore ties the regions together. Each
connected material region gets one quintic tensor-product B-spline on its
bounding box, with knots about 2-3 cells apart (2.5 shortest diffusion lengths,
clipped to that range). All regions and groups are fit at once by least squares
to the cell values, with heavily weighted rows at two points on every face:

- interface faces: $\phi$ and $D\,\partial\phi/\partial n$ continuous;
- boundary faces: the face's Robin condition.

Two variants are used, chosen from the mesh:

- **smooth**: also the diffusion equation and its gradient on both sides of
  every interface and boundary face. Accurate on layouts whose interfaces are
  straight or gently curved and run to the boundary.
- **corner**: picked when the interfaces meet at a corner (turning more than 30
  degrees) or enclose a small region (under 8 knot spans across). There the
  extra rows pose an ill-conditioned problem and the fit stops matching its
  data, so the diffusion equation is imposed softly at every cell center
  instead, and the singular functions $r^\alpha\Phi(\theta)$ of each material
  corner are added to the regions around it. The exponents $\alpha$ come from
  the corner angles and diffusion coefficients; each function is cut off
  radially before the shortest interface out of the corner ends, so it adds no
  kink where an interface stops.

A fixed-source fit always uses the corner variant, whose interior rows carry
the source.

## Eigenvalue problems

For a k-eigenvalue problem the fit also defines an eigenvalue,

$$
k_\text{fit} = \frac{\int F\phi_\text{fit}\,dV}{\int M\phi_\text{fit}\,dV},
$$

with $M$ the loss operator, and the residual is
$r = M\phi_\text{fit} - F\phi_\text{fit}/k_\text{fit}$. The nearby problem
$M\phi = F\phi/k + r$ is solved by a power iteration that reuses the
fixed-source solver for each outer step, and $k_\text{nearby} - k_\text{fit}$
estimates the eigenvalue error.

The same estimate is available without the nearby solve. With the adjoint flux
$\phi^+$ (from a solver run on the adjoint cross sections),

$$
k_h - k \approx \sum_c \eta_c, \qquad
\eta_c = \frac{\phi^+_c\,(k_h M\phi_\text{fit} - F\phi_\text{fit})_c V_c}
             {\sum_c \phi^+_c (M\phi_\text{fit})_c V_c},
$$

and the per-cell contributions $\eta_c$ show where the eigenvalue error comes
from. The two estimates agree closely on smooth layouts; where the error is a net
of large contributions of both signs (see below) they can differ by 5-15%.

## The discrete operator and its adjoint

The unstructured solvers never form a matrix, and they treat the non-orthogonal
part of the face flux as a deferred source that is iterated to convergence. What
a converged solve satisfies is nonetheless one linear system, and
`unstructured_operator` assembles it: the loss operator $\mathbf{M}$, including
the non-orthogonal correction $\mathbf{C}$ with the solver's least-squares
gradients, and the production operator $\mathbf{F}$, both volume-integrated.

The explicit matrix gives the exact discrete adjoint. A solver run on adjoint
cross sections transposes the scattering and fission but not $\mathbf{C}$,
because the least-squares gradient correction is not symmetric; on triangles and
hexagons that leaves an error of 1-2% in an adjoint-weighted estimate that does
not shrink with the mesh size. On orthogonal meshes $\mathbf{C} = 0$ and the two
agree.

With `method="direct"`, `nearby_k_eigenvalue` factorizes $\mathbf{M}$ once and
uses it for every step of the nearby power iteration, and takes the adjoint for
the one-shot estimate from the bordered system below. This is usually far faster
than the iterative route.

## Quantities of interest

For a normalized quantity of interest, a ratio of two reaction-rate integrals

$$
J(\phi) = \frac{\int a\cdot\phi\,dV}{\int b\cdot\phi\,dV}
$$

(a region power, or the absorption in an absorber per unit production), the
nearby estimate is $J(\phi_\text{nearby}) - J(\phi_\text{fit})$. `nearby_qoi`
also gives it without the nearby solve. Linearizing about the discrete solution,

$$
J(\phi_\text{nearby}) - J(\phi_\text{fit}) \approx
\Gamma^T\mathbf{V}r + J(\phi_h) - J(\phi_\text{fit}),
$$

where $\Gamma$ solves the generalized adjoint problem
$(\mathbf{M} - \mathbf{F}/k_h)^T\Gamma = \partial J/\partial\phi$ with
$\Gamma^T\mathbf{F}\phi_h = 0$. The ratio makes $J$ independent of the
eigenvector's normalization, so $\partial J/\partial\phi$ is orthogonal to
$\phi_h$, which is exactly the condition for this singular system to have a
solution. It is solved through the bordered matrix

$$
\begin{pmatrix}(\mathbf{M} - \mathbf{F}/k_h)^T & \mathbf{F}\phi_h\\
(\mathbf{F}\phi_h)^T & 0\end{pmatrix},
$$

one factorization for any number of quantities. $J(\phi_\text{fit})$ is
integrated with a degree-4 rule over each cell. The per-cell products
$\Gamma_c (\mathbf{V}r)_c$ form a goal-oriented error map that can drive mesh
refinement. The difference from the nearby-solve estimate is the curvature of
$J$; it shrinks with the mesh size.

## Limits

The estimate is only as good as the fit. The 1-D and structured fits are
independent on either side of a material interface and need not agree there,
so on those meshes the estimate is most reliable away from interfaces.

On layouts with material corners or small enclosed regions the eigenvalue error
is a small net of large contributions of both signs ($\sum_c|\eta_c|$ can be
hundreds of times $|\sum_c \eta_c|$). The local flux estimate stays good there,
but the eigenvalue estimate is better read as an indicator of size, with
$\sum_c|\eta_c|$ as a loose bound.

The fit also needs the mesh to resolve the layout: a few cells per knot span and
several knot spans across each region. On a curved interface it estimates the
error of the scheme on the polygons the mesh uses, not the error of replacing
the curve by those polygons. On a homogeneous problem it recovers the error
closely - see {doc}`../user_guide/verification` for an example.
