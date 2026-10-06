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
| 2-D unstructured | least-squares polynomial of up to fourth degree over a stencil of same-material neighbors |

The residual is the continuous operator evaluated pointwise at the cell
centers. Sources are per unit volume, so one definition serves all three
geometries.

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

## Limits

The estimate is only as good as the fit. Block-wise fits are independent on
either side of a material interface and need not agree there, so the estimate is
most reliable away from interfaces; on a strongly heterogeneous problem the
eigenvalue estimate is better read as an indicator of size than as a
correction. On a homogeneous problem it recovers the error closely - see
{doc}`../user_guide/verification` for an example.
