# Iterative solution

No solver assembles the full system matrix. Each works with the coefficients of
one group at a time and couples the groups, and fission, by iteration.

## Within a group

| Mesh | Within-group solve |
|---|---|
| 1-D | direct tridiagonal solve (Thomas algorithm) |
| 2-D structured | line relaxation: a tridiagonal solve along each row in $x$, sweeping the rows |
| 2-D unstructured | point Gauss-Seidel; successive over-relaxation in the fixed-source solver |

The within-group operator - leakage plus removal - is symmetric positive
definite, so the 2-D k-eigenvalue solvers can instead solve it with a
Jacobi-preconditioned conjugate gradient method (`use_cg=True`). Its iteration
count grows with the square root of the condition number rather than the
condition number itself, so it pulls ahead as the mesh is refined; the
k-eigenvalue and flux shape are the same either way.

## Across groups

Scattering couples the groups. Each sweep solves the groups in turn, from fast
to thermal, using the newest flux of every other group in the in-scatter
source (Gauss-Seidel). With down-scatter only and a direct within-group solve,
as in 1-D, one sweep is exact. Otherwise - up-scatter, the relaxation methods in
2-D, or the deferred correction on a non-orthogonal unstructured mesh - the
sweep repeats until the flux stops changing.

## Power iteration

The k-eigenvalue solvers find the fundamental mode of
$A\phi = \frac{1}{k} F \phi$ by power iteration. Starting from a flat flux of
unit norm, each outer iteration solves for the next iterate and takes the
eigenvalue from its norm:

$$
A\,\tilde\phi = F\,\phi^{(n)},
\qquad
k^{(n+1)} = \lVert\tilde\phi\rVert_2,
\qquad
\phi^{(n+1)} = \tilde\phi / k^{(n+1)},
$$

with the inner solves warm-started from the previous iterate. Since every
iterate has unit norm, $k^{(n+1)}$ converges to the eigenvalue. The iteration
stops when both the flux change $\lVert\phi^{(n+1)} - \phi^{(n)}\rVert_2$ and
the change in $k$ fall below `epsilon`. The flux check alone is not enough:
with a dominance ratio near 1 the flux can change slowly while $k$ is still
moving.

Power iteration converges at the dominance ratio, the ratio of the second
eigenvalue to the first. Large, loosely coupled cores have ratios close to 1
and need many outer iterations; `max_outer` caps them, and a solve that hits the
cap warns.

## Time steps

A time step is a fixed-source-like solve with a $1/(v_g \Delta t)$ term added to
the removal, and the fission source treated implicitly - reassembled from the
newest flux inside the Gauss-Seidel sweep. That keeps the scheme stable through
a supercritical transient, but the sweep then has to resolve the multiplication
as well as the scatter coupling, and only the $1/(v\Delta t)$ term keeps the
iteration contracting. Near critical it converges at roughly $k$ per sweep.

The solvers accelerate that fixed point with Aitken extrapolation
{cite:p}`aitken1926`: the convergence ratio is estimated from successive
changes, and once it has held steady the iterate jumps to the limit of the
geometric series. The jump is safeguarded - the ratio's stability is judged
against $1 - \sigma$, and the step is capped relative to the flux - and
self-correcting, since convergence is still measured across the sweep. In the
hardest case tested (exactly critical, no leakage, $1/(v\Delta t)$ at 3% of
$\Sigma_a$) this cuts the sweeps per step from thousands to a few dozen.

## Convergence warnings

Every solve that stops at an iteration cap - `max_outer`, `max_inner`, or the
inner cap within a time step - returns its last iterate, sets `converged` to
`False` and issues an `ndiffusion.ConvergenceWarning`. A transient's flag stays
`False` for the rest of the run once any step has hit the cap.
