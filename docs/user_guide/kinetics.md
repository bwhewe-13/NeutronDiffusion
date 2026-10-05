# Reactor kinetics

The time-dependent solvers model delayed neutron precursors:

$$
\begin{aligned}
\frac{1}{v_g}\frac{\partial \phi_g}{\partial t} &= -A_g\,\phi_g + \text{scatter}
  + (1-\beta)\,\chi_{p,g}\,F + \sum_i \chi_{d,i,g}\,\lambda_i C_i, \\
\frac{d C_i}{d t} &= \beta_i F - \lambda_i C_i,
\qquad F = \sum_{g'} \nu\Sigma_{f,g'}\,\phi_{g'}.
\end{aligned}
$$

## A transient

Start from a genuine steady state, perturb it, and step:

```python
import numpy as np
import ndiffusion as nd

cells = 60
edges = np.linspace(0.0, 100.0, cells + 1)
mmap = [0] * cells
bc = [nd.BoundaryCondition(A=1.0, B=0.0)]

mats = nd.materials.one_group(D=3.850204978408833, sigma_a=0.1532, nusigf=0.1570)
mats.velocity = [2.2e5]                                   # cm/s

res = nd.KEigenSolver(mats, mmap, edges, nd.Geometry.Sphere, bc,
                      epsilon=1e-10, max_outer=2000).solve()

# A k-eigenvalue flux is only stationary once nusigf is divided by keff.
critical = nd.scale_to_critical(mats, res.keff)
delayed = nd.make_delayed_data(nd.DELAYED_U235_6GROUP, G=1, n_mat=1,
                               chi=critical.chi)

solver = nd.TimeDependentSolver(critical, mmap, edges, nd.Geometry.Sphere, bc,
                                initial_flux=res.flux, delayed=delayed,
                                epsilon=1e-10, max_inner=2000)

perturbed = nd.scale_to_critical(mats, res.keff)
perturbed.removal = [0.1531]                              # remove some absorber
solver.update_materials(perturbed)                        # step insertion at t = 0
out = solver.run(dt=1e-2, n_steps=100)
out.precursors.shape                                      # (n_cells, n_precursor)
```

Precursors default to equilibrium with the initial flux, which is what a
transient starting from steady state needs; pass `initial_precursors` to
override. Omit `delayed` for prompt-only kinetics. `examples/kinetics.py` is a
complete version, comparing a 50 cent insertion with and without delayed
neutrons.

`update_materials` is the only way to perturb a transient - `step` takes no
external source. It swaps the cross sections and rebuilds the operator while
keeping the flux and precursors; `n_mat` and `n_groups` must not change. The new
cross sections take effect at the start of the next step, so a step insertion at
$t_n$ holds across the whole of $[t_n, t_n + \Delta t]$. Drive a ramp by calling
it once per step with interpolated cross sections.

## Delayed neutron data

`make_delayed_data` builds a `DelayedNeutronData` from per-material dicts with
`Beta` and `Lambda` and optionally `ChiDelayed` and `ChiPrompt`. A single dict is
broadcast to every material. `DELAYED_U235_6GROUP` is the Keepin six-group U-235
set.

The underlying arrays are flat: `lambda_` (with a trailing underscore, since
`lambda` is a Python keyword) is `[n_precursor]`, `beta` is
`[n_mat * n_precursor]`, `chi_delayed` is `[n_mat * n_precursor * n_groups]`
(each spectrum must sum to 1), and `chi_prompt` is `[n_mat * n_groups]` or
empty.

When `chi_prompt` is empty it is derived rather than defaulted to
`Materials.chi`:

$$
\chi_p = \frac{\chi - \sum_i \beta_i\,\chi_{d,i}}{1 - \beta},
$$

so the prompt and delayed parts always add back up to the total spectrum. With
the usual `chi_delayed = chi` this reduces to `chi_p = chi`. Pass `chi_prompt`
explicitly only if `Materials.chi` is itself the prompt spectrum.

Fission-matrix mode is supported. There is no separable spectrum, so the split
is applied to the matrix: the production cross section is the column sum
$P_{g'} = \sum_g F_{g g'}$, the delayed yield $\beta_i\,\chi_{d,i,g}\,P_{g'}$ is
subtracted from the tabulated matrix, and the part emitted within the step is
added back. The two representations agree exactly when the matrix is separable.
`Materials.chi` is all zeros in this mode, so it cannot serve as the
`ChiDelayed` fallback - supply one.

## Time differencing

All three time-dependent solvers take a weight `theta`, as a constructor
argument and as a settable property:

$$
\frac{\phi^{n+1} - \phi^n}{v\,\Delta t} = \theta\,R(\phi^{n+1}, C^{n+1})
  + (1-\theta)\,R(\phi^n, C^n).
$$

`theta = 1` (the default) is backward Euler, first order in $\Delta t$;
`theta = 0.5` is Crank-Nicolson, second order. The weighting is applied
consistently to the flux equation and the precursor balance, so the whole
transient is second order at `theta = 0.5`. On the infinite-medium
point-kinetics problem in `tests/test_kinetics.py` the observed order is 2.00,
and at a fixed step the error is about 140 times smaller than backward Euler's.
Values outside `[0.5, 1]` raise `ValueError`: that is exactly the A-stable
range, and below it the fast spatial modes would diverge at any useful step
size.

Whatever the weight, $C^{n+1}$ eliminates in closed form, which folds the
delayed source into a step-dependent effective fission spectrum plus a source
known from the old precursors:

$$
\begin{aligned}
\chi_{\text{eff},g} &= (1-\beta)\,\chi_{p,g}
  + \sum_i \chi_{d,i,g}\,\frac{\beta_i \lambda_i\,\theta\Delta t}{1 + \lambda_i\theta\Delta t}, \\
Q_{d,g} &= \sum_i \chi_{d,i,g}\,\lambda_i \left[
  C_i^n\,\frac{1 - (1-\theta)\lambda_i\Delta t}{1 + \theta\lambda_i\Delta t}
  + \frac{(1-\theta)\,\Delta t\,\beta_i F^n}{1 + \theta\lambda_i\Delta t}
  + \frac{1-\theta}{\theta}\,C_i^n \right].
\end{aligned}
$$

As $\theta\Delta t \to 0$ the effective spectrum tends to $(1-\beta)\chi_p$, prompt
only; as $\theta\Delta t \to \infty$ it tends to the total fission spectrum, so a
critical system with equilibrium precursors is an exact fixed point at any step
size and any `theta`. Fission is evaluated at the new time level inside the
Gauss-Seidel sweep, so the scheme stays unconditionally stable through a
supercritical transient.

### Stiff modes and `theta = 0.5`

Both ends of the range are unconditionally stable, but only backward Euler
damps the stiff modes. `theta = 0.5` is A-stable and not L-stable: a mode with
$\zeta = |\lambda|\Delta t \gg 1$ has amplification
$(1 - (1-\theta)\zeta)/(1 + \theta\zeta)$, which is about $1/\zeta$ at
`theta = 1` but tends to $-1$ at `theta = 0.5`, so it decays slowly with an
alternating sign instead of being killed.

Whether that ever shows up depends on how much stiff content the state actually
contains. The stiff modes are the mesh-scale spatial harmonics: on the 8-cell
slab in `tests/test_kinetics.py` the checkerboard mode decays at about
$6\times10^5$ per second, so $\Delta t = 10^{-3}$ puts it about 560 times beyond
the resolved range, and a localized flux bump still keeps about 80% of its size
after 20 steps, reversing sign on every one, while backward Euler removes it in
a single step. A smooth, mode-shaped perturbation excites almost none of that -
the TWIGL step insertion in `tests/test_benchmarks.py` moves the flux shape by
only about $10^{-5}$ and shows no ringing at all. When you do have a stiff
perturbation, damp it first and then switch:

```python
solver.update_materials(perturbed)   # step insertion at t = 0
solver.theta = 1.0
solver.run(1e-3, 2)                  # two damped steps
solver.theta = 0.5
solver.run(1e-3, 100)                # second order from here on
```

## Choosing `dt` and `max_inner`

With an implicit fission source, the inner Gauss-Seidel sweep resolves the
multiplication as well as the scatter coupling, and only the $1/(v_g\Delta t)$
diagonal term keeps that iteration contracting. When $1/(v\Delta t) \ll
\Sigma_r$, a near-critical problem converges at roughly $k$ per sweep.

The solvers apply Aitken extrapolation to that fixed point: the convergence
ratio is estimated from successive iterate changes and, once it has held steady,
the iterate jumps to the limit of the geometric series. In the worst case
tested - exactly critical, zero leakage, $1/(v\Delta t)$ at 3% of $\Sigma_a$ -
this cuts the iterations per step from thousands to a few dozen. The
extrapolation is safeguarded and self-correcting, since convergence is still
measured across the sweep.

The first step that still hits `max_inner` issues a `ConvergenceWarning` naming
the solver and the residual, once per solver, and `result().converged` stays
`False` from then on. Do not trust a transient that warned.
