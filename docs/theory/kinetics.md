# Kinetics

## Equations

With delayed neutron precursors in $I$ groups, the time-dependent multigroup
equations are {cite:p}`duderstadt1976,keepin1965`

$$
\begin{aligned}
\frac{1}{v_g}\frac{\partial \phi_g}{\partial t} &= -(A\phi)_g
  + (1-\beta)\,\chi_{p,g}\,F + \sum_i \chi_{d,i,g}\,\lambda_i C_i, \\
\frac{d C_i}{d t} &= \beta_i F - \lambda_i C_i,
\qquad F = \sum_{g'} \nu\Sigma_{f,g'}\,\phi_{g'} ,
\end{aligned}
$$

where $A$ is the leakage, removal and transfer operator, $\beta = \sum_i \beta_i$
the total delayed fraction, $\chi_p$ and $\chi_{d,i}$ the prompt and delayed
spectra, and $\lambda_i$ the decay constants. Precursor concentrations are per
unit volume.

## Prompt spectrum

When no prompt spectrum is given it is derived from the total one,

$$
\chi_p = \frac{\chi - \sum_i \beta_i\,\chi_{d,i}}{1 - \beta},
$$

so prompt and delayed emission always add back to $\chi$. With the usual
$\chi_{d,i} = \chi$ this is simply $\chi_p = \chi$. A delayed spectrum that
overshoots the total makes $\chi_p$ negative, and is rejected.

## Theta-weighted time differencing

Both equations are differenced with the same weight $\theta \in [0.5, 1]$:

$$
\frac{\phi^{n+1} - \phi^n}{v\,\Delta t} = \theta\,R^{n+1} + (1-\theta)\,R^n,
\qquad
\frac{C_i^{n+1} - C_i^n}{\Delta t}
  = \theta\left(\beta_i F^{n+1} - \lambda_i C_i^{n+1}\right)
  + (1-\theta)\left(\beta_i F^n - \lambda_i C_i^n\right),
$$

with $R$ the right-hand side of the flux equation. $\theta = 1$ is backward
Euler, first order in $\Delta t$; $\theta = 0.5$ is Crank-Nicolson, second order.
The interval $[0.5, 1]$ is exactly the A-stable range.

The precursor equation is linear in $C_i^{n+1}$, so it solves in closed form:

$$
C_i^{n+1} = \frac{C_i^n\left(1 - (1-\theta)\lambda_i\Delta t\right)
  + \Delta t\,\beta_i\left(\theta F^{n+1} + (1-\theta) F^n\right)}
  {1 + \theta\lambda_i\Delta t}.
$$

Substituting it into the delayed source of the flux equation and collecting the
$F^{n+1}$ terms folds the delayed neutrons into an effective fission spectrum and
a source known from the previous step, $\theta R^{n+1} \ni \theta\left(
\chi_{\text{eff}} F^{n+1} + Q_d\right)$:

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

At $\theta = 1$ these reduce to the backward-Euler forms
$\chi_{\text{eff},g} = (1-\beta)\chi_{p,g} + \sum_i \chi_{d,i,g}\,\beta_i\lambda_i\Delta t/(1 + \lambda_i\Delta t)$
and $Q_{d,g} = \sum_i \chi_{d,i,g}\,\lambda_i C_i^n/(1 + \lambda_i\Delta t)$.

The limits are what make the scheme behave. As $\theta\Delta t \to 0$ the
effective spectrum tends to $(1-\beta)\chi_p$, prompt neutrons only; as
$\theta\Delta t \to \infty$ it tends to $\chi$. A critical system with
equilibrium precursors is therefore an exact fixed point at any step size and
any $\theta$ - which is also why the prompt spectrum is derived as above rather
than defaulted to $\chi$.

Each step is then a source problem with $1/(v_g\theta\Delta t)$ added to the
removal. The fission source $\chi_{\text{eff}} F^{n+1}$ stays implicit,
reassembled from the newest flux inside the group sweep, so the step is stable
through a supercritical transient; {doc}`iteration` covers how that sweep is
accelerated. The $(1-\theta)$ part of the right-hand side is an explicit
residual of the previous flux, and on the unstructured solver it includes the
non-orthogonal correction, so both halves use the same operator.

## Fission-matrix mode

With a fission matrix $F_{g g'}$ there is no separable spectrum to reweight, so
the split is applied to the matrix itself. The production cross section is its
column sum, $P_{g'} = \sum_g F_{g g'}$, which equals $\nu\Sigma_{f,g'}$ when the
matrix is separable. The delayed yield $\beta_i\,\chi_{d,i,g}\,P_{g'}$ is
subtracted from the tabulated matrix and the part emitted within the step added
back, giving an effective matrix that agrees exactly with the separable form
when the matrix is separable.

## Starting a transient

A k-eigenvalue flux is a steady state only if $k = 1$.
{func}`ndiffusion.scale_to_critical` divides $\nu\Sigma_f$ by $k$, which makes
it one, and the precursors default to equilibrium with the initial flux,
$C_i = \beta_i F / \lambda_i$. Without the scaling a transient drifts from the
first step and a reactivity insertion cannot be separated from that drift.
