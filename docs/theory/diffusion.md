# The multigroup diffusion equation

## Equations

With the energy range split into $G$ groups, the steady multigroup diffusion
equation for the scalar flux $\phi_g$ of group $g$ is
{cite:p}`duderstadt1976`

$$
-\nabla \cdot D_g \nabla \phi_g + \Sigma_{r,g}\,\phi_g
  = \sum_{g' \neq g} \Sigma_{s,g \leftarrow g'}\,\phi_{g'}
  + \frac{\chi_g}{k} \sum_{g'} \nu\Sigma_{f,g'}\,\phi_{g'} + q_g ,
$$

with diffusion coefficient $D_g$, removal cross section $\Sigma_{r,g}$, group
transfer $\Sigma_{s,g \leftarrow g'}$, fission spectrum $\chi_g$, nu-fission
cross section $\nu\Sigma_{f,g'}$ and external source $q_g$. Writing $A$ for the
leakage, removal and transfer operator on the left and $F$ for the fission
production gives the three problems the solvers handle:

| Problem | Equation |
|---|---|
| k-eigenvalue | $A\phi = \frac{1}{k} F\phi$, no external source |
| fixed source | $A\phi = q$, no fission |
| time dependent | $\frac{1}{v_g}\frac{\partial \phi_g}{\partial t} = -(A\phi)_g + (F\phi)_g + \text{delayed}$ |

The fixed-source solvers carry no fission term: a subcritical multiplying system
with a source is outside their scope. The time-dependent equations and delayed
neutrons are in {doc}`kinetics`.

## Removal and scatter

The removal cross section is the absorption plus all scattering out of the
group, less the self-scatter:

$$
\Sigma_{r,g} = \Sigma_{a,g} + \sum_{g'} \Sigma_{s,g' \leftarrow g} - \Sigma_{s,g \leftarrow g}.
$$

Self-scatter cancels between the removal and the in-scatter, so it is left out
of both: the scatter matrix used by the solvers has a zero diagonal. It is
stored `scatter[m][g_to][g_from]`, transfer *into* the first index *from* the
second.

## Fission

In the usual separable form the fission source into group $g$ is
$\chi_g \sum_{g'} \nu\Sigma_{f,g'} \phi_{g'}$. When the spectrum depends on the
energy of the neutron causing fission, the source is a full matrix,
$\sum_{g'} F_{g g'} \phi_{g'}$. The solvers take that form when `chi` is all
zeros and `nusigf` holds `n_groups * n_groups` entries per material. The two
agree when $F_{g g'} = \chi_g\,\nu\Sigma_{f,g'}$.

## Boundary conditions

Every boundary condition is a Robin condition on the outward normal derivative,

$$
A\,\phi + B\,\frac{\partial \phi}{\partial n} = 0 .
$$

A reflective (symmetry) boundary is $A = 0$, $B = 1$, and a zero-flux boundary
$A = 1$, $B = 0$. A vacuum or partially reflecting surface follows from the
diffusion-theory partial currents
$J^{\pm} = \phi/4 \mp (D/2)\,\partial\phi/\partial n$. Requiring the incoming
current to be a fraction $\alpha$ (the albedo) of the outgoing one,
$J^- = \alpha J^+$, gives

$$
\frac{1 - \alpha}{4(1 + \alpha)}\,\phi + \frac{D}{2}\,\frac{\partial \phi}{\partial n} = 0 ,
$$

the Marshak condition. $\alpha = 0$ is vacuum and $\alpha = 1$ reflective.

On a symmetry axis of a cylinder or sphere the face area is zero, so the
condition there drops out of the equations entirely; the solvers require it to
be reflective rather than silently ignore anything else.

## Units

Lengths are in centimeters and cross sections in inverse centimeters. The
flux of a k-eigenvalue problem has an arbitrary amplitude;
{func}`ndiffusion.normalize_to_power` scales it to a total power. Sources and
fluxes are per unit volume, and a 1-D slab is per unit area, a 1-D cylinder or
2-D XY problem per unit height.
