"""
Reactor kinetics example: a $0.50 step reactivity insertion in a 1-D slab core.

Shows the whole workflow end to end:

  1. solve the k-eigenvalue problem for the steady-state flux;
  2. scale nusigf by 1/keff so that flux is an exact steady state;
  3. build 6-group delayed neutron data and start the transient with
     equilibrium precursors;
  4. perturb the core mid-transient with update_materials;
  5. compare against the same insertion computed prompt-only.

The point of the comparison is that delayed neutrons are not a refinement -
they set the timescale. The delayed run settles near a prompt jump of about 2x
and then creeps upward over seconds; the prompt-only run passes that within a
few milliseconds and keeps going.

Run after installing the package:
    pip install .
    python examples/kinetics.py
"""

import numpy as np

import ndiffusion as nd

# ---------------------------------------------------------------------------
# Core: 1-group slab, fuel surrounded by vacuum
# ---------------------------------------------------------------------------

CELLS = 60
EDGES = list(np.linspace(0.0, 100.0, CELLS + 1))
MEDIUM_MAP = [0] * CELLS
BC = [nd.BoundaryCondition(A=1.0, B=0.0)]  # zero flux at the outer face


def core(removal=0.1532):
    m = nd.Materials()
    m.n_mat = 1
    m.n_groups = 1
    m.D = [3.850204978408833]
    m.removal = [removal]
    m.scatter = [0.0]
    m.chi = [1.0]
    m.nusigf = [0.1570]
    m.velocity = [2.2e5]  # thermal neutron speed, cm/s
    return m


def keff_of(mats):
    solver = nd.KEigenSolver(
        mats, MEDIUM_MAP, EDGES, nd.Geometry.Sphere, BC,
        epsilon=1e-10, max_outer=2000,
    )
    res = solver.solve()
    assert res.converged, "k-eigenvalue solve did not converge"
    return res.keff, res.flux


def transient(mats, initial_flux, delayed=None, theta=1.0):
    kwargs = {} if delayed is None else {"delayed": delayed}
    return nd.TimeDependentSolver(
        mats=mats, medium_map=MEDIUM_MAP, edges_x=EDGES,
        geom=nd.Geometry.Sphere, bc=BC, initial_flux=initial_flux,
        epsilon=1e-10, max_inner=2000, theta=theta, **kwargs,
    )


# ---------------------------------------------------------------------------
# 1-2.  Steady state
# ---------------------------------------------------------------------------

keff, flux0 = keff_of(core())
print(f"unperturbed keff        = {keff:.8f}")

critical = nd.scale_to_critical(core(), keff)
print("nusigf scaled by 1/keff -> the eigenvalue flux is now a true steady state")

# ---------------------------------------------------------------------------
# 3.  Delayed neutron data (Keepin 6-group U-235) and the perturbation
# ---------------------------------------------------------------------------

delayed = nd.make_delayed_data(
    nd.DELAYED_U235_6GROUP, G=1, n_mat=1, chi=critical.chi
)
beta = sum(nd.DELAYED_U235_6GROUP["Beta"])

# Removing absorber raises k.  Search for the removal cross section worth $0.50.
target = 0.50 * beta
lo, hi = 0.1532 * 0.99, 0.1532
for _ in range(40):
    mid = 0.5 * (lo + hi)
    k_mid, _ = keff_of(nd.scale_to_critical(core(mid), keff))
    if (k_mid - 1.0) / k_mid < target:
        hi = mid
    else:
        lo = mid
perturbed = nd.scale_to_critical(core(0.5 * (lo + hi)), keff)
k_pert, _ = keff_of(perturbed)
rho = (k_pert - 1.0) / k_pert
print(f"perturbation worth      = {rho:.6f} = ${rho / beta:.3f}")
print(f"point-kinetics prompt jump beta/(beta-rho) = {beta / (beta - rho):.4f}")

# ---------------------------------------------------------------------------
# 4-5.  Transients
# ---------------------------------------------------------------------------


def power(solver, reference):
    return float(np.sum(solver.result().flux)) / reference


steady = transient(critical, flux0, delayed)
p0 = float(np.sum(steady.result().flux))
steady.run(1.0, 10)
print(f"\nunperturbed, 10 s:      power = {power(steady, p0):.10f}  (holds flat)")

print("\n  t (s)    delayed    prompt-only")
runs = {
    "delayed": transient(critical, flux0, delayed),
    "prompt": transient(critical, flux0),
}
for solver in runs.values():
    solver.update_materials(perturbed)

for t_end in (0.005, 0.02, 0.05, 0.2, 0.5):
    row = []
    for solver in runs.values():
        n_steps = int(round((t_end - solver.time) / 5e-5))
        solver.run(5e-5, n_steps)
        row.append(power(solver, p0))
    print(f"  {t_end:5.3f}  {row[0]:9.4f}  {row[1]:13.4g}")

print(
    "\nThe delayed run tracks the prompt jump and then the slow precursor decay;\n"
    "the prompt-only run is a prompt excursion on the ~3e-5 s generation time."
)
print(
    f"\nprecursor concentrations: {len(runs['delayed'].precursors)} values "
    f"({CELLS} cells x {delayed.n_precursor} groups), per unit volume"
)

# ---------------------------------------------------------------------------
# 6.  Time differencing: backward Euler vs Crank-Nicolson at the same dt
# ---------------------------------------------------------------------------

T_END = 0.05


def power_at_t_end(theta, dt, damped_steps=0):
    """Run to T_END, optionally damping the insertion with backward Euler first.

    theta = 0.5 is A-stable but not L-stable, so it does not damp the stiff
    modes a step insertion excites.  Taking the first couple of steps at
    theta = 1 kills those before switching to the second-order weighting.
    """
    solver = transient(critical, flux0, delayed, theta=1.0)
    solver.update_materials(perturbed)
    solver.run(dt, damped_steps)
    solver.theta = theta
    solver.run(dt, int(round((T_END - solver.time) / dt)))
    return power(solver, p0)


# The reference has to out-resolve everything it is compared against, so take
# it with the second-order scheme: backward Euler at this dt would still carry
# an O(dt) error larger than the Crank-Nicolson errors below.
converged = power_at_t_end(0.5, 1e-5, damped_steps=2)
print(f"\ntime differencing, power at t = {T_END} s (converged: {converged:.8f})")
print("\n     dt     backward Euler        Crank-Nicolson")
for dt in (4e-4, 2e-4, 1e-4):
    be = power_at_t_end(1.0, dt)
    cn = power_at_t_end(0.5, dt, damped_steps=2)
    print(f"  {dt:.0e}   {be:.8f} ({abs(be - converged):.1e})   "
          f"{cn:.8f} ({abs(cn - converged):.1e})")

print(
    "\nHalving dt halves the backward-Euler error but quarters the\n"
    "Crank-Nicolson one - first order against second."
)
