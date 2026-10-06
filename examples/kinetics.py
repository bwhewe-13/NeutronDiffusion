"""
A 50 cent step insertion
========================

The whole kinetics workflow on a 1-D sphere: solve the k-eigenvalue problem,
scale it to an exact steady state, add six-group delayed neutron data, insert
50 cents of reactivity, and compare against the same insertion with prompt
neutrons only.

Delayed neutrons are not a refinement - they set the timescale.  The delayed run
settles near a prompt jump of about 2 and then creeps up over seconds; the
prompt-only run passes that within milliseconds and keeps going.

Run after installing the package::

    pip install .
    python examples/kinetics.py
"""

import matplotlib.pyplot as plt
import numpy as np

import ndiffusion as nd

CELLS = 60
EDGES = np.linspace(0.0, 100.0, CELLS + 1)
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
    res = nd.KEigenSolver(mats, MEDIUM_MAP, EDGES, nd.Geometry.Sphere, BC,
                          epsilon=1e-10, max_outer=2000).solve()
    assert res.converged, "k-eigenvalue solve did not converge"
    return res.keff, res.flux


def transient(mats, initial_flux, delayed=None, theta=1.0):
    kwargs = {} if delayed is None else {"delayed": delayed}
    return nd.TimeDependentSolver(
        mats=mats, medium_map=MEDIUM_MAP, edges_x=EDGES,
        geom=nd.Geometry.Sphere, bc=BC, initial_flux=initial_flux,
        epsilon=1e-10, max_inner=2000, theta=theta, **kwargs,
    )


def power(solver, reference):
    return float(np.sum(solver.result().flux)) / reference


# %%
# Steady state
# ------------
# A k-eigenvalue flux is only stationary once ``nusigf`` is divided by keff;
# without that a transient drifts from the first step.

keff, flux0 = keff_of(core())
critical = nd.scale_to_critical(core(), keff)
print(f"unperturbed keff = {keff:.8f}")

# %%
# Delayed neutron data and the perturbation
# -----------------------------------------
# The Keepin six-group U-235 set, and a search for the removal cross section
# worth 50 cents.  The perturbed cross sections are scaled by the *unperturbed*
# keff, so the perturbed system is supercritical by exactly the inserted worth.

delayed = nd.make_delayed_data(nd.DELAYED_U235_6GROUP, G=1, n_mat=1, chi=critical.chi)
beta = sum(nd.DELAYED_U235_6GROUP["Beta"])

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
print(f"perturbation worth = {rho:.6f} = ${rho / beta:.3f}")
print(f"point-kinetics prompt jump beta/(beta - rho) = {beta / (beta - rho):.4f}")

# %%
# The unperturbed transient holds flat for 10 s - the check that the start is a
# genuine steady state with equilibrium precursors.

steady = transient(critical, flux0, delayed)
p0 = float(np.sum(steady.result().flux))
steady.run(1.0, 10)
print(f"unperturbed, 10 s: power = {power(steady, p0):.10f}")

# %%
# The insertion, with and without delayed neutrons
# ------------------------------------------------
# ``update_materials`` swaps the cross sections and keeps the flux and the
# precursors, which is a step insertion at the current time.

runs = {"delayed": transient(critical, flux0, delayed), "prompt only": transient(critical, flux0)}
history = {name: ([0.0], [1.0]) for name in runs}
for solver in runs.values():
    solver.update_materials(perturbed)

dt = 5e-5
for t_end in np.geomspace(1e-4, 0.5, 40):
    for name, solver in runs.items():
        n_steps = int(round((t_end - solver.time) / dt))
        if n_steps > 0:
            solver.run(dt, n_steps)
            history[name][0].append(solver.time)
            history[name][1].append(power(solver, p0))

for name, (t, p) in history.items():
    print(f"{name:<12} power at {t[-1]:.2f} s: {p[-1]:.4g}")

fig, ax = plt.subplots(figsize=(6, 3.5))
for name, (t, p) in history.items():
    ax.plot(t[1:], p[1:], label=name)
ax.axhline(beta / (beta - rho), color="0.5", ls=":", lw=1, label="prompt jump")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(0.8, 30)   # the prompt-only run leaves the top within milliseconds
ax.set_xlabel("time after insertion (s)")
ax.set_ylabel("power / initial")
ax.legend(frameon=False)
fig.tight_layout()

# %%
# Precursors are stored per unit volume, one column per precursor group.

print(f"precursors: shape {runs['delayed'].precursors.shape}")

# %%
# Time differencing
# -----------------
# Backward Euler (``theta = 1``) against Crank-Nicolson (``theta = 0.5``) at the
# same step sizes.  Crank-Nicolson is A-stable but not L-stable, so it does not
# damp the stiff modes a step insertion excites; two backward-Euler steps first
# remove them.  The reference is Crank-Nicolson at a much smaller step, since
# backward Euler there would still carry a larger error than the
# Crank-Nicolson runs.

T_END = 0.05


def power_at_t_end(theta, dt, damped_steps=0):
    solver = transient(critical, flux0, delayed, theta=1.0)
    solver.update_materials(perturbed)
    solver.run(dt, damped_steps)
    solver.theta = theta
    solver.run(dt, int(round((T_END - solver.time) / dt)))
    return power(solver, p0)


converged = power_at_t_end(0.5, 1e-5, damped_steps=2)
steps = np.array([4e-4, 2e-4, 1e-4])
be = np.array([abs(power_at_t_end(1.0, s) - converged) for s in steps])
cn = np.array([abs(power_at_t_end(0.5, s, damped_steps=2) - converged) for s in steps])
for s, e1, e2 in zip(steps, be, cn):
    print(f"dt = {s:.0e}: backward Euler error {e1:.1e}, Crank-Nicolson error {e2:.1e}")

# %%
# Halving the step halves the backward-Euler error and quarters the
# Crank-Nicolson one - first order against second.

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.loglog(steps, be, "o-", label=r"backward Euler, $\theta = 1$")
ax.loglog(steps, cn, "s-", label=r"Crank-Nicolson, $\theta = 0.5$")
ax.loglog(steps, be[-1] * steps / steps[-1], "k:", lw=1, label="slope 1")
ax.loglog(steps, cn[-1] * (steps / steps[-1]) ** 2, "k--", lw=1, label="slope 2")
ax.set_xlabel("time step (s)")
ax.set_ylabel(f"power error at t = {T_END} s")
ax.legend(frameon=False)
fig.tight_layout()

plt.show()
