"""
Prompt transients in a sphere
=============================

The bare sphere from the k-eigenvalue example, started from its fundamental
mode and stepped forward with the fission cross section scaled up, left alone,
or scaled down.  There are no delayed neutrons here, so the timescale is the
prompt generation time; see the kinetics example for delayed neutrons.

Run after installing the package::

    pip install .
    python examples/time_dependent.py
"""

import matplotlib.pyplot as plt
import numpy as np

import ndiffusion as nd

cells = 30
edges = np.linspace(0.0, 100.0, cells + 1)
bc = [nd.BoundaryCondition(A=1.0, B=0.0)]


def make_mat(nusigf_scale=1.0):
    m = nd.Materials()
    m.n_mat    = 1
    m.n_groups = 1
    m.D        = [3.850204978408833]
    m.removal  = [0.1532]
    m.scatter  = [0.0]
    m.chi      = [1.0]
    m.nusigf   = [0.1570 * nusigf_scale]
    m.velocity = [2.2e5]   # thermal neutron speed, cm/s
    return m


# %%
# Start from the fundamental mode
# -------------------------------
# A k-eigenvalue flux is a steady state only when :math:`k = 1`.  This sphere
# is critical to about 1 pcm on this mesh, so the unscaled case should hold
# its shape and level.

res = nd.KEigenSolver(make_mat(), [0] * cells, edges, nd.Geometry.Sphere, bc,
                      epsilon=1e-10, max_outer=2000).solve()
print(f"keff = {res.keff:.8f}")

# %%
# Step each case
# --------------
# ``step`` advances one time step and ``result()`` reports the current state,
# so the total flux can be recorded as the transient runs.

dt, n_steps = 2e-6, 100
history = {}
for label, scale in (("nusigf x 1.02", 1.02), ("critical", 1.0), ("nusigf x 0.98", 0.98)):
    solver = nd.TimeDependentSolver(make_mat(scale), [0] * cells, edges,
                                    nd.Geometry.Sphere, bc,
                                    initial_flux=res.flux, epsilon=1e-10)
    total0 = solver.result().flux.sum()
    times, totals = [0.0], [1.0]
    for _ in range(n_steps):
        solver.step(dt)
        times.append(solver.time)
        totals.append(solver.result().flux.sum() / total0)
    history[label] = (np.array(times), np.array(totals))
    print(f"{label:<14} total flux after {solver.time:.1e} s: {totals[-1]:.6f}")

# %%
# The critical case stays flat; the other two grow or decay exponentially on
# the prompt timescale.

fig, ax = plt.subplots(figsize=(6, 3.5))
for label, (t, total) in history.items():
    ax.plot(t * 1e6, total, label=label)
ax.set_xlabel(r"time ($\mu$s)")
ax.set_ylabel("total flux / initial")
ax.legend(frameon=False)
fig.tight_layout()

# %%
# The critical case also keeps its shape.

solver = nd.TimeDependentSolver(make_mat(), [0] * cells, edges, nd.Geometry.Sphere, bc,
                                initial_flux=res.flux, epsilon=1e-10)
solver.run(dt=1e-5, n_steps=200)
start, end = res.flux[:, 0], solver.result().flux[:, 0]
print(f"shape change after {solver.time:.1e} s: "
      f"{np.max(np.abs(end / end.max() - start / start.max())):.1e}")

plt.show()
