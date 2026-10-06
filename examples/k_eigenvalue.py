"""
k-eigenvalue problems in 1-D
============================

Two spheres: a bare one-group sphere at its critical radius, and a two-group
sphere with a reflective surface, which is the infinite medium.  Both have an
analytic eigenvalue to check against.

Run after installing the package::

    pip install .
    python examples/k_eigenvalue.py
"""

import matplotlib.pyplot as plt
import numpy as np
from _plotting import plot_groups

import ndiffusion as nd

# %%
# One group, bare sphere
# ----------------------
# With zero flux at the surface the critical radius is :math:`R = \pi / B`,
# where :math:`B^2 = (\nu\Sigma_f - \Sigma_a) / D`.  These cross sections put
# it at 100 cm, so the exact eigenvalue is 1.

m = nd.Materials()
m.n_mat    = 1
m.n_groups = 1
m.D        = [3.850204978408833]
m.removal  = [0.1532]
m.scatter  = [0.0]
m.chi      = [1.0]
m.nusigf   = [0.1570]

cells = 50
edges = np.linspace(0.0, 100.0, cells + 1)

solver = nd.KEigenSolver(
    mats       = m,
    medium_map = [0] * cells,
    edges_x    = edges,
    geom       = nd.Geometry.Sphere,
    bc         = [nd.BoundaryCondition(A=1.0, B=0.0)],
    epsilon    = 1e-8,
    max_outer  = 1000,
)
result = solver.solve()

print(f"keff       = {result.keff:.8f}  (exact: 1)")
print(f"iterations = {result.iterations}")
print(f"residual   = {result.residual:.2e}")

# %%
# The fundamental mode of a bare sphere is :math:`\sin(Br) / r`.

r = 0.5 * (edges[:-1] + edges[1:])
B = np.pi / 100.0
exact = np.sin(B * r) / (B * r)

fig, ax = plt.subplots(figsize=(6, 3.5))
plot_groups(ax, edges, result.flux, labels=["ndiffusion"])
ax.plot(r, exact, "k--", lw=1, label=r"$\sin(Br)/Br$")
ax.set_xlabel("radius (cm)")
ax.legend(frameon=False)
fig.tight_layout()

# %%
# Two groups, infinite medium
# ---------------------------
# A reflective surface removes all leakage, so the eigenvalue is
# :math:`k_\infty = (\nu\Sigma_{f,1} + \nu\Sigma_{f,2}\,\Sigma_{1\to2} /
# \Sigma_{r,2}) / \Sigma_{r,1}`.  Scatter is indexed ``[g_to][g_from]``, so the
# fast-to-thermal transfer is the second row, first column.

m2 = nd.Materials()
m2.n_mat    = 1
m2.n_groups = 2
m2.D        = [0.1, 0.1]
m2.removal  = [0.0362, 0.121]
m2.scatter  = [0.0,    0.0,
               0.0241, 0.0]
m2.chi      = [1.0, 0.0]
m2.nusigf   = [0.0085, 0.185]

reflective = nd.BoundaryCondition(A=0.0, B=1.0)
solver2 = nd.KEigenSolver(
    mats       = m2,
    medium_map = [0] * 50,
    edges_x    = np.linspace(0.0, 5.0, 51),
    geom       = nd.Geometry.Sphere,
    bc         = [reflective, reflective],
    epsilon    = 1e-8,
)
result2 = solver2.solve()

k_inf = (0.0085 + 0.185 * 0.0241 / 0.121) / 0.0362
print(f"keff  = {result2.keff:.8f}")
print(f"k_inf = {k_inf:.8f}")

# %%
# Without leakage the flux is flat, and its group ratio is the infinite-medium
# spectrum :math:`\phi_2 / \phi_1 = \Sigma_{1\to2} / \Sigma_{r,2}`.

flux2 = result2.flux
print(f"thermal/fast = {flux2[:, 1].mean() / flux2[:, 0].mean():.6f}"
      f"  (expected {0.0241 / 0.121:.6f})")

plt.show()
