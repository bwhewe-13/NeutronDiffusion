"""
Benchmark convergence
=====================

The TWIGL and IAEA two-group benchmarks, each solved on a sequence of meshes.
TWIGL runs on the structured 2-D solver; the IAEA core has a stepped outline, so
it runs on the unstructured one.  Both approach second order once the mesh
resolves the thermal diffusion length, about 2 cm in these fuels.

The cross sections come from :mod:`ndiffusion.materials`, which records each
benchmark's published eigenvalue.  The geometries are built in
``examples/_benchmarks.py``.
"""

import _benchmarks as B
import matplotlib.pyplot as plt
import numpy as np
from _plotting import plot_cells, plot_grid

# %%
# The two cores
# -------------
# TWIGL is a seed-and-blanket quarter core; IAEA a quarter PWR core with rodded
# assemblies and a reflector.  Both have reflective symmetry edges at x = 0 and
# y = 0.

mmap, edges = B.twigl_mesh(1.0)
iaea = B.iaea_mesh(5.0)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
plot_grid(axes[0], edges, edges, mmap, cmap="Pastel1")
axes[0].set_title("TWIGL: seed and blanket")
plot_cells(axes[1], iaea, np.asarray(iaea.material_id), cmap="Pastel1")
axes[1].set_title("IAEA: fuel 1, fuel 2, rodded, reflector")
fig.tight_layout()

# %%
# Mesh refinement
# ---------------
# Each benchmark is solved on a sequence of meshes, halving the cell size each
# time.  The reference is a Richardson extrapolation from two finer meshes than
# shown here - h = 0.5 and 0.25 cm for TWIGL, 1.25 and 0.625 cm for IAEA - which
# take too long to run every time the documentation is built.

studies = {
    "TWIGL (structured)": (B.solve_twigl, np.array([4.0, 2.0, 1.0, 0.5]), 0.91320965),
    "IAEA (unstructured)": (B.solve_iaea, np.array([5.0, 2.5, 1.25]), 1.02958822),
}
errors = {}
for name, (solve, hs, k_ref) in studies.items():
    k = np.array([solve(h).keff for h in hs])
    errors[name] = (hs, np.abs(k - k_ref))
    d = np.diff(k)
    print(f"{name}:")
    for h, kk in zip(hs, k):
        print(f"    h = {h:5.2f} cm   keff = {kk:.6f}   error {abs(kk - k_ref) * 1e5:6.2f} pcm")
    print("    observed order between successive meshes:",
          np.round(np.log2(d[:-1] / d[1:]), 2))

# %%
# On the coarsest meshes, with cells wider than the thermal diffusion length,
# the error does not yet follow a power law; it settles toward slope 2 as the
# mesh is refined.

fig, ax = plt.subplots(figsize=(5.5, 4))
for name, (hs, err) in errors.items():
    ax.loglog(hs, err * 1e5, "o-", label=name)
h_ref = np.array([0.5, 5.0])
ax.loglog(h_ref, 3.0 * h_ref ** 2, "k--", lw=1, label="slope 2")
ax.set_xlabel("cell size h (cm)")
ax.set_ylabel(r"$|k_h - k_\mathrm{ref}|$ (pcm)")
ax.legend(frameon=False)
fig.tight_layout()

plt.show()
