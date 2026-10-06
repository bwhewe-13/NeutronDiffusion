"""
Unstructured meshes and symmetry sectors
========================================

The unstructured solver takes any simple polygon, so a hexagonal lattice needs
no special treatment, and neither do the smaller polygons a symmetry cut leaves
behind.  ``ndiffusion.layouts`` builds both, and tags every boundary face as a
symmetry cut or the outer surface so the boundary conditions follow.

This example solves one hexagonal core three ways - the full core, a 60 degree
sector and a 30 degree sector.  The cuts slice cells differently in each, so the
three agree to within the discretization error, and the gap closes as the
lattice is refined.
"""

import matplotlib.pyplot as plt
import numpy as np
from _plotting import plot_cells

import ndiffusion as nd
from ndiffusion import layouts

# %%
# A two-group core and reflector
# ------------------------------
# Rows are ``(D1, D2, Sa1, Sa2, S12, nuSf1, nuSf2)``, the form two-group
# benchmarks are published in.

mats = nd.materials.two_group([
    (1.4, 0.40, 0.010, 0.080, 0.020, 0.006, 0.135),   # fuel
    (1.3, 0.30, 0.001, 0.020, 0.030, 0.000, 0.000),   # reflector
])
pitch = 10.0
painter = layouts.hex_rings(pitch, [0, 0, 0, 0, 1])   # four fuel rings, then reflector

# %%
# Three orientations of one geometry
# ----------------------------------
# A sector is cut out of the full hexagonal mesh with straight cuts through the
# origin.  Cells the cuts cross become smaller polygons.

orientations = ["full", "sector60", "sector30"]
meshes = {}
for orientation in orientations:
    mesh = layouts.hex_mesh(pitch, n_rings=5, orientation=orientation)
    nd.assign_materials(mesh, painter)
    meshes[orientation] = mesh

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
for ax, (orientation, mesh) in zip(axes, meshes.items()):
    plot_cells(ax, mesh, np.asarray(mesh.material_id), cmap="Pastel1",
               edgecolor="0.35", linewidth=0.5, clim=(0, 8))
    sides = np.diff(np.asarray(mesh.cell_offsets))
    ax.set_title(f"{orientation}: {len(sides)} cells, {sides.min()}-{sides.max()} sides")
fig.tight_layout()

# %%
# Solving each sector
# -------------------
# ``layouts.boundary_conditions`` puts reflective conditions on the cuts and a
# Marshak vacuum condition, using the reflector's diffusion coefficients, on
# the outer surface.  At a 10 cm pitch, against a 2 cm thermal diffusion length
# in the fuel, the mesh is coarse and the sectors differ by a few hundred pcm.

results = {}
for orientation, mesh in meshes.items():
    bc = layouts.boundary_conditions(mesh, D=[1.3, 0.3], albedo=0.0)
    results[orientation] = nd.KEigenSolverUnstructured2D(
        mats, mesh, bc, epsilon=1e-10, max_outer=2000).solve()
    print(f"{orientation:<9} keff = {results[orientation].keff:.8f}")

spread = max(r.keff for r in results.values()) - min(r.keff for r in results.values())
print(f"spread = {spread * 1e5:.0f} pcm")

# %%
# The thermal flux on the full core and the 30 degree sector.

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, orientation in zip(axes, ["full", "sector30"]):
    thermal = results[orientation].flux[:, 1]
    plot_cells(ax, meshes[orientation], thermal / thermal.max(), colorbar="thermal flux")
    ax.set_title(orientation)
fig.tight_layout()

# %%
# The gap closes under refinement
# -------------------------------
# A homogeneous core of fixed size built from ever smaller hexagons: pitch 30/n
# with n rings.  The spread between the full core and the two sectors falls by
# about 4 each time the pitch halves - second order.

spreads = []
ns = (3, 6, 12, 24)
for n in ns:
    p = 30.0 / n
    ks = []
    for orientation in orientations:
        mesh = layouts.hex_mesh(p, n_rings=n, orientation=orientation)
        nd.assign_materials(mesh, layouts.homogeneous())
        bc = layouts.boundary_conditions(mesh, D=[1.4, 0.4], albedo=0.0)
        ks.append(nd.KEigenSolverUnstructured2D(mats, mesh, bc, epsilon=1e-10,
                                                max_outer=5000, use_cg=True).solve().keff)
    spreads.append(max(ks) - min(ks))
    print(f"pitch {p:5.2f} cm: keff {ks[0]:.6f}, spread {spreads[-1] * 1e5:7.2f} pcm")

fig, ax = plt.subplots(figsize=(5, 3.5))
pitches = 30.0 / np.array(ns)
ax.loglog(pitches, np.array(spreads) * 1e5, "o-")
ax.set_xlabel("pitch (cm)")
ax.set_ylabel("spread between sectors (pcm)")
fig.tight_layout()

plt.show()
