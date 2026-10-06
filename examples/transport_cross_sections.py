"""
Transport cross sections to diffusion
=====================================

Multigroup transport libraries tabulate a total (or absorption) cross section, a
scatter matrix stored g_from -> g_to, and fission data.  The diffusion solvers
here instead want a diffusion coefficient, a removal cross section, and a scatter
matrix indexed [g_to][g_from] with the self-scatter diagonal removed.

``transport_to_diffusion(data, G)`` converts one material to a dict, and
``make_materials_from_transport(data, G)`` builds a ``Materials``.

Run after installing the package::

    pip install .
    python examples/transport_cross_sections.py
"""

import numpy as np

import ndiffusion as nd

# %%
# A two-group transport material
# ------------------------------
# ``Scat`` is stored the transport way, ``Scat[g_from][g_to]``: row 0 is
# scattering out of the fast group (self-scatter and fast to thermal), row 1
# out of the thermal group (self-scatter only here).  ``SigTr`` is tabulated, so
# ``D = 1 / (3 SigTr)`` needs no P1 correction.

fuel = {
    "SigTr":  np.array([2.3200e-01, 8.4000e-01]),
    "SigT":   np.array([2.5320e-01, 1.2100e00]),
    "Scat":   np.array([[2.2000e-01, 2.4100e-02],     # fast:  self, fast->thermal
                        [0.0000e00, 1.0800e00]]),      # thermal: self
    "nuSigf": np.array([8.5000e-03, 1.8500e-01]),
    "chi":    np.array([1.0, 0.0]),
}


# %%
# Inspect the conversion for one material
# ---------------------------------------

diff = nd.transport_to_diffusion(fuel, G=2)

print(f"D        = {np.round(diff['D'], 6).tolist()}   (= 1 / (3*SigTr))")
print(f"Removal  = {np.round(diff['Removal'], 6).tolist()}   (= SigT - self-scatter)")
print("scatter[g_to][g_from] (diagonal zeroed, input transposed):")
print(np.round(diff["Scat"], 6))
print(
    "  the fast->thermal transfer 0.0241 now sits at scatter[1][0] = "
    f"{diff['Scat'][1][0]:g}"
)


# %%
# Build a Materials and solve
# ---------------------------
# Reflective boundaries mean no leakage, so the solver's keff must equal the
# infinite-medium k_inf implied by the transport data - a check that removal and
# scatter were converted consistently.

mats = nd.make_materials_from_transport([fuel], G=2)

cells = 40
edges = np.linspace(0.0, 20.0, cells + 1)
reflective = [nd.BoundaryCondition(A=0.0, B=1.0) for _ in range(2)]

solver = nd.KEigenSolver(
    mats       = mats,
    medium_map = [0] * cells,
    edges_x    = edges,
    geom       = nd.Geometry.Slab,
    bc         = reflective,
    epsilon    = 1e-10,
    verbose    = False,
)
result = solver.solve()

# Analytic k_inf from the 0-D two-group balance: M phi = (1/k) F phi.
scat = fuel["Scat"]                       # [g_from][g_to]
s = scat.T.copy()                         # [g_to][g_from]
np.fill_diagonal(s, 0.0)
removal = fuel["SigT"] - np.diagonal(scat)
M = np.diag(removal) - s
F = np.outer(fuel["chi"], fuel["nuSigf"])
k_inf = float(np.max(np.linalg.eigvals(np.linalg.solve(M, F)).real))

print(f"keff (solver)   = {result.keff:.8f}")
print(f"k_inf (analytic)= {k_inf:.8f}")
print(f"difference      = {abs(result.keff - k_inf):.2e}")


# %%
# The C5G7 quarter-core example combines this conversion with an unstructured
# pin-cell mesh and the seven-group C5G7 cross sections.
