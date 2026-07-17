"""
Transport -> diffusion cross-section example.

Multigroup transport libraries tabulate a total (or absorption) cross section, a
scatter matrix stored g_from -> g_to, and fission data.  The diffusion solvers
here instead want a diffusion coefficient, a removal cross section, and a scatter
matrix indexed [g_to][g_from] with the self-scatter diagonal removed.

    ndiffusion.transport_to_diffusion(data, G)          -> dict (one material)
    ndiffusion.make_materials_from_transport(data, G)   -> Materials

Run after installing the package:
    pip install .
    python examples/transport_cross_sections.py
"""

import numpy as np

import ndiffusion as nd


def linspace(start, stop, n):
    return list(np.linspace(start, stop, n))


# ---------------------------------------------------------------------------
# A 2-group transport material.
#
# Scat is stored the transport way, Scat[g_from][g_to]:
#   row 0 = scattering OUT of the fast group (self-scatter + fast->thermal)
#   row 1 = scattering OUT of the thermal group (self-scatter only here)
# SigTr is tabulated, so D = 1 / (3 * SigTr) needs no P1 correction.
# ---------------------------------------------------------------------------

fuel = {
    "SigTr":  np.array([2.3200e-01, 8.4000e-01]),
    "SigT":   np.array([2.5320e-01, 1.2100e00]),
    "Scat":   np.array([[2.2000e-01, 2.4100e-02],     # fast:  self, fast->thermal
                        [0.0000e00, 1.0800e00]]),      # thermal: self
    "nuSigf": np.array([8.5000e-03, 1.8500e-01]),
    "chi":    np.array([1.0, 0.0]),
}


# ---------------------------------------------------------------------------
# Example 1: inspect the conversion for one material
# ---------------------------------------------------------------------------

print("=== transport_to_diffusion (one material) ===")

diff = nd.transport_to_diffusion(fuel, G=2)

print(f"D        = {np.round(diff['D'], 6).tolist()}   (= 1 / (3*SigTr))")
print(f"Removal  = {np.round(diff['Removal'], 6).tolist()}   (= SigT - self-scatter)")
print("scatter[g_to][g_from] (diagonal zeroed, input transposed):")
print(np.round(diff["Scat"], 6))
print(
    "  the fast->thermal transfer 0.0241 now sits at scatter[1][0] = "
    f"{diff['Scat'][1][0]:g}"
)
print()


# ---------------------------------------------------------------------------
# Example 2: build a Materials and run a k-eigenvalue solve
#
# Reflective boundaries mean no leakage, so the solver's keff must equal the
# analytic infinite-medium k_inf implied by the transport data - a check that
# removal and scatter were converted self-consistently.
# ---------------------------------------------------------------------------

print("=== make_materials_from_transport -> KEigenSolver ===")

mats = nd.make_materials_from_transport([fuel], G=2)

cells = 40
edges = linspace(0.0, 20.0, cells + 1)
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
print()


# ---------------------------------------------------------------------------
# Note: a full C5G7 quarter-core run combines this with the unstructured mesh
# from tools/c5g7_fuel_mesh.py -
#
#     mesh = nd.load_gmsh("c5g7_fuel.msh")
#     mats = nd.make_materials_from_transport(seven_group_xs, G=7)
#     solver = nd.KEigenSolverUnstructured2D(mats, mesh, bc)
#
# It is left out of this runnable example because it needs the 7-group C5G7
# cross sections and a large (~270k-element) mesh file.
# ---------------------------------------------------------------------------
