"""
k-eigenvalue example: 1-group and 2-group spherical problems.

Run after installing the package:
    pip install .
    python examples/k_eigenvalue.py
"""

import numpy as np
import ndiffusion as nd


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def linspace(start, stop, n):
    return list(np.linspace(start, stop, n))


# ---------------------------------------------------------------------------
# Example 1: 1-group sphere, single material
# ---------------------------------------------------------------------------

print("=== 1-group sphere (1 material) ===")

m = nd.Materials()
m.n_mat    = 1
m.n_groups = 1
m.D        = [3.850204978408833]
m.removal  = [0.1532]
m.scatter  = [0.0]
m.chi      = [1.0]
m.nusigf   = [0.1570]

cells = 50
edges = linspace(0.0, 100.0, cells + 1)

solver = nd.DiffusionSolver(
    mats       = m,
    medium_map = [0] * cells,
    edges_x    = edges,
    geom       = nd.Geometry.Sphere,
    bc         = [nd.BoundaryCondition(A=1.0, B=0.0)],
    epsilon    = 1e-8,
    verbose    = False,
)
result = solver.solve()

flux = np.array(result.flux).reshape(cells, m.n_groups)
print(f"keff       = {result.keff:.8f}  (reference: 1.00001244)")
print(f"iterations = {result.iterations}")
print(f"residual   = {result.residual:.2e}")
print(f"flux peak  = {flux[:, 0].max():.4f}  (cell 0, symmetric centre)")
print()


# ---------------------------------------------------------------------------
# Example 2: 2-group sphere with fast-to-thermal downscatter
# ---------------------------------------------------------------------------

print("=== 2-group sphere (downscatter, reflective BC) ===")

m2 = nd.Materials()
m2.n_mat    = 1
m2.n_groups = 2
m2.D        = [0.1, 0.1]
m2.removal  = [0.0362, 0.121]
# scatter[mat][g_to][g_from]: fast (g=0) -> thermal (g=1)
m2.scatter  = [0.0,    0.0,
               0.0241, 0.0]
m2.chi      = [1.0, 0.0]
m2.nusigf   = [0.0085, 0.185]

cells2 = 50
edges2 = linspace(0.0, 5.0, cells2 + 1)
bc_ref = nd.BoundaryCondition(A=0.0, B=1.0)   # reflective

solver2 = nd.DiffusionSolver(
    mats       = m2,
    medium_map = [0] * cells2,
    edges_x    = edges2,
    geom       = nd.Geometry.Sphere,
    bc         = [bc_ref, bc_ref],
    epsilon    = 1e-8,
    verbose    = False,
)
result2 = solver2.solve()

flux2 = np.array(result2.flux).reshape(cells2, m2.n_groups)
print(f"keff       = {result2.keff:.8f}  (reference: 1.25268252)")
print(f"iterations = {result2.iterations}")
print(f"fast flux peak    = {flux2[:, 0].max():.4f}")
print(f"thermal flux peak = {flux2[:, 1].max():.4f}")
