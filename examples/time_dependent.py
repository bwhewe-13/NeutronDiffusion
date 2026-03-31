"""
Time-dependent example: critical, supercritical, and subcritical 1-group spheres.

Run after installing the package:
    pip install .
    python examples/time_dependent.py
"""

import numpy as np
import ndiffusion as nd


def linspace(start, stop, n):
    return list(np.linspace(start, stop, n))


cells = 30
edges = linspace(0.0, 100.0, cells + 1)
bc    = [nd.BoundaryCondition(A=1.0, B=0.0)]


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


def keigenvalue_flux(mats):
    solver = nd.KEigenSolver(mats, [0] * cells, edges,
                                nd.Geometry.Sphere, bc, epsilon=1e-10)
    res = solver.solve()
    return res.keff, res.flux


# ---------------------------------------------------------------------------
# Start from the k-eigenvalue flux of the critical system
# ---------------------------------------------------------------------------
keff, init_flux = keigenvalue_flux(make_mat(nusigf_scale=1.0))
print(f"k-eigenvalue solution: keff = {keff:.8f}")
print()


# ---------------------------------------------------------------------------
# Critical system: flux shape should be preserved
# ---------------------------------------------------------------------------
print("=== Critical system (nusigf_scale = 1.0) ===")
tds = nd.TimeDependentSolver(
    make_mat(1.0), [0] * cells, edges, nd.Geometry.Sphere, bc,
    initial_flux=init_flux, epsilon=1e-10
)
res0 = tds.result()
tds.run(dt=1e-5, n_steps=200)
res_final = tds.result()

init_arr  = np.array(res0.flux)
final_arr = np.array(res_final.flux)
shape_err = np.max(np.abs(final_arr / final_arr.max() - init_arr / init_arr.max()))
print(f"Elapsed time : {res_final.time:.4e} s  ({res_final.steps} steps)")
print(f"Shape error  : {shape_err:.2e}  (expected < 1e-4 for near-critical)")
print()


# ---------------------------------------------------------------------------
# Supercritical system: flux should grow
# ---------------------------------------------------------------------------
print("=== Supercritical system (nusigf_scale = 1.5) ===")
tds_super = nd.TimeDependentSolver(
    make_mat(1.5), [0] * cells, edges, nd.Geometry.Sphere, bc,
    initial_flux=init_flux, epsilon=1e-10
)
sum_before = np.array(tds_super.result().flux).sum()
tds_super.run(dt=1e-6, n_steps=50)
sum_after = np.array(tds_super.result().flux).sum()
print(f"Sum flux before: {sum_before:.6f}")
print(f"Sum flux after : {sum_after:.6f}  (should be larger)")
print()


# ---------------------------------------------------------------------------
# Subcritical system: flux should decay
# ---------------------------------------------------------------------------
print("=== Subcritical system (nusigf_scale = 0.5) ===")
tds_sub = nd.TimeDependentSolver(
    make_mat(0.5), [0] * cells, edges, nd.Geometry.Sphere, bc,
    initial_flux=init_flux, epsilon=1e-10
)
sum_before = np.array(tds_sub.result().flux).sum()
tds_sub.run(dt=1e-6, n_steps=50)
sum_after = np.array(tds_sub.result().flux).sum()
print(f"Sum flux before: {sum_before:.6f}")
print(f"Sum flux after : {sum_after:.6f}  (should be smaller)")
