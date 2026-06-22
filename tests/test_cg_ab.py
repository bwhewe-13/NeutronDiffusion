"""
A/B regression for the Option B prototype: matrix-free Jacobi-preconditioned CG
as the within-group inner solver, versus the default Gauss-Seidel.

The CG path is a drop-in replacement for the spatial inner solve inside the same
power iteration, so for a *converged* problem it must reproduce the GS keff and
flux shape to tight tolerance. These tests:

  1. Assert GS and CG agree (structured + unstructured, 1- and 2-group).
  2. Assert CG keff is stable under mesh refinement with default settings
     (the property the O(n^2) GS sweep struggles with - see solver_2d.hpp).

The inner solver is selected per-instance with `set_use_cg(True/False)`; the
default (and the existing suite) is unaffected. It can also be forced globally
with the NDIFFUSION_KEIG_CG environment variable.
"""

import numpy as np

import ndiffusion as nd


# ---------------------------------------------------------------------------
# Helpers (kept self-contained; mirror tests/test_2d_k_eigenvalue.py)
# ---------------------------------------------------------------------------


def linspace(a, b, n):
    return list(np.linspace(a, b, n))


def vacuum():
    return nd.BoundaryCondition(A=1.0, B=0.0)


def reflective():
    return nd.BoundaryCondition(A=0.0, B=1.0)


def one_group_mat():
    m = nd.Materials()
    m.n_mat, m.n_groups = 1, 1
    m.D = [3.850204978408833]
    m.removal = [0.1532]
    m.scatter = [0.0]
    m.chi = [1.0]
    m.nusigf = [0.1570]
    m.velocity = [2.2e5]
    return m


def two_group_mat():
    m = nd.Materials()
    m.n_mat, m.n_groups = 1, 2
    m.D = [1.255, 0.211]
    m.removal = [0.00836, 0.1003]
    m.scatter = [0.0, 0.02533,
                 0.0, 0.0]
    m.chi = [1.0, 0.0]
    m.nusigf = [0.004602, 0.1091]
    m.velocity = [2.2e7, 2.2e5]
    return m


def make_quad_mesh(nx, ny, Lx, Ly, bc_tag_top=0, bc_tag_right=0):
    dx, dy = Lx / nx, Ly / ny
    vx, vy = [], []
    for i in range(nx + 1):
        for j in range(ny + 1):
            vx.append(i * dx)
            vy.append(j * dy)

    def vid(i, j):
        return i * (ny + 1) + j

    cell_vertices, cell_offsets, mat_ids = [], [0], []
    for i in range(nx):
        for j in range(ny):
            cell_vertices += [vid(i, j), vid(i + 1, j),
                              vid(i + 1, j + 1), vid(i, j + 1)]
            cell_offsets.append(len(cell_vertices))
            mat_ids.append(0)

    bv0, bv1, btag = [], [], []
    for i in range(nx):
        bv0.append(vid(i, 0)); bv1.append(vid(i + 1, 0)); btag.append(0)
    for j in range(ny):
        bv0.append(vid(nx, j)); bv1.append(vid(nx, j + 1)); btag.append(bc_tag_right)
    for i in range(nx - 1, -1, -1):
        bv0.append(vid(i + 1, ny)); bv1.append(vid(i, ny)); btag.append(bc_tag_top)
    for j in range(ny - 1, -1, -1):
        bv0.append(vid(0, j + 1)); bv1.append(vid(0, j)); btag.append(0)

    mesh = nd.UnstructuredMesh2D()
    mesh.vx, mesh.vy = vx, vy
    mesh.cell_vertices, mesh.cell_offsets = cell_vertices, cell_offsets
    mesh.material_id = mat_ids
    mesh.bface_v0, mesh.bface_v1, mesh.bface_bc_tag = bv0, bv1, btag
    return mesh


def flux_cosine(a, b):
    """|cos| between two flux vectors - 1.0 when they match up to sign/scale."""
    a, b = np.asarray(a), np.asarray(b)
    return abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------------------------------------------------------------------------
# Structured A/B
# ---------------------------------------------------------------------------


def _structured(mats, nx, ny, R, bc_x, bc_y, use_cg):
    s = nd.KEigenSolver2D(
        mats=mats,
        medium_map=[0] * (nx * ny),
        edges_x=linspace(0.0, R, nx + 1),
        edges_y=linspace(0.0, R, ny + 1),
        geom=nd.Geometry2D.XY,
        bc_x=bc_x, bc_y=bc_y,
        epsilon=1e-10, verbose=False,
    )
    s.set_use_cg(use_cg)
    return s.solve()


class TestStructuredAB:
    def test_one_group_agrees(self):
        m = one_group_mat()
        gs = _structured(m, 24, 24, 60.0, [vacuum()], [vacuum()], False)
        cg = _structured(m, 24, 24, 60.0, [vacuum()], [vacuum()], True)
        assert abs(gs.keff - cg.keff) < 1e-7, (gs.keff, cg.keff)
        assert flux_cosine(gs.flux, cg.flux) > 1 - 1e-8

    def test_two_group_agrees(self):
        m = two_group_mat()
        gs = _structured(m, 20, 20, 80.0, [vacuum()] * 2, [vacuum()] * 2, False)
        cg = _structured(m, 20, 20, 80.0, [vacuum()] * 2, [vacuum()] * 2, True)
        assert abs(gs.keff - cg.keff) < 1e-7, (gs.keff, cg.keff)
        assert flux_cosine(gs.flux, cg.flux) > 1 - 1e-8

    def test_refinement_stable(self):
        """CG keff stays put as the mesh is refined (default settings)."""
        m = two_group_mat()
        keffs = [
            _structured(m, n, n, 80.0, [vacuum()] * 2, [vacuum()] * 2, True).keff
            for n in (20, 40, 80)
        ]
        assert max(keffs) - min(keffs) < 5e-4, keffs


# ---------------------------------------------------------------------------
# Unstructured A/B
# ---------------------------------------------------------------------------


def _unstructured(mats, nx, ny, L, use_cg):
    mesh = make_quad_mesh(nx, ny, L, L, bc_tag_top=0, bc_tag_right=0)
    bc = [vacuum() for _ in range(mats.n_groups)]
    s = nd.KEigenSolverUnstructured2D(
        mats=mats, mesh=mesh, bc=bc, epsilon=1e-10, verbose=False,
    )
    s.set_use_cg(use_cg)
    return s.solve()


class TestUnstructuredAB:
    def test_one_group_agrees(self):
        m = one_group_mat()
        gs = _unstructured(m, 16, 16, 60.0, False)
        cg = _unstructured(m, 16, 16, 60.0, True)
        assert abs(gs.keff - cg.keff) < 1e-7, (gs.keff, cg.keff)
        assert flux_cosine(gs.flux, cg.flux) > 1 - 1e-8

    def test_two_group_agrees(self):
        m = two_group_mat()
        gs = _unstructured(m, 14, 14, 80.0, False)
        cg = _unstructured(m, 14, 14, 80.0, True)
        assert abs(gs.keff - cg.keff) < 1e-7, (gs.keff, cg.keff)
        assert flux_cosine(gs.flux, cg.flux) > 1 - 1e-8
