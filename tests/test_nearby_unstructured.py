"""Tests for the method of nearby problems on the unstructured 2-D FVM mesh.

Manufactured solution with zero flux on every edge (all-vacuum BCs):
    phi_exact(x, y) = sin(pi x / Lx) sin(pi y / Ly)

On a regular quad mesh the FVM scheme is second-order and MNP recovers the
(small) O(h^2) error quantitatively.  The right-triangle mesh is non-orthogonal,
so its accuracy rests on the deferred non-orthogonal correction; MNP tracks the
error there too, and it converges under refinement.
"""

import numpy as np
import pytest

import ndiffusion as nd
from ndiffusion import nearby

pytest.importorskip("scipy")

D = 1.0
SIG_A = 0.2
LX, LY = 10.0, 8.0
AX, AY = np.pi / LX, np.pi / LY


def one_group_absorber():
    m = nd.Materials()
    m.n_mat = 1
    m.n_groups = 1
    m.D = [D]
    m.removal = [SIG_A]
    m.scatter = [0.0]
    m.chi = [0.0]
    m.nusigf = [0.0]
    return m


def quad_mesh(nx, ny, Lx, Ly, bc_tag=0):
    """Regular nx x ny quad mesh on [0,Lx]x[0,Ly]; every boundary face -> bc_tag."""
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
            cell_vertices += [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
            cell_offsets.append(len(cell_vertices))
            mat_ids.append(0)

    bv0, bv1, btag = [], [], []

    def add(a, b):
        bv0.append(a)
        bv1.append(b)
        btag.append(bc_tag)

    for i in range(nx):
        add(vid(i, 0), vid(i + 1, 0))
    for j in range(ny):
        add(vid(nx, j), vid(nx, j + 1))
    for i in range(nx):
        add(vid(i, ny), vid(i + 1, ny))
    for j in range(ny):
        add(vid(0, j), vid(0, j + 1))

    mesh = nd.UnstructuredMesh2D()
    mesh.vx, mesh.vy = vx, vy
    mesh.cell_vertices, mesh.cell_offsets = cell_vertices, cell_offsets
    mesh.material_id = mat_ids
    mesh.bface_v0, mesh.bface_v1, mesh.bface_bc_tag = bv0, bv1, btag
    return mesh


def triangle_mesh(nx, ny, Lx, Ly, bc_tag=0):
    """Right-triangle mesh (each quad split in two); every boundary -> bc_tag."""
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
            cell_vertices += [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)]
            cell_offsets.append(len(cell_vertices))
            cell_vertices += [vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)]
            cell_offsets.append(len(cell_vertices))
            mat_ids += [0, 0]

    bv0, bv1, btag = [], [], []

    def add(a, b):
        bv0.append(a)
        bv1.append(b)
        btag.append(bc_tag)

    for i in range(nx):
        add(vid(i, 0), vid(i + 1, 0))
    for j in range(ny):
        add(vid(nx, j), vid(nx, j + 1))
    for i in range(nx):
        add(vid(i, ny), vid(i + 1, ny))
    for j in range(ny):
        add(vid(0, j), vid(0, j + 1))

    mesh = nd.UnstructuredMesh2D()
    mesh.vx, mesh.vy = vx, vy
    mesh.cell_vertices, mesh.cell_offsets = cell_vertices, cell_offsets
    mesh.material_id = mat_ids
    mesh.bface_v0, mesh.bface_v1, mesh.bface_bc_tag = bv0, bv1, btag
    return mesh


def _solve_mms(mesh):
    cx, cy, _area, _neigh = nearby._unstructured_geometry(mesh)
    phi_exact = np.sin(AX * cx) * np.sin(AY * cy)
    source = (D * (AX**2 + AY**2) + SIG_A) * phi_exact

    m = one_group_absorber()
    solver = nd.FixedSourceSolverUnstructured2D(
        m, mesh, [nd.BoundaryCondition(A=1.0, B=0.0)],
        epsilon=1e-12, max_inner=40000, omega=1.8, verbose=False)
    result = nd.nearby_fixed_source(solver, m, source, mesh=mesh)
    return result, phi_exact, cx, cy


class TestNearbyUnstructuredQuad:
    def test_shapes(self):
        mesh = quad_mesh(16, 16, LX, LY)
        result, _, _, _ = _solve_mms(mesh)
        n = 16 * 16
        assert result.curve_fit.shape == (n, 1)
        assert result.residual.shape == (n, 1)
        assert result.error_estimate.shape == (n, 1)

    def test_error_estimate_tracks_true_error(self):
        mesh = quad_mesh(40, 32, LX, LY)
        result, phi_exact, cx, cy = _solve_mms(mesh)
        num = result.numerical.flux[:, 0]
        te = num - phi_exact
        dx, dy = LX / 40, LY / 32
        interior = ((cx > 4 * dx) & (cx < LX - 4 * dx)
                    & (cy > 4 * dy) & (cy < LY - 4 * dy))
        ratio = np.linalg.norm(result.error_estimate[interior, 0]) / np.linalg.norm(te[interior])
        assert 0.7 < ratio < 1.3

    def test_residual_converges(self):
        r_coarse, _, _, _ = _solve_mms(quad_mesh(20, 16, LX, LY))
        r_fine, _, _, _ = _solve_mms(quad_mesh(40, 32, LX, LY))
        assert np.max(np.abs(r_fine.residual)) < np.max(np.abs(r_coarse.residual))


def _triangle_error(nx, ny):
    """Max interior error of the manufactured solution on a right-triangle mesh."""
    result, phi_exact, cx, cy = _solve_mms(triangle_mesh(nx, ny, LX, LY))
    te = result.numerical.flux[:, 0] - phi_exact
    dx, dy = LX / nx, LY / ny
    interior = ((cx > 4 * dx) & (cx < LX - 4 * dx)
                & (cy > 4 * dy) & (cy < LY - 4 * dy))
    return np.max(np.abs(te[interior])), te[interior], result.error_estimate[interior, 0]


class TestNearbyUnstructuredTriangle:
    """Non-orthogonal cells, so accuracy depends on the deferred correction.

    Without it the two-point flux is inconsistent: the error does not vanish
    under refinement, it grows toward a wrong limit.
    """

    def test_scheme_error_is_small(self):
        err, _, _ = _triangle_error(30, 24)
        assert err < 5e-4

    def test_error_converges_under_refinement(self):
        coarse, _, _ = _triangle_error(15, 12)
        fine, _, _ = _triangle_error(30, 24)
        assert fine < coarse / 2.0        # at least first order, heading to second

    def test_error_estimate_tracks_true_error(self):
        """MNP stays within a factor of ~2 on this mesh.

        Looser than the quad band (0.7-1.3): the corrected triangle error is
        several times smaller, so the estimator's own curve-fit error carries
        proportionally more weight. Measured 1.55-2.07 over 15x12 to 40x32.
        """
        _, te, est = _triangle_error(30, 24)
        ratio = np.linalg.norm(est) / np.linalg.norm(te)
        assert 0.5 < ratio < 3.0


ZERO_BC = [nd.BoundaryCondition(A=1.0, B=0.0)]


def _solve_mms_spline(mesh):
    cx, cy, _area, _neigh = nearby._unstructured_geometry(mesh)
    phi_exact = np.sin(AX * cx) * np.sin(AY * cy)
    source = (D * (AX**2 + AY**2) + SIG_A) * phi_exact
    m = one_group_absorber()
    solver = nd.FixedSourceSolverUnstructured2D(
        m, mesh, ZERO_BC, epsilon=1e-12, max_inner=40000, omega=1.8, verbose=False)
    result = nd.nearby_fixed_source(solver, m, source, mesh=mesh, bc=ZERO_BC)
    te = result.numerical.flux[:, 0] - phi_exact
    return np.linalg.norm(result.error_estimate[:, 0]) / np.linalg.norm(te)


class TestNearbyUnstructuredSpline:
    """The constrained spline fit (picked when bc= is given) on the same
    manufactured problem, over every cell, boundary rows included."""

    def test_quad_estimate_is_exact(self):
        assert _solve_mms_spline(quad_mesh(40, 32, LX, LY)) == pytest.approx(1.0, abs=0.01)

    def test_triangle_estimate_is_close(self):
        # the local fit gives 1.55-2.07 here; measured 1.04-1.07 with the spline
        assert 0.9 < _solve_mms_spline(triangle_mesh(30, 24, LX, LY)) < 1.2

    def test_spline_needs_bc(self):
        mesh = quad_mesh(8, 8, LX, LY)
        m = one_group_absorber()
        solver = nd.FixedSourceSolverUnstructured2D(m, mesh, ZERO_BC, epsilon=1e-10, verbose=False)
        with pytest.raises(ValueError, match="bc="):
            nd.nearby_fixed_source(solver, m, np.ones(64), mesh=mesh, fit="spline")
        with pytest.raises(ValueError, match="fit must be"):
            nd.nearby_fixed_source(solver, m, np.ones(64), mesh=mesh, bc=ZERO_BC, fit="global")


class TestNearbyKSpline:
    """Bare quarter square, one group: phi = cos(pi x / 2L) cos(pi y / 2L) and
    k = nu_f / (Sigma_a + D B^2) with B^2 = 2 (pi / 2L)^2."""

    L = 10.0

    def setup(self, n):
        m = nd.Materials()
        m.n_mat, m.n_groups = 1, 1
        m.D, m.removal, m.scatter, m.chi, m.nusigf = [1.0], [0.1], [0.0], [1.0], [0.13]
        mesh = quad_mesh(n, n, self.L, self.L)
        vx, vy = np.asarray(mesh.vx), np.asarray(mesh.vy)
        far = self.L - 1e-9   # zero flux on x = L and y = L, reflective on the axes
        mesh.bface_bc_tag = [int(min(vx[a], vx[b]) > far or min(vy[a], vy[b]) > far)
                             for a, b in zip(mesh.bface_v0, mesh.bface_v1)]
        bc = [nd.BoundaryCondition(A=0.0, B=1.0), nd.BoundaryCondition(A=1.0, B=0.0)]
        kw = dict(epsilon=1e-8, max_outer=10000, use_cg=True)
        keig = nd.KEigenSolverUnstructured2D(m, mesh, bc, **kw)
        adj = nd.KEigenSolverUnstructured2D(nd.make_adjoint_materials(m), mesh, bc, **kw)
        fixed = nd.FixedSourceSolverUnstructured2D(m, mesh, bc, epsilon=1e-12, max_inner=40000,
                                                   omega=1.8, verbose=False)
        k_exact = 0.13 / (0.1 + 2 * (np.pi / (2 * self.L)) ** 2)
        return m, mesh, bc, keig, adj, fixed, k_exact

    def test_adjoint_and_nearby_estimates_match_the_true_error(self):
        m, mesh, bc, keig, adj, fixed, k_exact = self.setup(20)
        r = nd.nearby_k_eigenvalue(keig, fixed, m, mesh=mesh, bc=bc, adjoint_solver=adj,
                                   tol=1e-10, max_iter=2000)
        true = r.numerical.keff - k_exact
        assert r.k_error_adjoint / true == pytest.approx(1.0, abs=0.05)
        assert (r.k_nearby - r.k_curve_fit) / true == pytest.approx(1.0, abs=0.05)
        assert r.k_error_cells.shape == (400,)
        assert r.k_error_cells.sum() == pytest.approx(r.k_error_adjoint)

    def test_one_shot_only(self):
        m, mesh, bc, keig, adj, fixed, _ = self.setup(10)
        r = nd.nearby_k_eigenvalue(keig, fixed, m, mesh=mesh, bc=bc, adjoint_solver=adj, nearby=False)
        assert r.nearby_flux is None and r.k_nearby is None
        assert np.isfinite(r.k_error_adjoint)
