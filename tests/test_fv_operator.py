"""The assembled unstructured operator reproduces the matrix-free solvers."""

import numpy as np
import pytest

import ndiffusion as nd
from ndiffusion.materials import two_group

pytest.importorskip("scipy")

from .test_nearby_unstructured import quad_mesh, triangle_mesh  # noqa: E402

# fuel and reflector, IAEA-like two-group constants
ROWS = [(1.5, 0.4, 0.010, 0.080, 0.020, 0.0, 0.135),
        (2.0, 0.3, 0.000, 0.010, 0.040, 0.0, 0.0)]


def problem(kind, n=12, L=20.0, jitter=0.0, seed=0):
    """Fuel in x < L/2, reflector beyond; reflective on x = 0 and y = 0, vacuum-like
    Robin elsewhere. ``jitter`` moves interior vertices off the interface by up to that
    fraction of a cell (n must be even so the interface falls on vertices)."""
    mesh = (triangle_mesh if kind == "tri" else quad_mesh)(n, n, L, L)
    vx, vy = np.asarray(mesh.vx, float), np.asarray(mesh.vy, float)
    if jitter:
        rng = np.random.default_rng(seed)
        inner = (vx > 1e-9) & (vx < L - 1e-9) & (vy > 1e-9) & (vy < L - 1e-9)
        inner &= np.abs(vx - L / 2) > 1e-9          # keep the material interface straight
        h = L / n
        vx[inner] += jitter * h * rng.uniform(-1, 1, inner.sum())
        vy[inner] += jitter * h * rng.uniform(-1, 1, inner.sum())
        mesh.vx, mesh.vy = vx, vy
    far = L - 1e-9
    mesh.bface_bc_tag = [int(min(vx[a], vx[b]) > far or min(vy[a], vy[b]) > far)
                         for a, b in zip(mesh.bface_v0, mesh.bface_v1)]
    cx, _cy = (np.asarray(a) for a in nd.cell_centroids(mesh))
    mesh.material_id = (cx > L / 2).astype(int)
    mats = two_group(ROWS)
    bc = [nd.BoundaryCondition(A=0.0, B=1.0), nd.BoundaryCondition(A=0.0, B=1.0),
          nd.BoundaryCondition(A=0.4692, B=1.5), nd.BoundaryCondition(A=0.4692, B=0.4)]
    return mats, mesh, bc


def keig(mats, mesh, bc):
    return nd.KEigenSolverUnstructured2D(mats, mesh, bc, epsilon=1e-12, max_outer=20000,
                                         max_inner=400, use_cg=True).solve()


class TestUnstructuredOperator:
    @pytest.mark.parametrize("kind,jitter", [("quad", 0.0), ("tri", 0.0), ("tri", 0.2), ("quad", 0.2)])
    def test_fixed_source_solution_satisfies_the_matrix(self, kind, jitter):
        mats, mesh, bc = problem(kind, jitter=jitter)
        op = nd.unstructured_operator(mats, mesh, bc)
        q = np.random.default_rng(1).random((len(op.area), 2))
        fs = nd.FixedSourceSolverUnstructured2D(mats, mesh, bc, epsilon=1e-13, max_inner=100000, omega=1.5)
        phi = np.asarray(fs.solve(q).flux)
        rhs = (q * op.area[:, None]).ravel()
        assert np.linalg.norm(op.loss @ phi.ravel() - rhs) / np.linalg.norm(rhs) < 1e-10

    def test_correction_vanishes_only_on_orthogonal_meshes(self):
        assert nd.unstructured_operator(*problem("quad")).correction.nnz == 0
        assert nd.unstructured_operator(*problem("tri")).correction.nnz > 0

    def test_eigenvalue_solution_satisfies_the_matrix(self):
        mats, mesh, bc = problem("tri", jitter=0.2)
        op = nd.unstructured_operator(mats, mesh, bc)
        r = keig(mats, mesh, bc)
        phi = np.asarray(r.flux).ravel()
        Fphi = op.production @ phi / r.keff
        assert np.linalg.norm(op.loss @ phi - Fphi) / np.linalg.norm(Fphi) < 1e-8

    def test_adjoint_materials_do_not_transpose_the_correction(self):
        mats, mesh, bc = problem("tri", jitter=0.2)
        op = nd.unstructured_operator(mats, mesh, bc)
        q = np.random.default_rng(2).random((len(op.area), 2))
        fs = nd.FixedSourceSolverUnstructured2D(nd.make_adjoint_materials(mats), mesh, bc,
                                                epsilon=1e-13, max_inner=100000, omega=1.5)
        x = np.asarray(fs.solve(q).flux).ravel()
        rhs = (q * op.area[:, None]).ravel()
        solver_op = (op.loss + op.correction).T - op.correction
        assert np.linalg.norm(solver_op @ x - rhs) / np.linalg.norm(rhs) < 1e-10
        assert np.linalg.norm(op.loss.T @ x - rhs) / np.linalg.norm(rhs) > 1e-3

    def test_fission_matrix_mode(self):
        mats, mesh, bc = problem("quad", n=4)
        F = nd.unstructured_operator(mats, mesh, bc).production
        chi = np.asarray(mats.chi).reshape(-1, 2)
        nsf = np.asarray(mats.nusigf).reshape(-1, 2)
        mats.nusigf = np.einsum("mg,mh->mgh", chi, nsf).ravel()
        mats.chi = np.zeros_like(chi).ravel()
        assert abs(nd.unstructured_operator(mats, mesh, bc).production - F).max() < 1e-15

    def test_periodic_mesh_is_rejected(self):
        mats, mesh, bc = problem("quad", n=4)
        mesh.periodic_a0, mesh.periodic_a1 = [0], [1]
        mesh.periodic_b0, mesh.periodic_b1 = [20], [21]
        with pytest.raises(ValueError, match="periodic"):
            nd.unstructured_operator(mats, mesh, bc)


def run(kind="tri", method="direct", nearby=True, n=30, **kw):
    """Resolved enough for the fit: knots about 2 cm apart, five spans across each region."""
    mats, mesh, bc = problem(kind, n=n, jitter=0.2)
    ks = nd.KEigenSolverUnstructured2D(mats, mesh, bc, epsilon=1e-12, max_outer=20000,
                                       max_inner=400, use_cg=True)
    fixed = None
    if method == "iterative":
        fixed = nd.FixedSourceSolverUnstructured2D(mats, mesh, bc, epsilon=1e-13,
                                                   max_inner=100000, omega=1.5)
    r = nd.nearby_k_eigenvalue(ks, fixed, mats, mesh=mesh, bc=bc, method=method, nearby=nearby,
                               tol=1e-12, max_iter=20000, **kw)
    return mats, mesh, bc, r


def region_weights(mats, mesh):
    """Absorption in the fuel's outer half (x > 5) over total production."""
    G = 2
    mat = np.asarray(mesh.material_id)
    cx, _ = (np.asarray(a) for a in nd.cell_centroids(mesh))
    rem = np.asarray(mats.removal).reshape(-1, G)
    sc = np.asarray(mats.scatter).reshape(-1, G, G)
    sig_a = rem - (sc.sum(axis=1) - np.einsum("mgg->mg", sc))
    nsf = np.asarray(mats.nusigf).reshape(-1, G)
    num = sig_a[mat] * ((mat == 0) & (cx > 5.0))[:, None]
    return num, nsf[mat]


class TestNearbyDirect:
    def test_direct_and_iterative_nearby_solves_agree(self):
        _, _, _, d = run(method="direct")
        _, _, _, i = run(method="iterative")
        assert d.k_nearby == pytest.approx(i.k_nearby, abs=1e-9)
        assert np.abs(d.nearby_flux - i.nearby_flux).max() < 1e-6 * np.abs(i.nearby_flux).max()
        assert d.operator is not None and i.operator is None

    def test_exact_adjoint_matches_adjoint_materials_on_quads(self):
        mats, mesh, bc = problem("quad")
        ks = nd.KEigenSolverUnstructured2D(mats, mesh, bc, epsilon=1e-12, max_outer=20000,
                                           max_inner=400, use_cg=True)
        adj = nd.KEigenSolverUnstructured2D(nd.make_adjoint_materials(mats), mesh, bc,
                                            epsilon=1e-12, max_outer=20000, max_inner=400, use_cg=True)
        d = nd.nearby_k_eigenvalue(ks, None, mats, mesh=mesh, bc=bc, method="direct", nearby=False)
        i = nd.nearby_k_eigenvalue(ks, None, mats, mesh=mesh, bc=bc, adjoint_solver=adj, nearby=False)
        assert d.k_error_adjoint == pytest.approx(i.k_error_adjoint, rel=1e-6)

    def test_direct_needs_mesh_and_bc(self):
        mats, mesh, bc = problem("quad", n=4)
        ks = nd.KEigenSolverUnstructured2D(mats, mesh, bc, epsilon=1e-10, use_cg=True)
        with pytest.raises(ValueError, match="direct"):
            nd.nearby_k_eigenvalue(ks, None, mats, mesh=mesh, method="direct")


class TestNearbyQoI:
    def test_one_shot_is_the_exact_linearization(self):
        """<Gamma, V r> equals dJ/d(eps) of the problem with source eps V r, by central
        differences, on a non-orthogonal mesh."""
        from scipy.sparse.linalg import splu

        mats, mesh, bc, r = run()
        num, den = region_weights(mats, mesh)
        q = nd.nearby_qoi(r, mats, mesh, bc, num, den)
        linear = q.cells.sum()

        op = r.operator
        lu = splu(op.loss.tocsc())
        phi0 = np.asarray(r.numerical.flux).ravel()
        Vr = (r.residual * op.area[:, None]).ravel()

        def J(eps):
            phi, k = phi0.copy(), r.numerical.keff
            for _ in range(20000):
                Fp = op.production @ phi
                new = lu.solve(Fp / k + eps * Vr)
                k_new = k * (op.production @ new).sum() / Fp.sum()
                done = abs(k_new - k) < 1e-15 and np.abs(new - phi).max() < 1e-13 * np.abs(phi).max()
                phi, k = new, k_new
                if done:
                    break
            p = phi.reshape(-1, 2)
            return np.sum(num * p * op.area[:, None]) / np.sum(den * p * op.area[:, None])

        eps = 0.01
        central = (J(eps) - J(-eps)) / (2 * eps)
        assert linear == pytest.approx(central, rel=1e-4)

    def test_one_shot_tracks_the_nearby_estimate(self):
        mats, mesh, bc, r = run()
        num, den = region_weights(mats, mesh)
        q = nd.nearby_qoi(r, mats, mesh, bc, num, den)
        assert q.error_nearby == pytest.approx(q.error_adjoint, rel=0.02)
        assert q.value == pytest.approx(
            np.sum(num * r.numerical.flux * r.operator.area[:, None])
            / np.sum(den * r.numerical.flux * r.operator.area[:, None]))

    def test_stacked_weights(self):
        mats, mesh, bc, r = run(nearby=False)
        num, den = region_weights(mats, mesh)
        one = nd.nearby_qoi(r, mats, mesh, bc, num, den)
        both = nd.nearby_qoi(r, mats, mesh, bc, np.stack([num, 2 * num]), den)
        assert both.error_adjoint.shape == (2,) and both.error_nearby is None
        assert both.error_adjoint[0] == pytest.approx(one.error_adjoint)
        assert both.error_adjoint[1] == pytest.approx(2 * one.error_adjoint)
        assert both.cells.shape == (2, len(r.operator.area))

    def test_needs_the_spline_fit(self):
        mats, mesh, bc = problem("quad", n=6)
        ks = nd.KEigenSolverUnstructured2D(mats, mesh, bc, epsilon=1e-10, use_cg=True)
        fixed = nd.FixedSourceSolverUnstructured2D(mats, mesh, bc, epsilon=1e-12, omega=1.5)
        r = nd.nearby_k_eigenvalue(ks, fixed, mats, mesh=mesh, fit="local", nearby=False)
        num, den = region_weights(mats, mesh)
        with pytest.raises(ValueError, match="spline"):
            nd.nearby_qoi(r, mats, mesh, bc, num, den)
