"""Tests for the constrained global spline fit used by MNP on unstructured meshes
(ndiffusion._spline_fit): corner exponents and modes, the choice between the
"smooth" and "corner" configurations, and the fit itself on a problem with an
analytic solution."""

import numpy as np
import pytest

import ndiffusion as nd

pytest.importorskip("scipy")
from ndiffusion import _spline_fit as sf  # noqa: E402

REFLECT = nd.BoundaryCondition(A=0.0, B=1.0)
ZERO = nd.BoundaryCondition(A=1.0, B=0.0)


def quad_mesh(n, x0, x1, material, tag):
    """n x n quads on [x0, x1]^2; material(x, y) per cell center, tag(x, y) per
    boundary face midpoint."""
    h = (x1 - x0) / n
    vid = lambda i, j: i * (n + 1) + j  # noqa: E731
    m = nd.UnstructuredMesh2D()
    m.vx = [x0 + i * h for i in range(n + 1) for j in range(n + 1)]
    m.vy = [x0 + j * h for i in range(n + 1) for j in range(n + 1)]
    cv, offs, mats = [], [0], []
    for i in range(n):
        for j in range(n):
            cv += [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
            offs.append(len(cv))
            mats.append(material(x0 + (i + 0.5) * h, x0 + (j + 0.5) * h))
    m.cell_vertices, m.cell_offsets, m.material_id = cv, offs, mats
    bv0, bv1, tags = [], [], []
    for k in range(n):
        for a, b in ((vid(k, 0), vid(k + 1, 0)), (vid(n, k), vid(n, k + 1)),
                     (vid(k + 1, n), vid(k, n)), (vid(0, k + 1), vid(0, k))):
            bv0.append(a)
            bv1.append(b)
            tags.append(tag(0.5 * (m.vx[a] + m.vx[b]), 0.5 * (m.vy[a] + m.vy[b])))
    m.bface_v0, m.bface_v1, m.bface_bc_tag = bv0, bv1, tags
    return m


def two_group(n_mat):
    data = [dict(D=[1.4, 0.4], Siga=[0.01, 0.08 + 0.02 * i], Scat=[[0.0, 0.0], [0.02, 0.0]],
                 nuSigf=[0.0, 0.12], chi=[1.0, 0.0]) for i in range(n_mat)]
    data[-1]["D"] = [1.0, 0.25]
    return nd.make_materials(data, 2)


class TestCornerExponents:
    @pytest.mark.parametrize("ratio", [0.1, 0.5, 10.0])
    def test_roots_satisfy_the_trace_condition(self, ratio):
        q = np.pi / 2
        sectors = [(q, 1.0), (q, 1.0), (q, ratio), (q, 1.0)]
        roots = sf.corner_exponents(sectors, amax=2.5)
        assert roots and min(roots) < 1.0          # a material contrast makes the corner singular
        for a in roots:
            assert abs(sf._trace_minus_two(a, sectors)) < 1e-6   # double roots only to ~1e-6

    def test_inverse_ratio_gives_the_same_exponents(self):
        q = np.pi / 2
        a = sf.corner_exponents([(q, 1.0), (q, 1.0), (q, 0.2), (q, 1.0)])
        b = sf.corner_exponents([(q, 1.0), (q, 1.0), (q, 5.0), (q, 1.0)])
        np.testing.assert_allclose(a, b, atol=1e-8)


def corner_problem():
    """A 270/90-degree material corner at the origin of [-10, 10]^2."""
    mats = two_group(2)
    mesh = quad_mesh(20, -10.0, 10.0, lambda x, y: int(x > 0 and y > 0), lambda x, y: 0)
    inter, _ = sf.mesh_faces(mesh, np.asarray(mesh.material_id))
    area = np.asarray(nd.cell_areas(mesh))
    regs, comp = sf.regions(mesh, np.asarray(mesh.material_id), area, 2.0)
    cx, cy = (np.asarray(a) for a in nd.cell_centroids(mesh))
    return mats, mesh, inter, comp, cx, cy


class TestCornerModes:
    def test_modes_found_at_the_corner_only(self):
        mats, mesh, inter, comp, cx, cy = corner_problem()
        modes = sf.corner_modes(mesh, inter, comp, mats, cx, cy, 1.0, min_turn=sf.CORNER_TURN)
        assert modes
        assert all(abs(md.x0) < 1e-9 and abs(md.y0) < 1e-9 for md in modes)
        assert all(np.isinf(md.R2) for md in modes)   # both interfaces run to the boundary

    def test_value_and_current_continuous_across_the_rays(self):
        mats, mesh, inter, comp, cx, cy = corner_problem()
        D = np.asarray(mats.D).reshape(-1, 2)
        for md in sf.corner_modes(mesh, inter, comp, mats, cx, cy, 1.0, min_turn=sf.CORNER_TURN):
            for th in md.starts:
                t = np.array([np.cos(th), np.sin(th)])
                n = np.array([-t[1], t[0]])
                p = 3.0 * t
                (va, gxa, gya), (vb, gxb, gyb) = (md.eval(*(p + s * 1e-7 * n)) for s in (1, -1))
                Da = D[int((p + 0.5 * n)[0] > 0 and (p + 0.5 * n)[1] > 0), md.group]
                Db = D[int((p - 0.5 * n)[0] > 0 and (p - 0.5 * n)[1] > 0), md.group]
                assert va[0] == pytest.approx(vb[0], abs=1e-5)
                ja = Da * (gxa[0] * n[0] + gya[0] * n[1])
                jb = Db * (gxb[0] * n[0] + gyb[0] * n[1])
                assert ja == pytest.approx(jb, rel=1e-4, abs=1e-6)

    def test_cut_off_laplacian_matches_finite_differences(self):
        md = sf.CornerMode(0.0, 0.0, 0.73, 0, [0.0], [2 * np.pi], np.array([[1.0, 0.4]]), [0], R1=2.0, R2=5.0)
        h = 1e-4
        for x, y in ((1.0, 0.7), (2.6, 1.9), (-3.1, 2.2), (0.4, -4.4)):
            fd = (md.eval(x + h, y)[0] + md.eval(x - h, y)[0] + md.eval(x, y + h)[0]
                  + md.eval(x, y - h)[0] - 4 * md.eval(x, y)[0]) / h**2
            assert md.lap(x, y)[0][0] == pytest.approx(fd[0], abs=1e-4)
        assert md.lap(1.0, 0.7)[0][0] == 0.0          # harmonic inside R1


class TestConfig:
    def test_material_corner_picks_corner(self):
        mats, mesh, *_ = corner_problem()
        assert sf.choose_config(mats, mesh, [REFLECT] * 2, 2.0)[0] == "corner"

    def test_straight_stripes_pick_smooth(self):
        mats = two_group(2)
        mesh = quad_mesh(20, 0.0, 20.0, lambda x, y: int(x > 10), lambda x, y: 0)
        assert sf.choose_config(mats, mesh, [REFLECT] * 2, 2.0) == ("smooth", [])

    def test_small_enclosed_region_is_an_inclusion(self):
        mesh = quad_mesh(30, 0.0, 30.0, lambda x, y: int(abs(x - 15) < 2 and abs(y - 15) < 2),
                         lambda x, y: 1 if x > 29.9 or y > 29.9 else 0)
        bc = [REFLECT, REFLECT, ZERO, ZERO]
        area = np.asarray(nd.cell_areas(mesh))
        mat = np.asarray(mesh.material_id)
        regs, comp = sf.regions(mesh, mat, area, 2.0)
        inc = sf.small_inclusions(mesh, bc, 2, regs, comp, area, 2.0)
        assert [m for m, _ in inc] == [1]             # the rod, not the open background
        assert sf.small_inclusions(mesh, bc, 2, regs, comp, area, 2.0, max_spans=1.0) == []


class TestFit:
    L = 10.0

    def solve(self, n):
        m = nd.Materials()
        m.n_mat, m.n_groups = 1, 1
        m.D, m.removal, m.scatter, m.chi, m.nusigf = [1.0], [0.1], [0.0], [1.0], [0.13]
        mesh = quad_mesh(n, 0.0, self.L, lambda x, y: 0, lambda x, y: 1 if max(x, y) > self.L - 1e-9 else 0)
        bc = [REFLECT, ZERO]
        r = nd.KEigenSolverUnstructured2D(m, mesh, bc, epsilon=1e-8, max_outer=10000, use_cg=True).solve()
        return m, mesh, bc, r

    def test_cosine_mode_curvature_and_k_estimate(self):
        """Bare quarter square: phi = cos(pi x / 2L) cos(pi y / 2L), so
        lap(phi) / phi = -2 (pi / 2L)^2 and k = nu_f / (Sigma_a + D B^2)."""
        m, mesh, bc, r = self.solve(20)
        phi = np.asarray(r.flux)
        f = sf.fit(m, mesh, phi, r.keff, bc, stride=2.0)
        assert f.config == "smooth"
        assert np.abs(f.curve - phi).max() < 1e-3 * np.abs(phi).max()
        B2 = 2 * (np.pi / (2 * self.L)) ** 2
        cx, cy = (np.asarray(a) for a in nd.cell_centroids(mesh))
        inner = (cx < 0.8 * self.L) & (cy < 0.8 * self.L)
        np.testing.assert_allclose(f.lap[inner, 0] / f.curve[inner, 0], -B2, rtol=1e-2)

        # one-shot adjoint estimate of the k error (the operator is self-adjoint here)
        area = np.asarray(nd.cell_areas(mesh))
        loss = m.D[0] * -f.lap[:, 0] + m.removal[0] * f.curve[:, 0]
        prod = m.nusigf[0] * f.curve[:, 0]
        eta = np.sum(phi[:, 0] * (r.keff * loss - prod) * area) / np.sum(phi[:, 0] * loss * area)
        true = r.keff - m.nusigf[0] / (m.removal[0] + m.D[0] * B2)
        assert eta / true == pytest.approx(1.0, abs=0.05)

    def test_fixed_source_needs_the_corner_config(self):
        m, mesh, bc, r = self.solve(10)
        with pytest.raises(ValueError, match="corner"):
            sf.fit(m, mesh, np.asarray(r.flux), np.inf, bc, config="smooth", source=np.ones((100, 1)))

    def test_evaluate_reproduces_the_centroid_values(self):
        m, mesh, bc, r = self.solve(10)
        f = sf.fit(m, mesh, np.asarray(r.flux), r.keff, bc)
        cx, cy = (np.asarray(a) for a in nd.cell_centroids(mesh))
        np.testing.assert_allclose(f.evaluate(np.arange(len(cx)), cx, cy), f.curve, rtol=1e-12, atol=1e-14)
