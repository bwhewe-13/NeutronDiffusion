"""Robin boundary conditions on the low-coordinate edges.

The 1-D left edge and the 2-D structured left and bottom edges take a BC like
the right and top ones, defaulting to reflective.  A full domain with vacuum on
every side must reproduce the half (1-D) or quarter (2-D) domain with symmetry
on the cut, and mirroring the mesh while swapping the two BCs must mirror the
flux - which pins the outward-normal sign convention on the new edges.
"""

import numpy as np
import pytest

import ndiffusion as nd

REFLECTIVE = nd.BoundaryCondition(A=0.0, B=1.0)
ZERO = nd.BoundaryCondition(A=1.0, B=0.0)


def two_group():
    m = nd.Materials()
    m.n_mat = 2
    m.n_groups = 2
    m.D = [1.4, 0.4, 1.2, 0.3]
    m.removal = [0.03, 0.1, 0.04, 0.02]
    m.scatter = [0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.035, 0.0]
    m.chi = [1.0, 0.0, 1.0, 0.0]
    m.nusigf = [0.005, 0.12, 0.0, 0.0]
    m.velocity = [2.0e7, 2.2e5]
    return m


def marshak(mats):
    return [nd.BoundaryCondition(A=0.25, B=0.5 * d) for d in mats.D[:mats.n_groups]]


# 1-D: 30 cm of fuel inside 10 cm of reflector, on a non-uniform mesh.
HALF_EDGES = np.concatenate([np.linspace(0.0, 30.0, 16), np.linspace(30.0, 40.0, 9)[1:]])
HALF_MAP = [0] * 15 + [1] * 8
FULL_EDGES = np.concatenate([-HALF_EDGES[::-1], HALF_EDGES[1:]])
FULL_MAP = HALF_MAP[::-1] + HALF_MAP


class TestSlab:
    def test_k_eigenvalue_full_equals_half(self):
        m = two_group()
        vac = marshak(m)
        half = nd.KEigenSolver(m, HALF_MAP, HALF_EDGES, nd.Geometry.Slab, vac,
                               epsilon=1e-11).solve()
        full = nd.KEigenSolver(m, FULL_MAP, FULL_EDGES, nd.Geometry.Slab, vac,
                               epsilon=1e-11, bc_left=vac).solve()
        assert full.keff == pytest.approx(half.keff, abs=1e-10)
        n = len(HALF_MAP)
        np.testing.assert_allclose(full.flux[n:] / full.flux[n:].max(),
                                   half.flux / half.flux.max(), atol=1e-8)
        np.testing.assert_allclose(full.flux[:n][::-1], full.flux[n:], rtol=1e-8)

    def test_fixed_source_full_equals_half(self):
        m = two_group()
        vac = marshak(m)
        q = np.zeros((len(HALF_MAP), 2))
        q[:5, 0] = 1.0
        half = nd.FixedSourceSolver(m, HALF_MAP, HALF_EDGES, nd.Geometry.Slab, vac,
                                    epsilon=1e-12, max_inner=5000).solve(q)
        full = nd.FixedSourceSolver(m, FULL_MAP, FULL_EDGES, nd.Geometry.Slab, vac,
                                    epsilon=1e-12, max_inner=5000,
                                    bc_left=vac).solve(np.vstack([q[::-1], q]))
        assert half.converged and full.converged
        np.testing.assert_allclose(full.flux[len(HALF_MAP):], half.flux, rtol=1e-9)

    def test_time_dependent_full_equals_half(self):
        m = two_group()
        vac = marshak(m)
        phi0 = np.ones((len(HALF_MAP), 2))
        half = nd.TimeDependentSolver(m, HALF_MAP, HALF_EDGES, nd.Geometry.Slab, vac,
                                      initial_flux=phi0, epsilon=1e-12,
                                      max_inner=500, theta=0.5)
        full = nd.TimeDependentSolver(m, FULL_MAP, FULL_EDGES, nd.Geometry.Slab, vac,
                                      initial_flux=np.vstack([phi0, phi0]),
                                      epsilon=1e-12, max_inner=500, theta=0.5,
                                      bc_left=vac)
        rh, rf = half.run(1e-4, 5), full.run(1e-4, 5)
        np.testing.assert_allclose(rf.flux[len(HALF_MAP):], rh.flux, rtol=1e-9)

    def test_mirrored_mesh_mirrors_flux(self):
        # Different conditions on the two ends: reversing the mesh and swapping
        # them must reverse the flux, so both edges use the outward normal.
        m = two_group()
        right, left = marshak(m), [ZERO, ZERO]
        edges = HALF_EDGES
        q = np.ones((len(HALF_MAP), 2))
        a = nd.FixedSourceSolver(m, HALF_MAP, edges, nd.Geometry.Slab, right,
                                 epsilon=1e-12, max_inner=5000, bc_left=left).solve(q)
        b = nd.FixedSourceSolver(m, HALF_MAP[::-1], (edges[-1] - edges)[::-1],
                                 nd.Geometry.Slab, left, epsilon=1e-12,
                                 max_inner=5000, bc_left=right).solve(q)
        np.testing.assert_allclose(b.flux[::-1], a.flux, rtol=1e-9)
        assert a.flux[0, 1] < a.flux[-1, 1]

    def test_default_is_reflective(self):
        m = two_group()
        vac = marshak(m)
        a = nd.KEigenSolver(m, HALF_MAP, HALF_EDGES, nd.Geometry.Slab, vac).solve()
        b = nd.KEigenSolver(m, HALF_MAP, HALF_EDGES, nd.Geometry.Slab, vac,
                            bc_left=[REFLECTIVE, REFLECTIVE]).solve()
        assert a.keff == b.keff
        np.testing.assert_array_equal(a.flux, b.flux)

    def test_wrong_length_raises(self):
        m = two_group()
        with pytest.raises(ValueError, match="bc_left"):
            nd.KEigenSolver(m, HALF_MAP, HALF_EDGES, nd.Geometry.Slab, marshak(m),
                            bc_left=[ZERO])


class TestCurvilinear:
    @pytest.mark.parametrize("geom", [nd.Geometry.Cylinder, nd.Geometry.Sphere])
    def test_axis_rejects_non_reflective(self, geom):
        m = two_group()
        with pytest.raises(ValueError, match="r = 0"):
            nd.KEigenSolver(m, HALF_MAP, HALF_EDGES, geom, marshak(m),
                            bc_left=[ZERO, REFLECTIVE])
        # Reflective on the axis is the default and stays allowed.
        nd.KEigenSolver(m, HALF_MAP, HALF_EDGES, geom, marshak(m),
                        bc_left=[REFLECTIVE, REFLECTIVE])

    def test_annulus_inner_surface(self):
        # A hollow cylinder: a black inner surface must lower keff.
        m = two_group()
        edges = HALF_EDGES + 5.0
        closed = nd.KEigenSolver(m, HALF_MAP, edges, nd.Geometry.Cylinder,
                                 marshak(m)).solve()
        black = nd.KEigenSolver(m, HALF_MAP, edges, nd.Geometry.Cylinder,
                                marshak(m), bc_left=[ZERO, ZERO]).solve()
        assert black.keff < closed.keff - 1e-3


# 2-D: the quarter is a 20 x 20 cm fuel block in reflector, non-uniform mesh.
N = 8
Q_EDGES = np.array([0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 29.0, 35.0])
F_EDGES = np.concatenate([-Q_EDGES[::-1], Q_EDGES[1:]])


def unfold(a):
    """Reflect a quarter-core (N, N, ...) array into the full (2N, 2N, ...) one."""
    a = np.concatenate([a[::-1], a], axis=0)
    return np.concatenate([a[:, ::-1], a], axis=1)


Q_MAP = np.where((np.arange(N)[:, None] < 5) & (np.arange(N)[None, :] < 5), 0, 1)
F_MAP = unfold(Q_MAP)


def quarter_of(flux):
    return flux.reshape(2 * N, 2 * N, -1)[N:, N:].reshape(N * N, -1)


class TestStructured2D:
    @pytest.mark.parametrize("use_cg", [False, True])
    def test_k_eigenvalue_full_equals_quarter(self, use_cg):
        m = two_group()
        vac = marshak(m)
        q = nd.KEigenSolver2D(m, Q_MAP, Q_EDGES, Q_EDGES, nd.Geometry2D.XY,
                              vac, vac, epsilon=1e-11, use_cg=use_cg).solve()
        f = nd.KEigenSolver2D(m, F_MAP, F_EDGES, F_EDGES, nd.Geometry2D.XY,
                              vac, vac, epsilon=1e-11, use_cg=use_cg,
                              bc_x_left=vac, bc_y_bottom=vac).solve()
        assert f.keff == pytest.approx(q.keff, abs=1e-9)
        fq = quarter_of(f.flux)
        np.testing.assert_allclose(fq / fq.max(), q.flux / q.flux.max(), atol=1e-7)

    def test_fixed_source_full_equals_quarter(self):
        m = two_group()
        vac = marshak(m)
        src = np.zeros((N, N, 2))
        src[:3, :3, 0] = 1.0
        q = nd.FixedSourceSolver2D(m, Q_MAP, Q_EDGES, Q_EDGES, nd.Geometry2D.XY,
                                   vac, vac, epsilon=1e-12, max_inner=5000
                                   ).solve(src.reshape(-1, 2))
        f = nd.FixedSourceSolver2D(m, F_MAP, F_EDGES, F_EDGES, nd.Geometry2D.XY,
                                   vac, vac, epsilon=1e-12, max_inner=5000,
                                   bc_x_left=vac, bc_y_bottom=vac
                                   ).solve(unfold(src).reshape(-1, 2))
        assert q.converged and f.converged
        np.testing.assert_allclose(quarter_of(f.flux), q.flux, rtol=1e-8)

    def test_time_dependent_full_equals_quarter(self):
        m = two_group()
        vac = marshak(m)
        q = nd.TimeDependentSolver2D(m, Q_MAP, Q_EDGES, Q_EDGES, nd.Geometry2D.XY,
                                     vac, vac, initial_flux=np.ones((N * N, 2)),
                                     epsilon=1e-12, max_inner=2000)
        f = nd.TimeDependentSolver2D(m, F_MAP, F_EDGES, F_EDGES, nd.Geometry2D.XY,
                                     vac, vac, initial_flux=np.ones((4 * N * N, 2)),
                                     epsilon=1e-12, max_inner=2000,
                                     bc_x_left=vac, bc_y_bottom=vac)
        rq, rf = q.run(1e-4, 3), f.run(1e-4, 3)
        np.testing.assert_allclose(quarter_of(rf.flux), rq.flux, rtol=1e-8)

    def test_mirrored_in_x_mirrors_flux(self):
        m = two_group()
        right, left = marshak(m), [ZERO, ZERO]
        src = np.ones((N * N, 2))
        a = nd.FixedSourceSolver2D(m, Q_MAP, Q_EDGES, Q_EDGES, nd.Geometry2D.XY,
                                   right, right, epsilon=1e-12, max_inner=5000,
                                   bc_x_left=left).solve(src)
        b = nd.FixedSourceSolver2D(m, Q_MAP[::-1], (Q_EDGES[-1] - Q_EDGES)[::-1],
                                   Q_EDGES, nd.Geometry2D.XY, left, right,
                                   epsilon=1e-12, max_inner=5000,
                                   bc_x_left=right).solve(src)
        np.testing.assert_allclose(b.flux.reshape(N, N, 2)[::-1],
                                   a.flux.reshape(N, N, 2), rtol=1e-8)

    def test_default_is_reflective(self):
        m = two_group()
        vac = marshak(m)
        a = nd.KEigenSolver2D(m, Q_MAP, Q_EDGES, Q_EDGES, nd.Geometry2D.XY,
                              vac, vac).solve()
        b = nd.KEigenSolver2D(m, Q_MAP, Q_EDGES, Q_EDGES, nd.Geometry2D.XY,
                              vac, vac, bc_x_left=[REFLECTIVE] * 2,
                              bc_y_bottom=[REFLECTIVE] * 2).solve()
        assert a.keff == b.keff
        np.testing.assert_array_equal(a.flux, b.flux)

    def test_rz_axis_rejects_non_reflective(self):
        m = two_group()
        vac = marshak(m)
        with pytest.raises(ValueError, match="bc_y_bottom"):
            nd.KEigenSolver2D(m, Q_MAP, Q_EDGES, Q_EDGES, nd.Geometry2D.RZ,
                              vac, vac, bc_y_bottom=vac)
        # The axial (z) bottom is an ordinary edge, as is r > 0.
        nd.KEigenSolver2D(m, Q_MAP, Q_EDGES, Q_EDGES, nd.Geometry2D.RZ,
                          vac, vac, bc_x_left=vac)
        nd.KEigenSolver2D(m, Q_MAP, Q_EDGES, Q_EDGES + 1.0, nd.Geometry2D.RZ,
                          vac, vac, bc_y_bottom=vac)
