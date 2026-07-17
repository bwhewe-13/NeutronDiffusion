"""Tests for the 2-D structured method of nearby problems.

Manufactured solution (reflective at x=0, y=0; zero-flux at x=Rx, y=Ry):
    phi_exact(x, y) = cos(pi x / (2 Rx)) cos(pi y / (2 Ry))
gives a distributed O(h^2) discretization error the estimate should recover.
"""

import numpy as np
import pytest

import ndiffusion as nd

pytest.importorskip("scipy")

D = 1.0
SIG_A = 0.2
RX, RY = 10.0, 8.0
BX, BY = np.pi / (2 * RX), np.pi / (2 * RY)


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


def zero_flux():
    return nd.BoundaryCondition(A=1.0, B=0.0)


def _solve_mms(nx, ny):
    ex = np.linspace(0.0, RX, nx + 1)
    ey = np.linspace(0.0, RY, ny + 1)
    xc = 0.5 * (ex[:-1] + ex[1:])
    yc = 0.5 * (ey[:-1] + ey[1:])
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    phi_exact = (np.cos(BX * X) * np.cos(BY * Y)).ravel()  # cell = i*ny + j
    source = (D * (BX**2 + BY**2) + SIG_A) * phi_exact

    mmap = [0] * (nx * ny)
    m = one_group_absorber()
    solver = nd.FixedSourceSolver2D(m, mmap, list(ex), list(ey), nd.Geometry2D.XY,
                                    bc_x=[zero_flux()], bc_y=[zero_flux()],
                                    epsilon=1e-12, max_inner=5000, verbose=False)
    result = nd.nearby_fixed_source(solver, m, source, medium_map=mmap,
                                    edges_x=list(ex), edges_y=list(ey),
                                    geometry=nd.Geometry2D.XY)
    return result, phi_exact, (nx, ny)


class TestNearbyFixedSource2DStructured:
    def test_shapes(self):
        result, _, (nx, ny) = _solve_mms(20, 16)
        assert result.curve_fit.shape == (nx * ny,)
        assert result.residual.shape == (nx * ny,)
        assert result.error_estimate.shape == (nx * ny,)

    def test_error_estimate_tracks_true_error(self):
        result, phi_exact, (nx, ny) = _solve_mms(40, 32)
        num = np.asarray(result.numerical.flux)
        te = (num - phi_exact).reshape(nx, ny)
        ee = result.error_estimate.reshape(nx, ny)
        s = (slice(2, -2), slice(2, -2))
        ratio = np.linalg.norm(ee[s]) / np.linalg.norm(te[s])
        assert 0.85 < ratio < 1.15

    def test_second_order_convergence(self):
        errs = []
        for grid in [(20, 16), (40, 32), (80, 64)]:
            result, phi_exact, (nx, ny) = _solve_mms(*grid)
            num = np.asarray(result.numerical.flux)
            te = (num - phi_exact).reshape(nx, ny)[2:-2, 2:-2]
            errs.append(np.max(np.abs(te)))
        assert np.log2(errs[0] / errs[1]) > 1.8
        assert np.log2(errs[1] / errs[2]) > 1.8
