"""Tests for the 1-D method of nearby problems (nearby_fixed_source /
nearby_k_eigenvalue).

The fixed-source checks use a manufactured non-polynomial solution
    phi_exact(x) = cos(pi x / (2R))   (reflective at x=0, zero-flux at x=R)
so the cell-centred scheme has genuine, distributed O(h^2) truncation error.
MNP should recover that error: in the interior the error *estimate* tracks the
true error, and both converge at second order.
"""

import numpy as np
import pytest

import ndiffusion as nd

pytest.importorskip("scipy")

D = 1.0
SIG_A = 0.2
R = 10.0
B = np.pi / (2 * R)


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


def _solve_mms(cells):
    edges = np.linspace(0.0, R, cells + 1)
    xc = 0.5 * (edges[:-1] + edges[1:])
    mmap = [0] * cells
    m = one_group_absorber()
    phi_exact = np.cos(B * xc)
    source = (D * B * B + SIG_A) * phi_exact

    solver = nd.FixedSourceSolver(m, mmap, list(edges), nd.Geometry.Slab,
                                  [zero_flux()], epsilon=1e-12, max_inner=2000,
                                  verbose=False)
    result = nd.nearby_fixed_source(solver, m, source, medium_map=mmap,
                                    edges_x=list(edges), geometry=nd.Geometry.Slab)
    return result, phi_exact


class TestNearbyFixedSource1D:
    def test_shapes_and_result_fields(self):
        result, _ = _solve_mms(40)
        assert result.numerical is not None
        assert result.nearby is not None
        assert result.curve_fit.shape == (40,)
        assert result.residual.shape == (40,)
        assert result.error_estimate.shape == (40,)

    def test_error_estimate_tracks_true_error(self):
        result, phi_exact = _solve_mms(80)
        num = np.asarray(result.numerical.flux)
        true_err = num - phi_exact
        s = slice(2, -2)  # interior (MNP is boundary-limited)
        ratio = np.linalg.norm(result.error_estimate[s]) / np.linalg.norm(true_err[s])
        assert 0.85 < ratio < 1.15

    def test_second_order_convergence(self):
        _, phi0 = _solve_mms(40)
        errs = []
        for cells in (40, 80, 160):
            result, phi_exact = _solve_mms(cells)
            num = np.asarray(result.numerical.flux)
            errs.append(np.max(np.abs((num - phi_exact)[2:-2])))
        order1 = np.log2(errs[0] / errs[1])
        order2 = np.log2(errs[1] / errs[2])
        assert order1 > 1.8 and order2 > 1.8

    def test_residual_converges(self):
        r40, _ = _solve_mms(40)
        r80, _ = _solve_mms(80)
        m40 = np.max(np.abs(r40.residual[2:-2]))
        m80 = np.max(np.abs(r80.residual[2:-2]))
        assert m80 < m40  # residual shrinks under refinement
        assert m40 < 1e-2

    def test_return_nearby_false(self):
        edges = np.linspace(0.0, R, 41)
        xc = 0.5 * (edges[:-1] + edges[1:])
        m = one_group_absorber()
        source = (D * B * B + SIG_A) * np.cos(B * xc)
        solver = nd.FixedSourceSolver(m, [0] * 40, list(edges), nd.Geometry.Slab,
                                      [zero_flux()], epsilon=1e-10, verbose=False)
        result = nd.nearby_fixed_source(solver, m, source, medium_map=[0] * 40,
                                        edges_x=list(edges),
                                        geometry=nd.Geometry.Slab,
                                        return_nearby=False)
        assert result.nearby is None
        assert result.error_estimate is None
        assert result.residual.shape == (40,)


class TestNearbyKEigenvalue1D:
    def two_group(self):
        m = nd.Materials()
        m.n_mat = 1
        m.n_groups = 2
        m.D = [1.5, 0.4]
        m.removal = [0.03, 0.10]
        m.scatter = [0.0, 0.0, 0.02, 0.0]
        m.chi = [1.0, 0.0]
        m.nusigf = [0.005, 0.15]
        return m

    def test_nearby_keff_consistent(self):
        m = self.two_group()
        cells = 80
        edges = list(np.linspace(0.0, 40.0, cells + 1))
        mmap = [0] * cells
        bc = [zero_flux(), zero_flux()]

        keig = nd.KEigenSolver(m, mmap, edges, nd.Geometry.Slab, bc,
                               epsilon=1e-10, verbose=False)
        fixed = nd.FixedSourceSolver(m, mmap, edges, nd.Geometry.Slab, bc,
                                     epsilon=1e-10, max_inner=1000, verbose=False)
        r = nd.nearby_k_eigenvalue(keig, fixed, m, medium_map=mmap,
                                   edges_x=edges, geometry=nd.Geometry.Slab)

        assert r.nearby_flux.shape == (cells * 2,)
        assert abs(r.k_nearby - r.numerical.keff) < 1e-3
        assert abs(r.k_curve_fit - r.numerical.keff) < 1e-2
        assert r.nearby_rate > 0.0


class TestFissionSourceHelper:
    def test_matches_manual_standard_mode(self):
        m = nd.Materials()
        m.n_mat = 1
        m.n_groups = 2
        m.D = [1.0, 1.0]
        m.removal = [0.1, 0.1]
        m.scatter = [0.0, 0.0, 0.0, 0.0]
        m.chi = [1.0, 0.0]
        m.nusigf = [0.01, 0.2]
        flux = np.array([2.0, 3.0])  # one cell, two groups
        fis = nd.fission_source(m, [0], flux)
        rate = 0.01 * 2.0 + 0.2 * 3.0
        assert np.allclose(fis, [1.0 * rate, 0.0 * rate])
