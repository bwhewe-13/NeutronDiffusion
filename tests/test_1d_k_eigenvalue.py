"""
Pytest suite for KEigenSolver (k-eigenvalue problems).

Reference values match those in src/main.cpp and the original
Python matrix_solutions.py reference suite.
"""

import numpy as np
import pytest

import ndiffusion as nd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def linspace(start, stop, n):
    return list(np.linspace(start, stop, n))


def uniform_map(cells):
    return [0] * cells


def two_mat_map(cells, interface):
    return [0] * interface + [1] * (cells - interface)


def zero_flux():
    return nd.BoundaryCondition(A=1.0, B=0.0)


def marshak(D):
    return nd.BoundaryCondition(A=0.25, B=D / 2.0)


def reflective():
    return nd.BoundaryCondition(A=0.0, B=1.0)


def one_group_mat():
    """1-group, 1-material cross sections."""
    m = nd.Materials()
    m.n_mat = 1
    m.n_groups = 1
    m.D = [3.850204978408833]
    m.removal = [0.1532]
    m.scatter = [0.0]
    m.chi = [1.0]
    m.nusigf = [0.1570]
    m.velocity = [2.2e5]  # not used by KEigenSolver, but valid to set
    return m


# ---------------------------------------------------------------------------
# 1-group slab
# ---------------------------------------------------------------------------


class TestOneGroupSlab:
    def test_one_material(self):
        m = one_group_mat()
        cells = 20
        solver = nd.KEigenSolver(
            m,
            uniform_map(cells),
            linspace(0.0, 50.0, cells + 1),
            nd.Geometry.Slab,
            [zero_flux()],
            epsilon=1e-8,
        )
        res = solver.solve()
        assert abs(res.keff - 1.00001244) < 1e-4

    def test_two_materials(self):
        m = nd.Materials()
        m.n_mat = 2
        m.n_groups = 1
        m.D = [5.0, 1.0]
        m.removal = [0.5, 0.01]
        m.scatter = [0.0, 0.0]
        m.chi = [1.0, 1.0]
        m.nusigf = [0.7, 0.0]
        cells = 100
        solver = nd.KEigenSolver(
            m,
            two_mat_map(cells, 50),
            linspace(0.0, 10.0, cells + 1),
            nd.Geometry.Slab,
            [zero_flux()],
            epsilon=1e-8,
        )
        res = solver.solve()
        assert abs(res.keff - 1.29524) < 1e-3

    def test_result_flux_shape(self):
        """Flux must be non-negative and symmetric (cos-shaped) for a uniform slab."""
        m = one_group_mat()
        cells = 40
        # Single outer vacuum BC; symmetry at i=0 is always enforced.
        solver = nd.KEigenSolver(
            m,
            uniform_map(cells),
            linspace(0.0, 50.0, cells + 1),
            nd.Geometry.Slab,
            [zero_flux()],
            epsilon=1e-8,
        )
        res = solver.solve()
        flux = np.array(res.flux).reshape(cells, 1)[:, 0]
        assert np.all(flux >= 0)
        # Flux peaks at the symmetric centre (cell 0) and decays to the edge
        assert flux[0] > flux[-1]

    def test_result_fields(self):
        m = one_group_mat()
        cells = 10
        solver = nd.KEigenSolver(
            m,
            uniform_map(cells),
            linspace(0.0, 50.0, cells + 1),
            nd.Geometry.Slab,
            [zero_flux()],
        )
        res = solver.solve()
        assert len(res.flux) == cells * m.n_groups
        assert res.iterations > 0
        assert res.residual >= 0.0


# ---------------------------------------------------------------------------
# 1-group cylinder
# ---------------------------------------------------------------------------


class TestOneGroupCylinder:
    def test_one_material(self):
        m = one_group_mat()
        cells = 20
        solver = nd.KEigenSolver(
            m,
            uniform_map(cells),
            linspace(0.0, 76.5535, cells + 1),
            nd.Geometry.Cylinder,
            [zero_flux()],
            epsilon=1e-8,
        )
        res = solver.solve()
        assert abs(res.keff - 1.00001244) < 1e-4

    def test_two_materials(self):
        m = nd.Materials()
        m.n_mat = 2
        m.n_groups = 1
        m.D = [5.0, 1.0]
        m.removal = [0.5, 0.01]
        m.scatter = [0.0, 0.0]
        m.chi = [1.0, 1.0]
        m.nusigf = [0.7, 0.0]
        cells = 100
        solver = nd.KEigenSolver(
            m,
            two_mat_map(cells, 50),
            linspace(0.0, 10.0, cells + 1),
            nd.Geometry.Cylinder,
            [zero_flux()],
            epsilon=1e-8,
        )
        res = solver.solve()
        assert abs(res.keff - 1.14068) < 1e-3


# ---------------------------------------------------------------------------
# 1-group sphere
# ---------------------------------------------------------------------------


class TestOneGroupSphere:
    def test_one_material(self):
        m = one_group_mat()
        cells = 20
        solver = nd.KEigenSolver(
            m,
            uniform_map(cells),
            linspace(0.0, 100.0, cells + 1),
            nd.Geometry.Sphere,
            [zero_flux()],
            epsilon=1e-8,
        )
        res = solver.solve()
        assert abs(res.keff - 1.00001244) < 1e-4

    def test_two_materials(self):
        m = nd.Materials()
        m.n_mat = 2
        m.n_groups = 1
        m.D = [5.0, 1.0]
        m.removal = [0.5, 0.01]
        m.scatter = [0.0, 0.0]
        m.chi = [1.0, 1.0]
        m.nusigf = [0.7, 0.0]
        cells = 150
        solver = nd.KEigenSolver(
            m,
            two_mat_map(cells, 75),
            linspace(0.0, 10.0, cells + 1),
            nd.Geometry.Sphere,
            [zero_flux()],
            epsilon=1e-8,
        )
        res = solver.solve()
        assert abs(res.keff - 0.95735) < 2e-3


# ---------------------------------------------------------------------------
# 2-group sphere
# ---------------------------------------------------------------------------


class TestTwoGroupSphere:
    def test_no_scatter(self):
        m = nd.Materials()
        m.n_mat = 1
        m.n_groups = 2
        m.D = [3.850204978408833, 3.850204978408833]
        m.removal = [0.1532, 0.1532]
        m.scatter = [0.0, 0.0, 0.0, 0.0]
        m.chi = [1.0, 0.0]
        m.nusigf = [0.1570, 0.1570]
        cells = 20
        solver = nd.KEigenSolver(
            m,
            uniform_map(cells),
            linspace(0.0, 100.0, cells + 1),
            nd.Geometry.Sphere,
            [zero_flux(), zero_flux()],
            epsilon=1e-8,
        )
        res = solver.solve()
        assert abs(res.keff - 1.00002955) < 1e-4

    def test_downscatter(self):
        m = nd.Materials()
        m.n_mat = 1
        m.n_groups = 2
        m.D = [0.1, 0.1]
        m.removal = [0.0362, 0.121]
        m.scatter = [0.0, 0.0, 0.0241, 0.0]
        m.chi = [1.0, 0.0]
        m.nusigf = [0.0085, 0.185]
        cells = 50
        solver = nd.KEigenSolver(
            m,
            uniform_map(cells),
            linspace(0.0, 5.0, cells + 1),
            nd.Geometry.Sphere,
            [reflective(), reflective()],
            epsilon=1e-8,
        )
        res = solver.solve()
        assert abs(res.keff - 1.25268252) < 1e-4

    def test_two_materials(self):
        m = nd.Materials()
        m.n_mat = 2
        m.n_groups = 2
        m.D = [1.0, 1.0, 1.0, 1.0]
        m.removal = [0.01, 0.01, 0.01, 0.00049]
        m.scatter = [0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.009, 0.0]
        m.chi = [1.0, 0.0, 1.0, 0.0]
        m.nusigf = [0.00085, 0.057, 0.0, 0.0]
        bc = marshak(1.0)
        cells = 100
        solver = nd.KEigenSolver(
            m,
            two_mat_map(cells, 50),
            linspace(0.0, 100.0, cells + 1),
            nd.Geometry.Sphere,
            [bc, bc],
            epsilon=1e-8,
        )
        res = solver.solve()
        assert abs(res.keff - 1.06508499) < 1e-5

    def test_subcritical(self):
        m = nd.Materials()
        m.n_mat = 1
        m.n_groups = 2
        m.D = [1.0, 1.0]
        m.removal = [0.01, 0.01]
        m.scatter = [0.0, 0.0, 0.001, 0.0]
        m.chi = [1.0, 0.0]
        m.nusigf = [0.00085, 0.057]
        bc = marshak(1.0)
        cells = 100
        solver = nd.KEigenSolver(
            m,
            uniform_map(cells),
            linspace(0.0, 50.0, cells + 1),
            nd.Geometry.Sphere,
            [bc, bc],
            epsilon=1e-8,
        )
        res = solver.solve()
        assert abs(res.keff - 0.36870290) < 1e-5

    def test_flux_layout(self):
        """Result flux is [cells * n_groups] row-major: flux[i*G + g]."""
        m = nd.Materials()
        m.n_mat = 1
        m.n_groups = 2
        m.D = [0.1, 0.1]
        m.removal = [0.0362, 0.121]
        m.scatter = [0.0, 0.0, 0.0241, 0.0]
        m.chi = [1.0, 0.0]
        m.nusigf = [0.0085, 0.185]
        cells = 20
        solver = nd.KEigenSolver(
            m,
            uniform_map(cells),
            linspace(0.0, 5.0, cells + 1),
            nd.Geometry.Sphere,
            [reflective(), reflective()],
        )
        res = solver.solve()
        assert len(res.flux) == cells * 2
        flux = np.array(res.flux).reshape(cells, 2)
        assert np.all(flux >= 0.0)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestKEigenSolverErrors:
    def test_bc_count_mismatch(self):
        """bc list length must equal n_groups."""
        m = one_group_mat()
        with pytest.raises(Exception):
            nd.KEigenSolver(
                m,
                uniform_map(10),
                linspace(0.0, 50.0, 11),
                nd.Geometry.Slab,
                [zero_flux(), zero_flux()],  # 2 BCs for 1 group
            )
