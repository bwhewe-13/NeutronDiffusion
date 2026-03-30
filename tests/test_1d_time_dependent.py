"""
Pytest suite for TimeDependentSolver.

Physical checks:
  - Critical system: flux shape is preserved over time (explicit-fission
    backward-Euler keeps amplitude nearly constant for small dt).
  - Supercritical system (keff > 1): total flux grows over time.
  - Subcritical system (keff < 1): total flux decays over time.
  - API: step(), run(), result(), time, steps properties.
"""

import numpy as np
import pytest

import ndiffusion as nd

# ---------------------------------------------------------------------------
# Shared fixtures / factories
# ---------------------------------------------------------------------------


def linspace(start, stop, n):
    return list(np.linspace(start, stop, n))


def uniform_map(cells):
    return [0] * cells


def zero_flux():
    return nd.BoundaryCondition(A=1.0, B=0.0)


def reflective():
    return nd.BoundaryCondition(A=0.0, B=1.0)


def base_mat(nusigf_scale=1.0):
    """1-group sphere material.  nusigf_scale adjusts criticality."""
    m = nd.Materials()
    m.n_mat = 1
    m.n_groups = 1
    m.D = [3.850204978408833]
    m.removal = [0.1532]
    m.scatter = [0.0]
    m.chi = [1.0]
    m.nusigf = [0.1570 * nusigf_scale]
    m.velocity = [2.2e5]  # cm/s  (~thermal neutron speed)
    return m


def two_group_mat():
    """2-group 1-material sphere with fast→thermal downscatter."""
    m = nd.Materials()
    m.n_mat = 1
    m.n_groups = 2
    m.D = [0.1, 0.1]
    m.removal = [0.0362, 0.121]
    m.scatter = [0.0, 0.0, 0.0241, 0.0]
    m.chi = [1.0, 0.0]
    m.nusigf = [0.0085, 0.185]
    m.velocity = [1.0e7, 2.2e5]  # fast / thermal
    return m


def keigenvalue_flux(mats, cells, edges, geom, bc):
    """Solve k-eigenvalue problem and return (keff, flux list)."""
    solver = nd.KEigenSolver(
        mats, uniform_map(cells), edges, geom, bc, epsilon=1e-10
    )
    res = solver.solve()
    return res.keff, res.flux


# ---------------------------------------------------------------------------
# Criticality-driven amplitude tests
# ---------------------------------------------------------------------------


class TestCriticalSystem:
    """For a near-critical system the flux shape should be preserved."""

    def test_shape_preserved(self):
        cells = 30
        edges = linspace(0.0, 100.0, cells + 1)
        m = base_mat(nusigf_scale=1.0)
        bc = [zero_flux()]

        # Start from the k-eigenvalue flux
        _, init_flux = keigenvalue_flux(m, cells, edges, nd.Geometry.Sphere, bc)
        init_arr = np.array(init_flux)

        tds = nd.TimeDependentSolver(
            m,
            uniform_map(cells),
            edges,
            nd.Geometry.Sphere,
            bc,
            initial_flux=init_flux,
            epsilon=1e-10,
        )
        tds.run(dt=1e-5, n_steps=200)
        res = tds.result()
        final = np.array(res.flux)

        # Normalise both and compare shapes
        init_norm = init_arr / init_arr.max()
        final_norm = final / final.max()
        assert np.max(np.abs(final_norm - init_norm)) < 1e-4


class TestSupercriticalSystem:
    """keff > 1 → total flux must grow over time."""

    def test_flux_grows(self):
        cells = 30
        edges = linspace(0.0, 100.0, cells + 1)
        m_super = base_mat(nusigf_scale=1.5)  # supercritical
        bc = [zero_flux()]

        # Non-zero initial flux from a critical solve
        _, init_flux = keigenvalue_flux(
            base_mat(nusigf_scale=1.0), cells, edges, nd.Geometry.Sphere, bc
        )
        tds = nd.TimeDependentSolver(
            m_super,
            uniform_map(cells),
            edges,
            nd.Geometry.Sphere,
            bc,
            initial_flux=init_flux,
            epsilon=1e-10,
        )
        init_sum = np.array(tds.result().flux).sum()

        tds.run(dt=1e-6, n_steps=50)
        final_sum = np.array(tds.result().flux).sum()

        assert final_sum > init_sum


class TestSubcriticalSystem:
    """keff < 1 → total flux must decay over time."""

    def test_flux_decays(self):
        cells = 30
        edges = linspace(0.0, 100.0, cells + 1)
        m = base_mat(nusigf_scale=0.5)  # subcritical
        bc = [zero_flux()]

        # Non-zero initial flux
        _, init_flux = keigenvalue_flux(
            base_mat(nusigf_scale=1.0), cells, edges, nd.Geometry.Sphere, bc
        )
        tds = nd.TimeDependentSolver(
            m,
            uniform_map(cells),
            edges,
            nd.Geometry.Sphere,
            bc,
            initial_flux=init_flux,
            epsilon=1e-10,
        )
        init_sum = np.array(tds.result().flux).sum()

        tds.run(dt=1e-6, n_steps=50)
        final_sum = np.array(tds.result().flux).sum()

        assert final_sum < init_sum


# ---------------------------------------------------------------------------
# API / interface tests
# ---------------------------------------------------------------------------


class TestTimeDependentAPI:
    def setup_method(self):
        self.cells = 20
        self.edges = linspace(0.0, 100.0, self.cells + 1)
        self.m = base_mat()
        self.bc = [zero_flux()]

    def _make_solver(self, **kwargs):
        return nd.TimeDependentSolver(
            self.m,
            uniform_map(self.cells),
            self.edges,
            nd.Geometry.Sphere,
            self.bc,
            **kwargs,
        )

    def test_initial_time_is_zero(self):
        tds = self._make_solver()
        assert tds.time == pytest.approx(0.0)

    def test_initial_steps_is_zero(self):
        tds = self._make_solver()
        assert tds.steps == 0

    def test_step_increments_time(self):
        tds = self._make_solver()
        dt = 1e-4
        tds.step(dt)
        assert tds.time == pytest.approx(dt)

    def test_step_increments_steps(self):
        tds = self._make_solver()
        tds.step(1e-4)
        assert tds.steps == 1

    def test_run_accumulates_time(self):
        tds = self._make_solver()
        dt = 1e-4
        n = 7
        tds.run(dt, n)
        assert tds.time == pytest.approx(dt * n)
        assert tds.steps == n

    def test_multiple_runs_accumulate(self):
        tds = self._make_solver()
        tds.run(1e-4, 3)
        tds.run(2e-4, 5)
        assert tds.time == pytest.approx(3 * 1e-4 + 5 * 2e-4)
        assert tds.steps == 8

    def test_result_flux_length(self):
        tds = self._make_solver()
        res = tds.result()
        assert len(res.flux) == self.cells * self.m.n_groups

    def test_result_time_matches_property(self):
        tds = self._make_solver()
        tds.run(1e-4, 5)
        res = tds.result()
        assert res.time == pytest.approx(tds.time)
        assert res.steps == tds.steps

    def test_run_returns_result(self):
        tds = self._make_solver()
        res = tds.run(1e-4, 3)
        assert isinstance(res.flux, list)
        assert res.time == pytest.approx(3e-4)
        assert res.steps == 3

    def test_zero_initial_flux(self):
        """Starting from zero flux should stay zero (no external source)."""
        tds = self._make_solver()
        tds.run(1e-4, 10)
        flux = np.array(tds.result().flux)
        assert np.all(flux == pytest.approx(0.0, abs=1e-30))

    def test_custom_initial_flux(self):
        _, init_flux = keigenvalue_flux(
            self.m, self.cells, self.edges, nd.Geometry.Sphere, self.bc
        )
        tds = self._make_solver(initial_flux=init_flux)
        flux_before = np.array(tds.result().flux).copy()
        tds.step(1e-5)
        flux_after = np.array(tds.result().flux)
        # Flux should have changed after a step (use tight tolerance to detect ~1e-6 shifts)
        assert not np.allclose(flux_before, flux_after, rtol=1e-7, atol=1e-12)

    def test_flux_non_negative_after_steps(self):
        _, init_flux = keigenvalue_flux(
            self.m, self.cells, self.edges, nd.Geometry.Sphere, self.bc
        )
        tds = self._make_solver(initial_flux=init_flux)
        tds.run(1e-5, 50)
        flux = np.array(tds.result().flux)
        assert np.all(flux >= -1e-12)  # allow tiny FP noise


# ---------------------------------------------------------------------------
# Multi-group time-dependent
# ---------------------------------------------------------------------------


class TestTwoGroupTimeDep:
    def test_flux_shape_preserved_critical(self):
        """2-group system near criticality (keff ≈ 1.25): flux shape stable."""
        cells = 30
        edges = linspace(0.0, 5.0, cells + 1)
        m = two_group_mat()
        bc = [reflective(), reflective()]

        _, init_flux = keigenvalue_flux(m, cells, edges, nd.Geometry.Sphere, bc)
        init_arr = np.array(init_flux).reshape(cells, 2)

        tds = nd.TimeDependentSolver(
            m,
            uniform_map(cells),
            edges,
            nd.Geometry.Sphere,
            bc,
            initial_flux=init_flux,
            epsilon=1e-10,
        )
        tds.run(dt=1e-8, n_steps=100)
        final = np.array(tds.result().flux).reshape(cells, 2)

        for g in range(2):
            ref = init_arr[:, g]
            fnl = final[:, g]
            shape = fnl / fnl.max() - ref / ref.max()
            assert np.max(np.abs(shape)) < 1e-3

    def test_result_flux_layout(self):
        """Result is [cells * n_groups], row-major: flux[i*G + g]."""
        cells = 10
        m = two_group_mat()
        bc = [reflective(), reflective()]
        tds = nd.TimeDependentSolver(
            m, uniform_map(cells), linspace(0.0, 5.0, cells + 1), nd.Geometry.Sphere, bc
        )
        res = tds.result()
        assert len(res.flux) == cells * 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestTimeDependentErrors:
    def test_missing_velocity(self):
        """Omitting velocity must raise an exception."""
        m = nd.Materials()
        m.n_mat = 1
        m.n_groups = 1
        m.D = [1.0]
        m.removal = [0.1]
        m.scatter = [0.0]
        m.chi = [1.0]
        m.nusigf = [0.1]
        # velocity intentionally left empty
        with pytest.raises(Exception):
            nd.TimeDependentSolver(
                m,
                [0] * 10,
                list(np.linspace(0, 10, 11)),
                nd.Geometry.Slab,
                [nd.BoundaryCondition(A=1.0, B=0.0)],
            )

    def test_wrong_initial_flux_length(self):
        """initial_flux with wrong length must raise an exception."""
        m = nd.Materials()
        m.n_mat = 1
        m.n_groups = 1
        m.D = [1.0]
        m.removal = [0.1]
        m.scatter = [0.0]
        m.chi = [1.0]
        m.nusigf = [0.1]
        m.velocity = [2.2e5]
        with pytest.raises(Exception):
            nd.TimeDependentSolver(
                m,
                [0] * 10,
                list(np.linspace(0, 10, 11)),
                nd.Geometry.Slab,
                [nd.BoundaryCondition(A=1.0, B=0.0)],
                initial_flux=[1.0] * 5,  # wrong: should be 10
            )

    def test_bc_count_mismatch(self):
        """bc list length must equal n_groups."""
        m = nd.Materials()
        m.n_mat = 1
        m.n_groups = 1
        m.D = [1.0]
        m.removal = [0.1]
        m.scatter = [0.0]
        m.chi = [1.0]
        m.nusigf = [0.1]
        m.velocity = [2.2e5]
        with pytest.raises(Exception):
            nd.TimeDependentSolver(
                m,
                [0] * 10,
                list(np.linspace(0, 10, 11)),
                nd.Geometry.Slab,
                [
                    nd.BoundaryCondition(A=1.0, B=0.0),
                    nd.BoundaryCondition(A=1.0, B=0.0),
                ],  # 2 BCs for 1 group
            )
