"""
Pytest suite for FixedSourceSolver (fixed-source problems).

Verification strategy
---------------------
1. 1-group slab with uniform source: compare to analytic solution
   phi(x) = q/sig_a * (1 - cosh(x/L) / cosh(R/L))  where L = sqrt(D/sig_a).
2. 2-group slab: verify fast→thermal scatter drives thermal flux when the
   source is fast-only, and that groups decouple when scatter is zero.
3. Geometry suite: Slab, Cylinder, Sphere all produce positive flux.
4. Error handling: wrong source length and BC count mismatch.
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


def zero_flux():
    return nd.BoundaryCondition(A=1.0, B=0.0)


def reflective():
    return nd.BoundaryCondition(A=0.0, B=1.0)


def one_group_absorber(D=1.0, sig_a=0.1):
    """1-group, 1-material, no fission, no scatter."""
    m = nd.Materials()
    m.n_mat = 1
    m.n_groups = 1
    m.D = [D]
    m.removal = [sig_a]
    m.scatter = [0.0]
    m.chi = [0.0]
    m.nusigf = [0.0]
    return m


# ---------------------------------------------------------------------------
# 1-group slab
# ---------------------------------------------------------------------------


class TestOneGroupSlab:
    """Analytic reference: uniform source q in a slab of half-width R.

    Continuous solution (symmetry at x=0, phi=0 at x=R):
        phi(x) = q/sig_a * (1 - cosh(x/L) / cosh(R/L))
    where L = sqrt(D/sig_a).
    """

    D = 1.0
    sig_a = 0.1
    q = 1.0
    R = 10.0
    cells = 200

    @property
    def edges(self):
        return linspace(0.0, self.R, self.cells + 1)

    @property
    def cell_centers(self):
        e = self.edges
        return [(e[i] + e[i + 1]) / 2.0 for i in range(self.cells)]

    def analytic(self, x):
        L = np.sqrt(self.D / self.sig_a)
        return self.q / self.sig_a * (1.0 - np.cosh(x / L) / np.cosh(self.R / L))

    def test_matches_analytic(self):
        solver = nd.FixedSourceSolver(
            one_group_absorber(self.D, self.sig_a),
            uniform_map(self.cells),
            self.edges,
            nd.Geometry.Slab,
            [zero_flux()],
            epsilon=1e-10,
        )
        res = solver.solve([self.q] * self.cells)

        numerical = np.array(res.flux)
        analytic = self.analytic(np.array(self.cell_centers))
        rel_err = np.abs(numerical - analytic) / analytic.max()
        assert np.max(rel_err) < 1e-3

    def test_flux_non_negative(self):
        solver = nd.FixedSourceSolver(
            one_group_absorber(),
            uniform_map(self.cells),
            self.edges,
            nd.Geometry.Slab,
            [zero_flux()],
        )
        res = solver.solve([1.0] * self.cells)
        assert np.all(np.array(res.flux) >= 0.0)

    def test_result_fields(self):
        cells = 20
        solver = nd.FixedSourceSolver(
            one_group_absorber(),
            uniform_map(cells),
            linspace(0.0, 10.0, cells + 1),
            nd.Geometry.Slab,
            [zero_flux()],
        )
        res = solver.solve([1.0] * cells)
        assert len(res.flux) == cells
        assert res.iterations > 0
        assert res.residual >= 0.0

    def test_stronger_absorption_gives_lower_flux(self):
        """Doubling sig_a should roughly halve the peak flux."""
        cells = 50
        edges = linspace(0.0, 10.0, cells + 1)

        def solve(sig_a):
            solver = nd.FixedSourceSolver(
                one_group_absorber(sig_a=sig_a),
                uniform_map(cells),
                edges,
                nd.Geometry.Slab,
                [zero_flux()],
                epsilon=1e-10,
            )
            return max(solver.solve([1.0] * cells).flux)

        assert solve(0.1) > solve(0.2)


# ---------------------------------------------------------------------------
# 2-group slab
# ---------------------------------------------------------------------------


class TestTwoGroupSlab:
    def _two_group_mat(self, scatter_01=0.02):
        """Fast (g=0) → thermal (g=1) downscatter material."""
        m = nd.Materials()
        m.n_mat = 1
        m.n_groups = 2
        m.D = [1.0, 0.5]
        m.removal = [0.05, 0.1]
        # scatter[g_to][g_from]: scatter_01 = fast→thermal
        m.scatter = [0.0, 0.0, scatter_01, 0.0]
        m.chi = [0.0, 0.0]
        m.nusigf = [0.0, 0.0]
        return m

    def test_downscatter_drives_thermal_flux(self):
        """Source in fast group only; thermal flux must be non-zero due to scatter."""
        cells = 50
        solver = nd.FixedSourceSolver(
            self._two_group_mat(scatter_01=0.02),
            uniform_map(cells),
            linspace(0.0, 10.0, cells + 1),
            nd.Geometry.Slab,
            [zero_flux(), zero_flux()],
        )
        # Source only in fast group (row-major: [q_fast, q_thermal, ...])
        source = [1.0, 0.0] * cells
        flux = np.array(solver.solve(source).flux).reshape(cells, 2)

        assert np.all(flux[:, 0] > 0)   # fast flux driven by source
        assert np.all(flux[:, 1] > 0)   # thermal flux driven by scatter

    def test_no_scatter_groups_independent(self):
        """Without scatter: source in thermal only → fast flux stays zero."""
        cells = 30
        solver = nd.FixedSourceSolver(
            self._two_group_mat(scatter_01=0.0),
            uniform_map(cells),
            linspace(0.0, 10.0, cells + 1),
            nd.Geometry.Slab,
            [zero_flux(), zero_flux()],
        )
        source = [0.0, 1.0] * cells  # thermal source only
        flux = np.array(solver.solve(source).flux).reshape(cells, 2)

        assert np.all(np.abs(flux[:, 0]) < 1e-12)  # no fast flux
        assert np.all(flux[:, 1] > 0)               # thermal flux from source

    def test_flux_layout(self):
        """Result is [cells * n_groups], row-major: flux[i*G + g]."""
        cells = 20
        solver = nd.FixedSourceSolver(
            self._two_group_mat(),
            uniform_map(cells),
            linspace(0.0, 10.0, cells + 1),
            nd.Geometry.Slab,
            [zero_flux(), zero_flux()],
        )
        res = solver.solve([1.0, 0.0] * cells)
        assert len(res.flux) == cells * 2


# ---------------------------------------------------------------------------
# Geometry suite
# ---------------------------------------------------------------------------


class TestGeometries:
    """Solver runs and returns positive flux for all three 1-D geometries."""

    def _make_solver(self, geom, cells=30):
        return nd.FixedSourceSolver(
            one_group_absorber(),
            uniform_map(cells),
            linspace(0.0, 10.0, cells + 1),
            geom,
            [zero_flux()],
        )

    def test_slab(self):
        res = self._make_solver(nd.Geometry.Slab).solve([1.0] * 30)
        assert np.all(np.array(res.flux) > 0)

    def test_cylinder(self):
        res = self._make_solver(nd.Geometry.Cylinder).solve([1.0] * 30)
        assert np.all(np.array(res.flux) > 0)

    def test_sphere(self):
        res = self._make_solver(nd.Geometry.Sphere).solve([1.0] * 30)
        assert np.all(np.array(res.flux) > 0)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestFixedSourceErrors:
    def test_wrong_source_length(self):
        """solve() must raise when source has the wrong number of elements."""
        cells = 20
        solver = nd.FixedSourceSolver(
            one_group_absorber(),
            uniform_map(cells),
            linspace(0.0, 10.0, cells + 1),
            nd.Geometry.Slab,
            [zero_flux()],
        )
        with pytest.raises(Exception):
            solver.solve([1.0] * 5)  # wrong: should be 20

    def test_bc_count_mismatch(self):
        """Constructor must raise when bc length != n_groups."""
        with pytest.raises(Exception):
            nd.FixedSourceSolver(
                one_group_absorber(),
                uniform_map(10),
                linspace(0.0, 10.0, 11),
                nd.Geometry.Slab,
                [zero_flux(), zero_flux()],  # 2 BCs for 1 group
            )
