"""Convergence warnings reach Python's warnings module.

The C++ core reports through a hook that the bindings point at PyErr_WarnEx, so
an iteration cap shows up as ndiffusion.ConvergenceWarning - catchable,
filterable, and raised as an exception under an "error" filter.  pyproject.toml
already turns it into an error for the whole suite, so every test here sets its
own filter.
"""

import warnings

import numpy as np
import pytest

import ndiffusion as nd

CELLS = 10
EDGES = np.linspace(0.0, 20.0, CELLS + 1)
ZERO = nd.BoundaryCondition(A=1.0, B=0.0)


def two_group():
    m = nd.Materials()
    m.n_mat = 1
    m.n_groups = 2
    m.D = [1.4, 0.4]
    m.removal = [0.03, 0.1]
    # Some upscatter, so the 1-D group sweep is not exact after two passes.
    m.scatter = [0.0, 0.01, 0.02, 0.0]
    m.chi = [1.0, 0.0]
    m.nusigf = [0.005, 0.12]
    m.velocity = [2.0e7, 2.2e5]
    return m


def quad_mesh(n=3, size=6.0):
    h = size / n
    vx, vy, cv, co = [], [], [], [0]
    for i in range(n + 1):
        for j in range(n + 1):
            vx.append(i * h)
            vy.append(j * h)
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            cv += [a, a + n + 1, a + n + 2, a + 1]
            co.append(len(cv))
    mesh = nd.UnstructuredMesh2D()
    mesh.vx, mesh.vy, mesh.cell_vertices, mesh.cell_offsets = vx, vy, cv, co
    mesh.material_id = [0] * (n * n)
    return mesh


def transient_1d(**kw):
    return nd.TimeDependentSolver(two_group(), [0] * CELLS, EDGES, nd.Geometry.Slab,
                                  [ZERO, ZERO], initial_flux=np.ones((CELLS, 2)), **kw)


def transient_2d(**kw):
    return nd.TimeDependentSolver2D(two_group(), [0] * 9, EDGES[:4], EDGES[:4],
                                    nd.Geometry2D.XY, [ZERO, ZERO], [ZERO, ZERO],
                                    initial_flux=np.ones((9, 2)), **kw)


def transient_unstructured(**kw):
    return nd.TimeDependentSolverUnstructured2D(two_group(), quad_mesh(), [ZERO, ZERO],
                                                initial_flux=np.ones((9, 2)), **kw)


TRANSIENTS = [transient_1d, transient_2d, transient_unstructured]
UNCONVERGED = {"epsilon": 1e-15, "max_inner": 1}


@pytest.fixture(autouse=True)
def default_filters():
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        yield


def test_category():
    assert issubclass(nd.ConvergenceWarning, UserWarning)
    assert nd.ConvergenceWarning.__module__ == "ndiffusion"
    assert "ConvergenceWarning" in nd.__all__


class TestKEigenvalue:
    def test_inner_cap(self):
        solver = nd.KEigenSolver(two_group(), [0] * CELLS, EDGES, nd.Geometry.Slab,
                                 [ZERO, ZERO], epsilon=1e-12, max_inner=1)
        with pytest.warns(nd.ConvergenceWarning, match="inner solve"):
            res = solver.solve()
        assert not res.converged

    def test_outer_cap(self):
        solver = nd.KEigenSolver2D(two_group(), [0] * 9, EDGES[:4], EDGES[:4],
                                   nd.Geometry2D.XY, [ZERO, ZERO], [ZERO, ZERO],
                                   epsilon=1e-12, max_outer=3)
        with pytest.warns(nd.ConvergenceWarning, match="max_outer=3"):
            res = solver.solve()
        assert not res.converged

    def test_converged_is_silent(self):
        solver = nd.KEigenSolverUnstructured2D(two_group(), quad_mesh(), [ZERO, ZERO])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert solver.solve().converged


class TestFixedSource:
    @pytest.mark.parametrize("make", [
        lambda: nd.FixedSourceSolver(two_group(), [0] * CELLS, EDGES, nd.Geometry.Slab,
                                     [ZERO, ZERO], epsilon=1e-15, max_inner=2),
        lambda: nd.FixedSourceSolver2D(two_group(), [0] * 9, EDGES[:4], EDGES[:4],
                                       nd.Geometry2D.XY, [ZERO, ZERO], [ZERO, ZERO],
                                       epsilon=1e-15, max_inner=2),
        lambda: nd.FixedSourceSolverUnstructured2D(two_group(), quad_mesh(), [ZERO, ZERO],
                                                   epsilon=1e-15, max_inner=2),
    ], ids=["1d", "2d", "unstructured"])
    def test_cap(self, make):
        solver = make()
        with pytest.warns(nd.ConvergenceWarning, match="max_inner=2"):
            res = solver.solve(np.ones((solver.n_cells, 2)))
        assert not res.converged


class TestTimeDependent:
    @pytest.mark.parametrize("make", TRANSIENTS)
    def test_warns_once_per_solver(self, make):
        solver = make(**UNCONVERGED)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = solver.run(1e-3, 4)
        assert [w.category for w in caught] == [nd.ConvergenceWarning]
        assert "time step" in str(caught[0].message)
        assert not res.converged

        # A new solver has its own budget of one.
        with pytest.warns(nd.ConvergenceWarning):
            make(**UNCONVERGED).step(1e-3)

    @pytest.mark.parametrize("make", TRANSIENTS)
    def test_converged_flag(self, make):
        solver = make(epsilon=1e-8, max_inner=500)
        assert solver.result().converged
        assert solver.run(1e-4, 3).converged

    @pytest.mark.parametrize("make", TRANSIENTS)
    def test_error_filter_leaves_a_complete_step(self, make):
        solver = make(**UNCONVERGED)
        with warnings.catch_warnings():
            warnings.simplefilter("error", nd.ConvergenceWarning)
            with pytest.raises(nd.ConvergenceWarning):
                solver.run(1e-3, 5)
            # The warning fires after the step is finished, so the solver
            # stopped cleanly after one step and keeps going (warned once).
            assert solver.steps == 1
            assert solver.time == pytest.approx(1e-3)
            res = solver.run(1e-3, 2)
        assert res.steps == 3
        assert not res.converged
        assert np.all(np.isfinite(res.flux))


class TestFilters:
    def test_error_filter_raises_from_solve(self):
        solver = nd.FixedSourceSolver(two_group(), [0] * CELLS, EDGES, nd.Geometry.Slab,
                                      [ZERO, ZERO], epsilon=1e-15, max_inner=2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(nd.ConvergenceWarning, match="FixedSourceSolver"):
                solver.solve(np.ones((CELLS, 2)))

    def test_ignore_filter_is_silent(self, capfd):
        solver = transient_1d(**UNCONVERGED)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("ignore", nd.ConvergenceWarning)
            solver.run(1e-3, 2)
        assert caught == []
        # Nothing falls back to stderr while a hook is installed.
        assert capfd.readouterr().err == ""

    def test_attributed_to_the_calling_line(self):
        solver = transient_1d(**UNCONVERGED)
        with pytest.warns(nd.ConvergenceWarning) as caught:
            solver.step(1e-3)
        assert caught[0].filename == __file__
