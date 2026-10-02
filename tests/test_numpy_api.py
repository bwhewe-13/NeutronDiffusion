"""numpy arrays in and out of the compiled solvers.

Inputs accept any array-like, flattened in C order, so an (n_cells, n_groups)
array lines up with the row-major [cell * n_groups + g] layout.  Results come
back as (n_cells, n_groups) arrays.  Per-cell inputs refuse any other 2-D shape:
a transposed array has the right size and would otherwise scramble silently.
"""

import numpy as np
import pytest

import ndiffusion as nd

CELLS = 12
EDGES = np.linspace(0.0, 12.0, CELLS + 1)
VACUUM = nd.BoundaryCondition(A=1.0, B=0.0)
REFLECTIVE = nd.BoundaryCondition(A=0.0, B=1.0)


def two_group():
    m = nd.Materials()
    m.n_mat = 1
    m.n_groups = 2
    m.D = [1.4, 0.4]
    m.removal = [0.03, 0.1]
    m.scatter = [0.0, 0.0, 0.02, 0.0]
    m.chi = [1.0, 0.0]
    m.nusigf = [0.005, 0.12]
    m.velocity = [2.0e7, 2.2e5]
    return m


def fixed_source():
    return nd.FixedSourceSolver(two_group(), [0] * CELLS, EDGES, nd.Geometry.Slab,
                                [VACUUM, VACUUM], epsilon=1e-12, max_inner=2000)


def transient(**kwargs):
    return nd.TimeDependentSolver(two_group(), [0] * CELLS, EDGES, nd.Geometry.Slab,
                                  [VACUUM, VACUUM], **kwargs)


def quad_mesh(n=4, size=4.0):
    h = size / n
    vx, vy, cv, co = [], [], [], [0]
    for i in range(n + 1):
        for j in range(n + 1):
            vx.append(i * h); vy.append(j * h)
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            cv += [a, a + n + 1, a + n + 2, a + 1]; co.append(len(cv))
    mesh = nd.UnstructuredMesh2D()
    mesh.vx, mesh.vy, mesh.cell_vertices, mesh.cell_offsets = vx, vy, cv, co
    mesh.material_id = [0] * (n * n)
    return mesh


class TestResults:
    def test_k_eigenvalue_flux_shape(self):
        res = nd.KEigenSolver(two_group(), [0] * CELLS, EDGES, nd.Geometry.Slab,
                              [VACUUM, VACUUM]).solve()
        assert isinstance(res.flux, np.ndarray)
        assert res.flux.dtype == np.float64
        assert res.flux.shape == (CELLS, 2)
        assert res.n_groups == 2

    def test_structured_2d_rows_are_cells(self):
        nx, ny = 4, 3
        solver = nd.KEigenSolver2D(two_group(), [0] * (nx * ny), np.arange(nx + 1.0),
                                   np.arange(ny + 1.0), nd.Geometry2D.XY,
                                   [VACUUM, VACUUM], [VACUUM, VACUUM])
        res = solver.solve()
        assert res.flux.shape == (nx * ny, 2)
        assert (solver.n_cells, solver.n_groups) == (nx * ny, 2)
        # Reflective at x = 0 and y = 0, vacuum at the far edges: the flux falls
        # off monotonically along both axes only if row i * ny + j is cell (i, j).
        grid = res.flux.reshape(nx, ny, 2)[..., 0]
        assert np.all(np.diff(grid, axis=0) < 0.0)
        assert np.all(np.diff(grid, axis=1) < 0.0)

    def test_unstructured_flux_shape(self):
        mesh = quad_mesh()
        res = nd.KEigenSolverUnstructured2D(two_group(), mesh, [VACUUM, VACUUM]).solve()
        assert res.flux.shape == (16, 2)

    def test_precursors_shape(self):
        delayed = nd.make_delayed_data(nd.DELAYED_U235_6GROUP, G=2, n_mat=1, chi=[1.0, 0.0])
        solver = transient(initial_flux=np.ones((CELLS, 2)), delayed=delayed)
        assert solver.precursors.shape == (CELLS, 6)
        res = solver.run(1e-4, 2)
        assert res.flux.shape == (CELLS, 2)
        assert res.precursors.shape == (CELLS, 6)
        assert res.n_precursor == 6

    def test_prompt_only_precursors_have_no_columns(self):
        solver = transient()
        assert solver.precursors.shape == (CELLS, 0)
        assert solver.result().precursors.shape == (CELLS, 0)

    def test_result_is_a_copy(self):
        res = fixed_source().solve(np.ones((CELLS, 2)))
        flux = res.flux
        flux[:] = 0.0
        assert np.all(res.flux > 0.0)


class TestInputs:
    def test_list_and_array_sources_agree(self):
        solver = fixed_source()
        q = np.column_stack((np.linspace(1.0, 2.0, CELLS), np.zeros(CELLS)))
        from_2d = solver.solve(q).flux
        from_flat = solver.solve(q.ravel()).flux
        from_list = solver.solve(q.ravel().tolist()).flux
        assert np.array_equal(from_2d, from_flat)
        assert np.array_equal(from_2d, from_list)

    def test_transposed_source_rejected(self):
        with pytest.raises(ValueError, match=r"source must be flat or shaped \(n_cells"):
            fixed_source().solve(np.ones((2, CELLS)))

    def test_three_dimensional_source_rejected(self):
        with pytest.raises(ValueError, match="source must be flat"):
            fixed_source().solve(np.ones((CELLS, 2, 1)))

    def test_transposed_initial_flux_rejected(self):
        with pytest.raises(ValueError, match="initial_flux must be flat"):
            transient(initial_flux=np.ones((2, CELLS)))

    def test_initial_flux_shapes_agree(self):
        phi = np.random.default_rng(1).uniform(0.5, 1.5, (CELLS, 2))
        a = transient(initial_flux=phi).run(1e-4, 3).flux
        b = transient(initial_flux=phi.ravel().tolist()).run(1e-4, 3).flux
        assert np.array_equal(a, b)

    def test_transposed_initial_precursors_rejected(self):
        delayed = nd.make_delayed_data(nd.DELAYED_U235_6GROUP, G=2, n_mat=1, chi=[1.0, 0.0])
        with pytest.raises(ValueError, match="initial_precursors must be flat"):
            transient(delayed=delayed, initial_precursors=np.ones((6, CELLS)))

    def test_material_indices_must_be_integers(self):
        with pytest.raises(TypeError):
            nd.KEigenSolver(two_group(), [0.5] * CELLS, EDGES, nd.Geometry.Slab,
                            [VACUUM, VACUUM])

    def test_scalar_is_not_an_array(self):
        with pytest.raises(TypeError):
            nd.KEigenSolver(two_group(), [0] * CELLS, 12.0, nd.Geometry.Slab,
                            [VACUUM, VACUUM])

    def test_integer_arrays_of_any_width(self):
        mmap = np.zeros(CELLS, dtype=np.int64)
        res = nd.KEigenSolver(two_group(), mmap, EDGES, nd.Geometry.Slab,
                              [VACUUM, VACUUM]).solve()
        assert res.converged


class TestFields:
    def test_materials_fields_are_arrays(self):
        m = two_group()
        assert isinstance(m.D, np.ndarray)
        assert m.scatter.shape == (4,)

    def test_multidimensional_assignment_is_flattened(self):
        m = two_group()
        m.scatter = np.array([[[0.0, 0.0], [0.02, 0.0]]])   # (n_mat, G, G)
        assert np.array_equal(m.scatter, [0.0, 0.0, 0.02, 0.0])

    def test_mesh_fields_are_integer_arrays(self):
        mesh = quad_mesh()
        assert mesh.cell_offsets.dtype.kind == "i"
        assert mesh.material_id.shape == (16,)

    def test_geometry_queries_return_arrays(self):
        cx, cy = nd.cell_centroids(quad_mesh())
        assert isinstance(cx, np.ndarray) and cx.shape == (16,)
        assert np.isclose(nd.cell_areas(quad_mesh()).sum(), 16.0)
