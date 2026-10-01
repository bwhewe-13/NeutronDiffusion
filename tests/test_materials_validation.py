"""Materials array-size validation and result convergence flags.

Every solver constructor must reject mis-sized Materials arrays with a clear
error instead of silently reading out of bounds.  The nastiest historical trap:
chi all zeros used to switch on fission-matrix mode regardless of nusigf's
size, so a non-fissile problem with a vector nusigf indexed it as a G x G
matrix (out-of-bounds read).
"""

import numpy as np
import pytest

import ndiffusion as nd


def one_group_materials():
    m = nd.Materials()
    m.n_mat = 1
    m.n_groups = 1
    m.D = [3.850204978408833]
    m.removal = [0.1532]
    m.scatter = [0.0]
    m.chi = [1.0]
    m.nusigf = [0.1570]
    return m


def two_group_materials():
    m = nd.Materials()
    m.n_mat = 1
    m.n_groups = 2
    m.D = [1.4, 0.4]
    m.removal = [0.03, 0.1]
    m.scatter = [0.0, 0.0, 0.02, 0.0]  # [g_to][g_from], diagonal zeroed
    m.chi = [1.0, 0.0]
    m.nusigf = [0.005, 0.12]
    return m


def slab_solver(m, **kwargs):
    cells = 10
    return nd.KEigenSolver(
        mats=m,
        medium_map=[0] * cells,
        edges_x=list(np.linspace(0.0, 10.0, cells + 1)),
        geom=nd.Geometry.Slab,
        bc=[nd.BoundaryCondition(A=1.0, B=0.0)] * m.n_groups,
        **kwargs,
    )


class TestMaterialsValidation:
    @pytest.mark.parametrize("field", ["D", "removal", "chi"])
    def test_short_vector_array_raises(self, field):
        m = two_group_materials()
        setattr(m, field, [1.0])  # needs n_mat * n_groups = 2
        with pytest.raises(ValueError, match=field):
            slab_solver(m)

    def test_short_scatter_raises(self):
        m = two_group_materials()
        m.scatter = [0.0, 0.0]  # needs n_mat * G * G = 4
        with pytest.raises(ValueError, match="scatter"):
            slab_solver(m)

    def test_bad_nusigf_size_raises(self):
        m = two_group_materials()
        m.nusigf = [0.005, 0.12, 0.0]  # neither G nor G*G
        with pytest.raises(ValueError, match="nusigf"):
            slab_solver(m)

    def test_matrix_nusigf_with_nonzero_chi_raises(self):
        m = two_group_materials()
        m.nusigf = [0.005, 0.0, 0.12, 0.0]  # matrix-sized but chi != 0
        with pytest.raises(ValueError, match="chi"):
            slab_solver(m)

    def test_zero_chi_vector_nusigf_is_standard_mode(self):
        # The historical out-of-bounds trap: chi == 0 with a vector nusigf
        # must NOT engage fission-matrix mode.  A non-fissile time-dependent
        # problem exercises accumulate_fission every step.
        m = two_group_materials()
        m.chi = [0.0, 0.0]
        m.nusigf = [0.0, 0.0]
        m.velocity = [1e7, 2e5]
        cells = 10
        solver = nd.TimeDependentSolver(
            mats=m,
            medium_map=[0] * cells,
            edges_x=list(np.linspace(0.0, 10.0, cells + 1)),
            geom=nd.Geometry.Slab,
            bc=[nd.BoundaryCondition(A=1.0, B=0.0)] * 2,
            initial_flux=[1.0] * (cells * 2),
        )
        res = solver.run(dt=1e-6, n_steps=3)
        assert res.steps == 3
        assert np.all(np.isfinite(res.flux))

    def test_fission_matrix_mode_still_works(self):
        m = two_group_materials()
        m.chi = [0.0, 0.0]
        # F[g_to][g_from] equivalent to chi = [1, 0] with the vector nusigf.
        m.nusigf = [0.005, 0.12, 0.0, 0.0]
        res = slab_solver(m).solve()
        assert res.keff > 0.0

    def test_2d_structured_validates(self):
        m = two_group_materials()
        m.D = [1.4]
        with pytest.raises(ValueError, match="D"):
            nd.KEigenSolver2D(
                mats=m,
                medium_map=[0] * 9,
                edges_x=list(np.linspace(0.0, 3.0, 4)),
                edges_y=list(np.linspace(0.0, 3.0, 4)),
                geom=nd.Geometry2D.XY,
                bc_x=[nd.BoundaryCondition(A=1.0, B=0.0)] * 2,
                bc_y=[nd.BoundaryCondition(A=1.0, B=0.0)] * 2,
            )

    def test_2d_unstructured_validates(self):
        m = two_group_materials()
        m.removal = [0.03]
        mesh = nd.UnstructuredMesh2D()
        mesh.vx = [0.0, 1.0, 0.0]
        mesh.vy = [0.0, 0.0, 1.0]
        mesh.cell_vertices = [0, 1, 2]
        mesh.cell_offsets = [0, 3]
        mesh.material_id = [0]
        with pytest.raises(ValueError, match="removal"):
            nd.KEigenSolverUnstructured2D(
                mats=m,
                mesh=mesh,
                bc=[nd.BoundaryCondition(A=1.0, B=0.0)] * 2,
            )


class TestConvergedFlag:
    def test_k_eigen_converged(self):
        res = slab_solver(one_group_materials(), epsilon=1e-8, max_outer=500).solve()
        assert res.converged
        assert res.iterations < 500

    def test_k_eigen_unconverged_when_capped(self):
        res = slab_solver(one_group_materials(), epsilon=1e-12, max_outer=2).solve()
        assert not res.converged

    def test_fixed_source_converged(self):
        m = one_group_materials()
        cells = 10
        solver = nd.FixedSourceSolver(
            mats=m,
            medium_map=[0] * cells,
            edges_x=list(np.linspace(0.0, 10.0, cells + 1)),
            geom=nd.Geometry.Slab,
            bc=[nd.BoundaryCondition(A=1.0, B=0.0)],
            epsilon=1e-10,
            max_inner=1000,
        )
        res = solver.solve([1.0] * cells)
        assert res.converged
        assert res.residual < 1e-10


# ---------------------------------------------------------------------------
# Unstructured boundary conditions
#
# `bc` is indexed bc[tag * n_groups + g], so its length implies the tag count.
# A tag the array does not reach contributes nothing to the diagonal, which is
# indistinguishable from a reflective boundary.
# ---------------------------------------------------------------------------


def unit_quad_mesh(nx=4, ny=4, size=10.0, tag_of_side=None):
    """nx x ny quad mesh on [0,size]^2.  tag_of_side(side) -> bc tag, default 0."""
    if tag_of_side is None:
        def tag_of_side(_side):
            return 0
    dx, dy = size / nx, size / ny
    vx, vy = [], []
    for i in range(nx + 1):
        for j in range(ny + 1):
            vx.append(i * dx); vy.append(j * dy)

    def vid(i, j):
        return i * (ny + 1) + j

    cv, co, mid = [], [0], []
    for i in range(nx):
        for j in range(ny):
            cv += [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
            co.append(len(cv)); mid.append(0)

    bv0, bv1, bt = [], [], []
    for i in range(nx):
        bv0.append(vid(i, 0));  bv1.append(vid(i + 1, 0));  bt.append(tag_of_side("bottom"))
        bv0.append(vid(i, ny)); bv1.append(vid(i + 1, ny)); bt.append(tag_of_side("top"))
    for j in range(ny):
        bv0.append(vid(0, j));  bv1.append(vid(0, j + 1));  bt.append(tag_of_side("left"))
        bv0.append(vid(nx, j)); bv1.append(vid(nx, j + 1)); bt.append(tag_of_side("right"))

    mesh = nd.UnstructuredMesh2D()
    mesh.vx = vx; mesh.vy = vy
    mesh.cell_vertices = cv; mesh.cell_offsets = co; mesh.material_id = mid
    mesh.bface_v0 = bv0; mesh.bface_v1 = bv1; mesh.bface_bc_tag = bt
    return mesh


VACUUM_2G = [nd.BoundaryCondition(A=0.25, B=0.7), nd.BoundaryCondition(A=0.25, B=0.2)]


class TestUnstructuredInput:
    """bc must cover n_bc_types * n_groups and every tag the mesh uses; the mesh
    itself must be manifold."""

    @pytest.mark.parametrize(
        "bc,match",
        [
            ([], "must not be empty"),
            ([nd.BoundaryCondition(A=1.0, B=0.0)], "not a multiple of n_groups"),
            ([nd.BoundaryCondition(A=1.0, B=0.0)] * 3, "not a multiple of n_groups"),
        ],
    )
    def test_bad_bc_length_raises(self, bc, match):
        with pytest.raises(ValueError, match=match):
            nd.KEigenSolverUnstructured2D(
                mats=two_group_materials(), mesh=unit_quad_mesh(), bc=bc
            )

    def test_tag_beyond_bc_raises(self):
        sides = {"bottom": 0, "top": 1, "left": 2, "right": 3}
        mesh = unit_quad_mesh(tag_of_side=sides.__getitem__)
        with pytest.raises(ValueError, match="boundary tag 3"):
            nd.KEigenSolverUnstructured2D(
                mats=two_group_materials(), mesh=mesh, bc=VACUUM_2G  # only tag 0
            )

    def test_negative_tag_raises(self):
        mesh = unit_quad_mesh(tag_of_side=lambda s: -1 if s == "top" else 0)
        with pytest.raises(ValueError, match="negative boundary tag"):
            nd.KEigenSolverUnstructured2D(
                mats=two_group_materials(), mesh=mesh, bc=VACUUM_2G
            )

    def test_three_cell_edge_raises(self):
        """A non-manifold edge is rejected, not treated as a boundary face."""
        mesh = unit_quad_mesh(nx=2, ny=1)
        cv = list(mesh.cell_vertices)
        cv += cv[0:4]              # duplicate cell 0 -> its edges are seen a third time
        mesh.cell_vertices = cv
        mesh.cell_offsets = list(mesh.cell_offsets) + [len(cv)]
        mesh.material_id = [0, 0, 0]
        with pytest.raises(ValueError, match="more than two cells"):
            nd.KEigenSolverUnstructured2D(
                mats=two_group_materials(), mesh=mesh, bc=VACUUM_2G
            )

    def test_all_solvers_validate(self):
        m = two_group_materials()
        m.velocity = [2.2e7, 2.2e5]
        mesh = unit_quad_mesh()
        with pytest.raises(ValueError, match="must not be empty"):
            nd.KEigenSolverUnstructured2D(mats=m, mesh=mesh, bc=[])
        with pytest.raises(ValueError, match="must not be empty"):
            nd.FixedSourceSolverUnstructured2D(mats=m, mesh=mesh, bc=[])
        with pytest.raises(ValueError, match="must not be empty"):
            nd.TimeDependentSolverUnstructured2D(mats=m, mesh=mesh, bc=[])

    def test_correct_bc_accepted(self):
        res = nd.KEigenSolverUnstructured2D(
            mats=two_group_materials(), mesh=unit_quad_mesh(), bc=VACUUM_2G
        ).solve()
        assert res.keff > 0.0

    def test_vacuum_leaks(self):
        """A vacuum bc must leak, i.e. give a lower keff than a reflective one."""
        mesh = unit_quad_mesh()
        m = two_group_materials()
        k_vac = nd.KEigenSolverUnstructured2D(mats=m, mesh=mesh, bc=VACUUM_2G).solve().keff
        reflective = [nd.BoundaryCondition(A=0.0, B=1.0)] * 2
        k_ref = nd.KEigenSolverUnstructured2D(mats=m, mesh=mesh, bc=reflective).solve().keff
        assert k_vac < k_ref


class TestTimeStepValidation:
    """dt must be positive and finite: 1/(v*dt) goes into the diagonal."""

    def _materials(self):
        m = one_group_materials()
        m.velocity = [2.2e5]
        return m

    def _solvers(self):
        m = self._materials()
        cells = 10
        edges = list(np.linspace(0.0, 10.0, cells + 1))
        bc = [nd.BoundaryCondition(A=1.0, B=0.0)]
        yield nd.TimeDependentSolver(
            mats=m, medium_map=[0] * cells, edges_x=edges,
            geom=nd.Geometry.Slab, bc=bc, initial_flux=[1.0] * cells,
        )
        yield nd.TimeDependentSolver2D(
            mats=m, medium_map=[0] * 9,
            edges_x=list(np.linspace(0.0, 3.0, 4)),
            edges_y=list(np.linspace(0.0, 3.0, 4)),
            geom=nd.Geometry2D.XY, bc_x=bc, bc_y=bc,
            initial_flux=[1.0] * 9,
        )
        mesh = unit_quad_mesh(nx=3, ny=3)
        yield nd.TimeDependentSolverUnstructured2D(
            mats=m, mesh=mesh, bc=bc, initial_flux=[1.0] * 9,
        )

    @pytest.mark.parametrize("dt", [0.0, -1e-3, float("inf"), float("nan")])
    def test_non_positive_dt_raises(self, dt):
        for solver in self._solvers():
            with pytest.raises(ValueError, match="dt must be a positive"):
                solver.step(dt)
            with pytest.raises(ValueError, match="dt must be a positive"):
                solver.run(dt, 3)

    def test_rejected_step_keeps_state(self):
        for solver in self._solvers():
            before = list(solver.result().flux)
            with pytest.raises(ValueError):
                solver.step(0.0)
            assert list(solver.result().flux) == before
            assert solver.time == 0.0
            assert solver.steps == 0

    def test_valid_dt_advances(self):
        for solver in self._solvers():
            solver.step(1e-4)
            assert solver.steps == 1
            assert solver.time == pytest.approx(1e-4)
            assert np.all(np.isfinite(np.array(solver.result().flux)))
