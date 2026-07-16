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
