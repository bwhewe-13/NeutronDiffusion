"""
Pytest suite for TimeDependentSolver2D and TimeDependentSolverUnstructured2D.

Verification strategy
---------------------
Structured (TimeDependentSolver2D):
  1. Time advances: step() increments time and step counter correctly.
  2. Flux non-negative: run() returns a non-negative flux after several steps.
  3. Result shape: result() flux has the correct length.

Unstructured (TimeDependentSolverUnstructured2D):
  4. Time advances: step() increments time and step counter correctly.
  5. Run returns correct steps: run() returns result with expected step count
     and a non-negative flux of the correct length.
"""

import numpy as np
import pytest

import ndiffusion as nd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def linspace(a, b, n):
    return list(np.linspace(a, b, n))


def vacuum():
    return nd.BoundaryCondition(A=1.0, B=0.0)


def reflective():
    return nd.BoundaryCondition(A=0.0, B=1.0)


def one_group_mat():
    """Standard 1-group fissile material (same as 1-D test suite)."""
    m = nd.Materials()
    m.n_mat    = 1
    m.n_groups = 1
    m.D        = [3.850204978408833]
    m.removal  = [0.1532]
    m.scatter  = [0.0]
    m.chi      = [1.0]
    m.nusigf   = [0.1570]
    m.velocity = [2.2e5]
    return m


def make_quad_mesh(nx, ny, Lx, Ly, bc_tag=0):
    """Build a regular nx×ny quad mesh on [0,Lx]×[0,Ly] with a uniform BC tag."""
    dx = Lx / nx
    dy = Ly / ny

    vx, vy = [], []
    for i in range(nx + 1):
        for j in range(ny + 1):
            vx.append(i * dx)
            vy.append(j * dy)

    def vid(i, j):
        return i * (ny + 1) + j

    cell_vertices = []
    cell_offsets = [0]
    mat_ids = []
    for i in range(nx):
        for j in range(ny):
            cell_vertices += [
                vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1),
            ]
            cell_offsets.append(len(cell_vertices))
            mat_ids.append(0)

    bface_v0, bface_v1, bface_bc_tag = [], [], []
    for i in range(nx):
        bface_v0.append(vid(i, 0));    bface_v1.append(vid(i + 1, 0));    bface_bc_tag.append(bc_tag)
    for j in range(ny):
        bface_v0.append(vid(nx, j));   bface_v1.append(vid(nx, j + 1));   bface_bc_tag.append(bc_tag)
    for i in range(nx - 1, -1, -1):
        bface_v0.append(vid(i + 1, ny)); bface_v1.append(vid(i, ny));     bface_bc_tag.append(bc_tag)
    for j in range(ny - 1, -1, -1):
        bface_v0.append(vid(0, j + 1)); bface_v1.append(vid(0, j));       bface_bc_tag.append(bc_tag)

    mesh = nd.UnstructuredMesh2D()
    mesh.vx = vx
    mesh.vy = vy
    mesh.cell_vertices = cell_vertices
    mesh.cell_offsets = cell_offsets
    mesh.material_id = mat_ids
    mesh.bface_v0 = bface_v0
    mesh.bface_v1 = bface_v1
    mesh.bface_bc_tag = bface_bc_tag
    return mesh


# ===========================================================================
# Structured 2-D time-dependent tests
# ===========================================================================


class TestTimeDependentSolver2D:
    nx, ny = 10, 10
    R = 185.0

    def _make_solver(self):
        m = one_group_mat()
        e = linspace(0.0, self.R, self.nx + 1)
        res0 = nd.KEigenSolver2D(
            mats=m,
            medium_map=[0] * (self.nx * self.ny),
            edges_x=e, edges_y=e,
            geom=nd.Geometry2D.XY,
            bc_x=[vacuum()], bc_y=[vacuum()],
            verbose=False,
        ).solve()

        return nd.TimeDependentSolver2D(
            mats=m,
            medium_map=[0] * (self.nx * self.ny),
            edges_x=e, edges_y=e,
            geom=nd.Geometry2D.XY,
            bc_x=[vacuum()], bc_y=[vacuum()],
            initial_flux=list(res0.flux),
            verbose=False,
        )

    def test_time_advances(self):
        solver = self._make_solver()
        assert solver.time == pytest.approx(0.0)
        assert solver.steps == 0
        solver.step(1e-4)
        assert solver.time == pytest.approx(1e-4)
        assert solver.steps == 1

    def test_flux_non_negative(self):
        solver = self._make_solver()
        res = solver.run(dt=1e-4, n_steps=5)
        assert np.all(np.array(res.flux) >= 0)
        assert res.steps == 5

    def test_result_shape(self):
        solver = self._make_solver()
        solver.step(1e-4)
        res = solver.result()
        assert len(res.flux) == self.nx * self.ny * 1


# ===========================================================================
# Unstructured 2-D time-dependent tests
# ===========================================================================


class TestTimeDependentSolverUnstructured2D:
    nx, ny = 8, 8
    R = 185.0

    def test_time_advances(self):
        m = one_group_mat()
        mesh = make_quad_mesh(self.nx, self.ny, self.R, self.R)

        res0 = nd.KEigenSolverUnstructured2D(
            mats=m, mesh=mesh, bc=[vacuum()], verbose=False
        ).solve()

        solver = nd.TimeDependentSolverUnstructured2D(
            mats=m,
            mesh=mesh,
            bc=[vacuum()],
            initial_flux=list(res0.flux),
            verbose=False,
        )
        assert solver.time == pytest.approx(0.0)
        solver.step(1e-4)
        assert solver.time == pytest.approx(1e-4)
        assert solver.steps == 1

    def test_run_returns_correct_steps(self):
        m = one_group_mat()
        mesh = make_quad_mesh(self.nx, self.ny, self.R, self.R)
        solver = nd.TimeDependentSolverUnstructured2D(
            mats=m, mesh=mesh, bc=[vacuum()], verbose=False
        )
        res = solver.run(dt=1e-4, n_steps=3)
        assert res.steps == 3
        flux = np.array(res.flux)
        assert len(flux) == self.nx * self.ny
        assert np.all(flux >= 0)
