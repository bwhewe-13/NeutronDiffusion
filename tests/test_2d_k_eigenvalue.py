"""
Pytest suite for KEigenSolver2D and KEigenSolverUnstructured2D.

Verification strategy
---------------------
Structured (KEigenSolver2D):
  1. 1-group XY: with ny=1 and a reflective top BC the 2-D problem collapses
     to 1-D; keff must match KEigenSolver on the same x-grid.
  2. 1-group RZ: with nx=1 the RZ problem collapses to a 1-D cylinder; keff
     must match KEigenSolver (Cylinder) on the same radial grid.
  3. 2-group XY smoke test: solver converges to a physically reasonable keff
     with the correct flux dimensions.

Unstructured (KEigenSolverUnstructured2D):
  4. Structured-equivalent quad mesh: keff matches KEigenSolver2D on the same
     Cartesian grid to within FVM vs FD discretisation error.
  5. Triangle mesh: solver runs without error on a fully unstructured mesh.
  6. BC effect: reflective BCs produce a higher keff than vacuum BCs.
"""

import numpy as np

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


def two_group_mat():
    """2-group, 1-material cross sections (no upscatter)."""
    m = nd.Materials()
    m.n_mat    = 1
    m.n_groups = 2
    m.D        = [1.255, 0.211]
    m.removal  = [0.00836, 0.1003]
    m.scatter  = [0.0,    0.02533,   # g_to=0: from g0, from g1
                  0.0,    0.0]       # g_to=1: from g0, from g1
    m.chi      = [1.0, 0.0]
    m.nusigf   = [0.004602, 0.1091]
    m.velocity = [2.2e7, 2.2e5]
    return m


def make_quad_mesh(
    nx,
    ny,
    Lx,
    Ly,
    material_id=None,
    bc_tag_bottom=0,
    bc_tag_right=0,
    bc_tag_top=0,
    bc_tag_left=0,
):
    """Build a regular nx×ny quad mesh on [0,Lx]×[0,Ly] with per-side BC tags."""
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
                vid(i, j),
                vid(i + 1, j),
                vid(i + 1, j + 1),
                vid(i, j + 1),
            ]
            cell_offsets.append(len(cell_vertices))
            mat_ids.append(material_id[i * ny + j] if material_id else 0)

    bface_v0, bface_v1, bface_bc_tag = [], [], []
    for i in range(nx):
        bface_v0.append(vid(i, 0));    bface_v1.append(vid(i + 1, 0));    bface_bc_tag.append(bc_tag_bottom)
    for j in range(ny):
        bface_v0.append(vid(nx, j));   bface_v1.append(vid(nx, j + 1));   bface_bc_tag.append(bc_tag_right)
    for i in range(nx - 1, -1, -1):
        bface_v0.append(vid(i + 1, ny)); bface_v1.append(vid(i, ny));     bface_bc_tag.append(bc_tag_top)
    for j in range(ny - 1, -1, -1):
        bface_v0.append(vid(0, j + 1)); bface_v1.append(vid(0, j));       bface_bc_tag.append(bc_tag_left)

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


def make_triangle_mesh(nx, ny, Lx, Ly, bc_tag=0):
    """Build a mesh of right triangles by splitting each quad in two."""
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
            cell_vertices += [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)]
            cell_offsets.append(len(cell_vertices))
            cell_vertices += [vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)]
            cell_offsets.append(len(cell_vertices))
            mat_ids.extend([0, 0])

    bface_v0, bface_v1, bface_bc_tag = [], [], []

    def add_face(va, vb):
        bface_v0.append(va)
        bface_v1.append(vb)
        bface_bc_tag.append(bc_tag)

    for i in range(nx):
        add_face(vid(i, 0), vid(i + 1, 0))
    for j in range(ny):
        add_face(vid(nx, j), vid(nx, j + 1))
    for i in range(nx - 1, -1, -1):
        add_face(vid(i + 1, ny), vid(i, ny))
    for j in range(ny - 1, -1, -1):
        add_face(vid(0, j + 1), vid(0, j))

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
# Structured 2-D k-eigenvalue tests
# ===========================================================================


class TestXYOneGroup:
    """
    A square XY domain.  Both axes use identical cross-sections and edges,
    so the 2-D problem should give the same keff as a 1-D slab of the same
    extent — provided the 1-D reference is also solved on the same mesh.

    We use reflective BC on one axis to reduce the 2-D problem exactly to 1-D.
    """

    nx, ny = 40, 1
    R = 185.0

    def _mats(self):
        return one_group_mat()

    def _edges_x(self):
        return linspace(0.0, self.R, self.nx + 1)

    def _edges_y(self):
        return linspace(0.0, 10.0, self.ny + 1)

    def test_keff_matches_1d(self):
        m = self._mats()
        ex = self._edges_x()
        ey = self._edges_y()

        # 2-D solve: vacuum on right (x), reflective on top (y).
        # With reflective top BC the y-direction contributes no leakage,
        # so the result should match a 1-D slab exactly.
        solver_2d = nd.KEigenSolver2D(
            mats=m,
            medium_map=[0] * (self.nx * self.ny),
            edges_x=ex,
            edges_y=ey,
            geom=nd.Geometry2D.XY,
            bc_x=[vacuum()],
            bc_y=[reflective()],
            epsilon=1e-10,
            verbose=False,
        )
        res_2d = solver_2d.solve()

        solver_1d = nd.KEigenSolver(
            mats=m,
            medium_map=[0] * self.nx,
            edges_x=ex,
            geom=nd.Geometry.Slab,
            bc=[vacuum()],
            epsilon=1e-10,
            verbose=False,
        )
        res_1d = solver_1d.solve()

        assert abs(res_2d.keff - res_1d.keff) < 1e-5, (
            f"2-D keff {res_2d.keff:.8f} differs from 1-D keff {res_1d.keff:.8f}"
        )

    def test_flux_shape(self):
        m = self._mats()
        ex = self._edges_x()
        ey = self._edges_y()
        solver = nd.KEigenSolver2D(
            mats=m,
            medium_map=[0] * (self.nx * self.ny),
            edges_x=ex, edges_y=ey,
            geom=nd.Geometry2D.XY,
            bc_x=[vacuum()], bc_y=[reflective()],
            verbose=False,
        )
        res = solver.solve()
        flux = np.array(res.flux).reshape(self.nx, self.ny, 1)
        assert flux.shape == (self.nx, self.ny, 1)
        assert np.all(flux > 0)
        mid = self.nx // 2
        assert flux[:mid, 0, 0].mean() > flux[mid:, 0, 0].mean()
        assert flux[0, 0, 0] > flux[-1, 0, 0]


class TestRZOneGroup:
    """
    When ny = 1 the RZ mesh is a single radial ring, so r-direction leakage
    vanishes (reflective top) and we recover the 1-D cylinder problem.
    """

    nx, ny = 1, 50
    R = 185.0

    def test_keff_matches_1d_cylinder(self):
        m = one_group_mat()
        ex = linspace(0.0, 10.0, self.nx + 1)
        ey = linspace(0.0, self.R, self.ny + 1)

        solver_2d = nd.KEigenSolver2D(
            mats=m,
            medium_map=[0] * (self.nx * self.ny),
            edges_x=ex, edges_y=ey,
            geom=nd.Geometry2D.RZ,
            bc_x=[reflective()],
            bc_y=[vacuum()],
            epsilon=1e-10,
            verbose=False,
        )
        res_2d = solver_2d.solve()

        solver_1d = nd.KEigenSolver(
            mats=m,
            medium_map=[0] * self.ny,
            edges_x=ey,
            geom=nd.Geometry.Cylinder,
            bc=[vacuum()],
            epsilon=1e-10,
            verbose=False,
        )
        res_1d = solver_1d.solve()

        assert abs(res_2d.keff - res_1d.keff) < 1e-4, (
            f"RZ keff {res_2d.keff:.8f} vs cylinder keff {res_1d.keff:.8f}"
        )


class TestXYTwoGroup:
    nx, ny = 20, 20
    R = 100.0

    def test_converges(self):
        m = two_group_mat()
        e = linspace(0.0, self.R, self.nx + 1)
        solver = nd.KEigenSolver2D(
            mats=m,
            medium_map=[0] * (self.nx * self.ny),
            edges_x=e, edges_y=e,
            geom=nd.Geometry2D.XY,
            bc_x=[vacuum(), vacuum()],
            bc_y=[vacuum(), vacuum()],
            verbose=False,
        )
        res = solver.solve()
        assert 0.5 < res.keff < 2.0
        assert np.all(np.array(res.flux).reshape(self.nx, self.ny, 2) >= 0)

    def test_flux_dimensions(self):
        m = two_group_mat()
        e = linspace(0.0, 100.0, self.nx + 1)
        solver = nd.KEigenSolver2D(
            mats=m,
            medium_map=[0] * (self.nx * self.ny),
            edges_x=e, edges_y=e,
            geom=nd.Geometry2D.XY,
            bc_x=[vacuum(), vacuum()],
            bc_y=[vacuum(), vacuum()],
            verbose=False,
        )
        res = solver.solve()
        assert len(res.flux) == self.nx * self.ny * 2


# ===========================================================================
# Unstructured 2-D k-eigenvalue tests
# ===========================================================================


class TestQuadVsStructured:
    nx, ny = 15, 15
    R = 185.0

    def test_keff_close_to_structured(self):
        m = one_group_mat()
        mesh = make_quad_mesh(
            self.nx, self.ny, self.R, self.R,
            bc_tag_bottom=1, bc_tag_right=0, bc_tag_top=0, bc_tag_left=1,
        )

        solver_u = nd.KEigenSolverUnstructured2D(
            mats=m,
            mesh=mesh,
            bc=[vacuum(), reflective()],
            epsilon=1e-8,
            max_inner=200,
            verbose=False,
        )
        res_u = solver_u.solve()

        e = list(np.linspace(0.0, self.R, self.nx + 1))
        solver_s = nd.KEigenSolver2D(
            mats=m,
            medium_map=[0] * (self.nx * self.ny),
            edges_x=e, edges_y=e,
            geom=nd.Geometry2D.XY,
            bc_x=[vacuum()], bc_y=[vacuum()],
            epsilon=1e-8,
            verbose=False,
        )
        res_s = solver_s.solve()

        assert abs(res_u.keff - res_s.keff) < 0.01, (
            f"Unstructured keff {res_u.keff:.6f} vs structured {res_s.keff:.6f}"
        )

    def test_flux_positive(self):
        m = one_group_mat()
        mesh = make_quad_mesh(self.nx, self.ny, self.R, self.R)
        solver = nd.KEigenSolverUnstructured2D(
            mats=m, mesh=mesh, bc=[vacuum()], verbose=False
        )
        res = solver.solve()
        flux = np.array(res.flux)
        assert np.all(flux > 0)
        assert len(flux) == self.nx * self.ny * 1


class TestTriangleMesh:
    nx, ny = 10, 10
    R = 185.0

    def test_converges(self):
        m = one_group_mat()
        mesh = make_triangle_mesh(self.nx, self.ny, self.R, self.R)
        solver = nd.KEigenSolverUnstructured2D(
            mats=m, mesh=mesh, bc=[vacuum()], verbose=False
        )
        res = solver.solve()
        assert 0.1 < res.keff < 5.0
        assert np.all(np.array(res.flux) >= 0)

    def test_flux_length(self):
        m = one_group_mat()
        mesh = make_triangle_mesh(self.nx, self.ny, self.R, self.R)
        solver = nd.KEigenSolverUnstructured2D(
            mats=m, mesh=mesh, bc=[vacuum()], verbose=False
        )
        res = solver.solve()
        assert len(res.flux) == 2 * self.nx * self.ny * 1


class TestBoundaryConditionEffect:
    nx, ny = 10, 10
    R = 100.0

    def test_reflective_higher_keff(self):
        m = one_group_mat()
        mesh_vac  = make_quad_mesh(self.nx, self.ny, self.R, self.R)
        mesh_refl = make_quad_mesh(self.nx, self.ny, self.R, self.R)

        keff_vac  = nd.KEigenSolverUnstructured2D(
            mats=m, mesh=mesh_vac,  bc=[vacuum()],     verbose=False
        ).solve().keff
        keff_refl = nd.KEigenSolverUnstructured2D(
            mats=m, mesh=mesh_refl, bc=[reflective()],  verbose=False
        ).solve().keff

        assert keff_refl > keff_vac
