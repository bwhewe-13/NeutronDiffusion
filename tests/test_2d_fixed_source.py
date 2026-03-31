"""
Pytest suite for FixedSourceSolver2D and FixedSourceSolverUnstructured2D.

Verification strategy
---------------------
Structured (FixedSourceSolver2D):
  1. 1-group XY analytic: with ny=1 and a reflective top BC the 2-D problem
     collapses to 1-D; compare to the analytic solution
         phi(x) = q/sig_a * (1 - cosh(x/L) / cosh(R/L)),  L = sqrt(D/sig_a).
  2. 2-group downscatter: fast source drives thermal flux via scatter.
  3. No-scatter decoupling: groups are independent when scatter is zero.
  4. Flux non-negativity and result-fields checks.
  5. RZ geometry smoke test: produces positive flux.
  6. Error handling: wrong source length, BC count mismatches.

Unstructured (FixedSourceSolverUnstructured2D):
  7. Structured-equivalent quad mesh analytic: same 1-D analytic reference on
     a single-row quad mesh (reflective left/bottom, vacuum right/top).
  8. Triangle mesh: solver runs and returns positive flux.
  9. 2-group downscatter on unstructured mesh.
  10. Source layout: output flux has correct length.
  11. Error handling: wrong source length.
"""

import numpy as np
import pytest

import ndiffusion as nd

# ---------------------------------------------------------------------------
# Helpers — materials
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


def two_group_absorber(scatter_01=0.02):
    """Fast (g=0) → thermal (g=1) downscatter, no fission."""
    m = nd.Materials()
    m.n_mat = 1
    m.n_groups = 2
    m.D = [1.0, 0.5]
    m.removal = [0.05, 0.1]
    # scatter[g_to * n_groups + g_from]: scatter_01 = fast→thermal
    m.scatter = [0.0, 0.0, scatter_01, 0.0]
    m.chi = [0.0, 0.0]
    m.nusigf = [0.0, 0.0]
    return m


# ---------------------------------------------------------------------------
# Helpers — mesh builders (mirrored from test_2d_unstructured.py)
# ---------------------------------------------------------------------------


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
        bface_v0.append(vid(i, 0))
        bface_v1.append(vid(i + 1, 0))
        bface_bc_tag.append(bc_tag_bottom)
    for j in range(ny):
        bface_v0.append(vid(nx, j))
        bface_v1.append(vid(nx, j + 1))
        bface_bc_tag.append(bc_tag_right)
    for i in range(nx - 1, -1, -1):
        bface_v0.append(vid(i + 1, ny))
        bface_v1.append(vid(i, ny))
        bface_bc_tag.append(bc_tag_top)
    for j in range(ny - 1, -1, -1):
        bface_v0.append(vid(0, j + 1))
        bface_v1.append(vid(0, j))
        bface_bc_tag.append(bc_tag_left)

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
# Structured 2-D fixed-source tests
# ===========================================================================


class TestFixedSource2DAnalytic:
    """Analytic reference: uniform source q in a slab of half-width R.

    Setting ny=1 with a reflective top BC collapses the 2-D problem to 1-D:
        phi(x) = q/sig_a * (1 - cosh(x/L) / cosh(R/L)),  L = sqrt(D/sig_a).

    Left (x=0) is always reflective; right (x=R) is zero-flux.
    """

    D = 1.0
    sig_a = 0.1
    q = 1.0
    R = 10.0
    nx = 200
    ny = 1

    @property
    def edges_x(self):
        return linspace(0.0, self.R, self.nx + 1)

    @property
    def edges_y(self):
        return linspace(0.0, 1.0, self.ny + 1)

    @property
    def cell_centers_x(self):
        e = self.edges_x
        return [(e[i] + e[i + 1]) / 2.0 for i in range(self.nx)]

    def analytic(self, x):
        L = np.sqrt(self.D / self.sig_a)
        return self.q / self.sig_a * (1.0 - np.cosh(x / L) / np.cosh(self.R / L))

    def test_matches_analytic_1d_collapse(self):
        solver = nd.FixedSourceSolver2D(
            one_group_absorber(self.D, self.sig_a),
            uniform_map(self.nx * self.ny),
            self.edges_x,
            self.edges_y,
            nd.Geometry2D.XY,
            bc_x=[zero_flux()],   # vacuum right
            bc_y=[reflective()],  # reflective top → reduces to 1-D
            epsilon=1e-10,
        )
        res = solver.solve([self.q] * (self.nx * self.ny))

        # flux layout: [nx*ny * 1], row-major — first (and only) group per cell
        flux = np.array(res.flux)
        analytic = self.analytic(np.array(self.cell_centers_x))
        # flux has one row (ny=1); reshape to (nx, ny) and take column 0
        phi_x = flux.reshape(self.nx, self.ny)[:, 0]
        rel_err = np.abs(phi_x - analytic) / analytic.max()
        assert np.max(rel_err) < 1e-3

    def test_flux_non_negative(self):
        solver = nd.FixedSourceSolver2D(
            one_group_absorber(),
            uniform_map(self.nx * self.ny),
            self.edges_x,
            self.edges_y,
            nd.Geometry2D.XY,
            bc_x=[zero_flux()],
            bc_y=[reflective()],
        )
        res = solver.solve([1.0] * (self.nx * self.ny))
        assert np.all(np.array(res.flux) >= 0.0)

    def test_result_fields(self):
        nx, ny = 10, 10
        solver = nd.FixedSourceSolver2D(
            one_group_absorber(),
            uniform_map(nx * ny),
            linspace(0.0, 10.0, nx + 1),
            linspace(0.0, 10.0, ny + 1),
            nd.Geometry2D.XY,
            bc_x=[zero_flux()],
            bc_y=[zero_flux()],
        )
        res = solver.solve([1.0] * (nx * ny))
        assert len(res.flux) == nx * ny * 1
        assert res.iterations > 0
        assert res.residual >= 0.0

    def test_stronger_absorption_lower_flux(self):
        """Doubling sig_a should reduce the peak flux."""
        nx, ny = 20, 5
        edges_x = linspace(0.0, 10.0, nx + 1)
        edges_y = linspace(0.0, 5.0, ny + 1)
        source = [1.0] * (nx * ny)

        def solve(sig_a):
            solver = nd.FixedSourceSolver2D(
                one_group_absorber(sig_a=sig_a),
                uniform_map(nx * ny),
                edges_x,
                edges_y,
                nd.Geometry2D.XY,
                bc_x=[zero_flux()],
                bc_y=[zero_flux()],
                epsilon=1e-10,
            )
            return max(solver.solve(source).flux)

        assert solve(0.1) > solve(0.2)


class TestFixedSource2DTwoGroup:
    """Two-group downscatter and decoupling tests on a structured mesh."""

    nx, ny = 15, 15

    @property
    def edges(self):
        return linspace(0.0, 10.0, self.nx + 1)

    def test_downscatter_drives_thermal_flux(self):
        """Source in fast group only; thermal flux must be non-zero via scatter."""
        solver = nd.FixedSourceSolver2D(
            two_group_absorber(scatter_01=0.02),
            uniform_map(self.nx * self.ny),
            self.edges,
            self.edges,
            nd.Geometry2D.XY,
            bc_x=[zero_flux(), zero_flux()],
            bc_y=[zero_flux(), zero_flux()],
        )
        # Source only in fast group; row-major: [q_fast, q_thermal, ...]
        source = [1.0, 0.0] * (self.nx * self.ny)
        flux = np.array(solver.solve(source).flux).reshape(self.nx * self.ny, 2)

        assert np.all(flux[:, 0] > 0)  # fast flux from source
        assert np.all(flux[:, 1] > 0)  # thermal flux from scatter

    def test_no_scatter_groups_independent(self):
        """Without scatter: source in thermal only → fast flux stays zero."""
        solver = nd.FixedSourceSolver2D(
            two_group_absorber(scatter_01=0.0),
            uniform_map(self.nx * self.ny),
            self.edges,
            self.edges,
            nd.Geometry2D.XY,
            bc_x=[zero_flux(), zero_flux()],
            bc_y=[zero_flux(), zero_flux()],
        )
        source = [0.0, 1.0] * (self.nx * self.ny)  # thermal source only
        flux = np.array(solver.solve(source).flux).reshape(self.nx * self.ny, 2)

        assert np.all(np.abs(flux[:, 0]) < 1e-12)  # no fast flux
        assert np.all(flux[:, 1] > 0)               # thermal flux from source

    def test_flux_layout(self):
        """Result is [nx*ny * n_groups], row-major: flux[(i*ny+j)*G + g]."""
        solver = nd.FixedSourceSolver2D(
            two_group_absorber(),
            uniform_map(self.nx * self.ny),
            self.edges,
            self.edges,
            nd.Geometry2D.XY,
            bc_x=[zero_flux(), zero_flux()],
            bc_y=[zero_flux(), zero_flux()],
        )
        res = solver.solve([1.0, 0.0] * (self.nx * self.ny))
        assert len(res.flux) == self.nx * self.ny * 2


class TestFixedSource2DGeometry:
    """Smoke tests for both supported 2-D geometries."""

    nx, ny = 20, 10

    def _make_solver(self, geom, nx=None, ny=None):
        nx = nx or self.nx
        ny = ny or self.ny
        return nd.FixedSourceSolver2D(
            one_group_absorber(),
            uniform_map(nx * ny),
            linspace(0.0, 10.0, nx + 1),
            linspace(0.0, 5.0, ny + 1),
            geom,
            bc_x=[zero_flux()],
            bc_y=[zero_flux()],
        )

    def test_xy_geometry(self):
        res = self._make_solver(nd.Geometry2D.XY).solve([1.0] * (self.nx * self.ny))
        assert np.all(np.array(res.flux) > 0)

    def test_rz_geometry(self):
        """RZ: x = z (axial), y = r (radial). ny=10 radial cells."""
        res = self._make_solver(nd.Geometry2D.RZ).solve([1.0] * (self.nx * self.ny))
        assert np.all(np.array(res.flux) > 0)


class TestFixedSource2DErrors:
    """Error-handling tests for FixedSourceSolver2D."""

    def test_wrong_source_length(self):
        cells = 10 * 10
        solver = nd.FixedSourceSolver2D(
            one_group_absorber(),
            uniform_map(cells),
            linspace(0.0, 10.0, 11),
            linspace(0.0, 10.0, 11),
            nd.Geometry2D.XY,
            bc_x=[zero_flux()],
            bc_y=[zero_flux()],
        )
        with pytest.raises(Exception):
            solver.solve([1.0] * 5)  # wrong: should be 100

    def test_bc_x_count_mismatch(self):
        """Constructor must raise when bc_x length != n_groups."""
        with pytest.raises(Exception):
            nd.FixedSourceSolver2D(
                one_group_absorber(),
                uniform_map(25),
                linspace(0.0, 5.0, 6),
                linspace(0.0, 5.0, 6),
                nd.Geometry2D.XY,
                bc_x=[zero_flux(), zero_flux()],  # 2 BCs for 1 group
                bc_y=[zero_flux()],
            )

    def test_bc_y_count_mismatch(self):
        """Constructor must raise when bc_y length != n_groups."""
        with pytest.raises(Exception):
            nd.FixedSourceSolver2D(
                one_group_absorber(),
                uniform_map(25),
                linspace(0.0, 5.0, 6),
                linspace(0.0, 5.0, 6),
                nd.Geometry2D.XY,
                bc_x=[zero_flux()],
                bc_y=[zero_flux(), zero_flux()],  # 2 BCs for 1 group
            )


# ===========================================================================
# Unstructured 2-D fixed-source tests
# ===========================================================================


class TestFixedSourceUnstructured2DAnalytic:
    """Analytic 1-D collapse on a single-row quad mesh.

    Build a mesh with nx cells in x and ny=1 row in y.  Assign:
      tag 0 = vacuum  (right and top boundaries)
      tag 1 = reflective  (left and bottom boundaries)
    This mirrors the structured solver's hardcoded left/bottom reflective
    convention and allows the same analytic comparison.
    """

    D = 1.0
    sig_a = 0.1
    q = 1.0
    R = 10.0
    nx = 100
    ny = 1

    def _mesh(self):
        return make_quad_mesh(
            self.nx, self.ny, self.R, 1.0,
            bc_tag_bottom=1,  # reflective — no y-leakage
            bc_tag_right=0,   # vacuum
            bc_tag_top=1,     # reflective — no y-leakage → collapses to 1-D
            bc_tag_left=1,    # reflective
        )

    def analytic(self, x):
        L = np.sqrt(self.D / self.sig_a)
        return self.q / self.sig_a * (1.0 - np.cosh(x / L) / np.cosh(self.R / L))

    def test_matches_analytic_1d_collapse(self):
        mesh = self._mesh()
        # Point GS needs O(N²) iterations for N=100; SOR with omega≈1.9 cuts
        # that to O(N), converging in ~900 iterations.
        solver = nd.FixedSourceSolverUnstructured2D(
            mats=one_group_absorber(self.D, self.sig_a),
            mesh=mesh,
            bc=[nd.BoundaryCondition(A=1.0, B=0.0),  # tag 0: vacuum
                nd.BoundaryCondition(A=0.0, B=1.0)], # tag 1: reflective
            epsilon=1e-10,
            max_inner=1000,
            omega=1.9,
        )
        n_cells = self.nx * self.ny
        res = solver.solve([self.q] * n_cells)

        # Cell centres in x (ny=1, cells ordered i*ny+j = i)
        dx = self.R / self.nx
        x_centers = np.array([(i + 0.5) * dx for i in range(self.nx)])
        flux = np.array(res.flux)  # [n_cells * 1]

        rel_err = np.abs(flux - self.analytic(x_centers)) / self.analytic(x_centers).max()
        assert np.max(rel_err) < 5e-3

    def test_flux_non_negative(self):
        mesh = self._mesh()
        solver = nd.FixedSourceSolverUnstructured2D(
            mats=one_group_absorber(),
            mesh=mesh,
            bc=[nd.BoundaryCondition(A=1.0, B=0.0),
                nd.BoundaryCondition(A=0.0, B=1.0)],
        )
        res = solver.solve([1.0] * (self.nx * self.ny))
        assert np.all(np.array(res.flux) >= 0.0)


class TestFixedSourceUnstructured2DTwoGroup:
    """Two-group tests on an unstructured quad mesh."""

    nx, ny = 12, 12

    def _mesh(self):
        return make_quad_mesh(self.nx, self.ny, 10.0, 10.0)

    def test_downscatter_drives_thermal_flux(self):
        """Source in fast group only; thermal flux must be non-zero via scatter."""
        mesh = self._mesh()
        n_cells = self.nx * self.ny
        solver = nd.FixedSourceSolverUnstructured2D(
            mats=two_group_absorber(scatter_01=0.02),
            mesh=mesh,
            bc=[nd.BoundaryCondition(A=1.0, B=0.0)],  # vacuum everywhere
        )
        source = [1.0, 0.0] * n_cells  # fast source only
        flux = np.array(solver.solve(source).flux).reshape(n_cells, 2)

        assert np.all(flux[:, 0] > 0)  # fast flux from source
        assert np.all(flux[:, 1] > 0)  # thermal flux from scatter

    def test_no_scatter_groups_independent(self):
        """Without scatter: source in thermal only → fast flux stays zero."""
        mesh = self._mesh()
        n_cells = self.nx * self.ny
        solver = nd.FixedSourceSolverUnstructured2D(
            mats=two_group_absorber(scatter_01=0.0),
            mesh=mesh,
            bc=[nd.BoundaryCondition(A=1.0, B=0.0)],
        )
        source = [0.0, 1.0] * n_cells  # thermal source only
        flux = np.array(solver.solve(source).flux).reshape(n_cells, 2)

        assert np.all(np.abs(flux[:, 0]) < 1e-12)  # no fast flux
        assert np.all(flux[:, 1] > 0)               # thermal flux from source


class TestFixedSourceUnstructured2DTriangle:
    """Smoke test on a triangle mesh."""

    nx, ny = 10, 10

    def test_positive_flux(self):
        mesh = make_triangle_mesh(self.nx, self.ny, 10.0, 10.0)
        n_cells = 2 * self.nx * self.ny
        solver = nd.FixedSourceSolverUnstructured2D(
            mats=one_group_absorber(),
            mesh=mesh,
            bc=[nd.BoundaryCondition(A=1.0, B=0.0)],
        )
        res = solver.solve([1.0] * n_cells)
        assert np.all(np.array(res.flux) > 0)

    def test_flux_length(self):
        """Output flux has length n_cells * n_groups = 2*nx*ny*1."""
        mesh = make_triangle_mesh(self.nx, self.ny, 10.0, 10.0)
        n_cells = 2 * self.nx * self.ny
        solver = nd.FixedSourceSolverUnstructured2D(
            mats=one_group_absorber(),
            mesh=mesh,
            bc=[nd.BoundaryCondition(A=1.0, B=0.0)],
        )
        res = solver.solve([1.0] * n_cells)
        assert len(res.flux) == n_cells * 1


class TestFixedSourceUnstructured2DErrors:
    """Error-handling tests for FixedSourceSolverUnstructured2D."""

    def test_wrong_source_length(self):
        mesh = make_quad_mesh(5, 5, 10.0, 10.0)
        solver = nd.FixedSourceSolverUnstructured2D(
            mats=one_group_absorber(),
            mesh=mesh,
            bc=[nd.BoundaryCondition(A=1.0, B=0.0)],
        )
        with pytest.raises(Exception):
            solver.solve([1.0] * 3)  # wrong: should be 25
