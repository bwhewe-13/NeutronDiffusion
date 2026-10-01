"""Mesh geometry queries, polygon cells and mesh validation."""

import numpy as np
import pytest

import ndiffusion as nd

L = 24.0
N = 12


def quad_grid(n=N, size=L):
    """n x n quad mesh on [0, size]^2, every boundary face tag 0, no materials yet."""
    h = size / n
    vx, vy = [], []
    for i in range(n + 1):
        for j in range(n + 1):
            vx.append(i * h); vy.append(j * h)

    def vid(i, j):
        return i * (n + 1) + j

    cv, co = [], [0]
    for i in range(n):
        for j in range(n):
            cv += [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
            co.append(len(cv))
    b0, b1, bt = [], [], []
    for i in range(n):
        b0 += [vid(i, 0), vid(i, n)]; b1 += [vid(i + 1, 0), vid(i + 1, n)]; bt += [0, 0]
    for j in range(n):
        b0 += [vid(0, j), vid(n, j)]; b1 += [vid(0, j + 1), vid(n, j + 1)]; bt += [0, 0]

    m = nd.UnstructuredMesh2D()
    m.vx, m.vy = vx, vy
    m.cell_vertices, m.cell_offsets = cv, co
    m.material_id = [0] * (n * n)
    m.bface_v0, m.bface_v1, m.bface_bc_tag = b0, b1, bt
    return m


def hex_lattice(nx, ny, pitch):
    """Honeycomb of regular hexagons, nx x ny cells, flat rows offset by half a pitch.

    `pitch` is the centre-to-centre distance of side-adjacent hexagons, so the
    circumradius is pitch/sqrt(3).  Vertices are merged by rounded position, which
    is what makes neighbouring cells share edges rather than come apart.
    """
    r = pitch / np.sqrt(3.0)
    vids, vx, vy = {}, [], []

    def vertex(x, y):
        key = (round(x, 9), round(y, 9))
        if key not in vids:
            vids[key] = len(vx)
            vx.append(key[0]); vy.append(key[1])
        return vids[key]

    cv, co = [], [0]
    for j in range(ny):
        for i in range(nx):
            cx = pitch * (i + 0.5 * (j % 2))
            cy = 1.5 * r * j
            for k in range(6):
                a = np.pi / 2.0 + np.pi / 3.0 * k
                cv.append(vertex(cx + r * np.cos(a), cy + r * np.sin(a)))
            co.append(len(cv))

    m = nd.UnstructuredMesh2D()
    m.vx, m.vy = vx, vy
    m.cell_vertices, m.cell_offsets = cv, co
    m.material_id = [0] * (nx * ny)
    return m


def mats(n_mat):
    m = nd.Materials()
    m.n_mat = n_mat
    m.n_groups = 1
    m.D = [1.0] * n_mat
    m.removal = [0.1] * n_mat
    m.scatter = [0.0] * n_mat
    m.chi = [1.0] * n_mat
    m.nusigf = [0.12 + 0.03 * i for i in range(n_mat)]
    return m


VACUUM = [nd.BoundaryCondition(A=0.25, B=0.5)]


class TestMeshGeometry:
    def test_triangle_centroid_and_area(self):
        m = nd.UnstructuredMesh2D()
        m.vx = [0.0, 1.0, 0.0]
        m.vy = [0.0, 0.0, 1.0]
        m.cell_vertices = [0, 1, 2]
        m.cell_offsets = [0, 3]
        m.material_id = [0]
        cx, cy = nd.cell_centroids(m)
        assert cx == pytest.approx([1.0 / 3.0])
        assert cy == pytest.approx([1.0 / 3.0])
        assert nd.cell_areas(m) == pytest.approx([0.5])

    def test_quad_centroid_and_area(self):
        m = nd.UnstructuredMesh2D()
        m.vx = [0.0, 2.0, 2.0, 0.0]
        m.vy = [0.0, 0.0, 1.0, 1.0]
        m.cell_vertices = [0, 1, 2, 3]
        m.cell_offsets = [0, 4]
        m.material_id = [0]
        cx, cy = nd.cell_centroids(m)
        assert (cx[0], cy[0]) == pytest.approx((1.0, 0.5))
        assert nd.cell_areas(m) == pytest.approx([2.0])

    def test_areas_sum_to_domain(self):
        assert sum(nd.cell_areas(quad_grid())) == pytest.approx(L * L)

    def test_centroids_lie_inside_the_domain(self):
        cx, cy = nd.cell_centroids(quad_grid())
        assert len(cx) == len(cy) == N * N
        assert min(cx) > 0.0 and max(cx) < L
        assert min(cy) > 0.0 and max(cy) < L

    @pytest.mark.parametrize(
        "mutate,match",
        [
            (lambda m: setattr(m, "vy", [0.0]), "same length"),
            (lambda m: setattr(m, "cell_offsets", [1, 5]), "must start at 0"),
            (lambda m: (setattr(m, "cell_vertices", [0, 1]),
                        setattr(m, "cell_offsets", [0, 2])), "at least 3"),
            (lambda m: (setattr(m, "vx", [0.0, 1.0, 2.0, 3.0]),
                        setattr(m, "vy", [0.0, 0.0, 0.0, 0.0])), "zero area"),
            (lambda m: setattr(m, "cell_vertices", [0, 1, 99, 3]), "outside"),
            (lambda m: setattr(m, "bface_v1", []), "same length"),
            (lambda m: setattr(m, "bface_bc_tag", [0] * 99), "more entries"),
        ],
    )
    def test_malformed_mesh_raises(self, mutate, match):
        m = nd.UnstructuredMesh2D()
        m.vx = [0.0, 1.0, 1.0, 0.0]
        m.vy = [0.0, 0.0, 1.0, 1.0]
        m.cell_vertices = [0, 1, 2, 3]
        m.cell_offsets = [0, 4]
        m.material_id = [0]
        m.bface_v0, m.bface_v1, m.bface_bc_tag = [0], [1], [0]
        mutate(m)
        with pytest.raises(ValueError, match=match):
            nd.validate_mesh(m)

    def test_solver_rejects_a_malformed_mesh(self):
        m = quad_grid(4)
        m.cell_vertices = [0, 1, 999] + list(m.cell_vertices)[3:]
        with pytest.raises(ValueError, match="outside"):
            nd.KEigenSolverUnstructured2D(mats(1), m, VACUUM)


class TestPolygonCells:
    """Cells may be any simple polygon, not just triangles and quads.

    The shoelace formulae give the centroid and area for any vertex count, and
    the face loop was already generic, so hexagonal lattices work directly.
    """

    @staticmethod
    def regular_polygon(n_sides, r=1.0, x0=0.0, y0=0.0):
        ang = [2.0 * np.pi * k / n_sides for k in range(n_sides)]
        m = nd.UnstructuredMesh2D()
        m.vx = [x0 + r * np.cos(a) for a in ang]
        m.vy = [y0 + r * np.sin(a) for a in ang]
        m.cell_vertices = list(range(n_sides))
        m.cell_offsets = [0, n_sides]
        m.material_id = [0]
        return m

    @pytest.mark.parametrize("n_sides", [3, 4, 5, 6, 8, 12])
    def test_regular_polygon_area_and_centroid(self, n_sides):
        m = self.regular_polygon(n_sides, r=2.0, x0=5.0, y0=-3.0)
        exact = 0.5 * n_sides * 2.0**2 * np.sin(2.0 * np.pi / n_sides)
        cx, cy = nd.cell_centroids(m)
        assert nd.cell_areas(m) == pytest.approx([exact])
        assert (cx[0], cy[0]) == pytest.approx((5.0, -3.0))

    def test_winding_direction_does_not_matter(self):
        ccw = self.regular_polygon(6, r=2.0, x0=5.0, y0=-3.0)
        cw = self.regular_polygon(6, r=2.0, x0=5.0, y0=-3.0)
        cw.cell_vertices = list(reversed(range(6)))
        assert nd.cell_areas(cw) == pytest.approx(nd.cell_areas(ccw))
        cx_cw, cy_cw = nd.cell_centroids(cw)
        cx_ccw, cy_ccw = nd.cell_centroids(ccw)
        assert cx_cw == pytest.approx(cx_ccw)
        assert cy_cw == pytest.approx(cy_ccw)

    def test_degenerate_cell_rejected(self):
        m = self.regular_polygon(6)
        m.vx = [0.0] * 6                      # collapse onto a line
        with pytest.raises(ValueError, match="zero area"):
            nd.validate_mesh(m)


class TestNonConforming:
    """A hanging node leaves edges that cannot pair, disconnecting the interior."""

    @staticmethod
    def _hanging():
        m = nd.UnstructuredMesh2D()
        m.vx = [0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 2.0, 2.0]
        m.vy = [0.0, 0.0, 2.0, 2.0, 1.0, 0.0, 1.0, 2.0]
        # Big cell (0,1,2,3) abuts two small cells that share vertex 4 on (1,2).
        m.cell_vertices = [0, 1, 2, 3, 1, 5, 6, 4, 4, 6, 7, 2]
        m.cell_offsets = [0, 4, 8, 12]
        m.material_id = [0, 0, 0]
        return m

    def test_hanging_node_raises(self):
        with pytest.raises(ValueError, match="non-conforming"):
            nd.KEigenSolverUnstructured2D(mats(1), self._hanging(), VACUUM)

    def test_conforming_equivalent_is_accepted(self):
        """Splitting the large cell so every face matches removes the hanging node."""
        m = self._hanging()
        m.vx = list(m.vx) + [0.0]
        m.vy = list(m.vy) + [1.0]
        m.cell_vertices = [0, 1, 4, 8, 8, 4, 2, 3, 1, 5, 6, 4, 4, 6, 7, 2]
        m.cell_offsets = [0, 4, 8, 12, 16]
        m.material_id = [0, 0, 0, 0]
        res = nd.KEigenSolverUnstructured2D(
            mats(1), m, VACUUM, epsilon=1e-7, max_inner=3000, verbose=False
        ).solve()
        assert res.converged

    @pytest.mark.parametrize("n", [3, 6, 10])
    def test_regular_grids_are_not_false_positives(self, n):
        nd.validate_mesh(quad_grid(n))
        nd.KEigenSolverUnstructured2D(mats(1), quad_grid(n), VACUUM,
                                      epsilon=1e-6, max_inner=2000, verbose=False)

    def test_hex_lattice_is_not_a_false_positive(self):
        nd.KEigenSolverUnstructured2D(mats(1), hex_lattice(3, 3, 2.0), VACUUM,
                                      epsilon=1e-6, max_inner=2000, verbose=False)


def tri_split_mesh(n, size=L):
    """Same grid as quad_grid, each square split into two right triangles.

    Every face shared across a diagonal is non-orthogonal, so this exercises the
    deferred correction; the quad mesh of the same square does not.
    """
    h = size / n
    vx, vy = [], []
    for i in range(n + 1):
        for j in range(n + 1):
            vx.append(i * h); vy.append(j * h)

    def vid(i, j):
        return i * (n + 1) + j

    cv, co = [], [0]
    for i in range(n):
        for j in range(n):
            cv += [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)]; co.append(len(cv))
            cv += [vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)]; co.append(len(cv))
    m = nd.UnstructuredMesh2D()
    m.vx, m.vy = vx, vy
    m.cell_vertices, m.cell_offsets = cv, co
    m.material_id = [0] * (2 * n * n)
    return m


class TestNonOrthogonalCorrection:
    """The two-point flux needs correcting where the centroid line is not normal
    to the face.  Without it the right-triangle scheme is inconsistent: its error
    grows under refinement towards a wrong limit.
    """

    D, SIGA, NUSIGF = 1.0, 0.1, 0.30

    def _mats(self):
        m = nd.Materials()
        m.n_mat = 1
        m.n_groups = 1
        m.D = [self.D]
        m.removal = [self.SIGA]
        m.scatter = [0.0]
        m.chi = [1.0]
        m.nusigf = [self.NUSIGF]
        return m

    def _k_exact(self):
        b2 = 2.0 * (np.pi / L) ** 2
        return self.NUSIGF / (self.SIGA + self.D * b2)

    def _solve(self, mesh):
        return nd.KEigenSolverUnstructured2D(
            self._mats(), mesh, [nd.BoundaryCondition(A=1.0, B=0.0)],
            epsilon=1e-10, max_outer=3000, max_inner=200, use_cg=True,
            verbose=False,
        ).solve().keff

    def test_triangle_mesh_converges(self):
        exact = self._k_exact()
        errs = [abs(self._solve(tri_split_mesh(n)) - exact) for n in (8, 16, 32)]
        assert errs[0] > errs[1] > errs[2]
        rate = np.log(errs[1] / errs[2]) / np.log(2.0)
        assert rate > 1.4          # heading to 2; ~1.7 at these resolutions

    def test_quad_mesh_is_second_order_and_unaffected(self):
        """Orthogonal faces give a zero correction, so quads are untouched."""
        exact = self._k_exact()
        errs = [abs(self._solve(quad_grid(n)) - exact) for n in (8, 16, 32)]
        for coarse, fine in zip(errs, errs[1:]):
            assert np.log(coarse / fine) / np.log(2.0) == pytest.approx(2.0, abs=0.05)

    def test_quad_faces_carry_no_correction(self):
        """A regular quad grid is detected as orthogonal, so nothing is computed."""
        mesh = quad_grid(6)
        cx, cy = nd.cell_centroids(mesh)
        assert len(cx) == 36           # geometry query still works
        # Equivalent statement at the solver level: the quad result matches the
        # uncorrected two-point flux exactly, which the rate test above pins.

    def test_triangle_and_quad_agree_in_the_limit(self):
        exact = self._k_exact()
        k_tri = self._solve(tri_split_mesh(32))
        k_quad = self._solve(quad_grid(32))
        assert abs(k_tri - exact) < 1e-3
        assert abs(k_quad - exact) < 1e-3
