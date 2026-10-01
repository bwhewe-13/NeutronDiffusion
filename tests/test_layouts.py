"""Preset layouts, symmetry orientations, and the boundary conditions they imply.

The orientation tests are the substantive ones: the same physical problem solved
on a full core and on each symmetry sector must give the same eigenvalue, which
exercises the sector clipping, the vertex merging, and the symmetry/outer tagging
together.  Residual differences are discretization - the sectors slice cells into
different polygons - so they are checked to shrink under refinement.
"""

import numpy as np
import pytest

import ndiffusion as nd
from ndiffusion import layouts as L

D1, SIGR, NUSIGF = 1.0, 0.1, 0.30


def mats(n_mat=1, nusigf=None):
    m = nd.Materials()
    m.n_mat = n_mat
    m.n_groups = 1
    m.D = [D1] * n_mat
    m.removal = [SIGR] * n_mat
    m.scatter = [0.0] * n_mat
    m.chi = [1.0] * n_mat
    m.nusigf = list(nusigf) if nusigf is not None else [NUSIGF] * n_mat
    return m


def solve(mesh, m=None, albedo=0.0):
    bc = L.boundary_conditions(mesh, D=[D1], albedo=albedo)
    return nd.KEigenSolverUnstructured2D(
        m if m is not None else mats(), mesh, bc, epsilon=1e-10,
        max_outer=5000, max_inner=500, use_cg=True, verbose=False,
    ).solve()


def n_cells(mesh):
    return len(mesh.cell_offsets) - 1


class TestOrientations:
    def test_infinite_gives_k_inf(self):
        """All boundaries reflective: keff is the infinite-medium value exactly."""
        mesh = L.cartesian_mesh(width=40.0, h=4.0, orientation="infinite")
        assert set(mesh.bface_bc_tag) == {L.SYMMETRY}
        assert solve(mesh).keff == pytest.approx(NUSIGF / SIGR, rel=1e-9)

    def test_hex_infinite_gives_k_inf(self):
        mesh = L.hex_mesh(pitch=4.0, n_rings=3, orientation="infinite")
        assert solve(mesh).keff == pytest.approx(NUSIGF / SIGR, rel=1e-9)

    @pytest.mark.parametrize("orientation,frac",
                             [("half", 2), ("quarter", 4), ("eighth", 8)])
    def test_cartesian_sector_area(self, orientation, frac):
        full = L.cartesian_mesh(width=40.0, h=2.0, orientation="full")
        part = L.cartesian_mesh(width=40.0, h=2.0, orientation=orientation)
        assert sum(nd.cell_areas(part)) == pytest.approx(
            sum(nd.cell_areas(full)) / frac)

    @pytest.mark.parametrize("orientation,frac",
                             [("half", 2), ("sector120", 3),
                              ("sector60", 6), ("sector30", 12)])
    def test_hex_sector_area(self, orientation, frac):
        full = L.hex_mesh(pitch=4.0, n_rings=4, orientation="full")
        part = L.hex_mesh(pitch=4.0, n_rings=4, orientation=orientation)
        assert sum(nd.cell_areas(part)) == pytest.approx(
            sum(nd.cell_areas(full)) / frac)

    def test_cartesian_sectors_agree(self):
        """Cuts fall on cell edges, so half and quarter are the same solve."""
        keff = {}
        for o in ("full", "half", "quarter"):
            mesh = L.cartesian_mesh(width=40.0, h=2.0, orientation=o)
            nd.assign_materials(mesh, L.core_reflector(core_radius=12.0))
            keff[o] = solve(mesh, mats(2, nusigf=[NUSIGF, 0.10])).keff
        assert keff["half"] == pytest.approx(keff["full"], rel=1e-8)
        assert keff["quarter"] == pytest.approx(keff["full"], rel=1e-8)

    def test_eighth_converges_to_quarter(self):
        """The 45-degree cut makes triangles, so agreement is O(h^2), not exact."""
        dev = []
        for h in (4.0, 1.0):
            kq = solve(L.cartesian_mesh(width=40.0, h=h, orientation="quarter")).keff
            ke = solve(L.cartesian_mesh(width=40.0, h=h, orientation="eighth")).keff
            dev.append(abs(ke - kq) / kq)
        assert dev[0] < 1e-3
        assert dev[1] < dev[0] / 4.0

    def test_hex_sectors_agree_and_converge(self):
        dev = []
        for n in (3, 8):
            pitch = 24.0 / n
            ks = [solve(L.hex_mesh(pitch=pitch, n_rings=n, orientation=o)).keff
                  for o in ("full", "sector120", "sector60", "sector30")]
            dev.append(max(abs(k - ks[0]) / ks[0] for k in ks))
        assert dev[0] < 5e-3
        assert dev[1] < dev[0] / 2.0

    def test_sector_cells_are_polygons(self):
        """Sector cuts slice hexagons, which the solver takes as-is."""
        mesh = L.hex_mesh(pitch=4.0, n_rings=4, orientation="sector30")
        off = list(mesh.cell_offsets)
        counts = {off[i + 1] - off[i] for i in range(n_cells(mesh))}
        assert counts - {3, 4, 5, 6} == set()
        assert len(counts) > 1          # a mix, not just whole hexagons

    def test_boundary_tags_split(self):
        mesh = L.cartesian_mesh(width=40.0, h=4.0, orientation="quarter")
        assert set(mesh.bface_bc_tag) == {L.SYMMETRY, L.OUTER}

    def test_full_core_has_no_symmetry_faces(self):
        mesh = L.cartesian_mesh(width=40.0, h=4.0, orientation="full")
        assert set(mesh.bface_bc_tag) == {L.OUTER}

    @pytest.mark.parametrize("gen,kwargs", [
        (L.cartesian_mesh, dict(width=10.0, h=1.0)),
        (L.hex_mesh, dict(pitch=4.0, n_rings=2)),
    ])
    def test_unknown_orientation_raises(self, gen, kwargs):
        with pytest.raises(ValueError, match="orientation must be one of"):
            gen(orientation="sideways", **kwargs)

    def test_meshes_validate(self):
        for o in ("full", "half", "quarter", "eighth"):
            nd.validate_mesh(L.cartesian_mesh(width=20.0, h=2.0, orientation=o))
        for o in ("full", "half", "sector120", "sector60", "sector30"):
            nd.validate_mesh(L.hex_mesh(pitch=4.0, n_rings=3, orientation=o))


class TestPainters:
    def _mesh(self, h=2.0):
        return L.cartesian_mesh(width=40.0, h=h, orientation="full")

    def test_homogeneous(self):
        mesh = self._mesh()
        nd.assign_materials(mesh, L.homogeneous())
        assert set(mesh.material_id) == {0}
        nd.assign_materials(mesh, L.homogeneous(material=3))
        assert set(mesh.material_id) == {3}

    def test_core_reflector_shapes(self):
        mesh = self._mesh()
        nd.assign_materials(mesh, L.core_reflector(core_radius=12.0))
        circ = list(mesh.material_id)
        nd.assign_materials(mesh, L.core_reflector(core_half_width=12.0))
        assert set(circ) == {0, 1}
        assert circ.count(0) < mesh.material_id.count(0)   # circle fits in square

    def test_core_reflector_needs_exactly_one_shape(self):
        with pytest.raises(ValueError, match="exactly one"):
            L.core_reflector()
        with pytest.raises(ValueError, match="exactly one"):
            L.core_reflector(core_radius=1.0, core_half_width=1.0)

    def test_reflector_savings(self):
        """A reflected core beats a bare core of the same fissile extent."""
        fuel_only = mats(1)
        bare = L.cartesian_mesh(width=24.0, h=1.0, orientation="full")
        nd.assign_materials(bare, L.homogeneous())
        k_bare = solve(bare, fuel_only).keff

        reflected = L.cartesian_mesh(width=40.0, h=1.0, orientation="full")
        nd.assign_materials(reflected, L.core_reflector(core_half_width=12.0))
        two = mats(2, nusigf=[NUSIGF, 0.0])
        two.removal = [SIGR, 0.02]        # reflector: scatters, barely absorbs
        assert solve(reflected, two).keff > k_bare

    def test_checkerboard_alternates(self):
        mesh = self._mesh(h=2.0)
        nd.assign_materials(mesh, L.checkerboard(pitch=4.0))
        assert set(mesh.material_id) == {0, 1}
        cx, cy = nd.cell_centroids(mesh)
        for x, y, m in zip(cx, cy, mesh.material_id):
            assert m == (int(np.floor(x / 4.0)) + int(np.floor(y / 4.0))) % 2

    def test_checkerboard_keff_between_pure(self):
        mesh = self._mesh(h=1.0)
        two = mats(2, nusigf=[0.40, 0.20])
        nd.assign_materials(mesh, L.homogeneous(0))
        hi = solve(mesh, two).keff
        nd.assign_materials(mesh, L.homogeneous(1))
        lo = solve(mesh, two).keff
        nd.assign_materials(mesh, L.checkerboard(pitch=4.0))
        assert lo < solve(mesh, two).keff < hi

    def test_checkerboard_needs_two_materials(self):
        with pytest.raises(ValueError, match="at least two"):
            L.checkerboard(pitch=1.0, materials=(0,))

    def test_annular_rings(self):
        mesh = self._mesh()
        nd.assign_materials(mesh, L.annular([6.0, 12.0], [0, 1, 2]))
        cx, cy = nd.cell_centroids(mesh)
        for x, y, m in zip(cx, cy, mesh.material_id):
            r = np.hypot(x, y)
            assert m == (0 if r <= 6.0 else 1 if r <= 12.0 else 2)

    def test_annular_validates(self):
        with pytest.raises(ValueError, match="one more entry"):
            L.annular([1.0, 2.0], [0, 1])
        with pytest.raises(ValueError, match="strictly increasing"):
            L.annular([2.0, 1.0], [0, 1, 2])

    def test_hex_rings(self):
        mesh = L.hex_mesh(pitch=4.0, n_rings=3, orientation="full")
        nd.assign_materials(mesh, L.hex_rings(4.0, [0, 0, 1, 2]))
        assert set(mesh.material_id) == {0, 1, 2}
        cx, cy = nd.cell_centroids(mesh)
        centre = int(np.argmin(np.hypot(cx, cy)))
        assert mesh.material_id[centre] == 0
        edge = int(np.argmax(np.hypot(cx, cy)))
        assert mesh.material_id[edge] == 2

    def test_hex_rings_needs_materials(self):
        with pytest.raises(ValueError, match="must not be empty"):
            L.hex_rings(1.0, [])

    def test_rods_lower_keff(self):
        """The perturbation pair: one geometry, two layouts, a rod worth."""
        mesh = self._mesh(h=1.0)
        three = mats(3, nusigf=[NUSIGF, 0.0, 0.0])
        three.removal = [SIGR, 0.02, 0.8]      # 2 = strong absorber

        unrodded = L.core_reflector(core_half_width=12.0)
        rodded = L.with_rods(unrodded, [(0.0, 0.0), (8.0, 8.0)], 3.0, material=2)

        nd.assign_materials(mesh, unrodded)
        k0 = solve(mesh, three).keff
        nd.assign_materials(mesh, rodded)
        k1 = solve(mesh, three).keff

        assert k1 < k0
        assert (k1 - k0) / (k1 * k0) < 0.0     # negative reactivity worth

    def test_rods_only_change_the_rodded_cells(self):
        mesh = self._mesh()
        base = L.core_reflector(core_half_width=12.0)
        nd.assign_materials(mesh, base)
        before = list(mesh.material_id)
        nd.assign_materials(mesh, L.with_rods(base, [(0.0, 0.0)], 3.0, material=2))
        after = list(mesh.material_id)
        changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        assert changed
        assert all(after[i] == 2 for i in changed)


class TestBoundaryConditions:
    def test_layout_and_length(self):
        mesh = L.cartesian_mesh(width=20.0, h=2.0, orientation="quarter")
        bc = L.boundary_conditions(mesh, D=[1.4, 0.4])
        assert len(bc) == 2 * 2                      # two tags x two groups
        for g in range(2):
            assert (bc[L.SYMMETRY * 2 + g].A, bc[L.SYMMETRY * 2 + g].B) == (0.0, 1.0)
        assert bc[L.OUTER * 2 + 0].B == pytest.approx(0.7)
        assert bc[L.OUTER * 2 + 1].B == pytest.approx(0.2)

    def test_albedo_one_is_reflective(self):
        """A fully reflective outer surface reproduces the infinite medium."""
        mesh = L.cartesian_mesh(width=40.0, h=4.0, orientation="quarter")
        assert solve(mesh, albedo=1.0).keff == pytest.approx(NUSIGF / SIGR, rel=1e-9)

    def test_albedo_raises_keff(self):
        mesh = L.cartesian_mesh(width=40.0, h=2.0, orientation="quarter")
        k = [solve(mesh, albedo=a).keff for a in (0.0, 0.5, 0.9)]
        assert k[0] < k[1] < k[2]


class TestPeriodic:
    """Periodic cuts impose rotational symmetry without imposing a mirror.

    A reflective cut assumes the loading is mirror-symmetric about it.  A spiral
    or pinwheel pattern is not, and would be solved wrongly; joining the two cuts
    periodically imposes only the rotation.
    """

    def test_rotational_matches_mirror_on_a_symmetric_core(self):
        """A homogeneous core has both symmetries, so the two must agree."""
        for o in ("half", "quarter"):
            km = solve(L.cartesian_mesh(width=40.0, h=2.0, orientation=o)).keff
            kr = solve(L.cartesian_mesh(width=40.0, h=2.0, orientation=o,
                                        symmetry="rotational")).keff
            assert kr == pytest.approx(km, rel=1e-10)

    def test_hex_rotational_matches_mirror(self):
        for o in ("sector120", "sector60"):
            km = solve(L.hex_mesh(pitch=4.0, n_rings=5, orientation=o)).keff
            kr = solve(L.hex_mesh(pitch=4.0, n_rings=5, orientation=o,
                                  symmetry="rotational")).keff
            assert kr == pytest.approx(km, rel=1e-5)

    def test_rotational_has_no_symmetry_faces(self):
        mesh = L.cartesian_mesh(width=20.0, h=2.0, orientation="quarter",
                                symmetry="rotational")
        assert set(mesh.bface_bc_tag) == {L.OUTER}
        assert len(mesh.periodic_a0) > 0
        assert (len(mesh.periodic_a1) == len(mesh.periodic_b0)
                == len(mesh.periodic_b1) == len(mesh.periodic_a0))

    @pytest.mark.parametrize("gen,kwargs,orientation", [
        (L.cartesian_mesh, dict(width=20.0, h=2.0), "eighth"),
        (L.cartesian_mesh, dict(width=20.0, h=2.0), "full"),
        (L.hex_mesh, dict(pitch=4.0, n_rings=3), "sector30"),
        (L.hex_mesh, dict(pitch=4.0, n_rings=3), "half"),
        (L.cartesian_mesh, dict(width=20.0, h=2.0), "infinite"),
        (L.hex_mesh, dict(pitch=4.0, n_rings=3), "infinite"),
    ])
    def test_non_period_sectors_rejected(self, gen, kwargs, orientation):
        with pytest.raises(ValueError, match="rotational"):
            gen(orientation=orientation, symmetry="rotational", **kwargs)

    @pytest.mark.parametrize("orientation", ["quarter", "full", "infinite"])
    def test_unknown_symmetry_raises(self, orientation):
        """Checked before the orientation branches, so no case swallows it."""
        with pytest.raises(ValueError, match='"mirror" or "rotational"'):
            L.cartesian_mesh(width=20.0, h=2.0, orientation=orientation,
                             symmetry="glide")

    def test_meshes_validate(self):
        nd.validate_mesh(L.cartesian_mesh(width=20.0, h=2.0,
                                          orientation="quarter",
                                          symmetry="rotational"))
        nd.validate_mesh(L.hex_mesh(pitch=4.0, n_rings=4, orientation="sector60",
                                    symmetry="rotational"))


def periodic_grid(n, size=20.0, sides=("x", "y")):
    """n x n quad grid with the named directions joined periodically.

    Non-periodic sides all carry tag 0, so a single-entry ``bc`` suffices.
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
            cv += [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
            co.append(len(cv))
    a0, a1, b0, b1 = [], [], [], []
    bv0, bv1, bt = [], [], []
    for j in range(n):
        if "x" in sides:
            a0.append(vid(0, j)); a1.append(vid(0, j + 1))
            b0.append(vid(n, j)); b1.append(vid(n, j + 1))
        else:
            bv0 += [vid(0, j), vid(n, j)]
            bv1 += [vid(0, j + 1), vid(n, j + 1)]
            bt += [0, 0]
    for i in range(n):
        if "y" in sides:
            a0.append(vid(i, 0)); a1.append(vid(i + 1, 0))
            b0.append(vid(i, n)); b1.append(vid(i + 1, n))
        else:
            bv0 += [vid(i, 0), vid(i, n)]
            bv1 += [vid(i + 1, 0), vid(i + 1, n)]
            bt += [0, 0]

    m = nd.UnstructuredMesh2D()
    m.vx, m.vy = vx, vy
    m.cell_vertices, m.cell_offsets = cv, co
    m.material_id = [0] * (n * n)
    m.periodic_a0, m.periodic_a1 = a0, a1
    m.periodic_b0, m.periodic_b1 = b0, b1
    m.bface_v0, m.bface_v1, m.bface_bc_tag = bv0, bv1, bt
    return m


class TestTranslationalPeriodic:
    @pytest.mark.parametrize("n", [4, 8, 16])
    def test_fully_periodic_gives_k_inf(self, n):
        """No leakage anywhere, so the eigenvalue is the infinite-medium value."""
        bc = [nd.BoundaryCondition(A=1.0, B=0.0)]
        res = nd.KEigenSolverUnstructured2D(
            mats(), periodic_grid(n), bc, epsilon=1e-11, max_outer=5000,
            max_inner=500, use_cg=True, verbose=False).solve()
        assert res.keff == pytest.approx(NUSIGF / SIGR, rel=1e-10)

    def test_periodic_and_reflective_differ_on_an_asymmetric_loading(self):
        """Reflecting a stripe doubles it; repeating it does not."""
        stripes = L.checkerboard(pitch=10.0, materials=(0, 1), origin=(0.0, 0.0))
        two = mats(2, nusigf=[0.40, 0.20])
        bc = [nd.BoundaryCondition(A=1.0, B=0.0)]

        p = periodic_grid(16)
        nd.assign_materials(p, stripes)
        k_per = nd.KEigenSolverUnstructured2D(
            two, p, bc, epsilon=1e-11, max_outer=5000, max_inner=500,
            use_cg=True, verbose=False).solve().keff

        r = periodic_grid(16, sides=())
        nd.assign_materials(r, stripes)
        k_ref = nd.KEigenSolverUnstructured2D(
            two, r, [nd.BoundaryCondition(A=0.0, B=1.0)], epsilon=1e-11,
            max_outer=5000, max_inner=500, use_cg=True, verbose=False).solve().keff

        assert k_ref > k_per          # mirroring widens the high-yield stripe
        assert abs(k_ref - k_per) > 1e-2

    def test_mismatched_pair_lengths_raise(self):
        m = periodic_grid(4)
        m.periodic_b1 = list(m.periodic_b1)[:-1]
        with pytest.raises(ValueError, match="same length"):
            nd.validate_mesh(m)

    def test_pair_on_an_interior_edge_raises(self):
        """Both sides must be boundary edges; an interior one is already paired."""
        m = periodic_grid(4, sides=())
        # vid(i, j) = i * 5 + j, so (6, 7) and (11, 12) are interior verticals.
        m.periodic_a0, m.periodic_a1 = [6], [7]
        m.periodic_b0, m.periodic_b1 = [11], [12]
        with pytest.raises(ValueError, match="unpaired boundary edge"):
            nd.KEigenSolverUnstructured2D(mats(), m, [nd.BoundaryCondition(1.0, 0.0)])

    def test_incongruent_pair_raises(self):
        """Joined edges must be the same length to be images of each other."""
        m = periodic_grid(4, sides=())
        vy = list(m.vy)
        vy[2] = 3.0 * (20.0 / 4)          # stretch one left-boundary edge
        m.vy = vy
        m.periodic_a0, m.periodic_a1 = [0], [1]      # length h
        m.periodic_b0, m.periodic_b1 = [1], [2]      # now length 2h
        with pytest.raises(ValueError, match="different length"):
            nd.KEigenSolverUnstructured2D(mats(), m, [nd.BoundaryCondition(1.0, 0.0)])
