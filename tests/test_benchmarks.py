"""Published-benchmark regression tests.

Three two-group k-eigenvalue benchmarks from Yu et al., "Solving Multi-Group
Neutron Diffusion Eigenvalue Problem with Decoupling Residual Loss Function"
(arXiv:2411.15693), which reproduces classic reactor-physics benchmarks and
reports fine-mesh FreeFEM++ reference eigenvalues:

- 1-D Swedish Ringhals-4 core/reflector slab   keff = 1.0037  (Table 2, Fig. 4)
- 2-D TWIGL seed/blanket quarter core          keff = 0.9133  (Table 4, Fig. 6)
- 2-D IAEA PWR stepped quarter core            keff = 1.0296  (Table 6, Fig. 12)

The model assumes fission neutrons are born fast: chi = (1, 0), with
down-scatter Sigma_1->2 only.  Removal_g = Sigma_a,g + out-scatter from g.

Each benchmark exercises a different solver: 1-D slab (symmetry at x=0), 2-D
structured (reflective left/bottom), and 2-D unstructured FVM (the IAEA
stepped domain is not a rectangle, and uses per-tag Robin BCs).
"""

import numpy as np

import ndiffusion as nd


def two_group_materials(data):
    """Build Materials from rows of (D1, D2, Sa1, Sa2, S12, nuSf1, nuSf2)."""
    m = nd.Materials()
    m.n_mat = len(data)
    m.n_groups = 2
    D, removal, scatter, chi, nusigf = [], [], [], [], []
    for d1, d2, sa1, sa2, s12, nsf1, nsf2 in data:
        D += [d1, d2]
        removal += [sa1 + s12, sa2]
        scatter += [0.0, 0.0, s12, 0.0]  # scatter[g_to][g_from]; only 1->2
        chi += [1.0, 0.0]
        nusigf += [nsf1, nsf2]
    m.D, m.removal, m.scatter, m.chi, m.nusigf = D, removal, scatter, chi, nusigf
    return m


class TestRinghals1D:
    """1-D Ringhals-4: reflector | core | reflector on [-279.5, 279.5] cm.

    Symmetric about x = 0, so the half domain [0, 279.5] matches the solver's
    built-in zero-gradient left edge.  Robin BC dphi/dn = -(c/D) phi with
    c = 0.5 maps to A = 0.5, B = D_g (reflector D at the outer surface).
    """

    A = 279.5   # half-width of the domain
    B = 161.25  # half-width of the core
    REF_KEFF = 1.0037

    def test_keff(self):
        mats = two_group_materials([
            # D1      D2      Sa1      Sa2     S12     nuSf1   nuSf2
            (1.4376, 0.3723, 0.0115, 0.1019, 0.0151, 0.0057, 0.1425),  # core
            (1.3116, 0.2624, -0.0098, 0.0284, 0.0238, 0.0, 0.0),       # reflector
        ])
        dx = 0.25
        cells = int(round(self.A / dx))
        edges = list(np.linspace(0.0, self.A, cells + 1))
        medium_map = nd.make_medium_map(
            [(0, self.B), (1, self.A - self.B)], edges=edges)

        solver = nd.KEigenSolver(
            mats=mats,
            medium_map=medium_map,
            edges_x=edges,
            geom=nd.Geometry.Slab,
            bc=[nd.BoundaryCondition(A=0.5, B=1.3116),
                nd.BoundaryCondition(A=0.5, B=0.2624)],
            epsilon=1e-8,
            max_outer=2000,
            max_inner=100,
        )
        res = solver.solve()
        assert res.converged
        assert abs(res.keff - self.REF_KEFF) < 5e-4, res.keff


class TestTwigl2D:
    """2-D TWIGL quarter core, 80 x 80 cm.

    Seed (mat 0) occupies [0,56]x[24,56] and [24,56]x[0,24]; blanket (mat 1)
    fills the rest.  Reflective on x=0 / y=0 (hardcoded), zero flux on the
    outer boundary (matches the benchmark's homogeneous Dirichlet).
    """

    REF_KEFF = 0.9133

    @staticmethod
    def in_seed(x, y):
        return (y > 24 and y < 56 and x < 56) or (x > 24 and x < 56 and y < 24)

    def test_keff(self):
        mats = two_group_materials([
            # D1   D2   Sa1    Sa2   S12   nuSf1  nuSf2
            (1.4, 0.4, 0.010, 0.15, 0.01, 0.007, 0.20),  # seed
            (1.3, 0.5, 0.008, 0.05, 0.01, 0.003, 0.06),  # blanket
        ])
        n = 80  # 1 cm cells; region boundaries at 24/56 fall on cell edges
        edges = list(np.linspace(0.0, 80.0, n + 1))
        medium_map = [0] * (n * n)
        for i in range(n):
            for j in range(n):
                xc, yc = i + 0.5, j + 0.5
                medium_map[i * n + j] = 0 if self.in_seed(xc, yc) else 1

        zero_flux = [nd.BoundaryCondition(A=1.0, B=0.0)] * 2
        solver = nd.KEigenSolver2D(
            mats=mats,
            medium_map=medium_map,
            edges_x=edges,
            edges_y=edges,
            geom=nd.Geometry2D.XY,
            bc_x=zero_flux,
            bc_y=zero_flux,
            epsilon=1e-7,
            max_outer=1000,
            max_inner=200,
            use_cg=True,
        )
        res = solver.solve()
        assert res.converged
        assert abs(res.keff - self.REF_KEFF) < 1e-3, res.keff


# ----------------------------------------------------------------------------
# 2-D IAEA PWR quarter core (stepped domain -> unstructured FVM solver)
# ----------------------------------------------------------------------------

# Block edges (cm) and the 9x9 region map read from Fig. 12 (octant symmetric).
# Region codes follow the paper's Table 6: 1 = outer fuel, 2 = inner fuel,
# 3 = fuel + rod, 4 = reflector; 0 = outside the stepped domain.
IAEA_EDGES = [0, 10, 30, 50, 70, 90, 110, 130, 150, 170]
IAEA_MAP = [  # [row j][col i], row 0 at the bottom
    [3, 2, 2, 2, 3, 2, 2, 1, 4],
    [2, 2, 2, 2, 2, 2, 2, 1, 4],
    [2, 2, 2, 2, 2, 2, 1, 1, 4],
    [2, 2, 2, 2, 2, 2, 1, 4, 4],
    [3, 2, 2, 2, 3, 1, 1, 4, 0],
    [2, 2, 2, 2, 1, 1, 4, 4, 0],
    [2, 2, 1, 1, 1, 4, 4, 0, 0],
    [1, 1, 1, 4, 4, 4, 0, 0, 0],
    [4, 4, 4, 4, 0, 0, 0, 0, 0],
]


def iaea_region(x, y):
    """Region code at point (x, y), or 0 outside the stepped domain."""
    i = int(np.searchsorted(IAEA_EDGES, x, side="right")) - 1
    j = int(np.searchsorted(IAEA_EDGES, y, side="right")) - 1
    if i < 0 or j < 0 or i > 8 or j > 8:
        return 0
    return IAEA_MAP[j][i]


def build_iaea_mesh(h):
    """Cartesian quad mesh of the stepped quarter core with cell size h.

    Boundary tags: 0 = symmetry edges on x=0 / y=0 (reflective),
    1 = external stepped boundary (Robin).
    """
    n = int(round(170.0 / h))
    vid = {}
    vx, vy = [], []

    def vertex(ix, jy):
        key = (ix, jy)
        if key not in vid:
            vid[key] = len(vx)
            vx.append(ix * h)
            vy.append(jy * h)
        return vid[key]

    cell_vertices, cell_offsets, material_id = [], [0], []
    edge_count = {}
    for i in range(n):
        for j in range(n):
            reg = iaea_region((i + 0.5) * h, (j + 0.5) * h)
            if reg == 0:
                continue
            v = [vertex(i, j), vertex(i + 1, j),
                 vertex(i + 1, j + 1), vertex(i, j + 1)]
            cell_vertices.extend(v)
            cell_offsets.append(len(cell_vertices))
            material_id.append(reg - 1)
            for k in range(4):
                a, b = v[k], v[(k + 1) % 4]
                edge_count[(min(a, b), max(a, b))] = \
                    edge_count.get((min(a, b), max(a, b)), 0) + 1

    bface_v0, bface_v1, bface_tag = [], [], []
    for (a, b), count in edge_count.items():
        if count != 1:
            continue
        on_axis = (vx[a] == 0.0 and vx[b] == 0.0) or \
                  (vy[a] == 0.0 and vy[b] == 0.0)
        bface_v0.append(a)
        bface_v1.append(b)
        bface_tag.append(0 if on_axis else 1)

    mesh = nd.UnstructuredMesh2D()
    mesh.vx, mesh.vy = vx, vy
    mesh.cell_vertices = cell_vertices
    mesh.cell_offsets = cell_offsets
    mesh.material_id = material_id
    mesh.bface_v0, mesh.bface_v1 = bface_v0, bface_v1
    mesh.bface_bc_tag = bface_tag
    return mesh


class TestIaea2D:
    """2-D IAEA PWR benchmark on the true stepped domain (unstructured FVM).

    Robin BC dphi/dn = -(c/D) phi with c = 0.4692 on the external boundary
    (A = c, B = D_g of the reflector, which borders the entire exterior);
    reflective on the x=0 / y=0 symmetry edges.

    Two corrections to the paper's literal Table 6 / figure legend, both
    required to reproduce its own reference keff = 1.0296 (canonical value
    1.029585; without them this setup yields 1.0340 and 1.0890 respectively):

    - The two fuel absorptions are swapped relative to the canonical
      benchmark: the *inner* zone (white, hosting the rods) is fuel 2 with
      Sigma_a2 = 0.085, and the outer ring (grey) is fuel 1 with 0.080.
    - The canonical spec includes an axial buckling B2_z = 0.8e-4 cm^-2 in
      every region: removal_g += D_g * B2_z.
    """

    REF_KEFF = 1.0296
    B2_AXIAL = 0.8e-4

    def materials(self):
        rows = [
            # D1   D2   Sa1   Sa2    S12   nuSf1  nuSf2
            (1.5, 0.4, 0.01, 0.080, 0.02, 0.0, 0.135),  # grey ring: fuel 1
            (1.5, 0.4, 0.01, 0.085, 0.02, 0.0, 0.135),  # white inner: fuel 2
            (1.5, 0.4, 0.01, 0.130, 0.02, 0.0, 0.135),  # red: fuel 2 + rod
            (2.0, 0.3, 0.00, 0.010, 0.04, 0.0, 0.0),    # blue: reflector
        ]
        mats = two_group_materials(rows)
        removal = list(mats.removal)
        for m, (d1, d2, *_rest) in enumerate(rows):
            removal[2 * m] += d1 * self.B2_AXIAL
            removal[2 * m + 1] += d2 * self.B2_AXIAL
        mats.removal = removal
        return mats

    def solve(self, h):
        mesh = build_iaea_mesh(h)
        cbou = 0.4692
        bc = [
            nd.BoundaryCondition(A=0.0, B=1.0),    # tag 0, g=0: reflective
            nd.BoundaryCondition(A=0.0, B=1.0),    # tag 0, g=1
            nd.BoundaryCondition(A=cbou, B=2.0),   # tag 1, g=0: Robin, D1(refl)
            nd.BoundaryCondition(A=cbou, B=0.3),   # tag 1, g=1: Robin, D2(refl)
        ]
        solver = nd.KEigenSolverUnstructured2D(
            mats=self.materials(),
            mesh=mesh,
            bc=bc,
            epsilon=1e-7,
            max_outer=1000,
            max_inner=200,
            use_cg=True,
        )
        return solver.solve()

    def test_keff(self):
        res = self.solve(h=2.5)
        assert res.converged
        # h=2.5 gives 1.02943; h=1.25 gives 1.02954 -> extrapolates to the
        # canonical 1.029585.
        assert abs(res.keff - self.REF_KEFF) < 1e-3, res.keff
