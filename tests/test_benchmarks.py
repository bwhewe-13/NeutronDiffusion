"""Published-benchmark regression tests.

Three two-group k-eigenvalue benchmarks from Yu et al., "Solving Multi-Group
Neutron Diffusion Eigenvalue Problem with Decoupling Residual Loss Function"
(arXiv:2411.15693), which reproduces classic reactor-physics benchmarks and
reports fine-mesh FreeFEM++ reference eigenvalues:

- 1-D Swedish Ringhals-4 core/reflector slab   keff = 1.0037  (Table 2, Fig. 4)
- 2-D TWIGL seed/blanket quarter core          keff = 0.9133  (Table 4, Fig. 6)
- 2-D IAEA PWR stepped quarter core            keff = 1.0296  (Table 6, Fig. 12)

plus one full-core benchmark from a different source:

- 2-D BIBLIS full-core PWR (8-composition checkerboard)  keff = 1.02535
  (FEMFFUSION validation report, Vidal-Ferrandiz et al. 2023; data originally
  Nakata & Martin, Nucl. Sci. Eng. 85 (1983)).

The model assumes fission neutrons are born fast: chi = (1, 0), with
down-scatter Sigma_1->2 only.  Removal_g = Sigma_a,g + out-scatter from g.

Each benchmark exercises a different solver: 1-D slab (symmetry at x=0), 2-D
structured (reflective left/bottom), and 2-D unstructured FVM (the IAEA stepped
domain and the BIBLIS full core are not rectangles the structured solver can
handle - IAEA uses per-tag Robin BCs, BIBLIS a full vacuum boundary).
"""

import numpy as np
import pytest

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
        mats = nd.materials.RINGHALS.materials()
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
        mats = nd.materials.TWIGL.materials()
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
      Sigma_a2 = 0.085, and the outer ring (gray) is fuel 1 with 0.080.
    - The canonical spec includes an axial buckling B2_z = 0.8e-4 cm^-2 in
      every region: removal_g += D_g * B2_z.
    """

    REF_KEFF = 1.0296
    B2_AXIAL = 0.8e-4

    def materials(self):
        return nd.materials.IAEA.materials()

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


# ----------------------------------------------------------------------------
# 2-D BIBLIS full-core PWR (8-composition checkerboard -> unstructured FVM)
# ----------------------------------------------------------------------------

# From the FEMFFUSION validation report (Vidal-Ferrandiz et al., 2023), Table 4
# and Fig. 4; cross sections originate in Nakata & Martin, Nucl. Sci. Eng. 85
# (1983).  257 homogenized assemblies (193 fuel + 64 reflector) of pitch
# 23.1226 cm on a 17x17 grid whose corners are void; vacuum on the whole outer
# boundary.  Numbers are taken from the FEMFFUSION input deck (biblis.xsec /
# biblis_SP1.prm), authoritative where it disagrees with the report's typeset
# Table 4 (e.g. material 1 Sigma_a2 = 0.0750058, not 0.0750580).

BIBLIS_PITCH = nd.materials.BIBLIS.pitch  # cm, homogenized assembly width

# Composition per assembly; 0 = void, 1..8 index the cross-section table
# (composition 3 is the non-fissile reflector).  The loading is symmetric, so row
# orientation does not affect keff.
BIBLIS_MAP = nd.materials.BIBLIS.assembly_map


def build_biblis_mesh(cells_per_assembly):
    """Cartesian quad mesh of the full BIBLIS core, cells_per_assembly per side.

    Void map entries are skipped; every exterior edge gets BC tag 0 (a single
    vacuum boundary - the full core has no symmetry axes to reflect on).  Same
    vertex-dedup + edge-count boundary extraction as build_iaea_mesh.
    """
    n = len(BIBLIS_MAP)
    h = BIBLIS_PITCH / cells_per_assembly
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
    for jrow in range(n):
        for icol in range(n):
            comp = BIBLIS_MAP[jrow][icol]
            if comp == 0:
                continue
            for dj in range(cells_per_assembly):
                for di in range(cells_per_assembly):
                    ix = icol * cells_per_assembly + di
                    jy = jrow * cells_per_assembly + dj
                    v = [vertex(ix, jy), vertex(ix + 1, jy),
                         vertex(ix + 1, jy + 1), vertex(ix, jy + 1)]
                    cell_vertices.extend(v)
                    cell_offsets.append(len(cell_vertices))
                    material_id.append(comp - 1)
                    for k in range(4):
                        a, b = v[k], v[(k + 1) % 4]
                        edge_count[(min(a, b), max(a, b))] = \
                            edge_count.get((min(a, b), max(a, b)), 0) + 1

    bface_v0, bface_v1, bface_tag = [], [], []
    for (a, b), count in edge_count.items():
        if count == 1:
            bface_v0.append(a)
            bface_v1.append(b)
            bface_tag.append(0)

    mesh = nd.UnstructuredMesh2D()
    mesh.vx, mesh.vy = vx, vy
    mesh.cell_vertices = cell_vertices
    mesh.cell_offsets = cell_offsets
    mesh.material_id = material_id
    mesh.bface_v0, mesh.bface_v1 = bface_v0, bface_v1
    mesh.bface_bc_tag = bface_tag
    return mesh


class TestBiblis2D:
    """2-D BIBLIS full-core PWR benchmark (unstructured FVM, vacuum boundary).

    A diffusion benchmark, so this diffusion solver reproduces the diffusion
    reference well: FEMFFUSION's mesh-converged SP1 (diffusion) keff is 1.02535
    (the higher-order Mueller-Weiss benchmark value is 1.02584).  This FVM
    solution converges toward the diffusion reference from above with mesh
    refinement: cells_per_assembly 2/4/6/8 give keff
    1.02865 / 1.02597 / 1.02553 / 1.02540.
    """

    REF_KEFF = 1.02535  # FEMFFUSION SP1 (diffusion), mesh-converged

    def test_keff(self):
        # Marshak vacuum on the whole outer boundary, using the reflector D
        # (composition 3) that borders the exterior.
        d1_refl, d2_refl = nd.materials.BIBLIS.rows[2][:2]
        bc = nd.boundary_conditions([d1_refl, d2_refl], alpha=0.0)
        solver = nd.KEigenSolverUnstructured2D(
            mats=nd.materials.BIBLIS.materials(),
            mesh=build_biblis_mesh(cells_per_assembly=6),
            bc=bc,
            epsilon=1e-7,
            max_outer=2000,
            max_inner=200,
            use_cg=True,
        )
        res = solver.solve()
        assert res.converged
        assert abs(res.keff - self.REF_KEFF) < 1e-3, res.keff


# ----------------------------------------------------------------------------
# 2-D TWIGL kinetics (delayed neutron precursors, step reactivity insertion)
# ----------------------------------------------------------------------------


class TestTwiglKinetics:
    """TWIGL seed/blanket transient on the geometry of TestTwigl2D.

    Cross sections, delayed data (one precursor group, beta = 0.0075,
    lambda = 0.08 1/s) and group velocities are the classic TWIGL kinetics set.
    The seed is split into the two symmetric arms (material 0, perturbed) and
    the central corner block (material 1); the blanket is material 2.

    This is an internal-consistency validation, **not** a comparison against
    published TWIGL power histories - the perturbation size here was chosen to
    give a $0.50 insertion under this code's own static reactivity, rather than
    copied from the benchmark specification.  What it pins down is the coupling
    between the spatial solver and the kinetics: the reactivity worth of the
    perturbation is computed independently from two k-eigenvalue solves, and
    the transient's prompt jump must match the point-kinetics value
    beta / (beta - rho) that worth implies.  That relation is independent of
    the generation time, so it needs no fitted parameter.

    The mesh is coarse (4 cm cells): the time behavior is what is under test,
    and TestTwigl2D already covers spatial accuracy at 1 cm.
    """

    BETA = 0.0075    # total delayed fraction
    LAMBDA = 0.08    # 1/s, single precursor group
    V_FAST, V_THERMAL = 1.0e7, 2.0e5  # cm/s
    DELTA_SA2 = -0.0015  # step change in seed-arm thermal absorption
    N = 20               # 4 cm cells; region edges 24/56 fall on cell edges

    def materials(self, delta_sa2=0.0):
        return two_group_materials([
            # D1   D2   Sa1    Sa2                  S12   nuSf1  nuSf2
            (1.4, 0.4, 0.010, 0.15 + delta_sa2, 0.01, 0.007, 0.20),  # seed arms
            (1.4, 0.4, 0.010, 0.15, 0.01, 0.007, 0.20),              # seed corner
            (1.3, 0.5, 0.008, 0.05, 0.01, 0.003, 0.06),              # blanket
        ])

    @staticmethod
    def region(x, y):
        if (x < 24 and 24 < y < 56) or (y < 24 and 24 < x < 56):
            return 0  # perturbed seed arms
        if 24 < x < 56 and 24 < y < 56:
            return 1  # seed corner
        return 2      # blanket

    def mesh(self):
        n, h = self.N, 80.0 / self.N
        edges = list(np.linspace(0.0, 80.0, n + 1))
        medium_map = [
            self.region((i + 0.5) * h, (j + 0.5) * h)
            for i in range(n) for j in range(n)
        ]
        return edges, medium_map

    def keff(self, mats, edges, medium_map):
        zero_flux = [nd.BoundaryCondition(A=1.0, B=0.0)] * 2
        solver = nd.KEigenSolver2D(
            mats=mats, medium_map=medium_map, edges_x=edges, edges_y=edges,
            geom=nd.Geometry2D.XY, bc_x=zero_flux, bc_y=zero_flux,
            epsilon=1e-9, max_outer=3000, max_inner=200, use_cg=True,
        )
        res = solver.solve()
        assert res.converged
        return res.keff, res.flux

    def transient(self, edges, medium_map, initial, delayed, mats):
        zero_flux = [nd.BoundaryCondition(A=1.0, B=0.0)] * 2
        kwargs = {} if delayed is None else {"delayed": delayed}
        return nd.TimeDependentSolver2D(
            mats=mats, medium_map=medium_map, edges_x=edges, edges_y=edges,
            geom=nd.Geometry2D.XY, bc_x=zero_flux, bc_y=zero_flux,
            initial_flux=initial, epsilon=1e-9, max_inner=200, **kwargs,
        )

    def setup_case(self):
        """Critical steady state plus the perturbed materials and their worth."""
        edges, medium_map = self.mesh()
        k0, flux0 = self.keff(self.materials(), edges, medium_map)

        # Scale by 1/k0 so the eigenvalue flux is an exact steady state; apply
        # the same scaling to the perturbed set so rho is the perturbation's
        # worth alone and not contaminated by the criticality offset.
        critical = nd.scale_to_critical(self.materials(), k0)
        perturbed = nd.scale_to_critical(self.materials(self.DELTA_SA2), k0)

        k1, _ = self.keff(perturbed, edges, medium_map)
        rho = (k1 - 1.0) / k1

        delayed = nd.make_delayed_data(
            {"Beta": [self.BETA], "Lambda": [self.LAMBDA]},
            G=2, n_mat=3, chi=[1.0, 0.0],
        )
        for m in (critical, perturbed):
            m.velocity = [self.V_FAST, self.V_THERMAL]
        return edges, medium_map, flux0, critical, perturbed, rho, delayed

    def test_step_insertion(self):
        (edges, medium_map, flux0, critical, perturbed, rho,
         delayed) = self.setup_case()

        # The perturbation is a sub-prompt-critical insertion.
        assert rho / self.BETA == pytest.approx(0.50, abs=0.02)

        # Unperturbed: an exact fixed point, even at 10 ms steps.
        steady = self.transient(edges, medium_map, flux0, delayed, critical)
        p0 = float(np.sum(steady.result().flux))
        steady.run(1e-2, 5)
        assert float(np.sum(steady.result().flux)) / p0 == pytest.approx(1.0, abs=1e-8)

        # Step insertion at t = 0.
        solver = self.transient(edges, medium_map, flux0, delayed, critical)
        solver.update_materials(perturbed)

        solver.run(2e-3, 50)  # t = 0.1 s: prompt jump complete
        power_jump = float(np.sum(solver.result().flux)) / p0
        assert power_jump == pytest.approx(self.BETA / (self.BETA - rho), rel=0.02)

        solver.run(2e-3, 200)  # t = 0.5 s
        power_late = float(np.sum(solver.result().flux)) / p0
        # Delayed timescale: still climbing, but only by a few percent over the
        # following 0.4 s - not the orders of magnitude a prompt excursion gives.
        assert power_late > power_jump
        assert power_late / power_jump < 1.1

    def test_prompt_only_runs_away(self):
        """The same insertion without precursors is a prompt excursion.

        This is the defect the delayed treatment fixes, asserted as a
        same-time comparison: at t = 0.05 s the delayed transient has settled
        near its prompt jump (~2x), while the prompt-only one is already ~150x
        and climbing on the ~3.8e-5 s generation time.
        """
        (edges, medium_map, flux0, critical, perturbed, _rho,
         delayed) = self.setup_case()

        powers = {}
        for label, delayed_data in (("delayed", delayed), ("prompt", None)):
            solver = self.transient(
                edges, medium_map, flux0, delayed_data, critical
            )
            p0 = float(np.sum(solver.result().flux))
            solver.update_materials(perturbed)
            solver.run(5e-4, 100)  # t = 0.05 s
            powers[label] = float(np.sum(solver.result().flux)) / p0

        assert powers["delayed"] == pytest.approx(2.0, rel=0.05)
        assert powers["prompt"] / powers["delayed"] > 10.0
