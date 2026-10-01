"""Named cross-section sets and the published benchmarks that carry them.

The benchmark tables are the same data `test_benchmarks.py` solves to its
published eigenvalues, so these tests cover the builders and the bundling; the
numbers themselves are pinned there.
"""

import numpy as np
import pytest

import ndiffusion as nd
from ndiffusion import materials as M


class TestBuilders:
    def test_one_group_scalar(self):
        m = M.one_group(D=1.2, sigma_a=0.05, nusigf=0.3)
        assert (m.n_mat, m.n_groups) == (1, 1)
        assert list(m.D) == [1.2]
        assert list(m.removal) == [0.05]
        assert list(m.nusigf) == [0.3]

    def test_one_group_broadcasts(self):
        m = M.one_group(D=1.0, sigma_a=0.1, nusigf=0.2, n_mat=3)
        assert m.n_mat == 3
        assert list(m.D) == [1.0] * 3

    def test_one_group_per_material(self):
        m = M.one_group(D=[1.0, 2.0], sigma_a=[0.1, 0.2], nusigf=[0.3, 0.0], n_mat=2)
        assert list(m.D) == [1.0, 2.0]
        assert list(m.nusigf) == [0.3, 0.0]

    def test_two_group_conventions(self):
        m = M.two_group([(1.4, 0.4, 0.010, 0.15, 0.01, 0.007, 0.20)])
        assert (m.n_mat, m.n_groups) == (1, 2)
        assert list(m.chi) == [1.0, 0.0]                  # born fast
        assert list(m.removal) == pytest.approx([0.020, 0.15])   # Sa + out-scatter
        assert list(m.scatter) == [0.0, 0.0, 0.01, 0.0]   # [g_to][g_from], 1->2
        assert list(m.nusigf) == pytest.approx([0.007, 0.20])

    def test_axial_buckling_adds_leakage(self):
        row = [(1.5, 0.4, 0.01, 0.08, 0.02, 0.0, 0.135)]
        plain = M.two_group(row)
        bucked = M.two_group(row, axial_buckling=0.8e-4)
        assert list(bucked.removal) == pytest.approx(
            [plain.removal[0] + 1.5 * 0.8e-4, plain.removal[1] + 0.4 * 0.8e-4])

    def test_two_group_rejects_wrong_row(self):
        with pytest.raises(ValueError, match="expected 7"):
            M.two_group([(1.0, 2.0, 3.0)])

    def test_builders_pass_solver_validation(self):
        mesh_mats = M.two_group([(1.4, 0.4, 0.01, 0.15, 0.01, 0.007, 0.20)])
        nd.KEigenSolver(mesh_mats, [0] * 10, list(np.linspace(0, 10, 11)),
                        nd.Geometry.Slab, [nd.BoundaryCondition(1.0, 0.0)] * 2)


class TestAssemblyMap:
    MAP = [
        [0, 2, 0],
        [2, 1, 2],
        [0, 2, 0],
    ]

    def test_reads_center_and_edges(self):
        paint = M.from_assembly_map(self.MAP, pitch=10.0)
        assert paint(0.0, 0.0) == 0          # center entry 1 -> index 0
        assert paint(0.0, 10.0) == 1         # entry 2 -> index 1
        assert paint(-10.0, 10.0) == 0       # void -> 0

    def test_row_zero_is_the_top(self):
        paint = M.from_assembly_map([[1, 1, 1], [2, 2, 2], [1, 1, 1]], pitch=10.0)
        assert paint(0.0, 10.0) == 0         # top row, entry 1
        assert paint(0.0, 0.0) == 1          # middle row, entry 2

    def test_outside_the_map(self):
        paint = M.from_assembly_map(self.MAP, pitch=10.0)
        assert paint(1000.0, 0.0) == 0

    def test_ragged_map_raises(self):
        with pytest.raises(ValueError, match="same length"):
            M.from_assembly_map([[1, 2], [1]], pitch=1.0)


class TestBenchmarks:
    @pytest.mark.parametrize("key", ["ringhals", "twigl", "iaea", "biblis"])
    def test_registry_entries(self, key):
        b = M.BENCHMARKS[key]
        assert b.reference_keff > 0.0
        assert b.source
        mats = b.materials()
        assert mats.n_groups == 2
        assert mats.n_mat == len(b.rows)

    def test_iaea_carries_its_buckling(self):
        """The axial leakage is part of the benchmark, not the caller's job."""
        plain = M.two_group(M.IAEA.rows)
        assert list(M.IAEA.materials().removal) != pytest.approx(list(plain.removal))

    def test_biblis_layout_is_a_painter(self):
        paint = M.BIBLIS.layout()
        assert paint(0.0, 0.0) in range(8)
        far = M.BIBLIS.pitch * 20
        assert paint(far, far) == 0            # outside the map

    def test_biblis_map_covers_every_composition(self):
        used = {v for row in M.BIBLIS.assembly_map for v in row if v != 0}
        assert used == set(range(1, len(M.BIBLIS.rows) + 1))

    @pytest.mark.parametrize("name", ["RINGHALS", "TWIGL", "IAEA"])
    def test_shape_benchmarks_have_no_map(self, name):
        with pytest.raises(ValueError, match="no assembly map"):
            getattr(M, name).layout()

    def test_benchmark_materials_solve(self):
        """A published table on an unrelated mesh: materials and geometry are free."""
        from ndiffusion import layouts as L

        mesh = L.cartesian_mesh(width=200.0, h=10.0, orientation="quarter")
        nd.assign_materials(mesh, L.homogeneous(0))
        bc = L.boundary_conditions(mesh, D=M.TWIGL.rows[0][:2])
        res = nd.KEigenSolverUnstructured2D(
            M.TWIGL.materials(), mesh, bc, epsilon=1e-9, max_outer=2000,
            max_inner=300, use_cg=True, verbose=False).solve()
        assert res.converged
        assert 0.5 < res.keff < 2.0
