"""Post-processing helpers: volumes, reaction rates, power and save/load.

Volumes are checked against the solvers through a neutron balance - with every
edge reflective, absorption integrated with cell_volumes has to equal the
integrated source, which only holds if they are the volumes the solver used.
Power normalization and peaking are checked against the analytic bare slab.
"""

import numpy as np
import pytest

import ndiffusion as nd

REFLECTIVE = nd.BoundaryCondition(A=0.0, B=1.0)
ZERO_FLUX = nd.BoundaryCondition(A=1.0, B=0.0)


def two_group():
    m = nd.Materials()
    m.n_mat = 2
    m.n_groups = 2
    m.D = [1.4, 0.4, 1.2, 0.3]
    m.removal = [0.03, 0.1, 0.04, 0.02]
    m.scatter = [0.0, 0.0, 0.02, 0.0,
                 0.0, 0.0, 0.035, 0.0]
    m.chi = [1.0, 0.0, 0.0, 0.0]
    m.nusigf = [0.005, 0.12, 0.0, 0.0]
    m.velocity = [2.0e7, 2.2e5]
    return m


def as_fission_matrix(m):
    out = nd.Materials()
    out.n_mat, out.n_groups = m.n_mat, m.n_groups
    out.D, out.removal, out.scatter = m.D, m.removal, m.scatter
    chi = m.chi.reshape(m.n_mat, m.n_groups)
    nusigf = m.nusigf.reshape(m.n_mat, m.n_groups)
    out.nusigf = np.einsum("mi,mj->mij", chi, nusigf)
    out.chi = np.zeros(m.n_mat * m.n_groups)
    return out


def unit_square_mesh(n=3):
    h = 1.0 / n
    vx, vy, cv, co = [], [], [], [0]
    for i in range(n + 1):
        for j in range(n + 1):
            vx.append(i * h)
            vy.append(j * h)
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            cv += [a, a + n + 1, a + n + 2, a + 1]
            co.append(len(cv))
    mesh = nd.UnstructuredMesh2D()
    mesh.vx, mesh.vy, mesh.cell_vertices, mesh.cell_offsets = vx, vy, cv, co
    mesh.material_id = [0] * (n * n)
    return mesh


class TestCellVolumes:
    def test_slab(self):
        np.testing.assert_allclose(nd.cell_volumes([0.0, 1.0, 3.0]), [1.0, 2.0])

    def test_slab_is_the_default(self):
        np.testing.assert_array_equal(
            nd.cell_volumes([0.0, 1.0, 3.0]),
            nd.cell_volumes([0.0, 1.0, 3.0], nd.Geometry.Slab))

    def test_cylinder(self):
        np.testing.assert_allclose(
            nd.cell_volumes([0.0, 1.0, 2.0], nd.Geometry.Cylinder),
            [np.pi, 3.0 * np.pi])

    def test_sphere(self):
        np.testing.assert_allclose(
            nd.cell_volumes([0.0, 1.0, 2.0], nd.Geometry.Sphere),
            [4.0 / 3.0 * np.pi, 28.0 / 3.0 * np.pi])

    def test_xy_is_row_major(self):
        vol = nd.cell_volumes([0.0, 1.0, 3.0], nd.Geometry2D.XY, [0.0, 0.5, 2.0, 3.0])
        np.testing.assert_allclose(vol.reshape(2, 3),
                                   [[0.5, 1.5, 1.0], [1.0, 3.0, 2.0]])

    def test_rz(self):
        # x is the axial z, y the radius.
        vol = nd.cell_volumes([0.0, 2.0], nd.Geometry2D.RZ, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(vol, [2.0 * np.pi, 6.0 * np.pi])

    def test_unstructured_mesh(self):
        mesh = unit_square_mesh()
        np.testing.assert_array_equal(nd.cell_volumes(mesh), nd.cell_areas(mesh))

    def test_2d_geometry_needs_edges_y(self):
        with pytest.raises(ValueError, match="edges_y"):
            nd.cell_volumes([0.0, 1.0], nd.Geometry2D.XY)

    def test_edges_y_needs_2d_geometry(self):
        with pytest.raises(ValueError, match="Geometry2D"):
            nd.cell_volumes([0.0, 1.0], nd.Geometry.Slab, [0.0, 1.0])


def balance(flux, mats, medium_map, vol, source):
    absorbed = nd.reaction_rate(flux, mats, medium_map, "absorption") @ vol
    return absorbed, np.asarray(source).sum(axis=1) @ vol


class TestVolumesMatchSolver:
    """All-reflective fixed source: absorption equals the source in total."""

    @pytest.mark.parametrize("geom", [nd.Geometry.Slab, nd.Geometry.Cylinder,
                                      nd.Geometry.Sphere])
    def test_1d(self, geom):
        mats = two_group()
        edges = np.array([0.0, 0.5, 1.5, 2.0, 4.0, 4.5, 7.0])
        medium_map = [0, 0, 1, 1, 0, 1]
        source = np.zeros((6, 2))
        source[1, 0] = 2.0
        source[4, 1] = 1.0
        solver = nd.FixedSourceSolver(mats, medium_map, edges, geom,
                                      [REFLECTIVE] * 2, epsilon=1e-13, max_inner=20000)
        flux = solver.solve(source).flux
        absorbed, emitted = balance(flux, mats, medium_map,
                                    nd.cell_volumes(edges, geom), source)
        assert absorbed == pytest.approx(emitted, rel=1e-8)

    @pytest.mark.parametrize("geom", [nd.Geometry2D.XY, nd.Geometry2D.RZ])
    def test_2d(self, geom):
        mats = two_group()
        ex = np.array([0.0, 1.0, 1.5, 3.0])
        ey = np.array([0.0, 0.5, 2.0, 2.5, 4.0])
        medium_map = [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0]
        source = np.zeros((12, 2))
        source[5, 0] = 1.0
        source[10, 1] = 3.0
        solver = nd.FixedSourceSolver2D(mats, medium_map, ex, ey, geom,
                                        [REFLECTIVE] * 2, [REFLECTIVE] * 2,
                                        epsilon=1e-13, max_inner=20000)
        flux = solver.solve(source).flux
        absorbed, emitted = balance(flux, mats, medium_map,
                                    nd.cell_volumes(ex, geom, ey), source)
        assert absorbed == pytest.approx(emitted, rel=1e-8)


class TestReactionRate:
    flux = np.array([[2.0, 1.0], [4.0, 3.0], [1.0, 1.0]])
    medium_map = [0, 0, 1]

    def test_absorption_subtracts_out_scatter(self):
        rate = nd.reaction_rate(self.flux, two_group(), self.medium_map,
                                "absorption", by_group=True)
        # absorption: material 0 (0.01, 0.1), material 1 (0.005, 0.02)
        np.testing.assert_allclose(rate, [[0.02, 0.1], [0.04, 0.3], [0.005, 0.02]])

    def test_removal(self):
        rate = nd.reaction_rate(self.flux, two_group(), self.medium_map, "removal")
        np.testing.assert_allclose(rate, [0.16, 0.42, 0.06])

    def test_nu_fission(self):
        rate = nd.reaction_rate(self.flux, two_group(), self.medium_map)
        np.testing.assert_allclose(rate, [0.13, 0.38, 0.0])

    def test_fission_divides_by_nu(self):
        mats = two_group()
        nu_fission = nd.reaction_rate(self.flux, mats, self.medium_map)
        fission = nd.reaction_rate(self.flux, mats, self.medium_map, "fission", nu=2.5)
        np.testing.assert_allclose(fission, nu_fission / 2.5)

    def test_fission_matrix_mode(self):
        mats = two_group()
        np.testing.assert_allclose(
            nd.reaction_rate(self.flux, as_fission_matrix(mats), self.medium_map),
            nd.reaction_rate(self.flux, mats, self.medium_map))

    def test_flat_flux(self):
        np.testing.assert_array_equal(
            nd.reaction_rate(self.flux.ravel(), two_group(), self.medium_map),
            nd.reaction_rate(self.flux, two_group(), self.medium_map))

    def test_unknown_kind(self):
        with pytest.raises(ValueError, match="kind"):
            nd.reaction_rate(self.flux, two_group(), self.medium_map, "capture")

    def test_material_map_length(self):
        with pytest.raises(ValueError, match="material_map"):
            nd.reaction_rate(self.flux, two_group(), [0, 0])

    def test_material_id_out_of_range(self):
        with pytest.raises(ValueError, match="material_map"):
            nd.reaction_rate(self.flux, two_group(), [0, 0, 2])

    def test_transposed_flux(self):
        with pytest.raises(ValueError, match="columns"):
            nd.reaction_rate(self.flux.T, two_group(), [0, 0])


# One-group bare half slab [0, a/2]: reflective at x = 0, zero flux at a/2, so
# the fundamental mode is cos(pi x / a).
HALF_WIDTH = 50.0
CELLS = 400
SIGMA_F = 0.01


@pytest.fixture(scope="module")
def half_slab():
    mats = nd.materials.one_group(D=1.0, sigma_a=0.02, nusigf=nd.NU_U235 * SIGMA_F)
    edges = np.linspace(0.0, HALF_WIDTH, CELLS + 1)
    medium_map = [0] * CELLS
    solver = nd.KEigenSolver(mats, medium_map, edges, nd.Geometry.Slab, [ZERO_FLUX],
                             epsilon=1e-12, max_outer=5000, max_inner=1000)
    res = solver.solve()
    return mats, medium_map, edges, res


class TestPower:
    def test_normalized_power_integrates_to_total(self, half_slab):
        mats, medium_map, edges, res = half_slab
        vol = nd.cell_volumes(edges)
        flux = nd.normalize_to_power(res.flux, mats, medium_map, vol, 1.0e6)
        assert flux.shape == res.flux.shape
        q = nd.power_density(flux, mats, medium_map)
        assert q @ vol == pytest.approx(1.0e6, rel=1e-12)

    def test_peak_flux_matches_analytic_slab(self, half_slab):
        # P = kappa Sigma_f phi0 * integral_0^{a/2} cos(pi x / a) dx
        #   = kappa Sigma_f phi0 a / pi
        mats, medium_map, edges, res = half_slab
        power = 1.0e6
        flux = nd.normalize_to_power(res.flux, mats, medium_map,
                                     nd.cell_volumes(edges), power)
        phi0 = power * np.pi / (nd.KAPPA_U235 * SIGMA_F * 2.0 * HALF_WIDTH)
        assert flux.max() == pytest.approx(phi0, rel=1e-4)

    def test_kappa_and_nu(self, half_slab):
        mats, medium_map, _, res = half_slab
        np.testing.assert_allclose(
            nd.power_density(res.flux, mats, medium_map, kappa=3.0, nu=1.5),
            3.0 * nd.reaction_rate(res.flux, mats, medium_map) / 1.5)

    def test_no_fission(self):
        mats = nd.materials.one_group(D=1.0, sigma_a=0.1)
        with pytest.raises(ValueError, match="no fission power"):
            nd.normalize_to_power(np.ones((3, 1)), mats, [0] * 3, np.ones(3), 1.0)

    def test_volumes_length(self, half_slab):
        mats, medium_map, _, res = half_slab
        with pytest.raises(ValueError, match="volumes"):
            nd.normalize_to_power(res.flux, mats, medium_map, np.ones(3), 1.0)


class TestRegionsAndPeaking:
    density = np.array([1.0, 3.0, 2.0, 0.0])
    volumes = np.array([1.0, 1.0, 2.0, 4.0])

    def test_region_powers(self):
        np.testing.assert_allclose(
            nd.region_powers(self.density, self.volumes, [0, 1, 1, 2]), [1.0, 7.0, 0.0])

    def test_region_powers_length(self):
        out = nd.region_powers(self.density, self.volumes, [0, 0, 0, 0], n_regions=3)
        np.testing.assert_allclose(out, [8.0, 0.0, 0.0])

    def test_region_powers_mismatch(self):
        with pytest.raises(ValueError, match="one entry per cell"):
            nd.region_powers(self.density, self.volumes, [0, 1])

    def test_uniform_is_one(self):
        assert nd.peaking_factors(np.full(5, 2.0), np.arange(1.0, 6.0)) == pytest.approx(1.0)

    def test_cell_peaking_ignores_unfueled_cells(self):
        # fueled average = (1 + 3 + 4) / 4 = 2
        assert nd.peaking_factors(self.density, self.volumes) == pytest.approx(1.5)

    def test_region_peaking(self):
        # region averages 1, 7/3, 0 over volumes 1, 3, 4 -> fueled average 2
        factor = nd.peaking_factors(self.density, self.volumes, [0, 1, 1, 2])
        assert factor == pytest.approx((7.0 / 3.0) / 2.0)

    def test_skipped_region_ids(self):
        assert nd.peaking_factors(self.density, self.volumes, [0, 3, 3, 5]) == \
            pytest.approx((7.0 / 3.0) / 2.0)

    def test_half_slab_peaking(self, half_slab):
        # max / average of cos(pi x / a) over [0, a/2] is pi / 2
        mats, medium_map, edges, res = half_slab
        q = nd.power_density(res.flux, mats, medium_map)
        assert nd.peaking_factors(q, nd.cell_volumes(edges)) == \
            pytest.approx(np.pi / 2.0, rel=1e-4)

    def test_no_power(self):
        with pytest.raises(ValueError, match="no cell"):
            nd.peaking_factors(np.zeros(3), np.ones(3))


class TestSaveLoad:
    def test_k_eigenvalue_round_trip(self, half_slab, tmp_path):
        mats, medium_map, edges, res = half_slab
        solver = nd.KEigenSolver(mats, medium_map, edges, nd.Geometry.Slab, [ZERO_FLUX])
        path = tmp_path / "slab.npz"
        nd.save_result(path, res, solver=solver, case="half slab", power=1.0e6)
        out = nd.load_result(path)
        np.testing.assert_array_equal(out.flux, res.flux)
        assert out.keff == res.keff
        assert out.iterations == res.iterations
        assert out.converged is res.converged
        assert out.metadata == {
            "ndiffusion_version": nd.__version__,
            "result_type": "DiffusionResult",
            "solver": "KEigenSolver",
            "n_cells": CELLS,
            "n_groups": 1,
            "case": "half slab",
            "power": 1.0e6,
        }

    def test_transient_round_trip(self, tmp_path):
        mats = two_group()
        absorber = dict(nd.DELAYED_U235_6GROUP, Beta=[0.0] * 6, ChiDelayed=[1.0, 0.0])
        delayed = nd.make_delayed_data([nd.DELAYED_U235_6GROUP, absorber], 2,
                                       chi=mats.chi)
        solver = nd.TimeDependentSolver(mats, [0, 1, 0, 1], np.linspace(0.0, 8.0, 5),
                                        nd.Geometry.Slab, [ZERO_FLUX] * 2,
                                        initial_flux=np.ones((4, 2)), delayed=delayed)
        solver.run(1.0e-3, 3)
        res = solver.result()
        path = tmp_path / "transient.npz"
        nd.save_result(path, res)
        out = nd.load_result(path)
        np.testing.assert_array_equal(out.flux, res.flux)
        np.testing.assert_array_equal(out.precursors, res.precursors)
        assert out.precursors.shape == (4, 6)
        assert (out.time, out.steps) == (res.time, res.steps)
        assert out.metadata["result_type"] == "TimeDependentResult"
        assert "solver" not in out.metadata
