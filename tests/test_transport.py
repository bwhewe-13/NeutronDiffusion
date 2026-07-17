"""Tests for the transport -> diffusion cross-section transform."""

import numpy as np
import pytest

import ndiffusion as nd


def _two_group_transport():
    """A 2-group transport material with up- and down-scatter.

    Scat is stored the transport way, Scat[g_from][g_to].
    """
    return {
        "SigT": np.array([0.35, 1.10]),
        # from 0: self 0.30, 0->1 0.02   (absorption 0.03)
        # from 1: 1->0 0.01, self 0.90   (absorption 0.19)
        "Scat": np.array([[0.30, 0.02],
                          [0.01, 0.90]]),
        "Scat1": np.array([[0.09, 0.006],
                           [0.002, 0.20]]),
        "nuSigf": np.array([0.02, 0.30]),
        "chi": np.array([1.0, 0.0]),
    }


class TestTransportToDiffusion:
    def test_removal_and_absorption(self):
        d = _two_group_transport()
        out = nd.transport_to_diffusion(d, 2, transport_correction="none")

        # removal = SigT - self-scatter
        assert out["Removal"] == pytest.approx([0.35 - 0.30, 1.10 - 0.90])
        # Siga = SigT - total out-scatter (self included)
        assert out["Siga"] == pytest.approx([0.35 - 0.32, 1.10 - 0.91])

    def test_scatter_is_transposed_and_diagonal_zeroed(self):
        d = _two_group_transport()
        out = nd.transport_to_diffusion(d, 2, transport_correction="none")

        # Input Scat[g_from][g_to] -> solver Scat[g_to][g_from].
        # 0->1 = 0.02 must land at scatter[1][0]; 1->0 = 0.01 at scatter[0][1].
        assert out["Scat"] == pytest.approx(np.array([[0.0, 0.01],
                                                      [0.02, 0.0]]))

    def test_to_from_orientation_is_used_as_is(self):
        d = _two_group_transport()
        from_to = nd.transport_to_diffusion(
            d, 2, transport_correction="none")

        # Feed the already-transposed matrix and declare it as such: the
        # result must be identical.
        d2 = dict(d)
        d2["Scat"] = d["Scat"].T.copy()
        d2["Scat1"] = d["Scat1"].T.copy()
        to_from = nd.transport_to_diffusion(
            d2, 2, scatter_orientation="to_from", transport_correction="none")

        assert to_from["Scat"] == pytest.approx(from_to["Scat"])
        assert to_from["Removal"] == pytest.approx(from_to["Removal"])

    def test_orientation_actually_changes_the_result(self):
        # Guard against the transpose being a no-op on a symmetric fixture.
        d = _two_group_transport()
        a = nd.transport_to_diffusion(d, 2, transport_correction="none")
        b = nd.transport_to_diffusion(
            d, 2, scatter_orientation="to_from", transport_correction="none")
        assert not np.allclose(a["Scat"], b["Scat"])

    def test_sigtr_tabulated_wins_in_auto(self):
        d = dict(_two_group_transport())
        d["SigTr"] = np.array([0.25, 0.80])
        out = nd.transport_to_diffusion(d, 2)
        assert out["D"] == pytest.approx([1 / (3 * 0.25), 1 / (3 * 0.80)])

    def test_p1_outflow_correction(self):
        d = _two_group_transport()
        out = nd.transport_to_diffusion(d, 2, transport_correction="p1")

        # SigTr = SigT - sum_g' Scat1[g->g']
        exp_tr = [0.35 - (0.09 + 0.006), 1.10 - (0.002 + 0.20)]
        assert out["SigTr"] == pytest.approx(exp_tr)
        assert out["D"] == pytest.approx([1 / (3 * t) for t in exp_tr])

    def test_p1_used_when_auto_and_no_sigtr(self):
        d = _two_group_transport()
        auto = nd.transport_to_diffusion(d, 2)
        p1 = nd.transport_to_diffusion(d, 2, transport_correction="p1")
        assert auto["D"] == pytest.approx(p1["D"])

    def test_no_correction(self):
        d = _two_group_transport()
        out = nd.transport_to_diffusion(d, 2, transport_correction="none")
        assert out["D"] == pytest.approx([1 / (3 * 0.35), 1 / (3 * 1.10)])

    def test_correction_does_not_change_removal(self):
        # The outflow correction cancels out of the removal term.
        d = _two_group_transport()
        a = nd.transport_to_diffusion(d, 2, transport_correction="none")
        b = nd.transport_to_diffusion(d, 2, transport_correction="p1")
        assert a["Removal"] == pytest.approx(b["Removal"])

    def test_siga_input_derives_total(self):
        d = _two_group_transport()
        d_abs = {k: v for k, v in d.items() if k != "SigT"}
        d_abs["Siga"] = np.array([0.03, 0.19])   # = SigT - out-scatter

        a = nd.transport_to_diffusion(d, 2, transport_correction="none")
        b = nd.transport_to_diffusion(d_abs, 2, transport_correction="none")
        assert b["SigT"] == pytest.approx(a["SigT"])
        assert b["Removal"] == pytest.approx(a["Removal"])

    def test_fission_matrix_is_reoriented(self):
        d = dict(_two_group_transport())
        del d["chi"]
        d["nuSigf"] = np.array([[0.0, 0.0],
                                [0.02, 0.30]])   # F[g_from][g_to]
        out = nd.transport_to_diffusion(d, 2, transport_correction="none")
        assert out["nuSigf"] == pytest.approx(np.array([[0.0, 0.02],
                                                        [0.0, 0.30]]))


class TestErrors:
    def test_bad_orientation(self):
        with pytest.raises(ValueError, match="scatter_orientation"):
            nd.transport_to_diffusion(
                _two_group_transport(), 2, scatter_orientation="sideways")

    def test_bad_correction(self):
        with pytest.raises(ValueError, match="transport_correction"):
            nd.transport_to_diffusion(
                _two_group_transport(), 2, transport_correction="p3")

    def test_auto_without_sigtr_or_scat1_raises(self):
        d = {k: v for k, v in _two_group_transport().items() if k != "Scat1"}
        with pytest.raises(KeyError, match="transport_correction='none'"):
            nd.transport_to_diffusion(d, 2)

    def test_p1_without_scat1_raises(self):
        d = {k: v for k, v in _two_group_transport().items() if k != "Scat1"}
        with pytest.raises(KeyError, match="Scat1"):
            nd.transport_to_diffusion(d, 2, transport_correction="p1")

    def test_missing_total_and_absorption(self):
        d = {k: v for k, v in _two_group_transport().items() if k != "SigT"}
        with pytest.raises(KeyError, match="SigT"):
            nd.transport_to_diffusion(d, 2, transport_correction="none")

    def test_wrong_scatter_shape(self):
        d = dict(_two_group_transport())
        d["Scat"] = np.zeros((3, 3))
        with pytest.raises(ValueError, match="Scat"):
            nd.transport_to_diffusion(d, 2, transport_correction="none")

    def test_nonpositive_sigma_tr(self):
        d = dict(_two_group_transport())
        d["SigTr"] = np.array([0.25, -0.1])
        with pytest.raises(ValueError, match="non-positive"):
            nd.transport_to_diffusion(d, 2)


class TestMakeMaterialsFromTransport:
    def test_matches_make_materials_on_equivalent_input(self):
        """The transport path must agree with the hand-built diffusion path."""
        d = _two_group_transport()
        mats_t = nd.make_materials_from_transport([d], 2)

        conv = nd.transport_to_diffusion(d, 2)
        equiv = {
            "D": conv["D"],
            "Siga": conv["Siga"],
            # make_materials wants [g_to][g_from] and zeroes the diagonal
            # itself, so hand it the transposed matrix *with* its diagonal.
            "Scat": np.array(d["Scat"]).T.copy(),
            "group_centers": np.array([1.0e6, 0.025]),
            "nuSigf": d["nuSigf"],
            "chi": d["chi"],
        }
        mats_d = nd.make_materials([equiv], 2, descending_energy=True)

        assert list(mats_t.D) == pytest.approx(list(mats_d.D))
        assert list(mats_t.removal) == pytest.approx(list(mats_d.removal))
        assert list(mats_t.scatter) == pytest.approx(list(mats_d.scatter))
        assert list(mats_t.chi) == pytest.approx(list(mats_d.chi))
        assert list(mats_t.nusigf) == pytest.approx(list(mats_d.nusigf))

    def test_material_order_is_preserved(self):
        a = _two_group_transport()
        b = dict(a)
        b["nuSigf"] = np.array([0.04, 0.60])
        mats = nd.make_materials_from_transport([a, b], 2)
        assert mats.n_mat == 2
        assert list(mats.nusigf) == pytest.approx([0.02, 0.30, 0.04, 0.60])

    def test_k_inf_matches_analytic(self):
        """End-to-end: an infinite medium must give the analytic k_inf.

        The 1-D solver is reflective at the left edge by symmetry; making the
        right edge reflective too leaves no leakage, so D drops out and k must
        equal the k_inf implied by the transport data.  This is the real check
        that removal and scatter were converted self-consistently - a
        transposed scatter matrix or a bad removal breaks it.
        """
        d = _two_group_transport()
        mats = nd.make_materials_from_transport([d], 2)

        n_cells = 20
        edges = list(np.linspace(0.0, 10.0, n_cells + 1))
        medium_map = [0] * n_cells
        reflective = [nd.BoundaryCondition(A=0.0, B=1.0) for _ in range(2)]

        solver = nd.KEigenSolver(mats, medium_map, edges, nd.Geometry.Slab,
                                 reflective, epsilon=1e-10, verbose=False)
        result = solver.solve()

        # Analytic k_inf: solve the 0-D two-group balance directly.
        # Sigma_r phi_g = sum_{g'!=g} Sigma_s[g<-g'] phi_g' + chi_g/k * F
        sig_t = np.asarray(d["SigT"], dtype=float)
        scat = np.asarray(d["Scat"], dtype=float)       # [g_from][g_to]
        s = scat.T.copy()                                # [g_to][g_from]
        np.fill_diagonal(s, 0.0)
        removal = sig_t - np.diagonal(scat)
        M = np.diag(removal) - s
        F = np.outer(d["chi"], d["nuSigf"])
        eigs = np.linalg.eigvals(np.linalg.solve(M, F))
        k_analytic = float(np.max(eigs.real))

        assert result.keff == pytest.approx(k_analytic, rel=1e-6)
