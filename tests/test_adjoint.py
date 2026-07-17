"""Tests for the adjoint materials transform (make_adjoint_materials).

The adjoint diffusion operator shares the forward eigenvalue; the adjoint flux
is the importance function.  The transform transposes the group scatter matrix
and swaps chi/nusigf (standard mode) or transposes the fission matrix.
"""

import numpy as np

import ndiffusion as nd


def zero_flux():
    return nd.BoundaryCondition(A=1.0, B=0.0)


def two_group_fissile():
    """Asymmetric 2-group fissile slab (fast->thermal downscatter)."""
    m = nd.Materials()
    m.n_mat = 1
    m.n_groups = 2
    m.D = [1.5, 0.4]
    m.removal = [0.03, 0.10]
    # scatter[g_to][g_from], zero diagonal
    m.scatter = [0.0, 0.0,
                 0.02, 0.0]
    m.chi = [1.0, 0.0]
    m.nusigf = [0.005, 0.15]
    return m


def slab(cells=80, length=40.0):
    edges = list(np.linspace(0.0, length, cells + 1))
    return [0] * cells, edges


class TestAdjointTransform:
    def test_scatter_transposed_and_fission_swapped(self):
        m = two_group_fissile()
        a = nd.make_adjoint_materials(m)

        s = np.array(m.scatter).reshape(1, 2, 2)
        sa = np.array(a.scatter).reshape(1, 2, 2)
        assert np.allclose(sa, np.transpose(s, (0, 2, 1)))

        # Standard mode: chi and nusigf swap.
        assert np.allclose(a.chi, m.nusigf)
        assert np.allclose(a.nusigf, m.chi)

        # D and removal are unchanged.
        assert np.allclose(a.D, m.D)
        assert np.allclose(a.removal, m.removal)

    def test_does_not_mutate_input(self):
        m = two_group_fissile()
        chi_before = list(m.chi)
        scatter_before = list(m.scatter)
        nd.make_adjoint_materials(m)
        assert list(m.chi) == chi_before
        assert list(m.scatter) == scatter_before

    def test_fission_matrix_mode_transposes(self):
        m = nd.Materials()
        m.n_mat = 1
        m.n_groups = 2
        m.D = [1.5, 0.4]
        m.removal = [0.03, 0.10]
        m.scatter = [0.0, 0.0, 0.02, 0.0]
        m.chi = [0.0, 0.0]  # activates fission-matrix mode
        # F[g_to][g_from]
        m.nusigf = [0.001, 0.20,
                    0.002, 0.05]
        a = nd.make_adjoint_materials(m)
        F = np.array(m.nusigf).reshape(1, 2, 2)
        Fa = np.array(a.nusigf).reshape(1, 2, 2)
        assert np.allclose(Fa, np.transpose(F, (0, 2, 1)))
        assert np.allclose(a.chi, 0.0)


class TestAdjointEigenvalue:
    def test_keff_matches_forward(self):
        """The adjoint eigenvalue equals the forward eigenvalue."""
        m = two_group_fissile()
        mmap, edges = slab()
        bc = [zero_flux(), zero_flux()]

        fwd = nd.KEigenSolver(m, mmap, edges, nd.Geometry.Slab, bc,
                              epsilon=1e-9, verbose=False)
        adj = nd.KEigenSolver(nd.make_adjoint_materials(m), mmap, edges,
                              nd.Geometry.Slab, bc, epsilon=1e-9, verbose=False)
        k_fwd = fwd.solve().keff
        k_adj = adj.solve().keff
        assert abs(k_fwd - k_adj) < 1e-6

    def test_adjoint_flux_differs_from_forward(self):
        """For an asymmetric spectrum the importance function has a different
        group shape than the forward flux."""
        m = two_group_fissile()
        mmap, edges = slab()
        bc = [zero_flux(), zero_flux()]

        fwd = nd.KEigenSolver(m, mmap, edges, nd.Geometry.Slab, bc,
                              epsilon=1e-9, verbose=False).solve()
        adj = nd.KEigenSolver(nd.make_adjoint_materials(m), mmap, edges,
                              nd.Geometry.Slab, bc, epsilon=1e-9,
                              verbose=False).solve()

        f = np.array(fwd.flux).reshape(-1, 2)
        a = np.array(adj.flux).reshape(-1, 2)
        # Normalize each to unit peak and compare the fast/thermal ratio.
        f_ratio = f[:, 0].max() / f[:, 1].max()
        a_ratio = a[:, 0].max() / a[:, 1].max()
        assert not np.isclose(f_ratio, a_ratio, rtol=1e-2)


def _quad_mesh(nx, ny, Lx, Ly):
    """Minimal regular quad mesh, all boundary faces tagged 0 (vacuum)."""
    dx, dy = Lx / nx, Ly / ny
    vx, vy = [], []
    for i in range(nx + 1):
        for j in range(ny + 1):
            vx.append(i * dx)
            vy.append(j * dy)

    def vid(i, j):
        return i * (ny + 1) + j

    cell_vertices, cell_offsets, mat_ids = [], [0], []
    for i in range(nx):
        for j in range(ny):
            cell_vertices += [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
            cell_offsets.append(len(cell_vertices))
            mat_ids.append(0)

    mesh = nd.UnstructuredMesh2D()
    mesh.vx, mesh.vy = vx, vy
    mesh.cell_vertices, mesh.cell_offsets = cell_vertices, cell_offsets
    mesh.material_id = mat_ids
    mesh.bface_v0, mesh.bface_v1, mesh.bface_bc_tag = [], [], []
    return mesh


class TestAdjointUnstructured:
    def test_keff_matches_forward_on_mesh(self):
        m = two_group_fissile()
        mesh = _quad_mesh(12, 12, 20.0, 20.0)
        bc = [zero_flux(), zero_flux()]

        fwd = nd.KEigenSolverUnstructured2D(m, mesh, bc, epsilon=1e-8,
                                            verbose=False).solve()
        adj = nd.KEigenSolverUnstructured2D(nd.make_adjoint_materials(m), mesh,
                                            bc, epsilon=1e-8,
                                            verbose=False).solve()
        assert abs(fwd.keff - adj.keff) < 1e-5
