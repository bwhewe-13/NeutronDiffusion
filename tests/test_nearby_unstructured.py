"""Tests for the method of nearby problems on the unstructured 2-D FVM mesh.

Manufactured solution with zero flux on every edge (all-vacuum BCs):
    phi_exact(x, y) = sin(pi x / Lx) sin(pi y / Ly)

On a regular quad mesh the FVM scheme is second-order and MNP recovers the
(small) O(h^2) error quantitatively.  On the right-triangle mesh the two-point
flux is inconsistent (non-orthogonal cells) and carries a large, non-vanishing
error - which MNP correctly flags (the estimate tracks the true error).
"""

import numpy as np
import pytest

import ndiffusion as nd
from ndiffusion import nearby

pytest.importorskip("scipy")

D = 1.0
SIG_A = 0.2
LX, LY = 10.0, 8.0
AX, AY = np.pi / LX, np.pi / LY


def one_group_absorber():
    m = nd.Materials()
    m.n_mat = 1
    m.n_groups = 1
    m.D = [D]
    m.removal = [SIG_A]
    m.scatter = [0.0]
    m.chi = [0.0]
    m.nusigf = [0.0]
    return m


def quad_mesh(nx, ny, Lx, Ly, bc_tag=0):
    """Regular nx x ny quad mesh on [0,Lx]x[0,Ly]; every boundary face -> bc_tag."""
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

    bv0, bv1, btag = [], [], []

    def add(a, b):
        bv0.append(a)
        bv1.append(b)
        btag.append(bc_tag)

    for i in range(nx):
        add(vid(i, 0), vid(i + 1, 0))
    for j in range(ny):
        add(vid(nx, j), vid(nx, j + 1))
    for i in range(nx):
        add(vid(i, ny), vid(i + 1, ny))
    for j in range(ny):
        add(vid(0, j), vid(0, j + 1))

    mesh = nd.UnstructuredMesh2D()
    mesh.vx, mesh.vy = vx, vy
    mesh.cell_vertices, mesh.cell_offsets = cell_vertices, cell_offsets
    mesh.material_id = mat_ids
    mesh.bface_v0, mesh.bface_v1, mesh.bface_bc_tag = bv0, bv1, btag
    return mesh


def triangle_mesh(nx, ny, Lx, Ly, bc_tag=0):
    """Right-triangle mesh (each quad split in two); every boundary -> bc_tag."""
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
            cell_vertices += [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)]
            cell_offsets.append(len(cell_vertices))
            cell_vertices += [vid(i, j), vid(i + 1, j + 1), vid(i, j + 1)]
            cell_offsets.append(len(cell_vertices))
            mat_ids += [0, 0]

    bv0, bv1, btag = [], [], []

    def add(a, b):
        bv0.append(a)
        bv1.append(b)
        btag.append(bc_tag)

    for i in range(nx):
        add(vid(i, 0), vid(i + 1, 0))
    for j in range(ny):
        add(vid(nx, j), vid(nx, j + 1))
    for i in range(nx):
        add(vid(i, ny), vid(i + 1, ny))
    for j in range(ny):
        add(vid(0, j), vid(0, j + 1))

    mesh = nd.UnstructuredMesh2D()
    mesh.vx, mesh.vy = vx, vy
    mesh.cell_vertices, mesh.cell_offsets = cell_vertices, cell_offsets
    mesh.material_id = mat_ids
    mesh.bface_v0, mesh.bface_v1, mesh.bface_bc_tag = bv0, bv1, btag
    return mesh


def _solve_mms(mesh):
    cx, cy, _area, _neigh = nearby._unstructured_geometry(mesh)
    phi_exact = np.sin(AX * cx) * np.sin(AY * cy)
    source = (D * (AX**2 + AY**2) + SIG_A) * phi_exact

    m = one_group_absorber()
    solver = nd.FixedSourceSolverUnstructured2D(
        m, mesh, [nd.BoundaryCondition(A=1.0, B=0.0)],
        epsilon=1e-12, max_inner=40000, omega=1.8, verbose=False)
    result = nd.nearby_fixed_source(solver, m, source, mesh=mesh)
    return result, phi_exact, cx, cy


class TestNearbyUnstructuredQuad:
    def test_shapes(self):
        mesh = quad_mesh(16, 16, LX, LY)
        result, _, _, _ = _solve_mms(mesh)
        n = 16 * 16
        assert result.curve_fit.shape == (n,)
        assert result.residual.shape == (n,)
        assert result.error_estimate.shape == (n,)

    def test_error_estimate_tracks_true_error(self):
        mesh = quad_mesh(40, 32, LX, LY)
        result, phi_exact, cx, cy = _solve_mms(mesh)
        num = np.asarray(result.numerical.flux)
        te = num - phi_exact
        dx, dy = LX / 40, LY / 32
        interior = ((cx > 4 * dx) & (cx < LX - 4 * dx)
                    & (cy > 4 * dy) & (cy < LY - 4 * dy))
        ratio = np.linalg.norm(result.error_estimate[interior]) / np.linalg.norm(te[interior])
        assert 0.7 < ratio < 1.3

    def test_residual_converges(self):
        r_coarse, _, _, _ = _solve_mms(quad_mesh(20, 16, LX, LY))
        r_fine, _, _, _ = _solve_mms(quad_mesh(40, 32, LX, LY))
        assert np.max(np.abs(r_fine.residual)) < np.max(np.abs(r_coarse.residual))


class TestNearbyUnstructuredTriangle:
    def test_flags_large_scheme_error(self):
        """The triangle FVM has a large, non-vanishing error; MNP flags it
        (the estimate is the same order of magnitude as the true error)."""
        mesh = triangle_mesh(30, 24, LX, LY)
        result, phi_exact, cx, cy = _solve_mms(mesh)
        num = np.asarray(result.numerical.flux)
        te = num - phi_exact
        dx, dy = LX / 30, LY / 24
        interior = ((cx > 4 * dx) & (cx < LX - 4 * dx)
                    & (cy > 4 * dy) & (cy < LY - 4 * dy))
        ratio = np.linalg.norm(result.error_estimate[interior]) / np.linalg.norm(te[interior])
        # The scheme error is large (O(1e-2)); the estimate should track it.
        assert np.max(np.abs(te[interior])) > 1e-3
        assert 0.5 < ratio < 2.0
