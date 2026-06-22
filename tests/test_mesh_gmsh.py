"""
Tests for ndiffusion.load_gmsh.

Gated on the optional ``gmsh`` dependency (``pip install ndiffusion[mesh]``);
skipped cleanly when gmsh is not installed.  Builds a small disk mesh with one
physical surface (fuel) and one physical boundary curve (vacuum), loads it via
load_gmsh, and feeds the result to the unstructured k-eigenvalue solver.
"""

import numpy as np
import pytest

import ndiffusion as nd

gmsh = pytest.importorskip("gmsh")


def one_group_mat():
    m = nd.Materials()
    m.n_mat = 1
    m.n_groups = 1
    m.D = [3.850204978408833]
    m.removal = [0.1532]
    m.scatter = [0.0]
    m.chi = [1.0]
    m.nusigf = [0.1570]
    return m


def _write_disk_msh(path, radius=50.0, mesh_size=8.0):
    gmsh.initialize()
    try:
        gmsh.model.add("disk")
        disk = gmsh.model.occ.addDisk(0, 0, 0, radius, radius)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [disk], tag=1, name="fuel")
        boundary = [t for _, t in gmsh.model.getBoundary([(2, disk)])]
        gmsh.model.addPhysicalGroup(1, boundary, tag=10, name="vacuum")
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def test_load_gmsh_disk_keigen(tmp_path):
    msh = tmp_path / "disk.msh"
    _write_disk_msh(msh)

    mesh = nd.load_gmsh(msh)

    n_cells = len(mesh.cell_offsets) - 1
    assert n_cells > 0
    assert len(mesh.material_id) == n_cells
    # Single physical surface -> all cells map to material 0.
    assert set(mesh.material_id) == {0}
    # Single physical boundary curve -> boundary faces all carry tag 0.
    assert mesh.bface_bc_tag and set(mesh.bface_bc_tag) == {0}

    res = nd.KEigenSolverUnstructured2D(
        mats=one_group_mat(),
        mesh=mesh,
        bc=[nd.BoundaryCondition(A=1.0, B=0.0)],  # tag 0 -> vacuum
        epsilon=1e-8,
        verbose=False,
    ).solve()

    flux = np.array(res.flux)
    assert len(flux) == n_cells
    assert np.all(flux >= 0)
    assert 0.1 < res.keff < 5.0
