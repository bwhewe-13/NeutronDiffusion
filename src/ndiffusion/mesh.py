"""Gmsh mesh import for the unstructured 2D diffusion solver."""

from pathlib import Path
from typing import Union


def load_gmsh(path: Union[str, Path]):
    """Load a Gmsh .msh file into an UnstructuredMesh2D.

    Physical surface groups (dim=2) map to 0-indexed material IDs, sorted by
    Gmsh physical group tag.  Physical curve groups (dim=1) map to 0-indexed
    BC tags the same way.  Cells or boundary edges not in any physical group
    default to index 0.

    Parameters
    ----------
    path :
        Path to a Gmsh .msh file (any format version supported by the
        installed gmsh Python package).

    Returns
    -------
    UnstructuredMesh2D

    Notes
    -----
    Supported 2D element types: triangles (Gmsh type 2) and quads (type 3).
    Higher-order or other element types in the file are silently skipped.

    The BC tag mapping is:
      - Sort all physical curve group tags numerically.
      - The group with the smallest tag -> bc_tag 0, next -> bc_tag 1, etc.
    Boundary edges not belonging to any physical curve group default to 0.

    Examples
    --------
    In Gmsh (Python API)::

        import gmsh
        gmsh.initialize()
        gmsh.model.add("reactor")
        core = gmsh.model.occ.addDisk(0, 0, 0, 100, 100)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [core], tag=1, name="fuel")
        boundary_curves = [t for _, t in gmsh.model.getBoundary([(2, core)])]
        gmsh.model.addPhysicalGroup(1, boundary_curves, tag=10, name="vacuum")
        gmsh.option.setNumber("Mesh.MeshSizeMax", 10.0)
        gmsh.model.mesh.generate(2)
        gmsh.write("reactor.msh")
        gmsh.finalize()

        import ndiffusion as nd
        mesh = nd.load_gmsh("reactor.msh")
    """
    try:
        import gmsh
    except ImportError as exc:
        raise ImportError(
            "The gmsh Python package is required for load_gmsh(). "
            "Install it with: pip install gmsh"
        ) from exc

    gmsh.initialize()
    try:
        gmsh.model.add("ndiffusion_import")
        gmsh.open(str(path))
        return _extract_mesh(gmsh)
    finally:
        gmsh.finalize()


def _extract_mesh(gmsh):
    """Extract an UnstructuredMesh2D from the current gmsh model."""
    from ndiffusion import UnstructuredMesh2D

    # ------------------------------------------------------------------
    # Nodes: build a dense 0-based index from (possibly non-contiguous)
    # Gmsh 1-based node tags.
    # ------------------------------------------------------------------
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
    n = len(node_tags)
    vx = [node_coords[3 * i]     for i in range(n)]
    vy = [node_coords[3 * i + 1] for i in range(n)]

    # ------------------------------------------------------------------
    # Material IDs from physical surface groups (dim=2).
    # Sort by Gmsh physical tag so the mapping is deterministic.
    # ------------------------------------------------------------------
    surf_phys = sorted(gmsh.model.getPhysicalGroups(dim=2), key=lambda x: x[1])
    mat_tag_to_id = {ptag: idx for idx, (_, ptag) in enumerate(surf_phys)}

    entity_to_mat: dict = {}
    for _, ptag in surf_phys:
        mid = mat_tag_to_id[ptag]
        for ent in gmsh.model.getEntitiesForPhysicalGroup(2, ptag):
            entity_to_mat[ent] = mid

    # ------------------------------------------------------------------
    # 2D cells: triangles (type 2, 3 nodes) and quads (type 3, 4 nodes).
    # ------------------------------------------------------------------
    _NODES_PER_TYPE = {2: 3, 3: 4}

    cell_vertices: list = []
    cell_offsets:  list = [0]
    material_id:   list = []

    for _, ent_tag in gmsh.model.getEntities(dim=2):
        mat_id = entity_to_mat.get(ent_tag, 0)
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2, tag=ent_tag)
        for etype, enodes in zip(elem_types, elem_node_tags):
            npe = _NODES_PER_TYPE.get(int(etype))
            if npe is None:
                continue
            n_elems = len(enodes) // npe
            for e in range(n_elems):
                verts = [tag_to_idx[int(enodes[e * npe + k])] for k in range(npe)]
                cell_vertices.extend(verts)
                cell_offsets.append(len(cell_vertices))
                material_id.append(mat_id)

    # ------------------------------------------------------------------
    # BC tags from physical curve groups (dim=1).
    # Sort by Gmsh physical tag -> 0-indexed BC tag.
    # ------------------------------------------------------------------
    curve_phys = sorted(gmsh.model.getPhysicalGroups(dim=1), key=lambda x: x[1])
    bc_tag_to_id = {ptag: idx for idx, (_, ptag) in enumerate(curve_phys)}

    entity_to_bc: dict = {}
    for _, ptag in curve_phys:
        bid = bc_tag_to_id[ptag]
        for ent in gmsh.model.getEntitiesForPhysicalGroup(1, ptag):
            entity_to_bc[ent] = bid

    # ------------------------------------------------------------------
    # Boundary faces from 1D line elements (Gmsh type 1, 2 nodes each).
    # ------------------------------------------------------------------
    bface_v0:    list = []
    bface_v1:    list = []
    bface_bc_tag: list = []

    for _, ent_tag in gmsh.model.getEntities(dim=1):
        bc_tag = entity_to_bc.get(ent_tag, 0)
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=1, tag=ent_tag)
        for etype, enodes in zip(elem_types, elem_node_tags):
            if int(etype) != 1:
                continue
            n_elems = len(enodes) // 2
            for e in range(n_elems):
                bface_v0.append(tag_to_idx[int(enodes[2 * e])])
                bface_v1.append(tag_to_idx[int(enodes[2 * e + 1])])
                bface_bc_tag.append(bc_tag)

    mesh = UnstructuredMesh2D()
    mesh.vx            = vx
    mesh.vy            = vy
    mesh.cell_vertices = cell_vertices
    mesh.cell_offsets  = cell_offsets
    mesh.material_id   = material_id
    mesh.bface_v0      = bface_v0
    mesh.bface_v1      = bface_v1
    mesh.bface_bc_tag  = bface_bc_tag
    return mesh
