"""Mesh import and material assignment for the unstructured 2D solvers.

Mesh generation and material assignment are separate steps.  A generator emits
geometry - vertices, connectivity, boundary faces, and (from Gmsh) region labels
- and :func:`assign_materials` paints material indices onto it.  One geometry can
therefore drive several material layouts: rodded and unrodded, a reflector
sensitivity sweep, or a set of perturbed states, all sharing a single mesh.
"""

import warnings
from pathlib import Path
from typing import Union

# Gmsh 2-D element type -> (nodes per element, corner nodes).  Gmsh lists the
# corner nodes first, so a cell-centered finite-volume scheme can take the corners
# and ignore the interior/edge nodes: a curved element is used as the straight-
# sided polygon through its corners.
_ELEMENT_TYPES = {
    2:  (3, 3),    # triangle
    3:  (4, 4),    # quadrangle
    9:  (6, 3),    # triangle, order 2
    10: (9, 4),    # quadrangle, order 2
    16: (8, 4),    # quadrangle, order 2 (serendipity)
    20: (9, 3),    # triangle, order 3 (incomplete)
    21: (10, 3),   # triangle, order 3
    22: (12, 3),   # triangle, order 4 (incomplete)
    23: (15, 3),   # triangle, order 4
    24: (15, 3),   # triangle, order 5 (incomplete)
    25: (21, 3),   # triangle, order 5
    36: (16, 4),   # quadrangle, order 3
    37: (25, 4),   # quadrangle, order 4
    38: (36, 4),   # quadrangle, order 5
    39: (12, 4),   # quadrangle, order 3 (incomplete)
    40: (16, 4),   # quadrangle, order 4 (incomplete)
}


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
        With ``region_names`` and ``bc_names`` attributes mapping each physical
        group name to its material id or BC tag.

    Notes
    -----
    Supported 2D element types: triangles and quadrangles of any order.  Only
    the corner nodes are used - the solver is cell-centered finite volume, so a
    curved element becomes the straight-sided polygon through its corners, which
    is reported once per load as a UserWarning.  Any element type that is
    neither a triangle nor a quadrangle is skipped, also with a warning.

    BC tags follow the physical curve group tags sorted numerically: the group
    with the smallest tag becomes bc_tag 0, the next bc_tag 1, and so on.
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
    # Keep the names: the index a region gets depends on where its physical tag
    # sorts, so referring to regions by name rather than by position is the only
    # way to stay correct when a group is added to the .msh.
    region_names = {}
    for _, ptag in surf_phys:
        name = gmsh.model.getPhysicalName(2, ptag)
        if name:
            region_names[name] = mat_tag_to_id[ptag]

    entity_to_mat: dict = {}
    for _, ptag in surf_phys:
        mid = mat_tag_to_id[ptag]
        for ent in gmsh.model.getEntitiesForPhysicalGroup(2, ptag):
            entity_to_mat[ent] = mid

    # ------------------------------------------------------------------
    # 2D cells: triangles (type 2, 3 nodes) and quads (type 3, 4 nodes).
    # ------------------------------------------------------------------
    cell_vertices: list = []
    cell_offsets:  list = [0]
    material_id:   list = []
    skipped: dict = {}
    n_curved = 0

    for _, ent_tag in gmsh.model.getEntities(dim=2):
        mat_id = entity_to_mat.get(ent_tag, 0)
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2, tag=ent_tag)
        for etype, enodes in zip(elem_types, elem_node_tags):
            spec = _ELEMENT_TYPES.get(int(etype))
            if spec is None:
                n_skipped = len(enodes)
                skipped[int(etype)] = skipped.get(int(etype), 0) + n_skipped
                continue
            npe, n_corner = spec
            n_elems = len(enodes) // npe
            if npe != n_corner:
                n_curved += n_elems
            for e in range(n_elems):
                verts = [tag_to_idx[int(enodes[e * npe + k])] for k in range(n_corner)]
                cell_vertices.extend(verts)
                cell_offsets.append(len(cell_vertices))
                material_id.append(mat_id)

    if n_curved:
        warnings.warn(
            f"{n_curved} higher-order element(s) were reduced to their corner "
            "nodes: the solver is cell-centered finite volume, so each becomes the "
            "straight-sided polygon through its corners. Mesh more finely near "
            "curved boundaries if that approximation matters.",
            stacklevel=3,
        )
    if skipped:
        listing = ", ".join(f"type {t}" for t in sorted(skipped))
        warnings.warn(
            f"skipped 2-D element(s) of unsupported {listing}; only triangles "
            "and quadrangles are supported. The imported mesh omits them, so it "
            "may not cover the whole domain.",
            stacklevel=3,
        )

    # ------------------------------------------------------------------
    # BC tags from physical curve groups (dim=1).
    # Sort by Gmsh physical tag -> 0-indexed BC tag.
    # ------------------------------------------------------------------
    curve_phys = sorted(gmsh.model.getPhysicalGroups(dim=1), key=lambda x: x[1])
    bc_tag_to_id = {ptag: idx for idx, (_, ptag) in enumerate(curve_phys)}
    bc_names = {}
    for _, ptag in curve_phys:
        name = gmsh.model.getPhysicalName(1, ptag)
        if name:
            bc_names[name] = bc_tag_to_id[ptag]

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
    # Attached rather than stored in the C++ struct (UnstructuredMesh2D allows
    # dynamic attributes); assign_materials() and the bc array both index by
    # these, so the names are what make a remap checkable.
    mesh.region_names  = region_names
    mesh.bc_names      = bc_names
    return mesh


_MESH_FIELDS = (
    "vx", "vy", "cell_vertices", "cell_offsets", "material_id",
    "bface_v0", "bface_v1", "bface_bc_tag",
    "periodic_a0", "periodic_a1", "periodic_b0", "periodic_b1",
)


def copy_mesh(mesh):
    """Return an independent copy of *mesh*, including any attached names.

    Copying every field costs roughly 20x what rewriting ``material_id`` alone
    does, which is why :func:`assign_materials` paints in place by default.
    """
    from ndiffusion import UnstructuredMesh2D

    out = UnstructuredMesh2D()
    for field in _MESH_FIELDS:
        setattr(out, field, list(getattr(mesh, field)))
    for attr in ("region_names", "bc_names"):
        if hasattr(mesh, attr):
            setattr(out, attr, dict(getattr(mesh, attr)))
    return out


def assign_materials(mesh, spec, copy=False):
    """Paint material indices onto an existing mesh.

    Parameters
    ----------
    mesh : UnstructuredMesh2D
        Geometry to assign materials to.
    spec : callable, dict, or sequence
        ``callable(x, y) -> int``
            Evaluated at each cell centroid, so the assignment agrees with the
            centroids the FVM solver itself uses.
        ``dict``
            Remaps the material ids already on the mesh.  Keys are either region
            ids (``int``) or region names (``str``, for a mesh from
            :func:`load_gmsh`).  Every region present in the mesh must appear as
            a key - a partial mapping is rejected rather than silently leaving
            some cells on their old index.
        sequence of int
            Used directly; must have one entry per cell.  This is the fast path
            on a large mesh - the callable form has to make one Python call per
            cell, so vectorizing it with :func:`cell_centroids` is roughly twice
            as quick::

                cx, cy = nd.cell_centroids(mesh)
                nd.assign_materials(mesh, np.where(np.asarray(cx) > x0, 1, 0))
    copy : bool, optional
        ``False`` (default) rewrites ``mesh.material_id`` in place and returns
        *mesh*, which is safe to do between solver constructions: each solver
        takes its own copy of the mesh at construction, so repainting afterward
        cannot disturb one already built.  ``True`` leaves the input untouched
        and returns a new mesh.

    Returns
    -------
    UnstructuredMesh2D
        The painted mesh (*mesh* itself unless *copy* is set).

    Raises
    ------
    ValueError
        If the spec does not cover every region, a name is unknown, a sequence
        has the wrong length, or any resulting index is negative.
    TypeError
        If *spec* is not a callable, dict, or sequence.

    Examples
    --------
    One geometry, three layouts::

        mesh = nd.load_gmsh("core.msh")
        nd.assign_materials(mesh, {"fuel": 0, "reflector": 1})
        unrodded = nd.KEigenSolverUnstructured2D(mats, mesh, bc).solve()

        nd.assign_materials(mesh, {"fuel": 0, "reflector": 1, "rod": 2})
        rodded = nd.KEigenSolverUnstructured2D(mats, mesh, bc).solve()

        nd.assign_materials(mesh, lambda x, y: 0 if x * x + y * y < R * R else 1)
        annular = nd.KEigenSolverUnstructured2D(mats, mesh, bc).solve()
    """
    from ndiffusion._core import cell_centroids

    target = copy_mesh(mesh) if copy else mesh
    n_cells = len(target.cell_offsets) - 1

    if callable(spec):
        cx, cy = cell_centroids(target)
        ids = [int(spec(x, y)) for x, y in zip(cx, cy)]
    elif isinstance(spec, dict):
        ids = _remap(target, spec, n_cells)
    else:
        try:
            ids = [int(v) for v in spec]
        except TypeError as exc:
            raise TypeError(
                "spec must be a callable (x, y) -> int, a dict keyed by region "
                f"id or name, or a sequence of length n_cells; got "
                f"{type(spec).__name__}"
            ) from exc
        if len(ids) != n_cells:
            raise ValueError(
                f"spec has {len(ids)} entries but the mesh has {n_cells} cells"
            )

    bad = [i for i in ids if i < 0]
    if bad:
        raise ValueError(
            f"material indices must be non-negative, got {sorted(set(bad))[:5]}"
        )

    target.material_id = ids
    return target


def _remap(mesh, spec, n_cells):
    """Resolve a dict spec against the mesh's current region ids."""
    present = sorted({int(r) for r in mesh.material_id})

    keys = list(spec)
    by_name = [k for k in keys if isinstance(k, str)]
    if by_name and len(by_name) != len(keys):
        raise ValueError(
            "spec mixes region names and region ids; use one or the other"
        )

    if by_name:
        names = getattr(mesh, "region_names", None)
        if not names:
            raise ValueError(
                "spec is keyed by region name but the mesh carries no region "
                "names; only meshes from load_gmsh() do (and only for named "
                "Gmsh physical groups). Key the spec by region id instead."
            )
        unknown = [k for k in keys if k not in names]
        if unknown:
            raise ValueError(
                f"unknown region name(s) {sorted(unknown)}; the mesh defines "
                f"{sorted(names)}"
            )
        spec = {names[k]: v for k, v in spec.items()}

    missing = [r for r in present if r not in spec]
    if missing:
        raise ValueError(
            f"spec does not cover region(s) {missing}; the mesh uses regions "
            f"{present}. Every region must be mapped, so that adding one to the "
            "mesh is an error rather than a silent carry-over of its old index."
        )
    return [int(spec[r]) for r in mesh.material_id]
