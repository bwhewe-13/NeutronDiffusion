"""Preset core layouts, symmetry orientations, and matching boundary conditions.

Three things compose here, and each is independent of the other two:

* a **geometry** - :func:`cartesian_mesh` or :func:`hex_mesh` produce cells and
  tagged boundary faces, and know nothing about cross sections;
* an **orientation** - the symmetry sector the geometry is cut down to, which
  also decides which boundary faces are symmetry cuts and which are the physical
  outer surface;
* a **layout** - a painter ``(x, y) -> material index`` handed to
  :func:`ndiffusion.assign_materials`.

So one geometry drives many layouts, and one layout can be evaluated on any
geometry::

    mesh = layouts.cartesian_mesh(width=100.0, h=2.0, orientation="quarter")
    nd.assign_materials(mesh, layouts.core_reflector(core_radius=35.0))
    bc = layouts.boundary_conditions(mesh, D=[1.4, 0.4], albedo=0.0)
    keff = nd.KEigenSolverUnstructured2D(mats, mesh, bc).solve().keff

Boundary tags
-------------
Generators tag every boundary face either :data:`SYMMETRY` or :data:`OUTER`.
:func:`boundary_conditions` then builds the tag-indexed ``bc`` array the
unstructured solvers expect - reflective on the symmetry cuts, Marshak with the
given albedo on the outer surface.  Getting that pairing wrong is silent (a
missed vacuum surface just reads as a higher keff), which is the main reason to
preset it.
"""

import numpy as np

#: Boundary tag for symmetry cuts introduced by an orientation (reflective).
SYMMETRY = 0
#: Boundary tag for the physical outer surface (vacuum, or an albedo).
OUTER = 1

_CARTESIAN_ORIENTATIONS = ("full", "half", "quarter", "eighth", "infinite")
_HEX_ORIENTATIONS = ("full", "half", "sector120", "sector60", "sector30", "infinite")

# Sectors that are a whole rotational period of the lattice, and so can be closed
# periodically.  A square lattice turns onto itself every 90 degrees and a
# hexagonal one every 60, so the 45-degree octant and the 30-degree hex wedge are
# *not* rotational periods: they are fundamental domains only because of the
# additional mirror symmetry, and can only be closed reflectively.
#
# Hex "half" is excluded for a second reason: the 180-degree cut runs along a
# full diameter of the central hexagon, so that cell is bisected and maps onto
# itself under the rotation.  Its two halves would have to be joined to each
# other, making the cell its own neighbour, which the finite-volume balance
# cannot express.  A Cartesian half core is fine - the grid puts a cell edge on
# the cut, so no cell straddles it.
_ROTATIONAL_ORIENTATIONS = {
    "cartesian": ("half", "quarter"),
    "hex": ("sector120", "sector60"),
}


def _check_rotational(orientation, family):
    allowed = _ROTATIONAL_ORIENTATIONS[family]
    if orientation in allowed:
        return
    if orientation in ("full", "infinite"):
        raise ValueError(
            f'orientation {orientation!r} has no symmetry cuts, so there is '
            'nothing for symmetry="rotational" to pair. Use one of '
            f'{allowed}, or leave symmetry at "mirror".'
        )
    period = 90 if family == "cartesian" else 60
    raise ValueError(
        f'orientation {orientation!r} cannot use symmetry="rotational": it is '
        f"not a whole rotational period of the lattice, which turns onto itself "
        f"every {period} degrees. It is a fundamental domain only by virtue of "
        f"the mirror symmetry, so close it with symmetry=\"mirror\", or use one "
        f"of {allowed} rotationally."
    )

# Vertices closer than this fraction of the cell size to a cut are projected onto
# it, which keeps a clip from shaving off slivers.  Projecting rather than merely
# tolerating puts them exactly in the cut plane, so the face can be recognised as
# a symmetry cut afterwards.
_SNAP_FRAC = 1e-6
# Precision at which vertices are merged.  Much finer than the snap: it only has
# to absorb round-off between the same point reached two ways (a polygon corner
# from cos/sin, or the same corner as a clip intersection).
_MERGE_FRAC = 1e-10


# ---------------------------------------------------------------------------
# Orientation -> clipping half-planes
# ---------------------------------------------------------------------------


def _wedge(angle_deg):
    """Half-planes for the wedge 0 <= theta <= angle, both through the origin."""
    a = np.radians(angle_deg)
    return [(0.0, 1.0, 0.0), (np.sin(a), -np.cos(a), 0.0)]


def _half_planes(orientation):
    """Inward half-planes ``a*x + b*y + c >= 0`` that define a symmetry sector."""
    return {
        "full":      [],
        "infinite":  [],
        "half":      [(1.0, 0.0, 0.0)],
        "quarter":   [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        "eighth":    [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, -1.0, 0.0)],
        "sector120": _wedge(120.0),
        "sector60":  _wedge(60.0),
        "sector30":  _wedge(30.0),
    }[orientation]


# ---------------------------------------------------------------------------
# Mesh assembly from polygons
# ---------------------------------------------------------------------------


def _build(polygons, scale):
    """Assemble an UnstructuredMesh2D from a list of vertex-coordinate polygons.

    Vertices are merged by rounded position: neighbouring cells must share vertex
    *indices*, not merely coincide, or no face pairs and the mesh falls apart.
    """
    from ndiffusion import UnstructuredMesh2D

    ndigits = max(0, int(round(-np.log10(scale * _MERGE_FRAC))))
    vids, vx, vy = {}, [], []

    def vertex(x, y):
        key = (round(float(x), ndigits) + 0.0, round(float(y), ndigits) + 0.0)
        if key not in vids:
            vids[key] = len(vx)
            vx.append(key[0])
            vy.append(key[1])
        return vids[key]

    cv, co = [], [0]
    for poly in polygons:
        ids = [vertex(x, y) for x, y in poly]
        # Drop any vertex repeated after snapping, including wrap-around.
        dedup = [v for k, v in enumerate(ids) if v != ids[k - 1]]
        if len(dedup) < 3:
            continue
        cv.extend(dedup)
        co.append(len(cv))

    mesh = UnstructuredMesh2D()
    mesh.vx, mesh.vy = vx, vy
    mesh.cell_vertices, mesh.cell_offsets = cv, co
    mesh.material_id = [0] * (len(co) - 1)
    return mesh


def _tag_boundary(mesh, half_planes, scale):
    """Tag boundary faces SYMMETRY when they lie in a cut plane, else OUTER."""
    edge_count = {}
    off = list(mesh.cell_offsets)
    cv = list(mesh.cell_vertices)
    for c in range(len(off) - 1):
        verts = cv[off[c]:off[c + 1]]
        for k, a in enumerate(verts):
            b = verts[(k + 1) % len(verts)]
            key = (min(a, b), max(a, b))
            edge_count[key] = edge_count.get(key, 0) + 1

    vx, vy = list(mesh.vx), list(mesh.vy)
    tol = scale * _MERGE_FRAC * 100.0
    b0, b1, bt = [], [], []
    for (a, b), count in edge_count.items():
        if count != 1:
            continue
        on_cut = any(
            abs(pa * vx[a] + pb * vy[a] + pc) <= tol
            and abs(pa * vx[b] + pb * vy[b] + pc) <= tol
            for pa, pb, pc in half_planes
        )
        b0.append(a)
        b1.append(b)
        bt.append(SYMMETRY if on_cut else OUTER)

    mesh.bface_v0, mesh.bface_v1, mesh.bface_bc_tag = b0, b1, bt
    return mesh


def _clip(poly, half_planes, tol):
    """Sutherland-Hodgman clip of a convex polygon against inward half-planes.

    Half-planes are normalised so the signed distance is metric, and a vertex
    within *tol* of a plane is projected onto it before the test - otherwise the
    clip leaves vertices a hair off the cut, and the boundary face they belong to
    is no longer recognisable as a symmetry cut.
    """
    pts = list(poly)
    for a, b, c in half_planes:
        norm = np.hypot(a, b)
        if norm == 0.0:
            continue
        a, b, c = a / norm, b / norm, c / norm

        snapped = []
        for x, y in pts:
            d = a * x + b * y + c
            if abs(d) < tol:
                x, y = x - d * a, y - d * b
            snapped.append((x, y))
        pts = snapped

        if not pts:
            return []
        out = []
        n = len(pts)
        for i in range(n):
            p, q = pts[i], pts[(i + 1) % n]
            dp = a * p[0] + b * p[1] + c
            dq = a * q[0] + b * q[1] + c
            if dp >= 0.0:
                out.append(p)
            if (dp > 0.0 and dq < 0.0) or (dp < 0.0 and dq > 0.0):
                t = dp / (dp - dq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
        pts = out
    return pts


def _polygon_area(poly):
    a2 = 0.0
    for i, (x0, y0) in enumerate(poly):
        x1, y1 = poly[(i + 1) % len(poly)]
        a2 += x0 * y1 - x1 * y0
    return 0.5 * abs(a2)


# ---------------------------------------------------------------------------
# Geometry generators
# ---------------------------------------------------------------------------


def cartesian_mesh(width, h, height=None, orientation="full",
                   symmetry="mirror"):
    """Quad mesh of a rectangular core, centred on the origin.

    Parameters
    ----------
    width, height : float
        Full core dimensions in cm; *height* defaults to *width*.
    h : float
        Target cell size.  Rounded so that x=0 and y=0 fall on cell edges, which
        keeps the symmetry cuts free of sliver cells.
    orientation : {"full", "half", "quarter", "eighth", "infinite"}
        Symmetry sector to keep.  ``"half"`` keeps x >= 0, ``"quarter"`` also
        y >= 0, ``"eighth"`` also y <= x (the 45-degree octant).  ``"infinite"``
        keeps the full domain but tags every boundary as a symmetry cut, giving
        k-infinity for a repeating lattice.

    symmetry : {"mirror", "rotational"}
        How the symmetry cuts are closed.  ``"mirror"`` (default) makes them
        reflective.  ``"rotational"`` joins the two cuts periodically instead,
        imposing rotational symmetry *without* mirror symmetry - the right choice
        for a spiral or pinwheel loading, where reflecting would solve a
        different problem.  Not available for ``"full"`` or ``"infinite"``, which
        have no cuts.
    Returns
    -------
    UnstructuredMesh2D
        Boundary faces tagged :data:`SYMMETRY` or :data:`OUTER`.
    """
    if orientation not in _CARTESIAN_ORIENTATIONS:
        raise ValueError(
            f"orientation must be one of {_CARTESIAN_ORIENTATIONS}, "
            f"got {orientation!r}."
        )
    height = width if height is None else height
    nx = max(2, 2 * int(round(width / (2.0 * h))))
    ny = max(2, 2 * int(round(height / (2.0 * h))))
    dx, dy = width / nx, height / ny

    planes = _half_planes(orientation)
    tol = min(dx, dy) * _SNAP_FRAC
    nominal = dx * dy

    polys = []
    for i in range(nx):
        for j in range(ny):
            x0 = -0.5 * width + i * dx
            y0 = -0.5 * height + j * dy
            cell = [(x0, y0), (x0 + dx, y0), (x0 + dx, y0 + dy), (x0, y0 + dy)]
            clipped = _clip(cell, planes, tol) if planes else cell
            if len(clipped) >= 3 and _polygon_area(clipped) > 1e-9 * nominal:
                polys.append(clipped)

    if symmetry not in ("mirror", "rotational"):
        raise ValueError(
            f'symmetry must be "mirror" or "rotational", got {symmetry!r}.')
    if symmetry == "rotational":
        _check_rotational(orientation, "cartesian")

    mesh = _build(polys, max(width, height))
    if orientation == "infinite":
        return _tag_all_symmetry(mesh)
    _tag_boundary(mesh, planes, max(width, height))
    if symmetry == "rotational":
        _make_rotational(mesh, max(width, height))
    return mesh


def hex_mesh(pitch, n_rings, orientation="full", symmetry="mirror"):
    """Honeycomb of regular hexagons arranged in rings about the origin.

    Ring 0 is the single central cell and ring k adds 6k cells, so the full core
    has ``1 + 3 n (n+1)`` cells.  Hexagons are pointy-top; *pitch* is the
    centre-to-centre distance of neighbours.

    Parameters
    ----------
    pitch : float
        Lattice pitch in cm.
    n_rings : int
        Number of rings beyond the central cell.
    orientation : {"full", "half", "sector120", "sector60", "sector30", "infinite"}
        Symmetry sector.  The sector cuts pass through the origin, so they slice
        hexagons into smaller polygons - which the solver handles directly.

    symmetry : {"mirror", "rotational"}
        How the symmetry cuts are closed.  ``"mirror"`` (default) makes them
        reflective.  ``"rotational"`` joins the two cuts periodically instead,
        imposing rotational symmetry *without* mirror symmetry - the right choice
        for a spiral or pinwheel loading, where reflecting would solve a
        different problem.  Not available for ``"full"`` or ``"infinite"``, which
        have no cuts.
    Returns
    -------
    UnstructuredMesh2D
    """
    if orientation not in _HEX_ORIENTATIONS:
        raise ValueError(
            f"orientation must be one of {_HEX_ORIENTATIONS}, got {orientation!r}."
        )
    r = pitch / np.sqrt(3.0)
    planes = _half_planes(orientation)
    tol = pitch * _SNAP_FRAC
    nominal = 1.5 * np.sqrt(3.0) * r * r

    polys = []
    for q in range(-n_rings, n_rings + 1):
        for s in range(-n_rings, n_rings + 1):
            if abs(q + s) > n_rings:
                continue
            cx = pitch * (q + 0.5 * s)
            cy = pitch * (np.sqrt(3.0) / 2.0) * s
            cell = [
                (cx + r * np.cos(np.pi / 2.0 + np.pi / 3.0 * k),
                 cy + r * np.sin(np.pi / 2.0 + np.pi / 3.0 * k))
                for k in range(6)
            ]
            clipped = _clip(cell, planes, tol) if planes else cell
            if len(clipped) >= 3 and _polygon_area(clipped) > 1e-6 * nominal:
                polys.append(clipped)

    extent = pitch * (n_rings + 1)
    if symmetry not in ("mirror", "rotational"):
        raise ValueError(
            f'symmetry must be "mirror" or "rotational", got {symmetry!r}.')
    if symmetry == "rotational":
        _check_rotational(orientation, "hex")

    mesh = _build(polys, extent)
    if orientation == "infinite":
        return _tag_all_symmetry(mesh)
    _tag_boundary(mesh, planes, extent)
    if symmetry == "rotational":
        _make_rotational(mesh, extent)
    return mesh


def _make_rotational(mesh, scale):
    """Join the two symmetry cuts periodically instead of reflecting on them.

    A reflective cut imposes mirror symmetry.  A core whose loading is only
    *rotationally* symmetric - a spiral or pinwheel pattern - does not have that,
    and reflecting would solve a different problem.  Pairing the two cut rays
    under the rotation between them imposes the rotational symmetry alone.

    Edges are grouped by the ray they lie on and matched by radius, so this works
    for a hex wedge (rotation = the sector angle), a Cartesian quarter (90
    degrees) and a half core (180 degrees) alike.
    """
    vx, vy = list(mesh.vx), list(mesh.vy)
    sym = [(a, b) for a, b, t in
           zip(mesh.bface_v0, mesh.bface_v1, mesh.bface_bc_tag) if t == SYMMETRY]
    if not sym:
        raise ValueError(
            "rotational symmetry needs an orientation with symmetry cuts; "
            '"full" and "infinite" have none'
        )

    def ray(a, b):
        mx, my = 0.5 * (vx[a] + vx[b]), 0.5 * (vy[a] + vy[b])
        return round(float(np.degrees(np.arctan2(my, mx))) % 360.0, 6)

    tol_apex = scale * 1e-9
    rays = {}
    for a, b in sym:
        mx, my = 0.5 * (vx[a] + vx[b]), 0.5 * (vy[a] + vy[b])
        if np.hypot(mx, my) <= tol_apex:
            raise ValueError(
                "a symmetry face is centred on the sector apex, so the cell it "
                "belongs to is bisected by the cut and maps onto itself under "
                "the rotation. It would have to be its own periodic neighbour, "
                'which the finite-volume balance cannot express; use '
                'symmetry="mirror" for this sector.'
            )
        rays.setdefault(ray(a, b), []).append((a, b))
    if len(rays) != 2:
        raise ValueError(
            f"expected two symmetry rays to pair, found {len(rays)}: "
            f"{sorted(rays)}. Rotational pairing needs a wedge with exactly two "
            "straight cuts meeting at the origin."
        )

    (ang_a, edges_a), (ang_b, edges_b) = sorted(rays.items())
    if len(edges_a) != len(edges_b):
        raise ValueError(
            f"the two symmetry cuts carry {len(edges_a)} and {len(edges_b)} "
            "faces; they must be congruent to pair periodically"
        )

    tol = scale * _MERGE_FRAC * 100.0

    def by_radius(edges):
        out = []
        for a, b in edges:
            ra = np.hypot(vx[a], vy[a])
            rb = np.hypot(vx[b], vy[b])
            out.append((min(ra, rb), max(ra, rb), a, b, ra, rb))
        return sorted(out)

    ordered_a, ordered_b = by_radius(edges_a), by_radius(edges_b)
    a0, a1, b0, b1 = [], [], [], []
    for (lo_a, hi_a, ea0, ea1, ra0, ra1), (lo_b, hi_b, eb0, eb1, rb0, rb1) in zip(
            ordered_a, ordered_b):
        if abs(lo_a - lo_b) > tol or abs(hi_a - hi_b) > tol:
            raise ValueError(
                "the two symmetry cuts do not match radius for radius "
                f"({lo_a:.6g}-{hi_a:.6g} against {lo_b:.6g}-{hi_b:.6g}); the "
                "sector is not rotationally congruent"
            )
        # Corresponding vertices are the ones at the same radius.
        if ra0 > ra1:
            ea0, ea1 = ea1, ea0
        if rb0 > rb1:
            eb0, eb1 = eb1, eb0
        a0.append(ea0)
        a1.append(ea1)
        b0.append(eb0)
        b1.append(eb1)

    mesh.periodic_a0, mesh.periodic_a1 = a0, a1
    mesh.periodic_b0, mesh.periodic_b1 = b0, b1

    keep = [(a, b, t) for a, b, t in
            zip(mesh.bface_v0, mesh.bface_v1, mesh.bface_bc_tag) if t != SYMMETRY]
    mesh.bface_v0 = [a for a, _, _ in keep]
    mesh.bface_v1 = [b for _, b, _ in keep]
    mesh.bface_bc_tag = [t for _, _, t in keep]
    return mesh


def _tag_all_symmetry(mesh):
    """Tag every boundary face SYMMETRY - an infinite repeating lattice."""
    _tag_boundary(mesh, [], 1.0)
    mesh.bface_bc_tag = [SYMMETRY] * len(mesh.bface_v0)
    return mesh


# ---------------------------------------------------------------------------
# Boundary conditions matching an orientation
# ---------------------------------------------------------------------------


def boundary_conditions(mesh, D, albedo=0.0):
    """Tag-indexed ``bc`` array pairing symmetry cuts with the outer surface.

    Parameters
    ----------
    mesh : UnstructuredMesh2D
        Mesh from one of the generators above, carrying :data:`SYMMETRY` /
        :data:`OUTER` tags.
    D : sequence of float
        Diffusion coefficient per energy group, for the Marshak outer condition.
    albedo : float
        0 = vacuum, 1 = fully reflective, in between = partial return.

    Returns
    -------
    list of BoundaryCondition
        Length ``2 * n_groups``, indexed ``bc[tag * n_groups + g]``.
    """
    from ndiffusion import BoundaryCondition

    D = np.asarray(D, dtype=float).ravel()
    a_outer = (1.0 - albedo) / (4.0 * (1.0 + albedo))
    bc = [BoundaryCondition(A=0.0, B=1.0) for _ in D]                 # SYMMETRY
    if albedo == 1.0:
        bc += [BoundaryCondition(A=a_outer, B=1.0) for _ in D]        # OUTER
    else:
        bc += [BoundaryCondition(A=a_outer, B=float(0.5 * d)) for d in D]
    return bc


# ---------------------------------------------------------------------------
# Layout painters
#
# Each returns a callable (x, y) -> material index, ready for assign_materials.
# They are plain functions of position, so they compose and can be evaluated on
# any geometry - Cartesian or hex, any orientation.
# ---------------------------------------------------------------------------


def homogeneous(material=0):
    """One material everywhere.  The baseline for sanity checks."""
    def paint(x, y):
        return material
    return paint


def core_reflector(core_radius=None, core_half_width=None,
                   core=0, reflector=1):
    """Two regions: a central core inside a reflector.

    Give exactly one of *core_radius* (circular core) or *core_half_width*
    (square core).  The circular form puts a curved material interface on the
    mesh, which is where the non-orthogonal correction earns its keep.
    """
    if (core_radius is None) == (core_half_width is None):
        raise ValueError(
            "give exactly one of core_radius (circular) or core_half_width "
            "(square)"
        )
    if core_radius is not None:
        r2 = float(core_radius) ** 2

        def paint(x, y):
            return core if x * x + y * y <= r2 else reflector
    else:
        a = float(core_half_width)

        def paint(x, y):
            return core if abs(x) <= a and abs(y) <= a else reflector
    return paint


def checkerboard(pitch, materials=(0, 1), origin=(0.0, 0.0)):
    """Alternating materials on a square lattice of the given pitch.

    The sharpest heterogeneity a two-material problem can have: every cell
    borders only the other material, which is the worst case for the
    group-sweep iteration.
    """
    materials = tuple(materials)
    if len(materials) < 2:
        raise ValueError("checkerboard needs at least two materials")
    x0, y0 = origin

    def paint(x, y):
        i = int(np.floor((x - x0) / pitch))
        j = int(np.floor((y - y0) / pitch))
        return materials[(i + j) % len(materials)]
    return paint


def annular(radii, materials, origin=(0.0, 0.0)):
    """Concentric rings.

    *radii* are the outer radii of the inner regions, ascending; *materials* is
    one longer, the last entry covering everything beyond the final radius.
    """
    radii = [float(r) for r in radii]
    materials = list(materials)
    if len(materials) != len(radii) + 1:
        raise ValueError(
            f"materials must have one more entry than radii "
            f"(got {len(materials)} and {len(radii)})"
        )
    if any(b <= a for a, b in zip(radii, radii[1:])):
        raise ValueError("radii must be strictly increasing")
    x0, y0 = origin

    def paint(x, y):
        r = np.hypot(x - x0, y - y0)
        for k, edge in enumerate(radii):
            if r <= edge:
                return materials[k]
        return materials[-1]
    return paint


def hex_rings(pitch, ring_materials, origin=(0.0, 0.0)):
    """Material per hexagonal ring, ring 0 being the central cell.

    *ring_materials* is indexed by ring number; anything beyond its length takes
    the last entry, so ``[0, 0, 1]`` is a two-ring core inside a reflector.
    """
    ring_materials = list(ring_materials)
    if not ring_materials:
        raise ValueError("ring_materials must not be empty")
    x0, y0 = origin

    def paint(x, y):
        # Invert the pointy-top axial mapping, then round in cube coordinates.
        s = (y - y0) / (pitch * np.sqrt(3.0) / 2.0)
        q = (x - x0) / pitch - 0.5 * s
        cq, cs, cr = q, s, -q - s
        rq, rs, rr = round(cq), round(cs), round(cr)
        dq, ds, dr = abs(rq - cq), abs(rs - cs), abs(rr - cr)
        if dq > ds and dq > dr:
            rq = -rs - rr
        elif ds > dr:
            rs = -rq - rr
        ring = int(max(abs(rq), abs(rs), abs(rq + rs)))
        return ring_materials[min(ring, len(ring_materials) - 1)]
    return paint


def with_rods(base, positions, radius, material):
    """Overlay control rods on another layout.

    Pairs with the un-rodded *base* to give the perturbation the mesh/material
    split was built for: paint one, solve, paint the other, solve, and the two
    eigenvalues give the rod worth on an identical geometry::

        unrodded = layouts.core_reflector(core_radius=35.0)
        rodded   = layouts.with_rods(unrodded, [(0.0, 0.0)], 5.0, material=2)
        rho = (k_rodded - k_unrodded) / (k_rodded * k_unrodded)
    """
    positions = [(float(px), float(py)) for px, py in positions]
    r2 = float(radius) ** 2

    def paint(x, y):
        for px, py in positions:
            if (x - px) ** 2 + (y - py) ** 2 <= r2:
                return material
        return base(x, y)
    return paint
