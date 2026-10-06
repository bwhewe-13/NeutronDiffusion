"""Benchmark geometries shared by the gallery and docs/scripts.

Each ``solve_*`` function builds the published problem on a given mesh size and
returns the DiffusionResult.  The geometry and boundary conditions follow
``tests/test_benchmarks.py``, which pins the cross-section tables in
``ndiffusion.materials`` to their published eigenvalues.
"""

import numpy as np

import ndiffusion as nd


def stepped_mesh(n, h, material_at, axis_tags=True):
    """Quad mesh of the cells of an n x n grid whose material_at(i, j) is not
    None.  Boundary faces on x = 0 or y = 0 get tag 0 when axis_tags is set
    (symmetry), every other boundary face tag 1."""
    vid, vx, vy = {}, [], []

    def vertex(i, j):
        if (i, j) not in vid:
            vid[(i, j)] = len(vx)
            vx.append(i * h)
            vy.append(j * h)
        return vid[(i, j)]

    cell_vertices, cell_offsets, material_id = [], [0], []
    edges = {}
    for i in range(n):
        for j in range(n):
            mat = material_at(i, j)
            if mat is None:
                continue
            v = [vertex(i, j), vertex(i + 1, j), vertex(i + 1, j + 1), vertex(i, j + 1)]
            cell_vertices += v
            cell_offsets.append(len(cell_vertices))
            material_id.append(mat)
            for a, b in zip(v, v[1:] + v[:1]):
                key = (min(a, b), max(a, b))
                edges[key] = edges.get(key, 0) + 1

    bv0, bv1, tags = [], [], []
    for (a, b), count in sorted(edges.items()):
        if count == 1:
            on_axis = (vx[a] == 0.0 and vx[b] == 0.0) or (vy[a] == 0.0 and vy[b] == 0.0)
            bv0.append(a)
            bv1.append(b)
            tags.append(0 if (axis_tags and on_axis) else 1)

    mesh = nd.UnstructuredMesh2D()
    mesh.vx, mesh.vy = vx, vy
    mesh.cell_vertices, mesh.cell_offsets = cell_vertices, cell_offsets
    mesh.material_id = material_id
    mesh.bface_v0, mesh.bface_v1, mesh.bface_bc_tag = bv0, bv1, tags
    return mesh


def solve_ringhals(h):
    """1-D Ringhals-4 slab, half domain [0, 279.5] cm with the core to 161.25."""
    bench = nd.materials.RINGHALS
    edges = np.linspace(0.0, 279.5, int(round(279.5 / h)) + 1)
    mmap = nd.make_medium_map([(0, 161.25), (1, 279.5 - 161.25)], edges=edges)
    bc = [nd.BoundaryCondition(A=0.5, B=1.3116), nd.BoundaryCondition(A=0.5, B=0.2624)]
    return nd.KEigenSolver(bench.materials(), mmap, edges, nd.Geometry.Slab, bc,
                           epsilon=1e-9, max_outer=5000, max_inner=200).solve()


def twigl_mesh(h):
    """TWIGL 80 x 80 cm quarter core: medium map and edges for cell size h."""
    n = int(round(80.0 / h))
    edges = np.linspace(0.0, 80.0, n + 1)
    c = 0.5 * (edges[:-1] + edges[1:])

    def seed(x, y):
        return (24 < y < 56 and x < 56) or (24 < x < 56 and y < 24)

    mmap = [0 if seed(x, y) else 1 for x in c for y in c]
    return mmap, edges


def solve_twigl(h):
    mmap, edges = twigl_mesh(h)
    zero = [nd.BoundaryCondition(A=1.0, B=0.0)] * 2
    return nd.KEigenSolver2D(nd.materials.TWIGL.materials(), mmap, edges, edges,
                             nd.Geometry2D.XY, zero, zero, epsilon=1e-9,
                             max_outer=2000, max_inner=200, use_cg=True).solve()


# IAEA block edges (cm) and 9 x 9 region map, row 0 at the bottom; 1 = outer
# fuel, 2 = inner fuel, 3 = fuel + rod, 4 = reflector, 0 = outside the core.
IAEA_EDGES = [0, 10, 30, 50, 70, 90, 110, 130, 150, 170]
IAEA_MAP = [
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


def iaea_mesh(h):
    n = int(round(170.0 / h))

    def material_at(i, j):
        x, y = (i + 0.5) * h, (j + 0.5) * h
        bi = int(np.searchsorted(IAEA_EDGES, x, side="right")) - 1
        bj = int(np.searchsorted(IAEA_EDGES, y, side="right")) - 1
        reg = IAEA_MAP[bj][bi]
        return None if reg == 0 else reg - 1

    return stepped_mesh(n, h, material_at)


def solve_iaea(h):
    # Reflective on the symmetry axes, Robin dphi/dn = -(0.4692/D) phi outside.
    refl = nd.BoundaryCondition(A=0.0, B=1.0)
    bc = [refl, refl, nd.BoundaryCondition(A=0.4692, B=2.0),
          nd.BoundaryCondition(A=0.4692, B=0.3)]
    return nd.KEigenSolverUnstructured2D(nd.materials.IAEA.materials(), iaea_mesh(h), bc,
                                         epsilon=1e-9, max_outer=2000, max_inner=200,
                                         use_cg=True).solve()


def biblis_mesh(cells_per_assembly):
    amap = nd.materials.BIBLIS.assembly_map
    n = len(amap) * cells_per_assembly
    h = nd.materials.BIBLIS.pitch / cells_per_assembly

    def material_at(i, j):
        comp = amap[j // cells_per_assembly][i // cells_per_assembly]
        return None if comp == 0 else comp - 1

    return stepped_mesh(n, h, material_at, axis_tags=False)


def solve_biblis(cells_per_assembly):
    d1, d2 = nd.materials.BIBLIS.rows[2][:2]   # reflector borders the exterior
    vacuum = nd.boundary_conditions([d1, d2], alpha=0.0)
    bc = [vacuum[0], vacuum[1], vacuum[0], vacuum[1]]
    return nd.KEigenSolverUnstructured2D(nd.materials.BIBLIS.materials(),
                                         biblis_mesh(cells_per_assembly), bc,
                                         epsilon=1e-9, max_outer=2000, max_inner=200,
                                         use_cg=True).solve()
