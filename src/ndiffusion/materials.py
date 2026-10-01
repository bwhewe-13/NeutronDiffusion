"""Named cross-section sets and the published benchmarks that use them.

Two kinds of thing live here.

**Builders** turn compact tabular data into a :class:`Materials`.  Most
two-group reactor benchmarks are quoted as rows of
``(D1, D2, Sa1, Sa2, S12, nuSf1, nuSf2)``, so :func:`two_group` takes exactly
that, and :func:`one_group` covers the simple cases used for sanity checks.

**Benchmarks** bundle a published cross-section table with its reference
eigenvalue and, where the loading is a map rather than a shape, a painter for
:func:`ndiffusion.assign_materials`.  Geometry is deliberately *not* bundled:
mesh and materials are separate concerns, so the same table can be run on
whatever mesh you like.

    mats = nd.materials.BIBLIS.materials()
    nd.assign_materials(mesh, nd.materials.BIBLIS.layout())
    assert abs(solve(mesh, mats) - nd.materials.BIBLIS.reference_keff) < 1e-3
"""

from collections import namedtuple

import numpy as np


def one_group(D=1.0, sigma_a=0.1, nusigf=0.0, n_mat=1):
    """Single-group Materials, optionally repeated over *n_mat* materials.

    Accepts scalars (broadcast to every material) or per-material sequences.
    """
    from ndiffusion import Materials

    def spread(v):
        arr = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
        return list(arr) if arr.size == n_mat else list(arr) * n_mat

    m = Materials()
    m.n_mat = n_mat
    m.n_groups = 1
    m.D = spread(D)
    m.removal = spread(sigma_a)
    m.scatter = [0.0] * n_mat
    m.chi = [1.0] * n_mat
    m.nusigf = spread(nusigf)
    return m


def two_group(rows, axial_buckling=0.0):
    """Materials from rows of ``(D1, D2, Sa1, Sa2, S12, nuSf1, nuSf2)``.

    The convention every benchmark here shares: fission neutrons are born fast
    (``chi = (1, 0)``), scattering is down only (``S12``), and the removal cross
    section is ``Sigma_a,g`` plus the out-scatter from group *g*.

    Parameters
    ----------
    rows : sequence of 7-tuples
        One row per material, in the order the ``medium_map`` indexes them.
    axial_buckling : float
        Optional transverse leakage ``D_g * B_z^2`` added to each removal cross
        section, the usual way a 2-D model stands in for a finite core height.

    Returns
    -------
    Materials
    """
    from ndiffusion import Materials

    rows = [tuple(float(v) for v in row) for row in rows]
    for k, row in enumerate(rows):
        if len(row) != 7:
            raise ValueError(
                f"row {k} has {len(row)} entries, expected 7: "
                "(D1, D2, Sa1, Sa2, S12, nuSf1, nuSf2)"
            )

    m = Materials()
    m.n_mat = len(rows)
    m.n_groups = 2
    D, removal, scatter, chi, nusigf = [], [], [], [], []
    for d1, d2, sa1, sa2, s12, nsf1, nsf2 in rows:
        D += [d1, d2]
        removal += [sa1 + s12 + d1 * axial_buckling, sa2 + d2 * axial_buckling]
        scatter += [0.0, 0.0, s12, 0.0]      # scatter[g_to][g_from]; 1 -> 2 only
        chi += [1.0, 0.0]
        nusigf += [nsf1, nsf2]
    m.D, m.removal, m.scatter, m.chi, m.nusigf = D, removal, scatter, chi, nusigf
    return m


def from_assembly_map(amap, pitch, origin=None, void=0, offset=1):
    """Painter reading material indices from a rectangular assembly map.

    Core loadings are usually published as a grid of composition numbers, one per
    assembly.  *amap* is that grid, row 0 at the **top** as printed.  Entries
    equal to *void* fall outside the core; they paint as material 0, so carve
    those assemblies out of the geometry rather than relying on this.

    Parameters
    ----------
    amap : 2-D sequence of int
        Composition per assembly.
    pitch : float
        Assembly width in cm.
    origin : (float, float) or None
        Position of the map's centre.  Defaults to the origin.
    void : int
        Map entry meaning "outside the core".
    offset : int
        Subtracted from non-void entries to give a 0-based material index.
    """
    grid = [list(row) for row in amap]
    ny = len(grid)
    nx = len(grid[0]) if ny else 0
    if any(len(row) != nx for row in grid):
        raise ValueError("assembly map rows must all be the same length")
    x0, y0 = origin if origin is not None else (0.0, 0.0)
    half_x = 0.5 * nx * pitch
    half_y = 0.5 * ny * pitch

    def paint(x, y):
        i = int(np.floor((x - x0 + half_x) / pitch))
        j = int(np.floor((y0 + half_y - y) / pitch))      # row 0 at the top
        if not (0 <= i < nx and 0 <= j < ny):
            return 0
        v = grid[j][i]
        return 0 if v == void else int(v) - offset
    return paint


# ---------------------------------------------------------------------------
# Published benchmarks
# ---------------------------------------------------------------------------

_Benchmark = namedtuple(
    "_Benchmark", "name rows reference_keff axial_buckling assembly_map pitch source"
)


class Benchmark(_Benchmark):
    """A published cross-section table with its reference eigenvalue.

    Geometry is not included - the same table runs on any mesh.  Where the core
    loading is a published assembly map, :meth:`layout` turns it into a painter.
    """

    __slots__ = ()

    def materials(self):
        """Materials for this benchmark, including any axial buckling."""
        return two_group(self.rows, axial_buckling=self.axial_buckling)

    def layout(self, origin=None):
        """Painter from the published assembly map.

        Raises
        ------
        ValueError
            If this benchmark's loading is a shape rather than a map; those are
            built from :mod:`ndiffusion.layouts` painters instead.
        """
        if self.assembly_map is None:
            raise ValueError(
                f"{self.name} has no assembly map; its regions are given as "
                "shapes, so build the layout with ndiffusion.layouts painters"
            )
        return from_assembly_map(self.assembly_map, self.pitch, origin=origin)


#: 1-D Ringhals-4 slab: reflector | core | reflector on [-279.5, 279.5] cm.
#: Core half-width 161.25 cm.  Note the negative fast absorption in the
#: reflector, which is how the published table folds in the transverse leakage.
RINGHALS = Benchmark(
    name="Ringhals-4 1-D slab",
    rows=[
        # D1      D2      Sa1      Sa2     S12     nuSf1   nuSf2
        (1.4376, 0.3723,  0.0115, 0.1019, 0.0151, 0.0057, 0.1425),   # core
        (1.3116, 0.2624, -0.0098, 0.0284, 0.0238, 0.0,    0.0),      # reflector
    ],
    reference_keff=1.0037,
    axial_buckling=0.0,
    assembly_map=None,
    pitch=None,
    source="Yu et al., arXiv:2411.15693, Table 2",
)

#: 2-D TWIGL seed/blanket quarter core, 80 x 80 cm.
TWIGL = Benchmark(
    name="TWIGL 2-D quarter core",
    rows=[
        # D1   D2   Sa1    Sa2   S12   nuSf1  nuSf2
        (1.4, 0.4, 0.010, 0.15, 0.01, 0.007, 0.20),   # seed
        (1.3, 0.5, 0.008, 0.05, 0.01, 0.003, 0.06),   # blanket
    ],
    reference_keff=0.9133,
    axial_buckling=0.0,
    assembly_map=None,
    pitch=None,
    source="Yu et al., arXiv:2411.15693, Table 4",
)

#: 2-D IAEA PWR stepped quarter core.  The axial buckling stands in for the
#: finite core height.
IAEA = Benchmark(
    name="IAEA PWR 2-D quarter core",
    rows=[
        # D1   D2   Sa1   Sa2    S12   nuSf1  nuSf2
        (1.5, 0.4, 0.01, 0.080, 0.02, 0.0, 0.135),    # fuel 1
        (1.5, 0.4, 0.01, 0.085, 0.02, 0.0, 0.135),    # fuel 2
        (1.5, 0.4, 0.01, 0.130, 0.02, 0.0, 0.135),    # fuel 2 + rod
        (2.0, 0.3, 0.00, 0.010, 0.04, 0.0, 0.0),      # reflector
    ],
    reference_keff=1.0296,
    axial_buckling=0.8e-4,
    assembly_map=None,
    pitch=20.0,
    source="Yu et al., arXiv:2411.15693, Table 6",
)

#: 2-D BIBLIS full-core PWR, eight compositions on a 17 x 17 assembly map.
#: Composition 3 (index 2) is the non-fissile reflector; 0 marks the void
#: outside the core outline.
BIBLIS = Benchmark(
    name="BIBLIS 2-D full core",
    rows=[
        # D1        D2        Sa1        Sa2        S12       nuSf1      nuSf2
        (1.436000, 0.363500, 0.0095042, 0.0750058, 0.017754, 0.0058708, 0.096067),
        (1.436600, 0.363600, 0.0096785, 0.078436,  0.017621, 0.0061908, 0.10358),
        (1.320000, 0.277200, 0.0026562, 0.071596,  0.023106, 0.0,       0.0),
        (1.438900, 0.363800, 0.010363,  0.091408,  0.017101, 0.0074527, 0.13236),
        (1.438100, 0.366500, 0.010003,  0.084828,  0.017290, 0.0061908, 0.10358),
        (1.438500, 0.366500, 0.010132,  0.087314,  0.017192, 0.0064285, 0.10911),
        (1.438900, 0.367900, 0.010165,  0.088024,  0.017125, 0.0061908, 0.10358),
        (1.439300, 0.368000, 0.010294,  0.09051,   0.017027, 0.0064285, 0.10911),
    ],
    reference_keff=1.02535,
    axial_buckling=0.0,
    assembly_map=[
        [0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0],
        [0, 0, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 0, 0],
        [0, 3, 3, 4, 4, 8, 1, 1, 1, 1, 1, 8, 4, 4, 3, 3, 0],
        [0, 3, 4, 4, 5, 1, 7, 1, 7, 1, 7, 1, 5, 4, 4, 3, 0],
        [3, 3, 4, 5, 2, 8, 2, 8, 1, 8, 2, 8, 2, 5, 4, 3, 3],
        [3, 4, 8, 1, 8, 2, 8, 2, 6, 2, 8, 2, 8, 1, 8, 4, 3],
        [3, 4, 1, 7, 2, 8, 1, 8, 2, 8, 1, 8, 2, 7, 1, 4, 3],
        [3, 4, 1, 1, 8, 2, 8, 1, 8, 1, 8, 2, 8, 1, 1, 4, 3],
        [3, 4, 1, 7, 1, 6, 2, 8, 1, 8, 2, 6, 1, 7, 1, 4, 3],
        [3, 4, 1, 1, 8, 2, 8, 1, 8, 1, 8, 2, 8, 1, 1, 4, 3],
        [3, 4, 1, 7, 2, 8, 1, 8, 2, 8, 1, 8, 2, 7, 1, 4, 3],
        [3, 4, 8, 1, 8, 2, 8, 2, 6, 2, 8, 2, 8, 1, 8, 4, 3],
        [3, 3, 4, 5, 2, 8, 2, 8, 1, 8, 2, 8, 2, 5, 4, 3, 3],
        [0, 3, 4, 4, 5, 1, 7, 1, 7, 1, 7, 1, 5, 4, 4, 3, 0],
        [0, 3, 3, 4, 4, 8, 1, 1, 1, 1, 1, 8, 4, 4, 3, 3, 0],
        [0, 0, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 0, 0],
        [0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0],
    ],
    pitch=23.1226,
    source=("FEMFFUSION validation report, Vidal-Ferrandiz et al. 2023; data "
            "from Nakata & Martin, Nucl. Sci. Eng. 85 (1983)"),
)

#: Every published benchmark, by short name.
BENCHMARKS = {
    "ringhals": RINGHALS,
    "twigl": TWIGL,
    "iaea": IAEA,
    "biblis": BIBLIS,
}
