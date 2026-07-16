"""
c5g7_fuel_mesh.py
-----------------
Generates a 2D gmsh mesh for the C5G7 benchmark fuel region shown in Figure 3:
a 34x34 pin core arranged as a 2x2 checkerboard of 17x17 assemblies.

    col  1-17   col 18-34
row  1-17  | UO2  | MOX  |
row 18-34  | MOX  | UO2  |

Geometry (origin at bottom-left, y increases upward):
  Total width  = 34 x 1.26 cm = 42.84 cm
  Total height = 34 x 1.26 cm = 42.84 cm

Physical group tags (2D surfaces):
  1  UO2      UO2 fuel pins
  2  GT       Guide tubes
  3  FC       Fission chambers (centre pin of each assembly)
  4  MOX_43   MOX 4.3 wt% pins
  5  MOX_70   MOX 7.0 wt% pins
  6  MOX_87   MOX 8.7 wt% pins
  7  Mod      Moderator (water) in all pin cells

Boundary physical group tags (1D curves):
  10  Bottom     y = 0
  11  Top        y = 2 x ASSEMBLY_W
  12  Left       x = 0
  13  Right      x = 2 x ASSEMBLY_W
  14  H_seam     horizontal assembly seam  (y = ASSEMBLY_W)
  15  V_seam     vertical   assembly seam  (x = ASSEMBLY_W)

Usage:
  pip install gmsh numpy
  python c5g7_fuel_mesh.py               # default lc = 0.15 cm
  python c5g7_fuel_mesh.py --lc 0.08    # finer
  python c5g7_fuel_mesh.py --lc 0.30    # coarser / faster
  python c5g7_fuel_mesh.py --gui        # open gmsh viewer after meshing
  python c5g7_fuel_mesh.py --output my.msh
"""

import argparse
import sys

import numpy as np

try:
    import gmsh
except ImportError:
    sys.exit("gmsh not found - install with:  pip install gmsh")

# -----------------------------------------------------------------------------
# Geometry constants  (centimetres)
# -----------------------------------------------------------------------------
PIN_PITCH   = 1.26   # pin-cell pitch
FUEL_RADIUS = 0.54   # fuel/clad mix radius (homogenised pin model)
N_PINS      = 17     # pins per assembly side
ASSEMBLY_W  = N_PINS * PIN_PITCH   # 21.42 cm
CORE_W      = 2 * ASSEMBLY_W       # 42.84 cm

# -----------------------------------------------------------------------------
# Physical group tags
# -----------------------------------------------------------------------------
TAG_UO2   = 1
TAG_GT    = 2
TAG_FC    = 3
TAG_MOX43 = 4
TAG_MOX70 = 5
TAG_MOX87 = 6
TAG_MOD   = 7

TAG_BC_BOTTOM = 10
TAG_BC_TOP    = 11
TAG_BC_LEFT   = 12
TAG_BC_RIGHT  = 13
TAG_BC_HSEAM  = 14   # horizontal seam at y = ASSEMBLY_W
TAG_BC_VSEAM  = 15   # vertical   seam at x = ASSEMBLY_W

PIN_TO_TAG = {
    'U': TAG_UO2,
    'G': TAG_GT,
    'F': TAG_FC,
    '4': TAG_MOX43,
    '7': TAG_MOX70,
    '8': TAG_MOX87,
}

GROUP_NAMES = {
    TAG_UO2:   "UO2",
    TAG_GT:    "GT",
    TAG_FC:    "FC",
    TAG_MOX43: "MOX_43",
    TAG_MOX70: "MOX_70",
    TAG_MOX87: "MOX_87",
    TAG_MOD:   "Moderator",
}

BC_NAMES = {
    TAG_BC_BOTTOM: "Bottom",
    TAG_BC_TOP:    "Top",
    TAG_BC_LEFT:   "Left",
    TAG_BC_RIGHT:  "Right",
    TAG_BC_HSEAM:  "H_seam",
    TAG_BC_VSEAM:  "V_seam",
}

# -----------------------------------------------------------------------------
# Pin maps  (17x17, row 0 = bottom row of assembly)
# Source: NEA/NSC/DOC(2003)16, Tables 6 & 7
# -----------------------------------------------------------------------------

# Standard Westinghouse 17x17: guide tubes at every 3rd position (2,5,8,11,14).
# 24 guide tubes + 1 fission chamber at centre (8,8).
_GT_POSITIONS = [
    (2,  2), (2,  5), (2,  8), (2, 11), (2, 14),
    (5,  2), (5,  5), (5,  8), (5, 11), (5, 14),
    (8,  2), (8,  5),           (8, 11), (8, 14),   # (8,8) -> FC
    (11, 2), (11, 5), (11, 8), (11,11), (11,14),
    (14, 2), (14, 5), (14, 8), (14,11), (14,14),
]


def _uo2_map() -> np.ndarray:
    m = np.full((N_PINS, N_PINS), 'U', dtype=object)
    for r, c in _GT_POSITIONS:
        m[r][c] = 'G'
    m[8][8] = 'F'
    return m


def _mox_map() -> np.ndarray:
    m = np.full((N_PINS, N_PINS), '8', dtype=object)
    # Outermost ring -> 4.3%
    m[0,  :] = '4';  m[16,  :] = '4'
    m[:,  0] = '4';  m[:, 16]  = '4'
    # Second ring -> 7.0%
    for i in range(1, 16):
        m[1,  i] = '7';  m[15, i] = '7'
        m[i,  1] = '7';  m[i, 15] = '7'
    for r, c in _GT_POSITIONS:
        m[r][c] = 'G'
    m[8][8] = 'F'
    return m


def _assembly_grid(uo2_map, mox_map):
    """
    Return 2x2 nested list: grid[row_q][col_q] = (x_offset, y_offset, pin_map)

    row_q = 0 -> bottom half (y in [0,          ASSEMBLY_W])
    row_q = 1 -> top    half (y in [ASSEMBLY_W, CORE_W    ])
    col_q = 0 -> left   half (x in [0,          ASSEMBLY_W])
    col_q = 1 -> right  half (x in [ASSEMBLY_W, CORE_W    ])

    Figure 3 layout (row 1 of figure = top = high y):
      top-left  = UO2,  top-right  = MOX
      bot-left  = MOX,  bot-right  = UO2
    """
    return [
        # row_q = 0  (bottom)
        [
            (0.0,        0.0,        mox_map),   # col 0: MOX
            (ASSEMBLY_W, 0.0,        uo2_map),   # col 1: UO2
        ],
        # row_q = 1  (top)
        [
            (0.0,        ASSEMBLY_W, uo2_map),   # col 0: UO2
            (ASSEMBLY_W, ASSEMBLY_W, mox_map),   # col 1: MOX
        ],
    ]


# -----------------------------------------------------------------------------
# Geometry builder
# -----------------------------------------------------------------------------

def _build_pin_geometry(occ, x_offset: float, y_offset: float,
                        pin_map: np.ndarray):
    """
    Add all squares and disks for one 17x17 assembly to the OCC kernel.
    Returns (squares, disks) where:
      squares : list[int]          - OCC surface tags for moderator squares
      disks   : list[(int, str)]   - (OCC tag, pin-type char) for fuel disks
    """
    squares, disks = [], []
    for row in range(N_PINS):
        for col in range(N_PINS):
            x0 = x_offset + col * PIN_PITCH
            y0 = y_offset + row * PIN_PITCH
            cx = x0 + PIN_PITCH / 2
            cy = y0 + PIN_PITCH / 2

            sq   = occ.addRectangle(x0, y0, 0, PIN_PITCH, PIN_PITCH)
            disk = occ.addDisk(cx, cy, 0, FUEL_RADIUS, FUEL_RADIUS)

            squares.append(sq)
            disks.append((disk, pin_map[row][col]))

    return squares, disks


def _classify_surfaces(out_surfaces, assembly_grid):
    """
    Classify each fragmented surface by material.

    Looks up which of the four assemblies the centroid belongs to, then
    checks whether it lies inside the fuel disk (-> fuel material) or
    outside it (-> moderator).

    Returns dict: surface_tag -> physical_group_tag
    """
    tag_to_material = {}

    for dim, stag in out_surfaces:
        if dim != 2:
            continue

        cx, cy, _ = gmsh.model.occ.getCenterOfMass(dim, stag)

        # Identify quadrant
        col_q = 0 if cx < ASSEMBLY_W else 1
        row_q = 0 if cy < ASSEMBLY_W else 1

        x_offset, y_offset, pm = assembly_grid[row_q][col_q]

        # Local coordinates within this assembly
        lx = cx - x_offset
        ly = cy - y_offset

        col = int(lx / PIN_PITCH)
        row = int(ly / PIN_PITCH)
        col = max(0, min(N_PINS - 1, col))
        row = max(0, min(N_PINS - 1, row))

        # Pin centre in global coordinates
        pcx = x_offset + col * PIN_PITCH + PIN_PITCH / 2
        pcy = y_offset + row * PIN_PITCH + PIN_PITCH / 2
        dist = np.hypot(cx - pcx, cy - pcy)

        ptype = pm[row][col]
        if dist < FUEL_RADIUS + 1e-6:
            tag_to_material[stag] = PIN_TO_TAG[ptype]
        else:
            tag_to_material[stag] = TAG_MOD

    return tag_to_material


def _collect_boundary_curves(tol: float = 1e-6):
    """Sort all 1D curves into boundary groups."""
    groups = {t: [] for t in BC_NAMES}

    for dim, ctag in gmsh.model.getEntities(1):
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, ctag)

        if ymax < tol and ymin < tol:
            groups[TAG_BC_BOTTOM].append(ctag)
        if ymin > CORE_W - tol and ymax > CORE_W - tol:
            groups[TAG_BC_TOP].append(ctag)
        if xmax < tol and xmin < tol:
            groups[TAG_BC_LEFT].append(ctag)
        if xmin > CORE_W - tol and xmax > CORE_W - tol:
            groups[TAG_BC_RIGHT].append(ctag)
        if abs(ymin - ASSEMBLY_W) < tol and abs(ymax - ASSEMBLY_W) < tol:
            groups[TAG_BC_HSEAM].append(ctag)
        if abs(xmin - ASSEMBLY_W) < tol and abs(xmax - ASSEMBLY_W) < tol:
            groups[TAG_BC_VSEAM].append(ctag)

    return groups


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def generate_mesh(lc: float = 0.15,
                  open_gui: bool = False,
                  output: str = "c5g7_fuel.msh"):
    """
    Build and write the C5G7 full fuel-region mesh (Figure 3).

    Parameters
    ----------
    lc        : base characteristic length (cm); refined near fuel surfaces
    open_gui  : if True, open the interactive gmsh viewer after meshing
    output    : output .msh filename
    """
    gmsh.initialize()
    gmsh.model.add("c5g7_fuel_34x34")
    occ = gmsh.model.occ

    uo2_map = _uo2_map()
    mox_map = _mox_map()
    grid    = _assembly_grid(uo2_map, mox_map)

    # -- 1. Create pin-cell geometry for all four assemblies -------------------
    print("Creating pin-cell geometry for 4 assemblies (34x34 pins) ...")
    all_squares, all_disks = [], []

    for row_q in range(2):
        for col_q in range(2):
            x_off, y_off, pm = grid[row_q][col_q]
            squares, disks = _build_pin_geometry(occ, x_off, y_off, pm)
            all_squares.extend(squares)
            all_disks.extend(disks)

    dim_squares = [(2, t) for t in all_squares]
    dim_disks   = [(2, t) for t, _ in all_disks]

    # -- 2. Fragment -> conforming interfaces across all materials --------------
    n_sq, n_dk = len(dim_squares), len(dim_disks)
    print(f"Fragmenting {n_sq} squares + {n_dk} disks "
          f"({n_sq + n_dk} total surfaces) ...")
    out, _ = occ.fragment(dim_squares + dim_disks, [])
    occ.synchronize()

    out_surfaces = [(d, t) for d, t in out if d == 2]
    print(f"  -> {len(out_surfaces)} surfaces after fragmentation")

    # -- 3. Classify surfaces by material -------------------------------------
    print("Classifying surfaces by material ...")
    tag_to_material = _classify_surfaces(out_surfaces, grid)

    # -- 4. Physical groups (2D) -----------------------------------------------
    print("Assigning physical groups ...")
    groups = {t: [] for t in list(PIN_TO_TAG.values()) + [TAG_MOD]}
    for stag, mat_tag in tag_to_material.items():
        groups[mat_tag].append(stag)

    for mat_tag, stags in groups.items():
        if stags:
            name = GROUP_NAMES[mat_tag]
            gmsh.model.addPhysicalGroup(2, stags, tag=mat_tag, name=name)
            print(f"  {name:12s} (tag {mat_tag}): {len(stags):4d} surfaces")

    # -- 5. Boundary physical groups (1D) --------------------------------------
    bc_groups = _collect_boundary_curves()
    for bc_tag, curves in bc_groups.items():
        if curves:
            name = BC_NAMES[bc_tag]
            gmsh.model.addPhysicalGroup(1, curves, tag=bc_tag, name=name)
            print(f"  {name:12s} (tag {bc_tag}): {len(curves):4d} curves")

    # -- 6. Mesh-size field: refine near fuel-moderator interfaces -------------
    print("Setting mesh-size field ...")

    fuel_mats = {TAG_UO2, TAG_GT, TAG_FC, TAG_MOX43, TAG_MOX70, TAG_MOX87}
    fuel_curves: set[int] = set()
    for stag, mat_tag in tag_to_material.items():
        if mat_tag in fuel_mats:
            for _, ctag in gmsh.model.getBoundary([(2, stag)], oriented=False):
                fuel_curves.add(abs(ctag))

    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", list(fuel_curves))
    gmsh.model.mesh.field.setNumber (f_dist, "Sampling",   60)

    f_thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh, "InField",  f_dist)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin",  lc / 3)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax",  lc)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMin",  0.02)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMax",  0.30)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)

    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints",         0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature",      0)

    # -- 7. Generate and optimise mesh -----------------------------------------
    gmsh.option.setNumber("Mesh.Algorithm",  6)   # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.Smoothing", 10)

    print("Generating 2D mesh ...")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Laplace2D")

    # -- 8. Write output -------------------------------------------------------
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(output)

    el_types, el_tags, _ = gmsh.model.mesh.getElements(2)
    n_el = sum(len(t) for t in el_tags)
    print(f"Mesh written to '{output}'  ({n_el} 2D elements)")

    if open_gui:
        print("Opening gmsh GUI ... (close the window to exit)")
        gmsh.fltk.run()

    gmsh.finalize()
    print("Done.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="C5G7 fuel mesh (Figure 3): 34x34 pin core, 4 assemblies."
    )
    p.add_argument("--lc",     type=float, default=0.15,
                   help="Base mesh size in cm (default 0.15)")
    p.add_argument("--gui",    action="store_true",
                   help="Open gmsh GUI after meshing")
    p.add_argument("--output", type=str, default="c5g7_fuel.msh",
                   help="Output filename (default: c5g7_fuel.msh)")
    args = p.parse_args()
    generate_mesh(lc=args.lc, open_gui=args.gui, output=args.output)
