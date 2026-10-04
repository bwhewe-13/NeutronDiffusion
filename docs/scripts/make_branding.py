"""Render the logo, favicon and social preview into docs/_static.

The mark is a real solve: the flux of a bare one-group hexagonal core, 37
assemblies, shaded with inferno.  The wordmark is Inter SemiBold converted to
outlines, so the SVGs render the same without the font installed.  Inter is
only needed to regenerate them:

    python docs/scripts/make_branding.py [--font path/to/Inter-SemiBold.otf]
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path as MplPath

import ndiffusion as nd

OUT = Path(__file__).resolve().parents[1] / "_static"

PITCH = 1.0
GAP = 0.09 * PITCH       # space between neighboring hexagons
CORNER = 0.045 * PITCH   # corner rounding radius

INK = {"light": "#1f2433", "dark": "#e8eaf0"}
DARK_BG = "#0f1117"
MUTED = "#9aa0b4"

# Logo layout, in viewBox units: the mark fills 112 of a 128 square, and the
# wordmark sits to its right.
BOX = 128.0
MARK = 112.0
WORD_X, WORD_BASELINE, WORD_SIZE, WORD_SPACING = 142.0, 84.0, 58.0, -1.0
WORD = "ndiffusion"


def hex_core_flux():
    mesh = nd.layouts.hex_mesh(pitch=PITCH, n_rings=3)
    nd.assign_materials(mesh, nd.layouts.homogeneous())
    mats = nd.materials.one_group(D=1.0, sigma_a=0.05, nusigf=0.06)
    bc = nd.layouts.boundary_conditions(mesh, [1.0], albedo=0.0)
    res = nd.KEigenSolverUnstructured2D(mats, mesh, bc, epsilon=1e-9,
                                        max_outer=5000).solve()
    phi = res.flux[:, 0]
    f = (phi - phi.min()) / (phi.max() - phi.min())

    offsets = mesh.cell_offsets
    polys = []
    for c in range(len(offsets) - 1):
        idx = mesh.cell_vertices[offsets[c]:offsets[c + 1]]
        polys.append(np.column_stack([mesh.vx[idx], mesh.vy[idx]]))
    return polys, f


def inset(poly):
    """Shrink a regular hexagon about its center so that, once its outline is
    stroked by CORNER with round joins, neighbors are GAP apart."""
    center = poly.mean(axis=0)
    apothem = 0.5 * PITCH
    scale = (apothem - 0.5 * GAP - CORNER) / apothem
    return center + scale * (poly - center)


def colors(f):
    cmap = matplotlib.colormaps["inferno"]
    return [matplotlib.colors.to_hex(cmap(0.22 + 0.76 * v)) for v in f]


def fmt(v):
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


# ---------------------------------------------------------------------------
# Mark
# ---------------------------------------------------------------------------


def mark_transform(polys, size, x0, y0):
    """Map mesh coordinates into a size x size box at (x0, y0), y down."""
    pts = np.vstack(polys)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = (hi - lo).max()
    scale = size / span
    mid = 0.5 * (lo + hi)

    def to_box(p):
        q = (p - mid) * scale
        return np.column_stack([x0 + 0.5 * size + q[:, 0], y0 + 0.5 * size - q[:, 1]])
    return to_box, scale


def mark_svg(polys, fills, size, x0, y0):
    to_box, scale = mark_transform(polys, size, x0, y0)
    width = fmt(2.0 * CORNER * scale)
    lines = [f'<g stroke-width="{width}" stroke-linejoin="round">']
    for poly, color in zip(polys, fills):
        pts = " ".join(f"{fmt(x)},{fmt(y)}" for x, y in to_box(inset(poly)))
        lines.append(f'<polygon points="{pts}" fill="{color}" stroke="{color}"/>')
    lines.append("</g>")
    return "\n".join(lines)


def draw_mark(ax, polys, fills, size, x0, y0):
    to_box, scale = mark_transform(polys, size, x0, y0)
    # Figure coordinates are in pixels with y down, so the linewidth in points
    # is pixels * 72 / dpi.
    lw = 2.0 * CORNER * scale * 72.0 / ax.figure.dpi
    for poly, color in zip(polys, fills):
        ax.add_patch(Polygon(to_box(inset(poly)), closed=True, facecolor=color,
                             edgecolor=color, linewidth=lw, joinstyle="round"))


# ---------------------------------------------------------------------------
# Wordmark
# ---------------------------------------------------------------------------


class OutlinePen(BasePen):
    """Collects glyph outlines as (code, points) for both SVG and matplotlib."""

    def __init__(self, glyph_set, transform):
        super().__init__(glyph_set)
        self.transform = transform
        self.segments = []

    def _xy(self, p):
        return self.transform(*p)

    def _moveTo(self, p):
        self.segments.append(("M", [self._xy(p)]))

    def _lineTo(self, p):
        self.segments.append(("L", [self._xy(p)]))

    def _curveToOne(self, p1, p2, p3):
        self.segments.append(("C", [self._xy(p) for p in (p1, p2, p3)]))

    def _qCurveToOne(self, p1, p2):
        self.segments.append(("Q", [self._xy(p) for p in (p1, p2)]))

    def _closePath(self):
        self.segments.append(("Z", []))


def wordmark(font_path, size, x, baseline, spacing):
    """Outline WORD; returns the segments and the x where the text ends."""
    font = TTFont(font_path)
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    scale = size / font["head"].unitsPerEm
    segments = []
    pen_x = x
    for ch in WORD:
        name = cmap[ord(ch)]
        ox = pen_x

        def transform(gx, gy, ox=ox):
            return (ox + gx * scale, baseline - gy * scale)

        pen = OutlinePen(glyphs, transform)
        glyphs[name].draw(pen)
        segments += pen.segments
        pen_x += font["hmtx"][name][0] * scale + spacing
    return segments, pen_x - spacing


def wordmark_svg(segments, color):
    d = []
    for code, pts in segments:
        d.append(code + " ".join(f"{fmt(px)} {fmt(py)}" for px, py in pts))
    return f'<path fill="{color}" d="{"".join(d)}"/>'


def wordmark_path(segments):
    codes_for = {"M": [MplPath.MOVETO], "L": [MplPath.LINETO],
                 "C": [MplPath.CURVE4] * 3, "Q": [MplPath.CURVE3] * 2}
    verts, codes = [], []
    start = None
    for code, pts in segments:
        if code == "Z":
            verts.append(start)
            codes.append(MplPath.CLOSEPOLY)
            continue
        if code == "M":
            start = pts[0]
        verts += pts
        codes += codes_for[code]
    return MplPath(verts, codes)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def svg(width, height, body, background=None):
    bg = (f'<rect width="{fmt(width)}" height="{fmt(height)}" fill="{background}"/>\n'
          if background else "")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {fmt(width)} {fmt(height)}" '
            f'width="{fmt(width)}" height="{fmt(height)}">\n'
            f"<title>ndiffusion</title>\n{bg}{body}\n</svg>\n")


def write_logos(polys, fills, font):
    margin = 0.5 * (BOX - MARK)
    mark = mark_svg(polys, fills, MARK, margin, margin)
    segments, end = wordmark(font, WORD_SIZE, WORD_X, WORD_BASELINE, WORD_SPACING)
    width = np.ceil(end + margin)
    for theme, name in (("light", "logo.svg"), ("dark", "logo-dark.svg")):
        body = mark + "\n" + wordmark_svg(segments, INK[theme])
        (OUT / name).write_text(svg(width, BOX, body))
    (OUT / "icon.svg").write_text(svg(BOX, BOX, mark))
    return segments


def write_favicon(polys, fills, px=64):
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, px)
    ax.set_ylim(px, 0)
    ax.axis("off")
    draw_mark(ax, polys, fills, px * MARK / BOX, px * (BOX - MARK) / (2 * BOX),
              px * (BOX - MARK) / (2 * BOX))
    fig.savefig(OUT / "favicon.png", transparent=True)
    plt.close(fig)


def write_social_preview(polys, fills, segments, regular_font):
    from matplotlib.font_manager import FontProperties

    w, h = 1280, 640
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor=DARK_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    size, gutter, scale = 360.0, 64.0, 1.9
    tag = FontProperties(fname=regular_font, size=21)
    lines = ["Multigroup neutron diffusion in 1-D and 2-D",
             "C++ core, Python interface"]

    # The logo wordmark, scaled up, with its origin moved to (0, 0) so the
    # text block can be placed once its width is known.
    path = wordmark_path(segments)
    word = (path.vertices - [WORD_X, WORD_BASELINE]) * scale
    texts = [ax.text(0.0, 0.0, s, color=MUTED, fontproperties=tag, va="baseline")
             for s in lines]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_w = max(t.get_window_extent(renderer).width for t in texts)
    block_w = max(word[:, 0].max() - word[:, 0].min(), text_w)

    x0 = 0.5 * (w - (size + gutter + block_w))
    draw_mark(ax, polys, fills, size, x0, 0.5 * (h - size))

    # Ascender of the wordmark to the last tagline baseline, centered on the mark.
    tx = x0 + size + gutter
    top = word[:, 1].min()
    spacing = 1.45 * tag.get_size_in_points() * fig.dpi / 72.0
    gap = 1.9 * spacing
    baseline = 0.5 * h - 0.5 * (gap + spacing * (len(lines) - 1) + top)
    ax.add_patch(PathPatch(MplPath(word + [tx - word[:, 0].min(), baseline], path.codes),
                           facecolor=INK["dark"], edgecolor="none"))
    for k, t in enumerate(texts):
        t.set_position((tx, baseline + gap + k * spacing))
    fig.savefig(OUT / "social-preview.png", facecolor=DARK_BG)
    plt.close(fig)


def find_font(style):
    roots = ["/usr/share/fonts", "/usr/local/share/fonts",
             os.path.expanduser("~/.local/share/fonts"), os.path.expanduser("~/.fonts"),
             "/Library/Fonts", os.path.expanduser("~/Library/Fonts"),
             "C:/Windows/Fonts"]
    for root in roots:
        for ext in ("otf", "ttf"):
            hits = glob.glob(f"{root}/**/Inter-{style}.{ext}", recursive=True)
            if hits:
                return hits[0]
    raise SystemExit(f"Inter-{style} not found; pass --font")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--font", help="Inter SemiBold (.otf or .ttf)")
    parser.add_argument("--regular-font", help="Inter Regular, for the preview tagline")
    args = parser.parse_args()
    font = args.font or find_font("SemiBold")
    regular = args.regular_font or find_font("Regular")

    OUT.mkdir(parents=True, exist_ok=True)
    polys, f = hex_core_flux()
    fills = colors(f)
    segments = write_logos(polys, fills, font)
    write_favicon(polys, fills)
    write_social_preview(polys, fills, segments, regular)
    for name in ("logo.svg", "logo-dark.svg", "icon.svg", "favicon.png",
                 "social-preview.png"):
        print(OUT / name)


if __name__ == "__main__":
    main()
