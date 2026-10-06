"""Plotting helpers shared by the examples."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection


def centers(edges):
    edges = np.asarray(edges, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def plot_groups(ax, edges, flux, labels=None, normalize=True, **kwargs):
    """One line per energy group against the cell centers of a 1-D mesh."""
    flux = np.asarray(flux, dtype=float)
    if normalize:
        flux = flux / flux.max()
    x = centers(edges)
    for g in range(flux.shape[1]):
        label = labels[g] if labels else f"group {g + 1}"
        ax.plot(x, flux[:, g], label=label, **kwargs)
    ax.set_xlabel("position (cm)")
    ax.set_ylabel("normalized flux" if normalize else "flux")
    if flux.shape[1] > 1 or labels:
        ax.legend(frameon=False)
    return ax


def polygons(mesh):
    """List of (n_vertices, 2) arrays, one per cell of an UnstructuredMesh2D."""
    vx, vy = np.asarray(mesh.vx), np.asarray(mesh.vy)
    offs, cv = np.asarray(mesh.cell_offsets), np.asarray(mesh.cell_vertices)
    return [np.column_stack([vx[cv[a:b]], vy[cv[a:b]]]) for a, b in zip(offs[:-1], offs[1:])]


def plot_cells(ax, mesh, values=None, cmap="inferno", edgecolor="face", linewidth=0.3,
               colorbar=None, **kwargs):
    """Fill each cell of an unstructured mesh by `values` (one per cell); with no
    values, draw the cell outlines only.  The default edge color matches each
    face, which hides the antialiasing seams between filled cells."""
    if values is None:
        edge = "0.3" if edgecolor in (None, "face", "none") else edgecolor
        coll = PolyCollection(polygons(mesh), facecolor="none", edgecolor=edge,
                              linewidth=linewidth, **kwargs)
    else:
        coll = PolyCollection(polygons(mesh), array=np.asarray(values, dtype=float),
                              cmap=cmap, edgecolor=edgecolor, linewidth=linewidth, **kwargs)
    ax.add_collection(coll)
    ax.margins(0)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    if colorbar and values is not None:
        plt.colorbar(coll, ax=ax, label=colorbar, shrink=0.8)
    return coll


def plot_grid(ax, edges_x, edges_y, values, cmap="inferno", colorbar=None):
    """Fill a 2-D structured mesh; values are row-major with y fastest, as the
    structured solvers return them."""
    nx, ny = len(edges_x) - 1, len(edges_y) - 1
    image = np.asarray(values, dtype=float).reshape(nx, ny).T
    mesh = ax.pcolormesh(edges_x, edges_y, image, cmap=cmap, shading="flat")
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    if colorbar:
        plt.colorbar(mesh, ax=ax, label=colorbar, shrink=0.8)
    return mesh
