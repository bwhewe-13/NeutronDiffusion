"""Render the README and quickstart figures into docs/_static/.

    python docs/scripts/make_readme_figures.py

The figures are committed, so the README renders on GitHub and PyPI without a
docs build.  Each one solves the same problem as the snippet next to it.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import ndiffusion as nd  # noqa: E402
from ndiffusion import layouts  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "..", "_static")
sys.path.insert(0, os.path.join(HERE, "..", "..", "examples"))
from _plotting import plot_cells, plot_grid  # noqa: E402

plt.rcParams.update({"font.size": 10, "savefig.dpi": 150, "savefig.bbox": "tight"})


def sphere():
    """The README example: bare one-group sphere at its critical radius."""
    m = nd.materials.one_group(D=3.850204978408833, sigma_a=0.1532, nusigf=0.1570)
    edges = np.linspace(0.0, 100.0, 51)
    res = nd.KEigenSolver(m, [0] * 50, edges, nd.Geometry.Sphere,
                          [nd.BoundaryCondition(A=1.0, B=0.0)], max_outer=500).solve()
    r = 0.5 * (edges[:-1] + edges[1:])
    B = np.pi / 100.0
    rr = np.linspace(0.01, 100.0, 400)

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.plot(rr, np.sin(B * rr) / (B * rr), color="0.6", lw=2.5, label=r"exact, $\sin(Br)/Br$")
    ax.plot(r, res.flux[:, 0] / res.flux[:, 0].max(), "o", ms=3.5, color="#87216b",
            label=f"ndiffusion, keff = {res.keff:.8f}")
    ax.set_xlabel("radius (cm)")
    ax.set_ylabel("normalized flux")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(STATIC, "readme_sphere.png"))
    plt.close(fig)


def quickstart_2d():
    """The two 2-D quickstart problems: structured k-eigenvalue and an
    unstructured fixed source."""
    m = nd.materials.one_group(D=3.850204978408833, sigma_a=0.1532, nusigf=0.1570)
    R, n = 100.0, 40
    edges = np.linspace(0.0, R, n + 1)
    zero = [nd.BoundaryCondition(A=1.0, B=0.0)]
    res = nd.KEigenSolver2D(m, [0] * (n * n), edges, edges, nd.Geometry2D.XY,
                            zero, zero, max_outer=500).solve()

    mats = nd.materials.two_group([(1.4, 0.4, 0.010, 0.080, 0.02, 0.0, 0.0)])
    mesh = layouts.cartesian_mesh(width=100.0, h=2.5, orientation="quarter")
    bc = layouts.boundary_conditions(mesh, D=[1.4, 0.4], albedo=0.0)
    solver = nd.FixedSourceSolverUnstructured2D(mats, mesh, bc, epsilon=1e-10,
                                                max_inner=5000, omega=1.9)
    source = np.zeros((solver.n_cells, 2))
    source[:, 0] = 1.0
    fixed = solver.solve(source)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    plot_grid(axes[0], edges, edges, res.flux[:, 0] / res.flux[:, 0].max(),
              colorbar="normalized flux")
    axes[0].set_title(f"KEigenSolver2D, keff = {res.keff:.5f}")
    plot_cells(axes[1], mesh, fixed.flux[:, 1], colorbar="thermal flux")
    axes[1].set_title("FixedSourceSolverUnstructured2D")
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC, "quickstart_2d.png"))
    plt.close(fig)


if __name__ == "__main__":
    sphere()
    quickstart_2d()
    print("wrote readme_sphere.png and quickstart_2d.png to", os.path.normpath(STATIC))
