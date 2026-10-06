"""
C5G7 quarter core
=================

Ties together the three pieces built for the C5G7 benchmark:

  1. the pin-resolved quarter-core mesh from ``tools/c5g7_fuel_mesh.py``
     (2x2 fuel assemblies in the top-left corner, moderator reflector filling
     the L-shaped remainder),
  2. the 7-group transport cross sections in ``critical_2d_C5G7_xs.h5``, mapped
     to diffusion constants by ``ndiffusion.make_materials_from_transport``,
  3. the unstructured FVM solver ``KEigenSolverUnstructured2D``.

Boundary conditions (C5G7 quarter core): reflective on the top and left faces,
vacuum on the bottom and right.

This is a *diffusion* solution of a benchmark defined for *transport*, so the
eigenvalue is not expected to match the published transport reference
(k = 1.18655) to pcm.  The cross-section file is named "critical", i.e. tuned so
the 2-D diffusion solution is close to critical (k ~ 1).

The solver stops on the flux-change norm (``--epsilon``); the eigenvalue itself
converges several iterations sooner, so keff is good to a few pcm well before the
flux fully settles.  The default mesh (``--lc 1.0``) is deliberately coarse so
the demo finishes in a few minutes.  The coarse mesh is *not* mesh-converged
(keff ~ 1.201 here vs ~ 1.189 at lc=0.5); a finer ``--lc`` lowers keff and is
more accurate but the per-iteration cost grows with the cell count.

Run after installing the package (needs the optional mesh + h5py deps)::

    pip install -e ".[mesh]"
    pip install h5py
    python examples/c5g7_quarter_core.py                     # default lc = 1.0
    python examples/c5g7_quarter_core.py --mesh my.msh       # reuse a mesh
    python examples/c5g7_quarter_core.py --lc 0.5            # finer, slower
    python examples/c5g7_quarter_core.py --plot flux.png     # save the figure below

This example is not run when the documentation is built - it takes minutes and
needs gmsh and h5py - so the figure is pre-rendered with ``--plot``:

.. image:: /_static/gallery/c5g7_flux.png
   :alt: Fast and thermal flux in the C5G7 quarter core
"""

import argparse
import os
import sys

import numpy as np

import ndiffusion as nd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XS   = os.path.join(HERE, "critical_2d_C5G7_xs.h5")
DEFAULT_MESH = os.path.join(HERE, "c5g7_fuel.msh")

G = 7

# Transport reference (NEA/NSC/DOC(2003)16) - for orientation only; a diffusion
# solve is not expected to reproduce it.
TRANSPORT_REFERENCE_KEFF = 1.18655

# load_gmsh assigns 0-indexed material IDs by sorted physical tag, which for
# tools/c5g7_fuel_mesh.py gives this order.  The list handed to
# make_materials_from_transport must match it, so map each mesh material to its
# group name in the HDF5 file.
MATERIAL_ORDER = [
    "UO2",              # 0
    "guide_tubes",      # 1  (GT)
    "fission_chamber",  # 2  (FC)
    "MOX43",            # 3
    "MOX70",            # 4
    "MOX87",            # 5
    "moderator",        # 6
]

# Same idea for the boundary tags (sorted curve tags -> 0-indexed bc tag).
BC_BOTTOM, BC_TOP, BC_LEFT, BC_RIGHT = 0, 1, 2, 3


def load_c5g7_xs(path):
    """Read the C5G7 HDF5 library into transport-cross-section dicts.

    Returns a list of 7 dicts in :data:`MATERIAL_ORDER` (= mesh material_id
    order), each shaped for :func:`ndiffusion.transport_to_diffusion`.

    File schema (one HDF5 group per material):
      - ``xs_total``   (7,)   - already the *transport* cross section Sigma_tr,
        so D = 1/(3*Sigma_tr) with no further correction.
      - ``xs_scatter`` (7, 7) - P0 scatter matrix stored ``[g_from][g_to]``.
      - ``chi``/``nu``/``sigmaf`` (7,) - present only for fissile materials;
        nu-fission is ``nu * sigmaf``.  Absent for guide tubes and moderator.
      - ``xs_fission`` (7, 7) - the fission production matrix; unused here since
        the chi and nu*sigmaf vectors carry the same information.
    """
    import h5py

    out = []
    with h5py.File(path, "r") as f:
        for name in MATERIAL_ORDER:
            g = f[name]
            tot  = g["xs_total"][:]      # Sigma_tr (transport-corrected total)
            scat = g["xs_scatter"][:]    # [g_from][g_to]
            if "nu" in g:
                nusigf = g["nu"][:] * g["sigmaf"][:]
                chi    = g["chi"][:]
            else:                        # non-fissile
                nusigf = np.zeros(G)
                chi    = np.zeros(G)
            out.append({
                "SigT":   np.asarray(tot,  dtype=float),
                "Scat":   np.asarray(scat, dtype=float),
                "nuSigf": np.asarray(nusigf, dtype=float),
                "chi":    np.asarray(chi,  dtype=float),
            })
    return out


def infinite_medium_k(mats, m):
    """0-D infinite-medium k for material *m* of a Materials object.

    A conversion self-check: solves ``M phi = (1/k) F phi`` with the loss
    operator ``M = diag(removal) - scatter`` and ``F = chi (x) nusigf`` built
    from the *converted* diffusion constants.  Returns 0 for non-fissile
    materials.
    """
    rem = np.asarray(mats.removal, dtype=float).reshape(mats.n_mat, G)[m]
    s   = np.asarray(mats.scatter, dtype=float).reshape(mats.n_mat, G, G)[m]
    chi = np.asarray(mats.chi,     dtype=float).reshape(mats.n_mat, G)[m]
    nsf = np.asarray(mats.nusigf,  dtype=float).reshape(mats.n_mat, G)[m]
    if nsf.sum() == 0.0:
        return 0.0
    M = np.diag(rem) - s
    F = np.outer(chi, nsf)
    return float(np.max(np.linalg.eigvals(np.linalg.solve(M, F)).real))


def build_boundary_conditions(mats):
    """Assemble the flat ``bc[tag*G + g]`` list (4 tags x 7 groups = 28).

    Reflective on the top and left faces; Marshak vacuum (albedo 0) on the
    bottom and right.  The vacuum coefficient B = D/2 uses the moderator D,
    since every outer face borders the reflector.
    """
    D_mod = np.asarray(mats.D, dtype=float).reshape(mats.n_mat, G)[
        MATERIAL_ORDER.index("moderator")
    ]
    reflective = [nd.BoundaryCondition(A=0.0, B=1.0) for _ in range(G)]
    vacuum = nd.boundary_conditions(D_mod, alpha=0.0)   # A=0.25, B=D/2 per group

    per_tag = {
        BC_BOTTOM: vacuum,
        BC_TOP:    reflective,
        BC_LEFT:   reflective,
        BC_RIGHT:  vacuum,
    }
    bc = []
    for tag in range(4):                # tag-major: bc[tag*G + g]
        bc.extend(per_tag[tag])
    return bc


def acquire_mesh(mesh_path, lc):
    """Load *mesh_path*, generating it with the C5G7 mesher if it is absent."""
    if not os.path.exists(mesh_path):
        print(f"Mesh {mesh_path} not found - generating (lc={lc}) ...")
        sys.path.insert(0, os.path.join(HERE, os.pardir, "tools"))
        import c5g7_fuel_mesh
        c5g7_fuel_mesh.generate_mesh(lc=lc, output=mesh_path)
    return nd.load_gmsh(mesh_path)


def main():
    p = argparse.ArgumentParser(description="C5G7 quarter-core diffusion k-eigenvalue.")
    p.add_argument("--mesh", default=DEFAULT_MESH, help="Gmsh .msh path (generated if absent)")
    p.add_argument("--xs",   default=DEFAULT_XS,   help="C5G7 HDF5 cross-section file")
    p.add_argument("--lc",   type=float, default=1.0, help="Mesh size if generating (cm)")
    p.add_argument("--epsilon", type=float, default=1e-3, help="flux-change tolerance")
    p.add_argument("--max-outer", type=int, default=500, help="Max power iterations")
    p.add_argument("--verbose", action="store_true", help="Print solver convergence")
    p.add_argument("--output", default=None, help="Flux output .npy (default: alongside mesh)")
    p.add_argument("--plot", default=None, help="Save a fast/thermal flux figure to this path")
    args = p.parse_args()

    # -- Cross sections -------------------------------------------------------
    print("Loading C5G7 cross sections ...")
    data = load_c5g7_xs(args.xs)
    mats = nd.make_materials_from_transport(data, G, transport_correction="none")

    print("Per-material infinite-medium k (conversion self-check):")
    for i, name in enumerate(MATERIAL_ORDER):
        print(f"  {i} {name:16s} k_inf = {infinite_medium_k(mats, i):.5f}")
    print()

    # -- Mesh -----------------------------------------------------------------
    print(f"Loading mesh {args.mesh} ...")
    mesh = acquire_mesh(args.mesh, args.lc)
    n_cells = len(mesh.material_id)
    mat_ids = set(mesh.material_id)
    bc_tags = set(mesh.bface_bc_tag)
    print(f"  {n_cells} cells, material_id={sorted(mat_ids)}, bc tags={sorted(bc_tags)}")
    if mat_ids != set(range(7)) or not bc_tags <= {0, 1, 2, 3}:
        raise SystemExit(
            "Mesh material/BC tags do not match the C5G7 layout - is this the "
            "mesh from tools/c5g7_fuel_mesh.py?"
        )

    # -- Solve ----------------------------------------------------------------
    bc = build_boundary_conditions(mats)
    print("Solving unstructured k-eigenvalue (this can take a few minutes) ...")
    solver = nd.KEigenSolverUnstructured2D(
        mats=mats, mesh=mesh, bc=bc,
        epsilon=args.epsilon, max_outer=args.max_outer, max_inner=200,
        use_cg=True, verbose=args.verbose,
    )
    res = solver.solve()

    print()
    print(f"keff        = {res.keff:.6f}")
    print(f"converged   = {res.converged}  (iterations={res.iterations}, "
          f"residual={res.residual:.2e})")
    print(f"transport reference (not a diffusion target): {TRANSPORT_REFERENCE_KEFF}")

    # -- Flux output ----------------------------------------------------------
    flux = np.asarray(res.flux, dtype=float).reshape(n_cells, G)
    if np.any(flux < 0):
        print("WARNING: negative flux entries present")
    out = args.output or os.path.splitext(args.mesh)[0] + "_flux.npy"
    np.save(out, flux)
    print("\nPer-group flux (cell max):")
    for g in range(G):
        print(f"  group {g}: max={flux[:, g].max():.4e}  mean={flux[:, g].mean():.4e}")
    print(f"Flux array ({n_cells}x{G}) saved to {out}")

    if args.plot:
        plot_flux(mesh, flux, args.plot)
        print(f"Figure saved to {args.plot}")


def plot_flux(mesh, flux, path):
    """Fast (group 1) and thermal (group 7) flux side by side."""
    import matplotlib.pyplot as plt

    sys.path.insert(0, HERE)
    from _plotting import plot_cells

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, g, title in ((axes[0], 0, "fast (group 1)"), (axes[1], G - 1, "thermal (group 7)")):
        plot_cells(ax, mesh, flux[:, g] / flux[:, g].max(), colorbar="normalized flux")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
