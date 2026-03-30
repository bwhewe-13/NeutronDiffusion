"""Utilities for building neutron diffusion problem inputs."""

import numpy as np


def boundary_conditions(Dg, alpha):
    """Compute Robin boundary condition coefficients.

    Marshak formula:
        A = (1 - alpha) / (4 * (1 + alpha))
        B = D / 2  (vacuum/partial reflection)
        B = 1      (reflective, alpha == 1)

    Parameters
    ----------
    Dg : array-like
        Diffusion coefficients per energy group.
    alpha : float
        Albedo (0 = vacuum, 1 = reflective, 0 < alpha < 1 = partial reflection).

    Returns
    -------
    list of BoundaryCondition
        One BoundaryCondition per energy group.
    """
    from ndiffusion._core import BoundaryCondition

    a_val = (1.0 - alpha) / (4.0 * (1.0 + alpha))
    Dg = np.asarray(Dg).ravel()
    if alpha == 1:
        return [BoundaryCondition(A=a_val, B=1.0) for _ in Dg]
    return [BoundaryCondition(A=a_val, B=float(0.5 * d)) for d in Dg]


def make_medium_map(regions, total_cells=None, edges=None):
    """Build a flat medium_map list from a compact region specification.

    Four calling conventions are supported:

    1. List of ``(mat_id, n_cells)`` tuples::

           make_medium_map([(0, 61), (1, 39)])

    2. List of int cell counts (mat IDs are assigned 0, 1, 2, ...)::

           make_medium_map([61, 39])

    3. List of widths (float or int) with *total_cells* — cells are distributed
       proportionally; the last region absorbs any rounding remainder::

           make_medium_map([R1, R2], total_cells=100)

    4. List of physical lengths in cm with *edges* — each cell is assigned by
       its centre position relative to cumulative region boundaries.  Also
       accepts ``(mat_id, length_cm)`` tuples for explicit material IDs::

           make_medium_map([R1, R2], edges=edges)
           make_medium_map([(0, R1), (1, R2)], edges=edges)

    Parameters
    ----------
    regions : list
        Region sizes as int cell counts, float physical lengths (cm),
        or ``(mat_id, size)`` tuples.
    total_cells : int or None
        Total number of spatial cells.  Required for convention 3.
    edges : array-like or None
        Cell-edge positions (cm), length ``n_cells + 1``.  Required for
        convention 4.  When provided, cell assignment is determined by each
        cell's centre position, giving exact results on non-uniform meshes.

    Returns
    -------
    list of int
        Flat medium_map of length equal to the total cell count.

    Raises
    ------
    ValueError
        If a float region size is passed without *total_cells* or *edges*.
    """
    if not regions:
        return []

    # Mode 4: assign cells by physical centre position using edges array
    if edges is not None:
        edges_arr = np.asarray(edges, dtype=float)
        if isinstance(regions[0], tuple):
            mat_ids = [r[0] for r in regions]
            widths = [float(r[1]) for r in regions]
        else:
            mat_ids = list(range(len(regions)))
            widths = [float(w) for w in regions]
        boundaries = [float(edges_arr[0])]
        for w in widths:
            boundaries.append(boundaries[-1] + w)
        cell_centers = 0.5 * (edges_arr[:-1] + edges_arr[1:])
        result = []
        for cx in cell_centers:
            assigned = mat_ids[-1]
            for i, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                if cx < hi:
                    assigned = mat_ids[i]
                    break
            result.append(assigned)
        return result

    # Mode 1: list of (mat_id, n_cells) tuples
    if isinstance(regions[0], tuple):
        result = []
        for mat_id, n_cells in regions:
            result.extend([mat_id] * int(n_cells))
        return result

    # Mode 3: proportional distribution from widths
    if total_cells is not None:
        widths = [float(w) for w in regions]
        total_width = sum(widths)
        counts = [int(total_cells * w / total_width) for w in widths]
        counts[-1] = total_cells - sum(counts[:-1])
        result = []
        for mat_id, count in enumerate(counts):
            result.extend([mat_id] * count)
        return result

    # Mode 2: list of int cell counts
    result = []
    for mat_id, n_cells in enumerate(regions):
        if not isinstance(n_cells, (int, np.integer)):
            raise ValueError(
                f"Region {mat_id} has a float size {n_cells!r}. "
                "Pass total_cells= or edges= to use physical lengths."
            )
        result.extend([mat_id] * n_cells)
    return result


def make_materials(data_list, G):
    """Build a configured Materials object from a list of cross-section dicts.

    Each dict (e.g., the result of ``np.load("material.npz")``) must contain:

    - ``D``             — diffusion coefficients, shape ``(G,)``
    - ``Siga``          — absorption cross sections, shape ``(G,)``
    - ``Scat``          — scatter matrix, shape ``(G, G)``, ``scatter[g_to][g_from]``
    - ``group_centers`` — energy group centres, shape ``(G,)``
    - ``nuSigf``        — nu-fission cross sections, shape ``(G,)`` or ``(G, G)``

    Optional keys:

    - ``Removal``  — precomputed removal cross sections, shape ``(G,)``.
                     If present, skips removal computation and does not zero
                     the scatter diagonal.
    - ``chi``      — fission spectrum, shape ``(G,)``.  Defaults to all-zeros
                     when absent (activates fission-matrix mode in the solver
                     when ``nuSigf`` is also a matrix).

    Parameters
    ----------
    data_list : list of dict-like
        Ordered list of cross-section data containers, one per material.
    G : int
        Number of energy groups.

    Returns
    -------
    Materials
        Fully configured Materials object ready to pass to a solver.
    """
    from ndiffusion._core import Materials

    D_all = []
    removal_all = []
    scatter_all = []
    chi_all = []
    nusigf_all = []

    for data in data_list:
        d = np.asarray(data["D"]).ravel()
        absorb = np.asarray(data["Siga"]).ravel()
        scatter = np.asarray(data["Scat"]).copy().astype(float)
        centers = np.asarray(data["group_centers"]).ravel()

        if "Removal" in data:
            removal = np.asarray(data["Removal"]).ravel().tolist()
        else:
            if np.argmax(centers) == 0:
                removal = [
                    absorb[gg] + np.sum(scatter, axis=0)[gg] - scatter[gg, gg]
                    for gg in range(G)
                ]
            else:
                removal = [
                    absorb[gg] + np.sum(scatter, axis=1)[gg] - scatter[gg, gg]
                    for gg in range(G)
                ]
            np.fill_diagonal(scatter, 0)

        chi = (
            np.asarray(data["chi"]).flatten().tolist()
            if "chi" in data
            else [0.0] * G
        )

        D_all.extend(d.tolist())
        removal_all.extend(removal)
        scatter_all.extend(scatter.flatten().tolist())
        chi_all.extend(chi)
        nusigf_all.extend(np.asarray(data["nuSigf"]).flatten().tolist())

    m = Materials()
    m.n_mat = len(data_list)
    m.n_groups = G
    m.D = D_all
    m.removal = removal_all
    m.scatter = scatter_all
    m.chi = chi_all
    m.nusigf = nusigf_all
    return m
