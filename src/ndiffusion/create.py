"""Utilities for building neutron diffusion problem inputs."""

import warnings

import numpy as np

# Same values as ndiffusion.transport.scatter_orientation.
_SCATTER_ORIENTATIONS = ("to_from", "from_to")


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

    3. List of widths (float or int) with *total_cells* - cells are distributed
       proportionally; the last region absorbs any rounding remainder::

           make_medium_map([R1, R2], total_cells=100)

    4. List of physical lengths in cm with *edges* - each cell is assigned by
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
            for i, hi in enumerate(boundaries[1:]):
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


def make_materials(data_list, G, descending_energy=None,
                   scatter_orientation="to_from"):
    """Build a configured Materials object from a list of cross-section dicts.

    Each dict (e.g., the result of ``np.load("material.npz")``) must contain:

    - ``D``             - diffusion coefficients, shape ``(G,)``
    - ``Siga``          - absorption cross sections, shape ``(G,)``
    - ``Scat``          - scatter matrix, shape ``(G, G)``, ``scatter[g_to][g_from]``
      by default (see *scatter_orientation*)
    - ``nuSigf``        - nu-fission cross sections, shape ``(G,)`` or ``(G, G)``

    Optional keys:

    - ``Removal``  - precomputed removal cross sections, shape ``(G,)``.
                     If present, skips removal computation and does not zero
                     the scatter diagonal.  ``Siga`` is then unused and may be
                     omitted (this is the path
                     :func:`ndiffusion.make_materials_from_transport` takes).
    - ``chi``      - fission spectrum, shape ``(G,)``.  Defaults to all-zeros
                     when absent (activates fission-matrix mode in the solver
                     when ``nuSigf`` is also a matrix).

    Removal computation and the scatter diagonal
    ---------------------------------------------
    The solver convention is ``scatter[g_to][g_from]`` with the self-scatter
    diagonal **excluded** (it cancels out of the removal term).  When ``Removal``
    is *not* supplied, this function derives it as
    ``Siga[g] + (total out-scatter from g) - Scat[g, g]`` and then zeroes the
    diagonal of ``Scat`` in place.  When ``Removal`` *is* supplied the data is
    taken as-is and the diagonal is left untouched - the caller is then
    responsible for providing a ``Scat`` matrix consistent with that ``Removal``.

    The "total out-scatter from g" is the column sum
    ``sum_{g_to} Scat[g_to][g]`` once ``Scat`` is in the solver's
    ``[g_to][g_from]`` order - that is, ``axis=0``.  Which axis that is in the
    *input* depends on how the caller stored the matrix, so it is selected by
    ``scatter_orientation``, as in
    :func:`ndiffusion.transport_to_diffusion`.  A ``"from_to"`` matrix is
    transposed into solver order first, keeping the stored transfer matrix and
    the derived removal consistent.

    Parameters
    ----------
    data_list : list of dict-like
        Ordered list of cross-section data containers, one per material.
    G : int
        Number of energy groups.
    descending_energy : bool or None, optional
        **Deprecated** - use *scatter_orientation*, which names the array
        orientation this argument actually selected.  ``True`` is equivalent to
        ``scatter_orientation="to_from"`` and warns; ``False`` raises, because it
        summed the out-scatter over one axis while passing the matrix through
        untransposed - pass ``scatter_orientation="from_to"`` instead.
    scatter_orientation : {"to_from", "from_to"}, optional
        Storage order of the input ``Scat`` matrix.  ``"to_from"`` (default) is
        the solver's own convention, ``Scat[g_to][g_from]``, and is used as-is;
        ``"from_to"`` is the usual transport-library convention,
        ``Scat[g_from][g_to]``, and is transposed into solver order.  Applies
        whether or not ``Removal`` is supplied.

    Returns
    -------
    Materials
        Fully configured Materials object ready to pass to a solver.
    """
    from ndiffusion._core import Materials

    if scatter_orientation not in _SCATTER_ORIENTATIONS:
        raise ValueError(
            f"scatter_orientation must be one of {_SCATTER_ORIENTATIONS}, "
            f"got {scatter_orientation!r}."
        )
    if descending_energy is not None:
        if not descending_energy:
            raise ValueError(
                "descending_energy=False is not supported: it summed the "
                "out-scatter over axis 1 while still handing the solver an "
                "untransposed matrix, so the removal cross section and the "
                "transfer matrix disagreed. Pass "
                'scatter_orientation="from_to" if the input is stored as '
                "Scat[g_from][g_to]."
            )
        warnings.warn(
            "descending_energy is deprecated: the out-scatter axis follows the "
            "storage orientation of Scat, not the energy ordering. "
            'descending_energy=True is equivalent to the default '
            'scatter_orientation="to_from"; use that instead.',
            DeprecationWarning,
            stacklevel=2,
        )

    D_all = []
    removal_all = []
    scatter_all = []
    chi_all = []
    nusigf_all = []

    for m, data in enumerate(data_list):
        d = np.asarray(data["D"]).ravel()

        scatter = np.asarray(data["Scat"], dtype=float)
        if scatter.size != G * G:
            raise ValueError(
                f"material {m}: Scat has {scatter.size} elements, expected "
                f"{G * G} for a ({G}, {G}) matrix."
            )
        scatter = scatter.reshape(G, G).copy()
        if scatter_orientation == "from_to":
            # Input is [g_from][g_to]; the solver indexes [g_to][g_from].
            scatter = scatter.T.copy()

        if "Removal" in data:
            removal = np.asarray(data["Removal"]).ravel().tolist()
        else:
            # Only needed to derive removal; read lazily so that data with a
            # precomputed Removal need not carry it.
            absorb = np.asarray(data["Siga"]).ravel()
            # Out-scatter from g is the column sum over destination groups,
            # sum_{g_to} Scat[g_to][g] - axis 0 of the solver-ordered matrix.
            out_scatter = np.sum(scatter, axis=0)
            removal = [
                absorb[gg] + out_scatter[gg] - scatter[gg, gg]
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
