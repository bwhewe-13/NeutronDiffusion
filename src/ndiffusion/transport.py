"""Transport -> diffusion cross-section transform.

Multigroup *transport* libraries (C5G7, NJOY/WIMS output, OpenMC MGXS, ...)
tabulate a total (or absorption) cross section, a scattering matrix, and fission
data.  The diffusion solvers here instead want a diffusion coefficient, a
*removal* cross section, and a scatter matrix with the self-scatter diagonal
removed.  This module performs that conversion.

The relations
-------------
Writing ``Sigma_s0[g->g']`` for the P0 (isotropic) group-transfer matrix:

* **Total.**  If only absorption is tabulated, the total follows from the
  balance ``Sigma_t[g] = Sigma_a[g] + sum_g' Sigma_s0[g->g']`` (the sum runs
  over *all* g', self-scatter included).

* **Diffusion coefficient.**  ``D[g] = 1 / (3 * Sigma_tr[g])``.  The transport
  cross section ``Sigma_tr`` comes from one of three routes, selected by
  *transport_correction*:

  - ``"auto"`` (default): use a tabulated ``SigTr`` if present, else apply the
    P1 outflow correction below.  Raises if neither is available, rather than
    silently falling back to an uncorrected D.
  - ``"p1"``: outflow (consistent-P1) correction from the P1 scatter moment,
    ``Sigma_tr[g] = Sigma_t[g] - sum_g' Sigma_s1[g->g']``.
  - ``"none"``: no correction, ``Sigma_tr = Sigma_t``.  Rarely accurate; useful
    as a baseline.

  The outflow correction subtracts the same quantity from the total and from
  the self-scatter term, so it cancels out of the removal cross section - which
  is why *removal* below is built from the **uncorrected** ``Sigma_t``.

* **Removal.**  ``Sigma_r[g] = Sigma_t[g] - Sigma_s0[g->g]``.  Expanding
  ``Sigma_t`` gives ``Sigma_a[g] + (out-scatter from g) - Sigma_s0[g->g]``,
  which is exactly the convention :func:`ndiffusion.make_materials` uses.

* **Scatter.**  Transport libraries almost always store the matrix
  ``g_from -> g_to``; the solvers index it ``scatter[g_to][g_from]``.  The
  transform transposes it (see *scatter_orientation*) and zeroes the diagonal,
  since self-scatter is already accounted for in the removal term.

* **Fission.**  ``chi`` and ``nuSigf`` pass through unchanged.  A ``(G, G)``
  ``nuSigf`` is treated as a fission *transfer* matrix and is reoriented with
  the same rule as the scatter matrix (see :class:`Materials` fission-matrix
  mode).

Nothing here is specific to a geometry or a solver: the result is an ordinary
:class:`Materials` accepted by every solver in the package.
"""

import numpy as np

_ORIENTATIONS = ("from_to", "to_from")
_CORRECTIONS = ("auto", "p1", "none")


def _oriented(matrix, G, orientation, name):
    """Return *matrix* as ``[g_from][g_to]`` regardless of input orientation."""
    m = np.asarray(matrix, dtype=float)
    if m.shape != (G, G):
        raise ValueError(
            f"{name} has shape {m.shape}, expected ({G}, {G})."
        )
    # "to_from" input is [g_to][g_from]; transpose it into [g_from][g_to].
    return m.T.copy() if orientation == "to_from" else m.copy()


def _transport_xs(data, sig_t, G, correction, orientation):
    """Resolve Sigma_tr from the tabulated data and the chosen correction."""
    if correction == "none":
        return sig_t

    if correction == "auto" and "SigTr" in data:
        sig_tr = np.asarray(data["SigTr"], dtype=float).ravel()
        if sig_tr.size != G:
            raise ValueError(
                f"SigTr has {sig_tr.size} entries, expected {G}."
            )
        return sig_tr

    if "Scat1" in data:
        scat1 = _oriented(data["Scat1"], G, orientation, "Scat1")
        return sig_t - scat1.sum(axis=1)

    if correction == "p1":
        raise KeyError(
            "transport_correction='p1' needs a P1 scatter moment matrix under "
            "the key 'Scat1'."
        )
    raise KeyError(
        "No transport cross section available: supply 'SigTr', or 'Scat1' for "
        "the P1 outflow correction, or pass transport_correction='none' to use "
        "the uncorrected D = 1/(3*SigT)."
    )


def transport_to_diffusion(data, G, scatter_orientation="from_to",
                           transport_correction="auto"):
    """Convert one material's transport cross sections to diffusion ones.

    Parameters
    ----------
    data : dict-like
        Transport cross sections for a single material.  Recognised keys:

        - ``SigT`` or ``Siga`` - total, or absorption; one is required and the
          other is derived from the scatter matrix.
        - ``Scat``   - P0 scatter matrix, shape ``(G, G)``.  Required.
        - ``SigTr``  - transport cross section, shape ``(G,)``.  Optional.
        - ``Scat1``  - P1 scatter moment matrix, shape ``(G, G)``.  Optional.
        - ``nuSigf`` - nu-fission, shape ``(G,)`` or ``(G, G)``.  Required.
        - ``chi``    - fission spectrum, shape ``(G,)``.  Optional; when absent
          the caller gets fission-matrix mode if ``nuSigf`` is ``(G, G)``.
    G : int
        Number of energy groups.
    scatter_orientation : {"from_to", "to_from"}
        Storage order of ``Scat`` (and ``Scat1``, and a matrix ``nuSigf``).
        ``"from_to"`` (the default, and the usual transport convention) means
        ``Scat[g_from][g_to]``; ``"to_from"`` means the data is already in the
        solver's ``[g_to][g_from]`` order and is used as-is.
    transport_correction : {"auto", "p1", "none"}
        How to obtain ``Sigma_tr``.  See the module docstring.

    Returns
    -------
    dict
        Diffusion cross sections for this material, ready to pass to
        :func:`ndiffusion.make_materials`: ``D``, ``Removal``, ``Scat``
        (``[g_to][g_from]``, diagonal zeroed), ``nuSigf``, and ``chi`` when the
        input had one.  The derived ``Siga``, ``SigT`` and ``SigTr`` are also
        included for inspection.  Because ``Removal`` is present,
        ``make_materials`` takes the data as-is and does not re-zero the
        diagonal.

    Raises
    ------
    ValueError
        On an unknown orientation or correction, a shape mismatch, or a
        non-positive ``Sigma_tr`` (which would give a negative or infinite D).
    KeyError
        When a required cross section is missing.
    """
    if scatter_orientation not in _ORIENTATIONS:
        raise ValueError(
            f"scatter_orientation must be one of {_ORIENTATIONS}, "
            f"got {scatter_orientation!r}."
        )
    if transport_correction not in _CORRECTIONS:
        raise ValueError(
            f"transport_correction must be one of {_CORRECTIONS}, "
            f"got {transport_correction!r}."
        )
    if "Scat" not in data:
        raise KeyError("Missing required scatter matrix 'Scat'.")
    if "nuSigf" not in data:
        raise KeyError("Missing required fission data 'nuSigf'.")

    # Internally everything is [g_from][g_to].
    scat = _oriented(data["Scat"], G, scatter_orientation, "Scat")
    out_scatter = scat.sum(axis=1)   # over g_to, self-scatter included

    if "SigT" in data:
        sig_t = np.asarray(data["SigT"], dtype=float).ravel()
        if sig_t.size != G:
            raise ValueError(f"SigT has {sig_t.size} entries, expected {G}.")
    elif "Siga" in data:
        sig_a = np.asarray(data["Siga"], dtype=float).ravel()
        if sig_a.size != G:
            raise ValueError(f"Siga has {sig_a.size} entries, expected {G}.")
        sig_t = sig_a + out_scatter
    else:
        raise KeyError(
            "Need a total cross section 'SigT' or an absorption cross section "
            "'Siga' (the other is derived from 'Scat')."
        )

    sig_tr = _transport_xs(data, sig_t, G, transport_correction,
                           scatter_orientation)
    if np.any(sig_tr <= 0.0):
        bad = np.flatnonzero(sig_tr <= 0.0).tolist()
        raise ValueError(
            f"Sigma_tr is non-positive in group(s) {bad}, so D = 1/(3*Sigma_tr) "
            "is undefined. Check the transport correction and the input data."
        )

    # Removal uses the uncorrected total: the outflow correction cancels here.
    removal = sig_t - np.diagonal(scat)

    # Solver order is [g_to][g_from], with self-scatter folded into removal.
    scatter = scat.T.copy()
    np.fill_diagonal(scatter, 0.0)

    nusigf = np.asarray(data["nuSigf"], dtype=float)
    if nusigf.shape == (G, G):
        # Fission transfer matrix: reorient like the scatter matrix.
        nusigf = _oriented(nusigf, G, scatter_orientation, "nuSigf").T.copy()
    elif nusigf.size != G:
        raise ValueError(
            f"nuSigf has shape {nusigf.shape}, expected ({G},) or ({G}, {G})."
        )

    out = {
        "D":       (1.0 / (3.0 * sig_tr)).tolist(),
        "Removal": removal.tolist(),
        "Scat":    scatter,
        "nuSigf":  nusigf,
        "Siga":    (sig_t - out_scatter).tolist(),
        "SigT":    sig_t.tolist(),
        "SigTr":   sig_tr.tolist(),
    }
    if "chi" in data:
        chi = np.asarray(data["chi"], dtype=float).ravel()
        if chi.size != G:
            raise ValueError(f"chi has {chi.size} entries, expected {G}.")
        out["chi"] = chi.tolist()
    return out


def make_materials_from_transport(data_list, G, scatter_orientation="from_to",
                                  transport_correction="auto"):
    """Build a :class:`Materials` from a list of *transport* cross-section dicts.

    The transport analogue of :func:`ndiffusion.make_materials`: each entry of
    *data_list* is converted by :func:`transport_to_diffusion` and the results
    are assembled in list order, so material ``m`` is index ``m`` in the
    ``medium_map`` / ``mesh.material_id``.

    Parameters
    ----------
    data_list : list of dict-like
        Ordered transport cross sections, one per material.  See
        :func:`transport_to_diffusion` for the recognised keys.
    G : int
        Number of energy groups.
    scatter_orientation : {"from_to", "to_from"}
        Storage order of the input matrices; see
        :func:`transport_to_diffusion`.
    transport_correction : {"auto", "p1", "none"}
        How to obtain ``Sigma_tr``; see :func:`transport_to_diffusion`.

    Returns
    -------
    Materials
        Fully configured, ready to pass to any solver.
    """
    from ndiffusion.create import make_materials

    return make_materials(
        [
            transport_to_diffusion(
                data, G,
                scatter_orientation=scatter_orientation,
                transport_correction=transport_correction,
            )
            for data in data_list
        ],
        G,
    )
