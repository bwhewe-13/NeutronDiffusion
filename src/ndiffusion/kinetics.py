"""Helpers for reactor kinetics: delayed neutron data and critical scaling.

The time-dependent solvers take a :class:`DelayedNeutronData` describing the
delayed neutron precursor groups.  ``make_delayed_data`` builds one from compact
per-material specifications, the way ``create.make_materials`` builds a
``Materials`` from cross-section dicts.

Pure Python - no rebuild is needed after editing this module.
"""

import numpy as np

#: Standard 6-group delayed neutron parameters for thermal fission in U-235
#: (Keepin, *Physics of Nuclear Kinetics*, 1965).  ``Beta`` entries are absolute
#: delayed fractions, summing to 0.0065.
DELAYED_U235_6GROUP = {
    "Beta": [
        0.000215,
        0.001424,
        0.001274,
        0.002568,
        0.000748,
        0.000273,
    ],
    "Lambda": [
        0.0124,
        0.0305,
        0.111,
        0.301,
        1.14,
        3.01,
    ],
}


def make_delayed_data(data_list, G, n_mat=None, chi=None):
    """Build a ``DelayedNeutronData`` from per-material precursor specs.

    Parameters
    ----------
    data_list : dict or list of dict
        One spec per material, or a single spec broadcast to every material
        (the common case - one fuel kinetics parameter set used throughout).
        Recognized keys:

        ``Beta``
            Delayed fraction per precursor group, length ``n_precursor``.
            Required.
        ``Lambda``
            Decay constant (1/s) per precursor group, length ``n_precursor``.
            Required, and must be identical across materials - the decay
            constants are properties of the precursor nuclides, not the fuel.
        ``ChiDelayed``
            Delayed fission spectrum, shape ``(n_precursor, G)`` or ``(G,)``
            (broadcast to every precursor group).  Each row must sum to 1 - a
            decaying precursor emits exactly one neutron.  Defaults to *chi*.
        ``ChiPrompt``
            Prompt fission spectrum, shape ``(G,)``.  Optional; when omitted
            the solver *derives* it as
            ``(chi - sum_i beta_i chi_d,i) / (1 - beta)`` so that prompt and
            delayed parts sum back to the total ``Materials.chi``.  Supply it
            explicitly only when ``Materials.chi`` is itself the prompt
            spectrum rather than the total.  All-or-nothing: the solver takes one
            ``chi_prompt`` array covering every material and cannot derive
            individual rows, so either all materials supply it or none do.

        A non-fissile material is written as an all-zero ``Beta``; it then
        produces no precursors regardless of the other entries.
    G : int
        Number of energy groups.
    n_mat : int or None
        Number of materials.  Required when *data_list* is a single spec to
        broadcast; inferred from the list length otherwise.
    chi : array-like or None
        Fission spectrum to use as the default ``ChiDelayed``, flat
        ``[n_mat * G]`` (i.e. ``Materials.chi``) or ``(G,)``.  Required only
        when a material omits ``ChiDelayed``.

    Returns
    -------
    DelayedNeutronData
        Ready to pass as the ``delayed`` argument of any time-dependent solver.

    Raises
    ------
    ValueError
        If required keys are missing, array shapes disagree, the precursor
        count is inconsistent between materials, the decay constants differ
        between materials, or ``ChiPrompt`` is supplied for only some materials.

    Notes
    -----
    In **fission-matrix mode** ``Materials.chi`` is all zeros, so it cannot
    serve as the ``ChiDelayed`` fallback - give ``ChiDelayed`` explicitly there.
    The solver applies the prompt/delayed split to the fission matrix itself,
    using its column sums as the production cross section.
    """
    from ndiffusion._core import DelayedNeutronData

    if isinstance(data_list, dict):
        if n_mat is None:
            raise ValueError("n_mat is required when broadcasting a single spec")
        data_list = [data_list] * n_mat
    else:
        data_list = list(data_list)
        if n_mat is None:
            n_mat = len(data_list)
        elif len(data_list) != n_mat:
            raise ValueError(
                f"data_list has {len(data_list)} entries but n_mat is {n_mat}"
            )
    if n_mat < 1:
        raise ValueError("n_mat must be at least 1")

    chi_flat = None
    if chi is not None:
        chi_arr = np.asarray(chi, dtype=float).ravel()
        if chi_arr.size == G:
            chi_flat = np.tile(chi_arr, (n_mat, 1))
        elif chi_arr.size == n_mat * G:
            chi_flat = chi_arr.reshape(n_mat, G)
        else:
            raise ValueError(
                f"chi must have {G} or {n_mat * G} elements, got {chi_arr.size}"
            )

    n_precursor = None
    lam_ref = None
    beta_rows, chi_d_rows, chi_p_rows = [], [], []
    # chi_prompt is one flat array over all materials, so it is emitted only when
    # every material declared one; a partial array would give the rest a wrong
    # prompt spectrum rather than the derivation they expect.
    chi_prompt_given = []

    for m, data in enumerate(data_list):
        if "Beta" not in data or "Lambda" not in data:
            raise ValueError(f"material {m}: both 'Beta' and 'Lambda' are required")

        beta = np.asarray(data["Beta"], dtype=float).ravel()
        lam = np.asarray(data["Lambda"], dtype=float).ravel()
        if beta.size != lam.size:
            raise ValueError(
                f"material {m}: Beta has {beta.size} entries but Lambda has "
                f"{lam.size}; both must be length n_precursor"
            )

        if n_precursor is None:
            n_precursor, lam_ref = beta.size, lam
        elif beta.size != n_precursor:
            raise ValueError(
                f"material {m}: {beta.size} precursor groups, but material 0 has "
                f"{n_precursor}; every material needs the same precursor count"
            )
        elif not np.allclose(lam, lam_ref):
            raise ValueError(
                f"material {m}: Lambda differs from material 0; decay constants "
                "are nuclide properties and must be the same for all materials"
            )

        # Delayed spectrum: (n_precursor, G), or one (G,) row broadcast.
        if "ChiDelayed" in data:
            chi_d = np.asarray(data["ChiDelayed"], dtype=float)
            if chi_d.size == G:
                chi_d = np.tile(chi_d.ravel(), (n_precursor, 1))
            elif chi_d.size == n_precursor * G:
                chi_d = chi_d.reshape(n_precursor, G)
            else:
                raise ValueError(
                    f"material {m}: ChiDelayed must have {G} or "
                    f"{n_precursor * G} elements, got {chi_d.size}"
                )
        elif chi_flat is not None:
            chi_d = np.tile(chi_flat[m], (n_precursor, 1))
        else:
            raise ValueError(
                f"material {m}: no 'ChiDelayed' and no 'chi' fallback supplied; "
                "pass chi=mats.chi to default the delayed spectrum to the "
                "steady-state fission spectrum"
            )

        if "ChiPrompt" in data:
            chi_p = np.asarray(data["ChiPrompt"], dtype=float).ravel()
            if chi_p.size != G:
                raise ValueError(
                    f"material {m}: ChiPrompt must have {G} elements, "
                    f"got {chi_p.size}"
                )
            chi_prompt_given.append(True)
        else:
            chi_p = np.zeros(G)  # unused unless every material declared one
            chi_prompt_given.append(False)

        # Caught here rather than in the solver because the usual cause is a
        # silent one: defaulting ChiDelayed from an all-zero fission-matrix chi.
        row_sums = chi_d.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError(
                f"material {m}: each ChiDelayed row must sum to 1 (a decaying "
                f"precursor emits one neutron), got {row_sums.tolist()}. In "
                "fission-matrix mode Materials.chi is all zeros and cannot be "
                "used as the fallback - pass ChiDelayed explicitly"
            )

        beta_rows.append(beta)
        chi_d_rows.append(chi_d)
        chi_p_rows.append(chi_p)

    if any(chi_prompt_given) and not all(chi_prompt_given):
        given = [m for m, ok in enumerate(chi_prompt_given) if ok]
        missing = [m for m, ok in enumerate(chi_prompt_given) if not ok]
        raise ValueError(
            f"ChiPrompt was supplied for material(s) {given} but not for "
            f"{missing}. The solver takes one chi_prompt array spanning all "
            "materials, so it cannot derive just the missing rows: supply "
            "ChiPrompt for every material, or for none and let the solver "
            "derive it from Materials.chi as "
            "(chi - sum_i beta_i chi_d,i) / (1 - beta)."
        )

    delayed = DelayedNeutronData()
    delayed.n_precursor = int(n_precursor)
    delayed.lambda_ = lam_ref.tolist()
    delayed.beta = np.concatenate(beta_rows).tolist()
    delayed.chi_delayed = np.concatenate([r.ravel() for r in chi_d_rows]).tolist()
    # Leave chi_prompt empty unless every material gave one - the solver then
    # derives the prompt spectrum from Materials.chi per material.
    delayed.chi_prompt = (
        np.concatenate(chi_p_rows).tolist() if all(chi_prompt_given) else []
    )
    return delayed


def scale_to_critical(mats, keff):
    """Return a copy of *mats* with ``nusigf`` divided by *keff*.

    A transient should start from a genuine steady state.  The flux from a
    k-eigenvalue solve satisfies ``A phi = (1/k) B phi``, so unless ``k`` is
    exactly 1 it is *not* a steady state of the time-dependent equation and the
    "unperturbed" transient drifts from the first step - which makes any
    reactivity insertion measured against it meaningless.  Scaling ``nusigf`` by
    ``1/k`` makes the same flux an exact fixed point.

    Parameters
    ----------
    mats : Materials
        Forward cross sections, typically the ones just handed to a
        k-eigenvalue solver.
    keff : float
        Eigenvalue from that solve.

    Returns
    -------
    Materials
        A new ``Materials``; the input is not modified.

    Raises
    ------
    ValueError
        If *keff* is not positive.
    """
    # Built field by field: copy.copy on a pybind11 class shares the underlying
    # C++ object, so the caller's Materials would be mutated too.
    from ndiffusion._core import Materials

    if keff <= 0.0:
        raise ValueError(f"keff must be positive, got {keff}")

    out = Materials()
    out.n_mat = mats.n_mat
    out.n_groups = mats.n_groups
    out.D = list(mats.D)
    out.removal = list(mats.removal)
    out.scatter = list(mats.scatter)
    out.chi = list(mats.chi)
    out.nusigf = (np.asarray(mats.nusigf, dtype=float) / keff).tolist()
    out.velocity = list(mats.velocity)
    return out
