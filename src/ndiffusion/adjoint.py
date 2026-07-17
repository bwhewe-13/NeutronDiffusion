"""Adjoint (importance) transform for the multigroup diffusion operator.

The multigroup diffusion loss operator

    M_{g,g'} = [ -div(D_g grad) + Sigma_r,g ] delta_{g,g'}  -  Sigma_s,g<-g'

and the fission production operator

    F_{g,g'} = chi_g * nu_Sigma_f,g'

define the forward k-eigenvalue problem  M phi = (1/k) F phi.  The adjoint
(importance) problem is  M^dagger phi^dagger = (1/k) F^dagger phi^dagger, where
M^dagger and F^dagger are the transposes of M and F over the energy-group index.

Two facts make the adjoint a pure :class:`Materials` transform - no solver code
changes are needed:

* The spatial part ``-div(D grad) + Sigma_r`` is **self-adjoint** for the Robin
  boundary conditions used here, so ``D``, ``removal``, and the ``bc`` arrays are
  unchanged.  Only the off-diagonal group coupling transposes.
* Transposing the group coupling means:

  - **scatter** transposes: ``scatter[m][g_to][g_from] -> scatter[m][g_from][g_to]``.
  - **fission** transposes: ``F^dagger_{g,g'} = F_{g',g} = chi_{g'} nu_Sigma_f,g``.
    In the standard ``chi`` / ``nusigf`` vector representation this is exactly a
    **swap of chi and nusigf**.  In fission-matrix mode (``chi`` all zeros,
    ``nusigf`` sized ``n_mat*G*G``) the ``nusigf`` matrix transposes instead.

The k-eigenvalue is identical for the forward and adjoint problems; the adjoint
flux is the neutron *importance* function used in perturbation theory and
detector-response calculations.
"""

import numpy as np


def make_adjoint_materials(mats):
    """Return a new :class:`Materials` for the adjoint diffusion problem.

    Running any existing solver (k-eigenvalue, fixed-source, or time-dependent,
    in any geometry) on the returned materials solves the corresponding adjoint
    equation.  The original ``mats`` is not modified.

    Parameters
    ----------
    mats : ndiffusion.Materials
        Forward-problem cross sections.

    Returns
    -------
    ndiffusion.Materials
        Adjoint cross sections: scatter transposed over the group indices and,
        for fission, chi/nusigf swapped (standard mode) or the nusigf matrix
        transposed (fission-matrix mode).  ``D``, ``removal``, and ``velocity``
        are copied unchanged.

    Notes
    -----
    Boundary conditions are unchanged - pass the same ``bc`` arrays to the
    solver.  For fixed-source adjoint problems the *source* is the adjoint
    source (e.g. a detector response cross section), which the caller supplies.
    """
    from ndiffusion._core import Materials

    n_mat = mats.n_mat
    G = mats.n_groups

    scatter = np.asarray(mats.scatter, dtype=float).reshape(n_mat, G, G)
    # Transpose g_to <-> g_from within each material.
    scatter_adj = np.transpose(scatter, (0, 2, 1)).copy()

    chi = np.asarray(mats.chi, dtype=float)
    nusigf = np.asarray(mats.nusigf, dtype=float)

    fission_matrix_mode = (
        nusigf.size == n_mat * G * G and not np.any(chi != 0.0)
    )

    if fission_matrix_mode:
        # F[g_to][g_from] -> F[g_from][g_to] per material; chi stays zero.
        nusigf_adj = np.transpose(
            nusigf.reshape(n_mat, G, G), (0, 2, 1)
        ).ravel()
        chi_adj = chi.copy()
    else:
        # F_{g,g'} = chi_g * nusigf_g'  ->  F^dagger = nusigf_g * chi_g':
        # swap the chi and nusigf vectors.
        chi_adj = nusigf.copy()
        nusigf_adj = chi.copy()

    m = Materials()
    m.n_mat = n_mat
    m.n_groups = G
    m.D = list(np.asarray(mats.D, dtype=float))
    m.removal = list(np.asarray(mats.removal, dtype=float))
    m.scatter = list(scatter_adj.ravel())
    m.chi = list(chi_adj)
    m.nusigf = list(nusigf_adj)
    if len(mats.velocity) > 0:
        m.velocity = list(np.asarray(mats.velocity, dtype=float))
    return m
