"""Utilities for loading neutron diffusion problem data."""

import json
from pathlib import Path

import numpy as np


def _candidate_data_paths():
    # Primary: data/ directory co-located with this package
    package_data = Path(__file__).resolve().parent / "data"
    # Fallback: repo-root data/ for cross-section files not shipped with the package
    repo_data = Path(__file__).resolve().parent.parent.parent / "data"

    seen = set()
    for path in (package_data, repo_data):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            yield path


def _find_data_file(filename):
    for data_path in _candidate_data_paths():
        candidate = data_path / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate data file '{filename}'.")


with _find_data_file("problem_setup.json").open("r", encoding="utf-8") as handle:
    problem_dictionary = json.load(handle)


def selection(name, G, I, BC=0):
    """Load a named problem configuration.

    Parameters
    ----------
    name : str
        Problem name from problem_setup.json.
    G : int
        Number of energy groups.
    I : int
        Number of spatial cells.
    BC : float
        Albedo value (0 = vacuum, 1 = reflective, 0 < BC < 1 = albedo).

    Returns
    -------
    tuple
        (G, R, I, dg, scatter, chi, fission, removal, bounds)
    """
    problem = problem_dictionary[name]

    R = []
    dg = []
    scatter = []
    chi = []
    fission = []
    removal = []

    for mat, r in problem.items():
        t1, t2, t3, t4, t5 = loading_data(G, mat)
        dg.append(t1)
        scatter.append(t2)
        chi.append(t3)
        fission.append(t4)
        removal.append(t5)
        R.append(r)
    bounds = boundary_conditions(dg[-1], alpha=BC)
    dg = np.array(dg)
    scatter = np.array(scatter)
    chi = np.array(chi)
    fission = np.array(fission)
    removal = np.array(removal)
    if np.all(chi == None):
        return G, R, I, dg, scatter, None, fission, removal, bounds
    return G, R, I, dg, scatter, chi, fission, removal, bounds


def loading_data(G, material):
    """Load cross-section data for a material.

    Tries .npz files first, then falls back to individual .csv files.

    Parameters
    ----------
    G : int
        Number of energy groups.
    material : str
        Material name (used to locate data files).

    Returns
    -------
    tuple
        (dg, scatter, chi, fission, removal)
    """
    try:
        data_dict = np.load(_find_data_file(f"{material}_{G}G.npz"))
        dg = data_dict["D"].copy()
        absorb = data_dict["Siga"].copy()
        scatter = data_dict["Scat"].copy()
        centers = data_dict["group_centers"].copy()
    except FileNotFoundError:
        dg = np.loadtxt(_find_data_file(f"D_{G}G_{material}.csv"))
        absorb = np.loadtxt(_find_data_file(f"Siga_{G}G_{material}.csv"))
        scatter = np.loadtxt(
            _find_data_file(f"Scat_{G}G_{material}.csv"), delimiter=","
        )
        centers = np.loadtxt(
            _find_data_file(f"group_centers_{G}G_{material}.csv"), delimiter=","
        )
    try:
        data_dict = np.load(_find_data_file(f"{material}_{G}G.npz"))
        removal = data_dict["Removal"].copy()
    except KeyError:
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

    try:
        data_dict = np.load(_find_data_file(f"{material}_{G}G.npz"))
        fission = data_dict["nuSigf"].copy()
        try:
            chi = data_dict["chi"].copy()
        except KeyError:
            chi = None
    except FileNotFoundError:
        try:
            chi = np.loadtxt(_find_data_file(f"chi_{G}G_{material}.csv"))
            fission = np.loadtxt(_find_data_file(f"nuSigf_{G}G_{material}.csv"))
        except OSError:
            chi = None
            fission = np.loadtxt(
                _find_data_file(f"nuSigf_{G}G_{material}.csv"), delimiter=","
            )
    return dg, scatter, chi, fission, removal


def boundary_conditions(Dg, alpha):
    """Compute Robin boundary condition coefficients.

    Marshak formula:
        A = (1 - alpha) / (4 * (1 + alpha))
        B = D / 2

    Parameters
    ----------
    Dg : array-like
        Diffusion coefficients per energy group.
    alpha : float
        Albedo (0 = vacuum, 1 = reflective, 0 < alpha < 1 = partial reflection).

    Returns
    -------
    numpy.ndarray, shape (n_groups, 2)
        Each row is [A, B] for the corresponding energy group.
    """
    A = lambda alpha: (1 - alpha) / (4 * (1 + alpha))
    bounds = np.zeros((len(Dg), 2)) + A(alpha)
    bounds[:, 1] = 0.5 * Dg
    if alpha == 1:
        bounds[:, 1] = 1
    return bounds
