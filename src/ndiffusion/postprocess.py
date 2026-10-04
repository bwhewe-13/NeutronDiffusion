"""Post-processing for solver results: volumes, reaction rates and power.

The solvers return a flux shape normalized however the iteration left it (a
k-eigenvalue flux has an arbitrary amplitude), per unit volume, on whatever
mesh was used.  The helpers here turn that into the quantities usually
reported from a core calculation:

    vol = nd.cell_volumes(edges_x, edges_y=edges_y, geom=nd.Geometry2D.XY)
    flux = nd.normalize_to_power(res.flux, mats, medium_map, vol, 3000e6)
    q = nd.power_density(flux, mats, medium_map)
    assembly_power = nd.region_powers(q, vol, assembly_id)
    fq = nd.peaking_factors(q, vol)

Units follow the inputs: with cross sections in 1/cm, volumes in cm^3 and the
energy per fission in J, a power in W gives a flux in n/cm^2/s.  A 1-D slab
volume is per unit transverse area and a 1-D cylinder or 2-D XY volume per unit
height, so the total power is too.

Pure Python - no rebuild is needed after editing this module.
"""

import json
from types import SimpleNamespace

import numpy as np

_MEV = 1.602176634e-13  # J

#: Recoverable energy per fission (J), the usual 200 MeV for U-235.
KAPPA_U235 = 200.0 * _MEV
#: Neutrons per thermal fission of U-235.
NU_U235 = 2.43

_KINDS = ("absorption", "removal", "nu-fission", "fission")

# Result attributes written by save_result, in the order they are looked for.
# Each result type carries a subset; missing ones are skipped.
_RESULT_FIELDS = (
    "flux", "precursors", "keff", "iterations", "residual", "converged",
    "time", "steps", "n_groups", "n_precursor",
)


def cell_volumes(edges_x, geom=None, edges_y=None):
    """Cell volumes, in the same order as the solver's cells.

    Parameters
    ----------
    edges_x : array-like or UnstructuredMesh2D
        Cell edges as passed to the solver.  An unstructured mesh returns its
        cell areas (volume per unit depth), the same ones the solver uses.
    geom : Geometry or Geometry2D
        ``Geometry.Slab`` (the default for 1-D), ``Cylinder`` or ``Sphere``;
        ``Geometry2D.XY`` or ``RZ`` together with *edges_y*.
    edges_y : array-like, optional
        Second set of edges for a 2-D structured mesh.  For RZ, *edges_x* is
        the axial z and *edges_y* the radius, as for the solvers.

    Returns
    -------
    numpy.ndarray
        ``(n_cells,)``.  2-D structured cell ``(i, j)`` is entry ``i * ny + j``.
    """
    from ndiffusion._core import Geometry, Geometry2D, UnstructuredMesh2D, cell_areas

    if isinstance(edges_x, UnstructuredMesh2D):
        return cell_areas(edges_x)

    ex = np.asarray(edges_x, dtype=float)
    if edges_y is None:
        if isinstance(geom, Geometry2D):
            raise ValueError(f"{geom} needs edges_y")
        if geom is None or geom == Geometry.Slab:
            return np.diff(ex)
        if geom == Geometry.Cylinder:
            return np.pi * np.diff(ex**2)
        return (4.0 / 3.0) * np.pi * np.diff(ex**3)

    ey = np.asarray(edges_y, dtype=float)
    if geom is None or geom == Geometry2D.XY:
        return np.outer(np.diff(ex), np.diff(ey)).ravel()
    if geom == Geometry2D.RZ:
        return np.outer(np.diff(ex), np.pi * np.diff(ey**2)).ravel()
    raise ValueError(f"edges_y was given, so geom must be a Geometry2D, got {geom}")


def _cell_xs(mats, material_map, n_cells):
    """Per-cell cross sections, each ``(n_cells, G)``, in solver group order."""
    n_mat, G = mats.n_mat, mats.n_groups
    ids = np.asarray(material_map, dtype=int).ravel()
    if ids.size != n_cells:
        raise ValueError(
            f"material_map has {ids.size} entries but the flux has {n_cells} cells")
    if ids.size and (ids.min() < 0 or ids.max() >= n_mat):
        raise ValueError(f"material_map holds ids outside [0, {n_mat})")

    removal = np.asarray(mats.removal, dtype=float).reshape(n_mat, G)
    scatter = np.asarray(mats.scatter, dtype=float).reshape(n_mat, G, G)
    nusigf = np.asarray(mats.nusigf, dtype=float)
    if nusigf.size == n_mat * G * G and not np.any(np.asarray(mats.chi)):
        # Fission-matrix mode: neutrons produced per fission in g' are the
        # column sum over the emitted group, as production_xs does in C++.
        nusigf = nusigf.reshape(n_mat, G, G).sum(axis=1)
    else:
        nusigf = nusigf.reshape(n_mat, G)

    # removal = absorption + out-scatter, with the self-scatter diagonal
    # already zeroed, so the column sum is exactly the out-scatter.
    absorption = removal - scatter.sum(axis=1)
    return {
        "removal": removal[ids],
        "absorption": absorption[ids],
        "nu-fission": nusigf[ids],
    }


def _as_table(flux, G):
    phi = np.asarray(flux, dtype=float)
    if phi.ndim == 2 and phi.shape[1] != G:
        raise ValueError(f"flux has {phi.shape[1]} columns but mats has {G} groups")
    return phi.reshape(-1, G)


def reaction_rate(flux, mats, material_map, kind="nu-fission", nu=NU_U235,
                  by_group=False):
    """Reaction rate density ``Sigma phi`` per cell.

    Parameters
    ----------
    flux : array-like
        ``(n_cells, G)`` as the solvers return it, or flat.
    mats : Materials
    material_map : array-like of int
        Material index per cell - the solver's ``medium_map``, or
        ``mesh.material_id`` on an unstructured mesh.
    kind : {"nu-fission", "fission", "absorption", "removal"}
        ``fission`` is ``nusigf / nu``, since ``Materials`` stores only the
        product.  ``absorption`` is ``removal`` less the out-scatter.
    nu : float
        Neutrons per fission, used only for ``kind="fission"``.
    by_group : bool
        Return ``(n_cells, G)`` instead of summing over groups.

    Returns
    -------
    numpy.ndarray
        ``(n_cells,)``, or ``(n_cells, G)`` with *by_group*.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    phi = _as_table(flux, mats.n_groups)
    xs = _cell_xs(mats, material_map, phi.shape[0])
    if kind == "fission":
        rate = xs["nu-fission"] * phi / nu
    else:
        rate = xs[kind] * phi
    return rate if by_group else rate.sum(axis=1)


def power_density(flux, mats, material_map, kappa=KAPPA_U235, nu=NU_U235):
    """Fission power density ``kappa * Sigma_f * phi`` per cell.

    *kappa* is the energy per fission and *nu* the neutrons per fission;
    together they convert the stored ``nusigf`` into a power.  Returns
    ``(n_cells,)``.
    """
    return kappa * reaction_rate(flux, mats, material_map, "fission", nu=nu)


def normalize_to_power(flux, mats, material_map, volumes, total_power,
                       kappa=KAPPA_U235, nu=NU_U235):
    """Scale *flux* so the fission power integrates to *total_power*.

    *volumes* comes from :func:`cell_volumes`.  The returned flux has the same
    shape as the input.  A k-eigenvalue flux is a shape only, so this is the
    step that gives it units.

    Raises
    ------
    ValueError
        If the flux produces no fission power to scale.
    """
    phi = np.asarray(flux, dtype=float)
    q = power_density(phi, mats, material_map, kappa=kappa, nu=nu)
    vol = np.asarray(volumes, dtype=float).ravel()
    if vol.size != q.size:
        raise ValueError(f"volumes has {vol.size} entries but the flux has {q.size} cells")
    power = float(np.dot(q, vol))
    if not power > 0.0:
        raise ValueError("the flux produces no fission power to normalize")
    return phi * (total_power / power)


def region_powers(density, volumes, regions, n_regions=None):
    """Integrate a per-cell density over regions.

    Parameters
    ----------
    density : array-like
        ``(n_cells,)`` - typically :func:`power_density`, but any density works
        (a reaction rate gives a region reaction rate).
    volumes : array-like
        ``(n_cells,)`` from :func:`cell_volumes`.
    regions : array-like of int
        Region id per cell: the material map for powers by material, or any
        other labeling (assembly, ring, ...).
    n_regions : int, optional
        Length of the result; defaults to ``max(regions) + 1``.

    Returns
    -------
    numpy.ndarray
        ``(n_regions,)``; entry ``r`` is the integral over cells labeled ``r``.
    """
    q = np.asarray(density, dtype=float).ravel()
    vol = np.asarray(volumes, dtype=float).ravel()
    ids = np.asarray(regions, dtype=int).ravel()
    if not q.size == vol.size == ids.size:
        raise ValueError(
            f"density, volumes and regions must have one entry per cell, got "
            f"{q.size}, {vol.size} and {ids.size}")
    return np.bincount(ids, weights=q * vol, minlength=n_regions or 0)


def peaking_factors(density, volumes, regions=None):
    """Peak-to-average ratio of a power density.

    The average is taken over the cells that produce power, so a reflector
    does not dilute it.  With *regions*, each region's density is first
    averaged over its volume and the ratio is between regions - an assembly
    or pin peaking factor rather than a local one.

    Returns
    -------
    float
    """
    q = np.asarray(density, dtype=float).ravel()
    vol = np.asarray(volumes, dtype=float).ravel()
    if regions is not None:
        ids = np.asarray(regions, dtype=int).ravel()
        region_vol = region_powers(np.ones_like(q), vol, ids)
        hot = region_powers(q, vol, ids)
        has_volume = region_vol > 0.0
        q = hot[has_volume] / region_vol[has_volume]
        vol = region_vol[has_volume]
    fueled = q > 0.0
    if not np.any(fueled):
        raise ValueError("no cell produces power")
    average = np.dot(q[fueled], vol[fueled]) / vol[fueled].sum()
    return float(q.max() / average)


def save_result(path, result, solver=None, **metadata):
    """Write a solver result to an ``.npz`` file.

    Every field the result carries is saved (``flux``, ``keff``,
    ``precursors``, ...), along with the ndiffusion version, the result type
    and, if *solver* is given, its class name and size.  Extra keyword
    arguments are stored as metadata and must be JSON serializable.
    """
    from ndiffusion import __version__

    arrays = {f: np.asarray(getattr(result, f)) for f in _RESULT_FIELDS
              if hasattr(result, f)}
    meta = {
        "ndiffusion_version": __version__,
        "result_type": type(result).__name__,
    }
    if solver is not None:
        meta["solver"] = type(solver).__name__
        meta["n_cells"] = solver.n_cells
        meta["n_groups"] = solver.n_groups
    meta.update(metadata)
    np.savez(path, metadata=np.array(json.dumps(meta)), **arrays)


def load_result(path):
    """Read a result written by :func:`save_result`.

    The compiled result classes cannot be constructed from Python, so this
    returns a namespace with the same attribute names plus a ``metadata``
    dict.  It works anywhere the original result's fields are read.
    """
    with np.load(path, allow_pickle=False) as data:
        fields = {}
        for name in data.files:
            value = data[name]
            fields[name] = value.item() if value.ndim == 0 else value
    fields["metadata"] = json.loads(fields["metadata"])
    return SimpleNamespace(**fields)
