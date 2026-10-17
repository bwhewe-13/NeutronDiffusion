"""Method of nearby problems (MNP) for multigroup neutron diffusion.

MNP is a solution-verification technique that estimates the spatial
discretization error of a numerical solution without an analytic reference.
The recipe (see ``ants/nearby1d.pyx`` for the transport analog):

1. Solve the numerical problem for the flux on the mesh.
2. Fit a smooth, twice-differentiable curve through the flux, **block-wise per
   material** so the fit never differentiates across a cross-section jump.
3. Substitute the fit into the *continuous* diffusion operator to obtain a
   **residual source** ``r = L[phi_cf] - S``.  The curve fit ``phi_cf`` is then
   the exact solution of the "nearby problem" ``L[phi] = S + r``.
4. Re-solve the nearby problem numerically.  Because ``phi_cf`` is its exact
   solution, ``phi_nearby - phi_cf`` estimates the true discretization error
   ``phi_num - phi_exact``.

The continuous fixed-source diffusion operator (matching the ``FixedSource*``
solvers, which carry no fission term) is, per group ``g``:

    L[phi]_g = -div(D_g grad phi_g) + Sigma_r,g phi_g
               - sum_{g'!=g} Sigma_s,g<-g' phi_g'

The k-eigenvalue variant adds the fission production and estimates the
eigenvalue error via a curve-fit ``k`` and a nearby ``k``.

The leakage term ``-div(D grad phi)`` needs the curve fit's second derivatives;
these come from quintic/cubic splines on structured meshes (via scipy). On the
unstructured mesh, given the boundary conditions, they come from one constrained
quintic tensor spline per material region, fit to all regions and groups at once
with flux and current continuity at the interfaces and the Robin conditions on
the boundary (``_spline_fit``); without them, from a per-cell least-squares
polynomial reconstruction (``fit="local"``).

For a k-eigenvalue problem the eigenvalue error can also be estimated in one
shot from the adjoint flux, ``sum phi+ (k L phi_cf - F phi_cf) V / sum phi+ L
phi_cf V``, which skips the nearby power iteration.

Everything here is orchestration in Python: the actual solves reuse the compiled
``FixedSourceSolver*`` classes, so no C++ changes are required.  Sources are
per-unit-volume throughout (the solvers integrate internally), and the residual
is the continuous operator evaluated pointwise at cell centers - one definition
for all three geometries.
"""

from collections import defaultdict, namedtuple

import numpy as np

from ndiffusion.postprocess import cell_volumes

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

NearbyFixedResult = namedtuple(
    "NearbyFixedResult",
    ["numerical", "curve_fit", "residual", "nearby", "error_estimate"],
)

NearbyKResult = namedtuple(
    "NearbyKResult",
    [
        "numerical",
        "curve_fit",
        "k_curve_fit",
        "nearby_flux",
        "k_nearby",
        "residual",
        "nearby_rate",
        "k_error_adjoint",
        "k_error_cells",
        "fit",
        "operator",
    ],
    defaults=(None, None, None, None),
)

NearbyQoIResult = namedtuple(
    "NearbyQoIResult",
    ["value", "fit_value", "error_adjoint", "error_nearby", "cells"],
)


# ---------------------------------------------------------------------------
# scipy is an optional dependency (extra: `pip install ndiffusion[nearby]`)
# ---------------------------------------------------------------------------


def _make_interp_spline():
    try:
        from scipy.interpolate import make_interp_spline
    except ImportError as exc:  # pragma: no cover - exercised only without scipy
        raise ImportError(
            "The method of nearby problems needs SciPy for the structured-grid "
            "curve fit. Install it with `pip install ndiffusion[nearby]` (or "
            "`pip install scipy`)."
        ) from exc
    return make_interp_spline


# ---------------------------------------------------------------------------
# Cross-section helpers (numpy views of the flat Materials arrays)
# ---------------------------------------------------------------------------


def _xs_arrays(mats):
    """Return (D, removal, chi, nusigf, scatter) reshaped for numpy indexing.

    D, removal, chi: (n_mat, G).  scatter: (n_mat, G, G) as [g_to][g_from].
    nusigf: (n_mat, G) in standard mode or (n_mat, G, G) in fission-matrix mode.
    """
    n_mat, G = mats.n_mat, mats.n_groups
    D = np.asarray(mats.D, dtype=float).reshape(n_mat, G)
    removal = np.asarray(mats.removal, dtype=float).reshape(n_mat, G)
    chi = np.asarray(mats.chi, dtype=float).reshape(n_mat, G)
    scatter = np.asarray(mats.scatter, dtype=float).reshape(n_mat, G, G)
    nusigf = np.asarray(mats.nusigf, dtype=float)
    if _fission_matrix_mode(mats):  # by size alone, one group would look like a matrix
        nusigf = nusigf.reshape(n_mat, G, G)
    else:
        nusigf = nusigf.reshape(n_mat, G)
    return D, removal, chi, nusigf, scatter


def _fission_matrix_mode(mats):
    n_mat, G = mats.n_mat, mats.n_groups
    chi = np.asarray(mats.chi, dtype=float)
    nusigf = np.asarray(mats.nusigf, dtype=float)
    return nusigf.size == n_mat * G * G and not np.any(chi != 0.0)


def _fission_apply(mats, mat_ids, curve):
    """Fission production ``F phi`` per cell/group (per unit volume).

    Mirrors ``accumulate_fission`` in solver_detail.hpp: handles both the
    standard ``chi_g * nu_sigf_g'`` product and the full fission transfer matrix.
    ``curve`` is (n_cells, G); returns (n_cells, G).
    """
    _, _, chi, nusigf, _ = _xs_arrays(mats)
    if _fission_matrix_mode(mats):
        F = nusigf[mat_ids]  # (n_cells, G, G) as [g_to][g_from]
        return np.einsum("cij,cj->ci", F, curve)
    chi_c = chi[mat_ids]  # (n_cells, G)
    nsf = nusigf[mat_ids]  # (n_cells, G)
    fis_rate = np.sum(nsf * curve, axis=1)  # (n_cells,)
    return chi_c * fis_rate[:, None]


def fission_source(mats, medium_map, flux):
    """Public helper: fission production ``F phi`` for a solver flux.

    Parameters
    ----------
    mats : Materials
    medium_map : sequence of int
        Material index per cell (``mesh.material_id`` for unstructured).
    flux : array-like
        Flux shaped ``(n_cells, G)`` as the solvers return it, or flat in the
        same row-major order.

    Returns
    -------
    numpy.ndarray
        ``(n_cells, G)`` fission source, per unit volume.
    """
    mat_ids = np.asarray(medium_map, dtype=int)
    G = mats.n_groups
    curve = np.asarray(flux, dtype=float).reshape(-1, G)
    return _fission_apply(mats, mat_ids, curve)


def _assemble_loss(mats, mat_ids, curve, lap):
    """Continuous fixed-source operator L[phi] = -D*lap + Sigma_r*phi - inscatter.

    ``curve`` and ``lap`` (the geometry Laplacian of the fit) are (n_cells, G).
    Returns the per-unit-volume loss (n_cells, G).
    """
    D, removal, _, _, scatter = _xs_arrays(mats)
    G = mats.n_groups
    Dc = D[mat_ids]  # (n_cells, G)
    remc = removal[mat_ids]  # (n_cells, G)

    # In-scatter: exclude the self-scatter diagonal (it lives in `removal`).
    Soff = scatter[mat_ids].copy()  # (n_cells, G, G) as [g_to][g_from]
    diag = np.arange(G)
    Soff[:, diag, diag] = 0.0
    inscatter = np.einsum("cij,cj->ci", Soff, curve)

    return -Dc * lap + remc * curve - inscatter


# ---------------------------------------------------------------------------
# Block-wise 1-D spline derivatives
# ---------------------------------------------------------------------------


def _contiguous_blocks(mat_line):
    """Yield (lo, hi) half-open index ranges of constant material."""
    mat_line = np.asarray(mat_line)
    n = len(mat_line)
    blocks = []
    lo = 0
    for i in range(1, n + 1):
        if i == n or mat_line[i] != mat_line[lo]:
            blocks.append((lo, i))
            lo = i
    return blocks


def _odd_k(n):
    """Largest odd spline degree usable with n samples (5, 3, 1, or 0)."""
    if n >= 6:
        return 5
    if n >= 4:
        return 3
    if n >= 2:
        return 1
    return 0


def _spline_derivs(x, y):
    """Value, first, and second derivative of a block spline at the samples x.

    Uses a quintic spline where possible (>= 6 points), dropping to cubic /
    linear for short material blocks.  A single-cell block yields zero
    derivatives (no curvature is estimable there).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    k = _odd_k(n)
    if k == 0:
        return y.copy(), np.zeros_like(y), np.zeros_like(y)
    if k == 1:
        d1 = np.gradient(y, x)
        return y.copy(), d1, np.zeros_like(y)
    make_interp_spline = _make_interp_spline()
    spl = make_interp_spline(x, y, k=k)
    val = spl(x)
    d1 = spl.derivative(1)(x)
    d2 = spl.derivative(2)(x)
    return val, d1, d2


# ---------------------------------------------------------------------------
# Geometry-specific curve fit + Laplacian
# ---------------------------------------------------------------------------


def _curvelap_1d(mats, medium_map, edges_x, geom, flux):
    from ndiffusion._core import Geometry

    mat_ids = np.asarray(medium_map, dtype=int)
    G = mats.n_groups
    flux = flux.reshape(-1, G)
    edges_x = np.asarray(edges_x, dtype=float)
    xc = 0.5 * (edges_x[:-1] + edges_x[1:])

    curve = flux.copy()
    lap = np.zeros_like(flux)

    for lo, hi in _contiguous_blocks(mat_ids):
        xb = xc[lo:hi]
        for g in range(G):
            val, d1, d2 = _spline_derivs(xb, flux[lo:hi, g])
            curve[lo:hi, g] = val
            if geom == Geometry.Slab:
                lap[lo:hi, g] = d2
            elif geom == Geometry.Cylinder:
                lap[lo:hi, g] = d2 + d1 / xb
            else:  # Sphere
                lap[lo:hi, g] = d2 + 2.0 * d1 / xb

    vol = cell_volumes(edges_x, geom)
    return curve, lap, vol, mat_ids


def _curvelap_2d(mats, medium_map, edges_x, edges_y, geom, flux):
    from ndiffusion._core import Geometry2D

    G = mats.n_groups
    ex = np.asarray(edges_x, dtype=float)
    ey = np.asarray(edges_y, dtype=float)
    nx = len(ex) - 1
    ny = len(ey) - 1
    mat = np.asarray(medium_map, dtype=int).reshape(nx, ny)
    xc = 0.5 * (ex[:-1] + ex[1:])
    yc = 0.5 * (ey[:-1] + ey[1:])

    f = flux.reshape(nx, ny, G)
    curve = f.copy()
    d2x = np.zeros_like(f)
    d2y = np.zeros_like(f)
    d1y = np.zeros_like(f)

    # x-lines (fixed j): second derivative in x, block-wise per material.
    for j in range(ny):
        for lo, hi in _contiguous_blocks(mat[:, j]):
            xb = xc[lo:hi]
            for g in range(G):
                val, _, d2 = _spline_derivs(xb, f[lo:hi, j, g])
                curve[lo:hi, j, g] = val
                d2x[lo:hi, j, g] = d2

    # y-lines (fixed i): second derivative in y (and first, for RZ).
    for i in range(nx):
        for lo, hi in _contiguous_blocks(mat[i, :]):
            yb = yc[lo:hi]
            for g in range(G):
                _, d1, d2 = _spline_derivs(yb, f[i, lo:hi, g])
                d2y[i, lo:hi, g] = d2
                d1y[i, lo:hi, g] = d1

    if geom == Geometry2D.XY:
        lap = d2x + d2y
    else:  # RZ: radial coordinate is y
        rr = yc[None, :, None]
        lap = d2x + d2y + d1y / rr

    vol = cell_volumes(ex, geom, ey)
    mat_ids = np.asarray(medium_map, dtype=int)
    return curve.reshape(nx * ny, G), lap.reshape(nx * ny, G), vol, mat_ids


def _unstructured_geometry(mesh):
    """Centroids, areas, and shared-vertex neighbor sets for an FVM mesh."""
    vx = np.asarray(mesh.vx, dtype=float)
    vy = np.asarray(mesh.vy, dtype=float)
    offs = np.asarray(mesh.cell_offsets, dtype=int)
    cv = np.asarray(mesh.cell_vertices, dtype=int)
    n_cells = len(offs) - 1

    cx = np.zeros(n_cells)
    cy = np.zeros(n_cells)
    area = np.zeros(n_cells)
    vert_cells = defaultdict(list)

    for c in range(n_cells):
        vs = cv[offs[c]:offs[c + 1]]
        x = vx[vs]
        y = vy[vs]
        x2 = np.roll(x, -1)
        y2 = np.roll(y, -1)
        cross = x * y2 - x2 * y
        A2 = np.sum(cross)  # 2 * signed area
        if abs(A2) > 1e-30:
            cx[c] = np.sum((x + x2) * cross) / (3.0 * A2)
            cy[c] = np.sum((y + y2) * cross) / (3.0 * A2)
        else:
            cx[c] = x.mean()
            cy[c] = y.mean()
        area[c] = 0.5 * abs(A2)
        for v in vs:
            vert_cells[int(v)].append(c)

    neighbors = [set() for _ in range(n_cells)]
    for cells in vert_cells.values():
        for a in cells:
            neighbors[a].update(cells)
    for c in range(n_cells):
        neighbors[c].discard(c)

    return cx, cy, area, neighbors


# --- Unstructured curve fit: high-order moving least-squares reconstruction ---
#
# The Laplacian in the residual must be reconstructed to *better* than the
# scheme's own O(h^2) discretization error, or the residual is dominated by
# reconstruction error and the nearby estimate is only an order-of-magnitude
# indicator.  The structured path gets this for free: a quintic interpolating
# spline has an O(h^4)-accurate second derivative.  To match it on an
# unstructured mesh we fit a **quartic** polynomial by least squares over a wide
# neighbor stencil - high order keeps the curvature bias below O(h^2), while the
# generous over-determination (>= ~2.5x the coefficient count) filters the
# cell-scale solution noise that a near-interpolating fit would amplify.
#
# Grow the same-material stencil by vertex-connectivity rings until it holds
# ~`_LSQ_TARGET_POINTS`; then pick the highest degree (capped at 4) that stays
# comfortably over-determined.  Material interfaces and small meshes fall back to
# a lower degree automatically.
_LSQ_TARGET_POINTS = 38
_LSQ_MAX_RINGS = 5
_LSQ_MAX_DEGREE = 4


def _poly_terms(deg):
    """Exponent pairs (px, py) of the 2-D polynomial basis up to total degree
    `deg`, and the basis-column indices of the pure x^2 and y^2 terms."""
    terms = []
    i_xx = i_yy = None
    for total in range(deg + 1):
        for py in range(total + 1):
            px = total - py
            if (px, py) == (2, 0):
                i_xx = len(terms)
            elif (px, py) == (0, 2):
                i_yy = len(terms)
            terms.append((px, py))
    return terms, i_xx, i_yy


def _n_coef(deg):
    return (deg + 1) * (deg + 2) // 2


def _choose_degree(npts):
    """Highest polynomial degree (<= _LSQ_MAX_DEGREE) that stays comfortably
    over-determined for `npts` samples; 0 when even a plane cannot be fit."""
    for deg in range(_LSQ_MAX_DEGREE, 0, -1):
        if npts >= int(2.5 * _n_coef(deg)):
            return deg
    return 0


def _grow_stencil(center, neighbors, mat_ids, min_points, max_rings):
    """Same-material cells within `max_rings` of `center`, stopping once the
    stencil holds `min_points` (the center is index 0 of the returned list)."""
    mat = mat_ids[center]
    collected = {center}
    frontier = {center}
    for _ in range(max_rings):
        nxt = set()
        for c in frontier:
            for nb in neighbors[c]:
                if mat_ids[nb] == mat and nb not in collected:
                    nxt.add(nb)
        if not nxt:
            break
        collected.update(nxt)
        frontier = nxt
        if len(collected) >= min_points:
            break
    return [center] + [c for c in collected if c != center]


def _curvelap_unstructured(mats, mesh, flux):
    G = mats.n_groups
    cx, cy, area, neighbors = _unstructured_geometry(mesh)
    mat_ids = np.asarray(mesh.material_id, dtype=int)
    n_cells = len(mat_ids)
    f = flux.reshape(n_cells, G)

    curve = f.copy()
    lap = np.zeros_like(f)

    for c in range(n_cells):
        stencil = _grow_stencil(
            c, neighbors, mat_ids, _LSQ_TARGET_POINTS, _LSQ_MAX_RINGS
        )
        dx = cx[stencil] - cx[c]
        dy = cy[stencil] - cy[c]
        npts = len(stencil)

        deg = _choose_degree(npts)
        if deg == 0:
            continue  # too few same-material neighbors: leave lap = 0

        # Normalize by a local length scale so the (up to quartic) Vandermonde is
        # well conditioned - raw dx^k terms are otherwise vanishingly small and
        # the high-order coefficients become numerically meaningless.  The second
        # derivatives pick up a 1/h^2 factor on the way back out.
        h = np.sqrt(np.mean(dx[1:] ** 2 + dy[1:] ** 2))
        if h <= 0.0:
            continue
        u = dx / h
        v = dy / h

        terms, i_xx, i_yy = _poly_terms(deg)
        A = np.column_stack([u ** px * v ** py for px, py in terms])

        for g in range(G):
            coef, *_ = np.linalg.lstsq(A, f[stencil, g], rcond=None)
            curve[c, g] = coef[0]
            if deg >= 2:
                # Laplacian d2/dx2 + d2/dy2 = 2*c_xx + 2*c_yy, undo the h scaling.
                lap[c, g] = (2.0 * coef[i_xx] + 2.0 * coef[i_yy]) / (h * h)

    return curve, lap, area, mat_ids


def _spline_stride(mats, area):
    """Knot spacing in cell sizes: 2.5 of the shortest diffusion length
    sqrt(D / Sigma_r), kept between 2 and 3 cells."""
    L_min = float(np.min(np.sqrt(np.asarray(mats.D) / np.asarray(mats.removal))))
    return float(np.clip(2.5 * L_min / np.sqrt(np.median(area)), 2.0, 3.0))


def _curvelap_spline(mats, mesh, flux, bc, k, source):
    from ndiffusion import _spline_fit
    from ndiffusion._core import cell_areas

    area = np.asarray(cell_areas(mesh))
    config = "auto" if source is None else "corner"  # only the corner rows carry a source
    f = _spline_fit.fit(mats, mesh, flux, k, bc, stride=_spline_stride(mats, area),
                        config=config, source=source)
    return f.curve, f.lap, area, np.asarray(mesh.material_id, dtype=int), f


def _fit_dispatch(mats, flux, medium_map, edges_x, edges_y, geometry, mesh,
                  fit=None, bc=None, k=np.inf, source=None):
    """Pick the curve fitter from the supplied geometry descriptor.

    Returns (curve, lap, volumes, material ids, fit), where fit is the
    ``SplineFit`` for the constrained spline and None otherwise."""
    flux = np.asarray(flux, dtype=float)
    if fit not in (None, "spline", "local"):
        raise ValueError(f"fit must be None, 'spline' or 'local', got {fit!r}")
    if mesh is not None:
        if fit == "spline" or (fit is None and bc is not None):
            if bc is None:
                raise ValueError("fit='spline' needs bc=, the tag-indexed boundary "
                                 "conditions the solver was built with")
            return _curvelap_spline(mats, mesh, flux, bc, k, source)
        return (*_curvelap_unstructured(mats, mesh, flux), None)
    if fit == "spline":
        raise ValueError("fit='spline' is for the unstructured mesh (mesh=)")
    if edges_y is not None:
        return (*_curvelap_2d(mats, medium_map, edges_x, edges_y, geometry, flux), None)
    if edges_x is not None:
        return (*_curvelap_1d(mats, medium_map, edges_x, geometry, flux), None)
    raise ValueError(
        "Provide a geometry descriptor: (medium_map, edges_x[, edges_y], "
        "geometry) for structured meshes, or mesh= for the unstructured mesh."
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def nearby_fixed_source(
    solver,
    mats,
    source,
    *,
    medium_map=None,
    edges_x=None,
    edges_y=None,
    geometry=None,
    mesh=None,
    bc=None,
    fit=None,
    return_nearby=True,
):
    """Run the method of nearby problems for a fixed-source diffusion solve.

    Parameters
    ----------
    solver : FixedSourceSolver / FixedSourceSolver2D / FixedSourceSolverUnstructured2D
        A constructed solver whose ``.solve(source)`` returns a
        ``FixedSourceResult``.
    mats : Materials
        The same cross sections the solver was built with.
    source : array-like
        Flat volumetric source ``[n_cells * G]`` (public layout).
    medium_map, edges_x, edges_y, geometry, mesh
        Geometry descriptor (the solver does not expose its own).  Use
        ``(medium_map, edges_x, geometry)`` for 1-D,
        ``(medium_map, edges_x, edges_y, geometry)`` for 2-D structured, or
        ``mesh=`` for the unstructured FVM mesh.
    bc : list of BoundaryCondition, optional
        Unstructured mesh only: the tag-indexed boundary conditions the solver
        was built with. Given ``bc``, the curve fit is the constrained spline
        fit; without it, the per-cell polynomial fit.
    fit : {None, "spline", "local"}
        Force the unstructured curve fit; ``None`` picks ``"spline"`` when
        ``bc`` is given.
    return_nearby : bool, default True
        If True, also solve the nearby problem and return the error estimate.

    Returns
    -------
    NearbyFixedResult
        ``numerical`` (raw solver result), ``curve_fit``, ``residual``, and -
        when ``return_nearby`` - ``nearby`` (raw solver result) and
        ``error_estimate = nearby.flux - curve_fit``, which estimates
        ``phi_num - phi_exact``.  The arrays are shaped ``(n_cells, G)``, like
        the solver flux.
    """
    G = mats.n_groups
    source = np.asarray(source, dtype=float).reshape(-1, G)

    numerical = solver.solve(source)
    flux = np.asarray(numerical.flux, dtype=float)

    curve, lap, _vol, mat_ids, _fit = _fit_dispatch(
        mats, flux, medium_map, edges_x, edges_y, geometry, mesh,
        fit=fit, bc=bc, k=np.inf, source=source,
    )
    loss = _assemble_loss(mats, mat_ids, curve, lap)  # fixed-source: no fission
    residual = loss - source

    if not return_nearby:
        return NearbyFixedResult(numerical, curve, residual, None, None)

    nearby = solver.solve(source + residual)
    error = nearby.flux - curve
    return NearbyFixedResult(numerical, curve, residual, nearby, error)


def nearby_k_eigenvalue(
    keig_solver,
    fixed_solver,
    mats,
    *,
    medium_map=None,
    edges_x=None,
    edges_y=None,
    geometry=None,
    mesh=None,
    bc=None,
    fit=None,
    adjoint_solver=None,
    nearby=True,
    method="iterative",
    tol=1e-8,
    max_iter=200,
):
    """Run the method of nearby problems for a k-eigenvalue diffusion solve.

    The nearby problem ``M phi = (1/k) F phi + r`` is solved by a power
    iteration. With ``method="iterative"`` each outer step is a
    ``fixed_solver`` solve (it inverts the same loss+scatter operator ``M`` as
    the k-eigenvalue solver). With ``method="direct"`` (unstructured mesh and
    ``bc=`` only) ``M`` is assembled by :func:`unstructured_operator` and
    factorized once, which is much faster, and ``fixed_solver`` may be None.

    Parameters
    ----------
    keig_solver : KEigenSolver / KEigenSolver2D / KEigenSolverUnstructured2D
        Solver whose ``.solve()`` returns a ``DiffusionResult`` (flux + keff).
    fixed_solver : matching FixedSource* solver, or None with ``method="direct"``
        Built with the *same* materials, mesh, and boundary conditions.
    mats : Materials
    medium_map, edges_x, edges_y, geometry, mesh
        Geometry descriptor, as in :func:`nearby_fixed_source`.
    bc, fit
        The unstructured curve fit, as in :func:`nearby_fixed_source`.
    adjoint_solver : matching KEigen* solver, optional
        Built on :func:`make_adjoint_materials` of ``mats`` (same mesh and
        boundary conditions). Its flux gives the one-shot eigenvalue error
        estimate ``k_error_adjoint`` and its per-cell contributions
        ``k_error_cells``. With ``method="direct"`` the adjoint is instead the
        exact discrete one, from the assembled operator, and this is ignored.
    nearby : bool, default True
        Run the nearby power iteration. With ``nearby=False`` only the fit and
        the one-shot estimate are formed; ``nearby_flux`` and ``k_nearby`` are
        then None.
    method : {"iterative", "direct"}
        How the nearby problem and the adjoint are solved (see above).
    tol : float
        Convergence tolerance on the flux change and ``|dk|``.
    max_iter : int
        Maximum nearby power-iteration steps.

    Returns
    -------
    NearbyKResult
        ``numerical`` (raw k result), ``curve_fit``, ``k_curve_fit``,
        ``nearby_flux``, ``k_nearby``, ``residual``, and ``nearby_rate`` (the
        curve-fit fission production integral).  The arrays are shaped
        ``(n_cells, G)``, like the solver flux.
        ``k_nearby - k_curve_fit`` estimates the eigenvalue discretization error,
        and so does ``k_error_adjoint`` when an adjoint is available
        (``k_error_cells`` sums to it). On layouts whose error contributions
        cancel - material corners, small inclusions - ``k_error_cells`` and the
        local error are more reliable than the net eigenvalue estimate; the sum
        of ``abs(k_error_cells)`` is a (loose) bound. ``fit`` is the
        constrained spline fit (``SplineFit``) when one was used and
        ``operator`` the assembled operator with ``method="direct"``; both feed
        :func:`nearby_qoi`.
    """
    if method not in ("iterative", "direct"):
        raise ValueError(f"method must be 'iterative' or 'direct', got {method!r}")
    if method == "direct" and (mesh is None or bc is None):
        raise ValueError("method='direct' needs the unstructured mesh= and bc=")
    if method == "iterative" and nearby and fixed_solver is None:
        raise ValueError("the iterative nearby solve needs fixed_solver")

    numerical = keig_solver.solve()
    flux = np.asarray(numerical.flux, dtype=float)
    G = mats.n_groups

    curve, lap, vol, mat_ids, spline = _fit_dispatch(
        mats, flux, medium_map, edges_x, edges_y, geometry, mesh,
        fit=fit, bc=bc, k=numerical.keff,
    )
    volG = vol[:, None]

    loss = _assemble_loss(mats, mat_ids, curve, lap)
    prod = _fission_apply(mats, mat_ids, curve)

    loss_int = float(np.sum(loss * volG))
    rate = float(np.sum(prod * volG))
    if loss_int == 0.0:
        raise ValueError("Curve-fit loss integral is zero; cannot form k.")
    k_cf = rate / loss_int

    residual = loss - prod / k_cf  # (n_cells, G), exact source of the nearby problem

    op = lu = None
    if method == "direct":
        from scipy.sparse.linalg import splu

        from ndiffusion.fv_operator import unstructured_operator

        op = unstructured_operator(mats, mesh, bc)
        lu = splu(op.loss.tocsc())

    k_error = cells = None
    adj = None
    if method == "direct":
        adj = _discrete_adjoint(op, flux, numerical.keff)
    elif adjoint_solver is not None:
        adj = np.asarray(adjoint_solver.solve().flux, dtype=float).reshape(-1, G)
    if adj is not None:
        cells = (np.sum(adj * (numerical.keff * loss - prod) * volG, axis=1)
                 / float(np.sum(adj * loss * volG)))
        k_error = float(cells.sum())
    if not nearby:
        return NearbyKResult(numerical, curve, k_cf, None, None, residual, rate, k_error, cells,
                             spline, op)

    # --- Nearby power iteration: M phi = F phi / k + r ---
    if lu is not None:
        def solve_m(q):
            return lu.solve((q * volG).ravel()).reshape(-1, G)
    else:
        def solve_m(q):
            return np.asarray(fixed_solver.solve(q).flux, dtype=float)

    phi = curve.copy()
    k = k_cf
    for _ in range(max_iter):
        Fphi = _fission_apply(mats, mat_ids, phi)
        prod_old = float(np.sum(Fphi * volG))
        phi_new = solve_m(Fphi / k + residual)

        prod_new = float(np.sum(_fission_apply(mats, mat_ids, phi_new) * volG))
        k_new = k * prod_new / prod_old if prod_old != 0.0 else k

        denom = np.linalg.norm(phi)
        dphi = np.linalg.norm(phi_new - phi) / (denom if denom > 0 else 1.0)
        dk = abs(k_new - k)
        phi = phi_new
        k = k_new
        if dphi < tol and dk < tol:
            break

    return NearbyKResult(
        numerical=numerical,
        curve_fit=curve,
        k_curve_fit=k_cf,
        nearby_flux=phi,
        k_nearby=k,
        residual=residual,
        nearby_rate=rate,
        k_error_adjoint=k_error,
        k_error_cells=cells,
        fit=spline,
        operator=op,
    )


def _bordered_adjoint_lu(op, flux, keff):
    """LU of [[(M - F/k)^T, F phi], [(F phi)^T, 0]], the generalized adjoint system.

    The appended row and column make the singular adjoint operator invertible:
    the column absorbs the small inconsistency of an iteratively converged
    (phi, k), and the row fixes the free multiple of the adjoint eigenvector."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu

    Fphi = op.production @ np.asarray(flux, dtype=float).ravel()
    col = sp.csc_matrix(Fphi[:, None])
    A = (op.loss - op.production / keff).T
    return splu(sp.bmat([[A, col], [col.T, None]], format="csc")), Fphi


def _discrete_adjoint(op, flux, keff):
    """The exact discrete adjoint eigenvector, (n_cells, G), normalized to <psi, F phi> = 1."""
    lu, Fphi = _bordered_adjoint_lu(op, flux, keff)
    N = len(Fphi)
    rhs = np.zeros(N + 1)
    rhs[N] = 1.0
    # (M - F/k)^T psi + mu F phi = 0 with <F phi, psi> = 1: mu vanishes and psi is the adjoint.
    return lu.solve(rhs)[:N].reshape(np.shape(flux))


# Dunavant degree-4 rule on a triangle: (barycentric a, weight) orbits.
_D4 = ((0.445948490915965, 0.223381589678011), (0.091576213509771, 0.109951743655322))


def _cell_quadrature(mesh, cx, cy):
    """Degree-4 quadrature on every cell, by fanning it into triangles from its
    centroid. Returns (cell, x, y, weight) arrays; the weights sum to the areas."""
    off = np.asarray(mesh.cell_offsets, dtype=np.int64)
    cv = np.asarray(mesh.cell_vertices, dtype=np.int64)
    vx, vy = np.asarray(mesh.vx, dtype=float), np.asarray(mesh.vy, dtype=float)
    cell = np.repeat(np.arange(len(off) - 1), np.diff(off))
    nxt = np.arange(len(cv)) + 1
    nxt[off[1:] - 1] = off[:-1]
    P0 = np.column_stack([cx[cell], cy[cell]])
    P1 = np.column_stack([vx[cv], vy[cv]])
    P2 = np.column_stack([vx[cv[nxt]], vy[cv[nxt]]])
    tri = 0.5 * np.abs((P1[:, 0] - P0[:, 0]) * (P2[:, 1] - P0[:, 1])
                       - (P2[:, 0] - P0[:, 0]) * (P1[:, 1] - P0[:, 1]))
    cells, xs, ys, ws = [], [], [], []
    for s_, w in _D4:
        for lam in ((s_, s_, 1 - 2 * s_), (s_, 1 - 2 * s_, s_), (1 - 2 * s_, s_, s_)):
            p = lam[0] * P0 + lam[1] * P1 + lam[2] * P2
            cells.append(cell)
            xs.append(p[:, 0])
            ys.append(p[:, 1])
            ws.append(w * tri)
    return np.concatenate(cells), np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


def _ratio(num, den, phi, vol):
    """J = sum num phi V / sum den phi V for a stack of weights (n_q, n, G)."""
    top = np.einsum("qcg,cg,c->q", num, phi, vol)
    bot = np.einsum("qcg,cg,c->q", den, phi, vol)
    return top / bot


def nearby_qoi(result, mats, mesh, bc, numerator, denominator):
    """Discretization error of normalized quantities of interest.

    A quantity of interest here is a ratio of two reaction-rate integrals,

        J(phi) = int a . phi dV / int b . phi dV,

    with per-unit-volume weights ``a`` (numerator) and ``b`` (denominator) per
    cell and group, e.g. the absorption cross section in one region over the
    production cross section everywhere (a normalized region power or
    absorption rate). The ratio makes ``J`` independent of the eigenvector's
    arbitrary normalization.

    The one-shot estimate solves the generalized adjoint problem
    ``(M - F/k)^T Gamma = dJ/dphi`` with ``<Gamma, F phi> = 0`` on the exact
    discrete operator (:func:`unstructured_operator`) and returns
    ``<Gamma, V r> + J(phi_h) - J(fit)``, the linearization of the nearby
    estimate ``J(phi_nearby) - J(fit)``. It needs the exact transpose: a solver
    run on adjoint cross sections does not transpose the non-orthogonal
    correction and leaves a bias of 1-2% on triangles and hexagons.

    Parameters
    ----------
    result : NearbyKResult
        From :func:`nearby_k_eigenvalue` with the constrained spline fit
        (``bc=`` given). Its ``operator`` is reused when present
        (``method="direct"``).
    mats, mesh, bc
        As passed to :func:`nearby_k_eigenvalue`.
    numerator, denominator : array-like
        Weights ``a`` and ``b``, shaped ``(n_cells, G)``, or ``(n_q, n_cells, G)``
        to evaluate several quantities with one factorization.

    Returns
    -------
    NearbyQoIResult
        ``value`` J(phi_h), ``fit_value`` J(fit) (by degree-4 quadrature of the
        fit over each cell), ``error_adjoint`` the one-shot estimate of
        ``J(phi_h) - J(phi_exact)``, ``error_nearby`` the nearby-solve estimate
        ``J(phi_nearby) - J(fit)`` (None without a nearby solve), and ``cells``,
        the per-cell contributions ``Gamma . V r`` (n_q, n_cells), a
        goal-oriented error map for adaptation. Arrays have a leading ``n_q``
        axis only when the weights do.
    """
    from ndiffusion._core import cell_centroids
    from ndiffusion.fv_operator import unstructured_operator

    if result.fit is None:
        raise ValueError("nearby_qoi needs a result from the constrained spline fit (bc= given)")
    G = mats.n_groups
    phi = np.asarray(result.numerical.flux, dtype=float).reshape(-1, G)
    n = phi.shape[0]
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    single = num.ndim == 2
    num = num.reshape(-1, n, G)
    den = np.broadcast_to(den.reshape(-1, n, G), num.shape)

    op = result.operator if result.operator is not None else unstructured_operator(mats, mesh, bc)
    area = op.area
    keff = result.numerical.keff
    J_h = _ratio(num, den, phi, area)

    cx, cy = (np.asarray(a, dtype=float) for a in cell_centroids(mesh))
    qc, qx, qy, qw = _cell_quadrature(mesh, cx, cy)
    fvals = result.fit.evaluate(qc, qx, qy)
    J_fit = (np.einsum("qkg,kg,k->q", num[:, qc], fvals, qw)
             / np.einsum("qkg,kg,k->q", den[:, qc], fvals, qw))

    # dJ/dphi in the volume-integrated layout: (a V - J b V) / int b phi dV
    bot = np.einsum("qcg,cg,c->q", den, phi, area)
    grads = (num - J_h[:, None, None] * den) * area[None, :, None] / bot[:, None, None]
    lu, _ = _bordered_adjoint_lu(op, phi, keff)
    N = n * G
    rhs = np.zeros((N + 1, len(J_h)))
    rhs[:N] = grads.reshape(len(J_h), N).T
    Gam = lu.solve(rhs)[:N].T.reshape(len(J_h), n, G)
    Vr = np.asarray(result.residual, dtype=float).reshape(n, G) * area[:, None]
    cells = np.einsum("qcg,cg->qc", Gam, Vr)
    est = cells.sum(axis=1) + J_h - J_fit
    near = None
    if result.nearby_flux is not None:
        phi_n = np.asarray(result.nearby_flux, dtype=float).reshape(n, G)
        near = _ratio(num, den, phi_n, area) - J_fit

    if single:
        return NearbyQoIResult(float(J_h[0]), float(J_fit[0]), float(est[0]),
                               None if near is None else float(near[0]), cells[0])
    return NearbyQoIResult(J_h, J_fit, est, near, cells)
