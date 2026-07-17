"""Method of nearby problems (MNP) for multigroup neutron diffusion.

MNP is a solution-verification technique that estimates the spatial
discretization error of a numerical solution without an analytic reference.
The recipe (see ``ants/nearby1d.pyx`` for the transport analogue):

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
these come from quintic/cubic splines on structured meshes (via scipy) and from
a per-cell least-squares quadratic reconstruction on the unstructured mesh.

Everything here is orchestration in Python: the actual solves reuse the compiled
``FixedSourceSolver*`` classes, so no C++ changes are required.  Sources are
per-unit-volume throughout (the solvers integrate internally), and the residual
is the continuous operator evaluated pointwise at cell centers - one definition
for all three geometries.
"""

from collections import defaultdict, namedtuple

import numpy as np

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
    ],
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
    if nusigf.size == n_mat * G * G:
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
    """Public helper: fission production ``F phi`` for a flat public flux.

    Parameters
    ----------
    mats : Materials
    medium_map : sequence of int
        Material index per cell (``mesh.material_id`` for unstructured).
    flux : array-like
        Flat public flux ``[n_cells * G]``, ``flux[cell*G+g]``.

    Returns
    -------
    numpy.ndarray
        Flat ``[n_cells * G]`` fission source, per unit volume.
    """
    mat_ids = np.asarray(medium_map, dtype=int)
    G = mats.n_groups
    curve = np.asarray(flux, dtype=float).reshape(-1, G)
    return _fission_apply(mats, mat_ids, curve).ravel()


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
# Cell volumes (match compute_geometry in the C++ solvers)
# ---------------------------------------------------------------------------

_PI = np.pi


def _cell_volumes_1d(edges_x, geom):
    from ndiffusion._core import Geometry

    e = np.asarray(edges_x, dtype=float)
    if geom == Geometry.Slab:
        return e[1:] - e[:-1]
    if geom == Geometry.Cylinder:
        return _PI * (e[1:] ** 2 - e[:-1] ** 2)
    return (4.0 / 3.0) * _PI * (e[1:] ** 3 - e[:-1] ** 3)  # Sphere


def _cell_volumes_2d(edges_x, edges_y, geom):
    from ndiffusion._core import Geometry2D

    ex = np.asarray(edges_x, dtype=float)
    ey = np.asarray(edges_y, dtype=float)
    dx = ex[1:] - ex[:-1]
    if geom == Geometry2D.XY:
        dy = ey[1:] - ey[:-1]
        return np.outer(dx, dy).ravel()  # cell = i*ny + j
    # RZ: x = z (axial), y = r (radial); vol = pi*(r_hi^2 - r_lo^2) * dz
    ring = _PI * (ey[1:] ** 2 - ey[:-1] ** 2)
    return np.outer(dx, ring).ravel()


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

    vol = _cell_volumes_1d(edges_x, geom)
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

    vol = _cell_volumes_2d(ex, ey, geom)
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


def _fit_dispatch(mats, flux, medium_map, edges_x, edges_y, geometry, mesh):
    """Pick the curve fitter from the supplied geometry descriptor."""
    flux = np.asarray(flux, dtype=float)
    if mesh is not None:
        return _curvelap_unstructured(mats, mesh, flux)
    if edges_y is not None:
        return _curvelap_2d(mats, medium_map, edges_x, edges_y, geometry, flux)
    if edges_x is not None:
        return _curvelap_1d(mats, medium_map, edges_x, geometry, flux)
    raise ValueError(
        "Provide a geometry descriptor: (medium_map, edges_x[, edges_y], "
        "geometry) for structured meshes, or mesh= for the unstructured mesh."
    )


def _as_source_list(arr):
    return np.asarray(arr, dtype=float).ravel().tolist()


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
    return_nearby : bool, default True
        If True, also solve the nearby problem and return the error estimate.

    Returns
    -------
    NearbyFixedResult
        ``numerical`` (raw solver result), ``curve_fit`` (flat), ``residual``
        (flat), and - when ``return_nearby`` - ``nearby`` (raw solver result)
        and ``error_estimate = nearby.flux - curve_fit`` (flat), which estimates
        ``phi_num - phi_exact``.
    """
    G = mats.n_groups
    source = np.asarray(source, dtype=float)

    numerical = solver.solve(_as_source_list(source))
    flux = np.asarray(numerical.flux, dtype=float)

    curve, lap, _vol, mat_ids = _fit_dispatch(
        mats, flux, medium_map, edges_x, edges_y, geometry, mesh
    )
    loss = _assemble_loss(mats, mat_ids, curve, lap)  # fixed-source: no fission
    residual = (loss - source.reshape(-1, G)).ravel()
    curve_flat = curve.ravel()

    if not return_nearby:
        return NearbyFixedResult(numerical, curve_flat, residual, None, None)

    nearby = solver.solve(_as_source_list(source + residual))
    error = np.asarray(nearby.flux, dtype=float) - curve_flat
    return NearbyFixedResult(numerical, curve_flat, residual, nearby, error)


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
    tol=1e-8,
    max_iter=200,
):
    """Run the method of nearby problems for a k-eigenvalue diffusion solve.

    The nearby problem is solved by a Python power iteration that reuses
    ``fixed_solver`` (which inverts the same loss+scatter operator ``M`` as the
    k-eigenvalue solver): each outer step solves ``M phi = (1/k) F phi + r`` and
    updates ``k`` from the production integral ratio.

    Parameters
    ----------
    keig_solver : KEigenSolver / KEigenSolver2D / KEigenSolverUnstructured2D
        Solver whose ``.solve()`` returns a ``DiffusionResult`` (flux + keff).
    fixed_solver : matching FixedSource* solver
        Built with the *same* materials, mesh, and boundary conditions.
    mats : Materials
    medium_map, edges_x, edges_y, geometry, mesh
        Geometry descriptor, as in :func:`nearby_fixed_source`.
    tol : float
        Convergence tolerance on the flux change and ``|dk|``.
    max_iter : int
        Maximum nearby power-iteration steps.

    Returns
    -------
    NearbyKResult
        ``numerical`` (raw k result), ``curve_fit`` (flat), ``k_curve_fit``,
        ``nearby_flux`` (flat), ``k_nearby``, ``residual`` (flat), and
        ``nearby_rate`` (the curve-fit fission production integral).
        ``k_nearby - k_curve_fit`` estimates the eigenvalue discretization error.
    """
    G = mats.n_groups

    numerical = keig_solver.solve()
    flux = np.asarray(numerical.flux, dtype=float)

    curve, lap, vol, mat_ids = _fit_dispatch(
        mats, flux, medium_map, edges_x, edges_y, geometry, mesh
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

    # --- Nearby power iteration (reuses the fixed-source solver's operator M) ---
    phi = curve.copy()
    k = k_cf
    for _ in range(max_iter):
        Fphi = _fission_apply(mats, mat_ids, phi)
        prod_old = float(np.sum(Fphi * volG))
        q = Fphi / k + residual
        res = fixed_solver.solve(_as_source_list(q))
        phi_new = np.asarray(res.flux, dtype=float).reshape(-1, G)

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
        curve_fit=curve.ravel(),
        k_curve_fit=k_cf,
        nearby_flux=phi.ravel(),
        k_nearby=k,
        residual=residual.ravel(),
        nearby_rate=rate,
    )
