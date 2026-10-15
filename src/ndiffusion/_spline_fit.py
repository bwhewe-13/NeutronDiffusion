"""Constrained global spline fit of an unstructured-mesh flux, for MNP.

One smooth function per connected material region and group: a quintic
tensor-product B-spline on the region's bounding box, with knots about `stride`
cell sizes apart, so the fit does not depend on how the region is meshed. All
regions and groups are fit together by sparse least squares to the cell
centroid values, plus heavily weighted constraint rows at two Gauss points on
every interface and boundary face:

    interface:  phi and D dphi/dn continuous across the face
    boundary:   the face's own Robin condition
    "smooth":   also the diffusion equation and its gradient on each side

Two configurations are needed. "smooth" (hard PDE rows on interfaces and
boundaries) is accurate on layouts without material corners, but around a
corner, or around a small region enclosed by interfaces, the PDE rows and the
continuity rows together pose an ill-conditioned Cauchy problem and the fit
stops matching its data. "corner" drops those PDE rows, imposes the diffusion
equation softly at every centroid instead, and adds the corner singular
functions r^alpha Phi(theta) of every interior material corner to the regions
around it. "auto" picks "corner" when the mesh has a material corner turning
more than 30 degrees or a small enclosed inclusion, and "smooth" otherwise.

The constraint sets are consistent but rank-deficient, so they enter as
weighted rows of one normal-equation system rather than as exact equalities.
"""

from collections import defaultdict, namedtuple

import numpy as np

K = 5  # spline degree

SplineFit = namedtuple("SplineFit", ["curve", "lap", "config", "evaluate", "info"])

CONFIGS = {
    "smooth": dict(pde_orders=(0, 1), interior_pde=None, enrich=False),
    "corner": dict(pde_orders=(), interior_pde=(1.0, (0, 1)), enrich=True),
}
CORNER_TURN = 30.0  # degrees; a polygonal curve turns a little at every vertex


def _scipy():
    try:
        import scipy.sparse as sp
        from scipy.interpolate import BSpline
        from scipy.optimize import brentq
        from scipy.sparse.linalg import spsolve
    except ImportError as exc:  # pragma: no cover - exercised only without scipy
        raise ImportError(
            "The method of nearby problems needs SciPy for the unstructured "
            "spline fit. Install it with `pip install ndiffusion[nearby]` (or "
            "`pip install scipy`)."
        ) from exc
    return sp, BSpline, brentq, spsolve


# ---------------------------------------------------------------------------
# Mesh topology
# ---------------------------------------------------------------------------


def _edge_owners(mesh):
    offs = np.asarray(mesh.cell_offsets)
    cv = np.asarray(mesh.cell_vertices)
    owner = defaultdict(list)
    for c in range(len(offs) - 1):
        vs = cv[offs[c]:offs[c + 1]]
        for a, b in zip(vs, np.roll(vs, -1)):
            owner[(min(a, b), max(a, b))].append(c)
    return owner


def mesh_faces(mesh, mat):
    """Interface faces (interior, materials differ) and boundary faces, each a
    dict with midpoint, unit tangent, length and cells; boundary faces also
    carry their tag and outward normal."""
    vx, vy = np.asarray(mesh.vx), np.asarray(mesh.vy)
    offs, cv = np.asarray(mesh.cell_offsets), np.asarray(mesh.cell_vertices)
    owner = _edge_owners(mesh)
    centre = np.array([[vx[cv[offs[c]:offs[c + 1]]].mean(), vy[cv[offs[c]:offs[c + 1]]].mean()]
                       for c in range(len(offs) - 1)])
    tag = {(min(a, b), max(a, b)): t
           for a, b, t in zip(mesh.bface_v0, mesh.bface_v1, mesh.bface_bc_tag)}
    inter, bound = [], []
    for (a, b), cs in owner.items():
        tx, ty = vx[b] - vx[a], vy[b] - vy[a]
        ln = np.hypot(tx, ty)
        mid = np.array([0.5 * (vx[a] + vx[b]), 0.5 * (vy[a] + vy[b])])
        f = dict(mid=mid, t=np.array([tx, ty]) / ln, cells=cs, length=ln)
        if len(cs) == 2 and mat[cs[0]] != mat[cs[1]]:
            inter.append(f)
        elif len(cs) == 1:
            nrm = np.array([ty, -tx]) / ln
            if np.dot(nrm, mid - centre[cs[0]]) < 0:
                nrm = -nrm
            f.update(n=nrm, tag=tag[(a, b)])
            bound.append(f)
    return inter, bound


def material_corners(inter, min_turn=0.0, tol=1e-6):
    """Interface vertices where the interface turns by more than `min_turn`
    degrees, or where three or more interface faces meet, as an (n, 2) array."""
    by_vertex = defaultdict(list)
    for f in inter:
        for s in (-0.5, 0.5):
            p = f["mid"] + s * f["length"] * f["t"]
            by_vertex[(round(p[0], 6), round(p[1], 6))].append(f["t"])
    out = []
    for key, ts in by_vertex.items():
        if len(ts) >= 3:
            out.append(key)
        elif len(ts) == 2:
            c = abs(np.dot(ts[0], ts[1]))
            if abs(c - 1.0) > tol and np.degrees(np.arccos(min(1.0, c))) > min_turn:
                out.append(key)
    return np.array(out, dtype=float).reshape(-1, 2)


def interface_breaks(inter, min_len, tol=1e-9):
    """x of every vertical and y of every horizontal run of interface faces
    whose total length reaches `min_len`. Used as knots, they put the
    re-entrant edges of a non-convex region on knot lines instead of cutting a
    knot span (a cut basis function makes the fit spike there)."""
    runs = ({}, {})
    for f in inter:
        tx, ty = f["t"]
        for axis, along in ((0, abs(tx) < tol), (1, abs(ty) < tol)):
            if along:
                c = round(float(f["mid"][axis]), 6)
                runs[axis][c] = runs[axis].get(c, 0.0) + f["length"]
    return tuple(tuple(sorted(c for c, L in r.items() if L >= min_len)) for r in runs)


# ---------------------------------------------------------------------------
# Tensor B-spline regions
# ---------------------------------------------------------------------------


def _knots(a, b, hk, breaks=()):
    """Clamped knot vector on [a, b] with spacing <= hk; every coordinate in
    `breaks` strictly inside (a, b) becomes a simple knot."""
    eps = 1e-9 * max(1.0, b - a)
    pts = [a] + sorted(p for p in set(breaks) if a + eps < p < b - eps) + [b]
    inner = []
    for lo, hi in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil((hi - lo) / hk - 1e-9)))
        inner.extend(np.linspace(lo, hi, n + 1)[1:])
    return np.r_[[a] * (K + 1), inner[:-1], [b] * (K + 1)]


def _basis(x, t, nu=0):
    """Dense (len(x), n_basis) matrix of the nu-th derivative of the B-splines."""
    _, BSpline, _, _ = _scipy()
    x = np.clip(np.asarray(x, float), t[K], t[-K - 1])
    if nu == 0:
        return BSpline.design_matrix(x, t, K).toarray()
    return BSpline(t, np.eye(len(t) - K - 1), K).derivative(nu)(x)


class _Region:
    def __init__(self, cells, material, box, hk, breaks):
        x0, x1, y0, y1 = box
        self.cells, self.material = cells, material
        self.tx, self.ty = _knots(x0, x1, hk, breaks[0]), _knots(y0, y1, hk, breaks[1])
        self.nx, self.ny = len(self.tx) - K - 1, len(self.ty) - K - 1
        self.n = self.nx * self.ny

    def rows(self, x, y, dx=0, dy=0):
        """Dense rows of d^dx/dx d^dy/dy of the tensor basis at the points."""
        Bx, By = _basis(x, self.tx, dx), _basis(y, self.ty, dy)
        return (Bx[:, :, None] * By[:, None, :]).reshape(len(x), -1)

    def rows_sparse(self, x, y, dx=0, dy=0):
        """The same rows, sparse: a tensor row has at most (K+1)^2 nonzeros."""
        sp = _scipy()[0]
        Bx, By = _basis(x, self.tx, dx), _basis(y, self.ty, dy)
        ix = np.argsort(-np.abs(Bx), axis=1)[:, :K + 1]
        iy = np.argsort(-np.abs(By), axis=1)[:, :K + 1]
        vx, vy = np.take_along_axis(Bx, ix, 1), np.take_along_axis(By, iy, 1)
        cols = (ix[:, :, None] * self.ny + iy[:, None, :]).reshape(len(x), -1)
        vals = (vx[:, :, None] * vy[:, None, :]).reshape(len(x), -1)
        rr = np.repeat(np.arange(len(x)), cols.shape[1])
        return sp.csr_matrix((vals.ravel(), (rr, cols.ravel())), shape=(len(x), self.n))


def regions(mesh, mat, area, stride, breaks=((), ())):
    """Connected same-material cell sets with their spline bases, and the
    region index of every cell."""
    vx, vy = np.asarray(mesh.vx), np.asarray(mesh.vy)
    offs, cv = np.asarray(mesh.cell_offsets), np.asarray(mesh.cell_vertices)
    nc = len(offs) - 1
    adj = defaultdict(list)
    for cs in _edge_owners(mesh).values():
        if len(cs) == 2 and mat[cs[0]] == mat[cs[1]]:
            adj[cs[0]].append(cs[1])
            adj[cs[1]].append(cs[0])
    comp = -np.ones(nc, int)
    out = []
    for c0 in range(nc):
        if comp[c0] >= 0:
            continue
        stack, members = [c0], []
        comp[c0] = len(out)
        while stack:
            c = stack.pop()
            members.append(c)
            for n in adj[c]:
                if comp[n] < 0:
                    comp[n] = len(out)
                    stack.append(n)
        members = np.array(sorted(members))
        vs = np.concatenate([cv[offs[c]:offs[c + 1]] for c in members])
        hk = stride * np.sqrt(np.median(area[members]))
        box = (vx[vs].min(), vx[vs].max(), vy[vs].min(), vy[vs].max())
        out.append(_Region(members, mat[c0], box, hk, breaks))
    return out, comp


def small_inclusions(mesh, bc, n_groups, regs, comp, area, stride, max_spans=8.0):
    """(material, spans) of every region that the mirrored domain encloses and
    whose equal-area diameter is under `max_spans` knot spans (absorber rods,
    small seeds). Enclosed: it touches no non-reflective boundary and does not
    run across the whole domain in x or y (the mirrors would extend that one
    into an infinite strip)."""
    owner = _edge_owners(mesh)
    vx, vy = np.asarray(mesh.vx), np.asarray(mesh.vy)
    offs, cv = np.asarray(mesh.cell_offsets), np.asarray(mesh.cell_vertices)
    tol = 1e-9 * max(np.ptp(vx), np.ptp(vy))

    def reflective(t):
        return all(bc[t * n_groups + g].A == 0.0 for g in range(n_groups))

    open_regions = {comp[owner[(min(a, b), max(a, b))][0]]
                    for a, b, t in zip(mesh.bface_v0, mesh.bface_v1, mesh.bface_bc_tag)
                    if not reflective(t)}
    out = []
    for i, R in enumerate(regs):
        if i in open_regions:
            continue
        vs = np.concatenate([cv[offs[c]:offs[c + 1]] for c in R.cells])
        if (vx[vs].min() - vx.min() < tol and vx.max() - vx[vs].max() < tol) or \
                (vy[vs].min() - vy.min() < tol and vy.max() - vy[vs].max() < tol):
            continue
        hk = stride * np.sqrt(np.median(area[R.cells]))
        spans = 2.0 * np.sqrt(area[R.cells].sum() / np.pi) / hk
        if spans < max_spans:
            out.append((int(R.material), float(spans)))
    return out


def _diff_penalty(nx, ny, order):
    """order-th coefficient differences along x and y (P-spline penalty) for a
    coefficient grid stored ix * ny + iy."""
    sp = _scipy()[0]

    def d1(n):
        D = sp.identity(n, format="csr")
        for _ in range(order):
            D = (sp.eye(D.shape[0] - 1, D.shape[0], 1) - sp.eye(D.shape[0] - 1, D.shape[0])) @ D
        return D

    return sp.vstack([sp.kron(d1(nx), sp.identity(ny)), sp.kron(sp.identity(nx), d1(ny))])


# ---------------------------------------------------------------------------
# Corner singular functions
# ---------------------------------------------------------------------------
#
# At an interior material corner the flux behaves like smooth + c r^alpha
# Phi(theta), with Phi = A_k cos(alpha t) + B_k sin(alpha t) in sector k.
# Continuity of Phi and D Phi' across the rays gives a transfer matrix per
# sector; an exponent is a root of trace(M) = 2 for the product M. Each mode
# is harmonic in every sector and continuous in value and current across the
# rays, so it can be added to every region around the corner. That only holds
# while the rays are interfaces: where an interface stops (IAEA assembly
# edges), the ray's extension crosses one material and an uncut mode kinks
# there, a line source the pointwise residual cannot see. So each mode is cut
# off radially, chi = 1 to R1 and 0 past R2, with R2 the shortest run of the
# same interface along the rays (infinite when every ray reaches the boundary).


def _transfer(alpha, w, D):
    c, s = np.cos(alpha * w), np.sin(alpha * w)
    return np.array([[c, s / (D * alpha)], [-D * alpha * s, c]])


def _trace_minus_two(alpha, sectors):
    M = np.eye(2)
    for w, D in sectors:
        M = _transfer(alpha, w, D) @ M
    return np.trace(M) - 2.0


def corner_exponents(sectors, amax=2.0, n=4000):
    """Roots alpha in (0, amax] of trace(M) = 2 for sectors [(width, D), ...]."""
    brentq = _scipy()[2]
    a = np.linspace(1e-4, amax, n)
    f = np.array([_trace_minus_two(x, sectors) for x in a])
    roots = []
    for i in range(n - 1):
        if f[i] == 0.0:
            roots.append(a[i])
        elif f[i] * f[i + 1] < 0:
            roots.append(brentq(_trace_minus_two, a[i], a[i + 1], args=(sectors,)))
    # double roots (alpha = 1 in a homogeneous medium) touch zero without a sign change
    for i in range(1, n - 1):
        if abs(f[i]) < 1e-6 and abs(f[i]) <= abs(f[i - 1]) and abs(f[i]) <= abs(f[i + 1]):
            if not any(abs(a[i] - r) < 1e-3 for r in roots):
                roots.append(a[i])
    return sorted(roots)


class CornerMode:
    """chi(r) r^alpha Phi(theta) for one group at one corner."""

    def __init__(self, x0, y0, alpha, group, starts, widths, AB, regions, R1=np.inf, R2=np.inf):
        self.x0, self.y0, self.alpha, self.group = x0, y0, alpha, group
        self.starts, self.widths, self.AB = np.asarray(starts), np.asarray(widths), np.asarray(AB)
        self.regions = set(regions)
        self.scale = 1.0
        self.R1, self.R2 = R1, R2

    def _chi(self, r):
        """Cutoff and its first three r-derivatives (C^3 septic step)."""
        out = [np.ones_like(r), np.zeros_like(r), np.zeros_like(r), np.zeros_like(r)]
        if not np.isfinite(self.R2):
            return out
        w = self.R2 - self.R1
        t = np.clip((r - self.R1) / w, 0.0, 1.0)
        s = 35 * t**4 - 84 * t**5 + 70 * t**6 - 20 * t**7
        s1 = 140 * t**3 - 420 * t**4 + 420 * t**5 - 140 * t**6
        s2 = 420 * t**2 - 1680 * t**3 + 2100 * t**4 - 840 * t**5
        s3 = 840 * t - 5040 * t**2 + 8400 * t**3 - 4200 * t**4
        return [1 - s, -s1 / w, -s2 / w**2, -s3 / w**3]

    def _sector(self, theta):
        rel = np.mod(theta - self.starts[0], 2 * np.pi)
        edges = np.cumsum(np.r_[0.0, self.widths])
        k = np.clip(np.searchsorted(edges, rel, side="right") - 1, 0, len(self.widths) - 1)
        return k, rel - edges[k]

    def _raw(self, x, y):
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        dx, dy = x - self.x0, y - self.y0
        r = np.maximum(np.hypot(dx, dy), 1e-12)
        th = np.arctan2(dy, dx)
        k, t = self._sector(th)
        A, B, a = self.AB[k, 0], self.AB[k, 1], self.alpha
        Phi = A * np.cos(a * t) + B * np.sin(a * t)
        dPhi = a * (-A * np.sin(a * t) + B * np.cos(a * t))
        val = self.scale * r**a * Phi
        dr = self.scale * a * r ** (a - 1) * Phi
        dth = self.scale * r**a * dPhi
        gx = np.cos(th) * dr - np.sin(th) / r * dth
        gy = np.sin(th) * dr + np.cos(th) / r * dth
        return r, th, val, gx, gy

    def eval(self, x, y):
        """Value and gradient of chi m."""
        r, th, val, gx, gy = self._raw(x, y)
        c, c1, _, _ = self._chi(r)
        return c * val, c * gx + c1 * np.cos(th) * val, c * gy + c1 * np.sin(th) * val

    def lap(self, x, y):
        """Laplacian of chi m and its gradient: m is harmonic, so
        lap(chi m) = m (chi'' + (2 alpha + 1) chi' / r)."""
        r, th, val, gx, gy = self._raw(x, y)
        _, c1, c2, c3 = self._chi(r)
        b = 2 * self.alpha + 1
        q = c2 + b * c1 / r
        q1 = c3 + b * (c2 / r - c1 / r**2)
        return q * val, q * gx + q1 * np.cos(th) * val, q * gy + q1 * np.sin(th) * val


def _ray_run(x0, y0, theta, inter, mat, pair, bverts, h_cell):
    """Length of the straight interface between materials `pair` leaving
    (x0, y0) at angle theta; inf if it reaches the domain boundary."""
    u = np.array([np.cos(theta), np.sin(theta)])
    tol = 1e-6 * max(1.0, h_cell)
    segs = []
    for f in inter:
        if abs(f["t"][0] * u[1] - f["t"][1] * u[0]) > 1e-6:
            continue
        d = f["mid"] - (x0, y0)
        if abs(d[0] * u[1] - d[1] * u[0]) > tol or {mat[c] for c in f["cells"]} != pair:
            continue
        s = d @ u
        segs.append((s - 0.5 * f["length"], s + 0.5 * f["length"]))
    end = 0.0
    for a, b in sorted(segs):
        if b <= tol:
            continue
        if a > end + tol:
            break
        end = max(end, b)
    p = np.array([x0, y0]) + end * u
    if len(bverts) and np.min(np.hypot(*(bverts - p).T)) < tol:
        return np.inf
    return end


def corner_modes(mesh, inter, comp, mats, cx, cy, h_cell, alpha_max=2.5, tol=1e-3,
                 min_turn=0.0, cutoff=0.5):
    """Singular modes at every interior material corner, each cut off (chi = 1
    to cutoff * R2, 0 past R2) before its shortest interface ray ends."""
    G = mats.n_groups
    D = np.asarray(mats.D).reshape(-1, G)
    mat = np.asarray(mesh.material_id)
    vx, vy = np.asarray(mesh.vx), np.asarray(mesh.vy)
    bv = np.unique(np.r_[np.asarray(mesh.bface_v0), np.asarray(mesh.bface_v1)])
    bverts = np.column_stack([vx[bv], vy[bv]])
    modes = []
    for x0, y0 in material_corners(inter, min_turn=min_turn):
        rays = []
        for f in inter:
            a = f["mid"] - 0.5 * f["length"] * f["t"]
            b = f["mid"] + 0.5 * f["length"] * f["t"]
            for p, q in ((a, b), (b, a)):
                if np.hypot(*(p - (x0, y0))) < 1e-6 * max(1.0, h_cell):
                    rays.append(np.arctan2(q[1] - y0, q[0] - x0))
        rays = np.sort(np.mod(rays, 2 * np.pi))
        if len(rays) < 2:
            continue
        rays = rays[np.r_[True, np.diff(rays) > 1e-9]]
        widths = np.diff(np.r_[rays, rays[0] + 2 * np.pi])
        sec_mat, sec_reg = [], []
        for th0, w in zip(rays, widths):
            th = th0 + 0.5 * w
            px, py = x0 + 2.0 * h_cell * np.cos(th), y0 + 2.0 * h_cell * np.sin(th)
            c = int(np.argmin(np.hypot(cx - px, cy - py)))
            sec_mat.append(mat[c])
            sec_reg.append(comp[c])
        R2 = min(_ray_run(x0, y0, th, inter, mat, {sec_mat[i - 1], sec_mat[i]}, bverts, h_cell)
                 for i, th in enumerate(rays))
        for g in range(G):
            sectors = [(w, D[m, g]) for w, m in zip(widths, sec_mat)]
            for alpha in corner_exponents(sectors, amax=alpha_max):
                if abs(alpha - round(alpha)) < tol:
                    continue
                M = np.eye(2)
                Ts = []
                for w, Dk in sectors:
                    T = _transfer(alpha, w, Dk)
                    Ts.append(T)
                    M = T @ M
                s = np.linalg.svd(M - np.eye(2))[2][-1]  # (Phi, D Phi') at the first ray
                AB = []
                for (w, Dk), T in zip(sectors, Ts):
                    AB.append((s[0], s[1] / (Dk * alpha)))
                    s = T @ s
                AB = np.array(AB)
                peak = max(np.abs(A * np.cos(alpha * t) + B * np.sin(alpha * t)).max()
                           for (A, B), t in zip(AB, (np.linspace(0, w, 50) for w, _ in sectors)))
                modes.append(CornerMode(x0, y0, alpha, g, rays, widths, AB / peak, sec_reg,
                                        R1=cutoff * R2, R2=R2))
    return modes


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def _interior_rows(regs, cx, cy, off, N, N_spline, modes, A_op, D, G, orders):
    """PDE rows at every centroid, in the cell's own region:
    -D_g lap(phi_g) + sum_g' A[g,g'] phi_g' (order 0) and its gradient (order 1)."""
    sp = _scipy()[0]
    blocks = []
    for r, R in enumerate(regs):
        x, y = cx[R.cells], cy[R.cells]
        mm = R.material
        A = A_op(mm)
        for order in orders:
            for ex, ey in ([(0, 0)] if order == 0 else [(1, 0), (0, 1)]):
                val = R.rows_sparse(x, y, ex, ey)
                lap = R.rows_sparse(x, y, 2 + ex, ey) + R.rows_sparse(x, y, ex, 2 + ey)
                mv = []
                for e, md in enumerate(modes):
                    if r in md.regions:
                        v, gx, gy = md.eval(x, y)
                        lv, lgx, lgy = md.lap(x, y)
                        mv.append((e, md.group, v if order == 0 else (gx if ex else gy),
                                   lv if order == 0 else (lgx if ex else lgy)))
                for g in range(G):
                    ii, jj, vv = [], [], []
                    for gp in range(G):
                        b = A[g, gp] * val
                        if gp == g:
                            b = b - D[mm, g] * lap
                        b = b.tocoo()
                        ii.append(b.row)
                        jj.append(off[(gp, r)] + b.col)
                        vv.append(b.data)
                    for e, ge, v, lv in mv:
                        col = A[g, ge] * v - (D[mm, g] * lv if ge == g else 0.0)
                        if np.any(col):
                            ii.append(np.arange(len(R.cells)))
                            jj.append(np.full(len(R.cells), N_spline + e))
                            vv.append(col)
                    blocks.append(sp.csr_matrix((np.concatenate(vv), (np.concatenate(ii), np.concatenate(jj))),
                                                shape=(len(R.cells), N)))
    return sp.vstack(blocks, format="csr")


def choose_config(mats, mesh, bc, stride, inclusion_spans=8.0):
    """"corner" if the mesh has a material corner turning more than 30 degrees
    or a small enclosed inclusion, else "smooth"; also returns the inclusions."""
    from ndiffusion._core import cell_areas

    mat = np.asarray(mesh.material_id)
    inter, _ = mesh_faces(mesh, mat)
    if len(material_corners(inter, min_turn=CORNER_TURN)):
        return "corner", []
    inclusions = []
    if inclusion_spans > 0:
        area = np.asarray(cell_areas(mesh))
        regs, comp = regions(mesh, mat, area, stride)
        inclusions = small_inclusions(mesh, bc, mats.n_groups, regs, comp, area, stride, inclusion_spans)
    return ("corner" if inclusions else "smooth"), inclusions


def fit(mats, mesh, flux, k, bc, stride=2.0, config="auto", weight=1e3, ridge=1e-10,
        penalty=None, source=None, inclusion_spans=8.0):
    """Fit the cell-centroid flux on an unstructured mesh.

    `k` is the eigenvalue whose fission term enters the PDE rows (pass
    ``np.inf`` for a fixed-source problem, with `source` the per-unit-volume
    source at the centroids); `bc` is the solver's tag-indexed Robin list.
    `stride` is the knot spacing in cell sizes. `penalty` = (order, lam) adds a
    P-spline difference penalty, which a strongly graded mesh needs (knots are
    spaced from the median cell, so large cells leave coefficients unsupported).

    Returns ``SplineFit(curve, lap, config, evaluate, info)``: the fit and its
    Laplacian at the centroids, both (n_cells, G); the configuration used; and
    ``evaluate(cells, x, y)``, the fit at points inside the given cells."""
    sp, _, _, spsolve = _scipy()
    from ndiffusion._core import cell_areas, cell_centroids

    inclusions = []
    if config == "auto":
        config, inclusions = choose_config(mats, mesh, bc, stride, inclusion_spans)
    if config not in CONFIGS:
        raise ValueError(f"config must be 'auto' or one of {sorted(CONFIGS)}, got {config!r}")
    pde_orders, interior_pde, enrich = (CONFIGS[config][key] for key in ("pde_orders", "interior_pde", "enrich"))
    if source is not None:
        if pde_orders:
            raise ValueError("a fixed-source fit needs config='corner' (the hard PDE rows carry no source)")
        interior_pde = (interior_pde[0], (0,))

    G = mats.n_groups
    D = np.asarray(mats.D).reshape(-1, G)
    Rm = np.asarray(mats.removal).reshape(-1, G)
    S = np.asarray(mats.scatter).reshape(-1, G, G)
    chi = np.asarray(mats.chi).reshape(-1, G)
    nf = np.asarray(mats.nusigf).reshape(-1, G)

    def A_op(m):
        return np.diag(Rm[m]) - S[m] - np.outer(chi[m], nf[m]) / k

    flux = np.asarray(flux, float).reshape(-1, G)
    mat = np.asarray(mesh.material_id)
    cx, cy = (np.asarray(a) for a in cell_centroids(mesh))
    area = np.asarray(cell_areas(mesh))
    inter, bound = mesh_faces(mesh, mat)
    h_cell = np.sqrt(np.median(area))
    breaks = interface_breaks(inter, 3.0 * h_cell)
    regs, comp = regions(mesh, mat, area, stride, breaks)
    off, N = {}, 0
    for g in range(G):
        for r, R in enumerate(regs):
            off[(g, r)] = N
            N += R.n
    gscale = 1.0 / np.abs(flux).max(axis=0)
    modes = []
    if enrich:
        modes = corner_modes(mesh, inter, comp, mats, cx, cy, h_cell, min_turn=CORNER_TURN)
        for md in modes:
            cells = np.where(np.isin(comp, list(md.regions)))[0]
            md.scale = 1.0 / max(np.abs(md.eval(cx[cells], cy[cells])[0]).max(), 1e-300)
    N_spline = N
    N = N_spline + len(modes)

    def mode_terms(r, px, py, deriv, toward=None):
        """{column: (group, value)} of the modes reaching region r at one point:
        deriv None for the value, (nx, ny) for a directional derivative, or
        ("lap", None | (nx, ny)) for the Laplacian or its derivative. A mode's
        gradient jumps on its rays, so `toward` (a point inside the region)
        moves the evaluation a hair to that side."""
        out = {}
        if toward is not None:
            d = np.array([toward[0] - px[0], toward[1] - py[0]])
            d /= np.linalg.norm(d)
            px, py = px + 1e-8 * d[0], py + 1e-8 * d[1]
        for e, md in enumerate(modes):
            if r not in md.regions:
                continue
            lap_ = deriv is not None and deriv[0] == "lap"
            v, gx, gy = md.lap(px, py) if lap_ else md.eval(px, py)
            d = deriv[1] if lap_ else deriv
            out[N_spline + e] = (md.group, v[0] if d is None else d[0] * gx[0] + d[1] * gy[0])
        return out

    # data rows
    di, dj, dv, rhs = [], [], [], []
    row = 0
    for r, R in enumerate(regs):
        B = R.rows_sparse(cx[R.cells], cy[R.cells]).tocoo()
        for g in range(G):
            di.append(row + B.row)
            dj.append(off[(g, r)] + B.col)
            dv.append(B.data * gscale[g])
            for e, md in enumerate(modes):
                if md.group == g and r in md.regions:
                    di.append(row + np.arange(len(R.cells)))
                    dj.append(np.full(len(R.cells), N_spline + e))
                    dv.append(md.eval(cx[R.cells], cy[R.cells])[0] * gscale[g])
            rhs.append(flux[R.cells, g] * gscale[g])
            row += len(R.cells)
    Adata = sp.csr_matrix((np.concatenate(dv), (np.concatenate(di), np.concatenate(dj))), shape=(row, N))
    y = np.concatenate(rhs)

    # constraint rows, each a {column offset: dense 1-row block}
    crows = []

    def pde(r, px, py, toward):
        R = regs[r]
        mm = R.material
        A = A_op(mm)
        for order in pde_orders:
            for ex, ey in ([(0, 0)] if order == 0 else [(1, 0), (0, 1)]):
                lapr = R.rows(px, py, 2 + ex, ey) + R.rows(px, py, ex, 2 + ey)
                val = R.rows(px, py, ex, ey)
                mt_ = mode_terms(r, px, py, None if order == 0 else (ex, ey), toward)
                ml_ = mode_terms(r, px, py, ("lap", None if order == 0 else (ex, ey)), toward)
                for g in range(G):
                    blocks = {}
                    for gp in range(G):
                        b = A[g, gp] * val
                        if gp == g:
                            b = b - D[mm, g] * lapr
                        if np.any(b):
                            blocks[off[(gp, r)]] = b
                    for col, (ge, v) in mt_.items():
                        c = A[g, ge] * v - (D[mm, g] * ml_[col][1] if ge == g else 0.0)
                        if c != 0.0:
                            blocks[col] = np.array([[c]])
                    crows.append(blocks)

    gp2 = np.array([-1, 1]) / np.sqrt(3) / 2  # two Gauss points per face
    for f in inter:
        c1, c2 = f["cells"]
        r1, r2 = comp[c1], comp[c2]
        nrm = np.array([f["t"][1], -f["t"][0]])
        m1, m2 = regs[r1].material, regs[r2].material
        t1, t2 = (cx[c1], cy[c1]), (cx[c2], cy[c2])
        for s_ in gp2:
            p = f["mid"] + s_ * f["length"] * f["t"]
            px, py = p[:1], p[1:]
            v1, v2 = regs[r1].rows(px, py), regs[r2].rows(px, py)
            n1 = nrm[0] * regs[r1].rows(px, py, 1, 0) + nrm[1] * regs[r1].rows(px, py, 0, 1)
            n2 = nrm[0] * regs[r2].rows(px, py, 1, 0) + nrm[1] * regs[r2].rows(px, py, 0, 1)
            val1, val2 = mode_terms(r1, px, py, None, t1), mode_terms(r2, px, py, None, t2)
            dn1, dn2 = mode_terms(r1, px, py, tuple(nrm), t1), mode_terms(r2, px, py, tuple(nrm), t2)
            for g in range(G):
                blk = {off[(g, r1)]: v1, off[(g, r2)]: -v2}
                for col in set(val1) | set(val2):
                    if (val1.get(col) or val2.get(col))[0] == g:
                        blk[col] = np.array([[val1.get(col, (g, 0.0))[1] - val2.get(col, (g, 0.0))[1]]])
                crows.append(blk)
                blk = {off[(g, r1)]: D[m1, g] * n1, off[(g, r2)]: -D[m2, g] * n2}
                for col in set(dn1) | set(dn2):
                    if (dn1.get(col) or dn2.get(col))[0] == g:
                        blk[col] = np.array([[D[m1, g] * dn1.get(col, (g, 0.0))[1]
                                              - D[m2, g] * dn2.get(col, (g, 0.0))[1]]])
                crows.append(blk)
            pde(r1, px, py, t1)
            pde(r2, px, py, t2)
    for f in bound:
        c0 = f["cells"][0]
        r = comp[c0]
        R = regs[r]
        tc = (cx[c0], cy[c0])
        for s_ in gp2:
            p = f["mid"] + s_ * f["length"] * f["t"]
            px, py = p[:1], p[1:]
            v = R.rows(px, py)
            dn = f["n"][0] * R.rows(px, py, 1, 0) + f["n"][1] * R.rows(px, py, 0, 1)
            mv, md_ = mode_terms(r, px, py, None, tc), mode_terms(r, px, py, tuple(f["n"]), tc)
            for g in range(G):
                b = bc[f["tag"] * G + g]
                blk = {off[(g, r)]: b.A * v + b.B * dn}
                for col, (ge, val_) in mv.items():
                    if ge == g:
                        blk[col] = np.array([[b.A * val_ + b.B * md_[col][1]]])
                crows.append(blk)
            pde(r, px, py, tc)

    ci, cj, cvv = [], [], []
    for i, blocks in enumerate(crows):
        norm = np.sqrt(sum(np.sum(b**2) for b in blocks.values()))
        for o, b in blocks.items():
            jj = np.nonzero(b[0])[0]
            ci.append(np.full(len(jj), i))
            cj.append(o + jj)
            cvv.append(b[0, jj] / norm)
    C = (sp.csr_matrix((np.concatenate(cvv), (np.concatenate(ci), np.concatenate(cj))), shape=(len(crows), N))
         if crows else sp.csr_matrix((0, N)))
    data_w = np.sqrt(Adata.multiply(Adata).sum(axis=1)).mean()
    w = weight * data_w
    Nm = Adata.T @ Adata + (w**2) * (C.T @ C) + ridge * sp.identity(N)
    rhs_total = Adata.T @ y
    if interior_pde is not None:
        w_int, orders = interior_pde
        Ci = _interior_rows(regs, cx, cy, off, N, N_spline, modes, A_op, D, G, orders)
        rn = np.maximum(np.sqrt(np.asarray(Ci.multiply(Ci).sum(axis=1)).ravel()), 1e-300)
        Ci = sp.diags(1.0 / rn) @ Ci
        wi = w_int * data_w
        Nm = Nm + (wi**2) * (Ci.T @ Ci)
        if source is not None:
            q = np.asarray(source, float).reshape(-1, G)
            qrows = np.concatenate([q[R.cells, g] for R in regs for g in range(G)])
            rhs_total = rhs_total + (wi**2) * (Ci.T @ (qrows / rn))
    if penalty is not None:
        order, lam = penalty
        Dp = sp.block_diag([_diff_penalty(R.nx, R.ny, order) for g in range(G) for R in regs], format="csr")
        if modes:
            Dp = sp.hstack([Dp, sp.csr_matrix((Dp.shape[0], len(modes)))], format="csr")
        Nm = Nm + lam * (Adata.multiply(Adata).sum() / N) * (Dp.T @ Dp)
    coef = spsolve(Nm.tocsc(), rhs_total)

    curve = np.zeros_like(flux)
    lap = np.zeros_like(flux)
    for r, R in enumerate(regs):
        x, yy = cx[R.cells], cy[R.cells]
        V = R.rows_sparse(x, yy)
        L = R.rows_sparse(x, yy, 2, 0) + R.rows_sparse(x, yy, 0, 2)
        for g in range(G):
            cg = coef[off[(g, r)]:off[(g, r)] + R.n]
            curve[R.cells, g] = V @ cg
            lap[R.cells, g] = L @ cg
        for e, md in enumerate(modes):
            if r in md.regions:
                curve[R.cells, md.group] += coef[N_spline + e] * md.eval(x, yy)[0]
                lap[R.cells, md.group] += coef[N_spline + e] * md.lap(x, yy)[0]

    def evaluate(cells, x, y):
        """The fit at points (x, y), each inside the matching cell, as (n, G)."""
        cells, x, y = np.asarray(cells), np.asarray(x, float), np.asarray(y, float)
        out = np.zeros((len(x), G))
        for r in np.unique(comp[cells]):
            sel = np.where(comp[cells] == r)[0]
            R = regs[r]
            V = R.rows_sparse(x[sel], y[sel])
            for g in range(G):
                out[sel, g] = V @ coef[off[(g, r)]:off[(g, r)] + R.n]
            for e, md in enumerate(modes):
                if r in md.regions:
                    out[sel, md.group] += coef[N_spline + e] * md.eval(x[sel], y[sel])[0]
        return out

    info = dict(n_unknowns=N, n_constraints=len(crows), regions=len(regs), inclusions=inclusions,
                modes=[(md.x0, md.y0, md.alpha, md.group, md.R2) for md in modes],
                constraint_residual=np.linalg.norm(C @ coef) / max(1.0, np.linalg.norm(coef)))
    return SplineFit(curve, lap, config, evaluate, info)
