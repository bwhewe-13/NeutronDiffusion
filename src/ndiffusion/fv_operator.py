"""The unstructured finite-volume operator as a sparse matrix.

The unstructured solvers are matrix-free and treat the non-orthogonal correction
as a deferred source, iterated to convergence. A converged fixed-source solve
therefore satisfies one linear system,

    M phi = V q,   M = diag(a_diag) - offdiag(D_harm a_coef) - scatter V - C,

where ``C phi`` is the face sum of ``D_f (w0 grad_0 + (1 - w0) grad_1) . T`` with
the solver's least-squares cell gradients, and a k-eigenvalue solve satisfies
``M phi = F phi / k`` with ``F`` the volume-integrated fission production.
:func:`unstructured_operator` assembles ``M`` and ``F`` explicitly, mirroring
``solver_2d_unstructured.cpp`` (``fill_face_geometry``, ``build_lsq_gradients``,
``build_diagonals``, ``cell_gradients``, ``non_orthogonal_correction``).

The explicit matrix is what an exact discrete adjoint needs: running a solver on
adjoint cross sections (:func:`make_adjoint_materials`) transposes the scattering
and fission but not ``C``, because the least-squares gradient correction is not
symmetric, so on a non-orthogonal mesh it solves ``(M + C)^T - C`` rather than
``M^T``. On an orthogonal mesh ``C`` vanishes and the two agree.

Unknowns are ordered like the solver flux, ``index = cell * G + g``. Periodic
meshes are not supported. Requires SciPy.
"""

from collections import namedtuple

import numpy as np

UnstructuredOperator = namedtuple("UnstructuredOperator",
                                  ["loss", "production", "correction", "area"])
UnstructuredOperator.__doc__ = """Sparse unstructured FVM operator.

``loss`` is ``M`` (CSR, volume-integrated, correction included), ``production`` is
``F``, ``correction`` is the non-orthogonal part ``C`` alone (``M = M_imp - C``),
and ``area`` holds the cell areas."""


def _faces(mesh):
    """Interior faces (c0, c1, v_lo, v_hi) and boundary faces (cell, v_lo, v_hi)."""
    off = np.asarray(mesh.cell_offsets, dtype=np.int64)
    cv = np.asarray(mesh.cell_vertices, dtype=np.int64)
    n = len(off) - 1
    cell = np.repeat(np.arange(n), np.diff(off))
    nxt = np.arange(len(cv)) + 1
    nxt[off[1:] - 1] = off[:-1]
    va, vb = cv, cv[nxt]
    lo, hi = np.minimum(va, vb), np.maximum(va, vb)
    key = lo * (int(cv.max()) + 1) + hi
    order = np.argsort(key, kind="stable")
    ks = key[order]
    first = np.r_[True, ks[1:] != ks[:-1]]
    start = np.flatnonzero(first)
    count = np.diff(np.r_[start, len(ks)])
    if count.max() > 2:
        raise ValueError("mesh has an edge shared by more than two cells")
    pair = count == 2
    i0, i1 = order[start[pair]], order[start[pair] + 1]
    ib = order[start[~pair]]
    return n, (cell[i0], cell[i1], lo[i0], hi[i0]), (cell[ib], lo[ib], hi[ib])


def unstructured_operator(mats, mesh, bc):
    """Assemble the sparse operator of the unstructured FVM solvers.

    Parameters
    ----------
    mats : Materials
    mesh : UnstructuredMesh2D
    bc : list of BoundaryCondition
        Tag-indexed, ``bc[tag * G + g]``, as passed to the solver.

    Returns
    -------
    UnstructuredOperator
        ``loss`` (M), ``production`` (F), ``correction`` (C), ``area``. A
        fixed-source solution satisfies ``loss @ phi.ravel() = (q * area[:, None]).ravel()``
        and a k-eigenvalue one ``loss @ phi.ravel() = production @ phi.ravel() / k``,
        both to the solver's tolerance.
    """
    import scipy.sparse as sp

    from ndiffusion._core import cell_areas, cell_centroids
    from ndiffusion.nearby import _fission_matrix_mode, _xs_arrays

    if len(mesh.periodic_a0):
        raise ValueError("unstructured_operator does not support periodic meshes")
    G = mats.n_groups
    D, removal, chi, nusigf, scatter = _xs_arrays(mats)
    mat = np.asarray(mesh.material_id, dtype=np.int64)
    vx, vy = np.asarray(mesh.vx, dtype=float), np.asarray(mesh.vy, dtype=float)
    cx, cy = (np.asarray(a, dtype=float) for a in cell_centroids(mesh))
    area = np.asarray(cell_areas(mesh), dtype=float)
    n, (c0, c1, a, b), (bcell, ba, bb) = _faces(mesh)

    # Boundary tags (an edge missing from the boundary list gets tag 0, as in the solver).
    bv0, bv1 = np.asarray(mesh.bface_v0), np.asarray(mesh.bface_v1)
    tags = np.asarray(mesh.bface_bc_tag) if len(mesh.bface_bc_tag) else np.zeros(len(bv0), int)
    tagmap = {(min(p, q), max(p, q)): t
              for p, q, t in zip(bv0.tolist(), bv1.tolist(), tags.tolist())}
    btag = np.array([tagmap.get((p, q), 0) for p, q in zip(ba.tolist(), bb.tolist())],
                    dtype=np.int64)
    n_tags = len(bc) // G
    A_bc = np.zeros((len(btag), G))
    B_bc = np.zeros((len(btag), G))
    valid = (btag >= 0) & (btag < n_tags)
    for g in range(G):
        A_bc[valid, g] = [bc[t * G + g].A for t in btag[valid]]
        B_bc[valid, g] = [bc[t * G + g].B for t in btag[valid]]

    # Interior faces: surface vector, over-relaxed split S = E + T, interpolation weight.
    ex, ey = cx[c1] - cx[c0], cy[c1] - cy[c0]
    sx, sy = vy[b] - vy[a], -(vx[b] - vx[a])
    flip = sx * ex + sy * ey < 0.0
    sx[flip] *= -1.0
    sy[flip] *= -1.0
    length = np.hypot(sx, sy)
    dist = np.hypot(ex, ey)
    eux, euy = ex / dist, ey / dist
    e_dot_s = eux * sx + euy * sy
    s_dot_s = sx * sx + sy * sy
    tangential = e_dot_s <= 1e-12 * np.sqrt(s_dot_s)
    e_mag = np.where(tangential, 0.0, s_dot_s / np.where(tangential, 1.0, e_dot_s))
    a_coef = np.where(tangential, length / dist, e_mag / dist)
    tx = np.where(tangential, 0.0, sx - e_mag * eux)
    ty = np.where(tangential, 0.0, sy - e_mag * euy)
    mx, my = 0.5 * (vx[a] + vx[b]), 0.5 * (vy[a] + vy[b])
    d0 = np.hypot(cx[c0] - mx, cy[c0] - my)
    d1 = np.hypot(cx[c1] - mx, cy[c1] - my)
    w0 = np.where(d0 + d1 > 0.0, d1 / np.where(d0 + d1 > 0.0, d0 + d1, 1.0), 0.5)

    # Boundary faces: normal distance from the centroid to the face.
    bsx, bsy = vy[bb] - vy[ba], -(vx[bb] - vx[ba])
    bex = 0.5 * (vx[ba] + vx[bb]) - cx[bcell]
    bey = 0.5 * (vy[ba] + vy[bb]) - cy[bcell]
    bflip = bsx * bex + bsy * bey < 0.0
    bsx[bflip] *= -1.0
    bsy[bflip] *= -1.0
    blen = np.hypot(bsx, bsy)
    bdn = (bex * bsx + bey * bsy) / blen
    pos = bdn > 0.0
    bdn_safe = np.where(pos, bdn, 1.0)
    b_acoef = np.where(pos, blen / bdn_safe, 0.0)

    # Weighted least-squares gradient coefficients (boundary samples at the face midpoint).
    odx = np.r_[ex, -ex, bsx / blen * bdn]
    ody = np.r_[ey, -ey, bsy / blen * bdn]
    oc = np.r_[c0, c1, bcell]
    w = 1.0 / (odx**2 + ody**2)
    Mxx = np.bincount(oc, w * odx * odx, n)
    Mxy = np.bincount(oc, w * odx * ody, n)
    Myy = np.bincount(oc, w * ody * ody, n)
    det = Mxx * Myy - Mxy**2
    ok = np.abs(det) > 1e-14 * (Mxx + Myy) ** 2
    det = np.where(ok, det, 1.0)
    ixx, ixy, iyy = (np.where(ok, v, 0.0) for v in (Myy / det, -Mxy / det, Mxx / det))
    lx = ixx[oc] * (w * odx) + ixy[oc] * (w * ody)
    ly = ixy[oc] * (w * odx) + iyy[oc] * (w * ody)
    nI = len(c0)
    l0x, l1x, lbx = lx[:nI], lx[nI:2 * nI], lx[2 * nI:]
    l0y, l1y, lby = ly[:nI], ly[nI:2 * nI], ly[2 * nI:]

    cells = np.arange(n)
    rows, cols, vals = [], [], []
    crows, ccols, cvals = [], [], []
    nonorth = np.flatnonzero(np.hypot(tx, ty) > 1e-10 * length)
    nf = len(nonorth)
    for g in range(G):
        Dg = D[mat, g]
        a_ij = 2.0 * Dg[c0] * Dg[c1] / (Dg[c0] + Dg[c1]) * a_coef
        diag = np.bincount(c0, a_ij, n) + np.bincount(c1, a_ij, n)
        A, B = A_bc[:, g], B_bc[:, g]
        den = A + np.where(pos, B / bdn_safe, 0.0)
        good = valid & (np.abs(den) > 1e-30)
        den_safe = np.where(good, den, 1.0)
        diag += np.bincount(bcell, np.where(good, Dg[bcell] * b_acoef * A / den_safe, 0.0), n)
        diag += removal[mat, g] * area
        rows += [cells * G + g, c0 * G + g, c1 * G + g]
        cols += [cells * G + g, c1 * G + g, c0 * G + g]
        vals += [diag, -a_ij, -a_ij]
        for gp in range(G):
            if gp != g:
                rows.append(cells * G + g)
                cols.append(cells * G + gp)
                vals.append(-scatter[mat, g, gp] * area)
        if nf == 0:
            continue
        # Cell gradients as n x n operators, grad_c = sum_f lsq_f (phi_nbr - phi_c); a boundary
        # face samples the Robin surface value, phi_s - phi_c = delta phi_c.
        delta = np.where(good, -A / den_safe, 0.0)
        gr = np.r_[c0, c0, c1, c1, bcell]
        gc = np.r_[c1, c0, c0, c1, bcell]
        Gx = sp.csr_matrix((np.r_[l0x, -l0x, l1x, -l1x, lbx * delta], (gr, gc)), shape=(n, n))
        Gy = sp.csr_matrix((np.r_[l0y, -l0y, l1y, -l1y, lby * delta], (gr, gc)), shape=(n, n))
        f = nonorth
        face = np.r_[np.arange(nf), np.arange(nf)]
        W = sp.csr_matrix((np.r_[w0[f], 1.0 - w0[f]], (face, np.r_[c0[f], c1[f]])), shape=(nf, n))
        Df = 2.0 * Dg[c0[f]] * Dg[c1[f]] / (Dg[c0[f]] + Dg[c1[f]])
        flux = sp.diags(Df * tx[f]) @ (W @ Gx) + sp.diags(Df * ty[f]) @ (W @ Gy)
        S = sp.csr_matrix((np.r_[np.ones(nf), -np.ones(nf)], (np.r_[c0[f], c1[f]], face)),
                          shape=(n, nf))
        Cg = (S @ flux).tocoo()
        crows.append(Cg.row * G + g)
        ccols.append(Cg.col * G + g)
        cvals.append(Cg.data)

    N = n * G
    cat = lambda parts: np.concatenate(parts) if parts else np.zeros(0)  # noqa: E731
    C = sp.csr_matrix((cat(cvals), (cat(crows).astype(np.int64), cat(ccols).astype(np.int64))),
                      shape=(N, N))
    M = sp.csr_matrix((cat(vals), (cat(rows), cat(cols))), shape=(N, N)) - C

    frows, fcols, fvals = [], [], []
    matrix_mode = _fission_matrix_mode(mats)
    for g in range(G):
        for gp in range(G):
            coef = nusigf[mat, g, gp] if matrix_mode else chi[mat, g] * nusigf[mat, gp]
            frows.append(cells * G + g)
            fcols.append(cells * G + gp)
            fvals.append(coef * area)
    F = sp.csr_matrix((cat(fvals), (cat(frows), cat(fcols))), shape=(N, N))
    F.eliminate_zeros()
    return UnstructuredOperator(M.tocsr(), F, C, area)
