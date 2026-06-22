#pragma once

#include <ndiffusion/types.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @file solver_detail.hpp
 * @brief Shared, dimension-agnostic helpers for the diffusion solvers.
 *
 * These free functions factor out logic that was previously copy-pasted across
 * the 1-D, 2-D structured, and 2-D unstructured solver translation units:
 * the L2 norms, the internal/external flux transpose, the fission-source
 * assembly, the power-iteration driver, and input validation.
 *
 * The geometry-specific spatial sweep (`solve_A` / `solve_step`) is **not**
 * here - it differs fundamentally per discretization (banded Thomas vs.
 * line-TDMA vs. point Gauss-Seidel) and stays in each solver.
 *
 * @par Flux storage convention
 * Internally each solver stores flux as `phi[g * stride + cell]`, where
 * `stride` is the per-group storage length (`cells + 1` for the 1-D / 2-D
 * structured solvers, which carry a ghost boundary row; `n_cells` for the
 * unstructured FVM solver). The public flux layout is the transpose,
 * `flux[cell * n_groups + g]`.
 */

namespace ndiffusion {
namespace detail {

// ============================================================================
// Norms
// ============================================================================

/// Euclidean (L2) norm of a vector.
inline double norm2(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

/// L2 norm of the difference (a - b).  `a` and `b` must be the same length.
inline double l2_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    const int n = static_cast<int>(a.size());
    for (int i = 0; i < n; ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

// ============================================================================
// Flux transpose between internal [g*stride+cell] and public [cell*groups+g]
// ============================================================================

/// Transpose internal `phi[g*stride+cell]` -> public `out[cell*groups+g]`.
/// Only the first `cells` cells of each group are emitted (drops any ghost row).
inline void pack_flux(const std::vector<double>& phi,
                      int cells, int groups, int stride,
                      std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(cells) * groups, 0.0);
    for (int g = 0; g < groups; ++g)
        for (int c = 0; c < cells; ++c)
            out[c * groups + g] = phi[g * stride + c];
}

/// Transpose public `ext[cell*groups+g]` -> internal `internal[g*stride+cell]`,
/// optionally scaling each cell by `weight[cell]` (e.g. cell area for FVM).
/// `internal` is sized to `groups*stride` and zero-filled (ghost rows stay 0).
inline void unpack_flux(const std::vector<double>& ext,
                        int cells, int groups, int stride,
                        const std::vector<double>* weight,
                        std::vector<double>& internal) {
    internal.assign(static_cast<std::size_t>(groups) * stride, 0.0);
    for (int g = 0; g < groups; ++g)
        for (int c = 0; c < cells; ++c) {
            const double w = weight ? (*weight)[c] : 1.0;
            internal[g * stride + c] = ext[c * groups + g] * w;
        }
}

// ============================================================================
// Fission source  b = B * phi
// ============================================================================

/// Assemble the fission source `out[g*stride+c]` from `phi[g*stride+c]`.
///
/// Handles both representations in one place: when `use_fission_matrix()` the
/// `nusigf` data is a full transfer matrix `F[g_to][g_from]`; otherwise it is
/// the standard `chi_g * nu_sigf_gp` product.  Each cell is optionally scaled
/// by `weight[c]` (cell area for the volume-integrated FVM solver; pass
/// `nullptr` for the per-unit-volume finite-difference solvers).
inline void accumulate_fission(const Materials& mats,
                               const std::vector<int>& material_id,
                               int groups, int cells, int stride,
                               const std::vector<double>* weight,
                               const std::vector<double>& phi,
                               std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(groups) * stride, 0.0);
    const bool fis_mat = mats.use_fission_matrix();
    for (int g = 0; g < groups; ++g) {
        for (int c = 0; c < cells; ++c) {
            const int mat = material_id[c];
            double src = 0.0;
            for (int gp = 0; gp < groups; ++gp) {
                if (fis_mat)
                    src += mats.nu_sigf_mat(mat, g, gp) * phi[gp * stride + c];
                else
                    src += mats.chi_g(mat, g) * mats.nu_sigf(mat, gp) *
                           phi[gp * stride + c];
            }
            const double w = weight ? (*weight)[c] : 1.0;
            out[g * stride + c] = src * w;
        }
    }
}

// ============================================================================
// Power iteration  A phi = (1/k) B phi
// ============================================================================

/// Result of a power-iteration solve, in internal `[g*stride+cell]` layout.
struct PowerResult {
    std::vector<double> phi;  ///< Converged flux, internal layout.
    double keff;              ///< Effective multiplication factor.
    int    iters;             ///< Power-iteration count.
    double change;            ///< Final L2 flux-change norm.
};

/// Generic power-iteration driver shared by every k-eigenvalue solver.
///
/// @param total      Length of the internal flux vector (`groups * stride`).
/// @param apply_B    Callable `(const vec& phi_in, vec& b_out)` - fission source.
/// @param solve_A    Callable `(const vec& b, vec& phi)` - in-place linear solve,
///                   warm-started from the current `phi`.
///
/// `srand(42)` seeds a fixed initial guess so eigen results are reproducible.
template <class ApplyB, class SolveA>
PowerResult power_iteration(int total, double epsilon, int max_outer,
                            bool verbose, ApplyB apply_B, SolveA solve_A) {
    std::srand(42);
    std::vector<double> phi(total);
    for (int i = 0; i < total; ++i)
        phi[i] = static_cast<double>(std::rand()) / RAND_MAX + 1e-10;
    double nrm = norm2(phi);
    for (double& v : phi) v /= nrm;

    std::vector<double> b(total);
    double keff   = 1.0;
    double change = 1.0;
    int    iter   = 0;

    while (change > epsilon && iter < max_outer) {
        const std::vector<double> phi_old = phi;

        apply_B(phi_old, b);
        solve_A(b, phi);

        keff = norm2(phi);
        for (double& v : phi) v /= keff;

        change = l2_diff(phi, phi_old);

        if (verbose)
            std::printf("Iter: %3d  keff: %.8f  change: %.2e\n",
                        iter + 1, keff, change);
        ++iter;
    }

    return {std::move(phi), keff, iter, change};
}

// ============================================================================
// Matrix-free Jacobi-preconditioned Conjugate Gradient  (Option B prototype)
// ============================================================================

/// Solve a symmetric-positive-definite system `A x = rhs` matrix-free, using
/// Jacobi (diagonal) preconditioned Conjugate Gradient.  `A` is never assembled;
/// it is supplied as a callable `apply_A(const vec& v, vec& out)` computing
/// `out = A v`.  `diag` points to the `n` diagonal entries of `A` (the
/// preconditioner `M = diag(A)`).  `x` is updated in place and warm-started from
/// its incoming value.
///
/// @return Number of CG iterations performed; `converged` reports whether the
///         relative residual `||rhs - A x|| / ||rhs||` fell below `tol`.
///
/// @note CG is only valid when `A` is SPD.  For the multigroup diffusion
///       operator this means the **within-group** leakage+removal block - the
///       cross-group scatter coupling (which is non-symmetric) must be handled
///       outside, by a Gauss-Seidel sweep over energy groups.
template <class ApplyA>
int cg_solve(int n,
             const std::vector<double>& rhs,
             std::vector<double>&       x,
             const double*              diag,
             double tol, int max_it,
             ApplyA apply_A,
             bool& converged) {
    std::vector<double> r(n), z(n), p(n), Ap(n);

    apply_A(x, Ap);
    for (int i = 0; i < n; ++i) r[i] = rhs[i] - Ap[i];

    double bnorm = norm2(rhs);
    if (bnorm == 0.0) bnorm = 1.0;

    for (int i = 0; i < n; ++i) z[i] = r[i] / diag[i];
    p = z;
    double rz = 0.0;
    for (int i = 0; i < n; ++i) rz += r[i] * z[i];

    converged = (norm2(r) / bnorm <= tol);
    int it = 0;
    for (; it < max_it && !converged; ++it) {
        apply_A(p, Ap);
        double pAp = 0.0;
        for (int i = 0; i < n; ++i) pAp += p[i] * Ap[i];
        if (pAp <= 0.0) break;  // breakdown / loss of positive-definiteness

        const double alpha = rz / pAp;
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        if (norm2(r) / bnorm <= tol) { converged = true; ++it; break; }

        for (int i = 0; i < n; ++i) z[i] = r[i] / diag[i];
        double rz_new = 0.0;
        for (int i = 0; i < n; ++i) rz_new += r[i] * z[i];
        const double beta = rz_new / rz;
        for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
        rz = rz_new;
    }
    return it;
}

/// Read a boolean from the environment: true unless unset/empty or starting with
/// one of `0 f F n N`.  Used to pick the inner linear solver at runtime
/// (`NDIFFUSION_KEIG_CG=1`) without recompiling, for A/B benchmarking.
inline bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') return false;
    const char c = v[0];
    return !(c == '0' || c == 'f' || c == 'F' || c == 'n' || c == 'N');
}

// ============================================================================
// Convergence reporting
// ============================================================================

/// Emit a stderr warning when an inner linear solve fails to converge within
/// its iteration cap.  This prevents the power iteration from *silently*
/// returning a wrong eigenvalue built on an under-converged inner solve.
inline void warn_inner_not_converged(const char* solver, int max_inner) {
    std::fprintf(stderr,
        "ndiffusion warning: %s inner Gauss-Seidel solve did not converge "
        "within max_inner=%d; the k-eigenvalue may be inaccurate. Increase "
        "max_inner (finer and multi-group meshes need more inner iterations).\n",
        solver, max_inner);
}

// ============================================================================
// Input validation
// ============================================================================

/// Throw std::invalid_argument unless `edges` is strictly increasing
/// (every cell has positive width).  `name` appears in the message.
inline void validate_increasing(const std::vector<double>& edges,
                                const char* name) {
    for (std::size_t i = 1; i < edges.size(); ++i)
        if (edges[i] <= edges[i - 1])
            throw std::invalid_argument(
                std::string(name) + " must be strictly increasing "
                "(every cell needs a positive width)");
}

/// Throw std::invalid_argument unless every id is in [0, n_mat).
inline void validate_material_ids(const std::vector<int>& ids, int n_mat,
                                  const char* name) {
    for (int id : ids)
        if (id < 0 || id >= n_mat)
            throw std::invalid_argument(
                std::string(name) + " contains a material index out of range "
                "[0, n_mat)");
}

}  // namespace detail
}  // namespace ndiffusion
