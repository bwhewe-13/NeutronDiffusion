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
// Cooperative interruption
// ============================================================================

/// Hook invoked periodically from the long-running iteration drivers so a host
/// can abort a solve.  A plain function pointer, installed once at start-up and
/// never reassigned, so concurrent solves only read it; the core library needs
/// to know nothing about the host (the Python bindings install one that checks
/// for a pending KeyboardInterrupt).
///
/// A hook aborts by throwing.  k-eigenvalue and fixed-source solves keep their
/// iterate in locals, so an abort discards it.  The transient solvers are
/// checked only *between* steps, never inside the Gauss-Seidel sweep that
/// rewrites `phi_` group by group, so an interrupted `run()` is left at the last
/// completed step and remains usable.
using InterruptHook = void (*)();

/// The single installed hook (null = no interruption support).
inline InterruptHook& interrupt_hook() {
    static InterruptHook hook = nullptr;
    return hook;
}

/// Give the host a chance to abort.  Cheap enough for a per-iteration call.
inline void check_interrupt() {
    if (const InterruptHook hook = interrupt_hook()) hook();
}

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

/// Relative L2 change ||a - b|| / ||a||, guarded against a zero denominator.
/// Mesh-size independent, unlike the raw absolute difference.
inline double rel_l2_diff(const std::vector<double>& a,
                          const std::vector<double>& b) {
    const double na = norm2(a);
    return l2_diff(a, b) / (na > 0.0 ? na : 1.0);
}

// ============================================================================
// Thomas / TDMA tridiagonal solver
// ============================================================================

/// Solve the n-equation tridiagonal system with lower, diag, upper bands.
/// `lower[0]` and `upper[n-1]` are not referenced.  `c` and `d` are caller
/// scratch vectors resized here, so hot loops can reuse the allocations.
inline void thomas(
    const std::vector<double>& lower,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& rhs,
          std::vector<double>& x,
    int n,
    std::vector<double>& c,
    std::vector<double>& d
) {
    c.resize(n);
    d.resize(n);

    c[0] = upper[0] / diag[0];
    d[0] = rhs[0]   / diag[0];
    for (int i = 1; i < n; ++i) {
        const double denom = diag[i] - lower[i] * c[i - 1];
        c[i] = upper[i] / denom;
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / denom;
    }

    x[n - 1] = d[n - 1];
    for (int i = n - 2; i >= 0; --i)
        x[i] = d[i] - c[i] * x[i + 1];
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
// Delayed neutron precursors and the theta time discretization
// ============================================================================
//
// Write the flux-driven half of the kinetics system as
//
//     L phi = -A phi + scatter + (1 - beta) chi_p F,
//     F(r)  = sum_g' nu_sigf_g' phi_g'     (production rate per unit volume)
//
// - that is, everything the flux drives, with fission contributing only its
// *prompt* part - and the delayed emission as D(C) = sum_i chi_d,i lambda_i C_i.
// The theta method weights the right-hand side between the two time levels:
//
//     (1/(v dt)) (phi^{n+1} - phi^n)
//         = theta [L phi^{n+1} + D(C^{n+1})] + (1-theta) [L phi^n + D(C^n)],
//
//     (C_i^{n+1} - C_i^n) / dt
//         = theta [beta_i F^{n+1} - lam_i C_i^{n+1}]
//         + (1-theta) [beta_i F^n - lam_i C_i^n].
//
// theta = 1 is backward Euler (first order); theta = 1/2 is Crank-Nicolson
// (second order).  The precursor balance still eliminates C^{n+1} in closed
// form,
//
//     C_i^{n+1} = [ C_i^n (1 - (1-theta) lam_i dt)
//                   + dt beta_i (theta F^{n+1} + (1-theta) F^n) ]
//                 / (1 + theta lam_i dt),                                  (*)
//
// and substituting that into theta D(C^{n+1}) + (1-theta) D(C^n), then dividing
// the flux equation through by theta, leaves exactly the shape the solvers
// already assemble:
//
//     [A_g + 1/(theta v_g dt)] phi_g^{n+1}
//       =  (1/(theta v_g dt)) phi_g^n
//        + chi_eff,g(m, theta*dt) * F^{n+1}       [implicit fission]
//        + scatter from phi^{n+1}                 [implicit, Gauss-Seidel]
//        + Q_d,g                                  [known: C^n and F^n]
//        + ((1-theta)/theta) E_g,                 [explicit flux residual]
//
//     chi_eff,g = (1 - beta_m) chi_p,g
//               + sum_i chi_d,i,g beta_i * lam_i (theta dt)/(1 + lam_i theta dt),
//     Q_d,g     = sum_i chi_d,i,g lam_i
//                 [ C_i^n (1 - (1-theta) lam_i dt) / (1 + theta lam_i dt)
//                 + (1-theta) dt beta_i F^n       / (1 + theta lam_i dt)
//                 + ((1-theta)/theta) C_i^n ],
//     E_g       = -[A phi^n]_g + scatter from phi^n + [(1-beta) chi_p F^n]_g.
//
// The first line is the point worth noticing: the F^{n+1}-proportional part of
// (*) carries the weight theta lam_i dt / (1 + theta lam_i dt), which is the
// *existing* backward-Euler weight evaluated at theta*dt.  So chi_eff needs no
// theta-specific code at all - build_chi_effective(mats, delayed, theta*dt)
// is the whole change - and it still has exactly the layout of Materials::chi,
// so the fission source is assembled by accumulate_fission() against a shadow
// Materials as before.  E_g is likewise assembled against the shadow materials
// at dt = 0, where the emitted weight vanishes and chi_eff collapses to the
// prompt spectrum (1-beta) chi_p.
//
// The two limits carry over with dt replaced by theta*dt: theta*dt -> 0 gives
// chi_eff -> (1-beta) chi_p (prompt only), and theta*dt -> inf gives the total
// fission spectrum.  Better still, a critical system with equilibrium
// precursors is an exact fixed point at *any* theta and any step size: if
// L phi^n + D(C^n) = 0 and C_i^n = beta_i F / lam_i, then phi^{n+1} = phi^n,
// C^{n+1} = C^n satisfies both discrete equations identically - (*) reduces to
// C_i lam_i dt = dt beta_i F.
//
// Stability.
// theta >= 1/2 is A-stable, which is why the solvers reject anything smaller.
// theta = 1/2 is *not* L-stable, though: a mode with |z| = |lam| dt >> 1 has
// amplification factor (1 - (1-theta) z)/(1 + theta z) -> -(1-theta)/theta,
// which approaches -1 as theta -> 1/2.  Stiff spatial harmonics (decaying at
// ~v (Sigma_r + D B^2), so |z| >> 1 at any practical dt) and the precursor
// recursion in (*) therefore ring rather than damp.  Both stay bounded, neither
// is monotone.  The cure is to damp a discontinuity - a step insertion via
// update_materials - with one or two theta = 1 steps before dropping back to
// theta = 1/2.
//
// Cost of the default.
// Every expression below is arranged so that theta == 1.0 reduces to the
// original backward-Euler expression *exactly* in floating point, not merely
// algebraically: 1.0*dt is dt, C*1.0 is C, and adding 0.0 is a no-op.  The
// explicit residual E is skipped entirely when theta == 1.  So the default
// costs nothing and is bit-for-bit what it always was.

/// Throw std::invalid_argument unless `theta` is in the A-stable range [1/2, 1].
///
/// Below 1/2 the scheme is only conditionally stable, and the condition is
/// hopeless here: the fast spatial modes decay at ~v Sigma_r, which is order
/// 1e4 per second, so any practically useful dt would diverge silently.
inline void validate_theta(double theta) {
    if (!(theta >= 0.5) || !(theta <= 1.0))
        throw std::invalid_argument(
            "theta must be in [0.5, 1]: 1 is backward Euler (first order), 0.5 "
            "is Crank-Nicolson (second order), and below 0.5 the time "
            "discretization is no longer unconditionally stable (got " +
            std::to_string(theta) + ")");
}

/// Total fission neutron production per unit flux in group `g_from`, for
/// material `m`.
///
/// In standard mode this is just `nu_sigf(m, g_from)`.  In fission-matrix mode
/// the fission spectrum is folded into `nusigf`, so the production rate is the
/// **column sum** `sum_g_to F[g_to][g_from]` - the total number of neutrons
/// emitted per fission caused by a group-`g_from` neutron.  The two agree when
/// the matrix is separable, since `sum_g chi_g = 1`.
inline double production_xs(const Materials& mats, int m, int g_from) {
    if (!mats.use_fission_matrix()) return mats.nu_sigf(m, g_from);
    double s = 0.0;
    for (int g = 0; g < mats.n_groups; ++g)
        s += mats.nu_sigf_mat(m, g, g_from);
    return s;
}

/// The prompt fission spectrum to use for material `m`, group `g`.
///
/// An explicit `chi_prompt` wins.  Otherwise it is *derived* from the total
/// spectrum so that prompt and delayed parts sum back to it:
///
///     chi_p = (chi - sum_i beta_i chi_d,i) / (1 - beta)
///
/// That keeps `chi_eff(dt -> infinity)` equal to the total spectrum for **any**
/// delayed spectrum, which is what makes a critical system a fixed point at any
/// step size.  When `chi_delayed` equals `chi` (the usual default) this reduces
/// to `chi_p = chi`, so the common case is unchanged.
inline double prompt_chi(const Materials& mats, const DelayedNeutronData& delayed,
                         int m, int g) {
    const int G = mats.n_groups;
    if (!delayed.chi_prompt.empty()) return delayed.chi_prompt[m * G + g];

    const double beta = delayed.beta_total(m);
    double delayed_part = 0.0;
    for (int i = 0; i < delayed.n_precursor; ++i)
        delayed_part += delayed.bet(m, i) * delayed.chi_d(m, i, g, G);
    return (mats.chi[m * G + g] - delayed_part) / (1.0 - beta);
}

/// Throw std::invalid_argument unless the delayed neutron data is consistent
/// with `mats` and physically sensible.  A default-constructed (empty) instance
/// always passes: it simply disables delayed neutrons.
inline void validate_delayed(const Materials& mats,
                             const DelayedNeutronData& delayed) {
    if (delayed.empty()) {
        // n_precursor == 0 must not carry stray data - that is a sizing mistake.
        if (!delayed.lambda.empty() || !delayed.beta.empty() ||
            !delayed.chi_delayed.empty())
            throw std::invalid_argument(
                "DelayedNeutronData.n_precursor is 0 but lambda/beta/chi_delayed "
                "are not empty; set n_precursor to enable delayed neutrons");
        return;
    }
    if (delayed.n_precursor < 0)
        throw std::invalid_argument(
            "DelayedNeutronData.n_precursor must not be negative");

    const int I = delayed.n_precursor;
    const int M = mats.n_mat;
    const int G = mats.n_groups;

    auto check = [](std::size_t actual, std::size_t expected, const char* name,
                    const char* layout) {
        if (actual != expected)
            throw std::invalid_argument(
                std::string("DelayedNeutronData.") + name + " must have " +
                layout + " elements (got " + std::to_string(actual) +
                ", expected " + std::to_string(expected) + ")");
    };
    check(delayed.lambda.size(), static_cast<std::size_t>(I),
          "lambda", "n_precursor");
    check(delayed.beta.size(), static_cast<std::size_t>(M) * I,
          "beta", "n_mat * n_precursor");
    check(delayed.chi_delayed.size(), static_cast<std::size_t>(M) * I * G,
          "chi_delayed", "n_mat * n_precursor * n_groups");
    if (!delayed.chi_prompt.empty())
        check(delayed.chi_prompt.size(), static_cast<std::size_t>(M) * G,
              "chi_prompt", "n_mat * n_groups (or empty to use Materials.chi)");

    for (int i = 0; i < I; ++i)
        if (!(delayed.lam(i) > 0.0))
            throw std::invalid_argument(
                "DelayedNeutronData.lambda entries must be positive "
                "(a zero decay constant is a precursor that never decays)");

    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < I; ++i)
            if (delayed.bet(m, i) < 0.0)
                throw std::invalid_argument(
                    "DelayedNeutronData.beta entries must not be negative");
        const double b = delayed.beta_total(m);
        if (b >= 1.0)
            throw std::invalid_argument(
                "DelayedNeutronData.beta sums to " + std::to_string(b) +
                " for material " + std::to_string(m) +
                "; the total delayed fraction must be below 1");
    }

    // A decaying precursor emits exactly one neutron, so each delayed spectrum
    // must be normalized.  This also catches the fission-matrix trap of
    // defaulting chi_delayed to Materials.chi, which is all zeros in that mode.
    for (int m = 0; m < M; ++m)
        for (int i = 0; i < I; ++i) {
            double s = 0.0;
            for (int g = 0; g < G; ++g) s += delayed.chi_d(m, i, g, G);
            if (std::fabs(s - 1.0) > 1e-6)
                throw std::invalid_argument(
                    "DelayedNeutronData.chi_delayed for material " +
                    std::to_string(m) + ", precursor group " + std::to_string(i) +
                    " sums to " + std::to_string(s) +
                    "; each delayed spectrum must sum to 1 (in fission-matrix "
                    "mode it cannot be defaulted from Materials.chi, which is "
                    "all zeros there - supply ChiDelayed explicitly)");
        }

    // In standard mode the prompt spectrum is derived by subtracting the
    // delayed part from the total; a delayed spectrum that overshoots the total
    // in some group would make it negative, which is unphysical and would show
    // up much later as a strange transient.  (Fission-matrix mode subtracts the
    // delayed part from the fission matrix instead - checked in
    // build_chi_effective, which has the matrix to hand.)
    if (!mats.use_fission_matrix() && delayed.chi_prompt.empty()) {
        for (int m = 0; m < M; ++m)
            for (int g = 0; g < G; ++g)
                if (prompt_chi(mats, delayed, m, g) < -1e-12)
                    throw std::invalid_argument(
                        "the delayed spectrum exceeds the total fission spectrum "
                        "in material " + std::to_string(m) + ", group " +
                        std::to_string(g) + ": the derived prompt spectrum "
                        "(chi - sum_i beta_i chi_d,i) / (1 - beta) is negative. "
                        "Supply chi_prompt explicitly if Materials.chi is meant "
                        "to be the prompt spectrum rather than the total");
    }
}

/// Build the dt-dependent effective fission operator, returned as a copy of
/// `mats`.  Handing the result to accumulate_fission() yields the implicit part
/// of the fission source.
///
/// In standard mode `chi` is replaced by
///
///     chi_eff,g = (1-beta) chi_p,g
///               + sum_i chi_d,i,g beta_i lambda_i dt / (1 + lambda_i dt).
///
/// In fission-matrix mode there is no separable spectrum, so the same weighting
/// is applied to the matrix itself.  The delayed part of `F` is `beta_i *
/// chi_d,i[g_to] * production_xs(g_from)`, so
///
///     F_eff[g][g'] = F[g][g'] - sum_i beta_i chi_d,i[g] P[g']
///                             + sum_i chi_d,i[g] beta_i lam_i dt/(1+lam_i dt) P[g']
///
/// where the first correction removes the delayed neutrons from the (total)
/// tabulated matrix and the second adds back the portion emitted within this
/// step.  The result is still matrix-shaped with `chi` all zeros, so it stays
/// in fission-matrix mode.  The two branches agree exactly when the matrix is
/// separable.
///
/// With no delayed data this is just `mats` unchanged, so callers need no
/// special case for prompt-only problems.
inline Materials build_chi_effective(const Materials& mats,
                                     const DelayedNeutronData& delayed,
                                     double dt) {
    Materials eff = mats;
    if (delayed.empty()) return eff;

    const int I = delayed.n_precursor;
    const int G = mats.n_groups;

    // Weight of precursor group i's neutrons emitted during a step of length dt.
    auto emitted = [&](int i) {
        const double lam = delayed.lam(i);
        return lam * dt / (1.0 + lam * dt);
    };

    if (!mats.use_fission_matrix()) {
        for (int m = 0; m < mats.n_mat; ++m) {
            const double beta = delayed.beta_total(m);
            for (int g = 0; g < G; ++g) {
                double c = (1.0 - beta) * prompt_chi(mats, delayed, m, g);
                for (int i = 0; i < I; ++i)
                    c += delayed.chi_d(m, i, g, G) * delayed.bet(m, i) * emitted(i);
                eff.chi[m * G + g] = c;
            }
        }
        return eff;
    }

    for (int m = 0; m < mats.n_mat; ++m) {
        for (int gf = 0; gf < G; ++gf) {
            const double prod = production_xs(mats, m, gf);
            for (int gt = 0; gt < G; ++gt) {
                double delayed_share = 0.0;   // removed from the total matrix
                double delayed_now   = 0.0;   // re-emitted within this step
                for (int i = 0; i < I; ++i) {
                    const double w = delayed.bet(m, i) *
                                     delayed.chi_d(m, i, gt, G) * prod;
                    delayed_share += w;
                    delayed_now   += w * emitted(i);
                }
                const double f = mats.nu_sigf_mat(m, gt, gf);
                if (f - delayed_share < -1e-12 * (std::fabs(f) + 1.0))
                    throw std::invalid_argument(
                        "the delayed neutron yield exceeds the tabulated fission "
                        "matrix in material " + std::to_string(m) + ", entry [" +
                        std::to_string(gt) + "][" + std::to_string(gf) +
                        "]: nusigf is treated as the *total* fission matrix, so "
                        "beta_i * chi_d,i must fit inside it");
                eff.nusigf[(m * G + gt) * G + gf] =
                    f - delayed_share + delayed_now;
            }
        }
    }
    return eff;
}

/// Fission neutron production rate `F[c] = sum_g' P_g' phi[g'*stride+c]`,
/// **per unit volume** - never area/volume weighted, because the precursor
/// balance is a pointwise ODE.  Output is indexed by cell, not by group.
///
/// `P` is `production_xs`, so this is correct in both fission representations.
inline void accumulate_production(const Materials& mats,
                                  const std::vector<int>& material_id,
                                  int groups, int cells, int stride,
                                  const std::vector<double>& phi,
                                  std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(cells), 0.0);

    // Cache the per-material production cross sections: in fission-matrix mode
    // each one is a column sum over groups, and this is called every step.
    std::vector<double> prod(static_cast<std::size_t>(mats.n_mat) * groups);
    for (int m = 0; m < mats.n_mat; ++m)
        for (int g = 0; g < groups; ++g)
            prod[m * groups + g] = production_xs(mats, m, g);

    for (int c = 0; c < cells; ++c) {
        const int mat = material_id[c];
        double f = 0.0;
        for (int gp = 0; gp < groups; ++gp)
            f += prod[mat * groups + gp] * phi[gp * stride + c];
        out[c] = f;
    }
}

/// Assemble the known delayed source `Q_d[g*stride+c]` in the solvers' internal
/// flux layout - everything in the theta-weighted delayed emission that does
/// *not* depend on the new flux.
///
/// That is the old precursors `C^n` plus, for theta < 1, the old production rate
/// `production_old` (`F^n`, from accumulate_production() applied to `phi^n`);
/// see the derivation above.  `production_old` is ignored when theta == 1 and
/// may be empty in that case.
///
/// Each cell is optionally scaled by `weight[c]` (cell area for the
/// volume-integrated FVM solver; `nullptr` for the per-unit-volume
/// finite-difference solvers).
inline void accumulate_delayed_source(const DelayedNeutronData& delayed,
                                      const std::vector<int>& material_id,
                                      int groups, int cells, int stride,
                                      double dt, double theta,
                                      const std::vector<double>* weight,
                                      const std::vector<double>& precursors,
                                      const std::vector<double>& production_old,
                                      std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(groups) * stride, 0.0);
    if (delayed.empty()) return;

    const int    I     = delayed.n_precursor;
    const double ex    = 1.0 - theta;   // explicit weight, 0 at backward Euler
    const double ratio = ex / theta;

    for (int c = 0; c < cells; ++c) {
        const int    mat   = material_id[c];
        const double w     = weight ? (*weight)[c] : 1.0;
        const double f_old = (ex > 0.0) ? production_old[c] : 0.0;
        for (int g = 0; g < groups; ++g) {
            double q = 0.0;
            for (int i = 0; i < I; ++i) {
                const double lam  = delayed.lam(i);
                const double C    = precursors[c * I + i];
                const double emit = delayed.chi_d(mat, i, g, groups) * lam;
                // Grouped so that theta == 1 leaves emit * C / (1 + lam*dt),
                // operation for operation: the numerator collapses to C * 1.0.
                q += emit * (C * (1.0 - ex * lam * dt) +
                             ex * dt * delayed.bet(mat, i) * f_old)
                     / (1.0 + theta * lam * dt);
                if (ex > 0.0) q += emit * ratio * C;
            }
            out[g * stride + c] = q * w;
        }
    }
}

/// Advance the precursors in place with the closed-form theta update, using the
/// production rate of the just-computed flux (`production`, i.e. `F^{n+1}`) and,
/// for theta < 1, that of the flux the step started from (`production_old`).
///
/// `production_old` is ignored when theta == 1 and may be empty in that case.
inline void update_precursors(const DelayedNeutronData& delayed,
                              const std::vector<int>& material_id,
                              int cells, double dt, double theta,
                              const std::vector<double>& production,
                              const std::vector<double>& production_old,
                              std::vector<double>& precursors) {
    if (delayed.empty()) return;
    const int    I  = delayed.n_precursor;
    const double ex = 1.0 - theta;
    for (int c = 0; c < cells; ++c) {
        const int    mat   = material_id[c];
        const double f_new = production[c];
        const double f_old = (ex > 0.0) ? production_old[c] : 0.0;
        // theta * f_new + 0.0 * f_old is exactly f_new, so theta == 1 reproduces
        // the backward-Euler update bit for bit.
        const double f_avg = theta * f_new + ex * f_old;
        for (int i = 0; i < I; ++i) {
            const double lam = delayed.lam(i);
            precursors[c * I + i] =
                (precursors[c * I + i] * (1.0 - ex * lam * dt) +
                 dt * delayed.bet(mat, i) * f_avg)
                / (1.0 + theta * lam * dt);
        }
    }
}

/// Equilibrium precursor concentrations `C_i = beta_i * F / lambda_i` for a
/// given production rate - the steady state of the precursor balance, and the
/// right default when a transient starts from a converged steady-state flux.
/// Seeding zeros instead would inject a large spurious prompt drop at t = 0.
inline void equilibrium_precursors(const DelayedNeutronData& delayed,
                                   const std::vector<int>& material_id,
                                   int cells,
                                   const std::vector<double>& production,
                                   std::vector<double>& out) {
    const int I = delayed.n_precursor;
    out.assign(static_cast<std::size_t>(cells) * (I > 0 ? I : 0), 0.0);
    if (delayed.empty()) return;
    for (int c = 0; c < cells; ++c) {
        const int mat = material_id[c];
        for (int i = 0; i < I; ++i)
            out[c * I + i] =
                delayed.bet(mat, i) * production[c] / delayed.lam(i);
    }
}

// ============================================================================
// Fixed-point acceleration for the implicit-fission inner iteration
// ============================================================================
//
// With an implicit fission source the inner sweep is the fixed point
//
//     phi <- (A + 1/(v dt))^-1 [ chi_eff F(phi) + S ],
//
// whose iteration matrix has spectral radius roughly
//
//     k_step = production / (loss + 1/(v dt)).
//
// The 1/(v dt) term is the only thing keeping that below 1, so a near-critical
// problem at a dt where 1/(v dt) << Sigma_r converges at nearly one digit per
// hundreds of sweeps.  The error is dominated by a single mode (the fundamental
// flux shape), which is exactly the situation Aitken's Delta^2 handles: estimate
// the convergence ratio from successive iterate changes and jump to the limit of
// the geometric series in one step.
//
//     sigma = ||d_m|| / ||d_{m-1}||,   phi <- phi + sigma/(1-sigma) * d_m
//
// Safeguards matter more than the extrapolation itself.  The ratio is only
// trusted after it has held steady for two iterations - measured relative to
// (1 - sigma), since near sigma = 1 a 5% change in sigma moves the extrapolation
// factor by orders of magnitude - and the jump is capped relative to the current
// flux magnitude.  A bad extrapolation is self-correcting: the residual is
// measured across the *sweep*, so an iterate thrown to the wrong place produces
// a large change on the next sweep rather than a false convergence.

/// Aitken (Delta^2) accelerator for the implicit-fission fixed point.
///
/// Usage: after each inner sweep, call accelerate() with the new iterate and
/// the previous one.  It may overwrite `phi` with an extrapolated iterate.
class FissionAccelerator {
public:
    /// @param enabled Set false to disable extrapolation entirely.
    explicit FissionAccelerator(bool enabled = true) : enabled_(enabled) {}

    /// Extrapolate along the current error mode when the convergence ratio has
    /// settled.  `phi` is the post-sweep iterate, `phi_prev` the pre-sweep one.
    ///
    /// @return True when `phi` was extrapolated.
    bool accelerate(std::vector<double>& phi,
                    const std::vector<double>& phi_prev) {
        if (!enabled_) return false;

        const int n = static_cast<int>(phi.size());
        double dn = 0.0;
        for (int i = 0; i < n; ++i) {
            const double d = phi[i] - phi_prev[i];
            dn += d * d;
        }
        dn = std::sqrt(dn);
        if (dn == 0.0) { reset(); return false; }

        const double prev = norm_prev_;
        norm_prev_ = dn;
        if (prev <= 0.0) return false;

        const double sigma = dn / prev;
        const double sigma_prev = sigma_prev_;
        sigma_prev_ = sigma;

        // Only a decaying, non-trivial mode is worth extrapolating: below
        // kSigmaMin the iteration is already fast, and at or above 1 the
        // geometric-series limit does not exist.
        if (!(sigma > kSigmaMin && sigma < kSigmaMax)) { stable_ = 0; return false; }

        // Stability is judged against the distance from 1, which is what sets
        // the extrapolation factor.
        if (sigma_prev > 0.0 &&
            std::fabs(sigma - sigma_prev) < kStableFrac * (1.0 - sigma))
            ++stable_;
        else
            stable_ = 0;

        if (cooldown_ > 0) { --cooldown_; return false; }
        if (stable_ < kStableNeeded) return false;

        const double factor = sigma / (1.0 - sigma);

        // Never jump further than kMaxJump times the current flux magnitude -
        // a mis-estimated ratio must not throw the iterate somewhere the sweep
        // cannot recover from.
        double phi_norm = 0.0;
        for (int i = 0; i < n; ++i) phi_norm += phi[i] * phi[i];
        phi_norm = std::sqrt(phi_norm);
        if (factor * dn > kMaxJump * phi_norm) { stable_ = 0; return false; }

        for (int i = 0; i < n; ++i)
            phi[i] += factor * (phi[i] - phi_prev[i]);

        // The change history describes the pre-extrapolation mode, so it is
        // meaningless now; start the estimate over and wait before trying again.
        reset();
        cooldown_ = kCooldown;
        return true;
    }

    /// Forget the convergence-ratio history (call between time steps).
    void reset() {
        norm_prev_  = -1.0;
        sigma_prev_ = -1.0;
        stable_     = 0;
    }

private:
    static constexpr double kSigmaMin    = 0.05;   ///< Below this, no help needed
    static constexpr double kSigmaMax    = 0.9999; ///< Above this, ratio untrustworthy
    static constexpr double kStableFrac  = 0.05;   ///< |dsigma| < 5% of (1-sigma)
    static constexpr int    kStableNeeded = 2;     ///< Consecutive stable estimates
    static constexpr int    kCooldown    = 2;      ///< Sweeps to wait after a jump
    static constexpr double kMaxJump     = 10.0;   ///< Max jump / ||phi||

    bool   enabled_;
    double norm_prev_  = -1.0;
    double sigma_prev_ = -1.0;
    int    stable_     = 0;
    int    cooldown_   = 0;
};

// ============================================================================
// Power iteration  A phi = (1/k) B phi
// ============================================================================

/// Result of a power-iteration solve, in internal `[g*stride+cell]` layout.
struct PowerResult {
    std::vector<double> phi;  ///< Converged flux, internal layout.
    double keff;              ///< Effective multiplication factor.
    int    iters;             ///< Power-iteration count.
    double change;            ///< Final L2 flux-change norm.
    bool   converged;         ///< True when both flux and keff criteria were met.
};

/// Generic power-iteration driver shared by every k-eigenvalue solver.
///
/// @param total      Length of the internal flux vector (`groups * stride`).
/// @param apply_B    Callable `(const vec& phi_in, vec& b_out)` - fission source.
/// @param solve_A    Callable `(const vec& b, vec& phi)` - in-place linear solve,
///                   warm-started from the current `phi`.
///
/// The initial guess is a flat unit-norm flux - deterministic on every
/// platform, and the iterates are renormalized each outer iteration so the
/// flux-change norm is scale-free.  Convergence requires both the flux change
/// and the eigenvalue change |dk| to fall below `epsilon` (the flux shape can
/// stall early when the dominance ratio is high, so keff gets its own check).
template <class ApplyB, class SolveA>
PowerResult power_iteration(int total, double epsilon, int max_outer,
                            bool verbose, ApplyB apply_B, SolveA solve_A) {
    std::vector<double> phi(total, 1.0 / std::sqrt(static_cast<double>(total)));

    std::vector<double> b(total);
    std::vector<double> phi_old;  // reused across iterations (no realloc)
    double keff   = 1.0;
    double change = 1.0;
    double dk     = 1.0;
    int    iter   = 0;

    while ((change > epsilon || dk > epsilon) && iter < max_outer) {
        check_interrupt();
        phi_old = phi;
        const double keff_old = keff;

        apply_B(phi_old, b);
        solve_A(b, phi);

        keff = norm2(phi);
        for (double& v : phi) v /= keff;

        change = l2_diff(phi, phi_old);
        dk     = std::abs(keff - keff_old);

        if (verbose)
            std::printf("Iter: %3d  keff: %.8f  change: %.2e  dk: %.2e\n",
                        iter + 1, keff, change, dk);
        ++iter;
    }

    const bool converged = (change <= epsilon && dk <= epsilon);
    return {std::move(phi), keff, iter, change, converged};
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

/// Emit a stderr warning when a backward-Euler time step's inner iteration hits
/// its cap.  With an implicit fission source the inner sweep must resolve the
/// multiplication as well as the scatter coupling; the 1/(v*dt) diagonal term
/// keeps that strongly diagonally dominant for small dt, but a near-critical
/// system at large dt converges slowly and silently wrong transients are worse
/// than slow ones.  Warned once per solver instance by the caller.
inline void warn_step_not_converged(const char* solver, int max_inner,
                                    double dt, double residual) {
    std::fprintf(stderr,
        "ndiffusion warning: %s time step (dt=%.3e) did not converge within "
        "max_inner=%d (relative change %.3e); the transient may be inaccurate. "
        "Reduce dt or increase max_inner - a near-critical system at large dt "
        "needs more inner iterations once fission is treated implicitly.\n",
        solver, dt, max_inner, residual);
}

// ============================================================================
// Input validation
// ============================================================================

/// Throw std::invalid_argument unless `dt` is a positive, finite time step.
/// The backward-Euler solvers add `1/(v*dt)` to the diagonal: dt == 0 makes the
/// flux NaN and dt < 0 runs time backwards.  Either poisons every later step,
/// since the flux is solver state.
inline void validate_dt(double dt) {
    if (!(dt > 0.0) || !std::isfinite(dt))
        throw std::invalid_argument(
            "dt must be a positive, finite time step (got " +
            std::to_string(dt) + "); the backward-Euler diagonal term "
            "1/(v*dt) is undefined otherwise");
}

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

/// Resolve the BC for a low-coordinate edge (1-D left, 2-D left / bottom).
/// Empty means reflective, which is what those edges were before they became
/// configurable.  Otherwise there must be one entry per group.  `on_axis` marks
/// an edge sitting on r = 0 of a cylinder or sphere: its face area is zero, so
/// any other condition would be silently ignored rather than applied - reject it.
inline std::vector<BoundaryCondition> low_edge_bc(
    std::vector<BoundaryCondition> bc, int groups, bool on_axis,
    const char* name) {
    if (bc.empty()) return std::vector<BoundaryCondition>(groups, {0.0, 1.0});
    if (static_cast<int>(bc.size()) != groups)
        throw std::invalid_argument(
            std::string(name) + " must be empty (reflective) or have one entry "
            "per energy group");
    if (on_axis)
        for (const BoundaryCondition& b : bc)
            if (b.A != 0.0)
                throw std::invalid_argument(
                    std::string(name) + " must be reflective (A = 0) on the "
                    "r = 0 axis, where the face area vanishes; start the mesh "
                    "at r > 0 for an inner surface");
    return bc;
}

/// Ratio phi_ghost / phi_cell for a Robin ghost node one cell width `h` outside
/// the edge, from A*(phi_g + phi_c)/2 + B*(phi_g - phi_c)/h = 0.
inline double robin_ghost_ratio(const BoundaryCondition& bc, double h) {
    const double num   = 0.5 * bc.A - bc.B / h;
    const double denom = 0.5 * bc.A + bc.B / h;
    return (std::abs(denom) > 1e-30) ? (-num / denom) : 1.0;
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

/// Throw std::invalid_argument unless every Materials array has the size its
/// layout convention requires.  Without this, a mis-sized array is read out of
/// bounds (undefined behavior) and results are silently wrong.  `velocity` is
/// not checked here - only the time-dependent solvers need it, and they
/// validate it themselves.
inline void validate_materials(const Materials& mats) {
    if (mats.n_mat < 1)
        throw std::invalid_argument("Materials.n_mat must be at least 1");
    if (mats.n_groups < 1)
        throw std::invalid_argument("Materials.n_groups must be at least 1");

    const std::size_t vec_size =
        static_cast<std::size_t>(mats.n_mat) * mats.n_groups;
    const std::size_t mat_size = vec_size * mats.n_groups;

    auto check = [](std::size_t actual, std::size_t expected, const char* name,
                    const char* layout) {
        if (actual != expected)
            throw std::invalid_argument(
                std::string("Materials.") + name + " must have " + layout +
                " elements (got " + std::to_string(actual) +
                ", expected " + std::to_string(expected) + ")");
    };
    check(mats.D.size(),       vec_size, "D",       "n_mat * n_groups");
    check(mats.removal.size(), vec_size, "removal", "n_mat * n_groups");
    check(mats.chi.size(),     vec_size, "chi",     "n_mat * n_groups");
    check(mats.scatter.size(), mat_size, "scatter",
          "n_mat * n_groups * n_groups");

    if (mats.nusigf.size() != vec_size && mats.nusigf.size() != mat_size)
        throw std::invalid_argument(
            "Materials.nusigf must have n_mat * n_groups elements (standard "
            "chi * nusigf mode) or n_mat * n_groups * n_groups elements "
            "(fission transfer matrix mode; requires chi all zeros), got " +
            std::to_string(mats.nusigf.size()));

    // A matrix-sized nusigf with a non-zero chi is almost certainly a mistake:
    // standard mode would read the first n_mat*n_groups entries as a vector.
    if (mats.n_groups > 1 && mats.nusigf.size() == mat_size &&
        !mats.use_fission_matrix())
        throw std::invalid_argument(
            "Materials.nusigf is sized as a fission transfer matrix "
            "(n_mat * n_groups * n_groups) but chi is not all zeros; "
            "fission-matrix mode requires chi == 0 everywhere");
}

}  // namespace detail
}  // namespace ndiffusion
