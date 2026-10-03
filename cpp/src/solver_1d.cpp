#include <ndiffusion/solver_1d.hpp>
#include <ndiffusion/solver_detail.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <stdexcept>

using namespace ndiffusion::detail;

// ============================================================================
// File-local helpers
// ============================================================================

namespace {

constexpr double PI = 3.14159265358979323846;

// Compute surface areas and cell volumes for the chosen geometry.
void compute_geometry(
    Geometry                    geom,
    const std::vector<double>&  edges,
    std::vector<double>&        sa,
    std::vector<double>&        vol
) {
    const int cells = static_cast<int>(edges.size()) - 1;
    sa.resize(cells + 1);
    vol.resize(cells);

    if (geom == Geometry::Slab) {
        std::fill(sa.begin(), sa.end(), 1.0);
        for (int i = 0; i < cells; ++i)
            vol[i] = edges[i + 1] - edges[i];

    } else if (geom == Geometry::Cylinder) {
        for (int i = 0; i <= cells; ++i)
            sa[i] = 2.0 * PI * edges[i];
        for (int i = 0; i < cells; ++i)
            vol[i] = PI * (edges[i + 1] * edges[i + 1] - edges[i] * edges[i]);

    } else {  // Sphere
        for (int i = 0; i <= cells; ++i)
            sa[i] = 4.0 * PI * edges[i] * edges[i];
        for (int i = 0; i < cells; ++i)
            vol[i] = (4.0 / 3.0) * PI *
                     (std::pow(edges[i + 1], 3) - std::pow(edges[i], 3));
    }
}

// A cylinder or sphere whose first edge is r = 0: the left face has no area.
bool starts_on_axis(Geometry geom, const std::vector<double>& edges) {
    return geom != Geometry::Slab && !edges.empty() && edges[0] <= 0.0;
}

// Build per-group tridiagonal bands from geometry and cross sections.
//
// For physical cell i and energy group g the finite-difference equation is:
//
//   - c_left                      * phi[i-1]
//   + ( c_right + c_left + sig_r[mat,g] ) * phi[i]
//   - c_right                     * phi[i+1]
//   - sum_{gp!=g} sig_s[mat,g,gp] * phi_gp[i]
//   = b[g][i]
//
//   c_right = D_harm(D_i, D_{i+1}) * SA[i+1] / (h_int_right * V[i])
//   c_left  = D_harm(D_i, D_{i-1}) * SA[i]   / (h_int_left  * V[i])
//
// where D_harm(a,b) = 2ab/(a+b) is the harmonic mean at the interface and h_int
// is the center-to-center distance across it, 0.5*(h_i + h_neighbor) - not the
// local cell width, which would leave the scheme inconsistent and
// non-conservative on a non-uniform mesh.  Matches build_coefficients_2d.
//
// Scatter coupling to other groups is handled by Gauss-Seidel and does not
// appear in the bands.
//
// The last row (i = cells) encodes the Robin boundary condition:
//   (0.5*A_bc + B_bc/dx_last) * phi[cells]
//   + (0.5*A_bc - B_bc/dx_last) * phi[cells-1] = 0
//
// The left edge uses the same ghost construction, but the ghost is eliminated
// into the row-0 diagonal (phi_ghost = alpha * phi[0]) so the bands keep their
// single trailing ghost row.  Reflective gives alpha = 1 and adds nothing.
void build_tridiagonals(
    const Materials&                     mats,
    const std::vector<int>&              medium_map,
    const std::vector<double>&           edges_x,
    const std::vector<double>&           surface_area,
    const std::vector<double>&           volume,
    const std::vector<BoundaryCondition>& bc,
    const std::vector<BoundaryCondition>& bc_left,
    int cells, int groups, int N,
    std::vector<double>& lower,
    std::vector<double>& diag,
    std::vector<double>& upper
) {
    lower.assign(groups * N, 0.0);
    diag .assign(groups * N, 0.0);
    upper.assign(groups * N, 0.0);

    for (int g = 0; g < groups; ++g) {
        for (int i = 0; i < cells; ++i) {
            const int    idx  = g * N + i;
            const double dx   = edges_x[i + 1] - edges_x[i];
            const int    mat  = medium_map[i];

            // Right-interface: harmonic-mean D over the center-to-center distance.
            // At the outer edge the ghost node sits one dx beyond the last
            // center, so mirroring dx there reproduces the Robin ghost spacing.
            const int    mat_r   = (i < cells - 1) ? medium_map[i + 1] : mat;
            const double dx_r    = (i < cells - 1) ? (edges_x[i + 2] - edges_x[i + 1]) : dx;
            const double D_i     = mats.d(mat,   g);
            const double D_r     = mats.d(mat_r, g);
            const double D_right = 2.0 * D_i * D_r / (D_i + D_r);
            const double coef_r  = D_right * surface_area[i + 1]
                                 / (0.5 * (dx + dx_r) * volume[i]);

            diag [idx] = coef_r + mats.sig_r(mat, g);
            upper[idx] = -coef_r;

            // Left-interface
            if (i > 0) {
                const int    mat_l  = medium_map[i - 1];
                const double dx_l   = edges_x[i] - edges_x[i - 1];
                const double D_l    = mats.d(mat_l, g);
                const double D_left = 2.0 * D_i * D_l / (D_i + D_l);
                const double coef_l = D_left * surface_area[i]
                                    / (0.5 * (dx_l + dx) * volume[i]);
                diag [idx] += coef_l;
                lower[idx]  = -coef_l;
            } else {
                const double alpha = robin_ghost_ratio(bc_left[g], dx);
                diag[idx] += D_i * surface_area[0] / (dx * volume[0]) *
                             (1.0 - alpha);
            }
        }

        // Boundary-condition ghost row
        const int    idx_bc  = g * N + cells;
        const double dx_last = edges_x[cells] - edges_x[cells - 1];
        diag [idx_bc] = 0.5 * bc[g].A + bc[g].B / dx_last;
        lower[idx_bc] = 0.5 * bc[g].A - bc[g].B / dx_last;
        // upper[idx_bc] = 0  (already zero-initialized)
    }
}

}  // namespace

// ============================================================================
// KEigenSolver - constructor
// ============================================================================

KEigenSolver::KEigenSolver(
    Materials                      mats,
    std::vector<int>               medium_map,
    std::vector<double>            edges_x,
    Geometry                       geom,
    std::vector<BoundaryCondition> bc,
    double epsilon,
    int    max_outer,
    int    max_inner,
    bool   verbose,
    std::vector<BoundaryCondition> bc_left
):
      mats_      (std::move(mats)),
      medium_map_(std::move(medium_map)),
      edges_x_   (std::move(edges_x)),
      geom_      (geom),
      bc_        (std::move(bc)),
      bc_left_   (std::move(bc_left)),
      epsilon_   (epsilon),
      max_outer_ (max_outer),
      max_inner_ (max_inner),
      verbose_   (verbose),
      cells_     (static_cast<int>(medium_map_.size())),
      groups_    (mats_.n_groups),
      N_         (cells_ + 1)
{
    if (static_cast<int>(bc_.size()) != groups_)
        throw std::invalid_argument("bc must have one entry per energy group");
    bc_left_ = low_edge_bc(std::move(bc_left_), groups_,
                           starts_on_axis(geom_, edges_x_), "bc_left");

    if (cells_ < 1)
        throw std::invalid_argument("medium_map must have at least one cell");
    if (static_cast<int>(edges_x_.size()) != cells_ + 1)
        throw std::invalid_argument("edges_x must have cells + 1 entries");
    validate_materials(mats_);
    validate_increasing(edges_x_, "edges_x");
    validate_material_ids(medium_map_, mats_.n_mat, "medium_map");

    compute_geometry(geom_, edges_x_, surface_area_, volume_);
    build_tridiagonals(mats_, medium_map_, edges_x_,
                       surface_area_, volume_, bc_, bc_left_,
                       cells_, groups_, N_,
                       lower_, diag_, upper_);
}

// ============================================================================
// KEigenSolver - fission source operator  b = B * phi
// ============================================================================

void KEigenSolver::apply_B(
    const std::vector<double>& phi,
          std::vector<double>& b
) const {
    // Ghost BC row (index cells_ in each group) stays zero - accumulate_fission
    // only writes the first cells_ entries per group.
    accumulate_fission(mats_, medium_map_, groups_, cells_, N_,
                       /*weight=*/nullptr, phi, b);
}

// ============================================================================
// KEigenSolver - matrix-free linear solve  A * phi = b
// ============================================================================

bool KEigenSolver::solve_A(
    const std::vector<double>& b,
          std::vector<double>& phi
) const {
    std::vector<double> lower_g(N_), diag_g(N_), upper_g(N_);
    std::vector<double> rhs(N_), phi_g(N_), tw_c, tw_d, phi_prev;

    for (int inner = 0; inner < max_inner_; ++inner) {
        phi_prev = phi;

        for (int g = 0; g < groups_; ++g) {
            for (int i = 0; i < cells_; ++i) {
                const int mat = medium_map_[i];
                rhs[i] = b[g * N_ + i];
                for (int gp = 0; gp < groups_; ++gp) {
                    if (gp != g)
                        rhs[i] += mats_.sig_s(mat, g, gp) * phi[gp * N_ + i];
                }
            }
            rhs[cells_] = 0.0;

            for (int i = 0; i < N_; ++i) {
                lower_g[i] = lower_[g * N_ + i];
                diag_g [i] = diag_ [g * N_ + i];
                upper_g[i] = upper_[g * N_ + i];
            }

            thomas(lower_g, diag_g, upper_g, rhs, phi_g, N_, tw_c, tw_d);

            for (int i = 0; i < N_; ++i)
                phi[g * N_ + i] = phi_g[i];
        }

        if (rel_l2_diff(phi, phi_prev) < epsilon_ * 1e-3)
            return true;
    }
    return false;
}

// ============================================================================
// KEigenSolver - power iteration
// ============================================================================

DiffusionResult KEigenSolver::solve() {
    bool inner_ok = true;
    PowerResult pr = power_iteration(
        groups_ * N_, epsilon_, max_outer_, verbose_,
        [this](const std::vector<double>& in, std::vector<double>& out) {
            apply_B(in, out);
        },
        [this, &inner_ok](const std::vector<double>& rhs, std::vector<double>& x) {
            if (!solve_A(rhs, x)) inner_ok = false;
        });

    if (!inner_ok)
        warn_inner_not_converged("KEigenSolver", max_inner_);

    std::vector<double> flux_out;
    pack_flux(pr.phi, cells_, groups_, N_, flux_out);
    return {flux_out, pr.keff, pr.iters, pr.change, pr.converged && inner_ok, groups_};
}

// ============================================================================
// FixedSourceSolver - constructor
// ============================================================================

FixedSourceSolver::FixedSourceSolver(
    Materials                      mats,
    std::vector<int>               medium_map,
    std::vector<double>            edges_x,
    Geometry                       geom,
    std::vector<BoundaryCondition> bc,
    double epsilon,
    int    max_inner,
    bool   verbose,
    std::vector<BoundaryCondition> bc_left
):
      mats_      (std::move(mats)),
      medium_map_(std::move(medium_map)),
      edges_x_   (std::move(edges_x)),
      geom_      (geom),
      bc_        (std::move(bc)),
      bc_left_   (std::move(bc_left)),
      epsilon_   (epsilon),
      max_inner_ (max_inner),
      verbose_   (verbose),
      cells_     (static_cast<int>(medium_map_.size())),
      groups_    (mats_.n_groups),
      N_         (cells_ + 1)
{
    if (static_cast<int>(bc_.size()) != groups_)
        throw std::invalid_argument("bc must have one entry per energy group");
    bc_left_ = low_edge_bc(std::move(bc_left_), groups_,
                           starts_on_axis(geom_, edges_x_), "bc_left");

    if (cells_ < 1)
        throw std::invalid_argument("medium_map must have at least one cell");
    if (static_cast<int>(edges_x_.size()) != cells_ + 1)
        throw std::invalid_argument("edges_x must have cells + 1 entries");
    validate_materials(mats_);
    validate_increasing(edges_x_, "edges_x");
    validate_material_ids(medium_map_, mats_.n_mat, "medium_map");

    compute_geometry(geom_, edges_x_, surface_area_, volume_);
    build_tridiagonals(mats_, medium_map_, edges_x_,
                       surface_area_, volume_, bc_, bc_left_,
                       cells_, groups_, N_,
                       lower_, diag_, upper_);
}

// ============================================================================
// FixedSourceSolver - solve  A*phi = q
// ============================================================================

FixedSourceResult FixedSourceSolver::solve(const std::vector<double>& source) const {
    if (static_cast<int>(source.size()) != cells_ * groups_)
        throw std::invalid_argument("source must have cells * n_groups elements");

    // Convert source from [cells * groups] to internal [groups * N]
    std::vector<double> src;
    unpack_flux(source, cells_, groups_, N_, /*weight=*/nullptr, src);

    std::vector<double> phi(groups_ * N_, 0.0);
    std::vector<double> lower_g(N_), diag_g(N_), upper_g(N_), rhs(N_), phi_g(N_);
    std::vector<double> tw_c, tw_d, phi_prev;

    double residual = 1.0;
    int    iter     = 0;   // sweeps performed

    while (iter < max_inner_) {
        check_interrupt();
        ++iter;
        phi_prev = phi;

        for (int g = 0; g < groups_; ++g) {
            for (int i = 0; i < cells_; ++i) {
                const int mat = medium_map_[i];
                rhs[i] = src[g * N_ + i];
                for (int gp = 0; gp < groups_; ++gp) {
                    if (gp != g)
                        rhs[i] += mats_.sig_s(mat, g, gp) * phi[gp * N_ + i];
                }
            }
            rhs[cells_] = 0.0;

            for (int i = 0; i < N_; ++i) {
                lower_g[i] = lower_[g * N_ + i];
                diag_g [i] = diag_ [g * N_ + i];
                upper_g[i] = upper_[g * N_ + i];
            }

            thomas(lower_g, diag_g, upper_g, rhs, phi_g, N_, tw_c, tw_d);

            for (int i = 0; i < N_; ++i)
                phi[g * N_ + i] = phi_g[i];
        }

        residual = rel_l2_diff(phi, phi_prev);

        if (verbose_)
            std::printf("Iter: %3d  residual: %.2e\n", iter, residual);

        if (residual < epsilon_)
            break;
    }

    std::vector<double> flux_out;
    pack_flux(phi, cells_, groups_, N_, flux_out);
    return {flux_out, iter, residual, residual < epsilon_, groups_};
}

// ============================================================================
// TimeDependentSolver - constructor
// ============================================================================

TimeDependentSolver::TimeDependentSolver(
    Materials                      mats,
    std::vector<int>               medium_map,
    std::vector<double>            edges_x,
    Geometry                       geom,
    std::vector<BoundaryCondition> bc,
    std::vector<double>            initial_flux,
    double epsilon,
    int    max_inner,
    bool   verbose,
    DelayedNeutronData             delayed,
    std::vector<double>            initial_precursors,
    double theta,
    std::vector<BoundaryCondition> bc_left
):
      mats_      (std::move(mats)),
      medium_map_(std::move(medium_map)),
      edges_x_   (std::move(edges_x)),
      geom_      (geom),
      bc_        (std::move(bc)),
      bc_left_   (std::move(bc_left)),
      epsilon_   (epsilon),
      max_inner_ (max_inner),
      verbose_   (verbose),
      delayed_   (std::move(delayed)),
      theta_     (theta),
      cells_     (static_cast<int>(medium_map_.size())),
      groups_    (mats_.n_groups),
      N_         (cells_ + 1),
      chi_eff_dt_   (-1.0),
      chi_eff_theta_(-1.0),
      warned_    (false),
      time_      (0.0),
      steps_     (0)
{
    if (static_cast<int>(bc_.size()) != groups_)
        throw std::invalid_argument("bc must have one entry per energy group");
    bc_left_ = low_edge_bc(std::move(bc_left_), groups_,
                           starts_on_axis(geom_, edges_x_), "bc_left");
    if (static_cast<int>(mats_.velocity.size()) != groups_)
        throw std::invalid_argument(
            "Materials.velocity must have one entry per energy group");

    if (cells_ < 1)
        throw std::invalid_argument("medium_map must have at least one cell");
    if (static_cast<int>(edges_x_.size()) != cells_ + 1)
        throw std::invalid_argument("edges_x must have cells + 1 entries");
    validate_materials(mats_);
    validate_delayed(mats_, delayed_);
    validate_increasing(edges_x_, "edges_x");
    validate_material_ids(medium_map_, mats_.n_mat, "medium_map");
    validate_theta(theta_);

    compute_geometry(geom_, edges_x_, surface_area_, volume_);
    build_tridiagonals(mats_, medium_map_, edges_x_,
                       surface_area_, volume_, bc_, bc_left_,
                       cells_, groups_, N_,
                       lower_base_, diag_base_, upper_base_);
    // chi_eff at dt = 0 is the prompt spectrum (1-beta) chi_p.
    prompt_mats_ = build_chi_effective(mats_, delayed_, 0.0);

    // Convert initial_flux from [cells * groups] to internal [groups * N]
    phi_.assign(groups_ * N_, 0.0);
    if (!initial_flux.empty()) {
        if (static_cast<int>(initial_flux.size()) != cells_ * groups_)
            throw std::invalid_argument(
                "initial_flux must have cells * n_groups elements");
        unpack_flux(initial_flux, cells_, groups_, N_, /*weight=*/nullptr, phi_);

        // unpack_flux leaves the ghost node zero, but it is a *constrained*
        // value, not a free one: the BC row demands
        // lower*phi[cells-1] + diag*phi[cells] = 0.  Every later step gets it
        // right because the Thomas sweep solves that row, and backward Euler
        // never reads it - but the theta method's explicit term does, so an
        // inconsistent t = 0 ghost would inject a spurious surface leakage into
        // the very first step.  Seed it here so the initial state is a genuine
        // solution of the boundary condition.
        for (int g = 0; g < groups_; ++g) {
            const int    idx_bc = g * N_ + cells_;
            const double d      = diag_base_[idx_bc];
            phi_[idx_bc] = (std::fabs(d) > 1e-30)
                           ? -lower_base_[idx_bc] * phi_[idx_bc - 1] / d
                           : phi_[idx_bc - 1];
        }
    }

    init_precursors(initial_precursors);
}

// ============================================================================
// TimeDependentSolver - kinetics state helpers
// ============================================================================

void TimeDependentSolver::init_precursors(
    const std::vector<double>& initial_precursors
) {
    const int I = delayed_.n_precursor;
    if (!initial_precursors.empty()) {
        if (static_cast<int>(initial_precursors.size()) != cells_ * I)
            throw std::invalid_argument(
                "initial_precursors must have cells * n_precursor elements");
        precursors_ = initial_precursors;
        return;
    }
    // Default: equilibrium with the initial flux.  Zeros would inject a
    // spurious prompt drop when starting from a steady state.
    std::vector<double> production;
    accumulate_production(mats_, medium_map_, groups_, cells_, N_,
                          phi_, production);
    equilibrium_precursors(delayed_, medium_map_, cells_, production,
                           precursors_);
}

void TimeDependentSolver::refresh_chi_effective(double dt) {
    if (dt == chi_eff_dt_ && theta_ == chi_eff_theta_) return;
    // The delayed neutrons emitted within the step are weighted by theta*dt, not
    // dt - see the derivation in solver_detail.hpp.
    chi_eff_mats_  = build_chi_effective(mats_, delayed_, theta_ * dt);
    chi_eff_dt_    = dt;
    chi_eff_theta_ = theta_;
}

void TimeDependentSolver::set_theta(double theta) {
    validate_theta(theta);
    theta_ = theta;
}

void TimeDependentSolver::update_materials(Materials mats) {
    if (mats.n_mat != mats_.n_mat || mats.n_groups != mats_.n_groups)
        throw std::invalid_argument(
            "update_materials must not change n_mat or n_groups; the mesh and "
            "material layout are fixed at construction");
    if (static_cast<int>(mats.velocity.size()) != groups_)
        throw std::invalid_argument(
            "Materials.velocity must have one entry per energy group");
    validate_materials(mats);
    validate_delayed(mats, delayed_);

    mats_ = std::move(mats);
    build_tridiagonals(mats_, medium_map_, edges_x_,
                       surface_area_, volume_, bc_, bc_left_,
                       cells_, groups_, N_,
                       lower_base_, diag_base_, upper_base_);
    prompt_mats_ = build_chi_effective(mats_, delayed_, 0.0);
    chi_eff_dt_  = -1.0;  // invalidate the cached effective spectrum
}

// ============================================================================
// TimeDependentSolver - explicit residual for the theta method
// ============================================================================
//
// E = -A phi_old + in-scatter(phi_old) + prompt fission(phi_old), i.e. the whole
// flux-driven right-hand side at the old time level.  Only the interior rows
// carry it: row i = cells_ is the Robin boundary *constraint*, not a balance
// equation, and it is imposed at the new time level with rhs = 0 as always.
//
// The ghost node lives in phi_ at index cells_, so applying the base bands over
// i = 0 .. cells_-1 needs no special case at either end - the lower band is
// zero at i = 0 (the left BC is folded into the diagonal) and phi_old[cells_]
// is a stored value.

void TimeDependentSolver::explicit_residual(const std::vector<double>& phi_old,
                                            std::vector<double>& out) const {
    std::vector<double> fis_prompt;
    accumulate_fission(prompt_mats_, medium_map_, groups_, cells_, N_,
                       /*weight=*/nullptr, phi_old, fis_prompt);

    out.assign(static_cast<std::size_t>(groups_) * N_, 0.0);
    for (int g = 0; g < groups_; ++g) {
        for (int i = 0; i < cells_; ++i) {
            const int    idx = g * N_ + i;
            const int    mat = medium_map_[i];
            double e = -(diag_base_[idx] * phi_old[idx] +
                         upper_base_[idx] * phi_old[idx + 1]);
            if (i > 0) e -= lower_base_[idx] * phi_old[idx - 1];
            e += fis_prompt[idx];
            for (int gp = 0; gp < groups_; ++gp)
                if (gp != g)
                    e += mats_.sig_s(mat, g, gp) * phi_old[gp * N_ + i];
            out[idx] = e;
        }
        // out[g * N_ + cells_] stays 0: the BC row has no source.
    }
}

// ============================================================================
// TimeDependentSolver - single theta-weighted time step
// ============================================================================
//
// The time-discretized equation for group g at cell i is:
//
//   [A_g + 1/(theta*v_g*dt) I] phi_g^{n+1}
//     = 1/(theta*v_g*dt) phi_g^n
//       + chi_eff,g * sum_gp( nu_sigf_gp * phi_gp^{n+1} )  [fission, implicit]
//       + Q_d,g                                            [delayed: C^n, F^n]
//       + sum_{gp!=g} sig_s(g<-gp) * phi_gp^{n+1}          [scatter, implicit GS]
//       + ((1-theta)/theta) * E_g                          [explicit residual]
//
// chi_eff (built at theta*dt) and Q_d come from integrating the precursor
// balance in closed form; with no delayed data chi_eff is just chi and Q_d is
// zero, leaving prompt-only kinetics.  See solver_detail.hpp for the derivation
// and for why theta = 1 recovers the old backward-Euler arithmetic exactly.
// Because fission is implicit, it is reassembled from the latest iterate inside
// the Gauss-Seidel loop, exactly like the cross-group scatter term.
//
// The 1/(theta*v_g*dt) term is added to the spatial diagonal at the start of
// each step; the base tridiagonals are left unchanged for reuse.

void TimeDependentSolver::step(double dt) {
    validate_dt(dt);

    const std::vector<double> phi_old = phi_;

    refresh_chi_effective(dt);

    // Production rate of the old flux; needed by the theta-weighted delayed
    // terms, and reused for the precursor advance at the end of the step.
    std::vector<double> production_old;
    if (theta_ < 1.0 && !delayed_.empty())
        accumulate_production(mats_, medium_map_, groups_, cells_, N_,
                              phi_old, production_old);

    // Delayed source from the old kinetics state - constant over the step.
    std::vector<double> qd;
    accumulate_delayed_source(delayed_, medium_map_, groups_, cells_, N_,
                              dt, theta_, /*weight=*/nullptr, precursors_,
                              production_old, qd);

    // Explicit half of the theta weighting.  Skipped entirely at theta = 1.
    std::vector<double> expl;
    const double ex_weight = (1.0 - theta_) / theta_;
    if (theta_ < 1.0) explicit_residual(phi_old, expl);

    // Gauss-Seidel inner iteration
    std::vector<double> lower_g(N_), diag_g(N_), upper_g(N_), rhs(N_), phi_g(N_);
    std::vector<double> tw_c, tw_d, phi_iter, fis;

    double residual  = 0.0;
    bool   converged = false;
    FissionAccelerator accel;

    for (int inner = 0; inner < max_inner_; ++inner) {
        phi_iter = phi_;

        // Implicit fission source from the latest iterate.
        accumulate_fission(chi_eff_mats_, medium_map_, groups_, cells_, N_,
                           /*weight=*/nullptr, phi_, fis);

        for (int g = 0; g < groups_; ++g) {
            const double inv_v_dt = 1.0 / (mats_.v(g) * theta_ * dt);

            // Build RHS for this group
            for (int i = 0; i < cells_; ++i) {
                const int mat = medium_map_[i];
                rhs[i] = inv_v_dt * phi_old[g * N_ + i]  // time-source
                        + fis[g * N_ + i]                 // fission (implicit)
                        + qd [g * N_ + i];                // delayed (C^n, F^n)
                if (theta_ < 1.0)
                    rhs[i] += ex_weight * expl[g * N_ + i];   // explicit residual
                // In-scatter from other groups (latest iterate)
                for (int gp = 0; gp < groups_; ++gp) {
                    if (gp != g)
                        rhs[i] += mats_.sig_s(mat, g, gp) * phi_[gp * N_ + i];
                }
            }
            rhs[cells_] = 0.0;  // ghost BC row: no source

            // Build per-group tridiagonal with time-absorption added to diagonal
            for (int i = 0; i < N_; ++i) {
                lower_g[i] = lower_base_[g * N_ + i];
                diag_g [i] = diag_base_ [g * N_ + i];
                upper_g[i] = upper_base_[g * N_ + i];
            }
            for (int i = 0; i < cells_; ++i)
                diag_g[i] += inv_v_dt;

            thomas(lower_g, diag_g, upper_g, rhs, phi_g, N_, tw_c, tw_d);

            for (int i = 0; i < N_; ++i)
                phi_[g * N_ + i] = phi_g[i];
        }

        // Check inner convergence (relative - physical flux can be large)
        residual = rel_l2_diff(phi_, phi_iter);
        if (residual < epsilon_) { converged = true; break; }

        accel.accelerate(phi_, phi_iter);
    }

    if (!converged && !warned_) {
        warn_step_not_converged("TimeDependentSolver", max_inner_, dt, residual);
        warned_ = true;
    }

    // Advance the precursors with the production rate of the new flux.
    if (!delayed_.empty()) {
        std::vector<double> production;
        accumulate_production(mats_, medium_map_, groups_, cells_, N_,
                              phi_, production);
        update_precursors(delayed_, medium_map_, cells_, dt, theta_,
                          production, production_old, precursors_);
    }

    time_  += dt;
    steps_ += 1;

    if (verbose_)
        std::printf("t = %.6e s  step %d  phi_max = %.6e\n",
                    time_, steps_,
                    *std::max_element(phi_.begin(), phi_.end()));
}

// ============================================================================
// TimeDependentSolver - run multiple steps
// ============================================================================

TimeDependentResult TimeDependentSolver::run(double dt, int n_steps) {
    for (int n = 0; n < n_steps; ++n) {
        check_interrupt();
        step(dt);
    }
    return result();
}

// ============================================================================
// TimeDependentSolver - extract current state
// ============================================================================

TimeDependentResult TimeDependentSolver::result() const {
    std::vector<double> flux_out;
    pack_flux(phi_, cells_, groups_, N_, flux_out);
    return {flux_out, time_, steps_, precursors_, groups_,
            delayed_.n_precursor};
}
