#include <ndiffusion/solver_1d.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <stdexcept>

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

// Thomas / TDMA tridiagonal solver.
// Solves the n-equation system with lower, diag, upper bands and rhs.
// lower[0] and upper[n-1] are not referenced.
void thomas(
    const std::vector<double>& lower,
    const std::vector<double>& diag,
    const std::vector<double>& upper,
    const std::vector<double>& rhs,
          std::vector<double>& x,
    int n
) {
    std::vector<double> c(n), d(n);

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

// Build per-group tridiagonal bands from geometry and cross sections.
//
// For physical cell i and energy group g the finite-difference equation is:
//
//   - D_left/(dx*V[i]) * SA[i]   * phi[i-1]
//   + ( D_right/(dx*V[i]) * SA[i+1]
//     + D_left /(dx*V[i]) * SA[i]
//     + sig_r[mat,g] ) * phi[i]
//   - D_right/(dx*V[i]) * SA[i+1] * phi[i+1]
//   - sum_{gp!=g} sig_s[mat,g,gp] * phi_gp[i]
//   = b[g][i]
//
// where D_left/right are harmonic-mean diffusion coefficients at the
// cell interfaces.  Scatter coupling to other groups is handled by
// Gauss-Seidel and does not appear in the bands.
//
// The last row (i = cells) encodes the Robin boundary condition:
//   (0.5*A_bc + B_bc/dx_last) * phi[cells]
//   + (0.5*A_bc - B_bc/dx_last) * phi[cells-1] = 0
void build_tridiagonals(
    const Materials&                     mats,
    const std::vector<int>&              medium_map,
    const std::vector<double>&           edges_x,
    const std::vector<double>&           surface_area,
    const std::vector<double>&           volume,
    const std::vector<BoundaryCondition>& bc,
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

            // Right-interface: half harmonic-mean D, combined with 2/(dx*V)*SA
            const int    mat_r   = (i < cells - 1) ? medium_map[i + 1] : mat;
            const double D_i     = mats.d(mat,   g);
            const double D_r     = mats.d(mat_r, g);
            const double D_right = D_i * D_r / (D_i + D_r);
            const double coef_r  = 2.0 / (dx * volume[i]) * D_right * surface_area[i + 1];

            diag [idx] = coef_r + mats.sig_r(mat, g);
            upper[idx] = -coef_r;

            // Left-interface: zero-gradient (symmetry) at i == 0
            if (i > 0) {
                const int    mat_l  = medium_map[i - 1];
                const double D_l    = mats.d(mat_l, g);
                const double D_left = D_i * D_l / (D_i + D_l);
                const double coef_l = 2.0 / (dx * volume[i]) * D_left * surface_area[i];
                diag [idx] += coef_l;
                lower[idx]  = -coef_l;
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

double norm2(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

}  // namespace

// ============================================================================
// DiffusionSolver — constructor
// ============================================================================

DiffusionSolver::DiffusionSolver(
    Materials                      mats,
    std::vector<int>               medium_map,
    std::vector<double>            edges_x,
    Geometry                       geom,
    std::vector<BoundaryCondition> bc,
    double epsilon,
    int    max_outer,
    int    max_inner,
    bool   verbose
):
      mats_      (std::move(mats)),
      medium_map_(std::move(medium_map)),
      edges_x_   (std::move(edges_x)),
      geom_      (geom),
      bc_        (std::move(bc)),
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

    compute_geometry(geom_, edges_x_, surface_area_, volume_);
    build_tridiagonals(mats_, medium_map_, edges_x_,
                       surface_area_, volume_, bc_,
                       cells_, groups_, N_,
                       lower_, diag_, upper_);
}

// ============================================================================
// DiffusionSolver — fission source operator  b = B * phi
// ============================================================================

void DiffusionSolver::apply_B(
    const std::vector<double>& phi,
          std::vector<double>& b
) const {
    b.assign(groups_ * N_, 0.0);

    for (int g = 0; g < groups_; ++g) {
        for (int i = 0; i < cells_; ++i) {
            const int mat = medium_map_[i];
            double src = 0.0;
            for (int gp = 0; gp < groups_; ++gp)
                src += mats_.chi_g(mat, g) * mats_.nu_sigf(mat, gp) * phi[gp * N_ + i];
            b[g * N_ + i] = src;
        }
        // b[g * N_ + cells_] stays zero (ghost BC row)
    }
}

// ============================================================================
// DiffusionSolver — matrix-free linear solve  A * phi = b
// ============================================================================

void DiffusionSolver::solve_A(
    const std::vector<double>& b,
          std::vector<double>& phi
) const {
    std::vector<double> lower_g(N_), diag_g(N_), upper_g(N_);
    std::vector<double> rhs(N_), phi_g(N_);

    for (int inner = 0; inner < max_inner_; ++inner) {
        const std::vector<double> phi_prev = phi;

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

            thomas(lower_g, diag_g, upper_g, rhs, phi_g, N_);

            for (int i = 0; i < N_; ++i)
                phi[g * N_ + i] = phi_g[i];
        }

        double change = 0.0;
        for (int i = 0; i < groups_ * N_; ++i) {
            const double diff = phi[i] - phi_prev[i];
            change += diff * diff;
        }
        if (std::sqrt(change) < epsilon_ * 1e-3)
            break;
    }
}

// ============================================================================
// DiffusionSolver — power iteration
// ============================================================================

DiffusionResult DiffusionSolver::solve() {
    const int total = groups_ * N_;

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

    while (change > epsilon_ && iter < max_outer_) {
        const std::vector<double> phi_old = phi;

        apply_B(phi_old, b);
        solve_A(b, phi);

        keff = norm2(phi);
        for (double& v : phi) v /= keff;

        change = 0.0;
        for (int i = 0; i < total; ++i) {
            const double d = phi[i] - phi_old[i];
            change += d * d;
        }
        change = std::sqrt(change);

        if (verbose_)
            std::printf("Iter: %3d  keff: %.8f  change: %.2e\n", iter + 1, keff, change);
        ++iter;
    }

    std::vector<double> flux_out(cells_ * groups_);
    for (int g = 0; g < groups_; ++g)
        for (int i = 0; i < cells_; ++i)
            flux_out[i * groups_ + g] = phi[g * N_ + i];

    return {flux_out, keff, iter, change};
}

// ============================================================================
// TimeDependentSolver — constructor
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
    bool   verbose
):
      mats_      (std::move(mats)),
      medium_map_(std::move(medium_map)),
      edges_x_   (std::move(edges_x)),
      geom_      (geom),
      bc_        (std::move(bc)),
      epsilon_   (epsilon),
      max_inner_ (max_inner),
      verbose_   (verbose),
      cells_     (static_cast<int>(medium_map_.size())),
      groups_    (mats_.n_groups),
      N_         (cells_ + 1),
      time_      (0.0),
      steps_     (0)
{
    if (static_cast<int>(bc_.size()) != groups_)
        throw std::invalid_argument("bc must have one entry per energy group");
    if (static_cast<int>(mats_.velocity.size()) != groups_)
        throw std::invalid_argument(
            "Materials.velocity must have one entry per energy group");

    compute_geometry(geom_, edges_x_, surface_area_, volume_);
    build_tridiagonals(mats_, medium_map_, edges_x_,
                       surface_area_, volume_, bc_,
                       cells_, groups_, N_,
                       lower_base_, diag_base_, upper_base_);

    // Convert initial_flux from [cells * groups] to internal [groups * N]
    phi_.assign(groups_ * N_, 0.0);
    if (!initial_flux.empty()) {
        if (static_cast<int>(initial_flux.size()) != cells_ * groups_)
            throw std::invalid_argument(
                "initial_flux must have cells * n_groups elements");
        for (int g = 0; g < groups_; ++g)
            for (int i = 0; i < cells_; ++i)
                phi_[g * N_ + i] = initial_flux[i * groups_ + g];
    }
}

// ============================================================================
// TimeDependentSolver — single backward-Euler time step
// ============================================================================
//
// The time-discretised equation for group g at cell i is:
//
//   [A_g + (1/v_g*dt) I] phi_g^{n+1}
//     = (1/v_g*dt) phi_g^n
//       + chi_g * sum_gp( nu_sigf_gp * phi_gp^n )    [fission, explicit]
//       + sum_{gp!=g} sig_s(g<-gp) * phi_gp^{n+1}    [scatter, implicit GS]
//
// The 1/(v_g*dt) term is added to the spatial diagonal at the start of
// each step; the base tridiagonals are left unchanged for reuse.

void TimeDependentSolver::step(double dt) {
    const std::vector<double> phi_old = phi_;

    // Explicit fission source from phi_old:  fis[g*N+i] = chi_g * sum_gp(nusigf*phi_old)
    std::vector<double> fis(groups_ * N_, 0.0);
    for (int g = 0; g < groups_; ++g) {
        for (int i = 0; i < cells_; ++i) {
            const int mat = medium_map_[i];
            double src = 0.0;
            for (int gp = 0; gp < groups_; ++gp)
                src += mats_.chi_g(mat, g) * mats_.nu_sigf(mat, gp) * phi_old[gp * N_ + i];
            fis[g * N_ + i] = src;
        }
    }

    // Gauss-Seidel inner iteration
    std::vector<double> lower_g(N_), diag_g(N_), upper_g(N_), rhs(N_), phi_g(N_);

    for (int inner = 0; inner < max_inner_; ++inner) {
        const std::vector<double> phi_iter = phi_;

        for (int g = 0; g < groups_; ++g) {
            const double inv_v_dt = 1.0 / (mats_.v(g) * dt);

            // Build RHS for this group
            for (int i = 0; i < cells_; ++i) {
                const int mat = medium_map_[i];
                rhs[i] = inv_v_dt * phi_old[g * N_ + i]  // time-source
                        + fis[g * N_ + i];                // fission (explicit)
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

            thomas(lower_g, diag_g, upper_g, rhs, phi_g, N_);

            for (int i = 0; i < N_; ++i)
                phi_[g * N_ + i] = phi_g[i];
        }

        // Check inner convergence
        double change = 0.0;
        for (int i = 0; i < groups_ * N_; ++i) {
            const double d = phi_[i] - phi_iter[i];
            change += d * d;
        }
        if (std::sqrt(change) < epsilon_)
            break;
    }

    time_  += dt;
    steps_ += 1;

    if (verbose_)
        std::printf("t = %.6e s  step %d  phi_max = %.6e\n",
                    time_, steps_,
                    *std::max_element(phi_.begin(), phi_.end()));
}

// ============================================================================
// TimeDependentSolver — run multiple steps
// ============================================================================

TimeDependentResult TimeDependentSolver::run(double dt, int n_steps) {
    for (int n = 0; n < n_steps; ++n)
        step(dt);
    return result();
}

// ============================================================================
// TimeDependentSolver — extract current state
// ============================================================================

TimeDependentResult TimeDependentSolver::result() const {
    std::vector<double> flux_out(cells_ * groups_);
    for (int g = 0; g < groups_; ++g)
        for (int i = 0; i < cells_; ++i)
            flux_out[i * groups_ + g] = phi_[g * N_ + i];
    return {flux_out, time_, steps_};
}
