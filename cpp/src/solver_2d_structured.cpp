#include <ndiffusion/solver_2d.hpp>

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

// Thomas / TDMA tridiagonal solver.
// Solves n equations; lower[0] and upper[n-1] are not referenced.
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
    for (int k = 1; k < n; ++k) {
        const double denom = diag[k] - lower[k] * c[k - 1];
        c[k] = upper[k] / denom;
        d[k] = (rhs[k] - lower[k] * d[k - 1]) / denom;
    }
    x[n - 1] = d[n - 1];
    for (int k = n - 2; k >= 0; --k)
        x[k] = d[k] - c[k] * x[k + 1];
}

double norm2(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

// Harmonic mean of two diffusion coefficients at a material interface.
// Returns the full harmonic mean 2ab/(a+b), matching the 1-D solver convention.
inline double d_harm(double a, double b) { return 2.0 * a * b / (a + b); }

// ============================================================================
// Geometry helpers
// ============================================================================

// Compute cell volumes and face areas for XY or RZ geometry.
//
// sa_x[(nx+1)*ny]: x-face area for each face; sa_x[i*ny+j] is the area of
//   the face between cells (i-1,j) and (i,j).  i=0 is the left boundary,
//   i=nx is the right boundary.
// sa_y[nx*(ny+1)]: y-face area for each face; sa_y[i*(ny+1)+j] is the area
//   of the face between cells (i,j-1) and (i,j).  j=0 is the bottom boundary,
//   j=ny is the top boundary.
void compute_geometry_2d(
    Geometry2D                  geom,
    const std::vector<double>&  edges_x,   // size nx+1
    const std::vector<double>&  edges_y,   // size ny+1
    int nx, int ny,
    std::vector<double>& vol,    // [nx*ny]
    std::vector<double>& sa_x,  // [(nx+1)*ny]
    std::vector<double>& sa_y   // [nx*(ny+1)]
) {
    vol .assign(nx * ny,       0.0);
    sa_x.assign((nx + 1) * ny, 0.0);
    sa_y.assign(nx * (ny + 1), 0.0);

    if (geom == Geometry2D::XY) {
        for (int i = 0; i <= nx; ++i)
            for (int j = 0; j < ny; ++j)
                sa_x[i * ny + j] = edges_y[j + 1] - edges_y[j];

        for (int i = 0; i < nx; ++i)
            for (int j = 0; j <= ny; ++j)
                sa_y[i * (ny + 1) + j] = edges_x[i + 1] - edges_x[i];

        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                vol[i * ny + j] = (edges_x[i + 1] - edges_x[i]) *
                                  (edges_y[j + 1] - edges_y[j]);

    } else {  // RZ: x = z (axial), y = r (radial)
        // z-face area = π*(r_{j+1}² − r_j²)
        for (int i = 0; i <= nx; ++i)
            for (int j = 0; j < ny; ++j)
                sa_x[i * ny + j] = PI * (edges_y[j + 1] * edges_y[j + 1] -
                                          edges_y[j]     * edges_y[j]);

        // r-face area = 2π*r_{j} * dz_i
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j <= ny; ++j)
                sa_y[i * (ny + 1) + j] = 2.0 * PI * edges_y[j] *
                                          (edges_x[i + 1] - edges_x[i]);

        // Cell volume = π*(r_{j+1}² − r_j²) * dz_i
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                vol[i * ny + j] = PI * (edges_y[j + 1] * edges_y[j + 1] -
                                         edges_y[j]     * edges_y[j]) *
                                    (edges_x[i + 1] - edges_x[i]);
    }
}

// Build precomputed stencil coefficients for the structured 2-D diffusion
// operator.
//
// Convention (cell (i,j), group g):
//   flat = g*(nx*ny) + i*ny + j
//
//   a_W[flat]  = west  (i-1,j) coupling; 0 at i=0 (reflective left)
//   a_E[flat]  = east  (i+1,j) coupling
//   a_S[flat]  = south (i,j-1) coupling; 0 at j=0 (reflective bottom)
//   a_N[flat]  = north (i,j+1) coupling; 0 at j=ny-1 (top-BC absorbed into diag)
//   diag[flat] = a_E + a_W + a_N_raw*(1-alpha_top)_or_a_N + a_S + sig_r
//
// The full 2-D equation for cell (i,j), group g is:
//   diag*phi[i,j] - a_W*phi[i-1,j] - a_E*phi[i+1,j]
//                 - a_S*phi[i,j-1] - a_N*phi[i,j+1]  = source
// where a_N == 0 at j=ny-1 (ghost absorbed into diag).
//
// Ghost row for right x-BC (one per group):
//   ghost_diag [g] = 0.5*A_x + B_x/dx_last
//   ghost_lower[g] = 0.5*A_x - B_x/dx_last
void build_coefficients_2d(
    const Materials&                      mats,
    const std::vector<int>&               medium_map,
    const std::vector<double>&            edges_x,
    const std::vector<double>&            edges_y,
    const std::vector<double>&            vol,
    const std::vector<double>&            sa_x,
    const std::vector<double>&            sa_y,
    const std::vector<BoundaryCondition>& bc_x,
    const std::vector<BoundaryCondition>& bc_y,
    int nx, int ny, int groups,
    std::vector<double>& a_W,
    std::vector<double>& a_E,
    std::vector<double>& a_S,
    std::vector<double>& a_N,
    std::vector<double>& diag,
    std::vector<double>& ghost_diag,
    std::vector<double>& ghost_lower
) {
    const int cells = nx * ny;
    a_W       .assign(groups * cells, 0.0);
    a_E       .assign(groups * cells, 0.0);
    a_S       .assign(groups * cells, 0.0);
    a_N       .assign(groups * cells, 0.0);
    diag      .assign(groups * cells, 0.0);
    ghost_diag .assign(groups, 0.0);
    ghost_lower.assign(groups, 0.0);

    const double dx_last = edges_x[nx] - edges_x[nx - 1];
    const double dy_last = edges_y[ny] - edges_y[ny - 1];

    for (int g = 0; g < groups; ++g) {

        // Ghost-row coefficients for the right x-BC.
        ghost_diag [g] = 0.5 * bc_x[g].A + bc_x[g].B / dx_last;
        ghost_lower[g] = 0.5 * bc_x[g].A - bc_x[g].B / dx_last;

        // Top-BC absorption factor for j = ny-1.
        // The ghost cell at j=ny satisfies: phi_ghost = alpha_top * phi[ny-1]
        // alpha_top = -(0.5*A_y - B_y/dy_last) / (0.5*A_y + B_y/dy_last)
        const double num_top   =  0.5 * bc_y[g].A - bc_y[g].B / dy_last;
        const double denom_top =  0.5 * bc_y[g].A + bc_y[g].B / dy_last;
        const double alpha_top = (std::abs(denom_top) > 1e-30) ? (-num_top / denom_top) : 1.0;

        for (int i = 0; i < nx; ++i) {
            const double dx = edges_x[i + 1] - edges_x[i];

            for (int j = 0; j < ny; ++j) {
                const int    flat = g * cells + i * ny + j;
                const int    mat  = medium_map[i * ny + j];
                const double dy   = edges_y[j + 1] - edges_y[j];
                const double V    = vol[i * ny + j];
                const double D_ij = mats.d(mat, g);

                // ---- East x-coupling ----
                const int    mat_e  = (i < nx - 1) ? medium_map[(i + 1) * ny + j] : mat;
                const double dx_e   = (i < nx - 1) ? (edges_x[i + 2] - edges_x[i + 1]) : dx;
                const double D_e    = d_harm(D_ij, mats.d(mat_e, g));
                const double coef_e = D_e * sa_x[(i + 1) * ny + j] /
                                      (0.5 * (dx + dx_e) * V);
                diag[flat] += coef_e;
                a_E [flat]  = coef_e;

                // ---- West x-coupling (reflective at i=0 — a_W stays 0) ----
                if (i > 0) {
                    const int    mat_w = medium_map[(i - 1) * ny + j];
                    const double dx_w  = edges_x[i] - edges_x[i - 1];
                    const double D_w   = d_harm(D_ij, mats.d(mat_w, g));
                    const double coef_w = D_w * sa_x[i * ny + j] /
                                          (0.5 * (dx_w + dx) * V);
                    diag[flat] += coef_w;
                    a_W [flat]  = coef_w;
                }

                // ---- North y-coupling ----
                const int    mat_n  = (j < ny - 1) ? medium_map[i * ny + (j + 1)] : mat;
                const double dy_n   = (j < ny - 1) ? (edges_y[j + 2] - edges_y[j + 1]) : dy;
                const double D_n    = d_harm(D_ij, mats.d(mat_n, g));
                const double coef_n = D_n * sa_y[i * (ny + 1) + (j + 1)] /
                                      (0.5 * (dy + dy_n) * V);

                if (j < ny - 1) {
                    // Interior north coupling: contribute to diagonal and store
                    // coupling coef in a_N so solve_A can add phi_north to RHS.
                    diag[flat] += coef_n;
                    a_N [flat]  = coef_n;
                } else {
                    // j == ny-1: absorb top-BC ghost into diagonal.
                    // diag_eff += coef_n * (1 - alpha_top).
                    // a_N stays 0 — no separate RHS contribution.
                    diag[flat] += coef_n * (1.0 - alpha_top);
                }

                // ---- South y-coupling (reflective at j=0 — a_S stays 0) ----
                if (j > 0) {
                    const int    mat_s = medium_map[i * ny + (j - 1)];
                    const double dy_s  = edges_y[j] - edges_y[j - 1];
                    const double D_s   = d_harm(D_ij, mats.d(mat_s, g));
                    const double coef_s = D_s * sa_y[i * (ny + 1) + j] /
                                          (0.5 * (dy_s + dy) * V);
                    diag[flat] += coef_s;
                    a_S [flat]  = coef_s;
                }

                // ---- Removal ----
                diag[flat] += mats.sig_r(mat, g);
            }
        }
    }
}

}  // namespace

// ============================================================================
// KEigenSolver2D — constructor
// ============================================================================

KEigenSolver2D::KEigenSolver2D(
    Materials                      mats,
    std::vector<int>               medium_map,
    std::vector<double>            edges_x,
    std::vector<double>            edges_y,
    Geometry2D                     geom,
    std::vector<BoundaryCondition> bc_x,
    std::vector<BoundaryCondition> bc_y,
    double epsilon, int max_outer, int max_inner, bool verbose
):
      mats_      (std::move(mats)),
      medium_map_(std::move(medium_map)),
      edges_x_   (std::move(edges_x)),
      edges_y_   (std::move(edges_y)),
      geom_      (geom),
      bc_x_      (std::move(bc_x)),
      bc_y_      (std::move(bc_y)),
      epsilon_   (epsilon),
      max_outer_ (max_outer),
      max_inner_ (max_inner),
      verbose_   (verbose),
      nx_        (static_cast<int>(edges_x_.size()) - 1),
      ny_        (static_cast<int>(edges_y_.size()) - 1),
      groups_    (mats_.n_groups)
{
    if (static_cast<int>(bc_x_.size()) != groups_)
        throw std::invalid_argument("bc_x must have one entry per energy group");
    if (static_cast<int>(bc_y_.size()) != groups_)
        throw std::invalid_argument("bc_y must have one entry per energy group");
    if (static_cast<int>(medium_map_.size()) != nx_ * ny_)
        throw std::invalid_argument("medium_map size must equal nx * ny");

    std::vector<double> vol, sa_x, sa_y;
    compute_geometry_2d(geom_, edges_x_, edges_y_, nx_, ny_, vol, sa_x, sa_y);
    build_coefficients_2d(mats_, medium_map_, edges_x_, edges_y_,
                          vol, sa_x, sa_y,
                          bc_x_, bc_y_, nx_, ny_, groups_,
                          a_W_, a_E_, a_S_, a_N_, diag_,
                          ghost_diag_, ghost_lower_);
}

// ============================================================================
// KEigenSolver2D — fission source  b = B * phi
// ============================================================================

void KEigenSolver2D::apply_B(
    const std::vector<double>& phi,
          std::vector<double>& b
) const {
    const int cells = nx_ * ny_;
    b.assign(groups_ * cells, 0.0);

    const bool fis_mat = mats_.use_fission_matrix();
    for (int g = 0; g < groups_; ++g) {
        for (int ij = 0; ij < cells; ++ij) {
            const int mat = medium_map_[ij];
            double src = 0.0;
            for (int gp = 0; gp < groups_; ++gp) {
                if (fis_mat)
                    src += mats_.nu_sigf_mat(mat, g, gp) * phi[gp * cells + ij];
                else
                    src += mats_.chi_g(mat, g) * mats_.nu_sigf(mat, gp) * phi[gp * cells + ij];
            }
            b[g * cells + ij] = src;
        }
    }
}

// ============================================================================
// KEigenSolver2D — linear solve  A * phi = b  (line TDMA + Gauss-Seidel)
//
// For each GS inner iteration:
//   For each group g:
//     Sweep columns j = 0 .. ny-1:
//       Build x-tridiagonal of size nx+1 (cells + ghost right-BC row).
//       RHS[i] = b[g,i,j]
//              + scatter from other groups  (latest iterate)
//              + a_S[g,i,j] * phi[g,i,j-1]  (known south; 0 at j=0)
//              + a_N[g,i,j] * phi[g,i,j+1]  (known north; 0 at j=ny-1)
//       Solve with Thomas → update phi[g, *, j].
// ============================================================================

void KEigenSolver2D::solve_A(
    const std::vector<double>& b,
          std::vector<double>& phi
) const {
    const int cells = nx_ * ny_;
    const int N_x   = nx_ + 1;  // interior cells + ghost

    std::vector<double> lower(N_x), diag_g(N_x), upper_g(N_x);
    std::vector<double> rhs(N_x), phi_x(N_x);

    for (int inner = 0; inner < max_inner_; ++inner) {
        const std::vector<double> phi_prev = phi;

        for (int g = 0; g < groups_; ++g) {
            for (int j = 0; j < ny_; ++j) {
                for (int i = 0; i < nx_; ++i) {
                    const int flat = g * cells + i * ny_ + j;
                    const int mat  = medium_map_[i * ny_ + j];

                    diag_g [i] = diag_[flat];
                    lower  [i] = (i > 0) ? -a_W_[flat] : 0.0;
                    upper_g[i] = -a_E_[flat];

                    // RHS: external source (fission/scatter from apply_B).
                    double r = b[flat];

                    // In-scatter from other groups.
                    for (int gp = 0; gp < groups_; ++gp)
                        if (gp != g)
                            r += mats_.sig_s(mat, g, gp) *
                                 phi[gp * cells + i * ny_ + j];

                    // South y-neighbour (reflective at j=0 → a_S=0).
                    if (j > 0)
                        r += a_S_[flat] * phi[g * cells + i * ny_ + (j - 1)];

                    // North y-neighbour (a_N=0 at j=ny-1 — absorbed into diag).
                    if (j < ny_ - 1)
                        r += a_N_[flat] * phi[g * cells + i * ny_ + (j + 1)];

                    rhs[i] = r;
                }

                // Ghost right-BC row.
                lower  [nx_] = ghost_lower_[g];
                diag_g [nx_] = ghost_diag_ [g];
                upper_g[nx_] = 0.0;
                rhs    [nx_] = 0.0;

                thomas(lower, diag_g, upper_g, rhs, phi_x, N_x);

                for (int i = 0; i < nx_; ++i)
                    phi[g * cells + i * ny_ + j] = phi_x[i];
            }
        }

        double change = 0.0;
        for (int k = 0; k < groups_ * cells; ++k) {
            const double d = phi[k] - phi_prev[k];
            change += d * d;
        }
        if (std::sqrt(change) < epsilon_ * 1e-3)
            break;
    }
}

// ============================================================================
// KEigenSolver2D — power iteration
// ============================================================================

DiffusionResult KEigenSolver2D::solve() {
    const int cells = nx_ * ny_;
    const int total = groups_ * cells;

    std::srand(42);
    std::vector<double> phi(total);
    for (int k = 0; k < total; ++k)
        phi[k] = static_cast<double>(std::rand()) / RAND_MAX + 1e-10;
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
        for (int k = 0; k < total; ++k) {
            const double d = phi[k] - phi_old[k];
            change += d * d;
        }
        change = std::sqrt(change);

        if (verbose_)
            std::printf("Iter: %3d  keff: %.8f  change: %.2e\n",
                        iter + 1, keff, change);
        ++iter;
    }

    std::vector<double> flux_out(cells * groups_);
    for (int g = 0; g < groups_; ++g)
        for (int ij = 0; ij < cells; ++ij)
            flux_out[ij * groups_ + g] = phi[g * cells + ij];

    return {flux_out, keff, iter, change};
}

// ============================================================================
// TimeDependentSolver2D — constructor
// ============================================================================

TimeDependentSolver2D::TimeDependentSolver2D(
    Materials                      mats,
    std::vector<int>               medium_map,
    std::vector<double>            edges_x,
    std::vector<double>            edges_y,
    Geometry2D                     geom,
    std::vector<BoundaryCondition> bc_x,
    std::vector<BoundaryCondition> bc_y,
    std::vector<double>            initial_flux,
    double epsilon, int max_inner, bool verbose
):
      mats_      (std::move(mats)),
      medium_map_(std::move(medium_map)),
      edges_x_   (std::move(edges_x)),
      edges_y_   (std::move(edges_y)),
      geom_      (geom),
      bc_x_      (std::move(bc_x)),
      bc_y_      (std::move(bc_y)),
      epsilon_   (epsilon),
      max_inner_ (max_inner),
      verbose_   (verbose),
      nx_        (static_cast<int>(edges_x_.size()) - 1),
      ny_        (static_cast<int>(edges_y_.size()) - 1),
      groups_    (mats_.n_groups),
      time_      (0.0),
      steps_     (0)
{
    if (static_cast<int>(bc_x_.size()) != groups_)
        throw std::invalid_argument("bc_x must have one entry per energy group");
    if (static_cast<int>(bc_y_.size()) != groups_)
        throw std::invalid_argument("bc_y must have one entry per energy group");
    if (static_cast<int>(mats_.velocity.size()) != groups_)
        throw std::invalid_argument(
            "Materials.velocity must have one entry per energy group");

    std::vector<double> sa_x, sa_y;
    compute_geometry_2d(geom_, edges_x_, edges_y_, nx_, ny_, vol_, sa_x, sa_y);
    build_coefficients_2d(mats_, medium_map_, edges_x_, edges_y_,
                          vol_, sa_x, sa_y,
                          bc_x_, bc_y_, nx_, ny_, groups_,
                          a_W_base_, a_E_base_, a_S_base_, a_N_base_, diag_base_,
                          ghost_diag_base_, ghost_lower_base_);

    const int cells = nx_ * ny_;
    phi_.assign(groups_ * cells, 0.0);
    if (!initial_flux.empty()) {
        if (static_cast<int>(initial_flux.size()) != cells * groups_)
            throw std::invalid_argument(
                "initial_flux must have nx*ny * n_groups elements");
        for (int g = 0; g < groups_; ++g)
            for (int ij = 0; ij < cells; ++ij)
                phi_[g * cells + ij] = initial_flux[ij * groups_ + g];
    }
}

// ============================================================================
// TimeDependentSolver2D — one backward-Euler step
// ============================================================================

void TimeDependentSolver2D::solve_step(
    const std::vector<double>& phi_old,
    const std::vector<double>& fis,
    double dt
) {
    const int cells = nx_ * ny_;
    const int N_x   = nx_ + 1;

    std::vector<double> lower(N_x), diag_g(N_x), upper_g(N_x);
    std::vector<double> rhs(N_x), phi_x(N_x);

    for (int inner = 0; inner < max_inner_; ++inner) {
        const std::vector<double> phi_iter = phi_;

        for (int g = 0; g < groups_; ++g) {
            const double inv_v_dt = 1.0 / (mats_.v(g) * dt);

            for (int j = 0; j < ny_; ++j) {
                for (int i = 0; i < nx_; ++i) {
                    const int flat = g * cells + i * ny_ + j;
                    const int mat  = medium_map_[i * ny_ + j];

                    // Diagonal with time-absorption added.
                    diag_g [i] = diag_base_[flat] + inv_v_dt;
                    lower  [i] = (i > 0) ? -a_W_base_[flat] : 0.0;
                    upper_g[i] = -a_E_base_[flat];

                    double r = inv_v_dt * phi_old[flat]   // time source
                             + fis[flat];                 // fission (explicit)

                    for (int gp = 0; gp < groups_; ++gp)
                        if (gp != g)
                            r += mats_.sig_s(mat, g, gp) *
                                 phi_[gp * cells + i * ny_ + j];

                    if (j > 0)
                        r += a_S_base_[flat] * phi_[g * cells + i * ny_ + (j - 1)];

                    if (j < ny_ - 1)
                        r += a_N_base_[flat] * phi_[g * cells + i * ny_ + (j + 1)];

                    rhs[i] = r;
                }

                lower  [nx_] = ghost_lower_base_[g];
                diag_g [nx_] = ghost_diag_base_ [g];
                upper_g[nx_] = 0.0;
                rhs    [nx_] = 0.0;

                thomas(lower, diag_g, upper_g, rhs, phi_x, N_x);

                for (int i = 0; i < nx_; ++i)
                    phi_[g * cells + i * ny_ + j] = phi_x[i];
            }
        }

        double change = 0.0;
        for (int k = 0; k < groups_ * cells; ++k) {
            const double d = phi_[k] - phi_iter[k];
            change += d * d;
        }
        if (std::sqrt(change) < epsilon_)
            break;
    }
}

void TimeDependentSolver2D::step(double dt) {
    const int cells = nx_ * ny_;
    const std::vector<double> phi_old = phi_;

    std::vector<double> fis(groups_ * cells, 0.0);
    const bool fis_mat = mats_.use_fission_matrix();
    for (int g = 0; g < groups_; ++g) {
        for (int ij = 0; ij < cells; ++ij) {
            const int mat = medium_map_[ij];
            double src = 0.0;
            for (int gp = 0; gp < groups_; ++gp) {
                if (fis_mat)
                    src += mats_.nu_sigf_mat(mat, g, gp) * phi_old[gp * cells + ij];
                else
                    src += mats_.chi_g(mat, g) * mats_.nu_sigf(mat, gp) *
                           phi_old[gp * cells + ij];
            }
            fis[g * cells + ij] = src;
        }
    }

    solve_step(phi_old, fis, dt);

    time_  += dt;
    steps_ += 1;

    if (verbose_)
        std::printf("t = %.6e s  step %d  phi_max = %.6e\n",
                    time_, steps_,
                    *std::max_element(phi_.begin(), phi_.end()));
}

TimeDependentResult TimeDependentSolver2D::run(double dt, int n_steps) {
    for (int n = 0; n < n_steps; ++n)
        step(dt);
    return result();
}

TimeDependentResult TimeDependentSolver2D::result() const {
    const int cells = nx_ * ny_;
    std::vector<double> flux_out(cells * groups_);
    for (int g = 0; g < groups_; ++g)
        for (int ij = 0; ij < cells; ++ij)
            flux_out[ij * groups_ + g] = phi_[g * cells + ij];
    return {flux_out, time_, steps_};
}

// ============================================================================
// FixedSourceSolver2D — constructor
// ============================================================================

FixedSourceSolver2D::FixedSourceSolver2D(
    Materials                      mats,
    std::vector<int>               medium_map,
    std::vector<double>            edges_x,
    std::vector<double>            edges_y,
    Geometry2D                     geom,
    std::vector<BoundaryCondition> bc_x,
    std::vector<BoundaryCondition> bc_y,
    double epsilon, int max_inner, bool verbose
):
      mats_      (std::move(mats)),
      medium_map_(std::move(medium_map)),
      edges_x_   (std::move(edges_x)),
      edges_y_   (std::move(edges_y)),
      geom_      (geom),
      bc_x_      (std::move(bc_x)),
      bc_y_      (std::move(bc_y)),
      epsilon_   (epsilon),
      max_inner_ (max_inner),
      verbose_   (verbose),
      nx_        (static_cast<int>(edges_x_.size()) - 1),
      ny_        (static_cast<int>(edges_y_.size()) - 1),
      groups_    (mats_.n_groups)
{
    if (static_cast<int>(bc_x_.size()) != groups_)
        throw std::invalid_argument("bc_x must have one entry per energy group");
    if (static_cast<int>(bc_y_.size()) != groups_)
        throw std::invalid_argument("bc_y must have one entry per energy group");
    if (static_cast<int>(medium_map_.size()) != nx_ * ny_)
        throw std::invalid_argument("medium_map size must equal nx * ny");

    std::vector<double> vol, sa_x, sa_y;
    compute_geometry_2d(geom_, edges_x_, edges_y_, nx_, ny_, vol, sa_x, sa_y);
    build_coefficients_2d(mats_, medium_map_, edges_x_, edges_y_,
                          vol, sa_x, sa_y,
                          bc_x_, bc_y_, nx_, ny_, groups_,
                          a_W_, a_E_, a_S_, a_N_, diag_,
                          ghost_diag_, ghost_lower_);
}

// ============================================================================
// FixedSourceSolver2D — solve  A·φ = source
// ============================================================================

FixedSourceResult FixedSourceSolver2D::solve(const std::vector<double>& source) const {
    const int cells = nx_ * ny_;
    if (static_cast<int>(source.size()) != cells * groups_)
        throw std::invalid_argument("source must have nx*ny * n_groups elements");

    const int N_x = nx_ + 1;  // interior cells + ghost

    // Convert source from [cells * groups] row-major to internal [groups * cells].
    // The structured FD equation is per unit volume (build_coefficients_2d divides
    // all coupling coefficients by V), so the volumetric source feeds directly.
    std::vector<double> src(groups_ * cells, 0.0);
    for (int g = 0; g < groups_; ++g)
        for (int ij = 0; ij < cells; ++ij)
            src[g * cells + ij] = source[ij * groups_ + g];

    std::vector<double> phi(groups_ * cells, 0.0);
    std::vector<double> lower(N_x), diag_g(N_x), upper_g(N_x), rhs(N_x), phi_x(N_x);

    double residual = 1.0;
    int    iter     = 0;

    for (; iter < max_inner_; ++iter) {
        const std::vector<double> phi_prev = phi;

        for (int g = 0; g < groups_; ++g) {
            for (int j = 0; j < ny_; ++j) {
                for (int i = 0; i < nx_; ++i) {
                    const int flat = g * cells + i * ny_ + j;
                    const int mat  = medium_map_[i * ny_ + j];

                    diag_g [i] = diag_[flat];
                    lower  [i] = (i > 0) ? -a_W_[flat] : 0.0;
                    upper_g[i] = -a_E_[flat];

                    double r = src[flat];  // external volumetric source

                    // In-scatter from other groups (latest iterate).
                    for (int gp = 0; gp < groups_; ++gp)
                        if (gp != g)
                            r += mats_.sig_s(mat, g, gp) *
                                 phi[gp * cells + i * ny_ + j];

                    // South y-neighbour (a_S=0 at j=0 — reflective).
                    if (j > 0)
                        r += a_S_[flat] * phi[g * cells + i * ny_ + (j - 1)];

                    // North y-neighbour (a_N=0 at j=ny_-1 — absorbed into diag).
                    if (j < ny_ - 1)
                        r += a_N_[flat] * phi[g * cells + i * ny_ + (j + 1)];

                    rhs[i] = r;
                }

                // Ghost right-BC row.
                lower  [nx_] = ghost_lower_[g];
                diag_g [nx_] = ghost_diag_ [g];
                upper_g[nx_] = 0.0;
                rhs    [nx_] = 0.0;

                thomas(lower, diag_g, upper_g, rhs, phi_x, N_x);

                for (int i = 0; i < nx_; ++i)
                    phi[g * cells + i * ny_ + j] = phi_x[i];
            }
        }

        double change = 0.0;
        for (int k = 0; k < groups_ * cells; ++k) {
            const double d = phi[k] - phi_prev[k];
            change += d * d;
        }
        residual = std::sqrt(change);

        if (verbose_)
            std::printf("Iter: %3d  residual: %.2e\n", iter + 1, residual);

        if (residual < epsilon_)
            break;
    }

    std::vector<double> flux_out(cells * groups_);
    for (int g = 0; g < groups_; ++g)
        for (int ij = 0; ij < cells; ++ij)
            flux_out[ij * groups_ + g] = phi[g * cells + ij];

    return {flux_out, iter + 1, residual};
}
