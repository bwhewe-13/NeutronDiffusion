#include <ndiffusion/solver_2d.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>

// ============================================================================
// File-local helpers
// ============================================================================

namespace {

double norm2(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

// Full harmonic mean 2ab/(a+b) for interface diffusion coefficients.
inline double d_harm(double a, double b) { return 2.0 * a * b / (a + b); }

inline double cross2d(double ax, double ay, double bx, double by) {
    return ax * by - ay * bx;
}

// Compute centroid and area of a single cell from its vertex indices.
void cell_geometry(
    const std::vector<double>& vx,
    const std::vector<double>& vy,
    const std::vector<int>&    verts,
    double& cx, double& cy, double& area
) {
    const int nv = static_cast<int>(verts.size());
    if (nv == 3) {
        const double x0 = vx[verts[0]], y0 = vy[verts[0]];
        const double x1 = vx[verts[1]], y1 = vy[verts[1]];
        const double x2 = vx[verts[2]], y2 = vy[verts[2]];
        cx   = (x0 + x1 + x2) / 3.0;
        cy   = (y0 + y1 + y2) / 3.0;
        area = 0.5 * std::abs(cross2d(x1 - x0, y1 - y0, x2 - x0, y2 - y0));
    } else {  // quad
        const double x0 = vx[verts[0]], y0 = vy[verts[0]];
        const double x1 = vx[verts[1]], y1 = vy[verts[1]];
        const double x2 = vx[verts[2]], y2 = vy[verts[2]];
        const double x3 = vx[verts[3]], y3 = vy[verts[3]];

        const double a012 = 0.5 * std::abs(
            cross2d(x1 - x0, y1 - y0, x2 - x0, y2 - y0));
        const double cx012 = (x0 + x1 + x2) / 3.0;
        const double cy012 = (y0 + y1 + y2) / 3.0;

        const double a023 = 0.5 * std::abs(
            cross2d(x2 - x0, y2 - y0, x3 - x0, y3 - y0));
        const double cx023 = (x0 + x2 + x3) / 3.0;
        const double cy023 = (y0 + y2 + y3) / 3.0;

        area = a012 + a023;
        if (area > 0.0) {
            cx = (a012 * cx012 + a023 * cx023) / area;
            cy = (a012 * cy012 + a023 * cy023) / area;
        } else {
            cx = (x0 + x1 + x2 + x3) / 4.0;
            cy = (y0 + y1 + y2 + y3) / 4.0;
        }
    }
}

using EdgeKey = std::pair<int, int>;
struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& k) const {
        return std::hash<long long>()(
            (static_cast<long long>(k.first) << 32) | k.second);
    }
};

// ============================================================================
// Shared mesh preprocessing: centroid/area + face building
// ============================================================================

void preprocess_mesh(
    const UnstructuredMesh2D&            mesh,
    int&                                 n_cells,
    std::vector<double>&                 cell_area,
    std::vector<double>&                 cell_cx,
    std::vector<double>&                 cell_cy,
    std::vector<FaceUnstructured2D>&     faces,
    std::vector<std::vector<int>>&       cell_faces
) {
    n_cells = static_cast<int>(mesh.cell_offsets.size()) - 1;

    cell_area  .resize(n_cells);
    cell_cx    .resize(n_cells);
    cell_cy    .resize(n_cells);
    cell_faces .resize(n_cells);

    // Compute centroid and area for each cell.
    for (int c = 0; c < n_cells; ++c) {
        const int off0 = mesh.cell_offsets[c];
        const int off1 = mesh.cell_offsets[c + 1];
        std::vector<int> verts(mesh.cell_vertices.begin() + off0,
                               mesh.cell_vertices.begin() + off1);
        cell_geometry(mesh.vx, mesh.vy, verts,
                      cell_cx[c], cell_cy[c], cell_area[c]);
    }

    // Build boundary-face BC lookup: canonical edge → bc_tag.
    std::unordered_map<EdgeKey, int, EdgeKeyHash> bface_map;
    const int nbf = static_cast<int>(mesh.bface_v0.size());
    for (int f = 0; f < nbf; ++f) {
        int v0 = mesh.bface_v0[f], v1 = mesh.bface_v1[f];
        if (v0 > v1) std::swap(v0, v1);
        const int tag = (f < static_cast<int>(mesh.bface_bc_tag.size()))
                        ? mesh.bface_bc_tag[f] : 0;
        bface_map[{v0, v1}] = tag;
    }

    // Hash all cell edges.  First encounter: record as half-face.
    // Second encounter: create interior face and erase from map.
    // After the loop, remaining entries are boundary faces.
    struct HalfFace { int cell; };
    std::unordered_map<EdgeKey, HalfFace, EdgeKeyHash> edge_map;

    for (int c = 0; c < n_cells; ++c) {
        const int off0 = mesh.cell_offsets[c];
        const int off1 = mesh.cell_offsets[c + 1];
        const int nv   = off1 - off0;

        for (int e = 0; e < nv; ++e) {
            int va = mesh.cell_vertices[off0 + e];
            int vb = mesh.cell_vertices[off0 + (e + 1) % nv];
            int vlo = va, vhi = vb;
            if (vlo > vhi) std::swap(vlo, vhi);
            const EdgeKey key{vlo, vhi};

            auto it = edge_map.find(key);
            if (it == edge_map.end()) {
                edge_map[key] = {c};
            } else {
                // Interior face between c0 = it->second.cell and c1 = c.
                const int c0 = it->second.cell;
                const int c1 = c;

                const double fx0 = mesh.vx[vlo], fy0 = mesh.vy[vlo];
                const double fx1 = mesh.vx[vhi], fy1 = mesh.vy[vhi];
                const double L   = std::hypot(fx1 - fx0, fy1 - fy0);
                const double d   = std::hypot(cell_cx[c1] - cell_cx[c0],
                                              cell_cy[c1] - cell_cy[c0]);

                FaceUnstructured2D face;
                face.c0     = c0;
                face.c1     = c1;
                face.length = L;
                face.dist   = d;
                face.a_coef = (d > 0.0) ? L / d : 0.0;
                face.bc_tag = -1;

                const int fidx = static_cast<int>(faces.size());
                faces.push_back(face);
                cell_faces[c0].push_back(fidx);
                cell_faces[c1].push_back(fidx);

                edge_map.erase(it);
            }
        }
    }

    // Remaining entries in edge_map are boundary faces.
    for (auto& [key, hf] : edge_map) {
        const int c0  = hf.cell;
        const int vlo = key.first, vhi = key.second;

        const double fx0 = mesh.vx[vlo], fy0 = mesh.vy[vlo];
        const double fx1 = mesh.vx[vhi], fy1 = mesh.vy[vhi];
        const double L   = std::hypot(fx1 - fx0, fy1 - fy0);

        // Distance from centroid to face midpoint.
        const double mx  = 0.5 * (fx0 + fx1);
        const double my  = 0.5 * (fy0 + fy1);
        const double d   = std::hypot(cell_cx[c0] - mx, cell_cy[c0] - my);

        auto bit = bface_map.find(key);
        const int bc_tag = (bit != bface_map.end()) ? bit->second : 0;

        FaceUnstructured2D face;
        face.c0     = c0;
        face.c1     = -1;
        face.length = L;
        face.dist   = d;
        face.a_coef = (d > 0.0) ? L / d : 0.0;
        face.bc_tag = bc_tag;

        const int fidx = static_cast<int>(faces.size());
        faces.push_back(face);
        cell_faces[c0].push_back(fidx);
    }
}

// ============================================================================
// Build per-group, per-cell diagonal (base, without time term).
//
// The unstructured FVM system (per-cell, volume-integrated) is:
//   a_diag[c,g] * phi[c,g]  -  Σ_f a_f[g] * phi[nbr,g]  =  rhs[c,g]
// where rhs[c,g] includes fission/scatter * cell_area[c].
//
// a_diag_base[g * n_cells + c] accumulates:
//   interior faces:  D_harm * L/d  (same contribution to both sides)
//   boundary faces:  D_c * L/d * A / (0.5*A + B/d)   (BC absorption)
//   removal:         sig_r * cell_area
// ============================================================================

void build_diagonals(
    const Materials&                      mats,
    const UnstructuredMesh2D&             mesh,
    const std::vector<BoundaryCondition>& bc,
    int n_cells, int groups,
    const std::vector<double>&            cell_area,
    const std::vector<FaceUnstructured2D>& faces,
    std::vector<double>&                  a_diag_base
) {
    const int n_bc_types = (groups > 0)
                           ? static_cast<int>(bc.size()) / groups : 1;
    a_diag_base.assign(groups * n_cells, 0.0);

    for (int g = 0; g < groups; ++g) {
        // Interior face contributions (add to both cells).
        for (const auto& f : faces) {
            if (f.c1 < 0) continue;
            const int c0 = f.c0, c1 = f.c1;
            const double D0  = mats.d(mesh.material_id[c0], g);
            const double D1  = mats.d(mesh.material_id[c1], g);
            const double aij = d_harm(D0, D1) * f.a_coef;
            a_diag_base[g * n_cells + c0] += aij;
            a_diag_base[g * n_cells + c1] += aij;
        }

        // Boundary face contributions (BC absorbed into diagonal).
        for (const auto& f : faces) {
            if (f.c1 >= 0) continue;
            const int c0  = f.c0;
            const int tag = f.bc_tag;
            if (tag < 0 || tag >= n_bc_types) continue;

            const BoundaryCondition& bci = bc[tag * groups + g];
            const double D_c   = mats.d(mesh.material_id[c0], g);
            const double A     = bci.A, B = bci.B;
            const double d     = f.dist;
            // FVM Robin BC: A*phi_s + B*(dphi/dn)_s = 0 with linear extrapolation
            // gives a_bc = D * (L/d) * A / (A + B/d).
            // Note: no 0.5 factor — that appears only in the structured ghost-node scheme.
            const double denom = A + (d > 0.0 ? B / d : 0.0);
            if (std::abs(denom) > 1e-30)
                a_diag_base[g * n_cells + c0] += D_c * f.a_coef * A / denom;
        }

        // Removal.
        for (int c = 0; c < n_cells; ++c)
            a_diag_base[g * n_cells + c] +=
                mats.sig_r(mesh.material_id[c], g) * cell_area[c];
    }
}

}  // namespace

// ============================================================================
// KEigenSolverUnstructured2D — constructor
// ============================================================================

KEigenSolverUnstructured2D::KEigenSolverUnstructured2D(
    Materials          mats,
    UnstructuredMesh2D mesh,
    std::vector<BoundaryCondition> bc,
    double epsilon, int max_outer, int max_inner, bool verbose
):
      mats_      (std::move(mats)),
      mesh_      (std::move(mesh)),
      bc_        (std::move(bc)),
      epsilon_   (epsilon),
      max_outer_ (max_outer),
      max_inner_ (max_inner),
      verbose_   (verbose),
      n_cells_   (0),
      groups_    (mats_.n_groups)
{
    if (mesh_.cell_offsets.empty())
        throw std::invalid_argument("mesh cell_offsets must not be empty");

    preprocess_mesh();
    build_diagonals();
}

void KEigenSolverUnstructured2D::preprocess_mesh() {
    ::preprocess_mesh(mesh_, n_cells_, cell_area_, cell_cx_, cell_cy_,
                      faces_, cell_faces_);
}

void KEigenSolverUnstructured2D::build_diagonals() {
    ::build_diagonals(mats_, mesh_, bc_, n_cells_, groups_,
                      cell_area_, faces_, a_diag_base_);
}

// ============================================================================
// KEigenSolverUnstructured2D — fission source  b = B * phi
// ============================================================================

void KEigenSolverUnstructured2D::apply_B(
    const std::vector<double>& phi,
          std::vector<double>& b
) const {
    b.assign(groups_ * n_cells_, 0.0);
    const bool fis_mat = mats_.use_fission_matrix();
    for (int g = 0; g < groups_; ++g) {
        for (int c = 0; c < n_cells_; ++c) {
            const int mat = mesh_.material_id[c];
            double src = 0.0;
            for (int gp = 0; gp < groups_; ++gp) {
                if (fis_mat)
                    src += mats_.nu_sigf_mat(mat, g, gp) * phi[gp * n_cells_ + c];
                else
                    src += mats_.chi_g(mat, g) * mats_.nu_sigf(mat, gp) *
                           phi[gp * n_cells_ + c];
            }
            b[g * n_cells_ + c] = src * cell_area_[c];
        }
    }
}

// ============================================================================
// KEigenSolverUnstructured2D — linear solve  A * phi = b  (point GS)
//
// For each sweep over cells 0..n_cells-1, for each group g:
//   phi[c,g] = (b[c,g] + scatter*area + Σ_f D_harm*a_coef*phi[nbr,g])
//              / a_diag_base[c,g]
// ============================================================================

void KEigenSolverUnstructured2D::solve_A(
    const std::vector<double>& b,
          std::vector<double>& phi
) const {
    for (int inner = 0; inner < max_inner_; ++inner) {
        const std::vector<double> phi_prev = phi;

        for (int c = 0; c < n_cells_; ++c) {
            const int    mat  = mesh_.material_id[c];
            const double area = cell_area_[c];

            for (int g = 0; g < groups_; ++g) {
                double rhs = b[g * n_cells_ + c];

                // In-scatter from other groups.
                for (int gp = 0; gp < groups_; ++gp)
                    if (gp != g)
                        rhs += mats_.sig_s(mat, g, gp) *
                               phi[gp * n_cells_ + c] * area;

                // Interior face neighbour contributions.
                for (int fi : cell_faces_[c]) {
                    const FaceUnstructured2D& f = faces_[fi];
                    if (f.c1 < 0) continue;
                    const int nbr = (f.c0 == c) ? f.c1 : f.c0;
                    const double D0 = mats_.d(mesh_.material_id[c],   g);
                    const double D1 = mats_.d(mesh_.material_id[nbr], g);
                    rhs += d_harm(D0, D1) * f.a_coef * phi[g * n_cells_ + nbr];
                }

                phi[g * n_cells_ + c] = rhs / a_diag_base_[g * n_cells_ + c];
            }
        }

        double change = 0.0;
        for (int k = 0; k < groups_ * n_cells_; ++k) {
            const double d = phi[k] - phi_prev[k];
            change += d * d;
        }
        if (std::sqrt(change) < epsilon_ * 1e-3)
            break;
    }
}

// ============================================================================
// KEigenSolverUnstructured2D — power iteration
// ============================================================================

DiffusionResult KEigenSolverUnstructured2D::solve() {
    const int total = groups_ * n_cells_;

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

    std::vector<double> flux_out(n_cells_ * groups_);
    for (int g = 0; g < groups_; ++g)
        for (int c = 0; c < n_cells_; ++c)
            flux_out[c * groups_ + g] = phi[g * n_cells_ + c];

    return {flux_out, keff, iter, change};
}

// ============================================================================
// TimeDependentSolverUnstructured2D — constructor
// ============================================================================

TimeDependentSolverUnstructured2D::TimeDependentSolverUnstructured2D(
    Materials          mats,
    UnstructuredMesh2D mesh,
    std::vector<BoundaryCondition> bc,
    std::vector<double>            initial_flux,
    double epsilon, int max_inner, bool verbose
):
      mats_      (std::move(mats)),
      mesh_      (std::move(mesh)),
      bc_        (std::move(bc)),
      epsilon_   (epsilon),
      max_inner_ (max_inner),
      verbose_   (verbose),
      n_cells_   (0),
      groups_    (mats_.n_groups),
      time_      (0.0),
      steps_     (0)
{
    if (static_cast<int>(mats_.velocity.size()) != groups_)
        throw std::invalid_argument(
            "Materials.velocity must have one entry per energy group");
    if (mesh_.cell_offsets.empty())
        throw std::invalid_argument("mesh cell_offsets must not be empty");

    preprocess_mesh();
    build_diagonals();

    phi_.assign(groups_ * n_cells_, 0.0);
    if (!initial_flux.empty()) {
        if (static_cast<int>(initial_flux.size()) != n_cells_ * groups_)
            throw std::invalid_argument(
                "initial_flux must have n_cells * n_groups elements");
        for (int g = 0; g < groups_; ++g)
            for (int c = 0; c < n_cells_; ++c)
                phi_[g * n_cells_ + c] = initial_flux[c * groups_ + g];
    }
}

void TimeDependentSolverUnstructured2D::preprocess_mesh() {
    ::preprocess_mesh(mesh_, n_cells_, cell_area_, cell_cx_, cell_cy_,
                      faces_, cell_faces_);
}

void TimeDependentSolverUnstructured2D::build_diagonals() {
    ::build_diagonals(mats_, mesh_, bc_, n_cells_, groups_,
                      cell_area_, faces_, a_diag_base_);
}

// ============================================================================
// TimeDependentSolverUnstructured2D — one backward-Euler step
// ============================================================================

void TimeDependentSolverUnstructured2D::solve_step(
    const std::vector<double>& phi_old,
    const std::vector<double>& fis,
    double dt
) {
    for (int inner = 0; inner < max_inner_; ++inner) {
        const std::vector<double> phi_iter = phi_;

        for (int c = 0; c < n_cells_; ++c) {
            const int    mat  = mesh_.material_id[c];
            const double area = cell_area_[c];

            for (int g = 0; g < groups_; ++g) {
                const double inv_v_dt = 1.0 / (mats_.v(g) * dt);
                const double diag = a_diag_base_[g * n_cells_ + c] + inv_v_dt * area;

                double rhs = inv_v_dt * phi_old[g * n_cells_ + c] * area
                           + fis[g * n_cells_ + c];

                for (int gp = 0; gp < groups_; ++gp)
                    if (gp != g)
                        rhs += mats_.sig_s(mat, g, gp) *
                               phi_[gp * n_cells_ + c] * area;

                for (int fi : cell_faces_[c]) {
                    const FaceUnstructured2D& f = faces_[fi];
                    if (f.c1 < 0) continue;
                    const int nbr = (f.c0 == c) ? f.c1 : f.c0;
                    const double D0 = mats_.d(mesh_.material_id[c],   g);
                    const double D1 = mats_.d(mesh_.material_id[nbr], g);
                    rhs += d_harm(D0, D1) * f.a_coef * phi_[g * n_cells_ + nbr];
                }

                phi_[g * n_cells_ + c] = rhs / diag;
            }
        }

        double change = 0.0;
        for (int k = 0; k < groups_ * n_cells_; ++k) {
            const double d = phi_[k] - phi_iter[k];
            change += d * d;
        }
        if (std::sqrt(change) < epsilon_)
            break;
    }
}

void TimeDependentSolverUnstructured2D::step(double dt) {
    const std::vector<double> phi_old = phi_;

    std::vector<double> fis(groups_ * n_cells_, 0.0);
    const bool fis_mat = mats_.use_fission_matrix();
    for (int g = 0; g < groups_; ++g) {
        for (int c = 0; c < n_cells_; ++c) {
            const int mat = mesh_.material_id[c];
            double src = 0.0;
            for (int gp = 0; gp < groups_; ++gp) {
                if (fis_mat)
                    src += mats_.nu_sigf_mat(mat, g, gp) * phi_old[gp * n_cells_ + c];
                else
                    src += mats_.chi_g(mat, g) * mats_.nu_sigf(mat, gp) *
                           phi_old[gp * n_cells_ + c];
            }
            fis[g * n_cells_ + c] = src * cell_area_[c];
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

TimeDependentResult TimeDependentSolverUnstructured2D::run(double dt, int n_steps) {
    for (int n = 0; n < n_steps; ++n)
        step(dt);
    return result();
}

TimeDependentResult TimeDependentSolverUnstructured2D::result() const {
    std::vector<double> flux_out(n_cells_ * groups_);
    for (int g = 0; g < groups_; ++g)
        for (int c = 0; c < n_cells_; ++c)
            flux_out[c * groups_ + g] = phi_[g * n_cells_ + c];
    return {flux_out, time_, steps_};
}

// ============================================================================
// FixedSourceSolverUnstructured2D — constructor
// ============================================================================

FixedSourceSolverUnstructured2D::FixedSourceSolverUnstructured2D(
    Materials          mats,
    UnstructuredMesh2D mesh,
    std::vector<BoundaryCondition> bc,
    double epsilon, int max_inner, double omega, bool verbose
):
      mats_      (std::move(mats)),
      mesh_      (std::move(mesh)),
      bc_        (std::move(bc)),
      epsilon_   (epsilon),
      max_inner_ (max_inner),
      omega_     (omega),
      verbose_   (verbose),
      n_cells_   (0),
      groups_    (mats_.n_groups)
{
    if (mesh_.cell_offsets.empty())
        throw std::invalid_argument("mesh cell_offsets must not be empty");

    preprocess_mesh();
    build_diagonals();
}

void FixedSourceSolverUnstructured2D::preprocess_mesh() {
    ::preprocess_mesh(mesh_, n_cells_, cell_area_, cell_cx_, cell_cy_,
                      faces_, cell_faces_);
}

void FixedSourceSolverUnstructured2D::build_diagonals() {
    ::build_diagonals(mats_, mesh_, bc_, n_cells_, groups_,
                      cell_area_, faces_, a_diag_base_);
}

// ============================================================================
// FixedSourceSolverUnstructured2D — solve  A·φ = source  (point GS)
// ============================================================================

FixedSourceResult FixedSourceSolverUnstructured2D::solve(
    const std::vector<double>& source
) const {
    if (static_cast<int>(source.size()) != n_cells_ * groups_)
        throw std::invalid_argument("source must have n_cells * n_groups elements");

    // Convert source from [n_cells * groups] row-major to internal [groups * n_cells]
    // and multiply by cell_area to form the volume-integrated RHS.
    // This matches how apply_B multiplies the fission density by cell_area in the
    // k-eigenvalue solver (the FVM equation is volume-integrated throughout).
    std::vector<double> src_vol(groups_ * n_cells_, 0.0);
    for (int g = 0; g < groups_; ++g)
        for (int c = 0; c < n_cells_; ++c)
            src_vol[g * n_cells_ + c] = source[c * groups_ + g] * cell_area_[c];

    std::vector<double> phi(groups_ * n_cells_, 0.0);

    double residual = 1.0;
    int    iter     = 0;

    for (; iter < max_inner_; ++iter) {
        const std::vector<double> phi_prev = phi;

        for (int c = 0; c < n_cells_; ++c) {
            const int    mat  = mesh_.material_id[c];
            const double area = cell_area_[c];

            for (int g = 0; g < groups_; ++g) {
                double rhs = src_vol[g * n_cells_ + c];  // volume-integrated source

                // In-scatter from other groups (latest iterate), volume-integrated.
                for (int gp = 0; gp < groups_; ++gp)
                    if (gp != g)
                        rhs += mats_.sig_s(mat, g, gp) *
                               phi[gp * n_cells_ + c] * area;

                // Interior face neighbour contributions.
                // Boundary faces (c1 < 0) are already absorbed into a_diag_base_.
                for (int fi : cell_faces_[c]) {
                    const FaceUnstructured2D& f = faces_[fi];
                    if (f.c1 < 0) continue;
                    const int nbr = (f.c0 == c) ? f.c1 : f.c0;
                    const double D0 = mats_.d(mesh_.material_id[c],   g);
                    const double D1 = mats_.d(mesh_.material_id[nbr], g);
                    rhs += d_harm(D0, D1) * f.a_coef * phi[g * n_cells_ + nbr];
                }

                const double phi_gs = rhs / a_diag_base_[g * n_cells_ + c];
                phi[g * n_cells_ + c] = (1.0 - omega_) * phi[g * n_cells_ + c]
                                       + omega_ * phi_gs;
            }
        }

        double change = 0.0;
        for (int k = 0; k < groups_ * n_cells_; ++k) {
            const double d = phi[k] - phi_prev[k];
            change += d * d;
        }
        residual = std::sqrt(change);

        if (verbose_)
            std::printf("Iter: %3d  residual: %.2e\n", iter + 1, residual);

        if (residual < epsilon_)
            break;
    }

    std::vector<double> flux_out(n_cells_ * groups_);
    for (int g = 0; g < groups_; ++g)
        for (int c = 0; c < n_cells_; ++c)
            flux_out[c * groups_ + g] = phi[g * n_cells_ + c];

    return {flux_out, iter + 1, residual};
}
