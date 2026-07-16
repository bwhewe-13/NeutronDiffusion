#include <ndiffusion/solver_2d.hpp>
#include <ndiffusion/solver_detail.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>

using namespace ndiffusion::detail;

// ============================================================================
// File-local helpers
// ============================================================================

namespace {

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

    // Build boundary-face BC lookup: canonical edge -> bc_tag.
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
//   a_diag[c,g] * phi[c,g]  -  Sigma_f a_f[g] * phi[nbr,g]  =  rhs[c,g]
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
            // Note: no 0.5 factor - that appears only in the structured ghost-node scheme.
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
// KEigenSolverUnstructured2D - constructor
// ============================================================================

KEigenSolverUnstructured2D::KEigenSolverUnstructured2D(
    Materials          mats,
    UnstructuredMesh2D mesh,
    std::vector<BoundaryCondition> bc,
    double epsilon, int max_outer, int max_inner, bool verbose,
    std::optional<bool> use_cg
):
      mats_      (std::move(mats)),
      mesh_      (std::move(mesh)),
      bc_        (std::move(bc)),
      epsilon_   (epsilon),
      max_outer_ (max_outer),
      max_inner_ (max_inner),
      verbose_   (verbose),
      use_cg_    (use_cg.value_or(
                      ndiffusion::detail::env_flag("NDIFFUSION_KEIG_CG"))),
      n_cells_   (0),
      groups_    (mats_.n_groups)
{
    if (mesh_.cell_offsets.empty())
        throw std::invalid_argument("mesh cell_offsets must not be empty");

    preprocess_mesh();
    if (n_cells_ < 1)
        throw std::invalid_argument("mesh must have at least one cell");
    if (static_cast<int>(mesh_.material_id.size()) != n_cells_)
        throw std::invalid_argument("material_id size must equal number of cells");
    validate_materials(mats_);
    validate_material_ids(mesh_.material_id, mats_.n_mat, "material_id");
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
// KEigenSolverUnstructured2D - fission source  b = B * phi
// ============================================================================

void KEigenSolverUnstructured2D::apply_B(
    const std::vector<double>& phi,
          std::vector<double>& b
) const {
    // FVM source is volume-integrated, so each cell is weighted by its area.
    accumulate_fission(mats_, mesh_.material_id, groups_, n_cells_, n_cells_,
                       &cell_area_, phi, b);
}

// ============================================================================
// KEigenSolverUnstructured2D - linear solve  A * phi = b  (point GS)
//
// For each sweep over cells 0..n_cells-1, for each group g:
//   phi[c,g] = (b[c,g] + scatter*area + Sigma_f D_harm*a_coef*phi[nbr,g])
//              / a_diag_base[c,g]
// ============================================================================

bool KEigenSolverUnstructured2D::solve_A(
    const std::vector<double>& b,
          std::vector<double>& phi
) const {
    return use_cg_ ? solve_A_cg(b, phi) : solve_A_gs(b, phi);
}

bool KEigenSolverUnstructured2D::solve_A_gs(
    const std::vector<double>& b,
          std::vector<double>& phi
) const {
    std::vector<double> phi_prev;
    for (int inner = 0; inner < max_inner_; ++inner) {
        phi_prev = phi;

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

        if (rel_l2_diff(phi, phi_prev) < epsilon_ * 1e-3)
            return true;
    }
    return false;
}

// ============================================================================
// KEigenSolverUnstructured2D - linear solve  (Option B: within-group CG)
//
// Block Gauss-Seidel over energy groups; each within-group SPD system is solved
// with matrix-free Jacobi-preconditioned CG. The FVM operator is already
// volume-integrated and symmetric (a_ij = D_harm * a_coef shared by both cells,
// boundary BCs absorbed into a_diag_base_), so no symmetrization is needed.
// ============================================================================

bool KEigenSolverUnstructured2D::solve_A_cg(
    const std::vector<double>& b,
          std::vector<double>& phi
) const {
    // Within-group symmetric operator for group g:  out = A_g * v.
    auto apply_Ag = [this](int g, const std::vector<double>& v,
                           std::vector<double>& out) {
        const int base = g * n_cells_;
        for (int c = 0; c < n_cells_; ++c) {
            double s = a_diag_base_[base + c] * v[c];
            for (int fi : cell_faces_[c]) {
                const FaceUnstructured2D& f = faces_[fi];
                if (f.c1 < 0) continue;
                const int nbr = (f.c0 == c) ? f.c1 : f.c0;
                const double D0 = mats_.d(mesh_.material_id[c],   g);
                const double D1 = mats_.d(mesh_.material_id[nbr], g);
                s -= d_harm(D0, D1) * f.a_coef * v[nbr];
            }
            out[c] = s;
        }
    };

    std::vector<double> rhs_g(n_cells_), x_g(n_cells_), phi_prev;
    const int    max_cg = 2 * n_cells_ + 50;
    const double cg_tol = std::min(epsilon_ * 1e-2, 1e-9);

    for (int sweep = 0; sweep < max_inner_; ++sweep) {
        phi_prev = phi;
        bool cg_all_ok = true;

        for (int g = 0; g < groups_; ++g) {
            const int base = g * n_cells_;

            // RHS: external source (already volume-weighted in b) + in-scatter*area.
            for (int c = 0; c < n_cells_; ++c) {
                const int mat = mesh_.material_id[c];
                double r = b[base + c];
                for (int gp = 0; gp < groups_; ++gp)
                    if (gp != g)
                        r += mats_.sig_s(mat, g, gp) *
                             phi[gp * n_cells_ + c] * cell_area_[c];
                rhs_g[c] = r;
            }

            for (int c = 0; c < n_cells_; ++c) x_g[c] = phi[base + c];

            bool cg_ok = false;
            cg_solve(n_cells_, rhs_g, x_g, &a_diag_base_[base], cg_tol, max_cg,
                     [&](const std::vector<double>& v, std::vector<double>& o) {
                         apply_Ag(g, v, o);
                     },
                     cg_ok);
            if (!cg_ok) cg_all_ok = false;

            for (int c = 0; c < n_cells_; ++c) phi[base + c] = x_g[c];
        }

        // Report failure if a within-group CG stalled so the caller can warn.
        if (groups_ == 1 || rel_l2_diff(phi, phi_prev) < epsilon_ * 1e-3)
            return cg_all_ok;
    }
    return false;
}

// ============================================================================
// KEigenSolverUnstructured2D - power iteration
// ============================================================================

DiffusionResult KEigenSolverUnstructured2D::solve() {
    bool inner_ok = true;
    PowerResult pr = power_iteration(
        groups_ * n_cells_, epsilon_, max_outer_, verbose_,
        [this](const std::vector<double>& in, std::vector<double>& out) {
            apply_B(in, out);
        },
        [this, &inner_ok](const std::vector<double>& rhs, std::vector<double>& x) {
            if (!solve_A(rhs, x)) inner_ok = false;
        });

    if (!inner_ok)
        warn_inner_not_converged("KEigenSolverUnstructured2D", max_inner_);

    std::vector<double> flux_out;
    pack_flux(pr.phi, n_cells_, groups_, n_cells_, flux_out);
    return {flux_out, pr.keff, pr.iters, pr.change, pr.converged && inner_ok};
}

// ============================================================================
// TimeDependentSolverUnstructured2D - constructor
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
    if (n_cells_ < 1)
        throw std::invalid_argument("mesh must have at least one cell");
    if (static_cast<int>(mesh_.material_id.size()) != n_cells_)
        throw std::invalid_argument("material_id size must equal number of cells");
    validate_materials(mats_);
    validate_material_ids(mesh_.material_id, mats_.n_mat, "material_id");
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
// TimeDependentSolverUnstructured2D - one backward-Euler step
// ============================================================================

void TimeDependentSolverUnstructured2D::solve_step(
    const std::vector<double>& phi_old,
    const std::vector<double>& fis,
    double dt
) {
    std::vector<double> phi_iter;
    for (int inner = 0; inner < max_inner_; ++inner) {
        phi_iter = phi_;

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

        // Relative criterion - the physical flux magnitude can be large.
        if (rel_l2_diff(phi_, phi_iter) < epsilon_)
            break;
    }
}

void TimeDependentSolverUnstructured2D::step(double dt) {
    const std::vector<double> phi_old = phi_;

    std::vector<double> fis;
    accumulate_fission(mats_, mesh_.material_id, groups_, n_cells_, n_cells_,
                       &cell_area_, phi_old, fis);

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
    std::vector<double> flux_out;
    pack_flux(phi_, n_cells_, groups_, n_cells_, flux_out);
    return {flux_out, time_, steps_};
}

// ============================================================================
// FixedSourceSolverUnstructured2D - constructor
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
    if (n_cells_ < 1)
        throw std::invalid_argument("mesh must have at least one cell");
    if (static_cast<int>(mesh_.material_id.size()) != n_cells_)
        throw std::invalid_argument("material_id size must equal number of cells");
    validate_materials(mats_);
    validate_material_ids(mesh_.material_id, mats_.n_mat, "material_id");
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
// FixedSourceSolverUnstructured2D - solve  A*phi = source  (point GS)
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
    std::vector<double> src_vol;
    unpack_flux(source, n_cells_, groups_, n_cells_, &cell_area_, src_vol);

    std::vector<double> phi(groups_ * n_cells_, 0.0);
    std::vector<double> phi_prev;

    double residual = 1.0;
    int    iter     = 0;

    for (; iter < max_inner_; ++iter) {
        phi_prev = phi;

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

        residual = rel_l2_diff(phi, phi_prev);

        if (verbose_)
            std::printf("Iter: %3d  residual: %.2e\n", iter + 1, residual);

        if (residual < epsilon_)
            break;
    }

    std::vector<double> flux_out;
    pack_flux(phi, n_cells_, groups_, n_cells_, flux_out);
    return {flux_out, iter + 1, residual, residual < epsilon_};
}
