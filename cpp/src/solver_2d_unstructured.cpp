#include <ndiffusion/solver_2d.hpp>
#include <ndiffusion/solver_detail.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

// Centroid and area of a single cell from its vertex indices.
//
// The shoelace formulae, valid for any simple polygon with at least 3 vertices
// listed in order (either winding):
//
//   2A     = sum_i (x_i y_{i+1} - x_{i+1} y_i)
//   6A Cx  = sum_i (x_i + x_{i+1}) (x_i y_{i+1} - x_{i+1} y_i)
//   6A Cy  = sum_i (y_i + y_{i+1}) (x_i y_{i+1} - x_{i+1} y_i)
//
// A degenerate (zero-area) cell falls back to the vertex average, which keeps
// the centroid finite; validate_mesh rejects such cells before a solve.
void cell_geometry(
    const std::vector<double>& vx,
    const std::vector<double>& vy,
    const std::vector<int>&    verts,
    double& cx, double& cy, double& area
) {
    const int nv = static_cast<int>(verts.size());

    double a2 = 0.0, cx6 = 0.0, cy6 = 0.0;
    for (int i = 0; i < nv; ++i) {
        const int j = (i + 1) % nv;
        const double xi = vx[verts[i]], yi = vy[verts[i]];
        const double xj = vx[verts[j]], yj = vy[verts[j]];
        const double cross = cross2d(xi, yi, xj, yj);
        a2  += cross;
        cx6 += (xi + xj) * cross;
        cy6 += (yi + yj) * cross;
    }

    area = 0.5 * std::abs(a2);
    if (std::abs(a2) > 0.0) {
        cx = cx6 / (3.0 * a2);
        cy = cy6 / (3.0 * a2);
    } else {
        cx = 0.0;
        cy = 0.0;
        for (int i = 0; i < nv; ++i) { cx += vx[verts[i]]; cy += vy[verts[i]]; }
        cx /= nv;
        cy /= nv;
    }
}

}  // namespace

// ============================================================================
// Unstructured mesh geometry (public free functions)
// ============================================================================

void validate_mesh(const UnstructuredMesh2D& mesh) {
    if (mesh.vx.size() != mesh.vy.size())
        throw std::invalid_argument(
            "mesh.vx and mesh.vy must have the same length (got " +
            std::to_string(mesh.vx.size()) + " and " +
            std::to_string(mesh.vy.size()) + ")");

    if (mesh.cell_offsets.empty())
        throw std::invalid_argument("mesh.cell_offsets must not be empty");
    if (mesh.cell_offsets.front() != 0)
        throw std::invalid_argument("mesh.cell_offsets must start at 0");

    const int n_cells = static_cast<int>(mesh.cell_offsets.size()) - 1;
    const int n_verts = static_cast<int>(mesh.vx.size());

    for (int c = 0; c < n_cells; ++c) {
        const int nv = mesh.cell_offsets[c + 1] - mesh.cell_offsets[c];
        if (nv < 3)
            throw std::invalid_argument(
                "mesh cell " + std::to_string(c) + " has " + std::to_string(nv) +
                " vertices; a cell needs at least 3 (triangles, quadrilaterals "
                "and higher polygons such as hexagons are all supported)");
    }
    if (mesh.cell_offsets.back() != static_cast<int>(mesh.cell_vertices.size()))
        throw std::invalid_argument(
            "mesh.cell_offsets must end at cell_vertices.size() (got " +
            std::to_string(mesh.cell_offsets.back()) + " and " +
            std::to_string(mesh.cell_vertices.size()) + ")");

    for (int v : mesh.cell_vertices)
        if (v < 0 || v >= n_verts)
            throw std::invalid_argument(
                "mesh.cell_vertices contains vertex index " + std::to_string(v) +
                " outside [0, " + std::to_string(n_verts) + ")");

    if (mesh.bface_v0.size() != mesh.bface_v1.size())
        throw std::invalid_argument(
            "mesh.bface_v0 and mesh.bface_v1 must have the same length (got " +
            std::to_string(mesh.bface_v0.size()) + " and " +
            std::to_string(mesh.bface_v1.size()) + "); each boundary face is a "
            "vertex pair");
    if (mesh.bface_bc_tag.size() > mesh.bface_v0.size())
        throw std::invalid_argument(
            "mesh.bface_bc_tag has more entries (" +
            std::to_string(mesh.bface_bc_tag.size()) + ") than there are "
            "boundary faces (" + std::to_string(mesh.bface_v0.size()) +
            "); a shorter array is padded with tag 0, a longer one is a sizing "
            "mistake");

    for (std::size_t f = 0; f < mesh.bface_v0.size(); ++f)
        for (int v : {mesh.bface_v0[f], mesh.bface_v1[f]})
            if (v < 0 || v >= n_verts)
                throw std::invalid_argument(
                    "mesh boundary face " + std::to_string(f) + " references "
                    "vertex index " + std::to_string(v) + " outside [0, " +
                    std::to_string(n_verts) + ")");

    const std::size_t np = mesh.periodic_a0.size();
    if (mesh.periodic_a1.size() != np || mesh.periodic_b0.size() != np ||
        mesh.periodic_b1.size() != np)
        throw std::invalid_argument(
            "mesh.periodic_a0/a1/b0/b1 must all have the same length (got " +
            std::to_string(mesh.periodic_a0.size()) + ", " +
            std::to_string(mesh.periodic_a1.size()) + ", " +
            std::to_string(mesh.periodic_b0.size()) + ", " +
            std::to_string(mesh.periodic_b1.size()) +
            "); each pair joins one edge to another, vertex to vertex");

    for (std::size_t k = 0; k < np; ++k)
        for (int v : {mesh.periodic_a0[k], mesh.periodic_a1[k],
                      mesh.periodic_b0[k], mesh.periodic_b1[k]})
            if (v < 0 || v >= n_verts)
                throw std::invalid_argument(
                    "periodic pair " + std::to_string(k) + " references vertex "
                    "index " + std::to_string(v) + " outside [0, " +
                    std::to_string(n_verts) + ")");

    // A zero-area cell carries no removal and no source, and its centroid is
    // not inside it, so the whole finite-volume balance for that cell is void.
    // Usually a repeated vertex or a collinear "polygon".
    for (int c = 0; c < n_cells; ++c) {
        const std::vector<int> verts(
            mesh.cell_vertices.begin() + mesh.cell_offsets[c],
            mesh.cell_vertices.begin() + mesh.cell_offsets[c + 1]);
        double cx = 0.0, cy = 0.0, area = 0.0;
        cell_geometry(mesh.vx, mesh.vy, verts, cx, cy, area);
        if (!(area > 0.0))
            throw std::invalid_argument(
                "mesh cell " + std::to_string(c) + " has zero area; its "
                "vertices are repeated or collinear");
    }
}

std::pair<std::vector<double>, std::vector<double>>
cell_centroids(const UnstructuredMesh2D& mesh) {
    validate_mesh(mesh);
    const int n_cells = static_cast<int>(mesh.cell_offsets.size()) - 1;
    std::vector<double> cx(n_cells), cy(n_cells);
    double area = 0.0;
    for (int c = 0; c < n_cells; ++c) {
        const std::vector<int> verts(
            mesh.cell_vertices.begin() + mesh.cell_offsets[c],
            mesh.cell_vertices.begin() + mesh.cell_offsets[c + 1]);
        cell_geometry(mesh.vx, mesh.vy, verts, cx[c], cy[c], area);
    }
    return {std::move(cx), std::move(cy)};
}

std::vector<double> cell_areas(const UnstructuredMesh2D& mesh) {
    validate_mesh(mesh);
    const int n_cells = static_cast<int>(mesh.cell_offsets.size()) - 1;
    std::vector<double> areas(n_cells);
    double cx = 0.0, cy = 0.0;
    for (int c = 0; c < n_cells; ++c) {
        const std::vector<int> verts(
            mesh.cell_vertices.begin() + mesh.cell_offsets[c],
            mesh.cell_vertices.begin() + mesh.cell_offsets[c + 1]);
        cell_geometry(mesh.vx, mesh.vy, verts, cx, cy, areas[c]);
    }
    return areas;
}

namespace {

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

// Fill a face's surface vector, over-relaxed implicit coefficient, and
// non-orthogonal correction vector.
//
//   S = length * n, oriented from (px,py) toward (qx,qy)
//   e = (q - p)/|q - p|
//   E = (S.S)/(e.S) e,   T = S - E,   a_coef = |E|/dist
//
// On an orthogonal face e is parallel to n, so E == S, T == 0 and a_coef is the
// familiar length/dist.  `(fx0,fy0)-(fx1,fy1)` are the face endpoints; (px,py)
// is the owning centroid and (qx,qy) the neighbor centroid (or, on a boundary
// face, the face midpoint).
void fill_face_geometry(FaceUnstructured2D& face,
                        double fx0, double fy0, double fx1, double fy1,
                        double px, double py, double qx, double qy,
                        bool boundary) {
    double ex = qx - px, ey = qy - py;

    // Defaults for an ordinary face: the neighbor is exactly where it looks,
    // and the two cells share a frame.  A periodic face overwrites these.
    face.d0x = ex;  face.d0y = ey;
    face.d1x = -ex; face.d1y = -ey;
    face.rot_cos = 1.0;
    face.rot_sin = 0.0;

    // Face normal: rotate the edge by 90 degrees, then orient it from p to q.
    double sx = (fy1 - fy0), sy = -(fx1 - fx0);
    if (sx * ex + sy * ey < 0.0) { sx = -sx; sy = -sy; }
    face.sx = sx;
    face.sy = sy;

    if (boundary) {
        // The boundary condition is a statement about the *normal* derivative at
        // the face, so the distance that enters it is the normal distance from
        // the centroid to the face plane - not the oblique centroid-to-midpoint
        // distance, which would make the BC and the flux coefficient disagree.
        const double len = std::hypot(sx, sy);
        if (len <= 0.0) {
            face.dist = 0.0; face.a_coef = 0.0; face.tx = face.ty = 0.0; return;
        }
        const double dn = (ex * sx + ey * sy) / len;
        face.dist   = dn;
        face.a_coef = (dn > 0.0) ? face.length / dn : 0.0;
        face.tx = face.ty = 0.0;
        return;
    }

    const double dist = std::hypot(ex, ey);
    face.dist = dist;

    if (dist <= 0.0) {
        face.a_coef = 0.0;
        face.tx = face.ty = 0.0;
        return;
    }

    const double eux = ex / dist, euy = ey / dist;
    const double e_dot_s = eux * sx + euy * sy;
    const double s_dot_s = sx * sx + sy * sy;

    if (e_dot_s <= 1e-12 * std::sqrt(s_dot_s)) {
        // Centroid line almost tangential to the face: the over-relaxed split
        // blows up, so fall back to the plain two-point flux with no correction.
        face.a_coef = face.length / dist;
        face.tx = face.ty = 0.0;
        return;
    }

    const double e_mag = s_dot_s / e_dot_s;      // |E|
    face.a_coef = e_mag / dist;
    face.tx = sx - e_mag * eux;
    face.ty = sy - e_mag * euy;
}

// Solve the per-cell weighted least-squares gradient fit and store the result as
// per-face coefficients.
//
// For cell P with neighbors k (a boundary face contributing its face midpoint):
//
//   minimize  sum_k w_k [ grad_P . d_k - (phi_k - phi_P) ]^2,   w_k = 1/|d_k|^2
//
// giving the 2x2 normal equations  M grad_P = sum_k w_k d_k (phi_k - phi_P)  with
// M = sum_k w_k d_k d_k^T.  M depends only on geometry, so M^-1 w_k d_k is
// precomputed here and the gradient becomes a plain weighted sum at run time.
//
// A cell whose neighbors are collinear gives a singular M; its coefficients are
// left at zero, which drops the correction for that cell rather than producing a
// meaningless gradient.
void build_lsq_gradients(
    int n_cells,
    std::vector<FaceUnstructured2D>&       faces,
    const std::vector<std::vector<int>>&   cell_faces
) {
    struct Normal { double xx, xy, yy; };
    std::vector<Normal> M(n_cells, {0.0, 0.0, 0.0});

    auto offset = [&](const FaceUnstructured2D& f, int c, double& dx, double& dy) {
        if (f.c1 >= 0) {
            // Stored per side, so a periodic neighbor contributes its image
            // rather than its actual position on the far side of the domain.
            if (c == f.c0) { dx = f.d0x; dy = f.d0y; }
            else           { dx = f.d1x; dy = f.d1y; }
        } else {
            // Boundary: the sample point is the face midpoint, which sits at
            // dist along the outward normal from the centroid.
            const double len = std::hypot(f.sx, f.sy);
            if (len <= 0.0) { dx = dy = 0.0; return; }
            dx = f.sx / len * f.dist;
            dy = f.sy / len * f.dist;
        }
    };

    for (int c = 0; c < n_cells; ++c)
        for (int fi : cell_faces[c]) {
            double dx, dy;
            offset(faces[fi], c, dx, dy);
            const double d2 = dx * dx + dy * dy;
            if (d2 <= 0.0) continue;
            const double w = 1.0 / d2;
            M[c].xx += w * dx * dx;
            M[c].xy += w * dx * dy;
            M[c].yy += w * dy * dy;
        }

    std::vector<double> ixx(n_cells, 0.0), ixy(n_cells, 0.0), iyy(n_cells, 0.0);
    for (int c = 0; c < n_cells; ++c) {
        const double det = M[c].xx * M[c].yy - M[c].xy * M[c].xy;
        const double scale = M[c].xx + M[c].yy;
        if (std::abs(det) <= 1e-14 * scale * scale) continue;  // singular: skip
        ixx[c] =  M[c].yy / det;
        ixy[c] = -M[c].xy / det;
        iyy[c] =  M[c].xx / det;
    }

    for (auto& f : faces) {
        f.lsq0x = f.lsq0y = f.lsq1x = f.lsq1y = 0.0;
        for (int side = 0; side < (f.c1 >= 0 ? 2 : 1); ++side) {
            const int c = side == 0 ? f.c0 : f.c1;
            double dx, dy;
            offset(f, c, dx, dy);
            const double d2 = dx * dx + dy * dy;
            if (d2 <= 0.0) continue;
            const double w = 1.0 / d2;
            const double cx = ixx[c] * (w * dx) + ixy[c] * (w * dy);
            const double cy = ixy[c] * (w * dx) + iyy[c] * (w * dy);
            if (side == 0) { f.lsq0x = cx; f.lsq0y = cy; }
            else           { f.lsq1x = cx; f.lsq1y = cy; }
        }
    }
}

// Reject a non-conforming (hanging-node) mesh.
//
// Where a large cell abuts two smaller ones, the large cell's edge (a,b) and the
// small cells' edges (a,m),(m,b) are three distinct keys that can never pair up,
// so all three are left looking like boundary faces and the mesh interior is
// quietly cut apart - the solve then succeeds and returns a wrong answer.  The
// signature is exactly that: a vertex lying strictly inside a boundary face.
//
// O(B^2) over boundary faces only, and B ~ sqrt(n_cells), so this costs little
// next to the solve it protects.
void check_conforming(
    const UnstructuredMesh2D& mesh,
    const std::vector<std::pair<EdgeKey, int>>& boundary_edges
) {
    std::vector<int> bverts;
    bverts.reserve(2 * boundary_edges.size());
    for (const auto& be : boundary_edges) {
        bverts.push_back(be.first.first);
        bverts.push_back(be.first.second);
    }
    std::sort(bverts.begin(), bverts.end());
    bverts.erase(std::unique(bverts.begin(), bverts.end()), bverts.end());

    // Two distinct vertex indices at the same point are the other way a mesh
    // comes apart: neighboring cells that were never merged share no index, so
    // none of their edges pair and every face looks like a boundary.  Those
    // duplicates sit at edge *endpoints*, so the interior test below cannot see
    // them.  Compare against the mesh extent so the tolerance scales.
    double xmin = 0.0, xmax = 0.0, ymin = 0.0, ymax = 0.0;
    if (!mesh.vx.empty()) {
        xmin = xmax = mesh.vx[0];
        ymin = ymax = mesh.vy[0];
        for (std::size_t i = 1; i < mesh.vx.size(); ++i) {
            xmin = std::min(xmin, mesh.vx[i]); xmax = std::max(xmax, mesh.vx[i]);
            ymin = std::min(ymin, mesh.vy[i]); ymax = std::max(ymax, mesh.vy[i]);
        }
    }
    const double diag = std::hypot(xmax - xmin, ymax - ymin);
    const double tol  = 1e-10 * (diag > 0.0 ? diag : 1.0);

    std::vector<int> by_pos = bverts;
    std::sort(by_pos.begin(), by_pos.end(), [&](int a, int b) {
        if (mesh.vx[a] != mesh.vx[b]) return mesh.vx[a] < mesh.vx[b];
        return mesh.vy[a] < mesh.vy[b];
    });
    for (std::size_t i = 1; i < by_pos.size(); ++i) {
        const int a = by_pos[i - 1], b = by_pos[i];
        if (std::abs(mesh.vx[a] - mesh.vx[b]) <= tol &&
            std::abs(mesh.vy[a] - mesh.vy[b]) <= tol)
            throw std::invalid_argument(
                "mesh vertices " + std::to_string(a) + " and " +
                std::to_string(b) + " are at the same point but have different "
                "indices, so the cells using them share no edge and the mesh "
                "comes apart into disconnected pieces. Merge coincident vertices "
                "when building the mesh (Gmsh does this on import)");
    }

    for (const auto& be : boundary_edges) {
        const int a = be.first.first, b = be.first.second;
        const double ax = mesh.vx[a], ay = mesh.vy[a];
        const double bx = mesh.vx[b], by = mesh.vy[b];
        const double ex = bx - ax, ey = by - ay;
        const double len2 = ex * ex + ey * ey;
        if (len2 <= 0.0) continue;

        for (int v : bverts) {
            if (v == a || v == b) continue;
            const double px = mesh.vx[v] - ax, py = mesh.vy[v] - ay;
            // Strictly between the endpoints along the edge?
            const double t = px * ex + py * ey;
            if (t <= 0.0 || t >= len2) continue;
            // And on the line, to a tolerance relative to the edge length?
            const double cross = std::abs(cross2d(ex, ey, px, py));
            if (cross <= 1e-9 * len2)
                throw std::invalid_argument(
                    "mesh vertex " + std::to_string(v) + " lies inside the edge ("
                    + std::to_string(a) + ", " + std::to_string(b) + "), so the "
                    "mesh is non-conforming (a hanging node). The finite-volume "
                    "discretization needs matching faces: that edge cannot pair "
                    "with its neighbors and would be treated as a boundary, "
                    "silently disconnecting part of the mesh interior");
        }
    }
}

void preprocess_mesh(
    const UnstructuredMesh2D&            mesh,
    int&                                 n_cells,
    std::vector<double>&                 cell_area,
    std::vector<double>&                 cell_cx,
    std::vector<double>&                 cell_cy,
    std::vector<FaceUnstructured2D>&     faces,
    std::vector<std::vector<int>>&       cell_faces
) {
    validate_mesh(mesh);
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

    // Hash all cell edges.  First encounter: record as a half-face holding its
    // owning cell.  Second encounter: create the interior face and mark the entry
    // kEdgeClosed - rather than erasing it, so a third cell on the same edge is
    // caught rather than reappearing below as a boundary face.  After the loop,
    // the entries still holding a cell are the boundary faces.
    constexpr int kEdgeClosed = -1;
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
            } else if (it->second.cell == kEdgeClosed) {
                throw std::invalid_argument(
                    "mesh edge (" + std::to_string(vlo) + ", " +
                    std::to_string(vhi) + ") is shared by more than two cells; "
                    "the finite-volume discretization needs a manifold mesh - "
                    "every edge must be interior to exactly two cells or on the "
                    "boundary of one");
            } else {
                // Interior face between c0 = it->second.cell and c1 = c.
                const int c0 = it->second.cell;
                const int c1 = c;

                const double fx0 = mesh.vx[vlo], fy0 = mesh.vy[vlo];
                const double fx1 = mesh.vx[vhi], fy1 = mesh.vy[vhi];

                FaceUnstructured2D face;
                face.c0     = c0;
                face.c1     = c1;
                face.length = std::hypot(fx1 - fx0, fy1 - fy0);
                face.bc_tag = -1;
                fill_face_geometry(face, fx0, fy0, fx1, fy1,
                                   cell_cx[c0], cell_cy[c0],
                                   cell_cx[c1], cell_cy[c1],
                                   /*boundary=*/false);

                // Interpolation weight from the centroid distances to the face
                // midpoint, used when averaging the two cell gradients onto it.
                const double mx = 0.5 * (fx0 + fx1), my = 0.5 * (fy0 + fy1);
                const double d0 = std::hypot(cell_cx[c0] - mx, cell_cy[c0] - my);
                const double d1 = std::hypot(cell_cx[c1] - mx, cell_cy[c1] - my);
                face.w0 = (d0 + d1 > 0.0) ? d1 / (d0 + d1) : 0.5;

                const int fidx = static_cast<int>(faces.size());
                faces.push_back(face);
                cell_faces[c0].push_back(fidx);
                cell_faces[c1].push_back(fidx);

                it->second.cell = kEdgeClosed;
            }
        }
    }

    // Periodic pairs are joined to each other rather than left as boundaries.
    // Each side is an edge that has been seen once, so both are still open here;
    // consuming them now keeps them out of the boundary list below.
    const int n_periodic = static_cast<int>(mesh.periodic_a0.size());
    for (int k = 0; k < n_periodic; ++k) {
        const int a0 = mesh.periodic_a0[k], a1 = mesh.periodic_a1[k];
        const int b0 = mesh.periodic_b0[k], b1 = mesh.periodic_b1[k];

        auto take = [&](int v0, int v1, const char* side) {
            const EdgeKey key{std::min(v0, v1), std::max(v0, v1)};
            auto it = edge_map.find(key);
            if (it == edge_map.end() || it->second.cell == kEdgeClosed)
                throw std::invalid_argument(
                    std::string("periodic pair ") + std::to_string(k) + " names " +
                    side + " edge (" + std::to_string(v0) + ", " +
                    std::to_string(v1) + "), which is not an unpaired boundary "
                    "edge of the mesh; both sides of a periodic pair must lie on "
                    "the boundary and be listed once");
            const int cell = it->second.cell;
            it->second.cell = kEdgeClosed;
            return cell;
        };
        const int ca = take(a0, a1, "the first");
        const int cb = take(b0, b1, "the second");

        // Rigid transform carrying side B onto side A, fixed by the vertex
        // correspondence a0<->b0, a1<->b1.  Cell B's interior then maps to the
        // far side of edge A, which is exactly where the neighbor belongs.
        const double ax = mesh.vx[a1] - mesh.vx[a0], ay = mesh.vy[a1] - mesh.vy[a0];
        const double bx = mesh.vx[b1] - mesh.vx[b0], by = mesh.vy[b1] - mesh.vy[b0];
        const double la = std::hypot(ax, ay), lb = std::hypot(bx, by);
        if (la <= 0.0 || lb <= 0.0)
            throw std::invalid_argument(
                "periodic pair " + std::to_string(k) + " has a zero-length edge");
        if (std::abs(la - lb) > 1e-8 * la)
            throw std::invalid_argument(
                "periodic pair " + std::to_string(k) + " joins edges of different "
                "length (" + std::to_string(la) + " and " + std::to_string(lb) +
                "); the two sides must be congruent, and their vertices must "
                "correspond in the order given");

        const double rc = (ax * bx + ay * by) / (la * lb);
        const double rs = (bx * ay - by * ax) / (la * lb);
        auto fwd = [&](double x, double y, double& ox, double& oy) {
            const double px = x - mesh.vx[b0], py = y - mesh.vy[b0];
            ox = mesh.vx[a0] + rc * px - rs * py;
            oy = mesh.vy[a0] + rs * px + rc * py;
        };
        auto inv = [&](double x, double y, double& ox, double& oy) {
            const double px = x - mesh.vx[a0], py = y - mesh.vy[a0];
            ox = mesh.vx[b0] + rc * px + rs * py;
            oy = mesh.vy[b0] - rs * px + rc * py;
        };

        double qx, qy, rx, ry;
        fwd(cell_cx[cb], cell_cy[cb], qx, qy);   // B's centroid, seen from A
        inv(cell_cx[ca], cell_cy[ca], rx, ry);   // A's centroid, seen from B

        FaceUnstructured2D face;
        face.c0     = ca;
        face.c1     = cb;
        face.length = la;
        face.bc_tag = -1;
        fill_face_geometry(face, mesh.vx[a0], mesh.vy[a0], mesh.vx[a1], mesh.vy[a1],
                           cell_cx[ca], cell_cy[ca], qx, qy, /*boundary=*/false);
        face.d1x = rx - cell_cx[cb];
        face.d1y = ry - cell_cy[cb];
        face.rot_cos = rc;
        face.rot_sin = rs;

        const double mx = 0.5 * (mesh.vx[a0] + mesh.vx[a1]);
        const double my = 0.5 * (mesh.vy[a0] + mesh.vy[a1]);
        const double da = std::hypot(cell_cx[ca] - mx, cell_cy[ca] - my);
        const double db = std::hypot(qx - mx, qy - my);
        face.w0 = (da + db > 0.0) ? db / (da + db) : 0.5;

        const int fidx = static_cast<int>(faces.size());
        faces.push_back(face);
        cell_faces[ca].push_back(fidx);
        cell_faces[cb].push_back(fidx);
    }

    // Sort before emitting: unordered_map iteration order differs between
    // standard-library implementations, and each boundary face is summed into the
    // diagonal, so an unsorted walk changes results in the last bits per platform.
    std::vector<std::pair<EdgeKey, int>> boundary_edges;  // (edge, owning cell)
    for (const auto& entry : edge_map)
        if (entry.second.cell != kEdgeClosed)
            boundary_edges.emplace_back(entry.first, entry.second.cell);
    std::sort(boundary_edges.begin(), boundary_edges.end(),
              [](const std::pair<EdgeKey, int>& a,
                 const std::pair<EdgeKey, int>& b) {
                  if (a.second != b.second) return a.second < b.second;
                  return a.first < b.first;
              });

    check_conforming(mesh, boundary_edges);

    for (const auto& [key, c0] : boundary_edges) {
        const int vlo = key.first, vhi = key.second;

        const double fx0 = mesh.vx[vlo], fy0 = mesh.vy[vlo];
        const double fx1 = mesh.vx[vhi], fy1 = mesh.vy[vhi];

        const double mx  = 0.5 * (fx0 + fx1);
        const double my  = 0.5 * (fy0 + fy1);

        auto bit = bface_map.find(key);
        const int bc_tag = (bit != bface_map.end()) ? bit->second : 0;

        FaceUnstructured2D face;
        face.c0     = c0;
        face.c1     = -1;
        face.length = std::hypot(fx1 - fx0, fy1 - fy0);
        face.bc_tag = bc_tag;
        face.w0     = 1.0;
        fill_face_geometry(face, fx0, fy0, fx1, fy1,
                           cell_cx[c0], cell_cy[c0], mx, my,
                           /*boundary=*/true);

        const int fidx = static_cast<int>(faces.size());
        faces.push_back(face);
        cell_faces[c0].push_back(fidx);
    }

    build_lsq_gradients(n_cells, faces, cell_faces);
}

// ============================================================================
// Boundary-condition validation
//
// `bc` is indexed bc[tag*groups + g], so the tag count is implied by the array
// length rather than given.  A tag the array does not reach contributes nothing
// to the diagonal, which is indistinguishable from a reflective boundary, so the
// length has to be checked against the tags the mesh actually uses.
// ============================================================================

void validate_bc_unstructured(
    const std::vector<BoundaryCondition>&  bc,
    int                                    groups,
    const std::vector<FaceUnstructured2D>& faces
) {
    if (bc.empty())
        throw std::invalid_argument(
            "bc must not be empty: the unstructured solvers index it as "
            "bc[tag * n_groups + g], so it needs n_bc_types * n_groups entries "
            "(one Robin boundary condition per boundary tag per energy group)");

    if (static_cast<int>(bc.size()) % groups != 0)
        throw std::invalid_argument(
            "bc has " + std::to_string(bc.size()) + " entries, which is not a "
            "multiple of n_groups = " + std::to_string(groups) + "; the layout "
            "is bc[tag * n_groups + g], so the length must be "
            "n_bc_types * n_groups");

    const int n_bc_types = static_cast<int>(bc.size()) / groups;

    int max_tag = -1;
    int min_tag = 0;
    for (const auto& f : faces) {
        if (f.c1 >= 0) continue;               // interior face
        max_tag = std::max(max_tag, f.bc_tag);
        min_tag = std::min(min_tag, f.bc_tag);
    }

    if (min_tag < 0)
        throw std::invalid_argument(
            "mesh.bface_bc_tag contains a negative boundary tag; tags index the "
            "bc array and must lie in [0, n_bc_types)");

    if (max_tag >= n_bc_types)
        throw std::invalid_argument(
            "the mesh uses boundary tag " + std::to_string(max_tag) +
            " but bc only supplies " + std::to_string(n_bc_types) +
            " tag(s) (" + std::to_string(bc.size()) + " entries / n_groups = " +
            std::to_string(groups) + "); bc needs n_bc_types * n_groups entries, "
            "indexed bc[tag * n_groups + g]");
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

// True when every face's correction vector is negligible, i.e. the centroid line
// is parallel to the face normal everywhere.  Regular quad grids satisfy this
// exactly, so they skip the gradient reconstruction and cost nothing.
bool faces_are_orthogonal(const std::vector<FaceUnstructured2D>& faces) {
    for (const auto& f : faces) {
        const double t = std::hypot(f.tx, f.ty);
        if (t > 1e-10 * f.length) return false;
    }
    return true;
}

// Cell gradients for every energy group, from the precomputed least-squares
// coefficients:
//
//   grad phi_P = sum_{faces f of P} lsq_f * (phi_neighbor - phi_P)
//
// Boundary faces contribute their Robin surface value,
// phi_s = phi_P (B/d)/(A + B/d) - the same linear extrapolation the diagonal
// uses - sampled at the face midpoint.
//
// Output layout matches the flux: gx[g * n_cells + c].
void cell_gradients(
    const std::vector<BoundaryCondition>&  bc,
    int groups, int n_cells,
    const std::vector<FaceUnstructured2D>& faces,
    const std::vector<double>&             phi,
    std::vector<double>&                   gx,
    std::vector<double>&                   gy
) {
    gx.assign(static_cast<std::size_t>(groups) * n_cells, 0.0);
    gy.assign(static_cast<std::size_t>(groups) * n_cells, 0.0);
    const int n_bc_types = static_cast<int>(bc.size()) / groups;

    for (int g = 0; g < groups; ++g) {
        const int base = g * n_cells;
        for (const auto& f : faces) {
            if (f.c1 >= 0) {
                const double d0 = phi[base + f.c1] - phi[base + f.c0];
                gx[base + f.c0] += f.lsq0x * d0;
                gy[base + f.c0] += f.lsq0y * d0;
                gx[base + f.c1] -= f.lsq1x * d0;
                gy[base + f.c1] -= f.lsq1y * d0;
            } else {
                const int tag = f.bc_tag;
                if (tag < 0 || tag >= n_bc_types) continue;
                const BoundaryCondition& b = bc[tag * groups + g];
                const double bd  = (f.dist > 0.0) ? b.B / f.dist : 0.0;
                const double den = b.A + bd;
                // phi_s - phi_P = -phi_P * A / (A + B/d)
                const double delta = (std::abs(den) > 1e-30)
                    ? -phi[base + f.c0] * b.A / den : 0.0;
                gx[base + f.c0] += f.lsq0x * delta;
                gy[base + f.c0] += f.lsq0y * delta;
            }
        }
    }
}

// Deferred non-orthogonal correction entering cell `c`'s right-hand side from
// face `f`: D_f (grad phi)_f . T, signed by which side of the face `c` is on.
// Zero on an orthogonal mesh, where T vanishes.
inline double non_orthogonal_correction(
    const FaceUnstructured2D& f, int c, int g, int n_cells,
    const Materials& mats, const std::vector<int>& material_id,
    const std::vector<double>& gx, const std::vector<double>& gy
) {
    if (f.tx == 0.0 && f.ty == 0.0) return 0.0;
    const int base = g * n_cells;

    double gfx, gfy, d_face;
    if (f.c1 >= 0) {
        // Both gradients must be in one frame before they can be averaged; the
        // face's correction vector T lives in c0's, so rotate c1's to match.
        // Identity except across a rotationally periodic face.
        const double g1x = f.rot_cos * gx[base + f.c1] - f.rot_sin * gy[base + f.c1];
        const double g1y = f.rot_sin * gx[base + f.c1] + f.rot_cos * gy[base + f.c1];
        gfx = f.w0 * gx[base + f.c0] + (1.0 - f.w0) * g1x;
        gfy = f.w0 * gy[base + f.c0] + (1.0 - f.w0) * g1y;
        d_face = d_harm(mats.d(material_id[f.c0], g), mats.d(material_id[f.c1], g));
    } else {
        gfx = gx[base + f.c0];
        gfy = gy[base + f.c0];
        d_face = mats.d(material_id[f.c0], g);
    }
    const double sign = (f.c0 == c) ? 1.0 : -1.0;
    return sign * d_face * (gfx * f.tx + gfy * f.ty);
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
    validate_bc_unstructured(bc_, groups_, faces_);
    build_diagonals();
}

void KEigenSolverUnstructured2D::preprocess_mesh() {
    ::preprocess_mesh(mesh_, n_cells_, cell_area_, cell_cx_, cell_cy_,
                      faces_, cell_faces_);
    orthogonal_ = faces_are_orthogonal(faces_);
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
    std::vector<double> phi_prev, gx, gy;
    for (int inner = 0; inner < max_inner_; ++inner) {
        check_interrupt();
        phi_prev = phi;
        if (!orthogonal_)
            cell_gradients(bc_, groups_, n_cells_, faces_, phi, gx, gy);

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

                // Interior face neighbor contributions.
                for (int fi : cell_faces_[c]) {
                    const FaceUnstructured2D& f = faces_[fi];
                    if (f.c1 < 0) continue;
                    const int nbr = (f.c0 == c) ? f.c1 : f.c0;
                    const double D0 = mats_.d(mesh_.material_id[c],   g);
                    const double D1 = mats_.d(mesh_.material_id[nbr], g);
                    rhs += d_harm(D0, D1) * f.a_coef * phi[g * n_cells_ + nbr];
                }

                if (!orthogonal_)
                    for (int fi : cell_faces_[c])
                        rhs += non_orthogonal_correction(
                            faces_[fi], c, g, n_cells_, mats_,
                            mesh_.material_id, gx, gy);

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

    std::vector<double> rhs_g(n_cells_), x_g(n_cells_), phi_prev, gx, gy;
    const int    max_cg = 2 * n_cells_ + 50;
    const double cg_tol = std::min(epsilon_ * 1e-2, 1e-9);

    for (int sweep = 0; sweep < max_inner_; ++sweep) {
        check_interrupt();
        phi_prev = phi;
        bool cg_all_ok = true;
        if (!orthogonal_)
            cell_gradients(bc_, groups_, n_cells_, faces_, phi, gx, gy);

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
                if (!orthogonal_)
                    for (int fi : cell_faces_[c])
                        r += non_orthogonal_correction(
                            faces_[fi], c, g, n_cells_, mats_,
                            mesh_.material_id, gx, gy);
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

        // A single group has no scatter coupling, so one sweep is enough - unless
        // the non-orthogonal correction is active, which is explicit and has to
        // be iterated to consistency like any deferred correction.
        // Report failure if a within-group CG stalled so the caller can warn.
        if ((groups_ == 1 && orthogonal_) ||
            rel_l2_diff(phi, phi_prev) < epsilon_ * 1e-3)
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
    double epsilon, int max_inner, bool verbose,
    DelayedNeutronData             delayed,
    std::vector<double>            initial_precursors,
    double theta
):
      mats_      (std::move(mats)),
      mesh_      (std::move(mesh)),
      bc_        (std::move(bc)),
      epsilon_   (epsilon),
      max_inner_ (max_inner),
      verbose_   (verbose),
      delayed_   (std::move(delayed)),
      theta_     (theta),
      n_cells_   (0),
      groups_    (mats_.n_groups),
      time_      (0.0),
      steps_     (0),
      chi_eff_dt_   (-1.0),
      chi_eff_theta_(-1.0),
      warned_    (false)
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
    validate_delayed(mats_, delayed_);
    validate_material_ids(mesh_.material_id, mats_.n_mat, "material_id");
    validate_bc_unstructured(bc_, groups_, faces_);
    validate_theta(theta_);
    build_diagonals();
    // chi_eff at dt = 0 is the prompt spectrum (1-beta) chi_p.
    prompt_mats_ = build_chi_effective(mats_, delayed_, 0.0);

    phi_.assign(groups_ * n_cells_, 0.0);
    if (!initial_flux.empty()) {
        if (static_cast<int>(initial_flux.size()) != n_cells_ * groups_)
            throw std::invalid_argument(
                "initial_flux must have n_cells * n_groups elements");
        for (int g = 0; g < groups_; ++g)
            for (int c = 0; c < n_cells_; ++c)
                phi_[g * n_cells_ + c] = initial_flux[c * groups_ + g];
    }

    init_precursors(initial_precursors);
}

// ============================================================================
// TimeDependentSolverUnstructured2D - kinetics state helpers
// ============================================================================

void TimeDependentSolverUnstructured2D::init_precursors(
    const std::vector<double>& initial_precursors
) {
    const int I = delayed_.n_precursor;
    if (!initial_precursors.empty()) {
        if (static_cast<int>(initial_precursors.size()) != n_cells_ * I)
            throw std::invalid_argument(
                "initial_precursors must have n_cells * n_precursor elements");
        precursors_ = initial_precursors;
        return;
    }
    // Production rate is per unit volume - no cell-area weight here.
    std::vector<double> production;
    accumulate_production(mats_, mesh_.material_id, groups_, n_cells_, n_cells_,
                          phi_, production);
    equilibrium_precursors(delayed_, mesh_.material_id, n_cells_, production,
                           precursors_);
}

void TimeDependentSolverUnstructured2D::refresh_chi_effective(double dt) {
    if (dt == chi_eff_dt_ && theta_ == chi_eff_theta_) return;
    // The delayed neutrons emitted within the step are weighted by theta*dt, not
    // dt - see the derivation in solver_detail.hpp.
    chi_eff_mats_  = build_chi_effective(mats_, delayed_, theta_ * dt);
    chi_eff_dt_    = dt;
    chi_eff_theta_ = theta_;
}

void TimeDependentSolverUnstructured2D::set_theta(double theta) {
    validate_theta(theta);
    theta_ = theta;
}

void TimeDependentSolverUnstructured2D::update_materials(Materials mats) {
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
    build_diagonals();
    prompt_mats_ = build_chi_effective(mats_, delayed_, 0.0);
    chi_eff_dt_  = -1.0;  // invalidate the cached effective spectrum
}

void TimeDependentSolverUnstructured2D::preprocess_mesh() {
    ::preprocess_mesh(mesh_, n_cells_, cell_area_, cell_cx_, cell_cy_,
                      faces_, cell_faces_);
    orthogonal_ = faces_are_orthogonal(faces_);
}

void TimeDependentSolverUnstructured2D::build_diagonals() {
    ::build_diagonals(mats_, mesh_, bc_, n_cells_, groups_,
                      cell_area_, faces_, a_diag_base_);
}

// ============================================================================
// TimeDependentSolverUnstructured2D - explicit residual for the theta method
// ============================================================================
//
// E = -A phi_old + in-scatter(phi_old) + prompt fission(phi_old), volume
// integrated like the rest of the FVM right-hand side, so the scatter term
// carries the cell area and the fission source is assembled with the cell-area
// weight.  Boundary faces are already absorbed into a_diag_base_, so only
// interior faces contribute off-diagonal terms.  On a non-orthogonal mesh the
// deferred correction is part of A too and has to appear here as it does in
// solve_step; otherwise the two halves of the step use different operators and
// a critical system is no longer a fixed point for theta < 1.

void TimeDependentSolverUnstructured2D::explicit_residual(
    const std::vector<double>& phi_old,
    std::vector<double>& out
) const {
    std::vector<double> fis_prompt, gx, gy;
    accumulate_fission(prompt_mats_, mesh_.material_id, groups_,
                       n_cells_, n_cells_, &cell_area_, phi_old, fis_prompt);
    if (!orthogonal_)
        cell_gradients(bc_, groups_, n_cells_, faces_, phi_old, gx, gy);

    out.assign(static_cast<std::size_t>(groups_) * n_cells_, 0.0);
    for (int c = 0; c < n_cells_; ++c) {
        const int    mat  = mesh_.material_id[c];
        const double area = cell_area_[c];

        for (int g = 0; g < groups_; ++g) {
            double e = -a_diag_base_[g * n_cells_ + c] * phi_old[g * n_cells_ + c]
                       + fis_prompt[g * n_cells_ + c];

            for (int gp = 0; gp < groups_; ++gp)
                if (gp != g)
                    e += mats_.sig_s(mat, g, gp) * phi_old[gp * n_cells_ + c] * area;

            for (int fi : cell_faces_[c]) {
                const FaceUnstructured2D& f = faces_[fi];
                if (f.c1 < 0) continue;
                const int nbr = (f.c0 == c) ? f.c1 : f.c0;
                const double D0 = mats_.d(mesh_.material_id[c],   g);
                const double D1 = mats_.d(mesh_.material_id[nbr], g);
                e += d_harm(D0, D1) * f.a_coef * phi_old[g * n_cells_ + nbr];
            }

            if (!orthogonal_)
                for (int fi : cell_faces_[c])
                    e += non_orthogonal_correction(
                        faces_[fi], c, g, n_cells_, mats_,
                        mesh_.material_id, gx, gy);

            out[g * n_cells_ + c] = e;
        }
    }
}

// ============================================================================
// TimeDependentSolverUnstructured2D - one theta-weighted step
// ============================================================================

void TimeDependentSolverUnstructured2D::solve_step(
    const std::vector<double>& phi_old,
    const std::vector<double>& qd,
    const std::vector<double>& expl,
    double dt
) {
    std::vector<double> phi_iter, fis, gx, gy;
    const double ex_weight = (1.0 - theta_) / theta_;
    double residual  = 0.0;
    bool   converged = false;
    FissionAccelerator accel;

    for (int inner = 0; inner < max_inner_; ++inner) {
        phi_iter = phi_;
        if (!orthogonal_)
            cell_gradients(bc_, groups_, n_cells_, faces_, phi_, gx, gy);

        // Implicit fission source from the latest iterate.  The FVM equations
        // are volume-integrated, so this one carries the cell-area weight.
        accumulate_fission(chi_eff_mats_, mesh_.material_id, groups_,
                           n_cells_, n_cells_, &cell_area_, phi_, fis);

        for (int c = 0; c < n_cells_; ++c) {
            const int    mat  = mesh_.material_id[c];
            const double area = cell_area_[c];

            for (int g = 0; g < groups_; ++g) {
                const double inv_v_dt = 1.0 / (mats_.v(g) * theta_ * dt);
                const double diag = a_diag_base_[g * n_cells_ + c] + inv_v_dt * area;

                double rhs = inv_v_dt * phi_old[g * n_cells_ + c] * area
                           + fis[g * n_cells_ + c]   // fission (implicit)
                           + qd [g * n_cells_ + c];  // delayed (C^n, F^n)

                if (theta_ < 1.0)
                    rhs += ex_weight * expl[g * n_cells_ + c];  // explicit residual

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

                if (!orthogonal_)
                    for (int fi : cell_faces_[c])
                        rhs += non_orthogonal_correction(
                            faces_[fi], c, g, n_cells_, mats_,
                            mesh_.material_id, gx, gy);

                phi_[g * n_cells_ + c] = rhs / diag;
            }
        }

        // Relative criterion - the physical flux magnitude can be large.
        residual = rel_l2_diff(phi_, phi_iter);
        if (residual < epsilon_) { converged = true; break; }

        accel.accelerate(phi_, phi_iter);
    }

    if (!converged && !warned_) {
        warn_step_not_converged("TimeDependentSolverUnstructured2D",
                                max_inner_, dt, residual);
        warned_ = true;
    }
}

void TimeDependentSolverUnstructured2D::step(double dt) {
    validate_dt(dt);

    const std::vector<double> phi_old = phi_;

    refresh_chi_effective(dt);

    // Production rate of the old flux - pointwise, so unweighted.  Needed by the
    // theta-weighted delayed terms and reused for the precursor advance below.
    std::vector<double> production_old;
    if (theta_ < 1.0 && !delayed_.empty())
        accumulate_production(mats_, mesh_.material_id, groups_,
                              n_cells_, n_cells_, phi_old, production_old);

    // Delayed source from the old kinetics state.  Precursors are stored per
    // unit volume, so the cell area is applied here, where Q_d enters the
    // volume-integrated FVM right-hand side.
    std::vector<double> qd;
    accumulate_delayed_source(delayed_, mesh_.material_id, groups_,
                              n_cells_, n_cells_, dt, theta_, &cell_area_,
                              precursors_, production_old, qd);

    // Explicit half of the theta weighting.  Skipped entirely at theta = 1.
    std::vector<double> expl;
    if (theta_ < 1.0) explicit_residual(phi_old, expl);

    solve_step(phi_old, qd, expl, dt);

    // Advance the precursors from the new flux.  The production rate is a
    // pointwise quantity - unweighted, unlike the fission source above.
    if (!delayed_.empty()) {
        std::vector<double> production;
        accumulate_production(mats_, mesh_.material_id, groups_,
                              n_cells_, n_cells_, phi_, production);
        update_precursors(delayed_, mesh_.material_id, n_cells_, dt, theta_,
                          production, production_old, precursors_);
    }

    time_  += dt;
    steps_ += 1;

    if (verbose_)
        std::printf("t = %.6e s  step %d  phi_max = %.6e\n",
                    time_, steps_,
                    *std::max_element(phi_.begin(), phi_.end()));
}

TimeDependentResult TimeDependentSolverUnstructured2D::run(double dt, int n_steps) {
    for (int n = 0; n < n_steps; ++n) {
        check_interrupt();
        step(dt);
    }
    return result();
}

TimeDependentResult TimeDependentSolverUnstructured2D::result() const {
    std::vector<double> flux_out;
    pack_flux(phi_, n_cells_, groups_, n_cells_, flux_out);
    return {flux_out, time_, steps_, precursors_};
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
    validate_bc_unstructured(bc_, groups_, faces_);
    build_diagonals();
}

void FixedSourceSolverUnstructured2D::preprocess_mesh() {
    ::preprocess_mesh(mesh_, n_cells_, cell_area_, cell_cx_, cell_cy_,
                      faces_, cell_faces_);
    orthogonal_ = faces_are_orthogonal(faces_);
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
    int    iter     = 0;   // sweeps performed
    std::vector<double> gx, gy;

    while (iter < max_inner_) {
        check_interrupt();
        ++iter;
        phi_prev = phi;
        if (!orthogonal_)
            cell_gradients(bc_, groups_, n_cells_, faces_, phi, gx, gy);

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

                // Interior face neighbor contributions.
                // Boundary faces (c1 < 0) are already absorbed into a_diag_base_.
                for (int fi : cell_faces_[c]) {
                    const FaceUnstructured2D& f = faces_[fi];
                    if (f.c1 < 0) continue;
                    const int nbr = (f.c0 == c) ? f.c1 : f.c0;
                    const double D0 = mats_.d(mesh_.material_id[c],   g);
                    const double D1 = mats_.d(mesh_.material_id[nbr], g);
                    rhs += d_harm(D0, D1) * f.a_coef * phi[g * n_cells_ + nbr];
                }

                if (!orthogonal_)
                    for (int fi : cell_faces_[c])
                        rhs += non_orthogonal_correction(
                            faces_[fi], c, g, n_cells_, mats_,
                            mesh_.material_id, gx, gy);

                const double phi_gs = rhs / a_diag_base_[g * n_cells_ + c];
                phi[g * n_cells_ + c] = (1.0 - omega_) * phi[g * n_cells_ + c]
                                       + omega_ * phi_gs;
            }
        }

        residual = rel_l2_diff(phi, phi_prev);

        if (verbose_)
            std::printf("Iter: %3d  residual: %.2e\n", iter, residual);

        if (residual < epsilon_)
            break;
    }

    std::vector<double> flux_out;
    pack_flux(phi, n_cells_, groups_, n_cells_, flux_out);
    return {flux_out, iter, residual, residual < epsilon_};
}
