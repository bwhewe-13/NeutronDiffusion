#pragma once

#include <vector>

/**
 * @file types.hpp
 * @brief Shared types for the ndiffusion solver library.
 *
 * Defines geometry, cross-section, boundary condition, and result types
 * used by all dimensionalities (1D, 2D, 3D) of the neutron diffusion solvers.
 */

// ============================================================================
// Geometry
// ============================================================================

/// Coordinate system for 1-D radial/slab problems.
enum class Geometry {
    Slab,     ///< Cartesian slab (x from 0 to R)
    Cylinder, ///< Infinite cylinder (r from 0 to R)
    Sphere    ///< Sphere (r from 0 to R)
};

/// Coordinate system for 2-D structured mesh problems.
enum class Geometry2D {
    XY, ///< Cartesian 2-D (x, y)
    RZ  ///< Axisymmetric cylindrical (z axial = x-axis, r radial = y-axis)
};

// ============================================================================
// Cross-section data
// ============================================================================

/**
 * @brief Cross-section data for all materials and energy groups.
 *
 * All multi-dimensional arrays are stored flat in row-major order.
 * Accessor methods provide convenient indexed access.
 *
 * @par Array layouts
 *  - `D`, `removal`, `chi`, `nusigf`: `[n_mat * n_groups]`
 *  - `scatter`: `[n_mat * n_groups * n_groups]`
 *    where `scatter[m][g_to][g_from]` is the scattering cross section
 *    that transfers neutrons **from** group `g_from` **into** group `g_to`
 *    in material `m`.
 *  - `velocity`: `[n_groups]` — average neutron speed (cm/s) per group.
 *    Required by TimeDependentSolver; unused by KEigenSolver.
 *
 * @note numpy arrays are automatically converted to `std::vector<double>`
 *       by the pybind11 bindings.
 */
struct Materials {
    int n_mat;    ///< Number of distinct materials
    int n_groups; ///< Number of energy groups

    std::vector<double> D;         ///< Diffusion coefficients  [n_mat * n_groups]
    std::vector<double> removal;   ///< Removal cross sections  [n_mat * n_groups]
    std::vector<double> scatter;   ///< Scatter cross sections  [n_mat * n_groups * n_groups]
    std::vector<double> chi;       ///< Fission spectrum        [n_mat * n_groups]
    std::vector<double> nusigf;    ///< ν·Σ_f  [n_mat * n_groups] (standard mode)
                                   ///<   or fission transfer matrix F[g_to][g_from]
                                   ///<   [n_mat * n_groups * n_groups] when chi is all zeros
    std::vector<double> velocity;  ///< Neutron speed (cm/s)   [n_groups]

    /**
     * @brief Diffusion coefficient for a material and energy group.
     *
     * @param m Material index.
     * @param g Energy-group index.
     * @return Diffusion coefficient for material @p m and group @p g.
     */
    double d      (int m, int g)                const { return D      [m * n_groups + g]; }
    /**
     * @brief Removal cross section for a material and energy group.
     *
     * @param m Material index.
     * @param g Energy-group index.
     * @return Removal cross section for material @p m and group @p g.
     */
    double sig_r  (int m, int g)                const { return removal[m * n_groups + g]; }
    /**
     * @brief Scattering cross section between two energy groups.
     *
     * @param m Material index.
     * @param g_to Destination energy-group index.
     * @param g_from Source energy-group index.
     * @return Scattering cross section in material @p m from group @p g_from
     *         into group @p g_to.
     */
    double sig_s  (int m, int g_to, int g_from) const {
        return scatter[(m * n_groups + g_to) * n_groups + g_from];
    }
    /**
     * @brief Fission spectrum entry for a material and energy group.
     *
     * @param m Material index.
     * @param g Energy-group index.
     * @return Fission spectrum value for material @p m and group @p g.
     */
    double chi_g  (int m, int g)                const { return chi   [m * n_groups + g]; }
    /**
     * @brief Standard-mode fission production cross section.
     *
     * @param m Material index.
     * @param g Energy-group index.
     * @return ν·Σ_f for material @p m and group @p g.
     */
    double nu_sigf(int m, int g)                const { return nusigf[m * n_groups + g]; }
    /**
     * @brief Fission transfer matrix entry in matrix mode.
     *
     * @param m Material index.
     * @param g_to Destination energy-group index for emitted neutrons.
     * @param g_from Source energy-group index causing fission.
     * @return Fission transfer matrix entry for material @p m. Only valid when
     *         use_fission_matrix() returns true.
     */
    double nu_sigf_mat(int m, int g_to, int g_from) const {
        return nusigf[(m * n_groups + g_to) * n_groups + g_from];
    }
    /**
     * @brief Report whether `nusigf` stores a full fission transfer matrix.
     *
     * @return True when `chi` is all zeros, meaning `nusigf` holds
     *         `F[g_to][g_from]` instead of the standard group-vector form.
     */
    bool use_fission_matrix() const {
        for (double v : chi) if (v != 0.0) return false;
        return true;
    }
    /**
     * @brief Average neutron speed for an energy group.
     *
     * @param g Energy-group index.
     * @return Average neutron speed (cm/s) for group @p g.
     */
    double v      (int g)                       const { return velocity[g]; }
};

// ============================================================================
// Boundary conditions
// ============================================================================

/**
 * @brief Robin boundary condition at the outer surface.
 *
 * Encodes the condition:
 * @code
 *   A · φ + B · (dφ/dx) = 0
 * @endcode
 *
 * | Type            | A                           | B     |
 * |-----------------|-----------------------------|-------|
 * | Zero-flux       | 1.0                         | 0.0   |
 * | Marshak vacuum  | (1−alpha)/(4(1+alpha))      | D/2   |
 * | Reflective      | 0.0                         | 1.0   |
 *
 * One `BoundaryCondition` is required per energy group.
 */
struct BoundaryCondition {
    double A; ///< Coefficient of phi
    double B; ///< Coefficient of dphi/dx
};

// ============================================================================
// Results
// ============================================================================

/**
 * @brief Output from a completed k-eigenvalue solve.
 */
struct DiffusionResult {
    std::vector<double> flux;  ///< Physical flux [cells * n_groups], row-major: flux[i*G+g]
    double keff;               ///< Effective multiplication factor
    int    iterations;         ///< Power-iteration count
    double residual;           ///< Final flux change norm (convergence indicator)
};

/**
 * @brief Output from a completed fixed-source solve.
 */
struct FixedSourceResult {
    std::vector<double> flux;  ///< Physical flux [cells * n_groups], row-major: flux[i*G+g]
    int    iterations;         ///< Gauss-Seidel iteration count
    double residual;           ///< Final flux change norm (convergence indicator)
};

/**
 * @brief Output snapshot from the time-dependent solver.
 */
struct TimeDependentResult {
    std::vector<double> flux;  ///< Physical flux [cells * n_groups], row-major: flux[i*G+g]
    double time;               ///< Total elapsed simulated time (s)
    int    steps;              ///< Number of time steps taken
};

// ============================================================================
// Unstructured mesh
// ============================================================================

// ============================================================================
// Unstructured face (shared by both unstructured 2-D solvers)
// ============================================================================

/**
 * @brief Preprocessed face data for the unstructured FVM solver.
 *
 * Interior faces have `c1 >= 0`.  Boundary faces have `c1 = -1`.
 */
struct FaceUnstructured2D {
    int    c0;      ///< First (or only) cell index
    int    c1;      ///< Second cell index, or -1 for boundary faces
    double length;  ///< Face length
    double dist;    ///< Centroid-to-centroid (interior) or centroid-to-face distance (boundary)
    double a_coef;  ///< Geometry factor: length / dist  (D-independent)
    int    bc_tag;  ///< BC tag for boundary faces; -1 for interior faces
};

/**
 * @brief 2-D unstructured mesh of triangles and/or quadrilaterals.
 *
 * Cells are defined by vertex coordinates and a flat connectivity list.
 * Cell @p c owns vertices `cell_vertices[cell_offsets[c] .. cell_offsets[c+1])`.
 * A cell with 3 vertices is a triangle; 4 vertices is a quadrilateral.
 *
 * Boundary faces are specified as pairs of vertex indices.  Any boundary face
 * not listed in `bface_v0/bface_v1` is assigned `bc_tag = 0` by default.
 */
struct UnstructuredMesh2D {
    std::vector<double> vx;            ///< Vertex x-coordinates [n_verts]
    std::vector<double> vy;            ///< Vertex y-coordinates [n_verts]
    std::vector<int>    cell_vertices; ///< Flat vertex-index list for all cells
    std::vector<int>    cell_offsets;  ///< Size n_cells+1; offsets into cell_vertices
    std::vector<int>    material_id;   ///< Material index per cell [n_cells]

    /// First vertex of each user-specified boundary face.
    std::vector<int>    bface_v0;
    /// Second vertex of each user-specified boundary face.
    std::vector<int>    bface_v1;
    /// BC tag for each user-specified boundary face (index into bc array).
    std::vector<int>    bface_bc_tag;
};
