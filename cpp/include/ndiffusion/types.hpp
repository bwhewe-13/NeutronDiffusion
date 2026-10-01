#pragma once

#include <cstddef>
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
 *  - `velocity`: `[n_groups]` - average neutron speed (cm/s) per group.
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
    std::vector<double> nusigf;    ///< nu*Sigma_f  [n_mat * n_groups] (standard mode)
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
     * @return nu*Sigma_f for material @p m and group @p g.
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
     * @return True when `chi` is all zeros and `nusigf` is matrix-sized
     *         (`n_mat * n_groups * n_groups`), meaning `nusigf` holds
     *         `F[g_to][g_from]` instead of the standard group-vector form.
     */
    bool use_fission_matrix() const {
        const std::size_t mat_size =
            static_cast<std::size_t>(n_mat) * n_groups * n_groups;
        if (nusigf.size() != mat_size) return false;
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
// Delayed neutron data
// ============================================================================

/**
 * @brief Delayed neutron precursor data for the time-dependent solvers.
 *
 * Kept separate from Materials because it is kinetics-only: the k-eigenvalue and
 * fixed-source solvers never see it, and a steady-state cross-section library
 * usually carries no precursor data at all.
 *
 * @par Array layouts
 *  - `lambda`:      `[n_precursor]`  decay constants (1/s)
 *  - `beta`:        `[n_mat * n_precursor]`  delayed fractions per material
 *  - `chi_delayed`: `[n_mat * n_precursor * n_groups]`  delayed fission spectrum
 *  - `chi_prompt`:  `[n_mat * n_groups]`, or empty to use `Materials::chi`
 *
 * A default-constructed instance (`n_precursor == 0`) disables delayed neutrons,
 * reducing the time-dependent solvers to prompt-only kinetics.
 *
 * @note Delayed neutrons require the standard `chi` / `nusigf` representation.
 *       Fission-matrix mode (see Materials::use_fission_matrix()) has no
 *       separable fission spectrum to split into prompt and delayed parts, so
 *       the combination is rejected by the solver constructors.
 */
struct DelayedNeutronData {
    int n_precursor = 0;  ///< Number of precursor groups; 0 disables delayed neutrons

    std::vector<double> lambda;       ///< Decay constants (1/s)  [n_precursor]
    std::vector<double> beta;         ///< Delayed fractions      [n_mat * n_precursor]
    std::vector<double> chi_delayed;  ///< Delayed fission spectrum
                                      ///<   [n_mat * n_precursor * n_groups]
    std::vector<double> chi_prompt;   ///< Prompt fission spectrum [n_mat * n_groups];
                                      ///<   empty -> use Materials::chi

    /// @return True when no precursor groups are defined (prompt-only kinetics).
    bool empty() const { return n_precursor == 0; }

    /**
     * @brief Decay constant of a precursor group.
     *
     * @param i Precursor-group index.
     * @return Decay constant (1/s) for precursor group @p i.
     */
    double lam (int i) const { return lambda[i]; }

    /**
     * @brief Delayed fraction for a material and precursor group.
     *
     * @param m Material index.
     * @param i Precursor-group index.
     * @return Delayed neutron fraction beta_i for material @p m.
     */
    double bet (int m, int i) const { return beta[m * n_precursor + i]; }

    /**
     * @brief Delayed fission spectrum entry.
     *
     * @param m Material index.
     * @param i Precursor-group index.
     * @param g Energy-group index.
     * @param n_groups Number of energy groups.
     * @return Fraction of precursor-group @p i decay neutrons born into group @p g.
     */
    double chi_d(int m, int i, int g, int n_groups) const {
        return chi_delayed[(m * n_precursor + i) * n_groups + g];
    }

    /**
     * @brief Total delayed fraction for a material, summed over precursor groups.
     *
     * @param m Material index.
     * @return beta = sum_i beta_i for material @p m.
     */
    double beta_total(int m) const {
        double b = 0.0;
        for (int i = 0; i < n_precursor; ++i) b += bet(m, i);
        return b;
    }
};

// ============================================================================
// Boundary conditions
// ============================================================================

/**
 * @brief Robin boundary condition at the outer surface.
 *
 * Encodes the condition:
 * @code
 *   A * phi + B * (dphi/dx) = 0
 * @endcode
 *
 * | Type            | A                           | B     |
 * |-----------------|-----------------------------|-------|
 * | Zero-flux       | 1.0                         | 0.0   |
 * | Marshak vacuum  | (1-alpha)/(4(1+alpha))      | D/2   |
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
    bool   converged;          ///< True when outer and inner solves both met their tolerances
};

/**
 * @brief Output from a completed fixed-source solve.
 */
struct FixedSourceResult {
    std::vector<double> flux;  ///< Physical flux [cells * n_groups], row-major: flux[i*G+g]
    int    iterations;         ///< Gauss-Seidel iteration count
    double residual;           ///< Final relative flux change norm (convergence indicator)
    bool   converged;          ///< True when the iteration met its tolerance
};

/**
 * @brief Output snapshot from the time-dependent solver.
 */
struct TimeDependentResult {
    std::vector<double> flux;  ///< Physical flux [cells * n_groups], row-major: flux[i*G+g]
    double time;               ///< Total elapsed simulated time (s)
    int    steps;              ///< Number of time steps taken

    /// Delayed neutron precursor concentrations per unit volume,
    /// `[cells * n_precursor]`, row-major: `precursors[i*I+p]`.
    /// Empty when the solver was built without delayed neutron data.
    std::vector<double> precursors;
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
 *
 * @par Non-orthogonal decomposition
 * The diffusion flux through a face is `D (grad phi)_f . S`, with surface vector
 * `S = length * n` pointing out of `c0`.  A two-point difference along the
 * centroid line `e` only captures that when `e` is parallel to `n`.  `S` is
 * therefore split (over-relaxed / Jasak) into a part along `e` and a remainder:
 * @code
 *   E = (S.S)/(e.S) e      T = S - E
 *   (grad phi)_f . S  =  |E| (phi_N - phi_P)/dist  +  (grad phi)_f . T
 * @endcode
 * The first term is implicit and gives `a_coef = |E|/dist`; the second is a
 * deferred correction evaluated from the reconstructed cell gradients.  On an
 * orthogonal mesh `E == S`, so `T` vanishes and `a_coef` reduces to
 * `length/dist` - the uncorrected two-point flux, unchanged.
 */
struct FaceUnstructured2D {
    int    c0;      ///< First (or only) cell index
    int    c1;      ///< Second cell index, or -1 for boundary faces
    double length;  ///< Face length
    double dist;    ///< Centroid-to-centroid (interior) or centroid-to-face normal
                    ///<   distance (boundary)
    double a_coef;  ///< Implicit geometry factor |E|/dist  (D-independent).
                    ///<   Equals length/dist on an orthogonal mesh.
    int    bc_tag;  ///< BC tag for boundary faces; -1 for interior faces

    double sx;      ///< Surface vector x-component, out of c0 (magnitude = length)
    double sy;      ///< Surface vector y-component, out of c0
    double tx;      ///< Non-orthogonal correction vector, x (zero if orthogonal)
    double ty;      ///< Non-orthogonal correction vector, y
    double w0;      ///< Interpolation weight of c0 at the face; c1 gets (1 - w0)

    /// Offset from c0's centroid to where c1's centroid sits *as seen across
    /// this face*, and the mirror offset from c1 back to c0.  For an ordinary
    /// face these are simply +/- (x_c1 - x_c0); across a periodic face they are
    /// the images under the periodic transform, which is what lets everything
    /// downstream treat a periodic face as an ordinary interior one.
    double d0x, d0y;
    double d1x, d1y;

    /// Rotation carrying a vector from c1's frame into c0's, as (cos, sin).
    /// Identity on every face except a rotationally periodic one, where a
    /// gradient has to be turned through the sector angle before it can be
    /// combined with its neighbor's.
    double rot_cos, rot_sin;

    /// Least-squares gradient coefficients.  The gradient of a cell is the sum
    /// over its faces of `lsq * (phi_neighbor - phi_cell)`, using `lsq0` when
    /// the cell is `c0` and `lsq1` when it is `c1`.  Purely geometric, so the
    /// weighted least-squares fit is solved once at construction; unlike a
    /// Green-Gauss reconstruction this stays second-order on triangles.
    double lsq0x, lsq0y;
    double lsq1x, lsq1y;
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

    /// @name Periodic boundary pairs
    ///
    /// Edges listed here are joined to each other instead of being treated as
    /// boundaries: the flux is continuous across the pair, as though the two
    /// sides were adjacent.  Pair `k` joins edge
    /// (`periodic_a0[k]`, `periodic_a1[k]`) to (`periodic_b0[k]`,
    /// `periodic_b1[k]`), **vertex to corresponding vertex** - `a0` maps to `b0`
    /// and `a1` to `b1`.  That correspondence fixes the rigid transform between
    /// the two sides, so a translation (a repeating lattice) and a rotation (a
    /// symmetry sector with no mirror symmetry) are both expressible, and the
    /// solver derives which from the geometry.
    ///
    /// Unlike a reflective condition, this imposes no symmetry of its own, so it
    /// is the right choice for a rotationally symmetric core - a spiral or
    /// pinwheel loading - where a mirror condition would be wrong.
    /// @{
    std::vector<int>    periodic_a0;
    std::vector<int>    periodic_a1;
    std::vector<int>    periodic_b0;
    std::vector<int>    periodic_b1;
    /// @}
};
