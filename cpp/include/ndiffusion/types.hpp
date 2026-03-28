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
 *    Required by TimeDependentSolver; unused by DiffusionSolver.
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
    std::vector<double> nusigf;    ///< ν·Σ_f                  [n_mat * n_groups]
    std::vector<double> velocity;  ///< Neutron speed (cm/s)   [n_groups]

    /// Diffusion coefficient for material @p m, group @p g.
    double d      (int m, int g)                const { return D      [m * n_groups + g]; }
    /// Removal cross section for material @p m, group @p g.
    double sig_r  (int m, int g)                const { return removal[m * n_groups + g]; }
    /**
     * @brief Scattering cross section in material @p m from group @p g_from
     *        into group @p g_to.
     */
    double sig_s  (int m, int g_to, int g_from) const {
        return scatter[(m * n_groups + g_to) * n_groups + g_from];
    }
    /// Fission spectrum for material @p m, group @p g.
    double chi_g  (int m, int g)                const { return chi   [m * n_groups + g]; }
    /// ν·Σ_f for material @p m, group @p g.
    double nu_sigf(int m, int g)                const { return nusigf[m * n_groups + g]; }
    /// Average neutron speed (cm/s) for group @p g.
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
 * | Marshak vacuum  | (1−α)/(4(1+α))              | D/2   |
 * | Reflective      | 0.0                         | 1.0   |
 *
 * One `BoundaryCondition` is required per energy group.
 */
struct BoundaryCondition {
    double A; ///< Coefficient of φ
    double B; ///< Coefficient of dφ/dx
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
 * @brief Output snapshot from the time-dependent solver.
 */
struct TimeDependentResult {
    std::vector<double> flux;  ///< Physical flux [cells * n_groups], row-major: flux[i*G+g]
    double time;               ///< Total elapsed simulated time (s)
    int    steps;              ///< Number of time steps taken
};
