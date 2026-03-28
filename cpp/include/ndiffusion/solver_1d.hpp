#pragma once

#include <ndiffusion/types.hpp>
#include <vector>
#include <string>
#include <cstdio>

/**
 * @file solver_1d.hpp
 * @brief 1-D multigroup neutron diffusion solver declarations.
 *
 * Provides two solvers:
 *  - DiffusionSolver  — k-eigenvalue power iteration
 *  - TimeDependentSolver — backward-Euler time stepping
 *
 * Both are matrix-free: only per-group tridiagonal bands are stored.
 * The spatial solve uses Gauss-Seidel over energy groups with Thomas
 * (TDMA) tridiagonal solves inside each sweep.
 */

// ============================================================================
// k-eigenvalue solver
// ============================================================================

/**
 * @brief Matrix-free 1-D multigroup neutron diffusion k-eigenvalue solver.
 *
 * Solves the generalized eigenvalue problem
 * @code
 *   A phi = (1/k) B phi
 * @endcode
 * using power iteration.  The operator @b A is applied implicitly via
 * precomputed per-group tridiagonal bands; no full N×N matrix is ever
 * assembled.  Each power-iteration inner solve uses Gauss-Seidel sweeps
 * over energy groups with a Thomas (TDMA) tridiagonal solve per sweep.
 *
 * @par Geometry
 * A symmetry boundary condition (zero flux gradient) is always enforced
 * at the left edge (r = 0).  The outer Robin BC is specified through the
 * `bc` parameter.
 *
 * @par Interface diffusion coefficients
 * Cell-interface diffusion coefficients use the half harmonic mean:
 * @code
 *   D_eff = D_i * D_j / (D_i + D_j)
 * @endcode
 * which, combined with the `2 / (dx * V) * SA` geometric pre-factor, gives the
 * correct finite-difference leakage coefficient.
 */
class DiffusionSolver {
public:
    /**
     * @brief Construct the solver and precompute tridiagonal bands.
     *
     * @param mats        Cross-section data for all materials.
     * @param medium_map  Material index for each spatial cell (length = cells).
     * @param edges_x     Cell-edge positions (cm), length = cells + 1.
     * @param geom        Geometry type (Slab, Cylinder, or Sphere).
     * @param bc          Outer Robin BC, one entry per energy group.
     * @param epsilon     Convergence tolerance on the flux change norm.
     * @param max_outer   Maximum power-iteration count.
     * @param max_inner   Maximum Gauss-Seidel inner iterations per outer step.
     * @param verbose     Print iteration diagnostics if true.
     *
     * @throws std::invalid_argument if `bc.size() != mats.n_groups`.
     */
    DiffusionSolver(
        Materials                      mats,
        std::vector<int>               medium_map,
        std::vector<double>            edges_x,
        Geometry                       geom,
        std::vector<BoundaryCondition> bc,
        double epsilon   = 1e-8,
        int    max_outer = 200,
        int    max_inner = 50,
        bool   verbose   = true
    );

    /**
     * @brief Run power iteration and return the converged solution.
     * @return DiffusionResult containing flux, keff, iterations, residual.
     */
    DiffusionResult solve();

private:
    /// Fission source operator: b[g*N+i] = χ_g · Σ_gp( νΣ_f,gp · φ_gp[i] )
    void apply_B(const std::vector<double>& phi,
                       std::vector<double>& b) const;

    /// Solve A·φ = b via Gauss-Seidel (groups) + Thomas (spatial).
    void solve_A(const std::vector<double>& b,
                       std::vector<double>& phi) const;

    Materials                      mats_;
    std::vector<int>               medium_map_;
    std::vector<double>            edges_x_;
    std::vector<double>            surface_area_;  ///< length = cells_ + 1
    std::vector<double>            volume_;         ///< length = cells_
    Geometry                       geom_;
    std::vector<BoundaryCondition> bc_;
    double epsilon_;
    int    max_outer_;
    int    max_inner_;
    bool   verbose_;

    int cells_;   ///< Number of physical spatial cells
    int groups_;  ///< Number of energy groups
    int N_;       ///< cells_ + 1 (includes the ghost BC row)

    /// Precomputed tridiagonal bands, length = groups_ * N_, indexed [g*N+i].
    std::vector<double> lower_;
    std::vector<double> diag_;
    std::vector<double> upper_;
};

// ============================================================================
// Time-dependent solver
// ============================================================================

/**
 * @brief 1-D multigroup time-dependent neutron diffusion solver.
 *
 * Advances the prompt-neutron diffusion equation
 * @code
 *   (1/v_g) dphi_g/dt = -A_g phi_g
 *                    + chi_g · sigma_gp( nu_sigma_f,gp · phi_gp )   [fission]
 *                    + sigma_{gpg} sigma_s(ggp) · phi_gp        [scatter]
 * @endcode
 * using **backward Euler** time differencing.
 *
 * @par Time-stepping algorithm
 * 1. Fission source is computed explicitly from phi^n.
 * 2. The time-absorption term `1/(v_g · dt)` is added to the spatial
 *    diagonal, so the modified per-group system is still tridiagonal.
 * 3. A Gauss-Seidel sweep over groups solves the implicit scatter coupling.
 *
 * This scheme is unconditionally stable for the diffusion operator and
 * first-order accurate in time.  Use small dt for physical accuracy.
 *
 * @par Requirements
 * `Materials::velocity` must be set (size = n_groups) before constructing
 * the solver; a `std::invalid_argument` is thrown otherwise.
 *
 * @par Usage
 * @code
 *   TimeDependentSolver tds(mats, medium_map, edges_x, geom, bc,
 *                           initial_flux);
 *   for (int n = 0; n < n_steps; ++n)
 *       tds.step(dt);
 *   TimeDependentResult res = tds.result();
 * @endcode
 */
class TimeDependentSolver {
public:
    /**
     * @brief Construct the time-dependent solver.
     *
     * @param mats          Cross-section data including `velocity` [n_groups].
     * @param medium_map    Material index for each cell (length = cells).
     * @param edges_x       Cell-edge positions (cm), length = cells + 1.
     * @param geom          Geometry type (Slab, Cylinder, or Sphere).
     * @param bc            Outer Robin BC, one entry per energy group.
     * @param initial_flux  Starting flux [cells * n_groups], row-major.
     *                      Defaults to zero if empty.
     * @param epsilon       Convergence tolerance for the Gauss-Seidel inner loop.
     * @param max_inner     Maximum Gauss-Seidel inner iterations per time step.
     * @param verbose       Print step diagnostics if true.
     *
     * @throws std::invalid_argument if `bc.size() != mats.n_groups`,
     *         `mats.velocity.size() != mats.n_groups`, or
     *         `initial_flux.size() != cells * n_groups` (when non-empty).
     */
    TimeDependentSolver(
        Materials                      mats,
        std::vector<int>               medium_map,
        std::vector<double>            edges_x,
        Geometry                       geom,
        std::vector<BoundaryCondition> bc,
        std::vector<double>            initial_flux = {},
        double epsilon   = 1e-6,
        int    max_inner = 50,
        bool   verbose   = false
    );

    /**
     * @brief Advance the solution by one backward-Euler time step.
     *
     * Updates the internal flux state and increments `time()` and `steps()`.
     *
     * @param dt Time step size (s).  Must be positive.
     */
    void step(double dt);

    /**
     * @brief Advance @p n_steps uniform steps of size @p dt and return the result.
     *
     * Equivalent to calling `step(dt)` n_steps times followed by `result()`.
     *
     * @param dt      Time step size (s).
     * @param n_steps Number of steps to take.
     * @return Current state as a TimeDependentResult.
     */
    TimeDependentResult run(double dt, int n_steps);

    /**
     * @brief Return the current state as a TimeDependentResult.
     *
     * The returned flux array has layout [cells * n_groups], row-major
     * (same as DiffusionResult::flux): `flux[i * n_groups + g]`.
     */
    TimeDependentResult result() const;

    /// Total elapsed simulated time (s).
    double time()  const { return time_; }
    /// Number of time steps taken so far.
    int    steps() const { return steps_; }

private:
    Materials                      mats_;
    std::vector<int>               medium_map_;
    std::vector<double>            edges_x_;
    std::vector<double>            surface_area_;
    std::vector<double>            volume_;
    Geometry                       geom_;
    std::vector<BoundaryCondition> bc_;
    double epsilon_;
    int    max_inner_;
    bool   verbose_;

    int cells_;
    int groups_;
    int N_;

    /// Base tridiagonal bands (no time-absorption term), reused every step.
    std::vector<double> lower_base_;
    std::vector<double> diag_base_;
    std::vector<double> upper_base_;

    /// Internal flux state [groups_ * N_], ghost row included.
    std::vector<double> phi_;

    double time_;   ///< Elapsed simulated time (s)
    int    steps_;  ///< Number of steps taken
};
