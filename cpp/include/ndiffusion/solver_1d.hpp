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
 *  - KEigenSolver  - k-eigenvalue power iteration
 *  - TimeDependentSolver - backward-Euler time stepping
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
 * precomputed per-group tridiagonal bands; no full NxN matrix is ever
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
class KEigenSolver {
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
    KEigenSolver(
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
    /// Fission source operator: b[g*N+i] = chi_g * Sigma_gp( nuSigma_f,gp * phi_gp[i] )
    void apply_B(const std::vector<double>& phi,
                       std::vector<double>& b) const;

    /// Solve A*phi = b via Gauss-Seidel (groups) + Thomas (spatial).
    /// @return true if the group Gauss-Seidel converged within max_inner_.
    bool solve_A(const std::vector<double>& b,
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
// Fixed-source solver
// ============================================================================

/**
 * @brief Matrix-free 1-D multigroup neutron diffusion fixed-source solver.
 *
 * Solves the linear system
 * @code
 *   A phi = q
 * @endcode
 * where @b q is a user-supplied external neutron source.  No fission or power
 * iteration is performed; this solver is appropriate for subcritical
 * assemblies, shielding problems, and source-multiplication calculations.
 *
 * @par Algorithm
 * Gauss-Seidel sweeps over energy groups with a Thomas (TDMA) tridiagonal
 * solve per sweep, exactly as in the inner loop of KEigenSolver.
 * In-scatter from other groups is treated with the latest available flux
 * iterate (Gauss-Seidel ordering).
 *
 * @par Source layout
 * The @p source array passed to solve() must have @c cells * n_groups elements
 * in the same row-major layout as the returned flux:
 * @code
 *   source[i * n_groups + g]  ->  external source in cell i, group g
 * @endcode
 */
class FixedSourceSolver {
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
     * @param max_inner   Maximum Gauss-Seidel iteration count.
     * @param verbose     Print iteration diagnostics if true.
     *
     * @throws std::invalid_argument if `bc.size() != mats.n_groups`.
     */
    FixedSourceSolver(
        Materials                      mats,
        std::vector<int>               medium_map,
        std::vector<double>            edges_x,
        Geometry                       geom,
        std::vector<BoundaryCondition> bc,
        double epsilon   = 1e-8,
        int    max_inner = 200,
        bool   verbose   = false
    );

    /**
     * @brief Solve A*phi = q and return the converged flux.
     *
     * @param source External source [cells * n_groups], row-major.
     * @return FixedSourceResult containing flux, iteration count, and residual.
     *
     * @throws std::invalid_argument if `source.size() != cells * n_groups`.
     */
    FixedSourceResult solve(const std::vector<double>& source) const;

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
 * Advances the multigroup diffusion equation
 * @code
 *   (1/v_g) dphi_g/dt = -A_g phi_g
 *                    + sum_{gp!=g} sigma_s(g<-gp) * phi_gp          [scatter]
 *                    + (1-beta) chi_p,g * F                         [prompt]
 *                    + sum_i chi_d,i,g * lambda_i * C_i             [delayed]
 *   dC_i/dt = beta_i * F - lambda_i * C_i,   F = sum_gp nu_sigf_gp phi_gp
 * @endcode
 * using **backward Euler** time differencing.
 *
 * @par Time-stepping algorithm
 * 1. The precursor equation is integrated in closed form, folding the delayed
 *    source into a dt-dependent effective fission spectrum `chi_eff` plus a
 *    source `Q_d` known from the old precursors (see solver_detail.hpp).
 * 2. The time-absorption term `1/(v_g * dt)` is added to the spatial
 *    diagonal, so the modified per-group system is still tridiagonal.
 * 3. A Gauss-Seidel sweep over groups resolves the implicit scatter coupling
 *    **and** the implicit fission source.
 * 4. The precursors are advanced from the converged new flux.
 *
 * Fission is treated implicitly, so the scheme is unconditionally stable and
 * first-order accurate in time.  Use small dt for physical accuracy.
 *
 * With no delayed data (the default) this reduces to prompt-only kinetics.
 *
 * @par Requirements
 * `Materials::velocity` must be set (size = n_groups) before constructing
 * the solver; a `std::invalid_argument` is thrown otherwise.
 *
 * @par Usage
 * @code
 *   TimeDependentSolver tds(mats, medium_map, edges_x, geom, bc,
 *                           initial_flux, 1e-6, 50, false, delayed);
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
     * @param delayed       Delayed neutron precursor data.  Defaults to empty,
     *                      giving prompt-only kinetics.
     * @param initial_precursors Starting precursor concentrations per unit
     *                      volume, `[cells * n_precursor]` row-major.  Defaults
     *                      to equilibrium with `initial_flux`, the right choice
     *                      when the transient starts from a steady state.
     * @param theta         Time-differencing weight in `[0.5, 1]`.  1 (the
     *                      default) is backward Euler, first order; 0.5 is
     *                      Crank-Nicolson, second order.  See `set_theta`.
     *
     * @throws std::invalid_argument if `bc.size() != mats.n_groups`,
     *         `mats.velocity.size() != mats.n_groups`,
     *         `initial_flux.size() != cells * n_groups` (when non-empty),
     *         the delayed data is inconsistent with `mats`,
     *         `initial_precursors.size() != cells * n_precursor` (when non-empty),
     *         or `theta` is outside `[0.5, 1]`.
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
        bool   verbose   = false,
        DelayedNeutronData             delayed = {},
        std::vector<double>            initial_precursors = {},
        double theta     = 1.0
    );

    /**
     * @brief Advance the solution by one theta-weighted time step.
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
     *
     * @return Current flux, time, and step count.
     */
    TimeDependentResult result() const;

    /**
     * @brief Replace the cross sections mid-transient and rebuild the operator.
     *
     * This is the perturbation mechanism for reactivity transients: the flux and
     * precursor state are preserved, only the spatial operator and fission data
     * change.  Drive a ramp by calling this once per step with interpolated
     * cross sections.
     *
     * The mesh, geometry, boundary conditions, and material layout are fixed at
     * construction, so `mats` must keep the same `n_mat` and `n_groups`.
     *
     * @param mats New cross-section data, including `velocity`.
     * @throws std::invalid_argument if the new materials fail validation, change
     *         `n_mat` or `n_groups`, or are inconsistent with the delayed data.
     *
     * @note The change takes effect at the *start* of the next step, which is
     *       what a step insertion at `t_n` means: the new cross sections hold
     *       across the whole of `[t_n, t_n + dt]`.  For `theta < 1` that
     *       includes the explicitly weighted term, which is therefore built
     *       from the new data too.
     */
    void update_materials(Materials mats);

    /// @return Total elapsed simulated time in seconds.
    double time()  const { return time_; }
    /// @return Number of time steps taken so far.
    int    steps() const { return steps_; }
    /// @return Precursor concentrations per unit volume, `[cells * n_precursor]`.
    const std::vector<double>& precursors() const { return precursors_; }
    /// @return The current time-differencing weight.
    double theta() const { return theta_; }

    /**
     * @brief Set the time-differencing weight for subsequent steps.
     *
     * 1 is backward Euler (first order), 0.5 is Crank-Nicolson (second order).
     * Both are unconditionally stable, but only backward Euler *damps* the
     * stiff modes: at `theta = 0.5` a mode too fast for the step size rings -
     * decaying slowly with an alternating sign - instead of being killed.
     * Whether that matters depends on how much stiff content a perturbation
     * excites; a smooth, mode-shaped one excites very little.  Hence this
     * setter: when it does matter, take one or two steps at `theta = 1` right
     * after the perturbation, then drop back to 0.5 for the smooth part of the
     * transient.
     *
     * @param theta Weight in `[0.5, 1]`.
     * @throws std::invalid_argument if `theta` is outside `[0.5, 1]`.
     */
    void set_theta(double theta);

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
    DelayedNeutronData             delayed_;
    double theta_;

    int cells_;
    int groups_;
    int N_;

    /// Base tridiagonal bands (no time-absorption term), reused every step.
    std::vector<double> lower_base_;
    std::vector<double> diag_base_;
    std::vector<double> upper_base_;

    /// Internal flux state [groups_ * N_], ghost row included.
    std::vector<double> phi_;

    /// Precursor concentrations per unit volume [cells_ * n_precursor].
    std::vector<double> precursors_;

    /// Cached effective-fission-spectrum materials and the (dt, theta) they were
    /// built for; rebuilt only when those or the cross sections change.
    Materials chi_eff_mats_;
    double    chi_eff_dt_;
    double    chi_eff_theta_;

    /// Shadow materials carrying the *prompt* fission spectrum `(1-beta) chi_p`,
    /// used only by the explicit term when `theta_ < 1`.  Depends on `mats_` and
    /// `delayed_` alone, so it is rebuilt with the operator, not per step.
    Materials prompt_mats_;

    /// True once a non-convergent step has been reported (warn once per solver).
    bool warned_;

    double time_;   ///< Elapsed simulated time (s)
    int    steps_;  ///< Number of steps taken

    /// Rebuild `chi_eff_mats_` if `dt` or `theta_` differs from the cached value.
    void refresh_chi_effective(double dt);
    /// Seed `precursors_` from `initial_precursors`, or from equilibrium.
    void init_precursors(const std::vector<double>& initial_precursors);
    /// Explicit residual `E = -A phi_old + scatter + prompt fission`, in the
    /// internal flux layout.  Only called when `theta_ < 1`.
    void explicit_residual(const std::vector<double>& phi_old,
                           std::vector<double>& out) const;
};
