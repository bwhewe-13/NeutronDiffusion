#pragma once

#include <ndiffusion/types.hpp>
#include <optional>
#include <utility>
#include <vector>

/**
 * @file solver_2d.hpp
 * @brief 2-D multigroup neutron diffusion solver declarations.
 *
 * Two solver families are provided:
 *
 * **Structured mesh** (KEigenSolver2D, TimeDependentSolver2D)
 *   - Cartesian (x,y) or axisymmetric cylindrical (r,z) geometry
 *   - Finite-difference 5-point stencil on an nx x ny grid
 *   - Spatial solve: line-by-line Thomas algorithm (x-direction) with
 *     Gauss-Seidel outer sweep - direct extension of the 1-D solver
 *   - Left (x=0) and bottom (y=0) boundaries: always reflective (zero gradient)
 *   - Right (x=nx) and top (y=ny) boundaries: user-specified Robin BC per group
 *
 * **Unstructured mesh** (KEigenSolverUnstructured2D, TimeDependentSolverUnstructured2D)
 *   - Triangles and/or quadrilaterals defined via vertex/connectivity arrays
 *   - Cell-centered finite-volume method (FVM)
 *   - Spatial solve: point Gauss-Seidel
 *   - Arbitrary Robin BCs per boundary face (identified by vertex pairs)
 *
 * Both families support k-eigenvalue (power iteration) and time-dependent
 * (backward Euler) physics, and reuse the shared Materials, BoundaryCondition,
 * DiffusionResult, and TimeDependentResult types from types.hpp.
 */

// ============================================================================
// Unstructured mesh geometry
// ============================================================================

/**
 * @brief Throw std::invalid_argument unless @p mesh is structurally sound.
 *
 * Checks the connectivity invariants every consumer of an UnstructuredMesh2D
 * relies on: matching vertex-coordinate lengths, a `cell_offsets` array that
 * starts at 0 and increases by at least 3 per cell up to `cell_vertices.size()`,
 * vertex indices in range, paired boundary-face arrays, and no zero-area cells.
 * Called by the solver constructors and by the geometry queries below, both of
 * which would otherwise index out of bounds on a malformed mesh.
 *
 * `material_id` is checked separately, against `Materials::n_mat`.
 */
void validate_mesh(const UnstructuredMesh2D& mesh);

/**
 * @brief Cell centroids of an unstructured mesh.
 *
 * Area centroids from the shoelace formulae, so any simple polygon works -
 * the same centroids the FVM solvers compute internally.
 *
 * @param mesh Unstructured mesh (validated on entry).
 * @return `(cx, cy)`, each of length `n_cells`.
 *
 * @throws std::invalid_argument if the mesh fails validate_mesh().
 */
std::pair<std::vector<double>, std::vector<double>>
cell_centroids(const UnstructuredMesh2D& mesh);

/**
 * @brief Cell areas of an unstructured mesh.
 *
 * @param mesh Unstructured mesh (validated on entry).
 * @return Area per cell, length `n_cells`.
 *
 * @throws std::invalid_argument if the mesh fails validate_mesh().
 */
std::vector<double> cell_areas(const UnstructuredMesh2D& mesh);

// ============================================================================
// Structured 2-D k-eigenvalue solver
// ============================================================================

/**
 * @brief Matrix-free 2-D multigroup neutron diffusion k-eigenvalue solver
 *        on a structured Cartesian or RZ mesh.
 *
 * Solves  A phi = (1/k) B phi  using power iteration.
 * The A operator is applied via per-group x-direction Thomas solves inside a
 * Gauss-Seidel sweep.  No full matrix is assembled.
 *
 * Flux is stored flat as `[nx*ny * n_groups]`, row-major: `flux[(i*ny+j)*G+g]`.
 * The user can reshape to `(nx, ny, G)` in Python/NumPy.
 */
class KEigenSolver2D {
public:
    /**
     * @param mats       Cross-section data.
     * @param medium_map Material index per cell [nx*ny], row-major: `map[i*ny+j]`.
     * @param edges_x    Cell-edge coordinates in x (size nx+1).
     * @param edges_y    Cell-edge coordinates in y (size ny+1).
     * @param geom       Coordinate system (XY or RZ).
     * @param bc_x       Robin BC for the right x-face, one entry per energy group.
     * @param bc_y       Robin BC for the top y-face, one entry per energy group.
     * @param epsilon    Convergence tolerance on the flux change norm.
     * @param max_outer  Maximum power iterations.
     * @param max_inner  Safety cap on inner Gauss-Seidel iterations per power
     *                   step. The inner solve stops early once converged; the
     *                   cap only bounds the worst case. Finer and multi-group
     *                   meshes need more inner iterations (roughly O(n^2) for
     *                   the spatial Gauss-Seidel), so a too-small cap yields a
     *                   silently inaccurate keff - a stderr warning is emitted
     *                   if the cap is hit without convergence.
     * @param verbose    Print iteration diagnostics if true.
     */
    KEigenSolver2D(
        Materials                      mats,
        std::vector<int>               medium_map,
        std::vector<double>            edges_x,
        std::vector<double>            edges_y,
        Geometry2D                     geom,
        std::vector<BoundaryCondition> bc_x,
        std::vector<BoundaryCondition> bc_y,
        double epsilon   = 1e-8,
        int    max_outer = 200,
        // 1000 (vs the older 50): the O(n^2) spatial Gauss-Seidel needs a high
        // cap so refined/multi-group keff problems converge rather than warn.
        int    max_inner = 1000,
        bool   verbose   = false,
        // Inner solver: unset = NDIFFUSION_KEIG_CG env var (GS if absent);
        // true = within-group Jacobi-PCG; false = line-TDMA Gauss-Seidel.
        std::optional<bool> use_cg = std::nullopt
    );

    /**
     * @brief Run power iteration to convergence.
     *
     * @return Converged k-eigenvalue solution, including flux and iteration
     *         metadata.
     */
    DiffusionResult solve();

    /// Select the inner within-group linear solver (Option B prototype).
    /// `false` (default) = line-TDMA Gauss-Seidel; `true` = matrix-free
    /// Jacobi-preconditioned CG on the symmetrized within-group operator.
    /// Default is taken from the `NDIFFUSION_KEIG_CG` environment variable.
    void set_use_cg(bool v) { use_cg_ = v; }

private:
    void apply_B(const std::vector<double>& phi, std::vector<double>& b) const;
    /// Dispatch to the GS or CG within-group solve based on `use_cg_`.
    /// @return true if the inner solve converged within max_inner_.
    bool solve_A(const std::vector<double>& b, std::vector<double>& phi) const;
    bool solve_A_gs(const std::vector<double>& b, std::vector<double>& phi) const;
    bool solve_A_cg(const std::vector<double>& b, std::vector<double>& phi) const;

    Materials                      mats_;
    std::vector<int>               medium_map_;
    std::vector<double>            edges_x_, edges_y_;
    Geometry2D                     geom_;
    std::vector<BoundaryCondition> bc_x_, bc_y_;
    double epsilon_;
    int    max_outer_, max_inner_;
    bool   verbose_;
    bool   use_cg_;

    int nx_, ny_, groups_;

    // Precomputed per-cell, per-group stencil coefficients.
    // Flat index: g*(nx_*ny_) + i*ny_ + j
    std::vector<double> a_W_;    ///< West  coupling (lower band of x-tridiagonal; 0 at i=0)
    std::vector<double> a_E_;    ///< East  coupling (upper band of x-tridiagonal)
    std::vector<double> a_S_;    ///< South coupling (RHS contribution; 0 at j=0)
    std::vector<double> a_N_;    ///< North coupling (RHS contribution; 0 at j=ny-1)
    std::vector<double> diag_;   ///< Effective diagonal (includes top-BC absorption)

    // Ghost-row coefficients for the right BC (per group).
    std::vector<double> ghost_diag_;  ///< diag  of ghost row [groups_]
    std::vector<double> ghost_lower_; ///< lower of ghost row [groups_]

    // Symmetrized (volume-integrated) coefficients for the CG path. Each row of
    // the per-unit-volume operator is scaled by its cell volume, and the right
    // and top BC ghosts are eliminated into the diagonal, giving an SPD
    // within-group operator. Same flat index as above; built in the constructor.
    std::vector<double> vol_;     ///< Cell volumes [nx_*ny_]
    std::vector<double> aWs_, aEs_, aSs_, aNs_, diags_;  ///< [groups_*nx_*ny_]
};

// ============================================================================
// Structured 2-D time-dependent solver
// ============================================================================

/**
 * @brief 2-D multigroup time-dependent neutron diffusion solver
 *        on a structured Cartesian or RZ mesh.
 *
 * Advances  (1/v_g) dphi_g/dt = -A_g phi_g + fission + scatter + delayed
 * using backward Euler time differencing, with the fission source treated
 * implicitly and the delayed precursor balance integrated in closed form
 * (see solver_detail.hpp).  With no delayed data this is prompt-only kinetics.
 *
 * @note `Materials::velocity` must be set (neutron speed per group, cm/s).
 */
class TimeDependentSolver2D {
public:
    /**
     * @brief Construct a structured 2-D time-dependent solver.
     *
     * @param mats Cross-section data.
     * @param medium_map Material index per cell [nx*ny], row-major:
     *        `map[i*ny+j]`.
     * @param edges_x Cell-edge coordinates in x (size nx+1).
     * @param edges_y Cell-edge coordinates in y (size ny+1).
     * @param geom Coordinate system (XY or RZ).
     * @param bc_x Robin BC for the right x-face, one entry per energy group.
     * @param bc_y Robin BC for the top y-face, one entry per energy group.
     * @param initial_flux  Starting flux [nx*ny * n_groups], row-major.
     *                      If empty, flux is initialized to zero.
     * @param epsilon Convergence tolerance for each implicit solve.
     * @param max_inner Maximum Gauss-Seidel iterations per time step.
     * @param verbose Print iteration diagnostics if true.
     * @param delayed Delayed neutron precursor data.  Defaults to empty,
     *        giving prompt-only kinetics.
     * @param initial_precursors Starting precursor concentrations per unit
     *        volume, `[nx*ny * n_precursor]` row-major.  Defaults to
     *        equilibrium with `initial_flux`.
     * @param theta Time-differencing weight in `[0.5, 1]`.  1 (the default) is
     *        backward Euler, first order; 0.5 is Crank-Nicolson, second order.
     *        See `set_theta`.
     * @throws std::invalid_argument if `theta` is outside `[0.5, 1]`, or on any
     *         of the usual shape and validation failures.
     */
    TimeDependentSolver2D(
        Materials                      mats,
        std::vector<int>               medium_map,
        std::vector<double>            edges_x,
        std::vector<double>            edges_y,
        Geometry2D                     geom,
        std::vector<BoundaryCondition> bc_x,
        std::vector<BoundaryCondition> bc_y,
        std::vector<double>            initial_flux = {},
        double epsilon   = 1e-6,
        int    max_inner = 50,
        bool   verbose   = false,
        DelayedNeutronData             delayed = {},
        std::vector<double>            initial_precursors = {},
        double theta     = 1.0
    );

    /**
     * @brief Advance one theta-weighted time step.
     *
     * @param dt Time step size in seconds.
     * @throws std::invalid_argument if `dt` is not positive and finite.
     */
    void step(double dt);

    /**
     * @brief Advance multiple uniform theta-weighted steps.
     *
     * @param dt Time step size in seconds.
     * @param n_steps Number of time steps to take.
     * @return Current time-dependent state after the requested steps.
     */
    TimeDependentResult run(double dt, int n_steps);

    /**
     * @brief Return the current time-dependent state.
     *
     * @return Current flux, time, and step count.
     */
    TimeDependentResult result() const;

    /**
     * @brief Replace the cross sections mid-transient and rebuild the operator.
     *
     * The perturbation mechanism for reactivity transients: flux and precursor
     * state are preserved, only the spatial operator and fission data change.
     * `mats` must keep the same `n_mat` and `n_groups`.
     *
     * @param mats New cross-section data, including `velocity`.
     * @throws std::invalid_argument on validation failure or a changed shape.
     *
     * @note The change takes effect at the *start* of the next step, which is
     *       what a step insertion at `t_n` means: the new cross sections hold
     *       across the whole of `[t_n, t_n + dt]`, the explicitly weighted term
     *       included.
     */
    void update_materials(Materials mats);

    /// @return Total elapsed simulated time in seconds.
    double time()  const { return time_; }
    /// @return Number of time steps completed so far.
    int    steps() const { return steps_; }
    /// @return Precursor concentrations per unit volume, `[nx*ny * n_precursor]`.
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
    std::vector<double>            edges_x_, edges_y_;
    Geometry2D                     geom_;
    std::vector<BoundaryCondition> bc_x_, bc_y_;
    double epsilon_;
    int    max_inner_;
    bool   verbose_;
    DelayedNeutronData             delayed_;
    double theta_;

    int nx_, ny_, groups_;
    double time_;
    int    steps_;

    std::vector<double> phi_;   ///< Internal flux state [groups_ * nx_ * ny_]

    /// Precursor concentrations per unit volume [nx_*ny_ * n_precursor].
    std::vector<double> precursors_;

    /// Cached effective-fission-spectrum materials and the (dt, theta) they were
    /// built for; rebuilt only when those or the cross sections change.
    Materials chi_eff_mats_;
    double    chi_eff_dt_;
    double    chi_eff_theta_;

    /// Shadow materials carrying the *prompt* fission spectrum `(1-beta) chi_p`,
    /// used only by the explicit term when `theta_ < 1`.
    Materials prompt_mats_;

    /// True once a non-convergent step has been reported (warn once).
    bool warned_;

    // Base (time-independent) stencil coefficients.
    std::vector<double> a_W_base_, a_E_base_, a_S_base_, a_N_base_, diag_base_;
    std::vector<double> ghost_diag_base_, ghost_lower_base_;

    // Per-cell volumes (for the time-absorption term 1/(v_g*dt)*vol).
    std::vector<double> vol_; ///< Cell volumes [nx_ * ny_]

    void solve_step(const std::vector<double>& phi_old,
                    const std::vector<double>& qd,
                    const std::vector<double>& expl,
                    double dt);

    void build_bands();
    void refresh_chi_effective(double dt);
    void init_precursors(const std::vector<double>& initial_precursors);
    /// Explicit residual `E = -A phi_old + scatter + prompt fission`, in the
    /// internal flux layout.  Only called when `theta_ < 1`.
    void explicit_residual(const std::vector<double>& phi_old,
                           std::vector<double>& out) const;
};

// ============================================================================
// Structured 2-D fixed-source solver
// ============================================================================

/**
 * @brief Matrix-free 2-D multigroup neutron diffusion fixed-source solver
 *        on a structured Cartesian or RZ mesh.
 *
 * Solves  A phi = q  where q is a user-supplied volumetric source.
 * No fission or power iteration is performed.
 *
 * Source layout: [nx*ny * n_groups], row-major: `source[(i*ny+j)*G+g]`.
 * Source values are volumetric - identical convention to FixedSourceSolver (1-D).
 *
 * Left (x=0) and bottom (y=0) boundaries are always reflective.
 * Right and top boundaries are user-specified Robin BCs per group.
 */
class FixedSourceSolver2D {
public:
    /**
     * @param mats       Cross-section data.
     * @param medium_map Material index per cell [nx*ny], row-major: `map[i*ny+j]`.
     * @param edges_x    Cell-edge coordinates in x (size nx+1).
     * @param edges_y    Cell-edge coordinates in y (size ny+1).
     * @param geom       Coordinate system (XY or RZ).
     * @param bc_x       Robin BC for the right x-face, one entry per energy group.
     * @param bc_y       Robin BC for the top y-face, one entry per energy group.
     * @param epsilon    Convergence tolerance on the flux change norm.
     * @param max_inner  Maximum Gauss-Seidel iterations.
     * @param verbose    Print iteration diagnostics if true.
     */
    FixedSourceSolver2D(
        Materials                      mats,
        std::vector<int>               medium_map,
        std::vector<double>            edges_x,
        std::vector<double>            edges_y,
        Geometry2D                     geom,
        std::vector<BoundaryCondition> bc_x,
        std::vector<BoundaryCondition> bc_y,
        double epsilon   = 1e-8,
        int    max_inner = 200,
        bool   verbose   = false
    );

    /**
     * @brief Solve A*phi = source and return the converged flux.
     *
     * @param source Volumetric source [nx*ny * n_groups], row-major.
     * @return FixedSourceResult with flux, iterations, residual.
     * @throws std::invalid_argument if source.size() != nx*ny * n_groups.
     */
    FixedSourceResult solve(const std::vector<double>& source) const;

private:
    Materials                      mats_;
    std::vector<int>               medium_map_;
    std::vector<double>            edges_x_, edges_y_;
    Geometry2D                     geom_;
    std::vector<BoundaryCondition> bc_x_, bc_y_;
    double epsilon_;
    int    max_inner_;
    bool   verbose_;

    int nx_, ny_, groups_;

    std::vector<double> a_W_, a_E_, a_S_, a_N_, diag_;
    std::vector<double> ghost_diag_, ghost_lower_;
};

// ============================================================================
// Unstructured 2-D k-eigenvalue solver
// ============================================================================

/**
 * @brief Matrix-free 2-D multigroup neutron diffusion k-eigenvalue solver
 *        on an unstructured triangular/quadrilateral mesh.
 *
 * Uses a cell-centered finite-volume method with point Gauss-Seidel spatial
 * solve inside power iteration.
 *
 * Flux is stored flat as `[n_cells * n_groups]`, row-major: `flux[c*G+g]`.
 */
class KEigenSolverUnstructured2D {
public:
    /**
     * @param mats  Cross-section data.
     * @param mesh  Unstructured mesh (vertices, connectivity, boundary faces).
     * @param bc    Robin BCs indexed by tag.  Size `n_bc_types * n_groups`;
     *              `bc[tag * n_groups + g]` is the BC for tag @p tag, group @p g.
     *              Must be a positive multiple of `n_groups` and cover every tag
     *              the mesh uses; the constructor throws otherwise.
     *              Boundary faces with no matching tag in `mesh.bface_bc_tag`
     *              use tag 0.
     * @param epsilon    Convergence tolerance.
     * @param max_outer  Maximum power iterations.
     * @param max_inner  Safety cap on inner point Gauss-Seidel iterations per
     *                   power step. The inner solve stops early once converged;
     *                   finer and multi-group meshes need more inner iterations,
     *                   so a too-small cap yields a silently inaccurate keff -
     *                   a stderr warning is emitted if the cap is hit without
     *                   convergence.
     * @param verbose    Print iteration diagnostics if true.
     */
    KEigenSolverUnstructured2D(
        Materials         mats,
        UnstructuredMesh2D mesh,
        std::vector<BoundaryCondition> bc,
        double epsilon   = 1e-8,
        int    max_outer = 200,
        // 1000 (vs the older 50): the O(n^2) spatial Gauss-Seidel needs a high
        // cap so refined/multi-group keff problems converge rather than warn.
        int    max_inner = 1000,
        bool   verbose   = false,
        // Inner solver: unset = NDIFFUSION_KEIG_CG env var (GS if absent);
        // true = within-group Jacobi-PCG; false = point Gauss-Seidel.
        std::optional<bool> use_cg = std::nullopt
    );

    /**
     * @brief Run power iteration to convergence.
     *
     * @return Converged k-eigenvalue solution, including flux and iteration
     *         metadata.
     */
    DiffusionResult solve();

    /// Select the inner within-group linear solver (Option B prototype).
    /// `false` (default) = point Gauss-Seidel; `true` = matrix-free
    /// Jacobi-preconditioned CG. The unstructured FVM operator is already
    /// volume-integrated and symmetric, so no symmetrization step is needed.
    /// Default is taken from the `NDIFFUSION_KEIG_CG` environment variable.
    void set_use_cg(bool v) { use_cg_ = v; }

private:
    void apply_B(const std::vector<double>& phi, std::vector<double>& b) const;
    /// Dispatch to the GS or CG within-group solve based on `use_cg_`.
    /// @return true if the inner solve converged within max_inner_.
    bool solve_A(const std::vector<double>& b, std::vector<double>& phi) const;
    bool solve_A_gs(const std::vector<double>& b, std::vector<double>& phi) const;
    bool solve_A_cg(const std::vector<double>& b, std::vector<double>& phi) const;

    Materials                      mats_;
    UnstructuredMesh2D             mesh_;
    std::vector<BoundaryCondition> bc_;
    double epsilon_;
    int    max_outer_, max_inner_;
    bool   verbose_;
    bool   use_cg_;

    int n_cells_, groups_;

    // Preprocessed mesh geometry.
    std::vector<double>            cell_area_;   ///< Cell areas [n_cells]
    std::vector<double>            cell_cx_;     ///< Cell centroid x [n_cells]
    std::vector<double>            cell_cy_;     ///< Cell centroid y [n_cells]
    std::vector<FaceUnstructured2D> faces_;      ///< All faces (interior + boundary)
    /// True when every face is orthogonal, so the deferred non-orthogonal
    /// correction is identically zero and is skipped entirely.
    bool orthogonal_ = true;
    std::vector<std::vector<int>>  cell_faces_;  ///< Face indices per cell [n_cells]

    // Per-group, per-cell diagonal (includes sig_r*area and BC contributions).
    // Index: g * n_cells_ + c
    std::vector<double> a_diag_base_;

    void preprocess_mesh();
    void build_diagonals();
};

// ============================================================================
// Unstructured 2-D time-dependent solver
// ============================================================================

/**
 * @brief 2-D multigroup time-dependent neutron diffusion solver
 *        on an unstructured triangular/quadrilateral mesh.
 *
 * Backward Euler with an implicit fission source and delayed neutron
 * precursors integrated in closed form (see solver_detail.hpp).  With no
 * delayed data this is prompt-only kinetics.
 *
 * @note The FVM equations are volume-integrated, so the RHS terms carry a
 *       cell-area factor.  Precursor concentrations and the production rate
 *       are stored **per unit volume** (unweighted); the area is applied only
 *       where the delayed source enters the RHS.
 *
 * @note `Materials::velocity` must be set (neutron speed per group, cm/s).
 */
class TimeDependentSolverUnstructured2D {
public:
    /**
     * @brief Construct an unstructured 2-D time-dependent solver.
     *
     * @param mats Cross-section data.
     * @param mesh Unstructured mesh (vertices, connectivity, boundary faces).
     * @param bc Robin BCs indexed by tag. Size `n_bc_types * n_groups`;
     *        `bc[tag * n_groups + g]` is the BC for tag @p tag, group @p g.
     *        Must be a positive multiple of `n_groups` and cover every tag the
     *        mesh uses; the constructor throws otherwise.
     * @param initial_flux  Starting flux [n_cells * n_groups], row-major.
     *                      If empty, flux is initialized to zero.
     * @param epsilon Convergence tolerance for each implicit solve.
     * @param max_inner Maximum Gauss-Seidel iterations per time step.
     * @param verbose Print iteration diagnostics if true.
     * @param delayed Delayed neutron precursor data.  Defaults to empty,
     *        giving prompt-only kinetics.
     * @param initial_precursors Starting precursor concentrations per unit
     *        volume, `[n_cells * n_precursor]` row-major.  Defaults to
     *        equilibrium with `initial_flux`.
     * @param theta Time-differencing weight in `[0.5, 1]`.  1 (the default) is
     *        backward Euler, first order; 0.5 is Crank-Nicolson, second order.
     *        See `set_theta`.
     * @throws std::invalid_argument if `theta` is outside `[0.5, 1]`, or on any
     *         of the usual shape and validation failures.
     */
    TimeDependentSolverUnstructured2D(
        Materials          mats,
        UnstructuredMesh2D mesh,
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
     * @brief Advance one theta-weighted time step.
     *
     * @param dt Time step size in seconds.
     * @throws std::invalid_argument if `dt` is not positive and finite.
     */
    void step(double dt);

    /**
     * @brief Advance multiple uniform theta-weighted steps.
     *
     * @param dt Time step size in seconds.
     * @param n_steps Number of time steps to take.
     * @return Current time-dependent state after the requested steps.
     */
    TimeDependentResult run(double dt, int n_steps);

    /**
     * @brief Return the current time-dependent state.
     *
     * @return Current flux, time, and step count.
     */
    TimeDependentResult result() const;

    /**
     * @brief Replace the cross sections mid-transient and rebuild the operator.
     *
     * The perturbation mechanism for reactivity transients: flux and precursor
     * state are preserved, only the spatial operator and fission data change.
     * `mats` must keep the same `n_mat` and `n_groups`.
     *
     * @param mats New cross-section data, including `velocity`.
     * @throws std::invalid_argument on validation failure or a changed shape.
     *
     * @note The change takes effect at the *start* of the next step, which is
     *       what a step insertion at `t_n` means: the new cross sections hold
     *       across the whole of `[t_n, t_n + dt]`, the explicitly weighted term
     *       included.
     */
    void update_materials(Materials mats);

    /// @return Total elapsed simulated time in seconds.
    double time()  const { return time_; }
    /// @return Number of time steps completed so far.
    int    steps() const { return steps_; }
    /// @return Precursor concentrations per unit volume, `[n_cells * n_precursor]`.
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
    UnstructuredMesh2D             mesh_;
    std::vector<BoundaryCondition> bc_;
    double epsilon_;
    int    max_inner_;
    bool   verbose_;
    DelayedNeutronData             delayed_;
    double theta_;

    int n_cells_, groups_;
    double time_;
    int    steps_;

    std::vector<double> phi_; ///< Internal flux state [groups_ * n_cells_]

    /// Precursor concentrations per unit volume [n_cells_ * n_precursor].
    std::vector<double> precursors_;

    /// Cached effective-fission-spectrum materials and the (dt, theta) they were
    /// built for; rebuilt only when those or the cross sections change.
    Materials chi_eff_mats_;
    double    chi_eff_dt_;
    double    chi_eff_theta_;

    /// Shadow materials carrying the *prompt* fission spectrum `(1-beta) chi_p`,
    /// used only by the explicit term when `theta_ < 1`.
    Materials prompt_mats_;

    /// True once a non-convergent step has been reported (warn once).
    bool warned_;

    // Mesh geometry (same fields as KEigenSolverUnstructured2D).
    std::vector<double>            cell_area_, cell_cx_, cell_cy_;
    std::vector<FaceUnstructured2D> faces_;
    /// True when every face is orthogonal, so the deferred non-orthogonal
    /// correction is identically zero and is skipped entirely.
    bool orthogonal_ = true;
    std::vector<std::vector<int>>  cell_faces_;

    std::vector<double> a_diag_base_; ///< Base diagonal (without time term)

    void preprocess_mesh();
    void build_diagonals();
    void solve_step(const std::vector<double>& phi_old,
                    const std::vector<double>& qd,
                    const std::vector<double>& expl,
                    double dt);
    void refresh_chi_effective(double dt);
    void init_precursors(const std::vector<double>& initial_precursors);
    /// Explicit residual `E = -A phi_old + scatter + prompt fission`, in the
    /// volume-integrated FVM form.  Only called when `theta_ < 1`.
    void explicit_residual(const std::vector<double>& phi_old,
                           std::vector<double>& out) const;
};

// ============================================================================
// Unstructured 2-D fixed-source solver
// ============================================================================

/**
 * @brief Matrix-free 2-D multigroup neutron diffusion fixed-source solver
 *        on an unstructured triangular/quadrilateral mesh.
 *
 * Solves  A phi = q  using point Gauss-Seidel.
 *
 * Source layout: [n_cells * n_groups], row-major: `source[c*G+g]`.
 * Values are volumetric; the solver multiplies by cell_area internally
 * to form the volume-integrated RHS (matching the FVM equation).
 */
class FixedSourceSolverUnstructured2D {
public:
    /**
     * @param mats  Cross-section data.
     * @param mesh  Unstructured mesh (vertices, connectivity, boundary faces).
     * @param bc    Robin BCs indexed by tag.  Size `n_bc_types * n_groups`;
     *              `bc[tag * n_groups + g]` is the BC for tag @p tag, group @p g.
     *              Must be a positive multiple of `n_groups` and cover every tag
     *              the mesh uses; the constructor throws otherwise.
     * @param epsilon    Convergence tolerance.
     * @param max_inner  Maximum SOR iterations.
     * @param omega      SOR relaxation factor (1.0 = Gauss-Seidel; 1.5-1.9 typical).
     * @param verbose    Print iteration diagnostics if true.
     */
    FixedSourceSolverUnstructured2D(
        Materials                      mats,
        UnstructuredMesh2D             mesh,
        std::vector<BoundaryCondition> bc,
        double epsilon   = 1e-8,
        int    max_inner = 200,
        double omega     = 1.0,
        bool   verbose   = false
    );

    /**
     * @brief Solve A*phi = source and return the converged flux.
     *
     * @param source Volumetric source [n_cells * n_groups], row-major.
     * @return FixedSourceResult with flux, iterations, residual.
     * @throws std::invalid_argument if source.size() != n_cells * n_groups.
     */
    FixedSourceResult solve(const std::vector<double>& source) const;

private:
    Materials                      mats_;
    UnstructuredMesh2D             mesh_;
    std::vector<BoundaryCondition> bc_;
    double epsilon_;
    int    max_inner_;
    double omega_;
    bool   verbose_;

    int n_cells_, groups_;

    std::vector<double>            cell_area_, cell_cx_, cell_cy_;
    std::vector<FaceUnstructured2D> faces_;
    /// True when every face is orthogonal, so the deferred non-orthogonal
    /// correction is identically zero and is skipped entirely.
    bool orthogonal_ = true;
    std::vector<std::vector<int>>  cell_faces_;

    std::vector<double> a_diag_base_;

    void preprocess_mesh();
    void build_diagonals();
};
