#pragma once

#include <ndiffusion/types.hpp>
#include <vector>

/**
 * @file solver_2d.hpp
 * @brief 2-D multigroup neutron diffusion solver declarations.
 *
 * Two solver families are provided:
 *
 * **Structured mesh** (KEigenSolver2D, TimeDependentSolver2D)
 *   - Cartesian (x,y) or axisymmetric cylindrical (r,z) geometry
 *   - Finite-difference 5-point stencil on an nx × ny grid
 *   - Spatial solve: line-by-line Thomas algorithm (x-direction) with
 *     Gauss-Seidel outer sweep — direct extension of the 1-D solver
 *   - Left (x=0) and bottom (y=0) boundaries: always reflective (zero gradient)
 *   - Right (x=nx) and top (y=ny) boundaries: user-specified Robin BC per group
 *
 * **Unstructured mesh** (KEigenSolverUnstructured2D, TimeDependentSolverUnstructured2D)
 *   - Triangles and/or quadrilaterals defined via vertex/connectivity arrays
 *   - Cell-centred finite-volume method (FVM)
 *   - Spatial solve: point Gauss-Seidel
 *   - Arbitrary Robin BCs per boundary face (identified by vertex pairs)
 *
 * Both families support k-eigenvalue (power iteration) and time-dependent
 * (backward Euler) physics, and reuse the shared Materials, BoundaryCondition,
 * DiffusionResult, and TimeDependentResult types from types.hpp.
 */

// ============================================================================
// Structured 2-D k-eigenvalue solver
// ============================================================================

/**
 * @brief Matrix-free 2-D multigroup neutron diffusion k-eigenvalue solver
 *        on a structured Cartesian or RZ mesh.
 *
 * Solves  A φ = (1/k) B φ  using power iteration.
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
     * @param max_inner  Maximum Gauss-Seidel iterations per power step.
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
        int    max_inner = 50,
        bool   verbose   = true
    );

    /**
     * @brief Run power iteration to convergence.
     *
     * @return Converged k-eigenvalue solution, including flux and iteration
     *         metadata.
     */
    DiffusionResult solve();

private:
    void apply_B(const std::vector<double>& phi, std::vector<double>& b) const;
    void solve_A(const std::vector<double>& b, std::vector<double>& phi) const;

    Materials                      mats_;
    std::vector<int>               medium_map_;
    std::vector<double>            edges_x_, edges_y_;
    Geometry2D                     geom_;
    std::vector<BoundaryCondition> bc_x_, bc_y_;
    double epsilon_;
    int    max_outer_, max_inner_;
    bool   verbose_;

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
};

// ============================================================================
// Structured 2-D time-dependent solver
// ============================================================================

/**
 * @brief 2-D multigroup time-dependent neutron diffusion solver
 *        on a structured Cartesian or RZ mesh.
 *
 * Advances  (1/v_g) dφ_g/dt = −A_g φ_g + fission + scatter
 * using backward Euler time differencing.
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
     *                      If empty, flux is initialised to zero.
     * @param epsilon Convergence tolerance for each implicit solve.
     * @param max_inner Maximum Gauss-Seidel iterations per time step.
     * @param verbose Print iteration diagnostics if true.
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
        bool   verbose   = true
    );

    /**
     * @brief Advance one backward-Euler time step.
     *
     * @param dt Time step size in seconds.
     */
    void step(double dt);

    /**
     * @brief Advance multiple uniform backward-Euler steps.
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

    /// @return Total elapsed simulated time in seconds.
    double time()  const { return time_; }
    /// @return Number of time steps completed so far.
    int    steps() const { return steps_; }

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
    double time_;
    int    steps_;

    std::vector<double> phi_;   ///< Internal flux state [groups_ * nx_ * ny_]

    // Base (time-independent) stencil coefficients.
    std::vector<double> a_W_base_, a_E_base_, a_S_base_, a_N_base_, diag_base_;
    std::vector<double> ghost_diag_base_, ghost_lower_base_;

    // Per-cell volumes (for the time-absorption term 1/(v_g*dt)*vol).
    std::vector<double> vol_; ///< Cell volumes [nx_ * ny_]

    void solve_step(const std::vector<double>& phi_old,
                    const std::vector<double>& fis,
                    double dt);
};

// ============================================================================
// Structured 2-D fixed-source solver
// ============================================================================

/**
 * @brief Matrix-free 2-D multigroup neutron diffusion fixed-source solver
 *        on a structured Cartesian or RZ mesh.
 *
 * Solves  A φ = q  where q is a user-supplied volumetric source.
 * No fission or power iteration is performed.
 *
 * Source layout: [nx*ny * n_groups], row-major: `source[(i*ny+j)*G+g]`.
 * Source values are volumetric — identical convention to FixedSourceSolver (1-D).
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
     * @brief Solve A·φ = source and return the converged flux.
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
 * Uses a cell-centred finite-volume method with point Gauss-Seidel spatial
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
     *              Boundary faces with no matching tag in `mesh.bface_bc_tag`
     *              use tag 0.
     * @param epsilon    Convergence tolerance.
     * @param max_outer  Maximum power iterations.
     * @param max_inner  Maximum Gauss-Seidel iterations per power step.
     * @param verbose    Print iteration diagnostics if true.
     */
    KEigenSolverUnstructured2D(
        Materials         mats,
        UnstructuredMesh2D mesh,
        std::vector<BoundaryCondition> bc,
        double epsilon   = 1e-8,
        int    max_outer = 200,
        int    max_inner = 50,
        bool   verbose   = true
    );

    /**
     * @brief Run power iteration to convergence.
     *
     * @return Converged k-eigenvalue solution, including flux and iteration
     *         metadata.
     */
    DiffusionResult solve();

private:
    void apply_B(const std::vector<double>& phi, std::vector<double>& b) const;
    void solve_A(const std::vector<double>& b, std::vector<double>& phi) const;

    Materials                      mats_;
    UnstructuredMesh2D             mesh_;
    std::vector<BoundaryCondition> bc_;
    double epsilon_;
    int    max_outer_, max_inner_;
    bool   verbose_;

    int n_cells_, groups_;

    // Preprocessed mesh geometry.
    std::vector<double>            cell_area_;   ///< Cell areas [n_cells]
    std::vector<double>            cell_cx_;     ///< Cell centroid x [n_cells]
    std::vector<double>            cell_cy_;     ///< Cell centroid y [n_cells]
    std::vector<FaceUnstructured2D> faces_;      ///< All faces (interior + boundary)
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
     * @param initial_flux  Starting flux [n_cells * n_groups], row-major.
     *                      If empty, flux is initialised to zero.
     * @param epsilon Convergence tolerance for each implicit solve.
     * @param max_inner Maximum Gauss-Seidel iterations per time step.
     * @param verbose Print iteration diagnostics if true.
     */
    TimeDependentSolverUnstructured2D(
        Materials          mats,
        UnstructuredMesh2D mesh,
        std::vector<BoundaryCondition> bc,
        std::vector<double>            initial_flux = {},
        double epsilon   = 1e-6,
        int    max_inner = 50,
        bool   verbose   = true
    );

    /**
     * @brief Advance one backward-Euler time step.
     *
     * @param dt Time step size in seconds.
     */
    void step(double dt);

    /**
     * @brief Advance multiple uniform backward-Euler steps.
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

    /// @return Total elapsed simulated time in seconds.
    double time()  const { return time_; }
    /// @return Number of time steps completed so far.
    int    steps() const { return steps_; }

private:
    Materials                      mats_;
    UnstructuredMesh2D             mesh_;
    std::vector<BoundaryCondition> bc_;
    double epsilon_;
    int    max_inner_;
    bool   verbose_;

    int n_cells_, groups_;
    double time_;
    int    steps_;

    std::vector<double> phi_; ///< Internal flux state [groups_ * n_cells_]

    // Mesh geometry (same fields as KEigenSolverUnstructured2D).
    std::vector<double>            cell_area_, cell_cx_, cell_cy_;
    std::vector<FaceUnstructured2D> faces_;
    std::vector<std::vector<int>>  cell_faces_;

    std::vector<double> a_diag_base_; ///< Base diagonal (without time term)

    void preprocess_mesh();
    void build_diagonals();
    void solve_step(const std::vector<double>& phi_old,
                    const std::vector<double>& fis,
                    double dt);
};

// ============================================================================
// Unstructured 2-D fixed-source solver
// ============================================================================

/**
 * @brief Matrix-free 2-D multigroup neutron diffusion fixed-source solver
 *        on an unstructured triangular/quadrilateral mesh.
 *
 * Solves  A φ = q  using point Gauss-Seidel.
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
     * @param epsilon    Convergence tolerance.
     * @param max_inner  Maximum SOR iterations.
     * @param omega      SOR relaxation factor (1.0 = Gauss-Seidel; 1.5–1.9 typical).
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
     * @brief Solve A·φ = source and return the converged flux.
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
    std::vector<std::vector<int>>  cell_faces_;

    std::vector<double> a_diag_base_;

    void preprocess_mesh();
    void build_diagonals();
};
