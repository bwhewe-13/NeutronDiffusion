#include <ndiffusion/solver_1d.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ============================================================================
// Mesh helpers
// ============================================================================

static std::vector<double> linspace(double start, double stop, int n) {
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i)
        v[i] = start + static_cast<double>(i) * (stop - start) / (n - 1);
    return v;
}

// All cells assigned to material 0
static std::vector<int> uniform_map(int cells) {
    return std::vector<int>(cells, 0);
}

// First 'interface_cell' cells = material 0, remainder = material 1
static std::vector<int> two_mat_map(int cells, int interface_cell) {
    std::vector<int> map(cells, 0);
    for (int i = interface_cell; i < cells; ++i)
        map[i] = 1;
    return map;
}

// ============================================================================
// Boundary-condition helpers
// ============================================================================

// Simple zero-flux approximation: phi[I] + phi[I-1] = 0
static BoundaryCondition zero_flux_vacuum() { return {1.0, 0.0}; }

// Marshak vacuum BC (Fick's law + partial-current condition at surface)
static BoundaryCondition marshak_vacuum(double D) { return {0.25, D / 2.0}; }

// Reflective (zero-current) BC
static BoundaryCondition reflective() { return {0.0, 1.0}; }

// ============================================================================
// Test runner
// ============================================================================

static void run_problem(
    const std::string&                   name,
    Materials                            mats,
    std::vector<int>                     medium_map,
    std::vector<double>                  edges_x,
    Geometry                             geom,
    std::vector<BoundaryCondition>       bc,
    double                               reference_keff,
    double                               tol
) {
    std::cout << "\n=== " << name << " ===\n";

    // Explicit verbose: the library default is quiet, the driver wants the trace.
    KEigenSolver solver(mats, medium_map, edges_x, geom, bc,
                        /*epsilon=*/1e-8, /*max_outer=*/200,
                        /*max_inner=*/1000, /*verbose=*/true);
    DiffusionResult result = solver.solve();

    const double err = std::abs(result.keff - reference_keff);
    std::cout << std::fixed << std::setprecision(8)
              << "keff:      " << result.keff       << "\n"
              << "Reference: " << reference_keff     << "\n"
              << "Error:     " << err                << "\n"
              << "PASS:      " << (err < tol ? "YES" : "NO") << "\n";
}

// ============================================================================
// Problem definitions
// ============================================================================

int main() {

    // -----------------------------------------------------------------------
    // 1.  1-group slab, 1 material
    // -----------------------------------------------------------------------
    {
        Materials m;
        m.n_mat    = 1;  m.n_groups = 1;
        m.D        = {3.850204978408833};
        m.removal  = {0.1532};
        m.scatter  = {0.0};
        m.chi      = {1.0};
        m.nusigf   = {0.1570};

        run_problem(
            "1-group slab, 1 material",
            m, uniform_map(20), linspace(0.0, 50.0, 21), Geometry::Slab,
            {zero_flux_vacuum()}, 1.00001243892, 1e-4
        );
    }

    // -----------------------------------------------------------------------
    // 2.  1-group slab, 2 materials
    // -----------------------------------------------------------------------
    {
        Materials m;
        m.n_mat    = 2;  m.n_groups = 1;
        m.D        = {5.0, 1.0};
        m.removal  = {0.5, 0.01};
        m.scatter  = {0.0, 0.0};
        m.chi      = {1.0, 1.0};
        m.nusigf   = {0.7, 0.0};

        run_problem(
            "1-group slab, 2 materials",
            m, two_mat_map(100, 50), linspace(0.0, 10.0, 101), Geometry::Slab,
            {zero_flux_vacuum()}, 1.29524, 1e-3
        );
    }

    // -----------------------------------------------------------------------
    // 3.  1-group cylinder, 1 material
    // -----------------------------------------------------------------------
    {
        Materials m;
        m.n_mat    = 1;  m.n_groups = 1;
        m.D        = {3.850204978408833};
        m.removal  = {0.1532};
        m.scatter  = {0.0};
        m.chi      = {1.0};
        m.nusigf   = {0.1570};

        run_problem(
            "1-group cylinder, 1 material",
            m, uniform_map(20), linspace(0.0, 76.5535, 21), Geometry::Cylinder,
            {zero_flux_vacuum()}, 1.00001243892, 1e-4
        );
    }

    // -----------------------------------------------------------------------
    // 4.  1-group cylinder, 2 materials
    // -----------------------------------------------------------------------
    {
        Materials m;
        m.n_mat    = 2;  m.n_groups = 1;
        m.D        = {5.0, 1.0};
        m.removal  = {0.5, 0.01};
        m.scatter  = {0.0, 0.0};
        m.chi      = {1.0, 1.0};
        m.nusigf   = {0.7, 0.0};

        run_problem(
            "1-group cylinder, 2 materials",
            m, two_mat_map(100, 50), linspace(0.0, 10.0, 101), Geometry::Cylinder,
            {zero_flux_vacuum()}, 1.14068, 1e-3
        );
    }

    // -----------------------------------------------------------------------
    // 5.  1-group sphere, 1 material
    // -----------------------------------------------------------------------
    {
        Materials m;
        m.n_mat    = 1;  m.n_groups = 1;
        m.D        = {3.850204978408833};
        m.removal  = {0.1532};
        m.scatter  = {0.0};
        m.chi      = {1.0};
        m.nusigf   = {0.1570};

        run_problem(
            "1-group sphere, 1 material",
            m, uniform_map(20), linspace(0.0, 100.0, 21), Geometry::Sphere,
            {zero_flux_vacuum()}, 1.00001243892, 1e-4
        );
    }

    // -----------------------------------------------------------------------
    // 6.  1-group sphere, 2 materials
    // -----------------------------------------------------------------------
    {
        Materials m;
        m.n_mat    = 2;  m.n_groups = 1;
        m.D        = {5.0, 1.0};
        m.removal  = {0.5, 0.01};
        m.scatter  = {0.0, 0.0};
        m.chi      = {1.0, 1.0};
        m.nusigf   = {0.7, 0.0};

        run_problem(
            "1-group sphere, 2 materials",
            m, two_mat_map(150, 75), linspace(0.0, 10.0, 151), Geometry::Sphere,
            {zero_flux_vacuum()}, 0.95735, 2e-3
        );
    }

    // -----------------------------------------------------------------------
    // 7.  2-group sphere, 1 material  (no scattering)
    // -----------------------------------------------------------------------
    {
        Materials m;
        m.n_mat    = 1;  m.n_groups = 2;
        m.D        = {3.850204978408833, 3.850204978408833};
        m.removal  = {0.1532, 0.1532};
        // scatter[mat][g_to][g_from]:  no coupling
        m.scatter  = {0.0, 0.0,
                      0.0, 0.0};
        m.chi      = {1.0, 0.0};
        m.nusigf   = {0.1570, 0.1570};

        run_problem(
            "2-group sphere, 1 material (no scatter)",
            m, uniform_map(20), linspace(0.0, 100.0, 21), Geometry::Sphere,
            {zero_flux_vacuum(), zero_flux_vacuum()}, 1.0000295511, 1e-4
        );
    }

    // -----------------------------------------------------------------------
    // 8.  2-group sphere, 1 material  (with fast-to-thermal downscatter)
    // -----------------------------------------------------------------------
    {
        Materials m;
        m.n_mat    = 1;  m.n_groups = 2;
        m.D        = {0.1, 0.1};
        m.removal  = {0.0362, 0.121};
        // scatter[0][g_to=1][g_from=0] = 0.0241  (fast -> thermal)
        m.scatter  = {0.0,    0.0,
                      0.0241, 0.0};
        m.chi      = {1.0, 0.0};
        m.nusigf   = {0.0085, 0.185};

        run_problem(
            "2-group sphere, 1 material (downscatter)",
            m, uniform_map(50), linspace(0.0, 5.0, 51), Geometry::Sphere,
            {reflective(), reflective()}, 1.25268252483, 1e-4
        );
    }

    // -----------------------------------------------------------------------
    // 9.  2-group sphere, 2 materials
    // -----------------------------------------------------------------------
    {
        Materials m;
        m.n_mat    = 2;  m.n_groups = 2;
        m.D        = {1.0, 1.0,
                      1.0, 1.0};
        m.removal  = {0.01, 0.01,
                      0.01, 0.00049};
        // mat 0: scatter[0][1][0] = 0.001;  mat 1: scatter[1][1][0] = 0.009
        m.scatter  = {0.0,   0.0,   0.001, 0.0,
                      0.0,   0.0,   0.009, 0.0};
        m.chi      = {1.0, 0.0,
                      1.0, 0.0};
        m.nusigf   = {0.00085, 0.057,
                      0.0,     0.0};

        BoundaryCondition bc_grp = marshak_vacuum(1.0);  // D = 1 => B = 0.5
        run_problem(
            "2-group sphere, 2 materials",
            m, two_mat_map(100, 50), linspace(0.0, 100.0, 101), Geometry::Sphere,
            {bc_grp, bc_grp}, 1.06508498598, 1e-5
        );
    }

    // -----------------------------------------------------------------------
    // 10. 2-group sphere, 1 material  (subcritical with albedo BC)
    // -----------------------------------------------------------------------
    {
        Materials m;
        m.n_mat    = 1;  m.n_groups = 2;
        m.D        = {1.0, 1.0};
        m.removal  = {0.01, 0.01};
        m.scatter  = {0.0,   0.0,
                      0.001, 0.0};
        m.chi      = {1.0, 0.0};
        m.nusigf   = {0.00085, 0.057};

        BoundaryCondition bc_grp = marshak_vacuum(1.0);
        run_problem(
            "2-group sphere, 1 material (subcritical)",
            m, uniform_map(100), linspace(0.0, 50.0, 101), Geometry::Sphere,
            {bc_grp, bc_grp}, 0.368702897492, 1e-5
        );
    }

    // -----------------------------------------------------------------------
    // 11. Kinetics: infinite medium with 6 delayed precursor groups
    //
    //     Reflective on both sides and a uniform initial flux means no
    //     leakage, so the problem reduces exactly to point kinetics with
    //     Lambda = 1/(v*nu_sigf) and rho = 1 - Sigma_a/nu_sigf.  Two checks:
    //     an unperturbed critical medium with equilibrium precursors must hold
    //     its amplitude, and a $0.50 step insertion must show the prompt jump
    //     beta/(beta - rho) = 2 rather than a prompt excursion.
    // -----------------------------------------------------------------------
    {
        const double nu_sigf = 0.1570;
        const double velocity = 2.2e5;

        // Keepin 6-group U-235 thermal fission data.
        DelayedNeutronData delayed;
        delayed.n_precursor = 6;
        delayed.lambda = {0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01};
        delayed.beta   = {0.000215, 0.001424, 0.001274,
                          0.002568, 0.000748, 0.000273};
        delayed.chi_delayed = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

        double beta_total = 0.0;
        for (double b : delayed.beta) beta_total += b;

        auto medium = [&](double rho) {
            Materials m;
            m.n_mat = 1;  m.n_groups = 1;
            m.D        = {1.0};
            m.removal  = {nu_sigf * (1.0 - rho)};
            m.scatter  = {0.0};
            m.chi      = {1.0};
            m.nusigf   = {nu_sigf};
            m.velocity = {velocity};
            return m;
        };

        const int cells = 8;
        const std::vector<double> initial_flux(cells, 1.0);
        const std::vector<BoundaryCondition> bc = {reflective()};

        std::cout << "\n=== Kinetics: infinite medium, 6 delayed groups ===\n";

        {
            TimeDependentSolver tds(
                medium(0.0), uniform_map(cells), linspace(0.0, 10.0, cells + 1),
                Geometry::Slab, bc, initial_flux, 1e-10, 500, false, delayed
            );
            tds.run(1.0, 10);
            const double amp = tds.result().flux[0];
            std::cout << "  critical, equilibrium precursors: amplitude = "
                      << std::setprecision(12) << amp
                      << (std::fabs(amp - 1.0) < 1e-8 ? "  [OK]" : "  [FAIL]")
                      << "\n";
        }

        {
            const double rho = 0.5 * beta_total;
            TimeDependentSolver tds(
                medium(0.0), uniform_map(cells), linspace(0.0, 10.0, cells + 1),
                Geometry::Slab, bc, initial_flux, 1e-10, 500, false, delayed
            );
            tds.update_materials(medium(rho));
            tds.run(1e-5, 2000);  // 0.02 s
            const double amp = tds.result().flux[0];
            const double jump = beta_total / (beta_total - rho);
            std::cout << "  $0.50 step, t = 0.02 s: amplitude = "
                      << std::setprecision(6) << amp
                      << "  (prompt jump " << jump << ")"
                      << (amp > 1.8 && amp < jump ? "  [OK]" : "  [FAIL]")
                      << "\n";
        }
    }

    return 0;
}
