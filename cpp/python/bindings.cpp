#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ndiffusion/solver_1d.hpp>
#include <ndiffusion/solver_2d.hpp>
#include <ndiffusion/solver_detail.hpp>

namespace py = pybind11;

namespace {

// Shared by the three time-dependent solvers' `theta` property.
const char* const THETA_DOC =
    "Time-differencing weight, in [0.5, 1].\n\n"
    "1 is backward Euler (first order in dt), 0.5 is Crank-Nicolson (second\n"
    "order).  Both are unconditionally stable, but only backward Euler *damps*\n"
    "the stiff modes: at theta = 0.5 a mode too fast for the step size rings -\n"
    "decaying slowly with an alternating sign - instead of being killed.\n"
    "Whether that matters depends on how much stiff content a perturbation\n"
    "excites; a smooth, mode-shaped one excites very little.  When it does\n"
    "matter, take one or two steps at theta = 1 right after the perturbation,\n"
    "then set it back to 0.5 for the smooth part of the transient.\n\n"
    "Raises ValueError if set outside [0.5, 1]: below 0.5 the scheme is only\n"
    "conditionally stable, and the condition is hopeless here (the fast spatial\n"
    "modes decay at ~v * Sigma_r, of order 1e4 per second).";

// Installed as the core library's interrupt hook.  The solves below release the
// GIL (py::call_guard<py::gil_scoped_release>), so this reacquires it, runs
// Python's pending signal handlers, and propagates whatever they raised -
// normally KeyboardInterrupt.  Without it a Ctrl-C is only delivered once the
// solve returns.
void check_python_signals() {
    py::gil_scoped_acquire gil;
    if (PyErr_CheckSignals() != 0)
        throw py::error_already_set();
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "ndiffusion C++ backend - 1-D and 2-D multigroup neutron diffusion solvers";

    // Installed once, before any solver exists, and never reassigned.
    ndiffusion::detail::interrupt_hook() = &check_python_signals;

    // ------------------------------------------------------------------
    // Geometry enum
    // ------------------------------------------------------------------
    py::enum_<Geometry>(m, "Geometry")
        .value("Slab",     Geometry::Slab)
        .value("Cylinder", Geometry::Cylinder)
        .value("Sphere",   Geometry::Sphere)
        .export_values();

    // ------------------------------------------------------------------
    // Materials
    // ------------------------------------------------------------------
    py::class_<Materials>(m, "Materials",
        "Cross-section data for all materials and energy groups.\n\n"
        "All arrays are flat (row-major):\n"
        "  D, removal, chi : [n_mat * n_groups]\n"
        "  nusigf          : [n_mat * n_groups]  (standard mode)\n"
        "                    [n_mat * n_groups * n_groups]  (fission-matrix mode,\n"
        "                     nusigf[m][g_to][g_from]) - activated when chi is all zeros\n"
        "  scatter         : [n_mat * n_groups * n_groups]  (scatter[m][g_to][g_from])\n"
        "  velocity        : [n_groups]  neutron speed (cm/s)\n\n"
        "numpy arrays are automatically converted to std::vector<double>.")
        .def(py::init<>())
        .def_readwrite("n_mat",    &Materials::n_mat)
        .def_readwrite("n_groups", &Materials::n_groups)
        .def_readwrite("D",        &Materials::D)
        .def_readwrite("removal",  &Materials::removal)
        .def_readwrite("scatter",  &Materials::scatter)
        .def_readwrite("chi",      &Materials::chi)
        .def_readwrite("nusigf",   &Materials::nusigf)
        .def_readwrite("velocity", &Materials::velocity);

    // ------------------------------------------------------------------
    // DelayedNeutronData
    // ------------------------------------------------------------------
    py::class_<DelayedNeutronData>(m, "DelayedNeutronData",
        "Delayed neutron precursor data for the time-dependent solvers.\n\n"
        "All arrays are flat (row-major):\n"
        "  lambda      : [n_precursor]                    decay constants (1/s)\n"
        "  beta        : [n_mat * n_precursor]            delayed fractions\n"
        "  chi_delayed : [n_mat * n_precursor * n_groups] delayed spectrum\n"
        "  chi_prompt  : [n_mat * n_groups], or empty to use Materials.chi\n\n"
        "A default-constructed instance (n_precursor = 0) disables delayed\n"
        "neutrons, giving prompt-only kinetics.  Delayed neutrons require the\n"
        "standard chi / nusigf representation - fission-matrix mode is rejected.")
        .def(py::init<>())
        .def_readwrite("n_precursor", &DelayedNeutronData::n_precursor)
        .def_readwrite("lambda_",     &DelayedNeutronData::lambda,
            "Decay constants (1/s) [n_precursor]. Named with a trailing "
            "underscore because 'lambda' is a Python keyword.")
        .def_readwrite("beta",        &DelayedNeutronData::beta)
        .def_readwrite("chi_delayed", &DelayedNeutronData::chi_delayed)
        .def_readwrite("chi_prompt",  &DelayedNeutronData::chi_prompt);

    // ------------------------------------------------------------------
    // BoundaryCondition
    // ------------------------------------------------------------------
    py::class_<BoundaryCondition>(m, "BoundaryCondition",
        "Robin BC at the outer surface:  A*phi + B*(dphi/dx) = 0\n\n"
        "Common choices:\n"
        "  vacuum (Marshak):   A = (1-alpha)/(4*(1+alpha)),  B = D/2\n"
        "  reflective:         A = 0,  B = 1\n"
        "  zero-flux approx:   A = 1,  B = 0")
        .def(py::init<>())
        .def(py::init<double, double>(), py::arg("A"), py::arg("B"))
        .def_readwrite("A", &BoundaryCondition::A)
        .def_readwrite("B", &BoundaryCondition::B);

    // ------------------------------------------------------------------
    // DiffusionResult
    // ------------------------------------------------------------------
    py::class_<DiffusionResult>(m, "DiffusionResult")
        .def_readonly("flux",       &DiffusionResult::flux,
            "Physical flux [cells * n_groups], row-major: flux[i * n_groups + g]")
        .def_readonly("keff",       &DiffusionResult::keff)
        .def_readonly("iterations", &DiffusionResult::iterations)
        .def_readonly("residual",   &DiffusionResult::residual)
        .def_readonly("converged",  &DiffusionResult::converged,
            "True when the outer power iteration and inner solves all met "
            "their tolerances");

    // ------------------------------------------------------------------
    // KEigenSolver
    // ------------------------------------------------------------------
    py::class_<KEigenSolver>(m, "KEigenSolver",
        "Matrix-free 1-D multigroup neutron diffusion k-eigenvalue solver.\n\n"
        "Solves  A phi = (1/k) B phi  using power iteration.\n"
        "The A operator is applied implicitly via per-group Thomas (TDMA) solves\n"
        "inside a Gauss-Seidel sweep over energy groups.  No full NxN matrix is\n"
        "ever assembled.")
        .def(py::init<Materials,
                      std::vector<int>,
                      std::vector<double>,
                      Geometry,
                      std::vector<BoundaryCondition>,
                      double, int, int, bool>(),
             py::arg("mats"),
             py::arg("medium_map"),
             py::arg("edges_x"),
             py::arg("geom"),
             py::arg("bc"),
             py::arg("epsilon")   = 1e-8,
             py::arg("max_outer") = 200,
             py::arg("max_inner") = 1000,
             py::arg("verbose")   = false)
        .def("solve", &KEigenSolver::solve,
             "Run power iteration and return a DiffusionResult.",
             py::call_guard<py::gil_scoped_release>());

    // ------------------------------------------------------------------
    // FixedSourceResult
    // ------------------------------------------------------------------
    py::class_<FixedSourceResult>(m, "FixedSourceResult")
        .def_readonly("flux",       &FixedSourceResult::flux,
            "Physical flux [cells * n_groups], row-major: flux[i * n_groups + g]")
        .def_readonly("iterations", &FixedSourceResult::iterations,
            "Gauss-Seidel iteration count")
        .def_readonly("residual",   &FixedSourceResult::residual,
            "Final relative flux change norm")
        .def_readonly("converged",  &FixedSourceResult::converged,
            "True when the iteration met its tolerance");

    // ------------------------------------------------------------------
    // FixedSourceSolver
    // ------------------------------------------------------------------
    py::class_<FixedSourceSolver>(m, "FixedSourceSolver",
        "Matrix-free 1-D multigroup neutron diffusion fixed-source solver.\n\n"
        "Solves  A phi = q  where q is a user-supplied external source.\n"
        "No fission or power iteration is performed.\n\n"
        "source layout: [cells * n_groups], row-major - same as flux output.")
        .def(py::init<Materials,
                      std::vector<int>,
                      std::vector<double>,
                      Geometry,
                      std::vector<BoundaryCondition>,
                      double, int, bool>(),
             py::arg("mats"),
             py::arg("medium_map"),
             py::arg("edges_x"),
             py::arg("geom"),
             py::arg("bc"),
             py::arg("epsilon")   = 1e-8,
             py::arg("max_inner") = 200,
             py::arg("verbose")   = false)
        .def("solve", &FixedSourceSolver::solve,
             py::arg("source"),
             "Solve A*phi = source and return a FixedSourceResult.",
             py::call_guard<py::gil_scoped_release>());

    // ------------------------------------------------------------------
    // TimeDependentResult
    // ------------------------------------------------------------------
    py::class_<TimeDependentResult>(m, "TimeDependentResult")
        .def_readonly("flux",  &TimeDependentResult::flux,
            "Physical flux [cells * n_groups], row-major: flux[i * n_groups + g]")
        .def_readonly("time",  &TimeDependentResult::time,
            "Total elapsed simulated time (s)")
        .def_readonly("steps", &TimeDependentResult::steps,
            "Number of time steps taken")
        .def_readonly("precursors", &TimeDependentResult::precursors,
            "Delayed neutron precursor concentrations per unit volume,\n"
            "[cells * n_precursor], row-major: precursors[i * n_precursor + p].\n"
            "Empty when the solver was built without delayed neutron data.");

    // ------------------------------------------------------------------
    // TimeDependentSolver
    // ------------------------------------------------------------------
    py::class_<TimeDependentSolver>(m, "TimeDependentSolver",
        "1-D multigroup time-dependent neutron diffusion solver.\n\n"
        "Advances  (1/v_g) d phi_g/dt = -A_g phi_g + fission + scatter + delayed\n"
        "using theta-weighted time differencing.\n\n"
        "Materials.velocity must be set (neutron speed per group, cm/s).\n\n"
        "Fission and scatter are both treated implicitly via Gauss-Seidel, and\n"
        "the delayed precursor balance is integrated in closed form, so the\n"
        "scheme is unconditionally stable.  The time-absorption term\n"
        "1/(theta * v_g * dt) is added to the spatial diagonal each step.\n\n"
        "Pass `delayed` to enable delayed neutron precursors; with the default\n"
        "empty data the solver reduces to prompt-only kinetics.")
        .def(py::init<Materials,
                      std::vector<int>,
                      std::vector<double>,
                      Geometry,
                      std::vector<BoundaryCondition>,
                      std::vector<double>,
                      double, int, bool,
                      DelayedNeutronData,
                      std::vector<double>,
                      double>(),
             py::arg("mats"),
             py::arg("medium_map"),
             py::arg("edges_x"),
             py::arg("geom"),
             py::arg("bc"),
             py::arg("initial_flux") = std::vector<double>{},
             py::arg("epsilon")      = 1e-6,
             py::arg("max_inner")    = 50,
             py::arg("verbose")      = false,
             py::arg("delayed")      = DelayedNeutronData{},
             py::arg("initial_precursors") = std::vector<double>{},
             py::arg("theta")        = 1.0)
        .def("step",   &TimeDependentSolver::step,
             py::arg("dt"),
             "Advance one theta-weighted time step of size dt (seconds).",
             py::call_guard<py::gil_scoped_release>())
        .def("run",    &TimeDependentSolver::run,
             py::arg("dt"), py::arg("n_steps"),
             "Advance n_steps uniform steps and return a TimeDependentResult.",
             py::call_guard<py::gil_scoped_release>())
        .def("result", &TimeDependentSolver::result,
             "Return the current state as a TimeDependentResult.")
        .def("update_materials", &TimeDependentSolver::update_materials,
             py::arg("mats"),
             "Replace the cross sections mid-transient and rebuild the operator.\n"
             "Flux and precursor state are preserved; n_mat and n_groups must not\n"
             "change.  Call once per step with interpolated data to drive a ramp.\n"
             "The change takes effect at the start of the next step - a step\n"
             "insertion at t_n, holding across the whole of [t_n, t_n + dt].")
        .def_property_readonly("time",  &TimeDependentSolver::time)
        .def_property_readonly("steps", &TimeDependentSolver::steps)
        .def_property_readonly("precursors", &TimeDependentSolver::precursors,
             "Precursor concentrations per unit volume [cells * n_precursor].")
        .def_property("theta", &TimeDependentSolver::theta,
                               &TimeDependentSolver::set_theta,
             THETA_DOC);

    // ------------------------------------------------------------------
    // Geometry2D enum
    // ------------------------------------------------------------------
    py::enum_<Geometry2D>(m, "Geometry2D",
        "Coordinate system for 2-D structured mesh problems.")
        .value("XY", Geometry2D::XY, "Cartesian 2-D (x, y)")
        .value("RZ", Geometry2D::RZ,
               "Axisymmetric cylindrical: x = z (axial), y = r (radial)")
        .export_values();

    // ------------------------------------------------------------------
    // UnstructuredMesh2D
    // ------------------------------------------------------------------
    py::class_<UnstructuredMesh2D>(m, "UnstructuredMesh2D",
        "2-D unstructured mesh of polygonal cells.\n\n"
        "Define vertices, cell connectivity, and (optionally) boundary faces.\n\n"
        "  vx, vy         : vertex coordinates [n_verts]\n"
        "  cell_vertices  : flat vertex-index list for all cells\n"
        "  cell_offsets   : size n_cells+1; offsets into cell_vertices\n"
        "                   cell c owns verts [offsets[c] .. offsets[c+1])\n"
        "                   any simple polygon, vertices in order (either winding)\n"
        "  material_id    : material index per cell [n_cells]\n"
        "  bface_v0/v1    : vertex-pair lists defining boundary faces\n"
        "  bface_bc_tag   : BC tag per boundary face (index into bc array)\n"
        "                   defaults to 0 if shorter than bface_v0")
        .def(py::init<>())
        .def_readwrite("vx",           &UnstructuredMesh2D::vx)
        .def_readwrite("vy",           &UnstructuredMesh2D::vy)
        .def_readwrite("cell_vertices",&UnstructuredMesh2D::cell_vertices)
        .def_readwrite("cell_offsets", &UnstructuredMesh2D::cell_offsets)
        .def_readwrite("material_id",  &UnstructuredMesh2D::material_id)
        .def_readwrite("bface_v0",     &UnstructuredMesh2D::bface_v0)
        .def_readwrite("bface_v1",     &UnstructuredMesh2D::bface_v1)
        .def_readwrite("bface_bc_tag", &UnstructuredMesh2D::bface_bc_tag);

    // ------------------------------------------------------------------
    // Unstructured mesh geometry queries
    // ------------------------------------------------------------------
    m.def("validate_mesh", &validate_mesh, py::arg("mesh"),
        "Raise ValueError unless the mesh connectivity is structurally sound:\n"
        "matching vx/vy lengths, cell_offsets starting at 0 and stepping by at\n"
        "least 3 per cell up to len(cell_vertices), vertex indices in range,\n"
        "paired bface_v0/bface_v1, and no zero-area cells.  The solver\n"
        "constructors call this themselves.");

    m.def("cell_centroids", &cell_centroids, py::arg("mesh"),
        "Return (cx, cy) cell centroids, each of length n_cells.\n\n"
        "Area centroids from the shoelace formulae, so any simple polygon works\n"
        "- the same centroids the FVM solvers use internally, so painting\n"
        "materials by centroid position agrees with the solve.");

    m.def("cell_areas", &cell_areas, py::arg("mesh"),
        "Return the area of each cell, length n_cells.");

    // ------------------------------------------------------------------
    // KEigenSolver2D
    // ------------------------------------------------------------------
    py::class_<KEigenSolver2D>(m, "KEigenSolver2D",
        "Matrix-free 2-D multigroup neutron diffusion k-eigenvalue solver\n"
        "on a structured Cartesian or RZ mesh.\n\n"
        "Flux output: flat [nx*ny * n_groups], row-major flux[(i*ny+j)*G+g].\n"
        "Reshape to (nx, ny, G) in NumPy.\n\n"
        "Left (x=0) and bottom (y=0) boundaries are always reflective.\n"
        "bc_x specifies the right (x=nx) Robin BC per group.\n"
        "bc_y specifies the top  (y=ny) Robin BC per group.")
        .def(py::init<Materials,
                      std::vector<int>,
                      std::vector<double>,
                      std::vector<double>,
                      Geometry2D,
                      std::vector<BoundaryCondition>,
                      std::vector<BoundaryCondition>,
                      double, int, int, bool, std::optional<bool>>(),
             py::arg("mats"),
             py::arg("medium_map"),
             py::arg("edges_x"),
             py::arg("edges_y"),
             py::arg("geom"),
             py::arg("bc_x"),
             py::arg("bc_y"),
             py::arg("epsilon")   = 1e-8,
             py::arg("max_outer") = 200,
             py::arg("max_inner") = 1000,
             py::arg("verbose")   = false,
             py::arg("use_cg")    = py::none())
        .def("solve", &KEigenSolver2D::solve,
             "Run power iteration and return a DiffusionResult.",
             py::call_guard<py::gil_scoped_release>())
        .def("set_use_cg", &KEigenSolver2D::set_use_cg, py::arg("use_cg"),
             "Select the within-group inner solver: False = line-TDMA\n"
             "Gauss-Seidel; True = matrix-free Jacobi-preconditioned CG.\n"
             "Also a constructor argument; the NDIFFUSION_KEIG_CG env var\n"
             "sets the default when neither is given.");

    // ------------------------------------------------------------------
    // TimeDependentSolver2D
    // ------------------------------------------------------------------
    py::class_<TimeDependentSolver2D>(m, "TimeDependentSolver2D",
        "2-D multigroup time-dependent neutron diffusion solver\n"
        "on a structured Cartesian or RZ mesh.\n\n"
        "Uses theta-weighted time differencing with an implicit fission source\n"
        "and delayed neutron precursors integrated in closed form.\n"
        "Materials.velocity must be set (neutron speed per group, cm/s).\n\n"
        "Pass `delayed` to enable delayed neutron precursors; with the default\n"
        "empty data the solver reduces to prompt-only kinetics.")
        .def(py::init<Materials,
                      std::vector<int>,
                      std::vector<double>,
                      std::vector<double>,
                      Geometry2D,
                      std::vector<BoundaryCondition>,
                      std::vector<BoundaryCondition>,
                      std::vector<double>,
                      double, int, bool,
                      DelayedNeutronData,
                      std::vector<double>,
                      double>(),
             py::arg("mats"),
             py::arg("medium_map"),
             py::arg("edges_x"),
             py::arg("edges_y"),
             py::arg("geom"),
             py::arg("bc_x"),
             py::arg("bc_y"),
             py::arg("initial_flux") = std::vector<double>{},
             py::arg("epsilon")      = 1e-6,
             py::arg("max_inner")    = 50,
             py::arg("verbose")      = false,
             py::arg("delayed")      = DelayedNeutronData{},
             py::arg("initial_precursors") = std::vector<double>{},
             py::arg("theta")        = 1.0)
        .def("step",   &TimeDependentSolver2D::step,   py::arg("dt"),
             "Advance one theta-weighted step of size dt (seconds).",
             py::call_guard<py::gil_scoped_release>())
        .def("run",    &TimeDependentSolver2D::run,
             py::arg("dt"), py::arg("n_steps"),
             "Advance n_steps uniform steps and return a TimeDependentResult.",
             py::call_guard<py::gil_scoped_release>())
        .def("result", &TimeDependentSolver2D::result,
             "Return the current state as a TimeDependentResult.")
        .def("update_materials", &TimeDependentSolver2D::update_materials,
             py::arg("mats"),
             "Replace the cross sections mid-transient and rebuild the operator.\n"
             "Flux and precursor state are preserved; n_mat and n_groups must not\n"
             "change.  Call once per step with interpolated data to drive a ramp.\n"
             "The change takes effect at the start of the next step - a step\n"
             "insertion at t_n, holding across the whole of [t_n, t_n + dt].")
        .def_property_readonly("time",  &TimeDependentSolver2D::time)
        .def_property_readonly("steps", &TimeDependentSolver2D::steps)
        .def_property_readonly("precursors", &TimeDependentSolver2D::precursors,
             "Precursor concentrations per unit volume [nx*ny * n_precursor].")
        .def_property("theta", &TimeDependentSolver2D::theta,
                               &TimeDependentSolver2D::set_theta,
             THETA_DOC);

    // ------------------------------------------------------------------
    // FixedSourceSolver2D
    // ------------------------------------------------------------------
    py::class_<FixedSourceSolver2D>(m, "FixedSourceSolver2D",
        "Matrix-free 2-D multigroup neutron diffusion fixed-source solver\n"
        "on a structured Cartesian or RZ mesh.\n\n"
        "Solves  A phi = q  where q is a user-supplied volumetric source.\n"
        "No fission or power iteration is performed.\n\n"
        "source layout: [nx*ny * n_groups], row-major - same as flux output.\n"
        "Left (x=0) and bottom (y=0) boundaries are always reflective.\n"
        "bc_x specifies the right (x=nx) Robin BC per group.\n"
        "bc_y specifies the top  (y=ny) Robin BC per group.")
        .def(py::init<Materials,
                      std::vector<int>,
                      std::vector<double>,
                      std::vector<double>,
                      Geometry2D,
                      std::vector<BoundaryCondition>,
                      std::vector<BoundaryCondition>,
                      double, int, bool>(),
             py::arg("mats"),
             py::arg("medium_map"),
             py::arg("edges_x"),
             py::arg("edges_y"),
             py::arg("geom"),
             py::arg("bc_x"),
             py::arg("bc_y"),
             py::arg("epsilon")   = 1e-8,
             py::arg("max_inner") = 200,
             py::arg("verbose")   = false)
        .def("solve", &FixedSourceSolver2D::solve,
             py::arg("source"),
             "Solve A*phi = source and return a FixedSourceResult.",
             py::call_guard<py::gil_scoped_release>());

    // ------------------------------------------------------------------
    // KEigenSolverUnstructured2D
    // ------------------------------------------------------------------
    py::class_<KEigenSolverUnstructured2D>(m, "KEigenSolverUnstructured2D",
        "Matrix-free 2-D multigroup neutron diffusion k-eigenvalue solver\n"
        "on an unstructured triangular/quadrilateral mesh.\n\n"
        "Uses cell-centred finite-volume method with point Gauss-Seidel.\n\n"
        "Flux output: flat [n_cells * n_groups], row-major flux[c*G+g].\n\n"
        "bc has size n_bc_types * n_groups; bc[tag*G+g] is the BC for\n"
        "tag 'tag', group g.  Boundary faces with no matching bc_tag use tag 0.")
        .def(py::init<Materials,
                      UnstructuredMesh2D,
                      std::vector<BoundaryCondition>,
                      double, int, int, bool, std::optional<bool>>(),
             py::arg("mats"),
             py::arg("mesh"),
             py::arg("bc"),
             py::arg("epsilon")   = 1e-8,
             py::arg("max_outer") = 200,
             py::arg("max_inner") = 1000,
             py::arg("verbose")   = false,
             py::arg("use_cg")    = py::none())
        .def("solve", &KEigenSolverUnstructured2D::solve,
             "Run power iteration and return a DiffusionResult.",
             py::call_guard<py::gil_scoped_release>())
        .def("set_use_cg", &KEigenSolverUnstructured2D::set_use_cg,
             py::arg("use_cg"),
             "Select the within-group inner solver: False = point\n"
             "Gauss-Seidel; True = matrix-free Jacobi-preconditioned CG.\n"
             "Also a constructor argument; the NDIFFUSION_KEIG_CG env var\n"
             "sets the default when neither is given.");

    // ------------------------------------------------------------------
    // TimeDependentSolverUnstructured2D
    // ------------------------------------------------------------------
    py::class_<TimeDependentSolverUnstructured2D>(m,
        "TimeDependentSolverUnstructured2D",
        "2-D multigroup time-dependent neutron diffusion solver\n"
        "on an unstructured triangular/quadrilateral mesh.\n\n"
        "Uses theta-weighted time differencing with an implicit fission source\n"
        "and delayed neutron precursors integrated in closed form.\n"
        "Materials.velocity must be set (neutron speed per group, cm/s).\n\n"
        "Precursor concentrations are stored per unit volume, matching the\n"
        "volumetric source convention of the fixed-source solver.")
        .def(py::init<Materials,
                      UnstructuredMesh2D,
                      std::vector<BoundaryCondition>,
                      std::vector<double>,
                      double, int, bool,
                      DelayedNeutronData,
                      std::vector<double>,
                      double>(),
             py::arg("mats"),
             py::arg("mesh"),
             py::arg("bc"),
             py::arg("initial_flux") = std::vector<double>{},
             py::arg("epsilon")      = 1e-6,
             py::arg("max_inner")    = 50,
             py::arg("verbose")      = false,
             py::arg("delayed")      = DelayedNeutronData{},
             py::arg("initial_precursors") = std::vector<double>{},
             py::arg("theta")        = 1.0)
        .def("step",   &TimeDependentSolverUnstructured2D::step,  py::arg("dt"),
             "Advance one theta-weighted step of size dt (seconds).",
             py::call_guard<py::gil_scoped_release>())
        .def("run",    &TimeDependentSolverUnstructured2D::run,
             py::arg("dt"), py::arg("n_steps"),
             "Advance n_steps uniform steps and return a TimeDependentResult.",
             py::call_guard<py::gil_scoped_release>())
        .def("result", &TimeDependentSolverUnstructured2D::result,
             "Return the current state as a TimeDependentResult.")
        .def("update_materials",
             &TimeDependentSolverUnstructured2D::update_materials,
             py::arg("mats"),
             "Replace the cross sections mid-transient and rebuild the operator.\n"
             "Flux and precursor state are preserved; n_mat and n_groups must not\n"
             "change.  Call once per step with interpolated data to drive a ramp.\n"
             "The change takes effect at the start of the next step - a step\n"
             "insertion at t_n, holding across the whole of [t_n, t_n + dt].")
        .def_property_readonly("time",  &TimeDependentSolverUnstructured2D::time)
        .def_property_readonly("steps", &TimeDependentSolverUnstructured2D::steps)
        .def_property_readonly("precursors",
             &TimeDependentSolverUnstructured2D::precursors,
             "Precursor concentrations per unit volume [n_cells * n_precursor].")
        .def_property("theta", &TimeDependentSolverUnstructured2D::theta,
                               &TimeDependentSolverUnstructured2D::set_theta,
             THETA_DOC);

    // ------------------------------------------------------------------
    // FixedSourceSolverUnstructured2D
    // ------------------------------------------------------------------
    py::class_<FixedSourceSolverUnstructured2D>(m,
        "FixedSourceSolverUnstructured2D",
        "Matrix-free 2-D multigroup neutron diffusion fixed-source solver\n"
        "on an unstructured triangular/quadrilateral mesh.\n\n"
        "Solves  A phi = q  using point Gauss-Seidel.\n\n"
        "source layout: [n_cells * n_groups], row-major - same as flux output.\n"
        "Source values are volumetric; the solver multiplies by cell_area\n"
        "internally to form the volume-integrated RHS.\n\n"
        "bc has size n_bc_types * n_groups; bc[tag*G+g] is the BC for\n"
        "tag 'tag', group g.")
        .def(py::init<Materials,
                      UnstructuredMesh2D,
                      std::vector<BoundaryCondition>,
                      double, int, double, bool>(),
             py::arg("mats"),
             py::arg("mesh"),
             py::arg("bc"),
             py::arg("epsilon")   = 1e-8,
             py::arg("max_inner") = 200,
             py::arg("omega")     = 1.0,
             py::arg("verbose")   = false)
        .def("solve", &FixedSourceSolverUnstructured2D::solve,
             py::arg("source"),
             "Solve A*phi = source and return a FixedSourceResult.",
             py::call_guard<py::gil_scoped_release>());
}
