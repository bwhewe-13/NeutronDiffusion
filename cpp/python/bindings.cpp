#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ndiffusion/solver_1d.hpp>
#include <ndiffusion/solver_2d.hpp>

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "ndiffusion C++ backend — 1-D and 2-D multigroup neutron diffusion solvers";

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
        "                     nusigf[m][g_to][g_from]) — activated when chi is all zeros\n"
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
        .def_readonly("residual",   &DiffusionResult::residual);

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
             py::arg("max_inner") = 50,
             py::arg("verbose")   = false)
        .def("solve", &KEigenSolver::solve,
             "Run power iteration and return a DiffusionResult.");

    // ------------------------------------------------------------------
    // FixedSourceResult
    // ------------------------------------------------------------------
    py::class_<FixedSourceResult>(m, "FixedSourceResult")
        .def_readonly("flux",       &FixedSourceResult::flux,
            "Physical flux [cells * n_groups], row-major: flux[i * n_groups + g]")
        .def_readonly("iterations", &FixedSourceResult::iterations,
            "Gauss-Seidel iteration count")
        .def_readonly("residual",   &FixedSourceResult::residual,
            "Final flux change norm");

    // ------------------------------------------------------------------
    // FixedSourceSolver
    // ------------------------------------------------------------------
    py::class_<FixedSourceSolver>(m, "FixedSourceSolver",
        "Matrix-free 1-D multigroup neutron diffusion fixed-source solver.\n\n"
        "Solves  A phi = q  where q is a user-supplied external source.\n"
        "No fission or power iteration is performed.\n\n"
        "source layout: [cells * n_groups], row-major — same as flux output.")
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
             "Solve A·phi = source and return a FixedSourceResult.");

    // ------------------------------------------------------------------
    // TimeDependentResult
    // ------------------------------------------------------------------
    py::class_<TimeDependentResult>(m, "TimeDependentResult")
        .def_readonly("flux",  &TimeDependentResult::flux,
            "Physical flux [cells * n_groups], row-major: flux[i * n_groups + g]")
        .def_readonly("time",  &TimeDependentResult::time,
            "Total elapsed simulated time (s)")
        .def_readonly("steps", &TimeDependentResult::steps,
            "Number of time steps taken");

    // ------------------------------------------------------------------
    // TimeDependentSolver
    // ------------------------------------------------------------------
    py::class_<TimeDependentSolver>(m, "TimeDependentSolver",
        "1-D multigroup time-dependent neutron diffusion solver.\n\n"
        "Advances  (1/v_g) d phi_g/dt = -A_g phi_g + fission + scatter\n"
        "using backward Euler time differencing.\n\n"
        "Materials.velocity must be set (neutron speed per group, cm/s).\n\n"
        "Fission is treated explicitly (from the previous time step);\n"
        "scatter is treated implicitly via Gauss-Seidel.  The time-absorption\n"
        "term 1/(v_g * dt) is added to the spatial diagonal each step.")
        .def(py::init<Materials,
                      std::vector<int>,
                      std::vector<double>,
                      Geometry,
                      std::vector<BoundaryCondition>,
                      std::vector<double>,
                      double, int, bool>(),
             py::arg("mats"),
             py::arg("medium_map"),
             py::arg("edges_x"),
             py::arg("geom"),
             py::arg("bc"),
             py::arg("initial_flux") = std::vector<double>{},
             py::arg("epsilon")      = 1e-6,
             py::arg("max_inner")    = 50,
             py::arg("verbose")      = false)
        .def("step",   &TimeDependentSolver::step,
             py::arg("dt"),
             "Advance one backward-Euler time step of size dt (seconds).")
        .def("run",    &TimeDependentSolver::run,
             py::arg("dt"), py::arg("n_steps"),
             "Advance n_steps uniform steps and return a TimeDependentResult.")
        .def("result", &TimeDependentSolver::result,
             "Return the current state as a TimeDependentResult.")
        .def_property_readonly("time",  &TimeDependentSolver::time)
        .def_property_readonly("steps", &TimeDependentSolver::steps);

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
        "2-D unstructured mesh of triangles and/or quadrilaterals.\n\n"
        "Define vertices, cell connectivity, and (optionally) boundary faces.\n\n"
        "  vx, vy         : vertex coordinates [n_verts]\n"
        "  cell_vertices  : flat vertex-index list for all cells\n"
        "  cell_offsets   : size n_cells+1; offsets into cell_vertices\n"
        "                   cell c owns verts [offsets[c] .. offsets[c+1])\n"
        "                   3 verts → triangle, 4 verts → quad\n"
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
                      double, int, int, bool>(),
             py::arg("mats"),
             py::arg("medium_map"),
             py::arg("edges_x"),
             py::arg("edges_y"),
             py::arg("geom"),
             py::arg("bc_x"),
             py::arg("bc_y"),
             py::arg("epsilon")   = 1e-8,
             py::arg("max_outer") = 200,
             py::arg("max_inner") = 50,
             py::arg("verbose")   = false)
        .def("solve", &KEigenSolver2D::solve,
             "Run power iteration and return a DiffusionResult.");

    // ------------------------------------------------------------------
    // TimeDependentSolver2D
    // ------------------------------------------------------------------
    py::class_<TimeDependentSolver2D>(m, "TimeDependentSolver2D",
        "2-D multigroup time-dependent neutron diffusion solver\n"
        "on a structured Cartesian or RZ mesh.\n\n"
        "Uses backward Euler time differencing.\n"
        "Materials.velocity must be set (neutron speed per group, cm/s).")
        .def(py::init<Materials,
                      std::vector<int>,
                      std::vector<double>,
                      std::vector<double>,
                      Geometry2D,
                      std::vector<BoundaryCondition>,
                      std::vector<BoundaryCondition>,
                      std::vector<double>,
                      double, int, bool>(),
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
             py::arg("verbose")      = false)
        .def("step",   &TimeDependentSolver2D::step,   py::arg("dt"),
             "Advance one backward-Euler step of size dt (seconds).")
        .def("run",    &TimeDependentSolver2D::run,
             py::arg("dt"), py::arg("n_steps"),
             "Advance n_steps uniform steps and return a TimeDependentResult.")
        .def("result", &TimeDependentSolver2D::result,
             "Return the current state as a TimeDependentResult.")
        .def_property_readonly("time",  &TimeDependentSolver2D::time)
        .def_property_readonly("steps", &TimeDependentSolver2D::steps);

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
                      double, int, int, bool>(),
             py::arg("mats"),
             py::arg("mesh"),
             py::arg("bc"),
             py::arg("epsilon")   = 1e-8,
             py::arg("max_outer") = 200,
             py::arg("max_inner") = 50,
             py::arg("verbose")   = false)
        .def("solve", &KEigenSolverUnstructured2D::solve,
             "Run power iteration and return a DiffusionResult.");

    // ------------------------------------------------------------------
    // TimeDependentSolverUnstructured2D
    // ------------------------------------------------------------------
    py::class_<TimeDependentSolverUnstructured2D>(m,
        "TimeDependentSolverUnstructured2D",
        "2-D multigroup time-dependent neutron diffusion solver\n"
        "on an unstructured triangular/quadrilateral mesh.\n\n"
        "Uses backward Euler time differencing.\n"
        "Materials.velocity must be set (neutron speed per group, cm/s).")
        .def(py::init<Materials,
                      UnstructuredMesh2D,
                      std::vector<BoundaryCondition>,
                      std::vector<double>,
                      double, int, bool>(),
             py::arg("mats"),
             py::arg("mesh"),
             py::arg("bc"),
             py::arg("initial_flux") = std::vector<double>{},
             py::arg("epsilon")      = 1e-6,
             py::arg("max_inner")    = 50,
             py::arg("verbose")      = false)
        .def("step",   &TimeDependentSolverUnstructured2D::step,  py::arg("dt"),
             "Advance one backward-Euler step of size dt (seconds).")
        .def("run",    &TimeDependentSolverUnstructured2D::run,
             py::arg("dt"), py::arg("n_steps"),
             "Advance n_steps uniform steps and return a TimeDependentResult.")
        .def("result", &TimeDependentSolverUnstructured2D::result,
             "Return the current state as a TimeDependentResult.")
        .def_property_readonly("time",  &TimeDependentSolverUnstructured2D::time)
        .def_property_readonly("steps", &TimeDependentSolverUnstructured2D::steps);
}
