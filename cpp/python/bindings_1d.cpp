#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ndiffusion/solver_1d.hpp>

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "ndiffusion C++ backend — 1-D multigroup neutron diffusion solvers";

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
        "  D, removal, chi, nusigf : [n_mat * n_groups]\n"
        "  scatter                 : [n_mat * n_groups * n_groups]  "
                                    "(scatter[m][g_to][g_from])\n"
        "  velocity                : [n_groups]  neutron speed (cm/s)\n\n"
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
    // DiffusionSolver
    // ------------------------------------------------------------------
    py::class_<DiffusionSolver>(m, "DiffusionSolver",
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
        .def("solve", &DiffusionSolver::solve,
             "Run power iteration and return a DiffusionResult.");

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
}
