# Using the C++ library

The solvers are an ordinary C++17 static library, `ndiffusion_core`, with no
Python dependency. The Python extension and the standalone driver are both thin
layers over it.

## Building

```bash
cmake -B build
cmake --build build
./build/cpp/ndiffusion_driver
```

A bare `cmake` configures a Release build and skips the Python extension
(`NDIFFUSION_BUILD_PYTHON` is on only under scikit-build). The driver solves a
set of 1-D reference problems and prints each eigenvalue against its reference.

## Calling the solvers

The classes and structs have the same names and arguments as in Python, from the
headers in `cpp/include/ndiffusion/`: `types.hpp` for `Materials`,
`BoundaryCondition` and the result structs, `solver_1d.hpp` for the 1-D solvers
and `solver_2d.hpp` for both kinds of 2-D solver.

```cpp
#include <ndiffusion/solver_1d.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    Materials m;
    m.n_mat    = 1;
    m.n_groups = 1;
    m.D        = {3.850204978408833};
    m.removal  = {0.1532};
    m.scatter  = {0.0};
    m.chi      = {1.0};
    m.nusigf   = {0.1570};

    const int cells = 50;
    std::vector<double> edges(cells + 1);
    for (int i = 0; i <= cells; ++i) edges[i] = 100.0 * i / cells;

    KEigenSolver solver(m, std::vector<int>(cells, 0), edges, Geometry::Sphere,
                        {BoundaryCondition{1.0, 0.0}}, 1e-8, 500);
    DiffusionResult result = solver.solve();
    std::cout << std::setprecision(9) << result.keff << "\n";   // 1.00000475
}
```

To use the library from another CMake project, add this repository as a
subdirectory and link the target:

```cmake
add_subdirectory(NeutronDiffusion)
target_link_libraries(my_app PRIVATE ndiffusion_core)
```

Arrays are `std::vector` throughout, in the flat row-major layouts described in
{doc}`materials` and {doc}`geometry`. Invalid input throws
`std::invalid_argument`. Without the Python hook, a convergence warning is
printed to standard error.

## Reference documentation

The C++ API reference is generated with Doxygen from the comments in the
headers and sources (see {doc}`../api/cpp`):

```bash
cmake --build build --target docs     # or: doxygen Doxyfile
```

The output goes to `docs/doxygen/html/`.
