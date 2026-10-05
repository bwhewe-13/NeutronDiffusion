# Contributing

Bug reports and pull requests are welcome on
[GitHub](https://github.com/bwhewe-13/NeutronDiffusion).

## Setting up

```bash
git clone https://github.com/bwhewe-13/NeutronDiffusion.git
cd NeutronDiffusion
pip install -e ".[dev]"
pytest
```

Python changes under `src/ndiffusion/` are live in the editable install; C++
changes need `pip install -e ".[dev]"` again before Python sees them.

## Layout

```text
cpp/
  include/ndiffusion/
    types.hpp               Materials, BoundaryCondition, geometry enums, results, meshes
    solver_1d.hpp           1-D solvers
    solver_2d.hpp           2-D structured and unstructured solvers, mesh queries
    solver_detail.hpp       shared internals (validation, BCs, kinetics algebra, hooks)
    solver_3d.hpp           placeholder
  src/
    solver_1d.cpp
    solver_2d_structured.cpp
    solver_2d_unstructured.cpp
    main.cpp                standalone driver (1-D reference problems)
  python/
    bindings.cpp            pybind11 bindings -> ndiffusion._core

src/ndiffusion/
  __init__.py               public API
  create.py                 make_materials, make_medium_map, boundary_conditions
  mesh.py                   load_gmsh, assign_materials, copy_mesh
  layouts.py                preset geometries, symmetry orientations, painters
  materials.py              cross-section builders and published benchmarks
  transport.py              transport -> diffusion cross sections
  adjoint.py                adjoint cross sections
  kinetics.py               delayed neutron data, critical scaling
  nearby.py                 method of nearby problems
  postprocess.py            volumes, reaction rates, power, save/load

tests/                      pytest suite, one file per area
examples/                   runnable scripts
tools/c5g7_fuel_mesh.py     regenerates the gitignored C5G7 meshes
docs/                       this site (Sphinx sources)
```

## What CI checks

- `pytest` on Linux, macOS and Windows, with the oldest and a recent supported
  Python
- `ruff check src tests examples tools docs`
- the C++ driver built and run under AddressSanitizer and
  UndefinedBehaviorSanitizer
- the documentation, built with warnings as errors

## Conventions

- A new C++ class or method needs a binding in `cpp/python/bindings.cpp`, with
  a docstring, and long-running methods release the GIL.
- Flux, source and cross-section arrays are flat and row-major in C++; see
  {doc}`user_guide/materials` and {doc}`user_guide/geometry` for the layouts.
- New behavior comes with a test. Convergence warnings are errors in the test
  suite, so a test that is meant to stop at an iteration cap uses
  `pytest.warns(nd.ConvergenceWarning)`.
- Add a line to the `Unreleased` section of `CHANGELOG.md`.
