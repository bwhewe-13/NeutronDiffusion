<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/bwhewe-13/NeutronDiffusion/master/docs/_static/logo-dark.svg">
  <img alt="ndiffusion" src="https://raw.githubusercontent.com/bwhewe-13/NeutronDiffusion/master/docs/_static/logo.svg" width="320">
</picture>

[![CI](https://github.com/bwhewe-13/NeutronDiffusion/actions/workflows/ci.yml/badge.svg)](https://github.com/bwhewe-13/NeutronDiffusion/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/bwhewe-13/NeutronDiffusion/branch/master/graph/badge.svg)](https://codecov.io/gh/bwhewe-13/NeutronDiffusion)
[![Docs](https://github.com/bwhewe-13/NeutronDiffusion/actions/workflows/docs.yml/badge.svg)](https://bwhewe-13.github.io/NeutronDiffusion/)
[![PyPI](https://img.shields.io/pypi/v/ndiffusion.svg)](https://pypi.org/project/ndiffusion/)
[![Python](https://img.shields.io/pypi/pyversions/ndiffusion.svg)](https://pypi.org/project/ndiffusion/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23212416.svg)](https://doi.org/10.5281/zenodo.23212416)

Multigroup neutron diffusion solver for 1-D and 2-D geometries. The solvers are
written in C++17 and exposed to Python through pybind11; the Python package adds
cross-section builders, meshes and core layouts, kinetics data, post-processing
and a discretization-error estimator.

**Documentation:** <https://bwhewe-13.github.io/NeutronDiffusion/>

## Features

- k-eigenvalue, fixed-source and time-dependent solvers on 1-D slab, cylinder
  and sphere meshes, 2-D structured XY and RZ meshes, and 2-D unstructured
  polygonal meshes - nine solvers, all built and run the same way
- Matrix-free throughout: tridiagonal (Thomas) solves inside a Gauss-Seidel
  group sweep on structured meshes, and a cell-centered finite volume scheme
  with a deferred non-orthogonal correction on unstructured ones, so triangles
  and skewed cells stay second order
- Robin boundary conditions on every edge (vacuum, reflective, albedo),
  boundary tags and periodic boundaries on unstructured meshes
- Delayed neutron precursors, theta-weighted time differencing up to
  Crank-Nicolson, and mid-transient perturbations
- Gmsh import, material assignment separate from the geometry, and preset
  Cartesian and hexagonal cores with symmetry sectors
- Published two-group benchmarks (Ringhals, TWIGL, IAEA, BIBLIS) and a
  transport-to-diffusion cross-section conversion
- Post-processing (power normalization, reaction rates, peaking factors), adjoint
  cross sections, and single-mesh error estimates by the method of nearby problems
- numpy in and out, Python convergence warnings, and Ctrl-C during a solve

## Installation

```bash
pip install ndiffusion
```

Wheels are published for Linux (x86_64, aarch64), macOS (Intel, Apple
silicon) and Windows, for Python 3.9 to 3.14. Elsewhere pip builds from source,
which needs a C++17 compiler.

For development, `pip install -e ".[dev]"` installs everything CI runs; re-run it
after editing C++ sources to rebuild the extension. See the
[installation guide](https://bwhewe-13.github.io/NeutronDiffusion/installation.html)
for the optional extras.

## Example

A bare uranium sphere, one energy group, zero flux at the surface:

```python
import numpy as np
import ndiffusion as nd

m = nd.Materials()
m.n_mat    = 1
m.n_groups = 1
m.D        = [3.850204978408833]
m.removal  = [0.1532]
m.scatter  = [0.0]
m.chi      = [1.0]
m.nusigf   = [0.1570]

cells = 50
solver = nd.KEigenSolver(
    mats       = m,
    medium_map = [0] * cells,
    edges_x    = np.linspace(0.0, 100.0, cells + 1),
    geom       = nd.Geometry.Sphere,
    bc         = [nd.BoundaryCondition(A=1.0, B=0.0)],
    max_outer  = 500,
)
result = solver.solve()
print(f"keff = {result.keff:.8f}")   # -> 1.00000475
```

<img alt="Flux in the bare sphere against the exact sin(Br)/Br" src="https://raw.githubusercontent.com/bwhewe-13/NeutronDiffusion/master/docs/_static/readme_sphere.png" width="520">

`result.flux` is a numpy array of shape `(n_cells, n_groups)`. The
[quickstart](https://bwhewe-13.github.io/NeutronDiffusion/quickstart.html) goes on
to 2-D structured and unstructured problems, and `examples/` has runnable scripts
for each solver family.

## Documentation

- [User guide](https://bwhewe-13.github.io/NeutronDiffusion/user_guide/geometry.html) -
  meshes, cross sections, boundary conditions, solvers, kinetics,
  post-processing and verification
- [Examples](https://bwhewe-13.github.io/NeutronDiffusion/auto_examples/index.html)
  and [benchmarks](https://bwhewe-13.github.io/NeutronDiffusion/benchmarks.html)
- [Theory](https://bwhewe-13.github.io/NeutronDiffusion/theory/diffusion.html) -
  the equations, the discretizations and the iterative methods
- [Python API](https://bwhewe-13.github.io/NeutronDiffusion/api/python.html) and
  [C++ API](https://bwhewe-13.github.io/NeutronDiffusion/cpp/)
- [Roadmap](TODO.md) and [changelog](CHANGELOG.md)

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Without the `dev` extra, parts of the suite are skipped rather than failing; run
`pytest -rs` to see what was skipped.

## Citing

If ndiffusion contributes to published work, please cite it:

```bibtex
@software{whewell_ndiffusion,
  author    = {Whewell, Ben},
  title     = {ndiffusion: multigroup neutron diffusion in {C++} and {Python}},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23212416},
  url       = {https://github.com/bwhewe-13/NeutronDiffusion},
}
```

That DOI always resolves to the latest release; each version also has its own
(see [Citing](https://bwhewe-13.github.io/NeutronDiffusion/citing.html)).

## License

MIT - see [LICENSE](LICENSE).
