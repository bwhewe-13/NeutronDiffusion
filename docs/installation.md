# Installation

ndiffusion needs Python 3.9 or newer and numpy. Building from source also needs
a C++17 compiler and CMake; pip fetches pybind11 and scikit-build-core itself.

```bash
pip install .
```

## Optional extras

| Extra | Installs | Needed for |
|---|---|---|
| `nearby` | scipy | the method of nearby problems ({doc}`user_guide/verification`) |
| `mesh` | gmsh | `load_gmsh`, reading `.msh` files |
| `test` | pytest, scipy | running the test suite |
| `dev` | pytest, scipy, ruff, matplotlib | everything CI runs |
| `docs` | sphinx, furo, myst-parser and friends | building this site |

```bash
pip install ".[nearby,mesh]"
```

## Development install

```bash
pip install -e ".[dev]"
```

This is an editable install: Python edits under `src/ndiffusion/` are picked up
immediately, but the compiled `_core` extension is not. After editing any C++
source, run `pip install -e ".[dev]"` again to rebuild it.

If an import still picks up old behavior after a rebuild, look for a stale
`_core*.so` left in `src/ndiffusion/` or an older non-editable copy of the
package in `site-packages`; either one shadows the new build.

## Running the tests

```bash
pytest
```

Optional dependencies are handled with `pytest.importorskip`, so a bare
`pip install -e .` gives a run that looks clean while quietly skipping tests.
Without scipy that is the point-kinetics reference comparisons in
`tests/test_kinetics.py` - the main kinetics validation - plus all of
`tests/test_nearby_*.py`. Run `pytest -rs` to see what was skipped and why.
`tests/test_mesh_gmsh.py` needs the `mesh` extra, which is heavier and not part
of `dev`.

The suite turns `ndiffusion.ConvergenceWarning` into an error, so a test that
stops at an iteration cap fails rather than passing on an unconverged answer.

## Building the documentation

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going docs docs/_build/html
```

The C++ reference is generated separately with Doxygen (see {doc}`api/cpp`).
