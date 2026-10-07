# Contributing

Bug reports and pull requests are welcome.

```bash
git clone https://github.com/bwhewe-13/NeutronDiffusion.git
cd NeutronDiffusion
pip install -e ".[dev]"
pytest
```

Python changes under `src/ndiffusion/` are picked up immediately; after a C++
change, run `pip install -e ".[dev]"` again to rebuild the extension.

Before opening a pull request:

- `pytest` passes. Convergence warnings are errors in the suite, so a test that
  stops at an iteration cap on purpose says so with
  `pytest.warns(nd.ConvergenceWarning)`.
- `ruff check src tests examples tools docs` is clean.
- New C++ classes and methods have a binding, with a docstring, in
  `cpp/python/bindings.cpp`.
- New behavior has a test and a line in the `Unreleased` section of
  `CHANGELOG.md`.

The [contributing page](https://bwhewe-13.github.io/NeutronDiffusion/contributing.html)
of the documentation describes the repository layout and what CI checks.
