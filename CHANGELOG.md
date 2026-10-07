# Changelog

All notable changes to ndiffusion are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- Delayed neutron precursors in all three time-dependent solvers, with an
  implicit fission source and `update_materials` for mid-transient
  perturbations; `make_delayed_data`, `scale_to_critical` and the Keepin
  6-group set `DELAYED_U235_6GROUP`
- Theta-weighted time differencing (`theta = 0.5` is Crank-Nicolson,
  `theta = 1` the default backward Euler)
- Deferred non-orthogonal correction with least-squares gradients, so the
  unstructured solvers stay second order on triangles and skewed cells
- Polygonal unstructured cells (hexagons and the clipped cells of a symmetry
  sector), plus `validate_mesh`, `cell_centroids` and `cell_areas`
- `assign_materials` and `copy_mesh`, so one mesh can carry several material
  layouts; Gmsh physical-group names are kept on the mesh as `region_names`
  and `bc_names`
- Periodic boundary pairs, translational or rotational
- `ndiffusion.layouts`: Cartesian and hex core meshes, symmetry orientations
  and material painters
- `ndiffusion.materials`: one- and two-group builders and the published
  Ringhals, TWIGL, IAEA and BIBLIS benchmark tables
- Transport-to-diffusion cross-section conversion
  (`make_materials_from_transport`, `transport_to_diffusion`)
- `make_adjoint_materials` for the adjoint (importance) problem
- Method of nearby problems discretization-error estimator (`nearby_*`)
- Within-group CG inner solver for the 2-D k-eigenvalue solvers (`use_cg`)
- Published benchmark regression tests and a C5G7 quarter-core example
- Ctrl-C interrupts a running solve instead of waiting for it to finish
- `ndiffusion.__version__`
- `n_cells` and `n_groups` on every solver, and `n_groups` (plus `n_precursor`
  for transients) on the result objects
- GitHub Actions CI on Linux, macOS and Windows, ruff linting and a
  sanitizer build of the C++ driver
- Robin boundary conditions on the 1-D left edge (`bc_left`) and the 2-D
  structured left and bottom edges (`bc_x_left`, `bc_y_bottom`). They default
  to reflective, the old fixed behavior, so a full core no longer has to be cut
  to a quarter, and a 1-D cylinder or sphere can have an inner surface. A
  non-reflective condition on an r = 0 axis raises.
- `ndiffusion.ConvergenceWarning`, and a `converged` flag on
  `TimeDependentResult` (false once any step has hit `max_inner`)
- Post-processing helpers: `cell_volumes`, `reaction_rate`, `power_density`,
  `normalize_to_power`, `region_powers`, `peaking_factors`, and
  `save_result` / `load_result` for `.npz` files
- A documentation site built with Sphinx - installation, quickstart, user
  guide and API reference - deployed to GitHub Pages along with the Doxygen
  C++ reference
- A logo, favicon and social preview image
- Theory pages (the diffusion equation, the 1-D, structured and unstructured
  discretizations, the iterative methods, kinetics and the method of nearby
  problems), an examples gallery with plots, a benchmarks page with
  published-versus-computed eigenvalues, and figures for the README and
  quickstart
- Two new examples: unstructured meshes and symmetry sectors, and benchmark
  convergence for TWIGL and IAEA
- Prebuilt wheels for Linux (x86_64, aarch64), macOS (Intel, Apple silicon) and
  Windows, Python 3.9 to 3.14, published to PyPI from tagged releases
- `CITATION.cff`, `.zenodo.json` and `CONTRIBUTING.md`

### Changed
- The README is shortened to an overview; the detailed usage moved to the
  documentation, and the 2021 notes in `docs/` to `docs/archive/`
- The examples are sphinx-gallery scripts with plots; `examples/kinetics.py`
  plots the power history and the time-differencing convergence
- CI tests Python 3.13 too, runs the examples, and reports coverage
- Arrays cross into and out of Python as numpy arrays instead of lists.
  `flux` is shaped `(n_cells, n_groups)` and `precursors`
  `(n_cells, n_precursor)`; the `Materials`, `DelayedNeutronData` and
  `UnstructuredMesh2D` fields read back as 1-D arrays. Inputs accept any
  array-like, and a fixed source, initial flux or initial precursors can be
  given in the same 2-D shape - a transposed array is rejected rather than
  read in the wrong order. The `nearby_*` results and `fission_source` use the
  same shape.
- `make_materials` takes `scatter_orientation`; `descending_energy=True` is
  deprecated and `descending_energy=False` raises
- The 1-D k-eigenvalue `max_inner` default is 1000, and the C++ solvers are
  quiet by default like the Python bindings
- A bare CMake configure builds Release, and warning flags apply to every
  target
- Convergence warnings go through Python's `warnings` module as
  `ConvergenceWarning` instead of to stderr, so they can be filtered, recorded
  or raised. The k-eigenvalue solvers now also warn when power iteration stops
  at `max_outer`, and the fixed-source solvers when they stop at `max_inner`.
  The transient warning is issued after the step completes, so turning it into
  an exception leaves the solver at a consistent state. The C++ driver still
  prints to stderr.

### Fixed
- 1-D interface coupling now divides by the center-to-center distance, so
  non-uniform meshes converge at second order
- Fixed-source iteration counts no longer overcount by one at the cap
- `make_materials` derives removal from the solver-ordered scatter matrix
- `make_delayed_data` rejects `ChiPrompt` given for only some materials
- Higher-order Gmsh elements are imported by their corner nodes rather than
  silently dropped
- Unstructured meshes are validated: the `bc` array must cover every boundary
  tag, and non-manifold edges, hanging nodes, coincident vertices and zero-area
  cells are rejected
- Boundary faces are emitted in sorted order, so results no longer depend on
  the standard library's hash ordering
- Time steps must be positive and finite
- `copy_mesh` keeps periodic boundary pairs
- The `DelayedNeutronData` docstring no longer says fission-matrix mode is
  rejected, or that an empty `chi_prompt` falls back to `Materials.chi`
- `examples/k_eigenvalue.py` printed the 20-cell slab reference for its sphere;
  the sphere is at its critical radius, so the exact eigenvalue is 1

## [0.3.0] - 2026-06-22

Gmsh import for unstructured meshes. Earlier history is in the git log.
