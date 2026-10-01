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
- GitHub Actions CI on Linux, macOS and Windows, ruff linting and a
  sanitizer build of the C++ driver

### Changed
- `make_materials` takes `scatter_orientation`; `descending_energy=True` is
  deprecated and `descending_energy=False` raises
- The 1-D k-eigenvalue `max_inner` default is 1000, and the C++ solvers are
  quiet by default like the Python bindings
- A bare CMake configure builds Release, and warning flags apply to every
  target

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

## [0.3.0] - 2026-06-22

Gmsh import for unstructured meshes. Earlier history is in the git log.
