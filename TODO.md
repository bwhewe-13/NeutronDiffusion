# TODO

## 1.0.0

API changes go in before 1.0 so they don't need a 2.0 later.

- [x] numpy arrays in and out of the solvers - inputs accept any array-like,
  results come back as `(n_cells, n_groups)` arrays instead of Python lists
- [x] Robin boundary conditions on every edge - `bc_left` in 1-D, `bc_x_left`
  and `bc_y_bottom` in 2-D structured, reflective by default
- [x] Convergence warnings raised through Python's `warnings` module
  (`ndiffusion.ConvergenceWarning`) instead of printed to stderr, and a
  `converged` flag on `TimeDependentResult` (the k-eigenvalue and fixed-source
  results already have one)
- [x] Post-processing - normalize to a total power, reaction rate densities,
  region powers and peaking factors, save/load results
- [x] Documentation site: theory, user guide, examples gallery, API reference
- [x] Logo and README badges
- [x] Prebuilt wheels on PyPI
- [x] `CITATION.cff`, `.zenodo.json` and a Zenodo DOI for the release

## After 1.0

**Geometry**
- 3-D structured geometry (x-y-z) and 3-D unstructured (tetrahedra/hexahedra)

**Physics**
- Automatic time-step control, using the difference between the `theta = 1` and
  `theta = 0.5` answers as a local error estimate
- Improved quasi-static or adiabatic kinetics, factoring the flux into a point
  kinetics amplitude and a slowly varying shape
- Thermal-hydraulic feedback (Doppler / moderator density) driving
  `update_materials` from the power distribution
- Sensitivity and perturbation analysis built on the adjoint importance
  function (`make_adjoint_materials` already gives the adjoint operator)
- Depletion coupling - Bateman equations for nuclide inventory evolution

**Solvers and performance**
- Flip the default inner solver for the 2-D k-eigenvalue solvers to the
  within-group CG (now a `use_cg` constructor option; default remains
  Gauss-Seidel, overridable via `NDIFFUSION_KEIG_CG=1`); extend CG to the
  fixed-source and time-dependent solvers, replacing hand-tuned SOR
- Power-iteration acceleration (Wielandt shift or Chebyshev extrapolation);
  CMFD (Coarse Mesh Finite Difference) for unstructured k-eigenvalue convergence.
  The transient inner iteration already uses Aitken extrapolation
  (`FissionAccelerator`); the same idea would apply to the k-eigenvalue outer
- OpenMP parallelism for the spatial sweep loops

**Testing**
- A CI-sized C5G7 diffusion regression. The quarter core already runs end-to-end
  in `examples/c5g7_quarter_core.py`, but it takes minutes and needs gmsh and
  h5py
- The TWIGL kinetics transient (`TestTwiglKinetics`) currently validates against
  its own static reactivity rather than the benchmark's published power history;
  digitizing that history would turn it into a true published regression
