# Python API

Everything listed here is importable from the top-level `ndiffusion` package.
The solver classes, `Materials`, the mesh and the result types are compiled
from C++; the rest is pure Python.

```{eval-rst}
.. currentmodule:: ndiffusion
```

## Solvers

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   KEigenSolver
   FixedSourceSolver
   TimeDependentSolver
   KEigenSolver2D
   FixedSourceSolver2D
   TimeDependentSolver2D
   KEigenSolverUnstructured2D
   FixedSourceSolverUnstructured2D
   TimeDependentSolverUnstructured2D
```

## Inputs and results

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Materials
   BoundaryCondition
   DelayedNeutronData
   Geometry
   Geometry2D
   UnstructuredMesh2D
   DiffusionResult
   FixedSourceResult
   TimeDependentResult
   ConvergenceWarning
```

## Building inputs

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   make_materials
   make_materials_from_transport
   transport_to_diffusion
   make_medium_map
   boundary_conditions
   make_adjoint_materials
   make_delayed_data
   scale_to_critical
   kinetics.DELAYED_U235_6GROUP
```

## Meshes

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   load_gmsh
   assign_materials
   copy_mesh
   validate_mesh
   cell_centroids
   cell_areas
```

## Post-processing

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   cell_volumes
   reaction_rate
   power_density
   normalize_to_power
   region_powers
   peaking_factors
   save_result
   load_result
   postprocess.KAPPA_U235
   postprocess.NU_U235
```

## Solution verification

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   nearby_fixed_source
   nearby_k_eigenvalue
   fission_source
   NearbyFixedResult
   NearbyKResult
```

## Modules

```{eval-rst}
.. autosummary::
   :toctree: generated

   layouts
   materials
```
