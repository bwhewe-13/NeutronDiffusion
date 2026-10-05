# ndiffusion

Multigroup neutron diffusion in 1-D and 2-D. The solvers are written in C++17
and exposed to Python through pybind11; the Python package adds the pieces
around them - cross-section builders, meshes and layouts, kinetics data,
post-processing and a discretization-error estimator.

```python
import numpy as np
import ndiffusion as nd

mats = nd.materials.one_group(D=3.850204978408833, sigma_a=0.1532, nusigf=0.1570)
solver = nd.KEigenSolver(mats, [0] * 50, np.linspace(0.0, 100.0, 51),
                         nd.Geometry.Sphere, [nd.BoundaryCondition(A=1.0, B=0.0)],
                         max_outer=500)
print(solver.solve().keff)   # 1.00000475
```

Every solver is built and run the same way, across three meshes and three
problem types:

| | k-eigenvalue | fixed source | time dependent |
|---|---|---|---|
| 1-D slab, cylinder, sphere | `KEigenSolver` | `FixedSourceSolver` | `TimeDependentSolver` |
| 2-D structured XY, RZ | `KEigenSolver2D` | `FixedSourceSolver2D` | `TimeDependentSolver2D` |
| 2-D unstructured polygons | `KEigenSolverUnstructured2D` | `FixedSourceSolverUnstructured2D` | `TimeDependentSolverUnstructured2D` |

No system matrix is ever assembled. The structured solvers use tridiagonal
(Thomas) solves inside a Gauss-Seidel sweep over groups, and the unstructured
solver is a cell-centered finite volume scheme with a deferred non-orthogonal
correction, so triangles and skewed cells stay second order.

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

user_guide/geometry
user_guide/materials
user_guide/boundary_conditions
user_guide/solvers
user_guide/kinetics
user_guide/postprocessing
user_guide/verification
user_guide/cpp
```

```{toctree}
:maxdepth: 1
:caption: Reference

api/python
api/cpp
```

```{toctree}
:maxdepth: 1
:caption: Project

roadmap
changelog
contributing
citing
```
