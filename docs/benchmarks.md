# Benchmarks

The cross sections of four published two-group diffusion benchmarks ship in
{mod}`ndiffusion.materials`, each with its reference eigenvalue and source.
`tests/test_benchmarks.py` solves all four and checks the eigenvalue, which is
what pins the tables.

## Eigenvalues

Each benchmark below is solved on two meshes, the second twice as fine, and
Richardson-extrapolated assuming second order. The table comes from
`docs/scripts/benchmark_table.py` (ndiffusion 1.0.0).

| Benchmark | Solver | Published | Coarse | Fine | Extrapolated | Difference |
|---|---|---|---|---|---|---|
| Ringhals-4 1-D slab {cite:p}`yu2024` | 1-D | 1.0037 | 1.00367 (h = 0.25 cm) | 1.00367 (h = 0.125 cm) | 1.00367 | -3 pcm |
| TWIGL 2-D quarter core {cite:p}`yu2024` | 2-D structured | 0.9133 | 0.91318 (h = 1 cm) | 0.91320 (h = 0.5 cm) | 0.91321 | -9 pcm |
| IAEA 2-D quarter core {cite:p}`yu2024` | 2-D unstructured | 1.0296 | 1.02943 (h = 2.5 cm) | 1.02954 (h = 1.25 cm) | 1.02958 | -2 pcm |
| BIBLIS 2-D full core {cite:p}`femffusion2023` | 2-D unstructured | 1.02535 | 1.02540 (8 per assembly) | 1.02529 (16 per assembly) | 1.02525 | -10 pcm |

The published values are themselves numerical solutions, quoted to four or five
digits, so differences of a few pcm are within their rounding. The BIBLIS
reference is FEMFFUSION's mesh-converged diffusion (SP1) eigenvalue; its cross
sections originate with Nakata and Martin (1983).

## Geometry and boundary conditions

| Benchmark | Geometry | Boundary conditions |
|---|---|---|
| Ringhals-4 | slab, core to 161.25 cm, reflector to 279.5 cm, solved on the half | symmetry at 0; Robin $0.5\,\phi + D\,\partial\phi/\partial n = 0$ outside |
| TWIGL | 80 cm square quarter core, seed and blanket | symmetry at x = 0, y = 0; zero flux outside |
| IAEA | stepped quarter core in 20 cm assemblies, with an axial buckling of $0.8\times10^{-4}\ \text{cm}^{-2}$ | symmetry at x = 0, y = 0; Robin $0.4692\,\phi + D\,\partial\phi/\partial n = 0$ outside |
| BIBLIS | full core, 17 x 17 assemblies of pitch 23.1226 cm | Marshak vacuum outside |

The IAEA and BIBLIS cores are not rectangles, so they run on the unstructured
solver with a quad mesh of the stepped outline. The geometries are built in
`examples/_benchmarks.py`.

## Convergence

The {doc}`auto_examples/plot_benchmarks` example solves TWIGL and IAEA on a
sequence of meshes and plots the eigenvalue error against the cell size.
