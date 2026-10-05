# Cross sections

## The `Materials` object

All cross sections live in one `Materials` object. Its arrays are flat and
row-major, indexed by material `m` and group `g`:

| Field | Size | Meaning |
|---|---|---|
| `D` | `n_mat * n_groups` | diffusion coefficient (cm) |
| `removal` | `n_mat * n_groups` | removal cross section (1/cm) |
| `scatter` | `n_mat * n_groups * n_groups` | group transfer, `scatter[m][g_to][g_from]` |
| `chi` | `n_mat * n_groups` | fission spectrum |
| `nusigf` | `n_mat * n_groups` | nu times the fission cross section |
| `velocity` | `n_groups` | neutron speed (cm/s), time-dependent solvers only |

Any array-like of the right total size can be assigned, and multi-dimensional
arrays are flattened in C order, so `scatter` can be given as an
`(n_mat, n_groups, n_groups)` array. Fields read back as numpy arrays. Each read
is a copy, so assign a whole array to change a value; editing an element of the
copy does nothing.

Two conventions are easy to get wrong:

- `scatter[m][g_to][g_from]` is the transfer *into* `g_to` *from* `g_from`.
  Transport libraries usually store the transpose.
- `removal` is absorption plus total out-scatter minus self-scatter, and the
  self-scatter diagonal of `scatter` must then be zero, since it is already
  accounted for. The builders below take care of this.

Every solver constructor checks the array sizes against `n_mat` and `n_groups`
and raises `ValueError` on a mismatch.

### Fission-matrix mode

If `chi` is all zeros and `nusigf` has `n_mat * n_groups * n_groups` entries,
`nusigf` is read as a full fission transfer matrix `F[m][g_to][g_from]` instead
of a separable `chi` and `nusigf` pair. This is how libraries with
incident-energy dependent spectra are represented.

## Builders

For the common one- and two-group cases:

```python
import numpy as np
import ndiffusion as nd

one = nd.materials.one_group(D=1.0, sigma_a=0.05, nusigf=0.06)

two = nd.materials.two_group([
    # D1,  D2,   Sa1,   Sa2,  S12,  nuSf1, nuSf2
    (1.4, 0.4, 0.010, 0.080, 0.02, 0.006, 0.135),   # fuel
    (1.3, 0.3, 0.001, 0.020, 0.03, 0.000, 0.000),   # reflector
])
```

`two_group` takes rows in the `(D1, D2, Sa1, Sa2, S12, nuSf1, nuSf2)` form that
two-group benchmarks are published in: fission neutrons born fast, down-scatter
only, removal built from absorption plus out-scatter. An optional
`axial_buckling` adds `D_g B^2` to the removal, standing in for the leakage of a
finite core height.

`make_materials(data_list, G)` builds a `Materials` from a list of dicts, one per
material - for example the contents of `.npz` files. Each needs `D`, `Siga`,
`Scat` and `nuSigf`, and optionally `chi` and a precomputed `Removal`. `Scat` is
in solver order `[g_to][g_from]` by default; pass `scatter_orientation="from_to"`
for data stored the other way round, and it is transposed on the way in.

```python
fuel = {
    "D": [1.4, 0.4], "Siga": [0.010, 0.080], "nuSigf": [0.006, 0.135],
    "chi": [1.0, 0.0], "Scat": [[0.0, 0.0], [0.02, 0.0]],
}
mats = nd.make_materials([fuel], G=2)
```

## Published benchmarks

`ndiffusion.materials` holds the cross sections of published benchmark problems
as `Benchmark` records, each with `.materials()`, `.reference_keff` and
`.source`, plus `.layout()` where the loading is an assembly map. Geometry is
deliberately not bundled, so a table runs on any mesh.

| Name | Problem | Reference keff |
|---|---|---|
| `RINGHALS` | Ringhals-4 1-D slab | 1.0037 |
| `TWIGL` | TWIGL 2-D quarter core | 0.9133 |
| `IAEA` | IAEA PWR 2-D quarter core | 1.0296 |
| `BIBLIS` | BIBLIS 2-D full core | 1.02535 |

The Ringhals slab, solved on its half domain:

```python
bench = nd.materials.RINGHALS
edges = np.linspace(0.0, 279.5, 1119)
mmap = nd.make_medium_map([(0, 161.25), (1, 279.5 - 161.25)], edges=edges)

solver = nd.KEigenSolver(bench.materials(), mmap, edges, nd.Geometry.Slab,
                         bc=[nd.BoundaryCondition(A=0.5, B=1.3116),
                             nd.BoundaryCondition(A=0.5, B=0.2624)],
                         max_outer=2000, max_inner=100)
keff = solver.solve().keff
print(keff, bench.reference_keff)
```

`tests/test_benchmarks.py` solves every table to its published eigenvalue, which
is what pins the numbers. `from_assembly_map(amap, pitch)` turns a published
assembly grid into a painter for {func}`ndiffusion.assign_materials`.

## Transport cross sections

Multigroup transport libraries tabulate a total (or absorption) cross section, a
scatter matrix and fission data; the solvers want `D`, a removal cross section
and a diagonal-free scatter matrix. `make_materials_from_transport` does the
conversion and returns a ready-to-use `Materials`:

```python
uo2 = {
    "SigT": [0.5, 1.2],
    "Scat": [[0.45, 0.03], [0.0, 1.1]],     # [g_from][g_to]
    "Scat1": [[0.08, 0.0], [0.0, 0.25]],    # P1 moments, for the outflow correction
    "nuSigf": [0.008, 0.18],
    "chi": [1.0, 0.0],
}
mats = nd.make_materials_from_transport([uo2], G=2)
```

Each input dict holds `SigT` or `Siga` (the other is derived from the scatter
matrix), a `Scat` matrix, `nuSigf`, optionally `chi`, and either `SigTr` or the
P1 scatter matrix `Scat1`:

| | |
|---|---|
| `D[g] = 1 / (3 Sigma_tr[g])` | `Sigma_tr` from a tabulated `SigTr`, else the P1 outflow correction `SigT - sum_g' Scat1[g->g']` (`transport_correction="none"` gives the uncorrected `1/(3 SigT)`) |
| `Sigma_r[g] = SigT[g] - Scat[g->g]` | the outflow correction cancels here, so removal uses the uncorrected total |
| `scatter[g_to][g_from]` | input is taken as `Scat[g_from][g_to]` (the transport convention) and transposed; pass `scatter_orientation="to_from"` for data already in solver order |

`transport_to_diffusion(data, G)` applies the same transform to a single
material and returns a plain dict, which is useful for inspecting the derived
`D`, `Removal` and `SigTr` before building a `Materials`.
`examples/transport_cross_sections.py` is a runnable end-to-end example.

## Adjoint cross sections

`make_adjoint_materials(mats)` returns the adjoint cross sections: the group
scatter transposed, `chi` and `nusigf` swapped (or the fission matrix
transposed), `D` and `removal` unchanged. Running any solver on them solves the
adjoint problem - the k-eigenvalue matches the forward one, and the flux is the
importance function.

```python
adj = nd.make_adjoint_materials(two)
```
