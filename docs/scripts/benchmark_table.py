"""Print the published-versus-computed keff table for docs/benchmarks.md.

Each benchmark is solved on two meshes, the second twice as fine, and
Richardson-extrapolated assuming second order.  Takes a few minutes.

    python docs/scripts/benchmark_table.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "examples"))

import _benchmarks as B  # noqa: E402

import ndiffusion as nd  # noqa: E402

CASES = [
    ("Ringhals-4 1-D slab", nd.materials.RINGHALS, B.solve_ringhals, (0.25, 0.125), "h = {} cm"),
    ("TWIGL 2-D quarter core", nd.materials.TWIGL, B.solve_twigl, (1.0, 0.5), "h = {} cm"),
    ("IAEA 2-D quarter core", nd.materials.IAEA, B.solve_iaea, (2.5, 1.25), "h = {} cm"),
    ("BIBLIS 2-D full core", nd.materials.BIBLIS, B.solve_biblis, (8, 16), "{} cells per assembly"),
]

print(f"ndiffusion {nd.__version__}\n")
print("| Benchmark | Published | Coarse | Fine | Extrapolated | Difference (pcm) |")
print("|---|---|---|---|---|---|")
for name, bench, solve, meshes, label in CASES:
    k = [solve(m).keff for m in meshes]
    k_ext = k[1] + (k[1] - k[0]) / 3.0
    diff = (k_ext - bench.reference_keff) * 1e5
    print(f"| {name} | {bench.reference_keff} | {k[0]:.5f} ({label.format(meshes[0])}) "
          f"| {k[1]:.5f} ({label.format(meshes[1])}) | {k_ext:.5f} | {diff:+.0f} |", flush=True)
