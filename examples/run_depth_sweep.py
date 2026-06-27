"""Depth sweep: how circuit depth drives scrambling and per-backend cost.

For a fixed width (N qubits) we vary the number of brickwork cycles (=> depth)
and report, at each depth:
    * circuit depth, two-qubit-gate count, and the MPS max bond dimension
      reached (a direct entanglement proxy),
    * the entropy of the exact output distribution (bits) and its support size --
      for a Clifford brickwork the support is a power of 2 (a stabilizer coset),
      so entropy = log2(support); it climbs then fluctuates with stabilizer rank,
    * the wall-clock time each backend needs to compute the EXACT distribution.

Run:  python examples/run_depth_sweep.py
"""

from __future__ import annotations

import math
import os
import time
import warnings

warnings.filterwarnings("ignore")

import _bootstrap  # noqa: F401
from proxysim import lnn_brickwork
from proxysim.backends import StabilizerBackend, StatevectorBackend, TensorNetworkBackend

N = 6
CYCLES = [1, 2, 3, 4, 5, 6, 8, 10]
SEED = 11
OUT = os.path.join(_bootstrap.RESULTS_DIR, "depth_sweep.txt")


def entropy_bits(dist):
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def time_exact(backend, circ):
    backend.exact_distribution(circ)  # warm
    t0 = time.perf_counter()
    backend.exact_distribution(circ)
    return (time.perf_counter() - t0) * 1e3


def main():
    tn, sv, stab = TensorNetworkBackend(), StatevectorBackend(), StabilizerBackend()
    lines = [f"Depth sweep on N={N} qubits (Clifford brickwork) -- EXACT distribution",
             f"{'cycles':>7}{'depth':>7}{'2q':>5}{'maxbond':>9}{'entropy(b)':>12}"
             f"{'support':>9}{'TN(ms)':>9}{'SV(ms)':>9}{'stim(ms)':>9}",
             "-" * 77]
    for c in CYCLES:
        circ = lnn_brickwork(N, c, twoq="cz", mode="clifford", seed=SEED)
        dist = sv.exact_distribution(circ)
        H, nout = entropy_bits(dist), len(dist)
        row = (f"{c:>7}{circ.depth:>7}{circ.n_two_qubit:>5}{tn.max_bond_of(circ):>9}"
               f"{H:>12.3f}{nout:>9}"
               f"{time_exact(tn, circ):>9.2f}{time_exact(sv, circ):>9.2f}"
               f"{time_exact(stab, circ):>9.2f}")
        lines.append(row)

    text = "\n".join(lines)
    print(text)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nWritten: {OUT}")


if __name__ == "__main__":
    main()
