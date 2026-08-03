"""Preliminary 5-qubit LNN brickwork simulation with quimb (tensor networks).

Builds precisely the 5-qubit, LNN fully-entangling brickwork circuit (the kind
of proxy circuit used in Merkel et al., arXiv:2503.05943) and reports the
final bitstring distribution + the wall-clock time.

The circuit is NOISELESS and unitary, so the output is a *deterministic*
distribution P(x)=|<x|psi>|^2.  We compute it EXACTLY, once, from an MPS built
with quimb's swap+split contraction -- not by Monte-Carlo'ing shots (shots would
only add sampling noise to an answer we can get exactly). Pass SHOTS>0 to also
draw a finite hardware-style sample.

Run:  python examples/run_5q_lnn.py
"""

from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore")

import _bootstrap  # noqa: F401
from proxysim import lnn_brickwork
from proxysim.backends import TensorNetworkBackend, total_variation_distance

N = 5
N_CYCLES = N - 1          # "cycle the two layers N-1 times" -> full entanglement
SHOTS = 0                 # 0 = exact distribution only; set >0 for a finite sample
SEED = 1234


def main():
    circ = lnn_brickwork(N, N_CYCLES, twoq="cz", mode="clifford", seed=SEED)
    print(circ.summary())

    tn = TensorNetworkBackend()
    if not tn.available:
        print("quimb not available:", tn.import_error)
        return

    print(f"  MPS max bond dimension: {tn.max_bond_of(circ)} "
          f"(small => low entanglement => cheap)")
    print()

    t0 = time.perf_counter()
    result = tn.run(circ, shots=SHOTS, seed=SEED)
    wall = time.perf_counter() - t0

    print(f"Tensor-network simulation ({tn.name})")
    print(f"  method        : {result.method}")
    print(f"  exact-dist time: {result.t_compute * 1e3:8.3f} ms")
    if SHOTS:
        print(f"  sample time    : {result.t_sample * 1e3:8.3f} ms   ({SHOTS} shots)")
    print(f"  total wall     : {wall * 1e3:8.3f} ms")
    print()
    print(f"Exact final distribution ({result.support()} outcomes with P>0):")
    for bs, p in result.top(16):
        bar = "#" * int(round(p * 50))
        print(f"  |{bs}>  P={p:.4f}  {bar}")
    print("\n(stabilizer state -> uniform over a coset; here every P = 1/support.)")

    if SHOTS:
        sp = result.sampled_probs()
        print(f"\n  TVD(sampled, exact) = {total_variation_distance(sp, result.distribution):.3e}"
              f"  (pure shot noise)")


if __name__ == "__main__":
    main()
