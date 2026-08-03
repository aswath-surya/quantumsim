"""Multi-backend comparison on the 5-qubit LNN brickwork circuits.

Each backend computes the EXACT output distribution by a totally different method
(dense statevector / MPS contraction / stabilizer tableau); since the circuit is
noiseless the three must agree to floating-point precision -- that is the headline
cross-check (TVD@ref ~ 1e-15).  Set SHOTS>0 to also draw finite samples.

Two circuits (mirroring Merkel et al.):
  1. Clifford brickwork -- efficiently simulable "proxy" (all 3 backends).
  2. Haar  brickwork    -- non-Clifford "target" (statevector + MPS; stim skipped).

Run:  python examples/run_compare.py
"""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")

import _bootstrap  # noqa: F401
from proxysim import lnn_brickwork, run_all

N = 5
N_CYCLES = N - 1
SHOTS = 0                 # 0 = exact only; >0 also draws a finite hardware sample
SEED = 1234
OUT = os.path.join(_bootstrap.RESULTS_DIR, "comparison.txt")


def main():
    blocks = []
    for title, mode in [("CLIFFORD PROXY", "clifford"), ("HAAR TARGET", "haar")]:
        circ = lnn_brickwork(N, N_CYCLES, twoq="cz", mode=mode, seed=SEED)
        rep = run_all(circ, shots=SHOTS, seed=SEED)
        block = f"\n##### {title} #####\n" + rep.to_text(top_k=8)
        blocks.append(block)
        print(block)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))
    print(f"\nWritten: {OUT}")


if __name__ == "__main__":
    main()
