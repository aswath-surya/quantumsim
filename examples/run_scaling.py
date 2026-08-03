"""Scaling contrast as N grows -- EXACT output distribution.

Shallow Clifford brickwork (fixed small depth) so all three backends apply.
Reporting the time to compute the EXACT distribution:
  * Every method must write down 2^N probabilities, so the FULL distribution is
    inherently exponential for all three -- they all top out ~25 qubits. The
    methods differ only in how they get there (dense evolution / MPS contraction
    / stabilizer tableau + dense readout).
  * The genuinely scalable Clifford operation is *sampling*, not the full
    distribution: stim's native CHP sampler handles thousands of qubits. The
    bottom table shows stim sampling (run(shots=...)) scaling far past where the
    exact distribution is computable.

Run:  python examples/run_scaling.py
"""

from __future__ import annotations

import os
import time
import warnings

warnings.filterwarnings("ignore")

import _bootstrap  # noqa: F401
from proxysim import lnn_brickwork
from proxysim.backends import StabilizerBackend, StatevectorBackend, TensorNetworkBackend

N_CYCLES = 2
SEED = 7
EXACT_MAX = 20
N_FULL = [5, 10, 14, 17, 20]
N_BIG = [50, 100, 500, 1000]
SAMPLE_SHOTS = 2000
OUT = os.path.join(_bootstrap.RESULTS_DIR, "scaling.txt")


def time_exact(backend, circ):
    backend.exact_distribution(circ)
    t0 = time.perf_counter()
    backend.exact_distribution(circ)
    return (time.perf_counter() - t0) * 1e3


def main():
    tn, sv, stab = TensorNetworkBackend(), StatevectorBackend(), StabilizerBackend()
    lines = ["EXACT full distribution -- time to compute (ms); shallow Clifford brickwork",
             f"{'N':>5}{'2q':>6}{'maxbond':>9}{'TN (ms)':>12}{'SV (ms)':>14}{'stim (ms)':>12}",
             "-" * 58]
    for n in N_FULL:
        circ = lnn_brickwork(n, N_CYCLES, twoq="cz", mode="clifford", seed=SEED)
        ok = n <= EXACT_MAX
        t_tn = f"{time_exact(tn, circ):>12.2f}" if ok else f"{'(2^N)':>12}"
        t_sv = f"{time_exact(sv, circ):>14.2f}" if ok else f"{'(2^N)':>14}"
        t_st = f"{time_exact(stab, circ):>12.2f}" if ok else f"{'(2^N)':>12}"
        lines.append(f"{n:>5}{circ.n_two_qubit:>6}{tn.max_bond_of(circ):>9}{t_tn}{t_sv}{t_st}")

    lines += ["", f"NATIVE stim SAMPLING (run shots={SAMPLE_SHOTS}) -- scales far past 2^N:",
              f"{'N':>6}{'2q':>7}{'stim sample (ms)':>20}", "-" * 33]
    for n in N_BIG:
        circ = lnn_brickwork(n, N_CYCLES, twoq="cz", mode="clifford", seed=SEED)
        stab.run(circ, shots=16, seed=SEED, want_exact=False)  # warm
        t0 = time.perf_counter()
        stab.run(circ, shots=SAMPLE_SHOTS, seed=SEED, want_exact=False)
        lines.append(f"{n:>6}{circ.n_two_qubit:>7}{(time.perf_counter() - t0) * 1e3:>20.2f}")

    text = "\n".join(lines)
    print(text)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nWritten: {OUT}")


if __name__ == "__main__":
    main()
