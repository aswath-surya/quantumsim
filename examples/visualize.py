"""Generate visualizations: the CIRCUIT (qiskit diagram + quimb tensor network)
and the OUTPUT distribution, saved as PNGs plus one combined panel.

Run:  python examples/visualize.py
Outputs: results/viz_circuit_qiskit.png, viz_tn_quimb.png,
         viz_distribution.png, viz_panel.png
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import _bootstrap  # noqa: F401
from proxysim import lnn_brickwork, run_all
from proxysim import viz

# A compact, fully-entangling Clifford brickwork so the diagrams stay legible
# while still exercising all three backends in the distribution panel.
N = 5
N_CYCLES = 2
SEED = 1234


def main():
    circ = lnn_brickwork(N, N_CYCLES, twoq="cz", mode="clifford", seed=SEED)
    print(circ.summary())

    report = run_all(circ, shots=0, seed=SEED)  # exact distribution
    paths = viz.panel(circ, report, _bootstrap.RESULTS_DIR, prefix="viz")

    print("\nWrote:")
    for k, p in paths.items():
        print(f"  {k:8s} {p}")


if __name__ == "__main__":
    main()
