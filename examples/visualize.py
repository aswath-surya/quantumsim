"""Generate visualizations: the CIRCUIT (qiskit diagram + quimb tensor network),
the OUTPUT distribution, and the Pauli-propagation coefficient spread -- saved as
PNGs plus one combined panel.

A lightly-magic brickwork is used: the injected T-gates make the output
distribution non-uniform AND give Pauli propagation a real coefficient spread to
show. Being non-Clifford, it also makes the stabilizer (stim) backend skip
automatically -- which is exactly the point (magic breaks stabilizer
simulability, so you fall back to TN / statevector / Pauli propagation).

Run:  python examples/visualize.py
Outputs: results/viz_circuit_qiskit.png, viz_tn_quimb.png,
         viz_distribution.png, viz_pauliprop.png, viz_panel.png
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import _bootstrap  # noqa: F401
from proxysim import brickwork_magic, run_all
from proxysim import viz

# Compact + a few magic gates: legible diagrams, non-trivial distribution, and a
# meaningful Pauli-propagation coefficient spread.
N = 5
N_CYCLES = 2
N_T = 6
SEED = 1234


def main():
    circ = brickwork_magic(N, N_CYCLES, n_t=N_T, seed=SEED)
    print(circ.summary())

    report = run_all(circ, shots=0, seed=SEED)  # exact distribution (stim auto-skipped)
    paths = viz.panel(circ, report, _bootstrap.RESULTS_DIR, prefix="viz")

    print("\nWrote:")
    for k, p in paths.items():
        print(f"  {k:8s} {p}")


if __name__ == "__main__":
    main()
