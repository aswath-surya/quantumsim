"""CB-based error bound vs actual TVD for a real circuit: MQT QPE (qpe11.qasm).

Loads the 12-qubit (11 counting + 1 ancilla) Quantum Phase Estimation circuit,
cycle-benchmarks its CX cycles, forms the QCAP bound, and compares it to the actual
noisy total-variation distance as the noise strength is swept.

Choices (see the chat that produced this):
  * CB cycles  = ASAP parallel CX layers (each distinct layer pattern is a cycle;
    under the uniform NoiseModel its e_F depends only on how many CX are in the
    layer, so identical-structure layers share one benchmark).
  * sweep      = noise strength.
  * actual     = full-distribution TVD. The ideal QPE output is a DELTA at the
    correct phase (585), so TVD(noisy, ideal) = 1 - P_measured(585) exactly.

Simulators: the ideal distribution is read off the tensor-network (MPS) backend
(exact; its final bond is 1 because the answer is a product state). The noisy TVD
is trajectory-averaged on the statevector backend, because at n=12 QPE's all-to-all
inverse-QFT makes MPS SWAP-bound and ~25x slower -- MPS pays off for local,
bounded-entanglement circuits, which QPE is not. Set SIM='tensornetwork' below to
run the (much slower) MPS version, e.g. on a larger circuit.

Run:  python examples/run_qpe_bounding.py
"""

from __future__ import annotations

import dataclasses
import os
import random
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector

import _bootstrap  # noqa: F401
from proxysim import NoiseModel, save_results
from proxysim.circuit import circuit_from_qasm
from proxysim.noise import sample_trajectory
from proxysim.backends import StatevectorBackend, TensorNetworkBackend
from proxysim.benchmarking import cycle_benchmark, qcap_bound, readout_fidelity

QASM = os.path.join(os.path.dirname(__file__), "qpe11.qasm")
BASE = NoiseModel(enabled=True, p1=2e-4, p2=1e-3, p_readout=1e-3, p_idle=2e-4)
SCALES = [0.0, 1.0, 2.0, 4.0, 8.0]
N_TRAJ = 1500
SEED = 5
OUT = _bootstrap.RESULTS_DIR + "/qpe_bounding.png"
OUT_DATA = _bootstrap.RESULTS_DIR + "/qpe_bounding_data.npz"

_SV = StatevectorBackend()


def cx_layer_patterns(circ):
    """ASAP-layerize; return Counter{frozenset(cx pairs): number of layers}."""
    patterns = Counter()
    for layer in circ.layers():
        pairs = frozenset(tuple(g.qubits) for g in layer if len(g.qubits) == 2)
        if pairs:
            patterns[pairs] += 1
    return patterns


def cb_bound(circ, patterns, n_measured, noise):
    """QCAP bound: benchmark each distinct CX-layer structure (cached by #CX under
    the uniform noise model), weight by how often it occurs."""
    ro_fid, ro_std = readout_fidelity(n_measured, noise, seed=3)
    ef_by_ncx = {}
    counts, efs = {}, {}
    for i, (pat, cnt) in enumerate(patterns.items()):
        k = len(pat)
        if k not in ef_by_ncx:
            cb = cycle_benchmark(list(pat), circ.n_qubits, [1, 2, 4, 8], noise,
                                 twoq="CX", n_decays=15, shots=800, seed=100 + k)
            ef_by_ncx[k] = (cb["e_F"], cb["e_F_std"])
        counts[f"pat{i}"] = cnt
        efs[f"pat{i}"] = ef_by_ncx[k]
    return qcap_bound(counts, efs, ro_fid, ro_std)["error"], ef_by_ncx, ro_fid


def p_target_sv(circ, idx0, idx1):
    d = Statevector(_SV._build(circ)).data
    return abs(d[idx0]) ** 2 + abs(d[idx1]) ** 2      # P(target) summed over ancilla


def noisy_tvd(circ, noise, idx0, idx1, ro_fid, seed):
    """1 - P_measured(target): the delta-ideal TVD, from statevector trajectories."""
    noise = dataclasses.replace(noise, p_readout=0.0)   # readout folded in via ro_fid
    rng = random.Random(seed)
    acc = sum(p_target_sv(sample_trajectory(circ, noise, rng), idx0, idx1)
              for _ in range(N_TRAJ)) / N_TRAJ
    return 1 - acc * ro_fid


def main():
    circ, measured = circuit_from_qasm(open(QASM).read())
    ancilla = [q for q in range(circ.n_qubits) if q not in measured]
    print(circ.summary(), "| measured:", len(measured), "| ancilla:", ancilla)

    # ideal distribution via the tensor-network backend -> the peak (target)
    ideal = TensorNetworkBackend().exact_distribution(circ)
    marg = Counter()
    for b, p in ideal.items():
        marg["".join(b[q] for q in measured)] += p
    target = max(marg, key=marg.get)
    print(f"ideal peak: |{target}> (int={int(target, 2)}) p={marg[target]:.4f}, "
          f"support={len(marg)}  -> TVD = 1 - P(peak)")

    # statevector indices for the target on the measured bits, summed over ancilla
    full0 = ["0"] * circ.n_qubits
    for q, b in zip(measured, target):
        full0[q] = b
    idx0 = sum(int(b) << j for j, b in enumerate(full0))
    full1 = list(full0); full1[ancilla[0]] = "1"
    idx1 = sum(int(b) << j for j, b in enumerate(full1))

    patterns = cx_layer_patterns(circ)
    print(f"CX layers: {sum(patterns.values())} total, {len(patterns)} distinct patterns, "
          f"sizes {sorted(set(len(p) for p in patterns))}\n")

    bounds, tvds = [], []
    print(f"{'scale':>6}{'bound':>10}{'actual TVD':>12}")
    for s in SCALES:
        noise = BASE.scaled(s) if s > 0 else dataclasses.replace(BASE, enabled=True,
                                                                 p1=0, p2=0, p_readout=0, p_idle=0)
        b, _, ro_fid = cb_bound(circ, patterns, len(measured), noise)
        t = noisy_tvd(circ, noise, idx0, idx1, ro_fid, seed=SEED)
        bounds.append(b)
        tvds.append(t)
        save_results(OUT_DATA, scales=np.array(SCALES[:len(bounds)]),
                     bound=np.array(bounds), tvd=np.array(tvds))
        print(f"{s:>6.1f}{b:>10.4f}{t:>12.4f}", flush=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(SCALES, bounds, "-", color="#D55E00", lw=2.5, label="CB bound")
    ax.plot(SCALES, tvds, "o-", color="#0072B2", ms=7, label="actual TVD (1 - P(585))")
    ax.set_xlabel("noise scale factor")
    ax.set_ylabel("probability of an error (TVD)")
    ax.set_title("QPE (12q, MQT qpe11): cycle-benchmark bound vs actual error\n"
                 "CB from ASAP CX layers; actual = full-distribution TVD (delta ideal)",
                 fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"\nWrote {OUT}, {OUT_DATA}")


if __name__ == "__main__":
    main()
