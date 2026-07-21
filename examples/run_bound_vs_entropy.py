"""How the bound-vs-actual gap depends on output entropy (the qcal notebook's
cells 29/45 diagnostic).

Idea: hold the circuit family and noise fixed so the CB bound is CONSTANT, and
only vary the Shannon entropy of the ideal output distribution. Then plot the
actual TVD, and the gap (bound - TVD), against that entropy. The gap should open
up as entropy grows -- errors on a spread output change the distribution less, so
the event-counting bound overshoots more.

The entropy is set exactly by k Hadamards (k free bits); the entangling cycles are
CNOTs (which spread the free bits in the Z basis, unlike diagonal CZ) plus
dimension-preserving single-qubit gates, so the output entropy stays k while the
distribution is genuinely spread and correlated.

TVD stays computable at 20 qubits because the ideal support is only 2^k: we know
those outcomes exactly and estimate the noisy distribution on them (plus the
leakage mass) from stim samples. Writes results/bound_vs_entropy.png.

Run:  python examples/run_bound_vs_entropy.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401
from proxysim import NoiseModel, bench_brickwork, even_pairs, odd_pairs, save_results
from proxysim.simulate import noisy_tvd_vs_support
from proxysim.benchmarking import cycle_benchmark, qcap_bound, readout_fidelity
from proxysim.backends.stabilizer import StabilizerBackend

N = 20
CYCLE_A = even_pairs(N)     # even pairs
CYCLE_B = odd_pairs(N)      # odd pairs
DEPTH = 8                                              # cycles (fixed -> bound fixed)
KS = [0, 2, 4, 6, 8, 10, 12]                          # target entropy (bits)
N_INST = 5                                            # circuit instances per k (smoothing)
NOISE = NoiseModel(enabled=True, p1=2e-4, p2=1e-3, p_readout=2e-3, p_idle=2e-4)
SHOTS = 300_000
SEED = 5
_SB = StabilizerBackend()
OUT = _bootstrap.RESULTS_DIR + "/bound_vs_entropy.png"
OUT_DATA = _bootstrap.RESULTS_DIR + "/bound_vs_entropy_data.npz"


def entropy_circuit(k, seed):
    """Clifford circuit with exactly k bits of output entropy (k Hadamards + DEPTH
    CNOT cycles with dimension-preserving single-qubit layers)."""
    return bench_brickwork(N, DEPTH, [CYCLE_A, CYCLE_B], oneq="dim_preserving",
                           twoq="cx", seed=seed, n_hadamards=k)


def save_fig(H, bound, tvd, tvd_std):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.axhline(bound, ls="-", color="#D55E00", lw=2, label="CB bound (fixed)")
    ax1.errorbar(H, tvd, yerr=tvd_std, fmt="o-", color="#0072B2", ms=7, capsize=3,
                 label="actual TVD (stim)")
    ax1.set_xlabel("Shannon entropy of ideal output (bits)")
    ax1.set_ylabel("probability of an error")
    ax1.set_title("bound is fixed; TVD falls as the output spreads", fontsize=11)
    ax1.set_ylim(bottom=0)
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(True, alpha=0.15)

    gap = [bound - t for t in tvd]
    ax2.errorbar(H, gap, yerr=tvd_std, fmt="o-", color="#009E73", ms=7, capsize=3)
    ax2.set_xlabel("Shannon entropy of ideal output (bits)")
    ax2.set_ylabel("bound - TVD  (looseness)")
    ax2.set_title("the gap opens up with entropy", fontsize=11)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.15)

    fig.suptitle(f"Bound looseness vs output entropy  (n={N}, depth {DEPTH}, "
                 "Clifford, stim)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)


def main():
    print("Cycle benchmarking the CNOT cycles (stim)...", flush=True)
    cbA = cycle_benchmark(CYCLE_A, N, [1, 2, 4, 8], NOISE, twoq="CX", seed=1)
    cbB = cycle_benchmark(CYCLE_B, N, [1, 2, 4, 8], NOISE, twoq="CX", seed=2)
    ro_fid, ro_std = readout_fidelity(N, NOISE, seed=3)
    efs = {"A": (cbA["e_F"], cbA["e_F_std"]), "B": (cbB["e_F"], cbB["e_F_std"])}
    bound = qcap_bound({"A": DEPTH, "B": DEPTH}, efs, ro_fid, ro_std)["error"]
    print(f"  e_F(A)={cbA['e_F']:.4f}  e_F(B)={cbB['e_F']:.4f}  ro_fid={ro_fid:.4f}", flush=True)
    print(f"  fixed CB bound (depth {DEPTH}) = {bound:.4f}\n", flush=True)

    H, tvd, tvd_std = [], [], []
    print(f"{'k':>3}{'entropy':>9}{'#out':>8}{'TVD(mean)':>11}{'+/-':>8}{'bound-TVD':>11}",
          flush=True)
    for k in KS:
        ts = []
        for i in range(N_INST):
            circ = entropy_circuit(k, SEED + 1000 * k + i)
            ideal = _SB.exact_distribution(circ)
            support = sorted(int(b, 2) for b in ideal)
            ts.append(noisy_tvd_vs_support(circ, NOISE, support, SHOTS, seed=SEED + 7 * i + k))
        Hk = float(k)                       # entropy is exactly k bits
        tmean, tstd = float(np.mean(ts)), float(np.std(ts))
        H.append(Hk)
        tvd.append(tmean)
        tvd_std.append(tstd)
        save_fig(H, bound, tvd, tvd_std)
        save_results(OUT_DATA, entropy=np.array(H), bound=bound,
                     tvd_mean=np.array(tvd), tvd_std=np.array(tvd_std))
        print(f"{k:>3}{Hk:>9.2f}{2**k:>8}{tmean:>11.4f}{tstd:>8.4f}{bound - tmean:>11.4f}",
              flush=True)

    print(f"\nWrote {OUT}, {OUT_DATA}")


if __name__ == "__main__":
    main()
