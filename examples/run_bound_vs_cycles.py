"""Bound vs actual TVD as a function of the number of cycles, at LARGE output
entropy (and a low-entropy reference).

Companion to run_bound_vs_entropy.py. There the depth was fixed and entropy varied;
here the entropy is fixed and the depth (number of entangling cycles) is swept. The
CB bound depends only on the cycles and their e_F, not on the output entropy, so a
single bound line serves every entropy -- which makes the contrast clean: at
entropy 0 (deterministic output) the measured TVD hugs the bound, while at large
entropy (k=12, uniform over 4096 outcomes) it falls well below, and the gap widens
with depth.

TVD stays computable at 20 qubits via the coset trick (known 2^k ideal support +
leakage mass, from stim samples). Writes results/bound_vs_cycles.png.

Run:  python examples/run_bound_vs_cycles.py
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
CYCLE_A = even_pairs(N)
CYCLE_B = odd_pairs(N)
K_LIST = [0, 12]                     # entropy 0 (reference) and 12 bits (large)
DEPTHS = [1, 2, 4, 8, 16]            # number of cycles
N_INST = 8                           # instances per point, plotted as individual points
NOISE = NoiseModel(enabled=True, p1=2e-4, p2=1e-3, p_readout=2e-3, p_idle=2e-4)
SHOTS = 500_000
SEED = 5
_SB = StabilizerBackend()
COLORS = {0: "#0072B2", 12: "#009E73"}
OUT = _bootstrap.RESULTS_DIR + "/bound_vs_cycles.png"
OUT_LINEAR = _bootstrap.RESULTS_DIR + "/bound_vs_cycles_linear.png"
OUT_DATA = _bootstrap.RESULTS_DIR + "/bound_vs_cycles_data.npz"


def save_data(bounds, tvd):
    """Persist all results (bounds + every instance's TVD) so the figures can be
    re-rendered later WITHOUT re-running the simulation."""
    kw = {"depths": np.array(DEPTHS[:len(bounds)]), "bounds": np.array(bounds)}
    for k in K_LIST:
        if tvd[k]:
            kw[f"tvd_k{k}"] = np.array(tvd[k])   # shape (n_depths, N_INST)
    save_results(OUT_DATA, **kw)


def entropy_circuit(k, depth, seed):
    """Clifford circuit with exactly k bits of output entropy (k Hadamards + CNOT
    brickwork with dimension-preserving single-qubit gates)."""
    return bench_brickwork(N, depth, [CYCLE_A, CYCLE_B], oneq="dim_preserving",
                           twoq="cx", seed=seed, n_hadamards=k)


def _draw(bounds, tvd, logx, out):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(DEPTHS[:len(bounds)], bounds, "-", color="#D55E00", lw=2.5, label="CB bound")
    for k in K_LIST:
        ys = tvd[k]                       # list over depths, each a list of instances
        for j, vals in enumerate(ys):     # individual instance points (the scatter)
            ax.plot([DEPTHS[j]] * len(vals), vals, "o", color=COLORS[k], ms=6,
                    alpha=0.55, label=f"TVD, entropy {k} bits" if j == 0 else None)
        if ys:                            # thin line through the means to guide the eye
            ax.plot(DEPTHS[:len(ys)], [np.mean(v) for v in ys], "-",
                    color=COLORS[k], lw=1.0, alpha=0.5)
    if logx:
        ax.set_xscale("log", base=2)
    else:
        ax.set_xticks(DEPTHS)
    ax.set_xlabel("number of cycles (depth)")
    ax.set_ylabel("probability of an error (TVD)")
    ax.set_title(f"Bound vs TVD vs depth  (n={N}, Clifford, stim)\n"
                 "one bound; TVD hugs it at entropy 0, falls away at large entropy",
                 fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, which="both", alpha=0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def save_fig(bounds, tvd):
    _draw(bounds, tvd, logx=True, out=OUT)
    _draw(bounds, tvd, logx=False, out=OUT_LINEAR)


def main():
    print("Cycle benchmarking the CNOT cycles (stim)...", flush=True)
    cbA = cycle_benchmark(CYCLE_A, N, [1, 2, 4, 8], NOISE, twoq="CX", seed=1)
    cbB = cycle_benchmark(CYCLE_B, N, [1, 2, 4, 8], NOISE, twoq="CX", seed=2)
    ro_fid, ro_std = readout_fidelity(N, NOISE, seed=3)
    efs = {"A": (cbA["e_F"], cbA["e_F_std"]), "B": (cbB["e_F"], cbB["e_F_std"])}
    print(f"  e_F(A)={cbA['e_F']:.4f}  e_F(B)={cbB['e_F']:.4f}  ro_fid={ro_fid:.4f}\n", flush=True)

    bounds, tvd = [], {k: [] for k in K_LIST}
    print(f"{'depth':>6}{'bound':>9}" + "".join(f"{'TVD k=' + str(k):>12}" for k in K_LIST),
          flush=True)
    for d in DEPTHS:
        bounds.append(qcap_bound({"A": d, "B": d}, efs, ro_fid, ro_std)["error"])
        row = ""
        for k in K_LIST:
            ts = []
            for i in range(N_INST):
                circ = entropy_circuit(k, d, SEED + 1000 * d + 13 * k + i)
                ideal = _SB.exact_distribution(circ)
                support = sorted(int(b, 2) for b in ideal)
                ts.append(noisy_tvd_vs_support(circ, NOISE, support, SHOTS,
                                               seed=SEED + 7 * i + d + k))
            tvd[k].append(ts)
            row += f"{np.mean(ts):>12.4f}"
        save_fig(bounds, tvd)
        save_data(bounds, tvd)            # persist after every depth (never regenerate)
        print(f"{d:>6}{bounds[-1]:>9.4f}{row}", flush=True)

    print(f"\nWrote {OUT}, {OUT_LINEAR}, {OUT_DATA}")


if __name__ == "__main__":
    main()
