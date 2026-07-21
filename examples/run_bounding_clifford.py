"""Bound vs actual error for ~20-qubit Clifford circuits (random + structured), stim.

Low-error-probability regime. Uses mirror / Loschmidt-echo circuits (U.U^-1, ideal
= |0...0>) so the error is 1 - P(0...0), resolvable from stim samples at 20 qubits
where a full distribution is hopeless. Cycle-benchmark the two entangling cycles
once, form the QCAP bound, and compare to the stim-measured error across depth for
both a random and a structured single-qubit gate set.

Circuits and metrics come from the package (proxysim.bench_brickwork / .mirror,
proxysim.noisy_survival) -- this script is just the experiment loop + plot + save.

Run:  python examples/run_bounding_clifford.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401
from proxysim import (NoiseModel, bench_brickwork, even_pairs, noisy_survival,
                      odd_pairs, save_results)
from proxysim.benchmarking import cycle_benchmark, qcap_bound, readout_fidelity

N = 20
CYCLE_A = even_pairs(N)       # even pairs (10 CZs)
CYCLE_B = odd_pairs(N)        # odd pairs  (9 CZs)
NOISE = NoiseModel(enabled=True, p1=1e-4, p2=5e-4, p_readout=1e-3, p_idle=1e-4)
DEPTHS = [1, 2, 4, 8, 16]     # cycles in U; mirror doubles the count
N_INSTANCES = 6
STIM_SHOTS = 100_000
SEED = 11
OUT = _bootstrap.RESULTS_DIR + "/bounding_clifford.png"
OUT_DATA = _bootstrap.RESULTS_DIR + "/bounding_clifford_data.npz"


def build(mode, n_cycles, seed):
    """Clifford mirror brickwork; mode 'random' -> random Cliffords, 'structured' -> {I,H,X}."""
    oneq = "clifford" if mode == "random" else "structured"
    return bench_brickwork(N, n_cycles, [CYCLE_A, CYCLE_B], oneq=oneq, twoq="cz",
                           seed=seed).mirror()


def error(circ, ro_fid, seed):
    """1 - P(0...0)*ro_fid, the mirror-circuit error (stim-native noisy survival)."""
    return 1 - noisy_survival(circ, NOISE, "stabilizer", n_traj=STIM_SHOTS, seed=seed) * ro_fid


def save_fig(state):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    xs = [2 * d for d in DEPTHS]
    for ax, mode in zip(axes, ("random", "structured")):
        bnd = state[mode]["bound"]
        if bnd:
            ax.plot(xs[:len(bnd)], bnd, "-", color="#D55E00", lw=2, label="CB bound")
        for j, pts in enumerate(state[mode]["err"]):
            ax.plot([xs[j]] * len(pts), pts, "o", color="#0072B2", ms=6, alpha=0.7,
                    label="stim (actual)" if j == 0 else None)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("mirror depth (2 x cycles)")
        ax.set_title(f"{mode} ansatz", fontsize=11)
        ax.grid(True, which="both", alpha=0.15)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("probability of an error  (1 - P(0...0))")
    fig.suptitle(f"Bound vs actual error, n={N} Clifford mirror circuits "
                 "(low error, stim)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)


def save_data(state):
    kw = {"depths": np.array(DEPTHS)}
    for mode in ("random", "structured"):
        if state[mode]["bound"]:
            kw[f"{mode}_bound"] = np.array(state[mode]["bound"])
            kw[f"{mode}_err"] = np.array(state[mode]["err"])   # (n_depths, N_INSTANCES)
    save_results(OUT_DATA, **kw)


def main():
    print("Cycle benchmarking (stim)...", flush=True)
    cbA = cycle_benchmark(CYCLE_A, N, [1, 2, 4, 8, 16], NOISE, mode="random", seed=1)
    cbB = cycle_benchmark(CYCLE_B, N, [1, 2, 4, 8, 16], NOISE, mode="random", seed=2)
    ro_fid, ro_std = readout_fidelity(N, NOISE, seed=3)
    efs = {"A": (cbA["e_F"], cbA["e_F_std"]), "B": (cbB["e_F"], cbB["e_F_std"])}
    print(f"  e_F(A even, 10 CZ)={cbA['e_F']:.4f}  e_F(B odd, 9 CZ)={cbB['e_F']:.4f}  "
          f"ro_fid={ro_fid:.4f}", flush=True)

    state = {m: {"bound": [], "err": []} for m in ("random", "structured")}
    for mode in ("random", "structured"):
        print(f"\n{mode} ansatz:", flush=True)
        for d in DEPTHS:
            b = qcap_bound({"A": 2 * d, "B": 2 * d}, efs, ro_fid, ro_std)["error"]
            errs = [error(build(mode, d, SEED + 100 * i + d), ro_fid, SEED + i)
                    for i in range(N_INSTANCES)]
            state[mode]["bound"].append(b)
            state[mode]["err"].append(errs)
            save_fig(state)
            save_data(state)
            print(f"  depth {2*d:>2}: bound={b:.4f}  mean stim error={np.mean(errs):.4f}",
                  flush=True)

    print(f"\nWrote {OUT}, {OUT_DATA}")


if __name__ == "__main__":
    main()
