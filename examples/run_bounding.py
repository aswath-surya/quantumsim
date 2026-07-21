"""Bound circuit error from cycle benchmarking -- a hardware-free replica of the
qcal `circuit_bounding` notebook.

Pipeline (all synthetic, driven by a NoiseModel):
  1. Cycle-benchmark each distinct entangling cycle in the ansatz -> e_F per cycle.
  2. Measure the readout (SPAM) fidelity.
  3. For circuits of growing depth, form the QCAP bound
        error <= 1 - ro_fid * prod_c (1 - e_F_c)^(times c is used),
     and compare it to the ACTUAL total-variation distance of the noisy circuit
     from its ideal distribution.

Two ansaetze mirror the notebook: "random" (random single-qubit Cliffords) and
"structured" (single-qubit gates drawn from {I, H, X}). Writes results/bounding.png.

Run:  python examples/run_bounding.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401
from proxysim import (NoiseModel, bench_brickwork, even_pairs, odd_pairs, save_results,
                      simulate, total_variation_distance)
from proxysim.backends.stabilizer import StabilizerBackend
from proxysim.benchmarking import cycle_benchmark, qcap_bound, readout_fidelity

N = 4
CYCLE_A = even_pairs(N)        # (0,1) (2,3)   (marker 1 in the notebook)
CYCLE_B = odd_pairs(N)         # (1,2)         (marker 2)
DEPTHS = [1, 2, 4, 8, 16, 32]
N_INSTANCES = 8                # random circuit instances per depth (TVD scatter)
TVD_SHOTS = 4000
SEED = 7
NOISE = NoiseModel(enabled=True, p1=1e-3, p2=1e-2, p_readout=1e-2, p_idle=1e-3)
OUT_DATA = _bootstrap.RESULTS_DIR + "/bounding_data.npz"
_SB = StabilizerBackend()


def test_circuit(depth, mode, seed):
    """depth reps of [1q layer, cycle A, 1q layer, cycle B] + a final 1q layer
    (non-mirror). Cycle A and cycle B each appear `depth` times."""
    oneq = "clifford" if mode == "random" else "structured"
    return bench_brickwork(N, depth, [CYCLE_A, CYCLE_B], oneq=oneq, twoq="cz",
                           seed=seed, final_oneq=True)


def noisy_tvd(circ):
    """Full-distribution TVD (small n) between the ideal and the stim-sampled noisy
    distribution."""
    ideal = _SB.exact_distribution(circ)
    noisy = simulate(circ, "stabilizer", "distribution", noise=NOISE,
                     shots=TVD_SHOTS, seed=SEED)
    return total_variation_distance(ideal, noisy)


def run(mode):
    print(f"\n=== {mode} ansatz ===")
    cbA = cycle_benchmark(CYCLE_A, N, [1, 2, 4, 8, 16, 24], NOISE, mode=mode, seed=1)
    cbB = cycle_benchmark(CYCLE_B, N, [1, 2, 4, 8, 16, 24], NOISE, mode=mode, seed=2)
    ro_fid, ro_std = readout_fidelity(N, NOISE, seed=3)
    print(f"  cycle A (even pairs): e_F = {cbA['e_F']:.4f}")
    print(f"  cycle B (middle):     e_F = {cbB['e_F']:.4f}")
    print(f"  readout fidelity:     {ro_fid:.4f} ({ro_std:.4f})")

    efs = {"A": (cbA["e_F"], cbA["e_F_std"]), "B": (cbB["e_F"], cbB["e_F_std"])}
    bounds, bstd, tvd_points = [], [], []
    print(f"  {'depth':>6}{'bound':>10}{'mean TVD':>10}")
    for d in DEPTHS:
        b = qcap_bound({"A": d, "B": d}, efs, ro_fid, ro_std)
        bounds.append(b["error"])
        bstd.append(b["std"])
        tvds = [noisy_tvd(test_circuit(d, mode, SEED + 100 * i)) for i in range(N_INSTANCES)]
        tvd_points.append(tvds)
        print(f"  {d:>6}{b['error']:>10.4f}{np.mean(tvds):>10.4f}")
    return bounds, bstd, tvd_points


def main():
    data = {m: run(m) for m in ("random", "structured")}

    kw = {"depths": np.array(DEPTHS)}
    for m in data:
        kw[f"{m}_bound"], kw[f"{m}_bound_std"], kw[f"{m}_tvd"] = \
            np.array(data[m][0]), np.array(data[m][1]), np.array(data[m][2])
    save_results(OUT_DATA, **kw)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, mode in zip(axes, ("random", "structured")):
        bounds, bstd, tvd_points = data[mode]
        b, bs = np.array(bounds), np.array(bstd)
        for d, tvds in zip(DEPTHS, tvd_points):
            ax.plot([d] * len(tvds), tvds, "o", color="#0072B2", ms=5, alpha=0.7,
                    label="measured TVD" if d == DEPTHS[0] else None)
        ax.plot(DEPTHS, b, "-", color="#D55E00", lw=2, label="QCAP bound")
        ax.fill_between(DEPTHS, b - 2.96 * bs, b + 2.96 * bs, color="#D55E00", alpha=0.2)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("circuit depth")
        ax.set_title(f"{mode} ansatz", fontsize=11)
        ax.grid(True, which="both", alpha=0.15)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("probability of an error (TVD)")
    fig.suptitle("Bounding circuit error from cycle benchmarking "
                 f"(n={N}, synthetic CB + noise model)", fontsize=12)
    fig.tight_layout()
    out = _bootstrap.RESULTS_DIR + "/bounding.png"
    fig.savefig(out, dpi=150)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
