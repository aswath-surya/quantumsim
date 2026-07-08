"""Circuit-level noise: compare simulators on TIME and FIDELITY.

Applies a toggle-able noise model (2-qubit depolarizing, 1-qubit depolarizing,
idling dephasing, readout error) to a Clifford brickwork and asks each simulator
for the noisy output distribution:

  * stabilizer (stim)     -- NATIVE Pauli noise, exact + scalable.
  * statevector (qiskit)  -- Monte-Carlo TRAJECTORIES (sample which Paulis fire,
                             evolve the pure state, measure).
  * tensor-network (MPS)  -- trajectories, same idea.

Then it reports, vs a scale factor on the noise rates:
  * FIDELITY of each simulator's noisy distribution to the ideal (noiseless) one
    -- they must agree (cross-validation); it decays as noise grows.
  * TIME per 1000 samples -- stim is orders of magnitude cheaper.

Run:  python examples/run_noise_compare.py
"""

from __future__ import annotations

import random
import time
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401
from proxysim import Circuit
from proxysim.noise import (NoiseModel, apply_readout, classical_fidelity,
                            sample_trajectory, to_stim_noisy)
from proxysim.backends import StatevectorBackend, TensorNetworkBackend

N = 6
SEED = 7
BASE = NoiseModel(enabled=True, p1=1e-3, p2=1e-2, p_readout=1e-2, p_idle=1e-3)
SCALES = [0.0, 1.0, 2.0, 4.0, 8.0]
STIM_SHOTS = 20000
TRAJ_SHOTS = 2000

# Okabe-Ito colorblind-safe categorical hues, fixed order (entity, not rank).
COLORS = {"stim": "#0072B2", "statevector": "#E69F00", "MPS": "#009E73"}


def ghz(n):
    """GHZ state (H + CNOT chain): a PEAKED Clifford output -- ideal = {0..0, 1..1}
    each 0.5 -- so depolarizing (X/Y) and readout noise visibly reduce fidelity.
    (A deep brickwork gives a near-uniform output that classical fidelity can't
    resolve; and Z-basis fidelity is blind to pure dephasing regardless.)"""
    c = Circuit(n, name=f"ghz_{n}")
    c.h(0)
    for i in range(n - 1):
        c.cx(i, i + 1)
    return c


circ = ghz(N)
IDEAL = StatevectorBackend().exact_distribution(circ)


def _norm(counts):
    tot = sum(counts.values()) or 1
    return {k: v / tot for k, v in counts.items()}


def stim_noisy(noise):
    c = to_stim_noisy(circ, noise)
    t0 = time.perf_counter()
    arr = c.compile_sampler(seed=SEED).sample(STIM_SHOTS)
    dt = time.perf_counter() - t0
    counts = {}
    for row in arr:
        k = "".join("1" if b else "0" for b in row)
        counts[k] = counts.get(k, 0) + 1
    return _norm(counts), dt / (STIM_SHOTS / 1000)


def traj_noisy(backend, noise, shots):
    rng = random.Random(SEED)
    counts = {}
    t0 = time.perf_counter()
    for _ in range(shots):
        traj = sample_trajectory(circ, noise, rng)
        r = backend.run(traj, shots=1, seed=rng.randint(0, 2**31 - 1))
        bs = apply_readout(next(iter(r.counts)), noise, rng)
        counts[bs] = counts.get(bs, 0) + 1
    dt = time.perf_counter() - t0
    return _norm(counts), dt / (shots / 1000)


def main():
    sv, tn = StatevectorBackend(), TensorNetworkBackend()

    # --- explicit toggle demonstration ---
    off = classical_fidelity(stim_noisy(NoiseModel(enabled=False))[0], IDEAL)
    on = classical_fidelity(stim_noisy(BASE)[0], IDEAL)
    print(f"toggle: noise OFF -> F={off:.4f}   noise ON (base) -> F={on:.4f}\n")

    results = {k: {"F": [], "t": []} for k in COLORS}
    print(f"{'scale':>6}{'F(stim)':>10}{'F(sv)':>10}{'F(mps)':>10}"
          f"{'t/1k stim':>12}{'t/1k sv':>12}{'t/1k mps':>12}  (ms)")
    for s in SCALES:
        noise = BASE.scaled(s)
        d_st, t_st = stim_noisy(noise)
        d_sv, t_sv = traj_noisy(sv, noise, TRAJ_SHOTS)
        d_mp, t_mp = traj_noisy(tn, noise, TRAJ_SHOTS)
        F = {"stim": classical_fidelity(d_st, IDEAL),
             "statevector": classical_fidelity(d_sv, IDEAL),
             "MPS": classical_fidelity(d_mp, IDEAL)}
        T = {"stim": t_st * 1e3, "statevector": t_sv * 1e3, "MPS": t_mp * 1e3}
        for k in COLORS:
            results[k]["F"].append(F[k])
            results[k]["t"].append(T[k])
        print(f"{s:>6.1f}{F['stim']:>10.4f}{F['statevector']:>10.4f}{F['MPS']:>10.4f}"
              f"{T['stim']:>12.3f}{T['statevector']:>12.3f}{T['MPS']:>12.3f}")

    # ---- plot: two panels (never a dual axis) ----
    fig, (axF, axT) = plt.subplots(1, 2, figsize=(12, 4.6))

    for k in COLORS:  # fidelity vs noise scale -- curves should coincide
        axF.plot(SCALES, results[k]["F"], "o-", color=COLORS[k], lw=2, ms=6, label=k)
    axF.set_xlabel("noise scale factor")
    axF.set_ylabel("fidelity  F(noisy, ideal)")
    axF.set_title("performance: fidelity vs noise (simulators agree)", fontsize=11)
    axF.set_ylim(0, 1.02)
    axF.legend(frameon=False, fontsize=9)
    axF.grid(True, alpha=0.15)

    # time per 1000 samples (median over scales), bar, log y
    labels = list(COLORS)
    med = [float(np.median(results[k]["t"])) for k in labels]
    axT.bar(labels, med, color=[COLORS[k] for k in labels], width=0.6)
    for i, v in enumerate(med):
        axT.text(i, v, f" {v:.2f}", ha="center", va="bottom", fontsize=9)
    axT.set_yscale("log")
    axT.set_ylabel("time per 1000 samples (ms)")
    axT.set_title("cost: stim samples noisy Clifford circuits far cheaper", fontsize=11)
    axT.grid(True, axis="y", which="both", alpha=0.15)

    fig.suptitle(f"Circuit-level noise across simulators "
                 f"(n={N} GHZ; 2q-depol/1q-depol/idle-dephase/readout, toggle-able)",
                 fontsize=12)
    fig.tight_layout()
    out = _bootstrap.RESULTS_DIR + "/noise_compare.png"
    fig.savefig(out, dpi=150)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
