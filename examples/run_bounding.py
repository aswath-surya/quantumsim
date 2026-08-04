"""Bound circuit error from cycle benchmarking, with COHERENT noise -- a hardware-free
replica of the qcal `circuit_bounding` notebook, extended past the Pauli-only regime.

Pipeline (all synthetic, driven by a NoiseModel):
  1. Cycle-benchmark each distinct entangling cycle in the ansatz -> e_F per cycle.
  2. Measure the readout (SPAM) fidelity.
  3. For circuits of growing depth, form the QCAP bound
        error <= 1 - ro_fid * prod_c (1 - e_F_c)^(times c is used),
     and compare it to the ACTUAL total-variation distance of the noisy circuit
     from its ideal distribution.

Two ansaetze mirror the notebook: "random" (random single-qubit Cliffords) and
"structured" (single-qubit gates drawn from {I, H, X}).

SINGLE-QUBIT GATE SET (`ONEQ_SET`) AND WHY THE CLIFFORD SCATTER IS BANDED
-------------------------------------------------------------------------
Under RC the effective error is a PAULI channel -- that is what RC is for, and what
`cycle_benchmark` measures. Every Pauli channel is unital, so the uniform distribution
is one of its fixed points. A Clifford circuit's ideal output is uniform on an affine
subspace of size 2^k, so writing D = TVD(ideal, uniform) the measured distance factors
as roughly

    TVD  ~=  eps * D,        D = 1 - 2^k / 2^n,   eps set by the gate count.

D therefore takes only n+1 possible values and eps is near-constant across instances
at fixed depth (identical gate counts, homogeneous noise), so the scatter collapses
onto a few horizontal bands -- including a hard band at exactly zero for the
full-support (k = n) instances, which is typically 30-60% of them. Measured at n=4,
depth 8: 13 of 30 Clifford instances sit at TVD = 0.000 and the rest start at 0.092.
Raising n does not help (random Clifford circuits are near-full-support at every n),
and neither does randomising eps, since eps enters multiplicatively and 0 * eps = 0.

    ONEQ_SET = "clifford"   -> both ansaetze Clifford: banded, exact, fast (default).
    ONEQ_SET = "t"          -> appends T to the RANDOM ansatz only. Its ideal state is
                               no longer a stabilizer state, D becomes continuous, and
                               the bands dissolve. Structured keeps {I, H, X} and stays
                               banded, so the two panels become a controlled comparison
                               -- same cycles, same noise, same bound, the only
                               difference being stabilizer vs non-stabilizer ideals.

Measured at n=2, depth 64, 30 instances: the Clifford scatter has a 0.214-wide void
between 0.050 and 0.264, while the +T scatter runs 0.047 -> 0.331 with a largest gap
of 0.041 (a 5x reduction, and consistent with 30 samples over that range).

The T ansatz keeps RC and keeps the bound valid -- `cycle_benchmark` measures the CZ
cycles and `qcap_bound` counts cycle applications; neither requires the single-qubit
dressing of the TEST circuit to be Clifford. What it costs is stim: a non-Clifford
circuit falls through to Monte-Carlo trajectories on the statevector backend, so
`N_TRAJ` becomes the runtime knob and imposes its own noise floor on those points.

WHY TWO TVD SCATTERS
--------------------
The noise model now carries a coherent term (`theta_zz`, a residual always-on ZZ) on
top of the Pauli channels. That breaks the assumption the bound rests on, so "the
measured TVD" is no longer one number -- it depends on whether the circuit was
randomly compiled, and the two answers are qualitatively different:

  * randomly compiled -- RC tailors the coherent error into a Pauli channel, which is
    the channel `cycle_benchmark` measures (it calls `.twirled()` internally, since it
    runs in stim). The bound is valid for this series.
  * NOT compiled -- the coherent error survives and accumulates in AMPLITUDE rather
    than in probability. The bound does not cover this series, and it is the one that
    lifts the "TVD == 0" band: uniform is a fixed point of every Pauli channel here,
    so full-support instances sit at zero under RC but not without it.

Plotting the bound against only the un-compiled series would be comparing a circuit to
a bound for a different circuit, which is why both are shown.

How each series is computed:
  * RC'd: `simulate(..., noise=NOISE.twirled())` on the exact stim path. `.twirled()`
    is the closed-form Pauli channel RC produces (cp(theta) -> p_IZ = p_ZI = p_ZZ =
    sin^2(theta/2)/4), so this is exact and costs nothing extra. It is not an
    assumption: `examples/run_coherent_rc.py` verifies it against actually running
    `rc.pauli_twirl` over 24k randomizations, agreeing to TVD 0.0011 with a no-RC
    control 47x further away.
  * un-compiled: Monte-Carlo trajectories on the statevector backend, since a coherent
    model is non-Clifford and stim cannot represent it.

Set THETA_ZZ = 0.0 to recover the original Pauli-only run exactly (the two series then
coincide, both taking the stim path).

Writes results/bounding_{N}qubits[_{ONEQ_SET}].png (the suffix is omitted for the
default Clifford ansatz, so the two gate sets never clobber each other's figures).

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
from proxysim import (GATE_SETS, NoiseModel, bench_brickwork, even_pairs, odd_pairs,
                      save_results, simulate, total_variation_distance)
from proxysim.backends import StatevectorBackend
from proxysim.backends.stabilizer import StabilizerBackend
from proxysim.benchmarking import cycle_benchmark, qcap_bound, readout_fidelity

N = 2
# even/odd brickwork layers. odd_pairs(2) is EMPTY -- at n=2 there is no (1,2) pair --
# so drop any empty layer: benchmarking it returns a spurious non-zero e_F (it still
# has 1q + idle noise) and qcap_bound would then charge the bound for a cycle the
# circuit never applies.
CYCLES = {k: v for k, v in (("A", even_pairs(N)), ("B", odd_pairs(N))) if v}
CYCLE_LIST = list(CYCLES.values())
DEPTHS = [1, 2, 4, 8, 16, 32, 64, 128]  # depth series to test, for both ansaetze
N_INSTANCES = 80                # random circuit instances per depth (TVD scatter)
TVD_SHOTS = 2000              # sets the TVD noise floor (~0.003 here); sampling is
                                 # setup-bound, so more shots are essentially free
THETA_ZZ = 0.00                  # coherent residual-ZZ angle, rad. 0.0 -> Pauli only.
#ONEQ_SET = "clifford"            # "clifford" -> stabilizer ansatz, exact stim path.
ONEQ_SET = "t"                                # "t"        -> append T to the 1q set. Non-Clifford,
                                 #               so every point goes through N_TRAJ
                                 #               trajectories, but the TVD bands
                                 #               dissolve. See the module docstring.
N_TRAJ = 1000                    # trajectories per point on any series that cannot go
                                 # through stim -- the un-compiled series always, and
                                 # BOTH series once ONEQ_SET = "t". Sets a noise floor
                                 # on those points: measured per-instance std is 0.011
                                 # at 200 trajectories and falls as 1/sqrt(N_TRAJ), so
                                 # 1500 puts it near the 0.003 shot floor. It is also
                                 # the whole runtime knob.
SEED = 42
NOISE = NoiseModel(enabled=True, p1=1e-3, p2=1e-2, p_readout=1e-2, p_idle=1e-3,
                   theta_zz=THETA_ZZ)
TWIRLED = NOISE.twirled()        # what RC produces, and what cycle_benchmark measures

# The 1q gate set per ansatz mode. T goes into the RANDOM ansatz only, so the two
# panels become a controlled comparison: same cycles, same noise, same bound, and the
# only difference is whether the ideal output is a stabilizer state. Structured stays
# on {I, H, X} and therefore stays banded (and stays on the fast stim path).
#
# cycle_benchmark is untouched either way -- it keeps its own Clifford twirl layers,
# because the Pauli twirl the CB protocol relies on is a twirl over the Clifford group.
_ONEQ = {"random": list(GATE_SETS["clifford"]),
         "structured": list(GATE_SETS["structured"])}
if ONEQ_SET == "t":
    _ONEQ["random"].append("t")
elif ONEQ_SET != "clifford":
    raise ValueError(f"ONEQ_SET must be 'clifford' or 't', got {ONEQ_SET!r}")

CLIFFORD_ANSATZ = ONEQ_SET == "clifford"
_TAG = "" if CLIFFORD_ANSATZ else f"_{ONEQ_SET}"
OUT_DATA = _bootstrap.RESULTS_DIR + f"/bounding_data{_TAG}.npz"
_SB, _SV = StabilizerBackend(), StatevectorBackend()


def test_circuit(depth, mode, seed):
    """depth reps of [1q layer, cycle, ...] over the non-empty cycles + a final 1q
    layer (non-mirror). Each cycle appears `depth` times."""
    return bench_brickwork(N, depth, CYCLE_LIST, oneq=_ONEQ[mode], twoq="cz",
                           seed=seed, final_oneq=True)


def noisy_tvds(circ, seed):
    """(TVD under randomized compiling, TVD without it) for one circuit instance.

    `seed` must vary per instance, otherwise every point shares one shot-noise
    realisation and the scatter understates the true spread.

    The path is chosen per CIRCUIT, not per ansatz: only the random ansatz carries T,
    and even there a shallow instance can draw no T at all (6 gate slots at depth 1,
    so ~49% of them), in which case it is Clifford and takes the exact stim path.
    """
    clifford = circ.is_clifford
    ideal = (_SB if clifford else _SV).exact_distribution(circ)
    rc = simulate(circ, "stabilizer" if clifford else "statevector", "distribution",
                  noise=TWIRLED, shots=TVD_SHOTS, n_traj=N_TRAJ, seed=seed)
    #raw = simulate(circ, "statevector", "distribution", noise=NOISE,
    #               shots=TVD_SHOTS, n_traj=N_TRAJ, seed=seed)
    #return (total_variation_distance(ideal, rc)), total_variation_distance(ideal, raw))
    return (total_variation_distance(ideal, rc)) 


def run(mode):
    print(f"\n=== {mode} ansatz ===")
    # cycle_benchmark runs in stim and so measures the TWIRLED channel -- the bound it
    # feeds is therefore a bound on the randomly-compiled circuit.
    cbs = {k: cycle_benchmark(pairs, N, [1, 2, 4, 8, 16, 24], NOISE, mode=mode,
                              seed=1 + j)
           for j, (k, pairs) in enumerate(CYCLES.items())}
    ro_fid, ro_std = readout_fidelity(N, NOISE, seed=3)
    for k, cb in cbs.items():
        print(f"  cycle {k} {str(CYCLES[k]):<16} e_F = {cb['e_F']:.4f}")
    print(f"  readout fidelity:     {ro_fid:.4f} ({ro_std:.4f})")

    efs = {k: (cb["e_F"], cb["e_F_std"]) for k, cb in cbs.items()}
    bounds, bstd, rc_points, raw_points = [], [], [], []
    #print(f"  {'depth':>6}{'bound':>10}{'TVD (RC)':>11}{'TVD (no RC)':>13}")
    print(f"  {'depth':>6}{'bound':>10}{'TVD (RC)':>11}")

    for d in DEPTHS:
        b = qcap_bound({k: d for k in CYCLES}, efs, ro_fid, ro_std)
        bounds.append(b["error"])
        bstd.append(b["std"])
        #rc_d, raw_d = [], []
        rc_d = []
        for i in range(N_INSTANCES):
            s = SEED + 100 * i     # stride leaves room for an independent shot seed
            #v_rc, v_raw = noisy_tvds(test_circuit(d, mode, s), s + 1)
            v_rc = noisy_tvds(test_circuit(d, mode, s), s + 1)
            rc_d.append(v_rc); #raw_d.append(v_raw)
        rc_points.append(rc_d); #raw_points.append(raw_d)
        #over = int(np.sum(np.array(raw_d) > b["error"]))
        #flag = f"   ({over}/{N_INSTANCES} over the bound)" if over else ""
        #print(f"  {d:>6}{b['error']:>10.4f}{np.mean(rc_d):>11.4f}"
        #      f"{np.mean(raw_d):>13.4f}{flag}", flush=True)
        print(f"  {d:>6}{b['error']:>10.4f}{np.mean(rc_d):>11.4f}", flush=True)
    return bounds, bstd, rc_points, raw_points


def main():
    for m in ("random", "structured"):
        cliff = "t" not in _ONEQ[m]
        print(f"1q gate set ({m:>10}): {_ONEQ[m]}  ->  "
              + ("Clifford, exact stim path" if cliff
                 else f"NON-Clifford, {N_TRAJ} trajectories per point"))
    print(f"coherent residual ZZ: theta_zz = {THETA_ZZ} rad  ->  twirled "
          f"p_IZ = p_ZI = p_ZZ = {TWIRLED.p_zz:.5f} per 2q gate "
          f"(vs p2 = {NOISE.p2} depolarizing)")
    data = {m: run(m) for m in ("random", "structured")}

    kw = {"depths": np.array(DEPTHS), "theta_zz": THETA_ZZ, "n_traj": N_TRAJ,
          "tvd_shots": TVD_SHOTS, "oneq_set": ONEQ_SET}
    for m in data:
        kw[f"{m}_bound"], kw[f"{m}_bound_std"] = np.array(data[m][0]), np.array(data[m][1])
        kw[f"{m}_tvd"] = np.array(data[m][2])        # RC'd -- the bound applies to this
        #kw[f"{m}_tvd_nonrc"] = np.array(data[m][3])  # un-compiled
    save_results(OUT_DATA, **kw)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, mode in zip(axes, ("random", "structured")):
        bounds, bstd, rc_points, raw_points = data[mode]
        b, bs = np.array(bounds), np.array(bstd)
        #for d, rc_d, raw_d in zip(DEPTHS, rc_points, raw_points):
        for d, rc_d in zip(DEPTHS, rc_points):
            #ax.plot([d] * len(raw_d), raw_d, "s", color="#009E73", ms=4, alpha=0.45,
            #        label="measured TVD, NOT compiled" if d == DEPTHS[0] else None)
            ax.plot([d] * len(rc_d), rc_d, "o", color="#0072B2", ms=5, alpha=0.7,
                    label="measured TVD, randomly compiled" if d == DEPTHS[0] else None)
        ax.plot(DEPTHS, b, "-", color="#D55E00", lw=2, label="QCAP bound (on the RC'd circuit)")
        ax.fill_between(DEPTHS, b - 2.96 * bs, b + 2.96 * bs, color="#D55E00", alpha=0.2)
        ax.set_xlabel("circuit depth")
        ax.set_title(f"{mode} ansatz", fontsize=11)
        ax.grid(True, which="both", alpha=0.15)
        ax.legend(frameon=False, fontsize=8.5)
    axes[0].set_ylabel("probability of an error (TVD)")
    ansatz = "Clifford 1q" if CLIFFORD_ANSATZ else "Clifford+T 1q on random only"
    fig.suptitle("Bounding circuit error from cycle benchmarking "
                 f"(n={N}, {ansatz}, synthetic CB + noise model, coherent "
                 rf"$\theta_{{zz}}={THETA_ZZ}$)", fontsize=12)
    fig.tight_layout()
    out = _bootstrap.RESULTS_DIR + f"/bounding_{N}qubits{_TAG}.png"
    fig.savefig(out, dpi=150)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
