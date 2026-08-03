"""Randomized compiling turns coherent noise into exactly the predicted Pauli channel.

This is the end-to-end validation of two features that only became meaningful together:
:class:`~proxysim.noise.NoiseModel`'s coherent terms (``theta_1q`` / ``theta_zz``) and
:mod:`proxysim.rc`'s Pauli twirl. Against a purely stochastic model the twirl is a
statistical no-op, so there was nothing to check; against a coherent model it has a
sharp, closed-form prediction:

    cp(theta)  --Pauli twirl-->  p_IZ = p_ZI = p_ZZ = sin^2(theta/2) / 4

which is precisely what :meth:`~proxysim.noise.NoiseModel.twirled` returns.

Panel 1 (the assertion). Two ways of computing the same distribution:

  * Path A -- run R independent Pauli-twirled randomizations of the circuit under the
    COHERENT model and average their distributions. This is randomized compiling as
    actually performed.
  * Path B -- run the circuit ONCE in stim under ``noise.twirled()``, the analytic
    Pauli channel.

PASS if TVD(A, B) is inside the combined Monte-Carlo error of the two paths, which is
estimated from the data itself by splitting each path into two independent halves (no
hand-tuned tolerance). A negative control -- the same circuit with NO randomized
compiling -- is printed alongside, and must be far outside that error, otherwise the
test has no power to detect a failure.

Why this run has ``p1 = p_idle = p_readout = 0`` and only the 2q terms active:
:mod:`proxysim.rc` emits the twirl UNCOMPILED (its docstring says so -- the random
Paulis appear as explicit gates rather than being absorbed into the neighbouring 1q
layer, as True-Q would do). Extra 1q gates would collect their own ``p1`` noise and
shift the idle sets, biasing A against B by an amount that has nothing to do with the
twirl identity. Restricting to the 2q coherent term makes the comparison exact. That is
a real limitation of testing RC against an analytic twirl, not a bug in either.

Panel 2 (why it matters for the QCAP bound). Sweeping ``theta_zz``: cycle benchmarking
runs in stim and therefore reports the e_F of the TWIRLED channel, so the QCAP bound
built from it describes a randomly-compiled circuit. The RC'd TVD stays under it at
every angle. The un-compiled TVD does not, and the reason is a difference in POWER, not
a constant factor: the twirled rate is sin^2(theta/2) ~ theta^2/4, so the bound vanishes
quadratically as theta -> 0, while an un-compiled coherent error adds amplitudes and its
TVD vanishes only LINEARLY. Small angles are therefore where the bound breaks -- exactly
the regime a real device sits in. The sweep is chosen to straddle the crossing.

Run:  python examples/run_coherent_rc.py
"""

from __future__ import annotations

import random
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

import _bootstrap  # noqa: F401
from proxysim import (NoiseModel, bench_brickwork, even_pairs, odd_pairs, pauli_twirl,
                      save_results, simulate, total_variation_distance)
from proxysim.backends.stabilizer import StabilizerBackend
from proxysim.benchmarking import cycle_benchmark, qcap_bound, readout_fidelity

N = 4
DEPTH = 6                      # brickwork reps, for the panel-1 identity check
THETA = 0.40                   # residual ZZ angle, rad (twirls to p_zz = 0.0099 / gate)
N_RAND = 24_000                # RC randomizations (panel 1); sets path-A MC error
STIM_SHOTS = 4_000_000         # stim shots (panel 1); sets path-B shot error

SWEEP_THETAS = [0.02, 0.04, 0.07, 0.12, 0.2, 0.35, 0.6]
SWEEP_DEPTH = 8
SWEEP_RAND = 3_000
CB_DEPTHS = [1, 2, 4, 8, 16, 24]
CB_DECAYS = 12                 # random decay curves per CB fit (default 30 is slower)

SEED = 20250730
OUT = _bootstrap.RESULTS_DIR + "/coherent_rc.png"
OUT_DATA = _bootstrap.RESULTS_DIR + "/coherent_rc_data.npz"

CYCLES = {k: v for k, v in (("A", even_pairs(N)), ("B", odd_pairs(N))) if v}
CYCLE_LIST = list(CYCLES.values())
_SB = StabilizerBackend()

# Only the 2q coherent term. See the module docstring for why the 1q/idle/readout
# channels are switched off rather than merely made small.
COHERENT = NoiseModel(enabled=True, p1=0.0, p2=0.0, p_readout=0.0, p_idle=0.0,
                      theta_zz=THETA)


def make_circuit(depth, seed):
    return bench_brickwork(N, depth, CYCLE_LIST, oneq="clifford", twoq="cz",
                           seed=seed, final_oneq=True)


def as_vec(dist, n=N):
    """dict bitstring -> prob, as a 2^n vector. Every path in the package indexes
    qubit j into bit j of the key, so this convention is shared."""
    v = np.zeros(2 ** n)
    for k, p in dist.items():
        v[sum(int(b) << j for j, b in enumerate(k))] += p
    return v


def tvd(u, v):
    return 0.5 * float(np.abs(u - v).sum())


def rc_distribution(circ, noise, n_rand, seed):
    """Randomized compiling, done for real: average the distributions of ``n_rand``
    independent Pauli-twirled randomizations under the COHERENT model.

    Returned as ``(mean, half0, half1)`` -- the two halves are independent estimates of
    the same quantity, which is how the script calibrates its own tolerance. With the
    stochastic channels off, each randomization is a single deterministic pure state, so
    ``n_traj=1`` is exact and the only error here is the finite number of twirls.
    """
    rng = random.Random(seed)
    halves = [np.zeros(2 ** N), np.zeros(2 ** N)]
    counts = [0, 0]
    for i in range(n_rand):
        tw = pauli_twirl(circ, rng)
        d = simulate(tw, "statevector", "distribution", noise=noise,
                     shots=0, n_traj=1, seed=seed + i)
        halves[i % 2] += as_vec(d)
        counts[i % 2] += 1
    h0, h1 = halves[0] / counts[0], halves[1] / counts[1]
    return 0.5 * (h0 + h1), h0, h1


# ---------------------------------------------------------------------------
# Panel 1: RC of a coherent model == the analytic twirled Pauli model
# ---------------------------------------------------------------------------
def check_twirl_identity():
    circ = make_circuit(DEPTH, SEED)
    n_2q = sum(1 for g in circ.gates if len(g.qubits) == 2)
    tw_noise = COHERENT.twirled()
    print("=== panel 1: RC(coherent) vs analytic twirl ===")
    print(f"  circuit: n={N}, depth={DEPTH}, {n_2q} two-qubit gates, Clifford="
          f"{circ.is_clifford}")
    print(f"  coherent: theta_zz = {THETA} rad")
    print(f"  analytic twirl: p_IZ = p_ZI = p_ZZ = sin^2(theta/2)/4 = "
          f"{tw_noise.p_zz:.6f}  ({3 * tw_noise.p_zz:.4%} error per 2q gate)")

    # Path A -- randomized compiling under the coherent model.
    a, a0, a1 = rc_distribution(circ, COHERENT, N_RAND, SEED + 1)
    # Path B -- one stim run under the analytic Pauli channel, twice for the error bar.
    b0 = as_vec(simulate(circ, "stabilizer", "distribution", noise=tw_noise,
                         shots=STIM_SHOTS // 2, seed=SEED + 2))
    b1 = as_vec(simulate(circ, "stabilizer", "distribution", noise=tw_noise,
                         shots=STIM_SHOTS // 2, seed=SEED + 3))
    b = 0.5 * (b0 + b1)
    # Negative control -- the same coherent model with NO randomized compiling.
    ctrl = as_vec(simulate(circ, "statevector", "distribution", noise=COHERENT,
                           shots=0, n_traj=1, seed=SEED + 4))

    # Self-calibrated tolerance. Two independent halves differ by ~2x the error of the
    # full-sample mean, so err ~ TVD(half0, half1)/2; 3 sigma on the sum of both paths.
    err_a, err_b = tvd(a0, a1) / 2.0, tvd(b0, b1) / 2.0
    tol = 3.0 * (err_a + err_b)
    d_ab, d_ctrl = tvd(a, b), tvd(ctrl, b)

    print(f"\n  TVD(RC-averaged coherent, analytic twirl) = {d_ab:.5f}")
    print(f"  MC error: path A (RC, {N_RAND} randomizations) = {err_a:.5f}, "
          f"path B (stim, {STIM_SHOTS} shots) = {err_b:.5f}")
    print(f"  tolerance (3 sigma)                       = {tol:.5f}")
    print(f"  negative control, NO RC                   = {d_ctrl:.5f} "
          f"({d_ctrl / max(tol, 1e-12):.0f}x the tolerance)")

    ok = d_ab < tol
    powered = d_ctrl > 5 * tol
    if not powered:
        print("  WARNING: the negative control is inside 5x the tolerance -- this run "
              "cannot distinguish RC from no-RC, so a PASS would be meaningless.")
    print(f"\n  {'PASS' if ok and powered else 'FAIL'}: randomized compiling reproduces "
          f"noise.twirled() to within sampling error.\n")
    return ok and powered, dict(rc=a, twirl=b, norc=ctrl, tvd_ab=d_ab, tvd_ctrl=d_ctrl,
                                tol=tol)


# ---------------------------------------------------------------------------
# Panel 2: the QCAP bound holds under RC, and need not hold without it
# ---------------------------------------------------------------------------
def sweep_theta():
    print("=== panel 2: QCAP bound vs measured TVD, with and without RC ===")
    circ = make_circuit(SWEEP_DEPTH, SEED + 77)
    ideal = as_vec(_SB.exact_distribution(circ))
    print(f"  circuit: n={N}, depth={SWEEP_DEPTH}, ideal support = "
          f"{int((ideal > 1e-12).sum())}/{2 ** N}")
    print(f"  {'theta':>7}{'p_zz':>10}{'e_F':>9}{'bound':>9}{'TVD (RC)':>11}"
          f"{'TVD (no RC)':>13}")

    bounds, tvd_rc, tvd_norc, efs = [], [], [], []
    for th in SWEEP_THETAS:
        noise = NoiseModel(enabled=True, p1=0.0, p2=0.0, p_readout=0.0, p_idle=0.0,
                           theta_zz=th)
        # cycle_benchmark twirls internally -- e_F is the twirled channel's, which is
        # exactly the channel an RC'd circuit experiences.
        cbs = {k: cycle_benchmark(pairs, N, CB_DEPTHS, noise, mode="random",
                                  n_decays=CB_DECAYS, seed=SEED + 10 + j)
               for j, (k, pairs) in enumerate(CYCLES.items())}
        ro_fid, ro_std = readout_fidelity(N, noise, seed=SEED + 20)
        efs_k = {k: (cb["e_F"], cb["e_F_std"]) for k, cb in cbs.items()}
        b = qcap_bound({k: SWEEP_DEPTH for k in CYCLES}, efs_k, ro_fid, ro_std)

        rc_vec, _, _ = rc_distribution(circ, noise, SWEEP_RAND, SEED + 30)
        norc = as_vec(simulate(circ, "statevector", "distribution", noise=noise,
                               shots=0, n_traj=1, seed=SEED + 40))

        mean_ef = float(np.mean([e for e, _ in efs_k.values()]))
        bounds.append(b["error"]); efs.append(mean_ef)
        tvd_rc.append(tvd(rc_vec, ideal)); tvd_norc.append(tvd(norc, ideal))
        flag = "  <-- bound VIOLATED" if tvd_norc[-1] > b["error"] else ""
        print(f"  {th:>7.2f}{noise.twirled().p_zz:>10.5f}{mean_ef:>9.4f}"
              f"{b['error']:>9.4f}{tvd_rc[-1]:>11.4f}{tvd_norc[-1]:>13.4f}{flag}",
              flush=True)
    bounds, tvd_rc, tvd_norc = map(np.array, (bounds, tvd_rc, tvd_norc))
    n_viol = int((tvd_norc > bounds).sum())
    print(f"\n  RC'd TVD under the bound at {int((tvd_rc <= bounds).sum())}/"
          f"{len(bounds)} angles;  un-compiled TVD violates it at {n_viol}/"
          f"{len(bounds)} (the small-angle end, where the bound is quadratic in theta "
          f"and the un-compiled error is linear).\n")
    return bounds, tvd_rc, tvd_norc, np.array(efs), ideal


def main():
    ok, p1 = check_twirl_identity()
    bounds, tvd_rc, tvd_norc, efs, ideal = sweep_theta()

    save_results(OUT_DATA, theta=THETA, n_rand=N_RAND, stim_shots=STIM_SHOTS,
                 rc_dist=p1["rc"], twirl_dist=p1["twirl"], norc_dist=p1["norc"],
                 tvd_rc_vs_twirl=p1["tvd_ab"], tvd_norc_vs_twirl=p1["tvd_ctrl"],
                 tolerance=p1["tol"], passed=ok,
                 sweep_thetas=np.array(SWEEP_THETAS), sweep_bound=bounds,
                 sweep_tvd_rc=tvd_rc, sweep_tvd_norc=tvd_norc, sweep_e_F=efs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    idx = np.arange(2 ** N)
    w = 0.4
    ax1.bar(idx - w / 2, p1["rc"], w, color="#0072B2", label="RC of coherent noise")
    ax1.bar(idx + w / 2, p1["twirl"], w, color="#E69F00",
            label=r"analytic $\mathtt{noise.twirled()}$ (stim)")
    ax1.plot(idx, p1["norc"], "kx", ms=7, mew=1.6, label="no RC (negative control)")
    ax1.set_xlabel("outcome")
    ax1.set_ylabel("probability")
    ax1.set_title(f"RC(coherent) == analytic twirl\n"
                  r"TVD $= {:.4f}$ vs tolerance ${:.4f}$   [{}]".format(
                      p1["tvd_ab"], p1["tol"], "PASS" if ok else "FAIL"), fontsize=10.5)
    ax1.legend(frameon=False, fontsize=8.5)
    ax1.grid(True, axis="y", alpha=0.15)

    th = np.array(SWEEP_THETAS)
    viol = tvd_norc > bounds
    if viol.any():
        ax2.fill_between(th, bounds, tvd_norc, where=viol, color="#009E73", alpha=0.13,
                         label="bound violated (no RC)")
    ax2.plot(th, bounds, "-", color="#D55E00", lw=2.5,
             label=r"QCAP bound (from twirled $e_F$) $\sim\theta^2$")
    ax2.plot(th, tvd_rc, "o-", color="#0072B2", ms=6,
             label=r"measured TVD, randomly compiled $\sim\theta^2$")
    ax2.plot(th, tvd_norc, "s--", color="#009E73", ms=6,
             label=r"measured TVD, NOT compiled $\sim\theta$")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xticks(th)                       # the swept angles, not decade minor ticks
    ax2.set_xticklabels([f"{v:g}" for v in th])
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_xlabel(r"coherent residual-ZZ angle $\theta_{zz}$ (rad)")
    ax2.set_ylabel("probability of an error (TVD)")
    ax2.set_title("CB measures the TWIRLED channel, so the bound covers an RC'd\n"
                  f"circuit -- and only that one (n={N}, depth={SWEEP_DEPTH})",
                  fontsize=10.5)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax2.grid(True, which="both", alpha=0.15)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"\nWrote {OUT}, {OUT_DATA}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
