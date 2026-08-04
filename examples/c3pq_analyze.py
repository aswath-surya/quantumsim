"""Post-process C-3PQ output into the `run_bounding` figure: measured TVD vs QCAP bound.

The last stage of the pipeline. Everything before it produced circuits and ran them;
nothing before it computed a physical quantity.

    build_bounding_bank.py -> qasm_bank_bounding/manifest.json   what each circuit means
    c3pq_stage.py          -> staging/jobs.json                  where each result lands
    c3pq_run.sh            -> results/c3pq/<run>/                the results
    c3pq_analyze.py        -> results/c3pq_bounding_*.png/.npz   (this file)

WHAT IS COMPUTED, AND WHERE READOUT ENTERS
------------------------------------------
* **TVD, the headline.** C-3PQ returns an *exact* probability vector per group, already
  weight-averaged over the K trajectories inside the binary. Readout error is then applied
  analytically with `noise.apply_readout_to_distribution` -- exact for independent
  bit-flips, and never baked into a circuit -- and the distance to the noiseless arm is
  `metrics.total_variation_distance`. There is no shot noise anywhere in this number; the
  only residual is the ~1/sqrt(K) trajectory noise. `run_bounding` at `TVD_SHOTS = 50000`
  and `N_TRAJ = 150` carries ~0.003 of the former and ~0.011 of the latter.

  Readout is applied to the noisy arms and NOT to the ideal one, matching what
  `run_bounding` compares (`simulate` folds readout into its returned distribution;
  `exact_distribution` does not) and matching what the bound describes: `ro_fid` is a
  separate factor in `qcap_bound`, so readout must be inside the measured error too.

* **e_F, per cycle.** Two paths, and by default the bound does *not* use the C-3PQ one.
  A CB sequence is Clifford, so at `theta_zz == 0` every stochastic-Pauli trajectory
  leaves a stabilizer state and its signed parity is exactly +-1: C-3PQ's exact
  probability vector removes shot noise, but for CB there is no shot noise left to
  remove, and the estimate stays binomial in `decays * K`. Matching stim's precision
  would cost ~1e4 CB circuits (~7 h of codegen+compile) against ~10 s. So `--ef-source
  auto` fits e_F on the Clifford path and keeps the C-3PQ arms as a cross-check of the
  chain (checks [2] and [5]); it flips to the C-3PQ arms once `theta_zz != 0`, where
  `cp(theta)` is non-Clifford and they are the exact ones. `clifford_efs.__doc__` has the
  full argument. Whichever path is used, both are written to the `.npz`.

  Readout is deliberately *not* applied on either path: it scales the fit amplitude `A`,
  not the decay rate `f`, so e_F is SPAM-independent by construction. Applying it here
  would be double-counting.

* **The bound.** `benchmarking.qcap_bound` over the measured per-cycle e_F with
  `benchmarking.readout_fidelity`, counting each cycle `depth` times exactly as
  run_bounding.py:200 does.

THE ASSERTIONS
--------------
These are not diagnostics printed for reassurance; each one fails a real, distinguishable
way of getting a plausible wrong answer, and `--no-assert` is the only way past them.

1. **Index convention** -- the calibration group's top-k must match a product of
   independent biased coins under `qubit i == bit i`. Nothing in C-3PQ documents its
   convention; the transposed one produces a valid-looking distribution and a wrong TVD.
2. **Noiseless CB survival == +1** to 1e-9. Fails if the propagated Pauli's sign or
   support was mis-recorded, or if lowering changed the circuit -- both of which would
   otherwise show up only as a quietly wrong e_F.
3. **C-3PQ vs proxysim** to 1e-12 on the noiseless arms, wherever a statevector is still
   affordable (`--sv-max-qubits`). This is the end-to-end check that the generated code
   computes the circuit that was staged.
4. **noisy vs rc agree to within 1/sqrt(K)** while `theta_zz == 0`. At zero coherent
   angle the channel is already Pauli-stochastic, so a twirl must return the same channel.
   This is the unbiasedness test of the whole chain -- twirl, virtual-gate noise
   accounting, trajectory sampling, batching and accumulation -- and it is only a valid
   test *because* the expected answer is known exactly. It is skipped, with a note, once
   theta_zz is nonzero, where the two series are supposed to differ.
5. **The two e_F paths agree** within their combined error bars. Weak at smoke
   statistics -- the C-3PQ arms set the tolerance and they are noisy -- but it is what
   makes preferring the Clifford path honest rather than merely cheap, and it tightens
   as `--cb-decays` and K grow.

Usage:
    python examples/c3pq_analyze.py
    python examples/c3pq_analyze.py --results results/c3pq/staging --sv-max-qubits 16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401
from proxysim import NoiseModel, c3pq, cb_emit, metrics, noise as noise_mod
from proxysim.benchmarking import cycle_benchmark, qcap_bound, readout_fidelity

DEFAULT_BANK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qasm_bank_bounding")
DEFAULT_STAGING = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "staging")
SV_MAX_QUBITS = 16       # above this the proxysim cross-check stops being affordable


class Results:
    """Everything the run produced, indexed by group name.

    `jobs.json` says which batch a group landed in and in what form; the batch output
    directory says where. Reading is lazy for `probs` (one 2^n vector per group is the
    only thing here big enough to care about) and eager for the two tabular modes.
    """

    def __init__(self, staging: str, results: str):
        self.results = results
        self.jobs = json.load(open(os.path.join(staging, "jobs.json")))
        self.batch_of, self.mode_of = {}, {}
        self.parity, self.topk, self.norm = {}, {}, {}
        missing = []
        for b in self.jobs["batches"]:
            d = os.path.join(results, b["name"])
            if not os.path.isdir(d):
                missing.append(b["name"])
                continue
            for g in b["groups"]:
                self.batch_of[g["name"]] = d
                self.mode_of[g["name"]] = b["mode"]
            if b["mode"] == "parity":
                self.parity.update(c3pq.read_parity(
                    os.path.join(d, f"{b['name']}.parity.tsv")))
            elif b["mode"] == "topk":
                self.topk.update(c3pq.read_topk(os.path.join(d, f"{b['name']}.topk.tsv")))
            norm = os.path.join(d, f"{b['name']}.norm.tsv")
            if os.path.exists(norm):
                self.norm.update(c3pq.read_norms(norm))
        if missing:
            raise SystemExit(f"{len(missing)} batches have no output directory under "
                             f"{results} (first: {missing[:3]}) -- run c3pq_run.sh")

    def probs(self, name: str, n: int) -> np.ndarray:
        return c3pq.read_probs(os.path.join(self.batch_of[name], f"{name}.f64"), n)

    def have(self, name: str) -> bool:
        return name in self.batch_of


# ---------------------------------------------------------------------------
# Assertion 1: the index convention
# ---------------------------------------------------------------------------
def check_calibration(groups, res: Results, tol: float = 1e-9):
    """Confirm C-3PQ puts qubit ``i`` in bit ``i``, from the calibration probe.

    The probe is a product state, so its reference is a product of 2x2 factors and this
    works at any width -- which matters, because the statevector cross-check below does
    not. Both conventions are scored and the wrong one must lose by a wide margin, not
    merely lose: at ``p_j`` all equal the two would be indistinguishable, and the point of
    `build_bounding_bank.calibration_circuit` choosing distinct increasing biases is that
    a tie here means the probe was built wrong rather than that the check passed.
    """
    out = []
    for g in groups:
        if g["kind"] != "cal" or not res.have(g["name"]):
            continue
        n, p1 = g["n_qubits"], np.array(g["p_one"])
        rec = res.topk[g["name"]]
        err = {}
        for conv in ("lsb", "msb"):
            ref = []
            for idx in rec["index"]:
                bits = [(idx >> (j if conv == "lsb" else n - 1 - j)) & 1
                        for j in range(n)]
                ref.append(np.prod(np.where(bits, p1, 1 - p1)))
            err[conv] = float(np.max(np.abs(np.array(ref) - rec["prob"])))
        ok = err["lsb"] < tol and err["msb"] > 100 * max(err["lsb"], 1e-15)
        out.append({"group": g["name"], "n": n, "ok": ok, **err})
        print(f"  n={n:<3} max|p - product| : qubit i = bit i  {err['lsb']:.2e}   "
              f"qubit i = bit n-1-i  {err['msb']:.2e}   {'OK' if ok else 'AMBIGUOUS'}")
    if not out:
        print("  (no calibration group in the results -- convention unverified)")
    return out


# ---------------------------------------------------------------------------
# Assertions 2-3: CB survival on the noiseless arm, and C-3PQ vs proxysim
# ---------------------------------------------------------------------------
def check_noiseless_cb(groups, res: Results, tol: float = 1e-9):
    """Every noiseless CB sequence must return survival exactly +1."""
    worst, n_checked = 0.0, 0
    for g in groups:
        if g["kind"] != "cb" or g["arm"] != "ideal" or not res.have(g["name"]):
            continue
        s = res.parity[g["name"]]["survival"]      # sign already applied on write
        worst = max(worst, abs(1.0 - s))
        n_checked += 1
    print(f"  {n_checked} noiseless CB sequences, worst |1 - survival| = {worst:.2e}")
    return {"n": n_checked, "worst": worst, "ok": n_checked > 0 and worst < tol}


def check_against_proxysim(groups, res: Results, bank: str, max_qubits: int,
                           tol: float = 1e-12, limit: int = 40):
    """Re-simulate the noiseless arms in proxysim and compare, where 2^n still fits.

    This is the only check that ties the generated C++ back to the circuit that was
    staged. It is also the one that runs out first -- hence `--sv-max-qubits`, and hence
    the calibration probe, which keeps *some* index-level check alive at every width.
    """
    from proxysim.backends.statevector import StatevectorBackend
    from proxysim.circuit import circuit_from_qasm

    sv, worst, checked = StatevectorBackend(), 0.0, []
    for g in groups:
        if g["arm"] != "ideal" or g["kind"] == "cal" or not res.have(g["name"]):
            continue
        n = g["n_qubits"]
        if n > max_qubits or res.mode_of[g["name"]] != "probs" or len(checked) >= limit:
            continue
        circ, _ = circuit_from_qasm(
            open(os.path.join(bank, g["dir"], g["files"][0]["file"])).read())
        ref = np.zeros(1 << n)
        for k, p in sv.exact_distribution(circ).items():
            ref[int(k[::-1], 2)] = p            # proxysim key char j is qubit j
        d = float(np.max(np.abs(ref - res.probs(g["name"], n))))
        worst = max(worst, d)
        checked.append(g["name"])
    if not checked:
        print(f"  skipped: no probs group at n <= {max_qubits}")
        return {"n": 0, "worst": float("nan"), "ok": True}
    print(f"  {len(checked)} noiseless arms re-simulated, worst max|dp| = {worst:.2e}")
    return {"n": len(checked), "worst": worst, "ok": worst < tol}


# ---------------------------------------------------------------------------
# Cycle benchmarking -> e_F
# ---------------------------------------------------------------------------
def cycle_efs(groups, res: Results):
    """``{(n, mode, cycle): analyze_cb(...)}`` from the noisy CB groups.

    The survival is exact rather than sampled: the harness evaluated the signed parity
    against the probability vector itself, so `read_parity` already returns it signed. `analyze_cb` then does
    the same `A*f^m` fit and the same ``e_F = (1 - 4^-n)(1 - f)`` conversion as
    `cycle_benchmark`, so the number is comparable to run_bounding's by construction.
    """
    by_cycle = defaultdict(lambda: defaultdict(list))
    for g in groups:
        if g["kind"] != "cb" or g["arm"] != "noisy" or not res.have(g["name"]):
            continue
        s = res.parity[g["name"]]["survival"]      # sign already applied on write
        by_cycle[(g["n_qubits"], g["mode"], g["cycle"])][g["cb_depth"]].append(s)
    return {key: cb_emit.analyze_cb(dict(depths), key[0])
            for key, depths in by_cycle.items()}


def clifford_efs(manifest, cfg, noise, decays, seed=1):
    """The same ``{(n, mode, cycle): analyze_cb(...)}``, from the *Clifford* path.

    WHY THIS EXISTS, AND WHY IT IS THE DEFAULT
    ------------------------------------------
    A CB sequence is Clifford by construction -- `build_cb` dresses the cycle with CB's
    own Clifford 1q set precisely so the propagated Pauli can be tracked with a tableau.
    At ``theta_zz == 0`` the noise is stochastic Pauli, so every trajectory leaves a
    stabilizer state and its signed parity is exactly **+1 or -1**, never anything
    between. The measured survivals show it: they come out quantised to multiples of
    ``2/K``.

    That is the whole argument. C-3PQ's exact probability vector removes shot noise, but
    for CB there is no shot noise left to remove -- the survival estimate is binomial in
    ``decays * K`` no matter how exactly each trajectory is evaluated. Reaching 8% on
    e_F needs ~1e4 CB circuits, which at the measured ~2 s/circuit codegen+compile is
    ~7 h of build per width, spent on a calculation stim finishes in ~10 s at n=2 and
    ~2 min at n=20. So the C-3PQ CB arms are kept as a *cross-check of the chain* (which
    is what they are genuinely good at, and what check [2] and check [5] use them for),
    and the number that enters the bound comes from here.

    The shortcut is only valid while the circuits really are Clifford. ``theta_zz != 0``
    makes the crosstalk ``cp(theta)`` non-Clifford, `cycle_benchmark` falls back to
    `noise.twirled()` -- a Pauli approximation of the coherent term -- and the C-3PQ arms
    become the exact ones. `--ef-source auto` switches on exactly that condition.
    """
    out = {}
    for n in cfg["widths"]:
        for mode in cfg["modes"]:
            for cycle, pairs in manifest["cycles"][str(n)].items():
                out[(n, mode, cycle)] = cycle_benchmark(
                    [tuple(p) for p in pairs], n, cfg["cb_depths"], noise,
                    mode=mode, n_decays=decays, seed=seed)
    return out


def check_ef_consistency(c3pq_ef, cliff_ef, sigma=4.0, label="c3pq"):
    """Check [5]: the two e_F paths must agree within their combined error bars.

    This is the check that makes using the Clifford path honest. It is one-sided in cost
    and two-sided in meaning: the C-3PQ arms are far noisier, so the tolerance is set by
    them, but a *disagreement* would say the two paths are not measuring the same cycle
    -- a lowering error, a wrong dressing set, or a noise budget that differs between
    `sample_trajectory` and `_apply_round`. At smoke statistics this check is weak by
    construction; it tightens as `--cb-decays` and K grow, and it is the reason those
    arms are still worth emitting.
    """
    rows, worst = [], 0.0
    for key in sorted(set(c3pq_ef) & set(cliff_ef)):
        a, b = c3pq_ef[key], cliff_ef[key]
        tol = sigma * math.hypot(a["e_F_std"], b["e_F_std"])
        d = abs(a["e_F"] - b["e_F"])
        rows.append((key, a["e_F"], a["e_F_std"], b["e_F"], b["e_F_std"], d, tol))
        worst = max(worst, d - tol)
    for key, ae, as_, be, bs, d, tol in rows:
        flag = "" if d <= tol else "   <-- disagrees"
        # `label` names the non-Clifford path, for the same reason `figure` takes `source`:
        # exatn_analyze.py reuses this check and its numbers are not C-3PQ's.
        print(f"  n={key[0]} {key[1]:>10} cycle {key[2]}:  {label} {ae:.4f}({as_:.4f})"
              f"   clifford {be:.4f}({bs:.4f})   |d|={d:.4f} tol={tol:.4f}{flag}")
    if not rows:
        print("  no cycle has both paths, skipped")
    return {"ok": worst <= 0.0, "rows": rows}


# ---------------------------------------------------------------------------
# TVD
# ---------------------------------------------------------------------------
def tvd_points(groups, res: Results, noise: NoiseModel):
    """``{(n, mode, depth): {arm: [tvd per instance]}}`` for the noisy and rc arms.

    Both arms are compared against the *same* ideal group, so any residual difference
    between them is a difference of channels and not of reference. Readout is applied to
    the noisy arms only -- see the module docstring.

    A group whose ``probs`` estimator was demoted to ``topk`` (too wide for a 2^n
    accumulator, `c3pq_stage.estimator_for`) yields a *lower bound* rather than the TVD;
    those points are collected separately under the ``bound`` flag -- see `tvd_from_topk`.
    """
    ideal = {}
    for g in groups:
        if g["kind"] == "tvd" and g["arm"] == "ideal" and res.have(g["name"]):
            ideal[(g["n_qubits"], g["mode"], g["depth"], g["instance"])] = g["name"]

    out = defaultdict(lambda: defaultdict(list))
    cache_key, cache_vec = None, None
    for g in sorted((g for g in groups if g["kind"] == "tvd" and g["arm"] != "ideal"),
                    key=lambda g: (g["n_qubits"], g["mode"], g["depth"], g["instance"])):
        key = (g["n_qubits"], g["mode"], g["depth"], g["instance"])
        if not res.have(g["name"]) or key not in ideal:
            continue
        n = g["n_qubits"]
        if res.mode_of[g["name"]] == "topk" or res.mode_of[ideal[key]] == "topk":
            out[key[:3]][g["arm"]].append(
                tvd_from_topk(res.topk[ideal[key]], res.topk[g["name"]]))
            out[key[:3]]["_bound"] = True
            continue
        if cache_key != key:                      # the ideal arm is read once per point
            cache_key = key
            cache_vec = res.probs(ideal[key], n)
        p = noise_mod.apply_readout_to_distribution(res.probs(g["name"], n), noise, n)
        out[key[:3]][g["arm"]].append(0.5 * float(np.abs(cache_vec - p).sum()))
    return out


def tvd_from_topk(ideal_rec, noisy_rec) -> float:
    """A rigorous LOWER BOUND on the TVD from two truncated distributions.

    When a width is too large to accumulate a 2^n vector, all that survives is each
    arm's largest few probabilities plus the mass they retain. The exact TVD is then not
    recoverable -- but a bound is, and a bound is still a usable y-value because the
    quantity it is being compared against is itself an upper bound (QCAP). A lower bound
    on the measured error and an upper bound on the predicted error can only ever *fail*
    to bracket, so this cannot manufacture agreement.

    Take ``A`` = the ideal's retained support. Then, exactly,

        TVD >= P_ideal(A) - P_noisy(A)

    because TVD is the supremum of that difference over all events. ``P_ideal(A)`` is the
    ideal's retained mass. ``P_noisy(A)`` is not known outright: the noisy arm reports
    only its own top-k, so for the elements of ``A`` it did not retain, each contributes
    at most ``t`` (its smallest retained probability -- nothing outside the top-k can
    exceed the smallest inside it) and they contribute at most ``1 - R_noisy`` in total.
    Taking the smaller of those two caps gives the tightest available upper bound on
    ``P_noisy(A)``, hence the tightest lower bound on the TVD.

    Readout is deliberately not applied here: `apply_readout_to_distribution` mixes mass
    across the whole 2^n index space and there is no truncated form of it that stays a
    bound. The bound is therefore against the pre-readout noisy arm, which is looser
    still -- conservative in the same direction.
    """
    keep = float(ideal_rec["retained"])
    known = dict(zip(noisy_rec["index"].tolist(), noisy_rec["prob"].tolist()))
    t = float(noisy_rec["prob"].min()) if len(noisy_rec["prob"]) else 0.0
    hit = [known[i] for i in ideal_rec["index"].tolist() if i in known]
    n_missing = len(ideal_rec["index"]) - len(hit)
    slack = min(n_missing * t, max(0.0, 1.0 - float(noisy_rec["retained"])))
    return max(0.0, keep - (sum(hit) + slack))


def check_rc_matches_noisy(points, k_traj: int, theta_zz: float, sigma: float = 4.0):
    """Assertion 4: at theta_zz == 0 the rc and noisy series must coincide.

    A Pauli twirl of a channel that is already Pauli-stochastic returns the same channel,
    so the two arms differ only by having drawn different trajectories. The scale of that
    difference is the trajectory noise, ~1/sqrt(K) per arm, hence sqrt(2/K) on the
    difference of two independent means; `sigma` is how many of those a point is allowed.
    """
    if theta_zz:
        print(f"  skipped: theta_zz = {theta_zz} != 0, so the two series are expected "
              f"to differ (that is what the un-compiled series is for)")
        return {"ok": True, "skipped": True}
    tol = sigma * math.sqrt(2.0 / max(k_traj, 1))
    worst, over, n_pts = 0.0, [], 0
    for key, arms in sorted(points.items()):
        if "noisy" not in arms or "rc" not in arms:
            continue
        d = abs(float(np.mean(arms["noisy"])) - float(np.mean(arms["rc"])))
        worst = max(worst, d)
        n_pts += 1
        if d > tol:
            over.append((key, d))
    print(f"  {n_pts} (n, mode, depth) points, worst |TVD_noisy - TVD_rc| = {worst:.4f}, "
          f"tolerance {tol:.4f} ({sigma:g} x sqrt(2/K), K={k_traj})")
    for key, d in over[:5]:
        print(f"    OVER  n={key[0]} {key[1]} depth={key[2]}: {d:.4f}")
    return {"ok": not over, "worst": worst, "tol": tol, "n": n_pts,
            "over": len(over), "skipped": False}


# ---------------------------------------------------------------------------
# The bound, and the figure
# ---------------------------------------------------------------------------
def bounds_for(n, mode, depths, efs, ro_fid, ro_std, cycles):
    """The QCAP bound at each depth: every cycle appears ``depth`` times
    (run_bounding.py:200), and its e_F is the one CB just measured."""
    ef = {c: (efs[(n, mode, c)]["e_F"], efs[(n, mode, c)]["e_F_std"]) for c in cycles}
    out = [qcap_bound({c: d for c in cycles}, ef, ro_fid, ro_std) for d in depths]
    return np.array([b["error"] for b in out]), np.array([b["std"] for b in out])


def figure(path, n, modes, depths, series, title_extra, source="C-3PQ exact trajectories"):
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4.8), sharey=True,
                             squeeze=False)
    for ax, mode in zip(axes[0], modes):
        s = series[mode]
        # A demoted (topk) width gives a lower bound, not the TVD; say so on the axis
        # rather than letting a caret-free marker imply a measurement -- see tvd_from_topk.
        lb = s.get("tvd_is_lower_bound")
        mk, suffix = ("^", r" (lower bound)") if lb else ("s", "")
        for d, pts in zip(depths, s["noisy"]):
            ax.plot([d] * len(pts), pts, mk, color="#009E73", ms=4, alpha=0.45,
                    label="measured TVD, NOT compiled" + suffix if d == depths[0] else None)
        for d, pts in zip(depths, s["rc"]):
            ax.plot([d] * len(pts), pts, "^" if lb else "o", color="#0072B2", ms=5,
                    alpha=0.7,
                    label="measured TVD, randomly compiled" + suffix if d == depths[0] else None)
        ax.plot(depths, s["bound"], "-", color="#D55E00", lw=2,
                label="QCAP bound (on the RC'd circuit)")
        ax.fill_between(depths, s["bound"] - 2.96 * s["bound_std"],
                        s["bound"] + 2.96 * s["bound_std"], color="#D55E00", alpha=0.2)
        ax.set_xlabel("circuit depth")
        ax.set_title(f"{mode} ansatz", fontsize=11)
        ax.grid(True, which="both", alpha=0.15)
        ax.legend(frameon=False, fontsize=8.5)
    axes[0][0].set_ylabel("probability of an error (TVD)")
    # `source` names what produced the probability vectors. It is a parameter rather than
    # a constant because exatn_analyze.py reuses this figure for sampled ExaTN runs, where
    # "exact trajectories" would be a false claim on the plot itself.
    fig.suptitle(f"Bounding circuit error from cycle benchmarking (n={n}, {source}, "
                 f"{title_extra})", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", default=DEFAULT_BANK)
    ap.add_argument("--staging", default=DEFAULT_STAGING)
    ap.add_argument("--results", default=None,
                    help="c3pq_run.sh output dir (default results/c3pq/<staging name>)")
    ap.add_argument("--out", default=_bootstrap.RESULTS_DIR)
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    ap.add_argument("--sv-max-qubits", type=int, default=SV_MAX_QUBITS,
                    help=f"widths above this skip the proxysim cross-check "
                         f"(default {SV_MAX_QUBITS})")
    ap.add_argument("--ef-source", choices=("auto", "clifford", "c3pq"), default="auto",
                    help="where the e_F that enters the bound comes from. auto = "
                         "clifford while theta_zz == 0 (the CB sequences are Clifford "
                         "there, so C-3PQ's exact vectors buy nothing and stim is ~1e3x "
                         "cheaper for the same precision), c3pq once it is nonzero. "
                         "See clifford_efs.__doc__")
    ap.add_argument("--cb-decays-ref", type=int, default=30,
                    help="random sequences per length on the Clifford e_F path "
                         "(default 30, ~11 s at n=2 and ~2 min at n=20)")
    ap.add_argument("--sigma", type=float, default=4.0,
                    help="how many trajectory-noise sigma the noisy/rc agreement check "
                         "allows (default 4)")
    ap.add_argument("--no-assert", action="store_true",
                    help="report the checks but do not fail on them")
    args = ap.parse_args()

    results = args.results or os.path.join(
        _bootstrap.RESULTS_DIR, "c3pq", os.path.basename(args.staging.rstrip("/")))
    manifest = json.load(open(os.path.join(args.bank, "manifest.json")))
    groups = manifest["groups"]
    cfg, nz = manifest["config"], manifest["noise"]
    noise = NoiseModel(**nz)
    res = Results(args.staging, results)

    print(f"bank    {args.bank}\nresults {results}")
    print(f"noise   p1={nz['p1']} p2={nz['p2']} p_idle={nz['p_idle']} "
          f"theta_zz={nz['theta_zz']} p_readout={nz['p_readout']}")
    print(f"config  K={cfg['trajectories']} instances={cfg['instances']} "
          f"cb_decays={cfg['cb_decays']} oneq_set={cfg['oneq_set']}\n")

    checks = {}
    print("[1] index convention (calibration probe)")
    cal = check_calibration(groups, res)
    checks["calibration"] = all(c["ok"] for c in cal) and bool(cal)
    print("[2] noiseless CB survival")
    checks["cb_noiseless"] = check_noiseless_cb(groups, res)["ok"]
    print("[3] C-3PQ vs proxysim statevector")
    checks["vs_proxysim"] = check_against_proxysim(
        groups, res, args.bank, args.sv_max_qubits)["ok"]

    c3pq_ef = cycle_efs(groups, res)
    points = tvd_points(groups, res, noise)
    print("[4] rc vs noisy agreement")
    rc_check = check_rc_matches_noisy(points, cfg["trajectories"], nz["theta_zz"],
                                      args.sigma)
    checks["rc_vs_noisy"] = rc_check["ok"]

    # e_F source. The CB arms are Clifford (build_cb dresses with CB's own Clifford set),
    # so while theta_zz == 0 the survival per trajectory is exactly +-1 and the C-3PQ
    # arms are binomial in decays*K -- see clifford_efs.__doc__ for the cost argument.
    source = args.ef_source
    if source == "auto":
        source = "clifford" if nz["theta_zz"] == 0 else "c3pq"
    cliff_ef = None
    if source == "clifford" or c3pq_ef:
        print(f"\n[5] e_F cross-check (bound uses: {source})")
        if nz["theta_zz"] != 0 and source == "clifford":
            print("  WARNING: theta_zz != 0, so the Clifford path only sees "
                  "noise.twirled() -- a Pauli approximation of the coherent term.")
        cliff_ef = clifford_efs(manifest, cfg, noise, args.cb_decays_ref)
        checks["e_F_agree"] = check_ef_consistency(c3pq_ef, cliff_ef, args.sigma)["ok"]
    efs = cliff_ef if source == "clifford" else c3pq_ef

    kw = {"theta_zz": nz["theta_zz"], "n_traj": cfg["trajectories"],
          "instances": cfg["instances"], "oneq_set": cfg["oneq_set"],
          "ef_source": source, "cb_decays_ref": args.cb_decays_ref}
    written = []
    for n in cfg["widths"]:
        cycles = list(manifest["cycles"][str(n)])
        modes = [m for m in cfg["modes"] if all((n, m, c) in efs for c in cycles)]
        if not modes:
            print(f"\nn={n}: no CB fit available, skipping")
            continue
        ro_fid, ro_std = readout_fidelity(n, noise, seed=3)
        print(f"\n=== n={n} === readout fidelity {ro_fid:.4f} ({ro_std:.4f})")
        series = {}
        depths = [d for d in cfg["depths"] if (n, modes[0], d) in points]
        for mode in modes:
            for c in cycles:
                fit = efs[(n, mode, c)]
                print(f"  {mode:>10} cycle {c}: e_F = {fit['e_F']:.4f} "
                      f"({fit['e_F_std']:.4f})  f = {fit['f']:.6f}  [{source}]")
            b, bs = bounds_for(n, mode, depths, efs, ro_fid, ro_std, cycles)
            lower = any(points[(n, mode, d)].get("_bound") for d in depths)
            series[mode] = {"bound": b, "bound_std": bs, "tvd_is_lower_bound": lower,
                            "noisy": [points[(n, mode, d)]["noisy"] for d in depths],
                            "rc": [points[(n, mode, d)]["rc"] for d in depths]}
            if lower:
                print(f"  {mode:>10} NOTE: this width was demoted to topk, so the TVD "
                      f"columns are LOWER BOUNDS (tvd_from_topk)")
            print(f"  {mode:>10} {'depth':>6}{'bound':>10}"
                  f"{'TVD noisy' if not lower else 'TVD>= noisy':>12}"
                  f"{'TVD rc' if not lower else 'TVD>= rc':>10}")
            for i, d in enumerate(depths):
                print(f"  {'':>10} {d:>6}{b[i]:>10.4f}"
                      f"{np.mean(series[mode]['noisy'][i]):>12.4f}"
                      f"{np.mean(series[mode]['rc'][i]):>10.4f}")
            kw[f"n{n:02d}_{mode}_bound"] = b
            kw[f"n{n:02d}_{mode}_bound_std"] = bs
            kw[f"n{n:02d}_{mode}_tvd"] = np.array(series[mode]["rc"])
            kw[f"n{n:02d}_{mode}_tvd_nonrc"] = np.array(series[mode]["noisy"])
            kw[f"n{n:02d}_{mode}_e_F"] = np.array(
                [efs[(n, mode, c)]["e_F"] for c in cycles])
            kw[f"n{n:02d}_{mode}_e_F_std"] = np.array(
                [efs[(n, mode, c)]["e_F_std"] for c in cycles])
            # both paths are kept whatever the bound used, so the choice stays auditable
            for tag_, src in (("c3pq", c3pq_ef), ("clifford", cliff_ef)):
                if src and all((n, mode, c) in src for c in cycles):
                    kw[f"n{n:02d}_{mode}_e_F_{tag_}"] = np.array(
                        [src[(n, mode, c)]["e_F"] for c in cycles])
        kw[f"n{n:02d}_depths"] = np.array(depths)
        tag = args.tag or f"_{cfg['oneq_set']}"
        written.append(figure(
            os.path.join(args.out, f"c3pq_bounding_{n}qubits{tag}.png"),
            n, modes, depths, series,
            f"K={cfg['trajectories']} trajectories, "
            rf"$\theta_{{zz}}$={nz['theta_zz']}, $e_F$ from {source}"))

    npz = metrics.save_results(
        os.path.join(args.out, f"c3pq_bounding_data{args.tag or '_' + cfg['oneq_set']}.npz"),
        **kw)
    print(f"\nwrote {npz}")
    for p in written:
        print(f"wrote {p}")

    print("\nchecks: " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}"
                                   for k, v in checks.items()))
    failed = [k for k, v in checks.items() if not v]
    if failed and not args.no_assert:
        print(f"FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
