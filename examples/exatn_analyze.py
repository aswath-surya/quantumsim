"""The C-3PQ bounding analysis, run against XACC/TNQVM/ExaTN output instead.

`c3pq_analyze.py` is the last stage of the bounding pipeline; this file is that same
analysis with the simulation engine swapped. The physics is not reimplemented -- every
check, the CB fit, the TVD assembly, the bound and the figure are imported from
`c3pq_analyze` -- so the two paths cannot silently drift apart.

    build_bounding_bank.py -> qasm_bank_bounding/manifest.json   what each circuit means
    run_exatn_bank.py      -> results_exatn/<stem>.f64           one vector per CIRCUIT
    exatn_analyze.py       -> results/exatn_bounding_*.png/.npz  (this file)

WHAT CHANGES, AND WHY
---------------------
* **Where the vectors come from.** C-3PQ emits one weight-averaged vector per *group*,
  having accumulated the K trajectories inside its own binary. ExaTN has no notion of a
  group: `run_exatn_bank.py` writes one `.f64` per QASM file. `ExatnResults` therefore does
  the accumulation here, reading the per-file weights the bank recorded
  (`build_bounding_bank.Bank.close`). This is the same weighted sum, moved from C++ to
  Python.

* **The parity and top-k tables are derived, not read.** C-3PQ's harness writes
  `.parity.tsv` and `.topk.tsv` because its accumulator never materialises a 2^n vector at
  the widths it targets. Here the vector always exists, so the CB survival and the
  calibration top-k are computed from it directly.

* **Shot noise is real, and the tolerances say so.** This is the substantive difference.
  C-3PQ returns exact probabilities and `c3pq_analyze` tests them at 1e-9 and 1e-12; ExaTN
  samples, so every one of those constants would fail by construction. Each is replaced
  with a statistical tolerance derived from `--shots`, and the noise floor is printed next
  to the number it governs. Two checks needed local variants rather than a rescaled
  constant -- see `check_calibration_sampled` and `check_rc_matches_noisy_sampled`.

* **The TVD is biased upward** by roughly `sum_i sqrt(p_i(1-p_i)/(2 pi S))`, because
  `|p_hat - q|` is convex. It is reported per width and stored in the `.npz`, and it is
  deliberately NOT subtracted: the QCAP bound is an upper bound and the measurement is
  being asked to sit under it, so shrinking the measurement is exactly the direction in
  which a correction could manufacture agreement. At the smoke config (n=2, TVDs ~1e-2)
  the bias is the same order as the signal -- printing it is what keeps that visible.

WHAT DOES NOT CHANGE
--------------------
The Clifford e_F path (`clifford_efs`) is pure stim and never touches ExaTN, so it remains
the reference the C-3PQ arms are cross-checked against. And check [1], the calibration
probe, is a genuine end-to-end test of the backend's bit order: it fails loudly if
`ExaTNBackend.normalize_counts` did not put qubit j in bit j.

Usage:
    python examples/exatn_analyze.py --bank qasm_bank_bounding --results results_exatn
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

import _bootstrap  # noqa: F401
import c3pq_analyze as base
from proxysim import NoiseModel, c3pq, metrics
from proxysim.benchmarking import readout_fidelity

DEFAULT_BANK = base.DEFAULT_BANK
DEFAULT_RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_exatn")


class ExatnResults:
    """`c3pq_analyze.Results` over a flat directory of per-circuit ExaTN vectors.

    Duck-types the interface the imported analysis uses -- `have`, `probs`, `mode_of`,
    `parity`, `topk` -- so nothing downstream knows which engine produced the numbers.

    The bank's group records supply everything needed to reassemble a group from loose
    files: `files[i].stem` names the `.f64`, `files[i].weight` is its share, and `estimator`
    says which of the three products the group is for. Stems are unique bank-wide
    (`c3pq.stage_name` encodes width, mode, depth, instance, arm and trajectory), so the
    flat output layout cannot collide.
    """

    def __init__(self, groups, results_dir: str, allow_partial: bool = False):
        self.results = results_dir
        self.entries = {}          # group -> [(path, weight)], weights summing to 1
        self.mode_of = {}
        self.parity, self.topk, self.norm = {}, {}, {}
        self.partial, self.absent = {}, []

        for g in groups:
            found, missing = [], 0
            n_files = max(len(g["files"]), 1)
            for f in g["files"]:
                path = os.path.join(results_dir, f"{f['stem']}.f64")
                if os.path.exists(path):
                    found.append((path, float(f.get("weight", 1.0 / n_files))))
                else:
                    missing += 1
            if not found:
                self.absent.append(g["name"])
                continue
            if missing:
                # Averaging over the survivors silently changes the estimator -- a group
                # missing half its trajectories is not a K/2 run, it is a biased K run --
                # so by default the group is dropped rather than quietly reweighted.
                self.partial[g["name"]] = (missing, len(g["files"]))
                if not allow_partial:
                    continue
                total = sum(w for _, w in found)
                found = [(p, w / total) for p, w in found]
            self.entries[g["name"]] = found
            self.mode_of[g["name"]] = g["estimator"]

        for g in groups:
            if g["name"] not in self.entries or g["estimator"] == "probs":
                continue
            vec = self._accumulate(g["name"], g["n_qubits"])
            if g["estimator"] == "parity":
                self.parity[g["name"]] = self._survival(vec, g)
            elif g["estimator"] == "topk":
                self.topk[g["name"]] = self._topk(vec, int(g.get("topk", vec.size)))
            self.norm[g["name"]] = float(vec.sum())

    # -- the Results interface -------------------------------------------------
    def have(self, name: str) -> bool:
        return name in self.entries

    def probs(self, name: str, n: int) -> np.ndarray:
        return self._accumulate(name, n)

    # -- assembly --------------------------------------------------------------
    def _accumulate(self, name: str, n: int) -> np.ndarray:
        """The group's weight-averaged probability vector.

        One file is resident at a time; the accumulator is the only 2^n array held, which
        matters at the widths where K files x 2^n doubles would not fit at once.
        """
        acc = np.zeros(1 << n, dtype=np.float64)
        for path, weight in self.entries[name]:
            acc += weight * c3pq.read_probs(path, n)
        return acc

    @staticmethod
    def _survival(vec: np.ndarray, g) -> dict:
        """The CB survival, `sign * sum_i p_i * (-1)^parity(i restricted to support)`.

        Mirrors `c3pq.read_parity`'s contract exactly, **including that the sign is already
        applied**: re-multiplying by `g["sign"]` downstream would flip half the sequences
        negative and yield a plausible-looking decay with a wrong e_F. Both index
        conventions are carried, as the harness does, so `survival_msb` stays available if
        the calibration probe ever contradicts the assumed one.
        """
        n = g["n_qubits"]
        idx = np.arange(vec.size, dtype=np.int64)
        out = {}
        for tag, qubits in (("lsb", g["support"]),
                            ("msb", [n - 1 - q for q in g["support"]])):
            phase = np.ones(vec.size, dtype=np.float64)
            for q in qubits:
                phase *= 1.0 - 2.0 * ((idx >> q) & 1)
            out[tag] = float(np.dot(vec, phase))
        sign = int(g.get("sign", 1))
        return {"sign": sign, "survival": sign * out["lsb"],
                "survival_msb": sign * out["msb"],
                "parity_lsb": sign * out["lsb"], "parity_msb": sign * out["msb"],
                "norm": float(vec.sum())}

    @staticmethod
    def _topk(vec: np.ndarray, k: int) -> dict:
        """The k largest entries, shaped as `c3pq.read_topk` returns them."""
        k = max(1, min(int(k), vec.size))
        idx = np.argpartition(vec, vec.size - k)[vec.size - k:]
        idx = idx[np.argsort(-vec[idx])]
        return {"index": idx.astype(np.int64), "prob": vec[idx].astype(np.float64),
                "norm": float(vec.sum()), "retained": float(vec[idx].sum())}

    def report(self) -> None:
        print(f"  {len(self.entries)} groups readable from {self.results}")
        if self.partial:
            shown = list(self.partial.items())[:3]
            detail = ", ".join(f"{k} ({m}/{t} files missing)" for k, (m, t) in shown)
            print(f"  {len(self.partial)} groups incomplete: {detail}")
        if self.absent:
            print(f"  {len(self.absent)} groups have no results at all "
                  f"(first: {self.absent[:3]})")

    def eff_shots(self, name: str, shots: int) -> float:
        """Shots' worth of independent samples behind this group's averaged vector.

        A weighted mean of K independent estimates has variance ``sum_k w_k^2 var_k``, so
        the weights that `Bank.close` wrote buy a factor ``1 / sum_k w_k^2`` -- exactly K
        while they are equal, and correctly less than K after `--allow-partial`
        renormalises a short group. Every tolerance below is derived from this number
        rather than from `--shots`, because a K=4 trajectory group and a K=1 ideal group
        run at the same `--shots` do not carry the same noise.
        """
        ws = [w for _, w in self.entries.get(name, ())]
        denom = sum(w * w for w in ws)
        return shots / denom if denom > 0 else float(shots)


# ---------------------------------------------------------------------------
# Shot-noise scales
# ---------------------------------------------------------------------------
def _binom_sd(p: np.ndarray, eff: float) -> np.ndarray:
    """Per-bin standard deviation of a sampled probability, floored at one count.

    The floor matters where it is used: an unpopulated bin has ``p_hat = 0`` and a naive
    ``sqrt(p_hat(1-p_hat)/S)`` of exactly zero, which would turn any deviation there into
    an infinite z-score. ``1/eff`` is the smallest resolvable probability, so it is the
    natural floor.
    """
    return np.sqrt(np.maximum(p * (1.0 - p), 1.0 / eff) / eff)


# ---------------------------------------------------------------------------
# Assertion 1: the index convention, at finite shots
# ---------------------------------------------------------------------------
def check_calibration_sampled(groups, res: ExatnResults, shots: int, sigma: float = 5.0):
    """Check [1] with a z-score margin in place of the imported version's ``100 x``.

    `c3pq_analyze.check_calibration` requires the wrong convention to lose by two orders
    of magnitude. That is the right test against exact probabilities, where the correct
    convention scores ~1e-15 and any factor at all is satisfiable. It cannot survive
    sampling: at n=2 the two conventions genuinely differ by ~0.10, while 10k shots put a
    ~0.005 floor under the correct one, so ``100 x`` demands a separation of 0.45 that no
    number of shots will ever produce -- the check would fail on a perfectly correct run.

    The fix is to score in units of the noise rather than in orders of magnitude. Each
    top-k entry gets its own binomial sigma; ``z`` is the largest standardised deviation
    over the k entries, and the threshold carries the ``sqrt(2 ln k)`` widening a maximum
    of k gaussians needs. The correct convention must sit under it and the transposed one
    must not, which is the same two-sided statement as before -- a tie still fails, so a
    mis-built probe (equal ``p_one``, making the conventions indistinguishable) is still
    caught rather than passed.
    """
    out = []
    for g in groups:
        if g["kind"] != "cal" or not res.have(g["name"]):
            continue
        n, p1 = g["n_qubits"], np.array(g["p_one"])
        rec = res.topk[g["name"]]
        eff = res.eff_shots(g["name"], shots)
        thresh = sigma + math.sqrt(2.0 * math.log(max(len(rec["index"]), 2)))
        err, z = {}, {}
        for conv in ("lsb", "msb"):
            ref = []
            for idx in rec["index"]:
                bits = [(idx >> (j if conv == "lsb" else n - 1 - j)) & 1
                        for j in range(n)]
                ref.append(np.prod(np.where(bits, p1, 1 - p1)))
            ref = np.array(ref, dtype=np.float64)
            dev = np.abs(ref - rec["prob"])
            err[conv] = float(np.max(dev))
            z[conv] = float(np.max(dev / _binom_sd(ref, eff)))
        ok = z["lsb"] <= thresh and z["msb"] > thresh
        out.append({"group": g["name"], "n": n, "ok": ok, "eff_shots": eff,
                    "thresh": thresh, **{f"err_{k}": v for k, v in err.items()},
                    **{f"z_{k}": v for k, v in z.items()}})
        print(f"  n={n:<3} max|p - product| : qubit i = bit i  {err['lsb']:.2e} "
              f"(z={z['lsb']:.1f})   qubit i = bit n-1-i  {err['msb']:.2e} "
              f"(z={z['msb']:.1f})   z threshold {thresh:.1f}   "
              f"{'OK' if ok else 'AMBIGUOUS'}")
        if not ok and z["msb"] <= thresh:
            print("       both conventions fit -- the probe cannot discriminate at this "
                  "width and shot count, not evidence that the convention is wrong")
    if not out:
        print("  (no calibration group in the results -- convention unverified)")
    return out


# ---------------------------------------------------------------------------
# Assertion 4: rc vs noisy, with the sampling term added
# ---------------------------------------------------------------------------
def check_rc_matches_noisy_sampled(points, bias, k_traj: int, theta_zz: float,
                                   sigma: float = 5.0):
    """Check [4] with the shot term added in quadrature to the trajectory term.

    The imported tolerance is ``sigma*sqrt(2/K)``, pure trajectory noise, because C-3PQ's
    per-trajectory vectors are exact. Here each arm's TVD also fluctuates with the
    sampling, and that term is bounded by the same per-bin sigmas the bias is built from:
    with ``T = 0.5*sum_i |X_i|``, ``sd(T) <= 0.5*sqrt(sum_i var_i) <= 0.5*sum_i sd_i``,
    and the bias is ``(1/sqrt(2 pi)) * sum_i sd_i``, so ``sd(T) <= sqrt(pi/2) * bias``.
    Two independent arms give a further ``sqrt(2)``.

    The two contributions are printed separately, because which one dominates decides what
    to do about a failure: a trajectory-dominated point needs more K, a shot-dominated one
    needs more shots, and at the smoke config it is the latter by a wide margin.
    """
    if theta_zz:
        print(f"  skipped: theta_zz = {theta_zz} != 0, so the two series are expected "
              f"to differ (that is what the un-compiled series is for)")
        return {"ok": True, "skipped": True}

    traj_var = 2.0 / max(k_traj, 1)
    worst, over, n_pts, tols = 0.0, [], 0, []
    for key, arms in sorted(points.items()):
        if "noisy" not in arms or "rc" not in arms:
            continue
        shot_var = 2.0 * (math.sqrt(math.pi / 2.0) * bias.get(key, 0.0)) ** 2
        tol = sigma * math.sqrt(traj_var + shot_var)
        tols.append(tol)
        d = abs(float(np.mean(arms["noisy"])) - float(np.mean(arms["rc"])))
        worst = max(worst, d)
        n_pts += 1
        if d > tol:
            over.append((key, d, tol))
    if not n_pts:
        print("  no point has both arms, skipped")
        return {"ok": True, "worst": 0.0, "n": 0, "over": 0, "skipped": False}

    mean_bias = float(np.mean([bias.get(k, 0.0) for k in points])) if points else 0.0
    print(f"  {n_pts} (n, mode, depth) points, worst |TVD_noisy - TVD_rc| = {worst:.4f}, "
          f"tolerance {min(tols):.4f}..{max(tols):.4f}")
    print(f"    trajectory term {sigma * math.sqrt(traj_var):.4f} "
          f"({sigma:g} x sqrt(2/K), K={k_traj});  shot term "
          f"{sigma * math.sqrt(2.0) * math.sqrt(math.pi / 2.0) * mean_bias:.4f} "
          f"(from the per-point bias, mean {mean_bias:.4f})")
    for key, d, tol in over[:5]:
        print(f"    OVER  n={key[0]} {key[1]} depth={key[2]}: {d:.4f} > {tol:.4f}")
    return {"ok": not over, "worst": worst, "tol": max(tols) if tols else 0.0,
            "n": n_pts, "over": len(over), "skipped": False}


# ---------------------------------------------------------------------------
# The sampling bias of the TVD
# ---------------------------------------------------------------------------
def tvd_bias(groups, res: ExatnResults, shots: int):
    """``{(n, mode, depth): mean upward bias of the measured TVD}``.

    ``|p_hat - q_hat|`` is convex, so a sampled TVD sits *above* the true one even when
    both arms are unbiased. Where the two distributions agree in a bin -- which is most of
    them, since the noisy arm is a perturbation of the ideal -- the excess is
    ``E|N(0, s)| = s*sqrt(2/pi)``, and summing over bins with the TVD's leading 1/2 gives

        bias ~ sum_i sqrt(var_i / (2 pi)),   var_i = p_i(1-p_i)/eff_p + q_i(1-q_i)/eff_q

    which is the largest the excess can be, hence a ceiling on how much of a measured TVD
    is an artefact. Both arms contribute, at their own effective shot counts: the ideal
    arm is a single circuit while the noisy arm averages K trajectories, so they are not
    interchangeable.

    Readout is not folded in. `apply_readout_to_distribution` is a stochastic matrix, and
    applying one can only contract the per-bin variances, so the un-mixed estimate is the
    conservative one.

    This is reported and stored, never subtracted -- see the module docstring.
    """
    ideal = {}
    for g in groups:
        if g["kind"] == "tvd" and g["arm"] == "ideal" and res.have(g["name"]):
            ideal[(g["n_qubits"], g["mode"], g["depth"], g["instance"])] = g["name"]

    acc = {}
    for g in groups:
        if g["kind"] != "tvd" or g["arm"] == "ideal" or not res.have(g["name"]):
            continue
        key = (g["n_qubits"], g["mode"], g["depth"], g["instance"])
        if key not in ideal:
            continue
        n = g["n_qubits"]
        q = res.probs(ideal[key], n)
        p = res.probs(g["name"], n)
        var = (p * (1.0 - p) / res.eff_shots(g["name"], shots)
               + q * (1.0 - q) / res.eff_shots(ideal[key], shots))
        acc.setdefault(key[:3], []).append(float(np.sqrt(var / (2.0 * math.pi)).sum()))
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ---------------------------------------------------------------------------
# Tolerances for the two imported checks that only need rescaling
# ---------------------------------------------------------------------------
def cb_tolerance(groups, res: ExatnResults, shots: int, sigma: float) -> float:
    """Tolerance for check [2], ``sigma / sqrt(eff)`` at the noisiest ideal CB group.

    This one is generous on purpose, and the reason is worth stating: at
    ``theta_zz == 0`` a noiseless CB sequence ends in a computational-basis eigenstate of
    the measured parity, so *every shot* returns the same sign and the survival is +1
    exactly -- sampling contributes nothing. The tolerance is therefore a ceiling for
    contraction error and for the case where that premise fails, not the expected scale.
    What the check is really for is unaffected: a mis-recorded sign or support moves the
    survival by O(1), not by a few 1/sqrt(S).
    """
    effs = [res.eff_shots(g["name"], shots) for g in groups
            if g["kind"] == "cb" and g["arm"] == "ideal" and res.have(g["name"])]
    return sigma / math.sqrt(min(effs)) if effs else sigma / math.sqrt(max(shots, 1))


def sv_tolerance(groups, res: ExatnResults, shots: int, sigma: float,
                 max_qubits: int) -> float:
    """Tolerance for check [3], which reports a *max* over 2^n bins.

    A per-bin ``sigma`` would be exceeded by chance somewhere in a 2^n-wide sweep, so the
    threshold carries the usual ``sqrt(2 ln M)`` widening for the maximum of M gaussians,
    with ``M = 2^n`` per group and the union over both tails. ``sqrt(0.25/eff)`` is the
    worst-case per-bin sigma (at p = 1/2); using the actual p per bin would tighten this,
    but the check compares a single scalar worst-case deviation, so the bound has to hold
    for whichever bin produced it.
    """
    cand = [(res.eff_shots(g["name"], shots), g["n_qubits"]) for g in groups
            if g["arm"] == "ideal" and g["kind"] != "cal" and res.have(g["name"])
            and g["n_qubits"] <= max_qubits and res.mode_of[g["name"]] == "probs"]
    if not cand:
        return float("inf")
    eff = min(e for e, _ in cand)
    n = max(n for _, n in cand)
    return sigma * math.sqrt(0.25 / eff) * math.sqrt(2.0 * math.log(2.0 * (1 << n)))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", default=DEFAULT_BANK,
                    help="the bounding bank build_bounding_bank.py wrote")
    ap.add_argument("--results", default=DEFAULT_RESULTS,
                    help="run_exatn_bank.py output dir (flat <stem>.f64 files)")
    ap.add_argument("--out", default=_bootstrap.RESULTS_DIR)
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    ap.add_argument("--shots", type=int, default=None,
                    help="shots per circuit; defaults to the value in the run manifest. "
                         "Every statistical tolerance is derived from this, so overriding "
                         "it to something the run did not use will silently mis-scale "
                         "every check")
    ap.add_argument("--allow-partial", action="store_true",
                    help="analyze groups that are missing some trajectories, "
                         "renormalising the surviving weights. Off by default: a group "
                         "missing half its trajectories is a biased estimate of the "
                         "channel, not an unbiased one at lower K")
    ap.add_argument("--sv-max-qubits", type=int, default=base.SV_MAX_QUBITS,
                    help=f"widths above this skip the proxysim cross-check "
                         f"(default {base.SV_MAX_QUBITS})")
    ap.add_argument("--ef-source", choices=("auto", "clifford", "exatn"), default="auto",
                    help="where the e_F that enters the bound comes from. auto = "
                         "clifford while theta_zz == 0. See clifford_efs.__doc__ in "
                         "c3pq_analyze -- the argument there is about exact vs sampled "
                         "trajectories and applies a fortiori here, where the CB arms "
                         "carry shot noise on top of the trajectory noise")
    ap.add_argument("--cb-decays-ref", type=int, default=30,
                    help="random sequences per length on the Clifford e_F path "
                         "(default 30, ~11 s at n=2 and ~2 min at n=20)")
    ap.add_argument("--sigma", type=float, default=5.0,
                    help="how many sigma every statistical check allows (default 5; "
                         "higher than c3pq_analyze's 4 because these tolerances govern "
                         "maxima over many bins and groups)")
    ap.add_argument("--arms", choices=("rc", "noisy", "both"), default="rc",
                    help="which measured series the figure draws (default rc). The QCAP "
                         "bound is stated for the randomly-compiled circuit, so the "
                         "un-compiled arm shares the axis without being what the curve "
                         "bounds. Both are written to the .npz and compared by check [4] "
                         "whatever this is set to")
    ap.add_argument("--no-assert", action="store_true",
                    help="report the checks but do not fail on them")
    args = ap.parse_args()
    arms = ("noisy", "rc") if args.arms == "both" else (args.arms,)

    # The bank manifest is what makes a directory of QASM a *bank*: it records which
    # circuits form a group, their weights, and each group's estimator. Nothing here can be
    # reconstructed from the .qasm files, so say what to run rather than raising the bare
    # FileNotFoundError -- the most likely cause is a results directory passed as --bank,
    # or a brickwork bank from generate_exatn_bank.py, which has no manifest at all.
    bank_manifest = os.path.join(args.bank, "manifest.json")
    if not os.path.exists(bank_manifest):
        raise SystemExit(
            f"no bank manifest at {bank_manifest}.\n"
            f"  --bank must point at a bounding bank, which build_bounding_bank.py writes:\n"
            f"      python examples/build_bounding_bank.py --out {args.bank}\n"
            f"  It is not the run output directory (that is --results), and not the "
            f"brickwork\n  bank from generate_exatn_bank.py, which has no groups to "
            f"analyze.")
    manifest = json.load(open(bank_manifest))
    groups = manifest["groups"]
    cfg, nz = manifest["config"], manifest["noise"]
    noise = NoiseModel(**nz)

    # The run manifest is the authority on how the results were produced. Falling back to
    # a default here would put a made-up number into every tolerance below, so a missing
    # manifest is an error unless --shots says otherwise.
    run = {}
    run_path = os.path.join(args.results, "manifest.json")
    if os.path.exists(run_path):
        run = json.load(open(run_path))
    shots = args.shots or run.get("shots")
    if not shots:
        raise SystemExit(
            f"no shot count: {run_path} is missing or has no 'shots', and --shots was "
            f"not given. Every tolerance in this analysis is derived from it.")

    print(f"bank    {args.bank}\nresults {args.results}")
    print(f"engine  ExaTN via {run.get('visitor', '?')} / compiler "
          f"{run.get('compiler', '?')}, {shots} shots/circuit, "
          f"lowered={run.get('lowered', '?')}")
    print(f"noise   p1={nz['p1']} p2={nz['p2']} p_idle={nz['p_idle']} "
          f"theta_zz={nz['theta_zz']} p_readout={nz['p_readout']}")
    print(f"config  K={cfg['trajectories']} instances={cfg['instances']} "
          f"cb_decays={cfg['cb_decays']} oneq_set={cfg['oneq_set']}")
    # Trajectory noise is 1/sqrt(K) and shot noise is 1/sqrt(S); at the bank defaults the
    # first is ~150x the second, so a run can look shot-rich and still be unmeasurable.
    # Worse, at small K the estimator is visibly *quantised*: most sampled trajectories
    # carry no error at all, so one bad draw out of K moves the averaged distribution by
    # 1/K of its own distance and the TVDs land in a comb at multiples of ~1/K rather than
    # scattering around a mean. That reads as outliers on the plot and is not.
    if cfg["trajectories"] < 32:
        print(f"        WARNING: K={cfg['trajectories']} trajectories gives ~"
              f"{1.0 / math.sqrt(cfg['trajectories']):.2f} trajectory noise per arm, "
              f"against ~{1.0 / math.sqrt(shots):.4f} from {shots} shots. The TVD points "
              f"will be quantised in steps of ~1/K, not scattered. Rebuild the bank with "
              f"--trajectories 150 before reading the figure quantitatively.")
    if cfg["instances"] < 5:
        print(f"        NOTE: instances={cfg['instances']}, so each depth gets only "
              f"{cfg['instances']} marker(s) per arm on the figure.")
    if run.get("failed"):
        print(f"        {len(run['failed'])} circuits FAILED in the run "
              f"(first: {list(run['failed'])[:2]})")
    res = ExatnResults(groups, args.results, allow_partial=args.allow_partial)
    res.report()
    if not res.entries:
        raise SystemExit(f"no group in {args.bank} has any result under {args.results} -- "
                         f"is this the bank the run was over?")
    print()

    checks = {}
    print("[1] index convention (calibration probe)")
    cal = check_calibration_sampled(groups, res, shots, args.sigma)
    checks["calibration"] = all(c["ok"] for c in cal) and bool(cal)

    cb_tol = cb_tolerance(groups, res, shots, args.sigma)
    print(f"[2] noiseless CB survival (tolerance {cb_tol:.2e})")
    checks["cb_noiseless"] = base.check_noiseless_cb(groups, res, tol=cb_tol)["ok"]

    sv_tol = sv_tolerance(groups, res, shots, args.sigma, args.sv_max_qubits)
    print(f"[3] ExaTN vs proxysim statevector (tolerance {sv_tol:.2e})")
    checks["vs_proxysim"] = base.check_against_proxysim(
        groups, res, args.bank, args.sv_max_qubits, tol=sv_tol)["ok"]

    exatn_ef = base.cycle_efs(groups, res)
    points = base.tvd_points(groups, res, noise)
    bias = tvd_bias(groups, res, shots)
    print("[4] rc vs noisy agreement")
    checks["rc_vs_noisy"] = check_rc_matches_noisy_sampled(
        points, bias, cfg["trajectories"], nz["theta_zz"], args.sigma)["ok"]

    source = args.ef_source
    if source == "auto":
        source = "clifford" if nz["theta_zz"] == 0 else "exatn"
    cliff_ef = None
    if source == "clifford" or exatn_ef:
        print(f"\n[5] e_F cross-check (bound uses: {source})")
        if nz["theta_zz"] != 0 and source == "clifford":
            print("  WARNING: theta_zz != 0, so the Clifford path only sees "
                  "noise.twirled() -- a Pauli approximation of the coherent term.")
        cliff_ef = base.clifford_efs(manifest, cfg, noise, args.cb_decays_ref)
        checks["e_F_agree"] = base.check_ef_consistency(
            exatn_ef, cliff_ef, args.sigma, label="exatn")["ok"]
    efs = cliff_ef if source == "clifford" else exatn_ef

    kw = {"theta_zz": nz["theta_zz"], "n_traj": cfg["trajectories"],
          "instances": cfg["instances"], "oneq_set": cfg["oneq_set"],
          "ef_source": source, "cb_decays_ref": args.cb_decays_ref,
          "engine": "exatn", "shots": shots}
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
        if not depths:
            print(f"  no TVD point at n={n}, skipping")
            continue
        for mode in modes:
            for c in cycles:
                fit = efs[(n, mode, c)]
                print(f"  {mode:>10} cycle {c}: e_F = {fit['e_F']:.4f} "
                      f"({fit['e_F_std']:.4f})  f = {fit['f']:.6f}  [{source}]")
            b, bs = base.bounds_for(n, mode, depths, efs, ro_fid, ro_std, cycles)
            bias_col = np.array([bias.get((n, mode, d), float("nan")) for d in depths])
            series[mode] = {"bound": b, "bound_std": bs, "tvd_is_lower_bound": False,
                            "noisy": [points[(n, mode, d)]["noisy"] for d in depths],
                            "rc": [points[(n, mode, d)]["rc"] for d in depths]}
            # The bias column is the whole reason this is not just c3pq_analyze's table:
            # a TVD that is not comfortably larger than its own sampling bias has not been
            # measured, whatever it plots as.
            print(f"  {mode:>10} {'depth':>6}{'bound':>10}{'TVD noisy':>12}"
                  f"{'TVD rc':>10}{'bias<=':>10}")
            for i, d in enumerate(depths):
                print(f"  {'':>10} {d:>6}{b[i]:>10.4f}"
                      f"{np.mean(series[mode]['noisy'][i]):>12.4f}"
                      f"{np.mean(series[mode]['rc'][i]):>10.4f}{bias_col[i]:>10.4f}")
            swamped = [d for i, d in enumerate(depths)
                       if np.mean(series[mode]["rc"][i]) < 2.0 * bias_col[i]]
            if swamped:
                print(f"  {mode:>10} NOTE: at depth(s) {swamped} the measured TVD is "
                      f"within 2x its sampling bias -- raise --shots before reading "
                      f"those points as a measurement")
            kw[f"n{n:02d}_{mode}_bound"] = b
            kw[f"n{n:02d}_{mode}_bound_std"] = bs
            kw[f"n{n:02d}_{mode}_tvd"] = np.array(series[mode]["rc"])
            kw[f"n{n:02d}_{mode}_tvd_nonrc"] = np.array(series[mode]["noisy"])
            kw[f"n{n:02d}_{mode}_tvd_bias"] = bias_col
            kw[f"n{n:02d}_{mode}_e_F"] = np.array(
                [efs[(n, mode, c)]["e_F"] for c in cycles])
            kw[f"n{n:02d}_{mode}_e_F_std"] = np.array(
                [efs[(n, mode, c)]["e_F_std"] for c in cycles])
            for tag_, src in (("exatn", exatn_ef), ("clifford", cliff_ef)):
                if src and all((n, mode, c) in src for c in cycles):
                    kw[f"n{n:02d}_{mode}_e_F_{tag_}"] = np.array(
                        [src[(n, mode, c)]["e_F"] for c in cycles])
        kw[f"n{n:02d}_depths"] = np.array(depths)
        tag = args.tag or f"_{cfg['oneq_set']}"
        written.append(base.figure(
            os.path.join(args.out, f"exatn_bounding_{n}qubits{tag}.png"),
            n, modes, depths, series,
            f"K={cfg['trajectories']} trajectories, "
            rf"$\theta_{{zz}}$={nz['theta_zz']}, $e_F$ from {source}",
            source=f"ExaTN, {shots} shots/circuit", arms=arms))

    npz = metrics.save_results(
        os.path.join(args.out,
                     f"exatn_bounding_data{args.tag or '_' + cfg['oneq_set']}.npz"),
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
