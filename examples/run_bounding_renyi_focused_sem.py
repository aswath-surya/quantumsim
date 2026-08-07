"""Bound circuit error from cycle benchmarking -- QCAP, exact white-noise, and Renyi-2.

A copy of ``examples/run_bounding.py`` (which is left untouched) with three additions:

  1. THREE bounds at every depth instead of one.
  2. A per-instance treatment: the two new bounds depend on the IDEAL distribution, so
     they are computed circuit by circuit and reported as a spread, not a single line.
  3. A full depth-by-N study with measured TVD, QCAP, exact-white-noise, and
     Renyi-2 statistics, plus fixed-depth fits versus N.

The pipeline that produces the noise numbers is unchanged:
  1. Cycle-benchmark each distinct entangling cycle in the ansatz -> e_F per cycle.
  2. Measure the readout (SPAM) fidelity.
  3. For circuits of growing depth, form the QCAP error parameter
        eps_qcap = 1 - ro_fid * prod_c (1 - e_F_c)^(times c is used),
     and compare it to the ACTUAL total-variation distance of the noisy circuit
     from its ideal distribution.

THE THREE BOUNDS
----------------
``qcap_bound`` returns eps_qcap. Call that the RAW QCAP BOUND: it bounds the error
probability, and using it as a TVD bound is the crude "all of the error mass could land
anywhere" statement.

Assume instead the global white-noise model, i.e. that the noisy distribution is

    q = (1 - eps_qcap) p + eps_qcap u,      u(x) = 2^-N,

with p the ideal distribution. Then the TVD is not bounded but EXACT:

    D_TV(p, q) = eps_qcap * D_TV(p, u).

That gives the circuit-specific EXACT WHITE-NOISE BOUND

    B_exact = eps_qcap * (1/2) sum_x |p(x) - 2^-N|.

It is "exact" ONLY for the assumed global white-noise mixture -- it is a model, not a
guarantee, and every label, legend and printout here says so.

D_TV(p, u) needs the full ideal distribution. The order-2 Renyi divergence

    D_2(p||u) = log( sum_x p(x)^2 / u(x) ) = log( 2^N sum_x p(x)^2 )

needs only the collision probability, and by the chi-squared / Renyi inequality
D_TV(p, u) <= (1/2) sqrt(e^{D_2(p||u)} - 1). Hence the RENYI BOUND

    B_Renyi = eps_qcap * min(1, (1/2) sqrt(2^N sum_x p(x)^2 - 1)),

clipped because a TVD can never exceed one. Both the clipped and the unclipped
square-root factor are retained: the unclipped one is what has interesting N-scaling.

By construction  B_exact <= B_Renyi <= raw QCAP  for every circuit, and the script
asserts exactly that (plus 0 <= bound <= 1) on every instance it generates.

Everything else -- circuit construction, cycle benchmarking, the randomly-compiled TVD
measurement, the two ansaetze, the plotting style, the .npz output -- is inherited from
``run_bounding.py``; see that file's docstring for the ONEQ_SET / THETA_ZZ / N_TRAJ
discussion, which applies verbatim.

Outputs (``_TAG`` is "" for the default Clifford ansatz, "_t" for Clifford+T):
    results/bounding_renyi_data_sem{_TAG}.npz
    results/renyi_scaling_data_sem{_TAG}.npz
    results/rc_bound_scaling_vs_N_depth{d}_sem{_TAG}.png
        One four-panel TVD/exact/Renyi/QCAP and ideal-factor plot per depth.
    results/renyi_factors_vs_N_depth{d}_sem{_TAG}.png
        Dedicated Renyi-factor plot versus N with exponential and Haar-crossover fits.
    results/rc_collision_entropy_density_grid_sem{_TAG}.png
        Combined depth-by-N collision-entropy-density heatmap.

Only these focused plotting functions are called by ``main``. Other analysis helpers
remain available in the file but are not called automatically.

Run:  python examples/run_bounding_renyi_focused.py
"""

from __future__ import annotations

import math
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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
DEPTHS = [1, 2, 4, 8, 16, 32, 64]
N_INSTANCES = 80                 # random circuit instances per depth (TVD scatter)
TVD_SHOTS = 50000                # sets the TVD noise floor (~0.003 here); sampling is
                                 # setup-bound, so more shots are essentially free
THETA_ZZ = 0.00                  # coherent residual-ZZ angle, rad. 0.0 -> Pauli only.
#ONEQ_SET = "clifford"           # "clifford" -> stabilizer ansatz, exact stim path.
ONEQ_SET = "t"                   # "t"        -> append T to the 1q set. Non-Clifford,
                                 #               so every point goes through N_TRAJ
                                 #               trajectories, but the TVD bands
                                 #               dissolve. See run_bounding.py.
N_TRAJ = 150                     # trajectories per point on any series that cannot go
                                 # through stim. Sets a noise floor on those points and
                                 # is the whole runtime knob.
SEED = 42

# Cycle-benchmark fit statistics. These larger values reduce the CB/readout
# parameter uncertainty that is propagated through QCAP, especially at large depth.
CB_DEPTHS = [1, 2, 4, 8, 16, 32, 64]
CB_SHOTS = 4000
CB_DECAYS = 100

# -- Full depth-by-N study: CB, noisy TVD, exact-WN, Renyi, and ideal factors. --
N_SCALING = [2, 3, 4, 5, 6, 7, 8, 9, 10]
SCALING_DEPTHS = [1, 2, 4, 8, 16, 32, 64, 128]
SCALING_INSTANCES = 40
PROJECTION_N_MAX = 32             # extrapolation endpoint for entropy/TVD projection plots
SCALING_SEED = 90000             # base seed; every (N, anåsatz, instance) gets its own
SCALING_MEM_BUDGET_GB = 2.0      # skip an N whose dense statevector would exceed this
BAND = "percentile"              # "percentile" -> 16th-84th across instances
                                 # "std"        -> mean +/- 1 standard deviation

NOISE = NoiseModel(enabled=True, p1=1e-3, p2=1e-2, p_readout=1e-2, p_idle=1e-3,
                   theta_zz=THETA_ZZ)
TWIRLED = NOISE.twirled()        # what RC produces, and what cycle_benchmark measures

# The 1q gate set per ansatz mode. T goes into the RANDOM ansatz only, so the two
# panels become a controlled comparison: same cycles, same noise, same bound, and the
# only difference is whether the ideal output is a stabilizer state.
_ONEQ = {"random": list(GATE_SETS["clifford"]),
         "structured": list(GATE_SETS["structured"])}
if ONEQ_SET == "t":
    _ONEQ["random"].append("t")
elif ONEQ_SET != "clifford":
    raise ValueError(f"ONEQ_SET must be 'clifford' or 't', got {ONEQ_SET!r}")

MODES = ("random", "structured")
CLIFFORD_ANSATZ = ONEQ_SET == "clifford"
_TAG = "" if CLIFFORD_ANSATZ else f"_{ONEQ_SET}"
OUT_DATA = _bootstrap.RESULTS_DIR + f"/bounding_renyi_data_sem{_TAG}.npz"
OUT_SCALING = _bootstrap.RESULTS_DIR + f"/renyi_scaling_data_sem{_TAG}.npz"
_SB, _SV = StabilizerBackend(), StatevectorBackend()

TOL = 1e-9                       # tolerance for the bound-ordering assertions
COLORS = {"measured": "#0072B2", "qcap": "#D55E00", "exact": "#009E73",
          "renyi": "#CC79A7"}


# ---------------------------------------------------------------------------
# Ideal-distribution helpers
# ---------------------------------------------------------------------------
def dense_probs(dist, n):
    """Sparse ``{bitstring: p}`` (q0 leftmost) -> dense probability vector of length 2^N.

    The backends return only the outcomes above their amplitude cutoff, so every key
    absent from the dict is a genuine ZERO-probability outcome and must be filled in as
    such: both  sum_x |p(x) - 2^-N|  and  sum_x p(x)^2  are sums over ALL 2^N outcomes,
    and taking them over the sparse support instead would silently drop the
    (1 - |support|) * 2^-N contribution to the uniform TVD.
    """
    p = np.zeros(2 ** n, dtype=float)
    for key, pr in dist.items():
        if len(key) != n:
            raise ValueError(f"bitstring {key!r} is not {n} bits wide")
        p[int(key[::-1], 2)] = pr      # q0 is the leftmost character -> bit 0
    tot = float(p.sum())
    if abs(tot - 1.0) > 1e-6:
        raise ValueError(f"ideal distribution sums to {tot:.9f}, not 1")
    return p / tot                     # absorb the sub-ppm cutoff loss


def ideal_factors(p, n):
    """The ideal-distribution-derived factors, for one circuit instance.

    Returns ``uniform_tvd`` = D_TV(p, u), the order-2 Renyi divergence D_2(p||u), and
    the square-root factor  S_2 = (1/2) sqrt(e^{D_2} - 1)  both unclipped and clipped
    at one. ``e^{D_2} - 1`` is clamped at zero before the root: for a uniform p it is
    algebraically zero and floating point can put it a few ulps below.
    """
    u = 2.0 ** -n
    uniform_tvd = 0.5 * float(np.abs(p - u).sum())
    exp_d2 = (2 ** n) * float((p ** 2).sum())          # e^{D_2(p||u)} = 2^N sum p^2
    d2 = math.log(exp_d2) if exp_d2 > 0 else -math.inf
    s2 = 0.5 * math.sqrt(max(exp_d2 - 1.0, 0.0))
    # Exact identities derived from the collision probability:
    #   Delta_2 = N - H_2(q) = log2(2^N sum_x q(x)^2) = log2(1 + 4 S_2^2).
    # Delta_2 is the collision-entropy deficit from uniformity.
    delta2 = math.log2(exp_d2) if exp_d2 > 0 else -math.inf
    h2 = n - delta2

    # Model-dependent exact-TVD analogue.  If q were uniform on an effective support
    # of K outcomes, then D_TV(q,u)=1-K/2^N and therefore
    #     H_support = log2 K = N + log2(1-D_TV(q,u)).
    # For a general q this is NOT an entropy identity; it is an interpretable bounded
    # effective-support coordinate used only to fit/project the exact TVD factor.
    one_minus_t = max(1.0 - uniform_tvd, np.finfo(float).tiny)
    support_h = n + math.log2(one_minus_t)
    return {"uniform_tvd": uniform_tvd,
            "renyi2_divergence": d2,
            "renyi_sqrt_factor_unclipped": s2,
            "renyi_sqrt_factor_clipped": min(1.0, s2),
            "collision_entropy_deficit": delta2,
            "collision_entropy": h2,
            "collision_entropy_density": h2 / n,
            "exact_support_entropy": support_h,
            "exact_support_entropy_density": support_h / n}


CHECKED = {"factors": 0, "bounds": 0}     # how many circuits passed each check


def check_factors(f, where=""):
    """D_TV(p,u) <= min(1, S_2) -- the chi-squared/Renyi inequality, per circuit."""
    lhs, rhs = f["uniform_tvd"], f["renyi_sqrt_factor_clipped"]
    if lhs > rhs + 1e-9:
        raise AssertionError(f"{where}: D_TV(p,u)={lhs:.12f} exceeds "
                             f"min(1, S_2)={rhs:.12f}")
    CHECKED["factors"] += 1


def ideal_distribution(circ):
    """Exact ideal distribution on the cheapest backend that can represent the circuit:
    stabilizer for Clifford, statevector otherwise. No noise, no shots."""
    return (_SB if circ.is_clifford else _SV).exact_distribution(circ)


# ---------------------------------------------------------------------------
# Deterministic self-test of the two closed-form cases (section 8)
# ---------------------------------------------------------------------------
def self_test(verbose=True):
    """Uniform and deterministic ideal distributions, where every factor is known.

        uniform p:        D_TV(p,u) = 0,          S_2 = 0
        deterministic p:  D_TV(p,u) = 1 - 2^-N,   S_2 = (1/2) sqrt(2^N - 1)

    The deterministic S_2 exceeds one for N >= 3 -- that is expected, and exactly why
    the bound uses the CLIPPED factor. Also checks that dense_probs() fills the
    outcomes missing from a sparse dict with zeros: the deterministic case is a
    one-entry dict whose D_TV is 1 - 2^-N only if the other 2^N - 1 outcomes count.
    """
    lines = []
    for n in (1, 2, 3, 5, 8):
        d = 2 ** n
        uni = {"".join(str((i >> j) & 1) for j in range(n)): 1.0 / d for i in range(d)}
        f = ideal_factors(dense_probs(uni, n), n)
        assert abs(f["uniform_tvd"]) < 1e-12, f
        assert abs(f["renyi_sqrt_factor_unclipped"]) < 1e-12, f
        assert abs(f["renyi2_divergence"]) < 1e-12, f
        check_factors(f, f"uniform n={n}")
        # the bounds themselves vanish for any eps
        for eps in (0.0, 0.37, 1.0):
            assert abs(eps * f["uniform_tvd"]) < 1e-12
            assert abs(eps * f["renyi_sqrt_factor_clipped"]) < 1e-12

        det = {"0" * n: 1.0}                       # sparse: 1 of 2^N keys present
        g = ideal_factors(dense_probs(det, n), n)
        want_tvd, want_s2 = 1.0 - 2.0 ** -n, 0.5 * math.sqrt(d - 1.0)
        assert abs(g["uniform_tvd"] - want_tvd) < 1e-12, (g, want_tvd)
        assert abs(g["renyi_sqrt_factor_unclipped"] - want_s2) < 1e-12, (g, want_s2)
        assert abs(g["renyi2_divergence"] - math.log(d)) < 1e-12, g
        assert abs(g["renyi_sqrt_factor_clipped"] - min(1.0, want_s2)) < 1e-12, g
        check_factors(g, f"deterministic n={n}")
        lines.append(f"    n={n}: uniform -> D_TV=0, S_2=0   |   deterministic -> "
                     f"D_TV={g['uniform_tvd']:.6f} (=1-2^-N), "
                     f"S_2={g['renyi_sqrt_factor_unclipped']:.6f} "
                     f"(=0.5*sqrt(2^N-1)), clipped={g['renyi_sqrt_factor_clipped']:.6f}")
    if verbose:
        print("self-test: closed-form uniform / deterministic distributions")
        print("\n".join(lines))
        print("    all assertions passed\n")
    return True


# ---------------------------------------------------------------------------
# Main experiment: measured TVD + the three bounds, per depth, per instance
# ---------------------------------------------------------------------------
def test_circuit(depth, mode, seed, n=N, cycles=None):
    """depth reps of [1q layer, cycle, ...] over the non-empty cycles + a final 1q
    layer (non-mirror). Each cycle appears `depth` times."""
    return bench_brickwork(n, depth, CYCLE_LIST if cycles is None else cycles,
                           oneq=_ONEQ[mode], twoq="cz", seed=seed, final_oneq=True)


def instance_result(circ, seed, eps_qcap):
    """Measured RC TVD + every ideal-distribution quantity, for ONE circuit instance.

    The ideal distribution is computed once here and reused for both the measured TVD
    and the bound factors (the original ``noisy_tvds`` computed it for the TVD only).

    `seed` must vary per instance, otherwise every point shares one shot-noise
    realisation and the scatter understates the true spread.

    The path is chosen per CIRCUIT, not per ansatz: only the random ansatz carries T,
    and even there a shallow instance can draw no T at all, in which case it is
    Clifford and takes the exact stim path.
    """
    n = circ.n_qubits
    clifford = circ.is_clifford
    ideal = (_SB if clifford else _SV).exact_distribution(circ)
    out = ideal_factors(dense_probs(ideal, n), n)
    check_factors(out, f"{circ.name} seed={seed}")

    rc = simulate(circ, "stabilizer" if clifford else "statevector", "distribution",
                  noise=TWIRLED, shots=TVD_SHOTS, n_traj=N_TRAJ, seed=seed)
    out["measured_tvd"] = total_variation_distance(ideal, rc)
    out["exact_white_noise_bound"] = eps_qcap * out["uniform_tvd"]
    out["renyi_bound"] = eps_qcap * out["renyi_sqrt_factor_clipped"]

    # section 8: B_exact <= B_Renyi <= raw QCAP, and all three live in [0, 1].
    assert out["exact_white_noise_bound"] <= out["renyi_bound"] + TOL, out
    assert out["renyi_bound"] <= eps_qcap + TOL, out
    for k in ("exact_white_noise_bound", "renyi_bound"):
        assert -TOL <= out[k] <= 1.0 + TOL, (k, out[k])
    assert -TOL <= eps_qcap <= 1.0 + TOL, eps_qcap
    CHECKED["bounds"] += 1
    return out


_PER_INSTANCE = ("measured_tvd", "exact_white_noise_bound", "renyi_bound",
                 "uniform_tvd", "renyi_sqrt_factor_unclipped",
                 "renyi_sqrt_factor_clipped", "renyi2_divergence",
                 "collision_entropy_deficit", "collision_entropy",
                 "collision_entropy_density", "exact_support_entropy",
                 "exact_support_entropy_density")


def run(mode):
    """Cycle-benchmark the ansatz, then sweep depth. Returns arrays keyed by name;
    per-instance arrays have shape (len(DEPTHS), N_INSTANCES)."""
    print(f"\n=== {mode} ansatz ===")
    # cycle_benchmark runs in stim and so measures the TWIRLED channel -- the bound it
    # feeds is therefore a bound on the randomly-compiled circuit.
    cbs = {k: cycle_benchmark(
               pairs, N, CB_DEPTHS, NOISE, mode=mode,
               n_decays=CB_DECAYS, shots=CB_SHOTS, seed=1 + j)
           for j, (k, pairs) in enumerate(CYCLES.items())}
    ro_fid, ro_std = readout_fidelity(N, NOISE, seed=3)
    for k, cb in cbs.items():
        print(f"  cycle {k} {str(CYCLES[k]):<16} e_F = {cb['e_F']:.4f}")
    print(f"  readout fidelity:     {ro_fid:.4f} ({ro_std:.4f})")

    efs = {k: (cb["e_F"], cb["e_F_std"]) for k, cb in cbs.items()}
    res = {k: [] for k in _PER_INSTANCE}
    bounds, bstd = [], []
    print(f"  {'depth':>6}{'QCAP':>9}{'measured':>11}{'exact-WN':>11}{'Renyi-2':>10}"
          f"{'<TVD(p,u)>':>13}{'<sqrt fac>':>12}{'clipped':>9}")

    for d in DEPTHS:
        b = qcap_bound({k: d for k in CYCLES}, efs, ro_fid, ro_std)
        eps = b["error"]
        bounds.append(eps)
        bstd.append(b["std"])
        rows = []
        for i in range(N_INSTANCES):
            s = SEED + 100 * i     # stride leaves room for an independent shot seed
            rows.append(instance_result(test_circuit(d, mode, s), s + 1, eps))
        for k in _PER_INSTANCE:
            res[k].append([r[k] for r in rows])
        clipped = float(np.mean([r["renyi_sqrt_factor_unclipped"] > 1.0 for r in rows]))
        res.setdefault("clipped_fraction", []).append(clipped)
        m = {k: np.mean(res[k][-1]) for k in _PER_INSTANCE}
        print(f"  {d:>6}{eps:>9.4f}{m['measured_tvd']:>11.4f}"
              f"{m['exact_white_noise_bound']:>11.4f}{m['renyi_bound']:>10.4f}"
              f"{m['uniform_tvd']:>13.4f}"
              f"{m['renyi_sqrt_factor_unclipped']:>12.4f}{clipped:>9.2f}", flush=True)

    out = {k: np.array(v) for k, v in res.items()}
    out["bound"], out["bound_std"] = np.array(bounds), np.array(bstd)
    return out


def band(arr, axis=1):
    """(lo, hi) uncertainty band across circuit instances, per BAND."""
    if BAND == "std":
        m, s = arr.mean(axis=axis), arr.std(axis=axis)
        return m - s, m + s
    return np.percentile(arr, 16, axis=axis), np.percentile(arr, 84, axis=axis)


BAND_LABEL = ("mean $\\pm$ 1 s.d. across circuit instances" if BAND == "std"
              else "16th-84th pct across circuit instances")


# ---------------------------------------------------------------------------
# Full depth-by-N sweep of measured TVD, QCAP, exact-WN, Renyi, and factors
# ---------------------------------------------------------------------------
def scaling_seed(n, depth, mode, i):
    """A distinct deterministic seed for every (N, depth, ansatz, instance)."""
    return (SCALING_SEED + 1000000 * n + 10000 * depth
            + 1000 * MODES.index(mode) + i)


def dense_state_gb(n):
    """Estimated peak memory for a dense 2^N readout, in GiB.

    Both backends go through a dense statevector (stim's ``state_vector()`` as much as
    qiskit's ``Statevector``): 2^N complex128 for the state, plus a float64 probability
    array and a copy inside the backend, so ~4x the raw amplitude array.
    """
    return (2 ** n) * 16 * 4 / 2 ** 30


def scaling_study():
    """Run the full TVD/QCAP/exact/Renyi experiment over both depth and N.

    Output arrays have shape (n_depths, n_qubits, n_instances), except QCAP arrays,
    which have shape (n_depths, n_qubits). Every plotted mean and error bar is taken
    across the instance axis at fixed (depth, N).
    """
    print(f"\n=== full depth-by-N scaling study "
          f"({SCALING_INSTANCES} instances per (depth,N,ansatz)) ===")
    ns, skipped = [], []
    for n in N_SCALING:
        gb = dense_state_gb(n)
        if n > _SV.max_exact_qubits or gb > SCALING_MEM_BUDGET_GB:
            skipped.append((n, gb))
            print(f"  WARNING: skipping N={n} -- dense readout ~{gb:.2f} GiB")
            continue
        ns.append(n)
    ns = np.asarray(ns, dtype=int)
    depths = np.asarray(SCALING_DEPTHS, dtype=int)

    keys = ("measured_tvd", "exact_white_noise_bound", "renyi_bound",
            "uniform_tvd", "renyi_sqrt_factor_unclipped",
            "renyi_sqrt_factor_clipped", "renyi2_divergence",
            "collision_entropy_deficit", "collision_entropy",
            "collision_entropy_density", "exact_support_entropy",
            "exact_support_entropy_density")
    out = {m: {k: np.empty((len(depths), len(ns), SCALING_INSTANCES), float)
               for k in keys} for m in MODES}
    for m in MODES:
        out[m]["bound"] = np.empty((len(depths), len(ns)), float)
        out[m]["bound_std"] = np.empty((len(depths), len(ns)), float)

    for ni, n in enumerate(ns):
        cycles_dict = {k: v for k, v in (("A", even_pairs(int(n))),
                                          ("B", odd_pairs(int(n)))) if v}
        cycles = list(cycles_dict.values())
        for mode in MODES:
            cbs = {k: cycle_benchmark(
                       pairs, int(n), CB_DEPTHS, NOISE, mode=mode,
                       n_decays=CB_DECAYS, shots=CB_SHOTS,
                       seed=SCALING_SEED + 1000 * int(n) + j)
                   for j, (k, pairs) in enumerate(cycles_dict.items())}
            ro_fid, ro_std = readout_fidelity(
                int(n), NOISE, seed=SCALING_SEED + 2000 * int(n) + MODES.index(mode))
            efs = {k: (cb["e_F"], cb["e_F_std"]) for k, cb in cbs.items()}

            for di, depth in enumerate(depths):
                b = qcap_bound({k: int(depth) for k in cycles_dict},
                               efs, ro_fid, ro_std)
                eps = b["error"]
                out[mode]["bound"][di, ni] = eps
                out[mode]["bound_std"][di, ni] = b["std"]
                for i in range(SCALING_INSTANCES):
                    seed = scaling_seed(int(n), int(depth), mode, i)
                    circ = test_circuit(int(depth), mode, seed, n=int(n), cycles=cycles)
                    row = instance_result(circ, seed + 1, eps)
                    for key in keys:
                        out[mode][key][di, ni, i] = row[key]

                print(f"  d={depth:>3} N={n:>2} {mode:>10}: "
                      f"TVD={out[mode]['measured_tvd'][di,ni].mean():.4f}, "
                      f"exact={out[mode]['exact_white_noise_bound'][di,ni].mean():.4f}, "
                      f"Renyi={out[mode]['renyi_bound'][di,ni].mean():.4f}, "
                      f"S2={out[mode]['renyi_sqrt_factor_unclipped'][di,ni].mean():.4f}",
                      flush=True)
    return depths, ns, out, skipped


# ---------------------------------------------------------------------------
# Fits of the ENSEMBLE MEAN of the UNCLIPPED square-root factor (section 5)
# ---------------------------------------------------------------------------
def _r2(y, f):
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((y - f) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_scaling(ns, means, sem=None):
    """Fit the ensemble-mean unclipped S2 versus N.

    Models:
        S2(N) = A exp(b N)
        S2(N) = A 2^(alpha N)
        S2(N) = C 2^(N/2)

    The fit is performed in log space.  When ``sem`` is supplied, the uncertainty of
    the ensemble mean is propagated as sigma_log ~= sem / mean and used as the fit
    weight.  The plotted error bars may still show the instance standard deviation;
    SEM is the correct uncertainty for fitting the ensemble mean.
    """
    ns, means = np.asarray(ns, float), np.asarray(means, float)
    if sem is None:
        sem = np.ones_like(means)
        weighted = False
    else:
        sem = np.asarray(sem, float)
        weighted = True
    ok = np.isfinite(means) & (means > 0) & np.isfinite(sem) & (sem >= 0)
    used, y, sy = ns[ok], means[ok], sem[ok]
    fits = {"n_used": int(ok.sum()), "n_total": int(means.size),
            "dropped_N": ns[~ok].astype(int).tolist(), "N_used": used.astype(int),
            "weighted_by_sem": weighted}
    if ok.sum() < 3:
        fits["ok"] = False
        return fits
    fits["ok"] = True
    ly = np.log(y)
    # Avoid zero sigma for an accidentally identical finite ensemble.
    sigma_log = np.maximum(sy / y, 1e-12) if weighted else None

    def _log_fit(model, p0):
        popt, pcov = curve_fit(model, used, ly, p0=p0,
                               sigma=sigma_log, absolute_sigma=weighted,
                               maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        pred = np.exp(model(used, *popt))
        return popt, perr, pcov, _r2(y, pred), _r2(ly, model(used, *popt))

    slope, icept = np.polyfit(used, ly, 1)
    p, e, cov, r2, r2l = _log_fit(lambda x, la, b: la + b * x, [icept, slope])
    fits["exp"] = {"A": math.exp(p[0]), "A_err": math.exp(p[0]) * e[0], "b": p[1],
                   "b_err": e[1], "r2": r2, "r2_log": r2l, "cov": cov}
    p, e, cov, r2, r2l = _log_fit(
        lambda x, la, al: la + al * math.log(2.0) * x,
        [icept, slope / math.log(2.0)])
    fits["pow2"] = {"A": math.exp(p[0]), "A_err": math.exp(p[0]) * e[0],
                    "alpha": p[1], "alpha_err": e[1], "r2": r2,
                    "r2_log": r2l, "cov": cov}
    p, e, cov, r2, r2l = _log_fit(
        lambda x, lc: lc + 0.5 * math.log(2.0) * x, [icept])
    fits["sqrt_dim"] = {"C": math.exp(p[0]), "C_err": math.exp(p[0]) * e[0],
                        "r2": r2, "r2_log": r2l, "cov": cov}
    return fits



def haar_renyi_factor(n):
    """Finite-N Haar/Porter-Thomas prediction for the unclipped Renyi factor.

    For a Haar-random pure state in dimension D=2^N,

        E[sum_x p(x)^2] = 2/(D+1),

    so the corresponding collision-based factor is

        S2_Haar(N) = 0.5 * sqrt((D-1)/(D+1)),

    which approaches 1/2 as N grows.
    """
    n = np.asarray(n, dtype=float)
    d = np.exp2(n)
    return 0.5 * np.sqrt(np.maximum((d - 1.0) / (d + 1.0), 0.0))


def fit_haar_relaxation(ns, means, sem=None):
    """Fit approach to the finite-N Haar prediction.

    The fitted crossover model is

        S2(N) = S2_Haar(N) + A exp(-b N).

    A may be positive or negative, allowing approach to the Haar curve from
    above or below.  The fixed Haar prediction itself is also scored with R^2.
    """
    ns = np.asarray(ns, dtype=float)
    means = np.asarray(means, dtype=float)
    weighted = sem is not None
    sem = np.ones_like(means) if sem is None else np.asarray(sem, dtype=float)

    ok = (np.isfinite(ns) & np.isfinite(means) & np.isfinite(sem)
          & (sem >= 0.0))
    x, y, sy = ns[ok], means[ok], sem[ok]
    out = {
        "ok": bool(ok.sum() >= 3),
        "n_used": int(ok.sum()),
        "n_total": int(means.size),
        "N_used": x.astype(int),
        "dropped_N": ns[~ok].astype(int).tolist(),
        "weighted_by_sem": weighted,
    }
    if not out["ok"]:
        return out

    haar = haar_renyi_factor(x)
    out["fixed_haar_r2"] = _r2(y, haar)
    out["fixed_haar_rmse"] = float(np.sqrt(np.mean((y - haar) ** 2)))

    sigma = np.maximum(sy, 1e-12) if weighted else None

    def model(z, A, b):
        return haar_renyi_factor(z) + A * np.exp(-b * z)

    A0 = float(y[0] - haar[0])
    try:
        par, cov = curve_fit(
            model,
            x,
            y,
            p0=[A0, 0.2],
            sigma=sigma,
            absolute_sigma=weighted,
            bounds=([-10.0, 0.0], [10.0, 10.0]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        out["ok"] = False
        return out

    err = np.sqrt(np.diag(cov))
    pred = model(x, *par)
    out.update({
        "A": float(par[0]),
        "A_err": float(err[0]),
        "b": float(par[1]),
        "b_err": float(err[1]),
        "r2": _r2(y, pred),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "cov": cov,
    })
    return out

def fit_exact_scaling(ns, means, sem=None):
    """Fit the mean exact factor D_TV(p,u) versus N.

    The primary model has a free asymptote,

        T(N) = L - A exp(-b N),

    because the data need not approach one.  The constrained L=1 model is retained as
    a diagnostic.  When supplied, ``sem`` weights the fit by uncertainty in the mean.
    """
    ns, means = np.asarray(ns, float), np.asarray(means, float)
    if sem is None:
        sem = np.ones_like(means)
        weighted = False
    else:
        sem = np.asarray(sem, float)
        weighted = True
    ok = (np.isfinite(means) & (means >= 0.0) & (means <= 1.0)
          & np.isfinite(sem) & (sem >= 0.0))
    used, y, sy = ns[ok], means[ok], sem[ok]
    fits = {"n_used": int(ok.sum()), "n_total": int(means.size),
            "dropped_N": ns[~ok].astype(int).tolist(), "N_used": used.astype(int),
            "weighted_by_sem": weighted}
    if ok.sum() < 3:
        fits["ok"] = False
        return fits
    fits["ok"] = True
    sigma = np.maximum(sy, 1e-12) if weighted else None

    def sat1(x, A, b):
        return 1.0 - A * np.exp(-b * x)

    def satL(x, L, A, b):
        return L - A * np.exp(-b * x)

    A0 = max(1.0 - y[0], 1e-3)
    par, cov = curve_fit(sat1, used, y, p0=[A0, 0.3], sigma=sigma,
                         absolute_sigma=weighted,
                         bounds=([0.0, 0.0], [2.0, 10.0]), maxfev=20000)
    err = np.sqrt(np.diag(cov))
    fits["sat1"] = {"A": par[0], "A_err": err[0], "b": par[1],
                    "b_err": err[1], "r2": _r2(y, sat1(used, *par)), "cov": cov}

    L0 = min(1.0, max(float(y.max()), 0.5))
    par, cov = curve_fit(satL, used, y,
                         p0=[L0, max(L0 - y[0], 1e-3), 0.3], sigma=sigma,
                         absolute_sigma=weighted,
                         bounds=([0.0, 0.0, 0.0], [1.0, 2.0, 10.0]),
                         maxfev=20000)
    err = np.sqrt(np.diag(cov))
    fits["satL"] = {"L": par[0], "L_err": err[0], "A": par[1],
                    "A_err": err[1], "b": par[2], "b_err": err[2],
                    "r2": _r2(y, satL(used, *par)), "cov": cov}
    return fits



def fit_collision_entropy_deficit(ns, means, sem=None):
    """Fit the collision-entropy deficit Delta_2=N-H_2(q) to aN+c.

    This is the preferred Rényi extrapolation because the transformation

        Delta_2 = log2(1 + 4 S_2^2)

    is an exact identity for every ideal distribution.  The fitted model implies

        H_2(q) = (1-a)N-c,
        S_2(N) = 1/2 sqrt(2^(aN+c)-1).

    ``a`` is the extensive collision-entropy-deficit density and ``1-a`` is the
    projected collision-entropy density.
    """
    ns, means = np.asarray(ns, float), np.asarray(means, float)
    weighted = sem is not None
    sem = np.ones_like(means) if sem is None else np.asarray(sem, float)
    ok = np.isfinite(means) & np.isfinite(sem) & (sem >= 0)
    x, y, sy = ns[ok], means[ok], sem[ok]
    fit = {"ok": bool(ok.sum() >= 3), "N_used": x.astype(int),
           "dropped_N": ns[~ok].astype(int).tolist(), "weighted_by_sem": weighted}
    if not fit["ok"]:
        return fit
    sigma = np.maximum(sy, 1e-12) if weighted else None
    par, cov = curve_fit(lambda z, a, c: a*z + c, x, y,
                         sigma=sigma, absolute_sigma=weighted, maxfev=20000)
    err = np.sqrt(np.diag(cov))
    pred = par[0]*x + par[1]
    fit.update({"a": par[0], "a_err": err[0], "c": par[1], "c_err": err[1],
                "h2_density": 1.0-par[0], "h2_density_err": err[0],
                "r2": _r2(y, pred), "cov": cov})
    return fit


def fit_exact_logit_scaling(ns, means, sem=None):
    """Fit the exact factor T=D_TV(q,u) with a bounded base-2 logistic model.

        log2(T/(1-T)) = kappa N + c,
        T(N) = 1/(1 + 2^(-(kappa N+c))).

    Unlike the collision-entropy transformation, this is an empirical model: TVD does
    not uniquely determine the full distribution or an entropy.  It is used only to
    extrapolate the exact white-noise multiplier while preserving 0<T<1.
    """
    ns, means = np.asarray(ns, float), np.asarray(means, float)
    weighted = sem is not None
    sem = np.ones_like(means) if sem is None else np.asarray(sem, float)
    ok = (np.isfinite(means) & np.isfinite(sem) & (sem >= 0)
          & (means > 0.0) & (means < 1.0))
    x, y, sy = ns[ok], means[ok], sem[ok]
    fit = {"ok": bool(ok.sum() >= 3), "N_used": x.astype(int),
           "dropped_N": ns[~ok].astype(int).tolist(), "weighted_by_sem": weighted}
    if not fit["ok"]:
        return fit

    def logistic2(z, kappa, c):
        w = np.clip(kappa*z + c, -80.0, 80.0)
        return 1.0/(1.0 + 2.0**(-w))

    sigma = np.maximum(sy, 1e-12) if weighted else None
    logit = np.log2(y/(1.0-y))
    k0, c0 = np.polyfit(x, logit, 1)
    par, cov = curve_fit(logistic2, x, y, p0=[k0, c0], sigma=sigma,
                         absolute_sigma=weighted, maxfev=20000)
    err = np.sqrt(np.diag(cov))
    pred = logistic2(x, *par)
    fit.update({"kappa": par[0], "kappa_err": err[0],
                "c": par[1], "c_err": err[1],
                "r2": _r2(y, pred), "cov": cov})
    return fit


def fit_extensive_entropy(ns, means, sem=None, *, name="entropy"):
    """Fit an extensive entropy-like quantity to H(N)=h*N+c.

    For ``collision_entropy`` this is a direct fit of the true order-2 entropy.
    For ``exact_support_entropy`` it is a model-dependent effective-support fit.
    The fit is weighted by the SEM of the ensemble mean when supplied.
    """
    ns = np.asarray(ns, float)
    means = np.asarray(means, float)
    weighted = sem is not None
    sem = np.ones_like(means) if sem is None else np.asarray(sem, float)
    ok = np.isfinite(ns) & np.isfinite(means) & np.isfinite(sem) & (sem >= 0.0)
    x, y, sy = ns[ok], means[ok], sem[ok]
    fit = {"ok": bool(ok.sum() >= 3), "name": name,
           "N_used": x.astype(int), "dropped_N": ns[~ok].astype(int).tolist(),
           "weighted_by_sem": weighted}
    if not fit["ok"]:
        return fit
    sigma = np.maximum(sy, 1e-12) if weighted else None
    par, cov = curve_fit(lambda z, h, c: h*z + c, x, y,
                         sigma=sigma, absolute_sigma=weighted, maxfev=20000)
    err = np.sqrt(np.diag(cov))
    pred = par[0]*x + par[1]
    fit.update({"h": par[0], "h_err": err[0], "c": par[1], "c_err": err[1],
                "r2": _r2(y, pred), "cov": cov})
    return fit


def reconstruct_s2_from_h2(n, h, c):
    """Reconstruct S2 from H2(N)=h*N+c using the exact identity."""
    n = np.asarray(n, float)
    delta = np.clip(n - (h*n + c), 0.0, 1024.0)
    return 0.5*np.sqrt(np.maximum(np.exp2(delta) - 1.0, 0.0))


def reconstruct_exact_from_support_entropy(n, h, c):
    """Reconstruct exact TVD under the uniform-effective-support model."""
    n = np.asarray(n, float)
    ratio = np.exp2(np.clip((h*n + c) - n, -1024.0, 0.0))
    return np.clip(1.0 - ratio, 0.0, 1.0)


def print_fits(all_fits):
    print("\n=== fits versus N at each fixed depth ===")
    for depth, by_mode in all_fits.items():
        print(f"depth {depth}:")
        for mode in MODES:
            fs2 = by_mode[mode]["s2"]
            fex = by_mode[mode]["exact"]
            if fs2["ok"]:
                print(f"  {mode:>10} S2: alpha={fs2['pow2']['alpha']:.4g} "
                      f"+/- {fs2['pow2']['alpha_err']:.2g}, "
                      f"R2={fs2['pow2']['r2']:.4f}")
            fhaar = by_mode[mode].get("haar")
            if fhaar and fhaar["ok"]:
                print(f"  {mode:>10} Haar crossover: A={fhaar['A']:.4g} "
                      f"+/- {fhaar['A_err']:.2g}, b={fhaar['b']:.4g} "
                      f"+/- {fhaar['b_err']:.2g}, R2={fhaar['r2']:.4f}; "
                      f"fixed-Haar R2={fhaar['fixed_haar_r2']:.4f}")
            if fex["ok"]:
                print(f"  {mode:>10} exact factor: L={fex['satL']['L']:.4g} "
                      f"+/- {fex['satL']['L_err']:.2g}, "
                      f"b={fex['satL']['b']:.4g} +/- {fex['satL']['b_err']:.2g}, "
                      f"R2={fex['satL']['r2']:.4f}")
            fent = by_mode[mode].get("entropy")
            flog = by_mode[mode].get("exact_logit")
            if fent and fent["ok"]:
                print(f"  {mode:>10} entropy: Delta2={fent['a']:.4g} N "
                      f"+ {fent['c']:.4g}; H2/N -> {fent['h2_density']:.4g}; "
                      f"R2={fent['r2']:.4f}")
            if flog and flog["ok"]:
                print(f"  {mode:>10} exact-logit: kappa={flog['kappa']:.4g} "
                      f"+/- {flog['kappa_err']:.2g}, R2={flog['r2']:.4f}")
            fh2 = by_mode[mode].get("h2_linear")
            fhs = by_mode[mode].get("exact_support")
            if fh2 and fh2["ok"]:
                print(f"  {mode:>10} H2: H2={fh2['h']:.4g} N + {fh2['c']:.4g}; "
                      f"R2={fh2['r2']:.4f}")
            if fhs and fhs["ok"]:
                print(f"  {mode:>10} exact-support: Heff={fhs['h']:.4g} N "
                      f"+ {fhs['c']:.4g}; R2={fhs['r2']:.4f}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_bounds(data):
    """The main two-panel figure: measured TVD scatter + the three bounds."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    x = np.array(DEPTHS, dtype=float)
    # everything is a TVD, so [0, 1] is the meaningful range -- but leaving the axis at
    # the full [0, 1] when nothing reaches 0.6 wastes half the panel, so cap it just
    # above the largest plotted value (bounds are still CLIPPED into [0, 1] below).
    top = min(1.0, 1.12 * max(
        max(float(np.max(data[m][k])) for k in ("measured_tvd", "bound",
                                                "exact_white_noise_bound",
                                                "renyi_bound"))
        for m in MODES))
    for ax, mode in zip(axes, MODES):
        d = data[mode]
        qcap = np.clip(d["bound"], 0.0, 1.0)
        for j, dep in enumerate(DEPTHS):
            pts = d["measured_tvd"][j]
            ax.plot([dep] * len(pts), pts, "o", color=COLORS["measured"], ms=4,
                    alpha=0.45, zorder=2,
                    label="Measured TVD, randomly compiled" if j == 0 else None)
        # raw QCAP: one number per depth, common to every instance. Its error bars are
        # the CB/readout PARAMETER uncertainty -- a different thing from the
        # instance-to-instance spread of the two bounds below, so it is drawn as caps
        # rather than as a band, to keep the two visually separate.
        ax.errorbar(x, qcap, yerr=np.clip(d["bound_std"], 0, 1), fmt="-",
                    color=COLORS["qcap"], lw=2, capsize=3, elinewidth=1, zorder=5,
                    label="Raw QCAP bound ($\\pm$1$\\sigma$ CB parameter unc.)")
        for key, col, lab in (
                ("exact_white_noise_bound", "exact", "Exact under white-noise model"),
                ("renyi_bound", "renyi", "Rényi-2 upper bound")):
            arr = np.clip(d[key], 0.0, 1.0)
            lo, hi = band(arr)
            ax.plot(x, arr.mean(axis=1), "-", color=COLORS[col], lw=2, zorder=4,
                    label=f"{lab} (mean over instances)")
            ax.fill_between(x, np.clip(lo, 0, 1), np.clip(hi, 0, 1),
                            color=COLORS[col], alpha=0.22, lw=0, zorder=3,
                            label=f"{lab}: {BAND_LABEL}")
            ax.plot(x, arr.max(axis=1), "--", color=COLORS[col], lw=1.7, zorder=5,
                    label=f"{lab} (maximum over instances)")
        ax.set_xlabel("circuit depth")
        ax.set_xlim(0, max(DEPTHS) * 1.03)          # linear depth axis
        ax.set_ylim(0, top)
        ax.set_title(f"{mode} ansatz", fontsize=11)
        ax.grid(True, alpha=0.15)
        ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    axes[0].set_ylabel("total-variation distance from the ideal distribution")
    ansatz = "Clifford 1q" if CLIFFORD_ANSATZ else "Clifford+T 1q on random only"
    fig.suptitle("Bounding circuit error from cycle benchmarking: raw QCAP vs "
                 "white-noise-exact vs Rényi-2\n"
                 f"(n={N}, {ansatz}, {N_INSTANCES} instances/depth, synthetic CB + "
                 rf"noise model, coherent $\theta_{{zz}}={THETA_ZZ}$; "
                 '"exact" holds only under the global white-noise mixture)',
                 fontsize=11.5)
    fig.tight_layout()
    out = _bootstrap.RESULTS_DIR + f"/bounding_renyi_{N}qubits{_TAG}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _mean_sd(arr):
    return arr.mean(axis=-1), arr.std(axis=-1, ddof=1)


def _mean_sd_sem(arr):
    mean = arr.mean(axis=-1)
    sd = arr.std(axis=-1, ddof=1)
    sem = sd / math.sqrt(arr.shape[-1])
    return mean, sd, sem


def figure_scaling_summary(depths, ns, sc):
    """Combined summary: one curve per depth, with mean +/- instance s.d."""
    outputs = []
    metrics = (("measured_tvd", "Measured TVD", COLORS["measured"]),
               ("exact_white_noise_bound", "Exact-WN prediction", COLORS["exact"]),
               ("renyi_bound", "Renyi-2 bound", COLORS["renyi"]))
    for mode in MODES:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
        for ax, (key, title, color) in zip(axes, metrics):
            mean, sd = _mean_sd(sc[mode][key])
            for di, depth in enumerate(depths):
                ax.errorbar(ns, mean[di], yerr=sd[di], marker="o", capsize=2,
                            lw=1.5, label=f"depth {depth}")
            ax.set_title(title)
            ax.set_xlabel("number of qubits $N$")
            ax.set_xticks(list(ns))
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.2)
        axes[0].set_ylabel("mean across circuit instances")
        axes[-1].legend(frameon=False, fontsize=8, ncol=2)
        fig.suptitle(f"{mode} ansatz: full TVD/bound scaling versus N by depth\n"
                     f"error bars = 1 instance standard deviation")
        fig.tight_layout()
        path = (_bootstrap.RESULTS_DIR +
                f"/tvd_exact_renyi_vs_N_all_depths_{mode}{_TAG}.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)
    return outputs


def figure_each_depth(depths, ns, sc, all_fits):
    """Write one four-panel N-sweep figure per depth using SEM error bars."""
    outputs = []
    for di, depth in enumerate(depths):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
        for col, mode in enumerate(MODES):
            ax = axes[0, col]
            for key, label, marker, color in (
                    ("measured_tvd", "Measured TVD", "o", COLORS["measured"]),
                    ("exact_white_noise_bound", "Exact-WN prediction", "^", COLORS["exact"]),
                    ("renyi_bound", "Renyi-2 bound", "s", COLORS["renyi"])):
                mean, sd, sem = _mean_sd_sem(sc[mode][key][di])
                ax.errorbar(ns, mean, yerr=sem, fmt=marker + "-", color=color,
                            capsize=3, lw=1.8, ms=5,
                            label=label + r" ensemble mean $\pm$ SEM")
            ax.plot(ns, sc[mode]["bound"][di], "--", color=COLORS["qcap"],
                    lw=1.7, label="Raw QCAP")
            ax.set_title(f"{mode} ansatz: TVD and bounds")
            ax.set_ylabel("TVD / bound")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.2)
            ax.legend(frameon=False, fontsize=8)

            ax = axes[1, col]
            s2m, s2sd, s2sem = _mean_sd_sem(sc[mode]["renyi_sqrt_factor_unclipped"][di])
            exm, exsd, exsem = _mean_sd_sem(sc[mode]["uniform_tvd"][di])
            ax.errorbar(ns, s2m, yerr=s2sem, fmt="o", color=COLORS["renyi"],
                        capsize=3, label=r"$S_2$ ensemble mean $\pm$ SEM")
            ax.errorbar(ns, exm, yerr=exsem, fmt="^", color=COLORS["exact"],
                        capsize=3, label=r"$D_{TV}(p,u)$ ensemble mean $\pm$ SEM")
            g = np.linspace(ns.min(), ns.max(), 300)
            fs2 = all_fits[int(depth)][mode]["s2"]
            if fs2["ok"]:
                f = fs2["pow2"]
                ax.plot(g, f["A"] * 2.0 ** (f["alpha"] * g), "--",
                        label=fr"$A2^{{\alpha N}}$, $\alpha={f['alpha']:.3f}$")
            fhaar = all_fits[int(depth)][mode].get("haar")
            ax.plot(g, haar_renyi_factor(g), "-.", color="0.35", lw=1.6,
                    label=r"finite-$N$ Haar prediction $S_2^{\rm Haar}$")
            if fhaar and fhaar["ok"]:
                ax.plot(g, haar_renyi_factor(g)
                        + fhaar["A"] * np.exp(-fhaar["b"] * g),
                        linestyle=(0, (5, 2, 1, 2)), lw=1.8,
                        label=fr"Haar crossover: $R^2={fhaar['r2']:.3f}$")
            fex = all_fits[int(depth)][mode]["exact"]
            if fex["ok"]:
                f = fex["satL"]
                ax.plot(g, f["L"] - f["A"] * np.exp(-f["b"] * g), ":",
                        label=fr"$L-Ae^{{-bN}}$, $L={f['L']:.2f}$, $b={f['b']:.3f}$")
            ax.set_title(f"{mode} ansatz: factors and fits")
            ax.set_xlabel("number of qubits $N$")
            ax.set_ylabel("Renyi-2 factor")
            ax.set_xticks(list(ns))
            ax.grid(True, alpha=0.2)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(f"Fixed depth {depth}: means and error bars versus N\n"
                     f"{SCALING_INSTANCES} instances per point; error bars = SEM")
        fig.tight_layout()
        path = (_bootstrap.RESULTS_DIR +
                f"/rc_bound_scaling_vs_N_depth{int(depth)}_sem{_TAG}.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)
    return outputs


def figure_fit_exponents(depths, all_fits):
    """Plot fitted ideal-distribution scaling parameters versus circuit depth."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for mode in MODES:
        alpha = np.array([all_fits[int(d)][mode]["s2"]["pow2"]["alpha"]
                          if all_fits[int(d)][mode]["s2"]["ok"] else np.nan
                          for d in depths])
        alpha_err = np.array([all_fits[int(d)][mode]["s2"]["pow2"]["alpha_err"]
                              if all_fits[int(d)][mode]["s2"]["ok"] else np.nan
                              for d in depths])
        beta = np.array([all_fits[int(d)][mode]["exact"]["satL"]["b"]
                         if all_fits[int(d)][mode]["exact"]["ok"] else np.nan
                         for d in depths])
        beta_err = np.array([all_fits[int(d)][mode]["exact"]["satL"]["b_err"]
                             if all_fits[int(d)][mode]["exact"]["ok"] else np.nan
                             for d in depths])
        asym = np.array([all_fits[int(d)][mode]["exact"]["satL"]["L"]
                         if all_fits[int(d)][mode]["exact"]["ok"] else np.nan
                         for d in depths])
        asym_err = np.array([all_fits[int(d)][mode]["exact"]["satL"]["L_err"]
                             if all_fits[int(d)][mode]["exact"]["ok"] else np.nan
                             for d in depths])
        axes[0].errorbar(depths, alpha, yerr=alpha_err, marker="o", capsize=3,
                         label=mode)
        axes[1].errorbar(depths, beta, yerr=beta_err, marker="o", capsize=3,
                         label=mode)
        axes[2].errorbar(depths, asym, yerr=asym_err, marker="o", capsize=3,
                         label=mode)
    axes[0].axhline(0.5, ls=":", color="0.4", label=r"$2^{N/2}$ exponent")
    axes[0].set_ylabel(r"Rényi scaling exponent $\alpha(d)$")
    axes[1].set_ylabel(r"exact-factor rate $b(d)$")
    axes[2].set_ylabel(r"exact-factor asymptote $L(d)$")
    for ax in axes:
        ax.set_xlabel("circuit depth")
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(depths))
        ax.set_xticklabels([str(int(d)) for d in depths])
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle("Renyi-2 fit parameters versus circuit depth")
    fig.tight_layout()
    path = _bootstrap.RESULTS_DIR + f"/renyi_factor_fit_parameters_vs_depth{_TAG}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_renyi_factors_each_depth(depths, ns, sc, all_fits):
    """Clean ideal-distribution figure for each fixed depth.

    Each panel contains exactly the two requested series versus N: the unclipped
    Rényi square-root factor S2 and the exact factor D_TV(p,u), with instance standard
    errors of the ensemble means as visible error bars and SEM-weighted fits.
    """
    outputs = []
    for di, depth in enumerate(depths):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
        for ax, mode in zip(axes, MODES):
            s2m, s2sd, s2sem = _mean_sd_sem(sc[mode]["renyi_sqrt_factor_unclipped"][di])
            exm, exsd, exsem = _mean_sd_sem(sc[mode]["uniform_tvd"][di])
            ax.errorbar(ns, s2m, yerr=s2sem, fmt="o", capsize=3,
                        color=COLORS["renyi"], label=r"$S_2$ ensemble mean $\pm$ SEM")
            #ax.errorbar(ns, exm, yerr=exsem, fmt="^", capsize=3,
            #            color=COLORS["exact"], label=r"$D_{TV}(p,u)$ ensemble mean $\pm$ SEM")
            g = np.linspace(ns.min(), ns.max(), 300)
            fs2 = all_fits[int(depth)][mode]["s2"]
            if fs2["ok"]:
                f = fs2["pow2"]
                ax.plot(g, f["A"] * 2.0 ** (f["alpha"] * g), "--",
                        color=COLORS["renyi"],
                        label=fr"$S_2=A2^{{\alpha N}}$: $\alpha={f['alpha']:.3f}\pm{f['alpha_err']:.3f}$, $R^2={f['r2']:.3f}$")
            fhaar = all_fits[int(depth)][mode].get("haar")
            ax.plot(g, haar_renyi_factor(g), "-.", color="0.35", lw=1.7,
                    label=r"finite-$N$ Haar prediction $S_2^{\rm Haar}$")
            if fhaar and fhaar["ok"]:
                ax.plot(g, haar_renyi_factor(g)
                        + fhaar["A"] * np.exp(-fhaar["b"] * g),
                        linestyle=(0, (5, 2, 1, 2)), color="#7A5195", lw=1.8,
                        label=fr"Haar crossover: $A={fhaar['A']:.2f}$, "
                              fr"$b={fhaar['b']:.3f}$, $R^2={fhaar['r2']:.3f}$; "
                              fr"fixed Haar $R^2={fhaar['fixed_haar_r2']:.3f}$")
            fex = all_fits[int(depth)][mode]["exact"]
            if fex["ok"]:
                f = fex["satL"]
                ax.plot(g, f["L"] - f["A"] * np.exp(-f["b"] * g), ":",
                        color=COLORS["exact"],
                        label=fr"$T=L-Ae^{{-bN}}$: $L={f['L']:.2f}$, $b={f['b']:.3f}$, $R^2={f['r2']:.3f}$")
            ax.set_title(f"{mode} ansatz")
            ax.set_xlabel("number of qubits $N$")
            ax.set_ylabel("Renyi-2 factor")
            ax.set_xticks(list(ns))
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.2)
            ax.legend(frameon=False, fontsize=7.5)
        fig.suptitle(f"Renyi-2 factors versus N at fixed depth {int(depth)}\n"
                     f"error bars: SEM of ensemble mean; fits weighted by SEM")
        fig.tight_layout()
        path = (_bootstrap.RESULTS_DIR
                + f"/renyi_factors_vs_N_depth{int(depth)}_sem{_TAG}.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)
    return outputs


def figure_ideal_factors_all_depths(depths, ns, sc):
    """For each ansatz, show S2 and D_TV(p,u) versus N with one curve per depth."""
    outputs = []
    for mode in MODES:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
        for di, depth in enumerate(depths):
            s2m, s2sd = _mean_sd(sc[mode]["renyi_sqrt_factor_unclipped"][di])
            exm, exsd = _mean_sd(sc[mode]["uniform_tvd"][di])
            axes[0].errorbar(ns, s2m, yerr=s2sd, marker="o", capsize=2,
                             lw=1.4, label=f"depth {int(depth)}")
            axes[1].errorbar(ns, exm, yerr=exsd, marker="^", capsize=2,
                             lw=1.4, label=f"depth {int(depth)}")
        axes[0].set_title(r"Unclipped Rényi square-root factor $S_2$")
        axes[1].set_title(r"Exact factor $D_{TV}(p,u)$")
        for ax in axes:
            ax.set_xlabel("number of qubits $N$")
            ax.set_ylabel("Renyi-2factor")
            ax.set_xticks(list(ns))
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.2)
            ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.suptitle(f"{mode} ansatz: Renyi-2 factors versus N across depths\n"
                     "error bars = 1 instance standard deviation")
        fig.tight_layout()
        path = (_bootstrap.RESULTS_DIR
                + f"/renyi_factors_vs_N_all_depths_{mode}{_TAG}.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)
    return outputs



def figure_entropy_projection_each_depth(depths, ns, sc, all_fits):
    """Projection plots for the identity-derived Rényi quantity and exact TVD factor."""
    outputs = []
    nproj = np.linspace(float(ns.min()), float(PROJECTION_N_MAX), 500)
    for di, depth in enumerate(depths):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
        for col, mode in enumerate(MODES):
            # Collision-entropy deficit: transform each instance first, then average.
            dm, dsd, _ = _mean_sd_sem(sc[mode]["collision_entropy_deficit"][di])
            fent = all_fits[int(depth)][mode]["entropy"]
            ax = axes[0, col]
            ax.errorbar(ns, dm, yerr=dsd, fmt="o", capsize=3,
                        color=COLORS["renyi"], label=r"mean $\Delta_2=N-H_2(q)$ $\pm$ 1 s.d.")
            if fent["ok"]:
                pred = fent["a"]*nproj + fent["c"]
                ax.plot(nproj, pred, "--", color=COLORS["renyi"],
                        label=fr"$\Delta_2=aN+c$: $a={fent['a']:.3f}\pm{fent['a_err']:.3f}$; "
                              fr"$h_2=1-a={fent['h2_density']:.3f}$")
                ax.axvspan(ns.max(), PROJECTION_N_MAX, color="0.5", alpha=0.08,
                           label="extrapolation region")
            ax.set_title(f"{mode}: collision-entropy deficit")
            ax.set_ylabel(r"$\Delta_2=\log_2(1+4S_2^2)$")
            ax.grid(True, alpha=0.2)
            ax.legend(frameon=False, fontsize=7.5)

            # Exact factor with bounded empirical logit extrapolation.
            tm, tsd, _ = _mean_sd_sem(sc[mode]["uniform_tvd"][di])
            flog = all_fits[int(depth)][mode]["exact_logit"]
            ax = axes[1, col]
            ax.errorbar(ns, tm, yerr=tsd, fmt="^", capsize=3,
                        color=COLORS["exact"], label=r"mean $D_{TV}(q,u)$ $\pm$ 1 s.d.")
            if flog["ok"]:
                z = np.clip(flog["kappa"]*nproj + flog["c"], -80.0, 80.0)
                pred = 1.0/(1.0 + 2.0**(-z))
                ax.plot(nproj, pred, ":", color=COLORS["exact"],
                        label=fr"$\log_2[T/(1-T)]=\kappa N+c$: "
                              fr"$\kappa={flog['kappa']:.3f}\pm{flog['kappa_err']:.3f}$")
                ax.axvspan(ns.max(), PROJECTION_N_MAX, color="0.5", alpha=0.08,
                           label="extrapolation region")
            ax.set_title(f"{mode}: exact factor (empirical projection)")
            ax.set_xlabel("number of qubits $N$")
            ax.set_ylabel(r"$T=D_{TV}(q,u)$")
            ax.set_ylim(0, 1.02)
            ax.grid(True, alpha=0.2)
            ax.legend(frameon=False, fontsize=7.5)
        fig.suptitle(f"Renyi-2 projection at fixed depth {int(depth)}\n"
                     "Rényi: exact entropy identity; exact TVD: bounded empirical logit model")
        fig.tight_layout()
        path = (_bootstrap.RESULTS_DIR
                + f"/renyi_distribution_projection_depth{int(depth)}{_TAG}.png")
        fig.savefig(path, dpi=170)
        plt.close(fig)
        outputs.append(path)
    return outputs


def figure_projection_parameters_vs_depth(depths, all_fits):
    """Summarize entropy-density and exact-logit projection parameters versus depth."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for mode in MODES:
        a = np.array([all_fits[int(d)][mode]["entropy"].get("a", np.nan) for d in depths])
        ae = np.array([all_fits[int(d)][mode]["entropy"].get("a_err", np.nan) for d in depths])
        h = 1.0-a
        k = np.array([all_fits[int(d)][mode]["exact_logit"].get("kappa", np.nan) for d in depths])
        ke = np.array([all_fits[int(d)][mode]["exact_logit"].get("kappa_err", np.nan) for d in depths])
        c = np.array([all_fits[int(d)][mode]["exact_logit"].get("c", np.nan) for d in depths])
        ce = np.array([all_fits[int(d)][mode]["exact_logit"].get("c_err", np.nan) for d in depths])
        axes[0].errorbar(depths, h, yerr=ae, marker="o", capsize=3, label=mode)
        axes[1].errorbar(depths, k, yerr=ke, marker="o", capsize=3, label=mode)
        axes[2].errorbar(depths, c, yerr=ce, marker="o", capsize=3, label=mode)
    axes[0].set_ylabel(r"projected collision-entropy density $h_2(d)=1-a(d)$")
    axes[1].set_ylabel(r"exact-TVD logit slope $\kappa(d)$")
    axes[2].set_ylabel(r"exact-TVD logit intercept $c(d)$")
    for ax in axes:
        ax.set_xlabel("circuit depth")
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(depths))
        ax.set_xticklabels([str(int(d)) for d in depths])
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle("Projection parameters versus circuit depth")
    fig.tight_layout()
    path = (_bootstrap.RESULTS_DIR
            + f"/renyi_distribution_projection_parameters_vs_depth{_TAG}.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def figure_entropy_scaling_each_depth(depths, ns, sc, all_fits):
    """At each depth, fit H2=hN+c and reconstruct the observed S2."""
    outputs = []
    nproj = np.linspace(float(ns.min()), float(PROJECTION_N_MAX), 500)
    for di, depth in enumerate(depths):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
        for col, mode in enumerate(MODES):
            hm, hsd, _ = _mean_sd_sem(sc[mode]["collision_entropy"][di])
            hfit = all_fits[int(depth)][mode]["h2_linear"]
            ax = axes[0, col]
            ax.errorbar(ns, hm/ns, yerr=hsd/ns, fmt="o", capsize=3,
                        color=COLORS["renyi"], label=r"mean $H_2/N$ $\pm$ 1 s.d.")
            if hfit["ok"]:
                pred_density = hfit["h"] + hfit["c"]/nproj
                ax.plot(nproj, pred_density, "--", color=COLORS["renyi"],
                        label=fr"$H_2=hN+c$: $h={hfit['h']:.3f}\pm{hfit['h_err']:.3f}$, "
                              fr"$R^2={hfit['r2']:.3f}$")
                ax.axvspan(ns.max(), PROJECTION_N_MAX, color="0.5", alpha=0.08,
                           label="extrapolation region")
            ax.set_title(f"{mode}: collision-entropy density")
            ax.set_ylabel(r"$H_2(q)/N$")
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.2)
            ax.legend(frameon=False, fontsize=7.5)

            s2m, s2sd, _ = _mean_sd_sem(sc[mode]["renyi_sqrt_factor_unclipped"][di])
            ax = axes[1, col]
            ax.errorbar(ns, s2m, yerr=s2sd, fmt="o", capsize=3,
                        color=COLORS["renyi"], label=r"observed mean $S_2$ $\pm$ 1 s.d.")
            if hfit["ok"]:
                pred = reconstruct_s2_from_h2(nproj, hfit["h"], hfit["c"])
                ax.plot(nproj, pred, "--", color=COLORS["renyi"],
                        label="reconstructed from fitted $H_2$")
                ax.axvspan(ns.max(), PROJECTION_N_MAX, color="0.5", alpha=0.08)
            ax.set_title(f"{mode}: Rényi factor reconstructed from entropy")
            ax.set_xlabel("number of qubits $N$")
            ax.set_ylabel(r"$S_2=\frac{1}{2}\sqrt{2^{N-H_2}-1}$")
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.2)
            ax.legend(frameon=False, fontsize=7.5)
        fig.suptitle(f"Collision-entropy extrapolation at fixed depth {int(depth)}\n"
                     "fit the entropy, then reconstruct the Rényi factor")
        fig.tight_layout()
        path = (_bootstrap.RESULTS_DIR
                + f"/collision_entropy_scaling_depth{int(depth)}{_TAG}.png")
        fig.savefig(path, dpi=170)
        plt.close(fig)
        outputs.append(path)
    return outputs


def figure_exact_support_scaling_each_depth(depths, ns, sc, all_fits):
    """Exact-factor analogue using a uniform-effective-support coordinate.

    H_eff=N+log2(1-T) is exact only for a distribution uniform on K effective outcomes.
    The figure labels this explicitly as a model-dependent projection.
    """
    outputs = []
    nproj = np.linspace(float(ns.min()), float(PROJECTION_N_MAX), 500)
    for di, depth in enumerate(depths):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
        for col, mode in enumerate(MODES):
            hm, hsd, _ = _mean_sd_sem(sc[mode]["exact_support_entropy"][di])
            hfit = all_fits[int(depth)][mode]["exact_support"]
            ax = axes[0, col]
            ax.errorbar(ns, hm/ns, yerr=hsd/ns, fmt="^", capsize=3,
                        color=COLORS["exact"],
                        label=r"mean $H_{\rm supp}^{\rm eff}/N$ $\pm$ 1 s.d.")
            if hfit["ok"]:
                pred_density = hfit["h"] + hfit["c"]/nproj
                ax.plot(nproj, pred_density, ":", color=COLORS["exact"],
                        label=fr"$H_{{\rm supp}}^{{\rm eff}}=hN+c$: "
                              fr"$h={hfit['h']:.3f}\pm{hfit['h_err']:.3f}$, "
                              fr"$R^2={hfit['r2']:.3f}$")
                ax.axvspan(ns.max(), PROJECTION_N_MAX, color="0.5", alpha=0.08,
                           label="extrapolation region")
            ax.set_title(f"{mode}: effective-support density")
            ax.set_ylabel(r"$H_{\rm supp}^{\rm eff}/N$")
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.2)
            ax.legend(frameon=False, fontsize=7.3)

            tm, tsd, _ = _mean_sd_sem(sc[mode]["uniform_tvd"][di])
            ax = axes[1, col]
            ax.errorbar(ns, tm, yerr=tsd, fmt="^", capsize=3,
                        color=COLORS["exact"],
                        label=r"observed mean $D_{TV}(q,u)$ $\pm$ 1 s.d.")
            if hfit["ok"]:
                pred = reconstruct_exact_from_support_entropy(
                    nproj, hfit["h"], hfit["c"])
                ax.plot(nproj, pred, ":", color=COLORS["exact"],
                        label="reconstructed from effective support")
                ax.axvspan(ns.max(), PROJECTION_N_MAX, color="0.5", alpha=0.08)
            ax.set_title(f"{mode}: exact factor reconstructed from support model")
            ax.set_xlabel("number of qubits $N$")
            ax.set_ylabel(r"$T=D_{TV}(q,u)$")
            ax.set_ylim(0, 1.02)
            ax.grid(True, alpha=0.2)
            ax.legend(frameon=False, fontsize=7.3)
        fig.suptitle(f"Exact-factor effective-support projection at fixed depth {int(depth)}\n"
                     r"model: $T=1-2^{H_{\rm supp}^{\rm eff}-N}$ (not an entropy identity)")
        fig.tight_layout()
        path = (_bootstrap.RESULTS_DIR
                + f"/exact_support_entropy_scaling_depth{int(depth)}{_TAG}.png")
        fig.savefig(path, dpi=170)
        plt.close(fig)
        outputs.append(path)
    return outputs



def figure_collision_entropy_heatmap(depths, ns, sc):
    """Combine all fixed-depth collision-entropy plots into one depth-by-N heatmap.

    Each cell is the ensemble mean of H_2(q)/N at one simulated (depth, N) point.
    Rows are categorical depth values rather than a linearly spaced numerical axis,
    which keeps powers-of-two depths equally readable.  The two ansatz modes share
    one color scale so their entropy densities can be compared directly.
    """
    means = {
        mode: sc[mode]["collision_entropy_density"].mean(axis=-1)
        for mode in MODES
    }
    finite = np.concatenate([arr[np.isfinite(arr)] for arr in means.values()])
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    fig, axes = plt.subplots(1, len(MODES), figsize=(12.5, 5.6),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    image = None
    for ax, mode in zip(axes, MODES):
        arr = means[mode]
        image = ax.imshow(arr, origin="lower", aspect="auto",
                          interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(f"{mode} ansatz")
        ax.set_xlabel("number of qubits $N$")
        ax.set_xticks(np.arange(len(ns)))
        ax.set_xticklabels([str(int(n)) for n in ns])
        ax.set_yticks(np.arange(len(depths)))
        ax.set_yticklabels([str(int(d)) for d in depths])

        # Print the numerical mean in every cell so the heatmap remains useful when
        # the dynamic range is narrow or when viewed in grayscale.
        midpoint = 0.5 * (vmin + vmax)
        for di in range(len(depths)):
            for ni in range(len(ns)):
                value = arr[di, ni]
                if np.isfinite(value):
                    ax.text(ni, di, f"{value:.3f}", ha="center", va="center",
                            color="white" if value < midpoint else "black",
                            fontsize=7.5)

    axes[0].set_ylabel("circuit depth")
    cbar = fig.colorbar(image, ax=axes, shrink=0.92, pad=0.03)
    cbar.set_label(r"ensemble mean collision-entropy density $H_2(q)/N$")
    fig.suptitle("Collision-entropy scaling across circuit depth and system size\n"
                 f"each cell averages {SCALING_INSTANCES} circuit instances")
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.12, top=0.84, wspace=0.08)
    path = (_bootstrap.RESULTS_DIR
            + f"/rc_collision_entropy_density_grid_sem{_TAG}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path

def figure_entropy_density_heatmaps(depths, ns, sc):
    """Heatmaps of the two entropy-density coordinates over the simulated grid."""
    outputs = []
    for mode in MODES:
        h2 = sc[mode]["collision_entropy_density"].mean(axis=-1)
        hs = sc[mode]["exact_support_entropy_density"].mean(axis=-1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
        for ax, arr, title in (
                (axes[0], h2, r"collision-entropy density $H_2/N$"),
                (axes[1], hs, r"effective-support density $H_{\rm supp}^{\rm eff}/N$")):
            im = ax.imshow(arr, origin="lower", aspect="auto",
                           extent=[ns.min()-0.5, ns.max()+0.5,
                                   depths.min()-0.5, depths.max()+0.5])
            ax.set_title(title)
            ax.set_xlabel("number of qubits $N$")
            ax.set_xticks(list(ns))
            fig.colorbar(im, ax=ax)
        axes[0].set_ylabel("circuit depth")
        axes[0].set_yticks(list(depths))
        fig.suptitle(f"{mode} ansatz: entropy-density surfaces over $(N,d)$")
        fig.tight_layout()
        path = (_bootstrap.RESULTS_DIR
                + f"/entropy_density_heatmaps_{mode}{_TAG}.png")
        fig.savefig(path, dpi=170)
        plt.close(fig)
        outputs.append(path)
    return outputs


def figure_entropy_density_parameters(depths, all_fits):
    """Fitted entropy densities h(d) for Rényi and exact-support projections."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for mode in MODES:
        h2 = np.array([all_fits[int(d)][mode]["h2_linear"].get("h", np.nan)
                       for d in depths])
        h2e = np.array([all_fits[int(d)][mode]["h2_linear"].get("h_err", np.nan)
                        for d in depths])
        hs = np.array([all_fits[int(d)][mode]["exact_support"].get("h", np.nan)
                       for d in depths])
        hse = np.array([all_fits[int(d)][mode]["exact_support"].get("h_err", np.nan)
                        for d in depths])
        axes[0].errorbar(depths, h2, yerr=h2e, marker="o", capsize=3, label=mode)
        axes[1].errorbar(depths, hs, yerr=hse, marker="^", capsize=3, label=mode)
    axes[0].set_ylabel(r"collision-entropy density $h_2(d)$")
    axes[1].set_ylabel(r"effective-support density $h_{\rm supp}^{\rm eff}(d)$")
    for ax in axes:
        ax.set_xlabel("circuit depth")
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(depths))
        ax.set_xticklabels([str(int(d)) for d in depths])
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle("Entropy-density fit parameters versus depth")
    fig.tight_layout()
    path = (_bootstrap.RESULTS_DIR
            + f"/entropy_density_parameters_vs_depth{_TAG}.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def save_main(data):
    """Per-depth AND per-instance results of the main experiment (section 6)."""
    kw = {"depths": np.array(DEPTHS), "theta_zz": THETA_ZZ, "n_traj": N_TRAJ,
          "tvd_shots": TVD_SHOTS, "oneq_set": ONEQ_SET, "n_qubits": N, "seed": SEED,
          "n_instances": N_INSTANCES, "band": BAND,
          "oneq_random": np.array(_ONEQ["random"]),
          "oneq_structured": np.array(_ONEQ["structured"])}
    for m, d in data.items():
        kw[f"{m}_bound"] = d["bound"]                 # raw QCAP, one per depth
        kw[f"{m}_bound_std"] = d["bound_std"]         # CB parameter uncertainty
        kw[f"{m}_tvd"] = d["measured_tvd"]            # (depth, instance), RC'd
        kw[f"{m}_clipped_fraction"] = d["clipped_fraction"]
        for k in _PER_INSTANCE:                       # all (depth, instance)
            kw[f"{m}_{k}"] = d[k]
            kw[f"{m}_{k}_mean"] = d[k].mean(axis=1)
            kw[f"{m}_{k}_std"] = d[k].std(axis=1)
    return save_results(OUT_DATA, **kw)


def save_scaling(depths, ns, sc, all_fits, skipped):
    kw = {"N_scaling": np.array(N_SCALING), "N_scaling_used": ns,
          "scaling_depths": depths,
          "N_scaling_skipped": np.array([n for n, _ in skipped], dtype=int),
          "scaling_instances": SCALING_INSTANCES, "scaling_seed": SCALING_SEED,
          "oneq_set": ONEQ_SET, "mem_budget_gb": SCALING_MEM_BUDGET_GB}
    for mode in MODES:
        for key, arr in sc[mode].items():
            kw[f"{mode}_{key}"] = arr
            if arr.ndim == 3:
                kw[f"{mode}_{key}_mean"] = arr.mean(axis=-1)
                kw[f"{mode}_{key}_std"] = arr.std(axis=-1, ddof=1)
                kw[f"{mode}_{key}_sem"] = (
                    arr.std(axis=-1, ddof=1) / math.sqrt(arr.shape[-1]))
        for depth in depths:
            di = int(np.where(depths == depth)[0][0])
            fs2 = all_fits[int(depth)][mode]["s2"]
            fex = all_fits[int(depth)][mode]["exact"]
            prefix = f"fit_d{int(depth)}_{mode}"
            kw[prefix + "_s2_ok"] = fs2["ok"]
            kw[prefix + "_exact_ok"] = fex["ok"]
            if fs2["ok"]:
                for model, keys in (("exp", ("A", "b")),
                                    ("pow2", ("A", "alpha")),
                                    ("sqrt_dim", ("C",))):
                    for k in keys:
                        kw[f"{prefix}_s2_{model}_{k}"] = fs2[model][k]
                        kw[f"{prefix}_s2_{model}_{k}_err"] = fs2[model][f"{k}_err"]
                    kw[f"{prefix}_s2_{model}_r2"] = fs2[model]["r2"]
            fhaar = all_fits[int(depth)][mode]["haar"]
            kw[prefix + "_haar_ok"] = fhaar["ok"]
            if fhaar["ok"]:
                for k in ("A", "b"):
                    kw[f"{prefix}_haar_{k}"] = fhaar[k]
                    kw[f"{prefix}_haar_{k}_err"] = fhaar[f"{k}_err"]
                kw[f"{prefix}_haar_r2"] = fhaar["r2"]
                kw[f"{prefix}_haar_rmse"] = fhaar["rmse"]
                kw[f"{prefix}_haar_fixed_r2"] = fhaar["fixed_haar_r2"]
                kw[f"{prefix}_haar_fixed_rmse"] = fhaar["fixed_haar_rmse"]
                kw[f"{prefix}_haar_cov"] = fhaar["cov"]
            if fex["ok"]:
                for model, keys in (("sat1", ("A", "b")),
                                    ("satL", ("L", "A", "b"))):
                    for k in keys:
                        kw[f"{prefix}_exact_{model}_{k}"] = fex[model][k]
                        kw[f"{prefix}_exact_{model}_{k}_err"] = fex[model][f"{k}_err"]
                    kw[f"{prefix}_exact_{model}_r2"] = fex[model]["r2"]
            fent = all_fits[int(depth)][mode]["entropy"]
            flog = all_fits[int(depth)][mode]["exact_logit"]
            kw[prefix + "_entropy_ok"] = fent["ok"]
            kw[prefix + "_exact_logit_ok"] = flog["ok"]
            if fent["ok"]:
                for k in ("a", "c", "h2_density"):
                    kw[f"{prefix}_entropy_{k}"] = fent[k]
                kw[f"{prefix}_entropy_a_err"] = fent["a_err"]
                kw[f"{prefix}_entropy_c_err"] = fent["c_err"]
                kw[f"{prefix}_entropy_h2_density_err"] = fent["h2_density_err"]
                kw[f"{prefix}_entropy_r2"] = fent["r2"]
                kw[f"{prefix}_entropy_cov"] = fent["cov"]
            if flog["ok"]:
                for k in ("kappa", "c"):
                    kw[f"{prefix}_exact_logit_{k}"] = flog[k]
                    kw[f"{prefix}_exact_logit_{k}_err"] = flog[f"{k}_err"]
                kw[f"{prefix}_exact_logit_r2"] = flog["r2"]
                kw[f"{prefix}_exact_logit_cov"] = flog["cov"]
            for fit_name in ("h2_linear", "exact_support"):
                ff = all_fits[int(depth)][mode][fit_name]
                kw[f"{prefix}_{fit_name}_ok"] = ff["ok"]
                if ff["ok"]:
                    for k in ("h", "c"):
                        kw[f"{prefix}_{fit_name}_{k}"] = ff[k]
                        kw[f"{prefix}_{fit_name}_{k}_err"] = ff[f"{k}_err"]
                    kw[f"{prefix}_{fit_name}_r2"] = ff["r2"]
                    kw[f"{prefix}_{fit_name}_cov"] = ff["cov"]
    return save_results(OUT_SCALING, **kw)


def main():
    self_test()
    for m in MODES:
        cliff = "t" not in _ONEQ[m]
        print(f"1q gate set ({m:>10}): {_ONEQ[m]}  ->  "
              + ("Clifford, exact stim path" if cliff
                 else f"NON-Clifford, {N_TRAJ} trajectories per point"))
    print(f"coherent residual ZZ: theta_zz = {THETA_ZZ} rad  ->  twirled "
          f"p_IZ = p_ZI = p_ZZ = {TWIRLED.p_zz:.5f} per 2q gate "
          f"(vs p2 = {NOISE.p2} depolarizing)")
    print('note: "exact-WN" is exact ONLY under the global white-noise mixture '
          "q = (1-eps)p + eps*u; it is a model, not a guarantee.")

    data = {m: run(m) for m in MODES}
    out_data = save_main(data)

    scaling_depths, ns, sc, skipped = scaling_study()
    all_fits = {
        int(depth): {
            mode: {
                "s2": fit_scaling(
                    ns,
                    sc[mode]["renyi_sqrt_factor_unclipped"][di].mean(axis=-1),
                    sc[mode]["renyi_sqrt_factor_unclipped"][di].std(axis=-1, ddof=1)
                    / math.sqrt(SCALING_INSTANCES)),
                "haar": fit_haar_relaxation(
                    ns,
                    sc[mode]["renyi_sqrt_factor_unclipped"][di].mean(axis=-1),
                    sc[mode]["renyi_sqrt_factor_unclipped"][di].std(axis=-1, ddof=1)
                    / math.sqrt(SCALING_INSTANCES)),
                "exact": fit_exact_scaling(
                    ns,
                    sc[mode]["uniform_tvd"][di].mean(axis=-1),
                    sc[mode]["uniform_tvd"][di].std(axis=-1, ddof=1)
                    / math.sqrt(SCALING_INSTANCES)),
                "entropy": fit_collision_entropy_deficit(
                    ns,
                    sc[mode]["collision_entropy_deficit"][di].mean(axis=-1),
                    sc[mode]["collision_entropy_deficit"][di].std(axis=-1, ddof=1)
                    / math.sqrt(SCALING_INSTANCES)),
                "exact_logit": fit_exact_logit_scaling(
                    ns,
                    sc[mode]["uniform_tvd"][di].mean(axis=-1),
                    sc[mode]["uniform_tvd"][di].std(axis=-1, ddof=1)
                    / math.sqrt(SCALING_INSTANCES)),
                "h2_linear": fit_extensive_entropy(
                    ns,
                    sc[mode]["collision_entropy"][di].mean(axis=-1),
                    sc[mode]["collision_entropy"][di].std(axis=-1, ddof=1)
                    / math.sqrt(SCALING_INSTANCES), name="collision_entropy"),
                "exact_support": fit_extensive_entropy(
                    ns,
                    sc[mode]["exact_support_entropy"][di].mean(axis=-1),
                    sc[mode]["exact_support_entropy"][di].std(axis=-1, ddof=1)
                    / math.sqrt(SCALING_INSTANCES), name="exact_support_entropy"),
            }
            for mode in MODES
        }
        for di, depth in enumerate(scaling_depths)
    }
    print_fits(all_fits)
    out_scaling = save_scaling(scaling_depths, ns, sc, all_fits, skipped)
    # Requested plot products:
    #   1. one four-panel TVD/exact/Renyi/QCAP figure for each fixed depth;
    #   2. one dedicated unclipped S2 and exact-factor figure versus N for
    #      each fixed depth;
    #   3. one collision-entropy density heatmap over the depth-by-N grid.
    depth_figs = figure_each_depth(scaling_depths, ns, sc, all_fits)
    renyi_factor_figs = figure_renyi_factors_each_depth(
        scaling_depths, ns, sc, all_fits)
    collision_entropy_heatmap = figure_collision_entropy_heatmap(
        scaling_depths, ns, sc)

    print(f"\nvalidation: D_TV(p,u) <= min(1, S_2) verified on {CHECKED['factors']} "
          f"circuits; B_exact <= B_Renyi <= raw QCAP and 0 <= bound <= 1 verified on "
          f"{CHECKED['bounds']} circuits; closed-form self-test passed "
          f"(tolerance {TOL:g})")
    print("\nWrote:")
    for f in (out_data, out_scaling, *depth_figs, *renyi_factor_figs,
              collision_entropy_heatmap):
        print(f"  {f}")
    if skipped:
        print("Skipped N (memory): "
              + ", ".join(f"{n} (~{gb:.2f} GiB)" for n, gb in skipped))


if __name__ == "__main__":
    main()
