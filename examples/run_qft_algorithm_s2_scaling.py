"""Algorithm-family explicit-RC TVD, three bounds, and ideal Renyi-2 scaling.

For every width N and every complete-circuit repetition d of an MQT Bench algorithm
this script measures the explicitly randomly-compiled Z-basis TVD and compares it to
three upper bounds, in increasing order of looseness:

    B_wn    = eps_qcap * D_TV(p_ideal, u)          exact white-noise model
    B_renyi = eps_qcap * min(1, S_2)               Renyi-2 bound
    eps_qcap                                       raw QCAP process-fidelity bound

with S_2 = (1/2) sqrt(2^N sum_x p(x)^2 - 1) the unclipped Renyi-2 factor of the ideal
output.  The identity B_wn <= B_renyi <= eps_qcap is asserted per point.

eps_qcap is built from a *per-cycle* census: cycle benchmarking is run once on every
distinct entangling ASAP layer the circuit actually executes, and the bound is the
product over those layers with their true multiplicities.  Counting one proxy cycle per
two-qubit *gate* instead over-charges the single-qubit and idling layers by the circuit's
parallelism factor, which for QFT is roughly 3.5x.

Figures:

    {alg}{N}_explicit_rc_bound_tvd.png    per width: TVD + three bounds vs cycle count
    {alg}_tvd_vs_N.png                    per repetition: TVD + three bounds vs N
    {alg}_s2_factor_scaling_vs_N.png      per repetition: exact ideal S_2 vs N, with fits
    {alg}_s2_fit_parameters_vs_repetition.png   fitted rates vs d

NOTE ON THE ALGORITHM CHOICE.  Plain ``qft`` has a degenerate ideal output: for any
computational basis state |x>, |<y|QFT|x>| = 2^(-N/2) exactly, so QFT|0...0> is the
uniform distribution (S_2 = 0, D_TV(p,u) = 0) and QFT^2|0...0> = |0...0> is
deterministic (S_2 = (1/2) sqrt(2^N - 1), exactly the deterministic reference curve).
S_2 therefore alternates between the two closed-form extremes with period two in d and
carries no scaling information -- the power-law fit has nothing to fit at odd d and is
trivially exact at even d.  ``qftentangled`` (a GHZ prep ahead of the transform) has a
genuinely non-trivial output and is the default.  Set ALGORITHM = "qft" to reproduce the
degenerate case anyway; the fits detect it and report why they were skipped rather than
emitting silent NaNs.
"""

from __future__ import annotations

import os
import random
import warnings

# Third-party deprecation chatter only.  This script reports its own diagnostics
# with print() precisely so a blanket filter cannot swallow them.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector
from scipy.optimize import curve_fit

import _bootstrap  # noqa: F401
from proxysim import NoiseModel, save_results
from proxysim.backends import StatevectorBackend
from proxysim.benchmarking import (
    cycle_benchmark,
    qcap_bound,
    readout_fidelity,
)
from proxysim.circuit import circuit_from_qasm
from proxysim.mqtbank import cycle_census, fetch, repeat, two_qubit_families
from proxysim.noise import (
    apply_readout_to_distribution,
    sample_trajectory,
)
from proxysim.parallel import pmap
from proxysim.rc import pauli_twirl


# =============================================================================
# Configuration
# =============================================================================

# "qftentangled" (default) or "qft" -- see the module docstring on degeneracy.
ALGORITHM = "qftentangled"

QUBIT_SIZES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14]

# Complete repetitions of the imported circuit: target = U^d.
ALGORITHM_REPETITIONS = [1, 2, 3, 4, 6, 8]

# A local QASM file is preferred when present, so a hand-placed circuit still wins;
# otherwise the circuit is generated with mqt.bench.
QASM_TEMPLATE = "{algorithm}{n}.qasm"

# Sequence lengths used internally by cycle benchmarking.
# Unrelated to the target-circuit repetition counts above.
CB_DEPTHS = [1, 2, 4, 8]

# Fixed physical noise model throughout the sweep.
NOISE = NoiseModel(
    enabled=True,
    p1=5e-5,
    p2=2.5e-4,
    p_readout=2.5e-4,
    p_idle=5e-5,
    p_z1=0.0,
    p_zz=0.0,
    theta_1q=0.0,
    theta_zz=0.0,
)

# Number of independently estimated TVD values shown at every cycle depth.
N_TVD_POINTS = 30

# Number of independently dressed circuits averaged to produce one TVD point.
N_RC_REALIZATIONS = 24

# Stochastic noise trajectories per dressed RC realization.  Total trajectories
# behind one plotted TVD point is N_RC_REALIZATIONS * N_TRAJ_PER_RC.
N_TRAJ_PER_RC = 12

# Cycle-benchmarking statistics.
CB_SHOTS = 800
CB_DECAYS = 20

# Two-qubit carrier used for every proxy cycle.  The QFT family mixes cp / cx / swap
# in a single ASAP layer, which one CB cycle cannot represent; the stochastic part of
# the noise model (DEPOLARIZE2 at p2) is gate-independent, so the carrier choice does
# not change e_F, only the Clifford frame the twirl runs in.
CB_CARRIER = "CZ"

SEED = 42

# Coverage factor for the shaded QCAP uncertainty band (1.96 = 95%).
QCAP_Z = 1.96

_SV = StatevectorBackend()

OUT_TVD_TEMPLATE = (
    _bootstrap.RESULTS_DIR + "/{algorithm}{n}_explicit_rc_bound_tvd.png"
)
OUT_TVD_VS_N = _bootstrap.RESULTS_DIR + f"/{ALGORITHM}_tvd_vs_N.png"
OUT_S2 = _bootstrap.RESULTS_DIR + f"/{ALGORITHM}_s2_factor_scaling_vs_N.png"
OUT_FIT_PARAMETERS = (
    _bootstrap.RESULTS_DIR + f"/{ALGORITHM}_s2_fit_parameters_vs_repetition.png"
)
OUT_FAMILY_DATA = (
    _bootstrap.RESULTS_DIR + f"/{ALGORITHM}_explicit_rc_s2_scaling_data.npz"
)

TOL = 1e-9


# =============================================================================
# Circuit loading
# =============================================================================

def qasm_path(n: int) -> str:
    """Path a hand-placed QASM file for this width would live at."""
    return os.path.join(
        os.path.dirname(__file__),
        QASM_TEMPLATE.format(algorithm=ALGORITHM, n=n),
    )


def load_algorithm_circuit(n: int):
    """Load one family member: local QASM if present, else generate via mqt.bench."""
    path = qasm_path(n)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            circ, _measured = circuit_from_qasm(handle.read())
        source = os.path.basename(path)
    else:
        circ, _measured = fetch(ALGORITHM, n)
        source = "mqt.bench"

    if circ.n_qubits != n:
        raise ValueError(
            f"{source} gave {circ.n_qubits} qubits for {ALGORITHM}{n}, expected {n}"
        )
    return circ, source


# =============================================================================
# Explicit randomized compiling
# =============================================================================

def explicitly_randomized_compile(circ, seed: int):
    """Return ``(dressed_circuit, virtual_gate_indices)`` for one RC realization.

    ``mark_virtual=True`` is essential: the twirling Paulis are frame changes, and
    without the report they are charged physical p1 and lengthen the idle accounting,
    so an RC-vs-raw comparison would measure the gate count instead of the channel.
    """
    return pauli_twirl(circ, rng=random.Random(seed), mark_virtual=True)


# =============================================================================
# Workers (module level: pmap uses a process pool)
# =============================================================================

def cycle_ef(args):
    """Run cycle benchmarking on one entangling ASAP layer of the target circuit."""
    pairs, n, seed = args
    cb = cycle_benchmark(
        [tuple(pair) for pair in pairs],
        n,
        CB_DEPTHS,
        NOISE,
        twoq=CB_CARRIER,
        n_decays=CB_DECAYS,
        shots=CB_SHOTS,
        seed=seed,
    )
    return float(cb["e_F"]), float(cb["e_F_std"])


def rc_averaged_tvd(args):
    """One raw TVD estimate for the explicitly RC'd target circuit.

    The averaging hierarchy is trajectory average within each dressed circuit, then
    RC-realization average across dressed circuits, then a single TVD against the
    ideal distribution -- not the mean of per-realization TVDs.
    """
    (
        target,
        noise,
        ideal_prob,
        n_rc_realizations,
        n_traj_per_rc,
        seed,
    ) = args

    rc_average_prob = np.zeros_like(ideal_prob, dtype=float)

    for rc_index in range(n_rc_realizations):
        rc_seed = seed + 1_000_003 * rc_index
        dressed, virtual_indices = explicitly_randomized_compile(target, seed=rc_seed)
        trajectory_rng = random.Random(rc_seed + 1)

        dressed_prob = np.zeros_like(ideal_prob, dtype=float)
        for _ in range(n_traj_per_rc):
            noisy_dressed = sample_trajectory(
                dressed, noise, trajectory_rng, virtual=virtual_indices
            )
            psi = np.asarray(
                Statevector(_SV._build(noisy_dressed)).data, dtype=complex
            )
            dressed_prob += np.abs(psi) ** 2
        dressed_prob /= n_traj_per_rc

        # sample_trajectory does not apply measurement error; the bit-flip channel is
        # linear, so applying it to the trajectory average is exact.
        dressed_prob = apply_readout_to_distribution(
            dressed_prob, noise, target.n_qubits
        )
        rc_average_prob += dressed_prob

    rc_average_prob /= n_rc_realizations
    return 0.5 * float(np.abs(rc_average_prob - ideal_prob).sum())


def scatter_summary(values):
    """Mean and range for one TVD scatter column."""
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.5f} [{values.min():.5f}, {values.max():.5f}]"


# =============================================================================
# Ideal-distribution factors and the three bounds
# =============================================================================

def ideal_statevector(circ) -> np.ndarray:
    """Ideal statevector of a proxysim circuit."""
    return np.asarray(Statevector(_SV._build(circ)).data, dtype=complex)


def ideal_factors(prob: np.ndarray) -> dict:
    """Exact ideal-output S_2, D_TV(p,u) and the collision-entropy quantities.

    ``exp_d2 = 2^N sum_x p(x)^2 = e^{D_2(p||u)}`` and

        S_2      = (1/2) sqrt(exp_d2 - 1)
        Delta_2  = log2(exp_d2) = N - H_2(p) = log2(1 + 4 S_2^2)

    ``exp_d2 - 1`` is clamped at zero before the root: for uniform p it is
    algebraically zero and floating point can put it a few ulps below.
    """
    prob = np.asarray(prob, dtype=float)
    total = float(prob.sum())
    if not np.isclose(total, 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError(f"ideal probabilities sum to {total}, not one")
    prob = prob / total

    dimension = prob.size
    n = int(round(np.log2(dimension)))
    uniform_prob = 1.0 / dimension

    exp_d2 = dimension * float(np.sum(prob**2))
    s2 = 0.5 * float(np.sqrt(max(exp_d2 - 1.0, 0.0)))
    delta2 = float(np.log2(exp_d2)) if exp_d2 > 0.0 else -np.inf
    h2 = n - delta2

    return {
        "s2": s2,
        "s2_clipped": min(1.0, s2),
        "uniform_tvd": 0.5 * float(np.abs(prob - uniform_prob).sum()),
        "exp_d2": float(exp_d2),
        "delta2": delta2,
        "h2": float(h2),
        "h2_density": float(h2 / n),
        "support": int(np.count_nonzero(prob > 1e-12)),
        "uniform": bool(np.allclose(prob, uniform_prob, atol=1e-10, rtol=1e-10)),
    }


def bounds_from(eps_qcap: float, factors: dict) -> dict:
    """The white-noise and Renyi-2 bounds, with their ordering asserted.

        B_wn = eps * D_TV(p,u)  <=  B_renyi = eps * min(1, S_2)  <=  eps
    """
    wn = eps_qcap * factors["uniform_tvd"]
    renyi = eps_qcap * factors["s2_clipped"]

    if not wn <= renyi + TOL:
        raise AssertionError(f"white-noise bound {wn} exceeds Renyi bound {renyi}")
    if not renyi <= eps_qcap + TOL:
        raise AssertionError(f"Renyi bound {renyi} exceeds QCAP {eps_qcap}")
    for name, value in (("wn_bound", wn), ("renyi_bound", renyi)):
        if not -TOL <= value <= 1.0 + TOL:
            raise AssertionError(f"{name}={value} outside [0, 1]")

    return {"wn_bound": float(wn), "renyi_bound": float(renyi)}


def state_fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """Pure-state fidelity, insensitive to global phase."""
    return float(abs(np.vdot(psi, phi)) ** 2)


def validate_one_rc_realization(base_circ):
    """Verify that the explicit RC pass preserves the ideal logical circuit."""
    dressed, virtual_indices = explicitly_randomized_compile(base_circ, seed=SEED)
    fidelity = state_fidelity(
        ideal_statevector(base_circ), ideal_statevector(dressed)
    )
    original_twoq = sum(len(g.qubits) == 2 for g in base_circ.gates)
    dressed_twoq = sum(len(g.qubits) == 2 for g in dressed.gates)

    print(
        "explicit-RC smoke test:\n"
        f"    original gate count: {len(base_circ.gates)}\n"
        f"    dressed gate count:  {len(dressed.gates)}\n"
        f"    original 2q gates:   {original_twoq}\n"
        f"    dressed 2q gates:    {dressed_twoq}\n"
        f"    virtual RC gates:    {len(virtual_indices)}\n"
        f"    ideal state fidelity: {fidelity:.12f}"
    )

    if len(virtual_indices) == 0:
        print(
            "    WARNING: pauli_twirl reported no virtual gate indices; dressing "
            "gates would be charged physical p1 and idle noise."
        )
    if not np.isclose(fidelity, 1.0, atol=1e-9, rtol=1e-9):
        raise RuntimeError(
            "The dressed circuit is not logically equivalent to the original circuit."
        )


# =============================================================================
# Fits across N at fixed repetition
# =============================================================================

def _r2(y: np.ndarray, fitted: np.ndarray) -> float:
    """Coefficient of determination, NaN when the data carry no variance.

    A bare ``ss_tot > 0`` test is not enough: a series that is constant up to rounding
    has ss_tot of order 1e-31, and dividing by it turns float noise into an R^2 of -15.
    """
    y = np.asarray(y, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((y - fitted) ** 2))
    scale = max(1.0, float(np.max(y**2))) * y.size
    if ss_tot <= 1e-20 * scale:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _pm(err: float) -> str:
    """Mathtext ``\\pm`` clause; non-finite errors would render as ``\\pminf``."""
    return rf"\pm{err:.3f}" if np.isfinite(err) else r"\pm\mathrm{n/a}"


def fit_s2_power2(ns, values):
    """Fit strictly positive S_2 points to S_2 = A * 2^(alpha N).

    Returns ``ok=False`` with a human-readable ``reason`` rather than NaNs when the
    ideal output is uniform at every width (S_2 identically zero), which is what plain
    ``qft`` does at odd repetition.
    """
    ns = np.asarray(ns, dtype=float)
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values) & (values > 0.0)
    out = {
        "ok": False,
        "reason": "",
        "N_used": ns[good].astype(int),
        "dropped_N": ns[~good].astype(int).tolist(),
    }

    if not np.any(good):
        out["reason"] = "S2 = 0 at every N (ideal output is uniform); nothing to fit"
        return out
    if np.count_nonzero(good) < 3:
        out["reason"] = (
            f"only {np.count_nonzero(good)} positive S2 points; need at least three"
        )
        return out

    x, y = ns[good], values[good]
    log2y = np.log2(y)
    alpha0, log2a0 = np.polyfit(x, log2y, 1)
    par, cov = curve_fit(
        lambda z, log2a, alpha: log2a + alpha * z,
        x,
        log2y,
        p0=[log2a0, alpha0],
        maxfev=20000,
    )
    err = np.sqrt(np.diag(cov))
    pred = 2.0 ** (par[0] + par[1] * x)
    out.update({
        "ok": True,
        # A flat S_2 is a real result (the collision entropy tracks N), but alpha is
        # then zero by construction and its error is undefined -- say so.
        "degenerate": bool(np.allclose(log2y, log2y[0], atol=1e-12)),
        "A": float(2.0 ** par[0]),
        "A_err": float(np.log(2.0) * 2.0 ** par[0] * err[0]),
        "alpha": float(par[1]),
        "alpha_err": float(err[1]),
        "r2": _r2(y, pred),
    })
    return out


def fit_entropy_deficit(ns, delta2):
    """Fit the exact transformed quantity Delta_2 = log2(1 + 4 S_2^2) = a N + c."""
    ns = np.asarray(ns, dtype=float)
    delta2 = np.asarray(delta2, dtype=float)
    good = np.isfinite(delta2)
    out = {
        "ok": False,
        "reason": "",
        "N_used": ns[good].astype(int),
        "dropped_N": ns[~good].astype(int).tolist(),
    }
    if np.count_nonzero(good) < 3:
        out["reason"] = "fewer than three finite Delta2 points"
        return out

    x, y = ns[good], delta2[good]
    par, cov = curve_fit(lambda z, a, c: a * z + c, x, y, maxfev=20000)
    err = np.sqrt(np.diag(cov))
    pred = par[0] * x + par[1]
    out.update({
        "ok": True,
        "a": float(par[0]),
        "a_err": float(err[0]),
        "c": float(par[1]),
        "c_err": float(err[1]),
        # H_2/N -> 1 - a, and Var(1 - a) = Var(a).
        "h2_density": float(1.0 - par[0]),
        "h2_density_err": float(err[0]),
        "r2": _r2(y, pred),
        # Delta2 is flat when the ideal output stays uniform: report it rather than
        # letting a degenerate a = 0 look like a measurement.
        "degenerate": bool(np.allclose(y, y[0], atol=1e-12)),
    })
    return out


def reconstruct_s2_from_delta_fit(n, fit):
    n = np.asarray(n, dtype=float)
    delta = fit["a"] * n + fit["c"]
    return 0.5 * np.sqrt(np.maximum(np.exp2(delta) - 1.0, 0.0))


def deterministic_s2(n):
    """S_2 of a point-mass ideal output: (1/2) sqrt(2^N - 1)."""
    n = np.asarray(n, dtype=float)
    return 0.5 * np.sqrt(np.maximum(np.exp2(n) - 1.0, 0.0))


def haar_s2(n):
    """S_2 of a finite-N Haar-random pure state's Born distribution, in expectation."""
    n = np.asarray(n, dtype=float)
    d = np.exp2(n)
    return 0.5 * np.sqrt(np.maximum((d - 1.0) / (d + 1.0), 0.0))


# =============================================================================
# Per-width run
# =============================================================================

def _cycle_key(entry) -> tuple:
    """Hashable, order-stable identifier for one entangling ASAP layer."""
    return tuple(tuple(int(q) for q in pair) for pair in entry["pairs"])


def benchmark_cycles(n: int, cycle_keys, n_workers: int) -> dict:
    """Cycle-benchmark every distinct entangling layer once.

    Returns ``{cycle_key: (e_F, e_F_std)}``.  The distinct-cycle set does not grow with
    the repetition count -- U^d executes the same layers d times -- so this is run once
    per width and reused across every d.
    """
    keys = list(cycle_keys)
    jobs = [
        (key, n, SEED + 3000 * n + 17 * index)
        for index, key in enumerate(keys)
    ]
    results = pmap(cycle_ef, jobs, n_workers=n_workers)
    return dict(zip(keys, results))


def run_one_size(n: int) -> dict:
    """Explicit-RC TVD, the three bounds, and exact ideal S_2 for one width."""
    base_circ, source = load_algorithm_circuit(n)
    twoq_per_rep = sum(len(gate.qubits) == 2 for gate in base_circ.gates)
    if twoq_per_rep == 0:
        raise ValueError(f"{ALGORITHM}{n} has no two-qubit gates")

    print(f"\n=== {ALGORITHM.upper()}-{n}  (from {source}) ===")
    print(base_circ.summary())
    print(
        f"two-qubit gates per repetition: {twoq_per_rep}   "
        f"families: {two_qubit_families(base_circ)}"
    )
    validate_one_rc_realization(base_circ)

    n_workers = max(1, (os.cpu_count() or 2) - 2)

    # Build every target first so the CB set covers all repetitions in one pass.
    targets, censuses = {}, {}
    for repetition in ALGORITHM_REPETITIONS:
        target = repeat(base_circ, repetition)
        expected = repetition * twoq_per_rep
        found = sum(len(gate.qubits) == 2 for gate in target.gates)
        if found != expected:
            raise RuntimeError(
                f"expected {expected} two-qubit gates for N={n}, d={repetition}; "
                f"found {found}"
            )
        targets[repetition] = target
        censuses[repetition] = cycle_census(target)

    distinct = []
    for repetition in ALGORITHM_REPETITIONS:
        for entry in censuses[repetition]:
            key = _cycle_key(entry)
            if key not in distinct:
                distinct.append(key)

    total_layers_d1 = sum(e["count"] for e in censuses[ALGORITHM_REPETITIONS[0]])
    print(
        f"cycle census: {len(distinct)} distinct entangling layers; "
        f"{total_layers_d1} layers at d={ALGORITHM_REPETITIONS[0]} "
        f"(vs {twoq_per_rep * ALGORITHM_REPETITIONS[0]} two-qubit gates)"
    )

    efs = benchmark_cycles(n, distinct, n_workers)
    ro_fid, ro_std = readout_fidelity(n, NOISE, seed=SEED + 2000 * n)
    e_f_values = np.asarray([efs[key][0] for key in distinct], dtype=float)
    print(
        f"per-cycle e_F: min={e_f_values.min():.3e} max={e_f_values.max():.3e} "
        f"mean={e_f_values.mean():.3e}   readout fidelity={ro_fid:.6f}"
    )

    rows, tvd_columns = [], []
    for repetition in ALGORITHM_REPETITIONS:
        target = targets[repetition]
        census = censuses[repetition]

        counts = {}
        cycle_efs = {}
        for index, entry in enumerate(census):
            key = _cycle_key(entry)
            label = f"cycle{index}"
            counts[label] = int(entry["count"])
            cycle_efs[label] = efs[key]

        bound = qcap_bound(counts, cycle_efs, ro_fid, ro_std)
        eps_qcap = float(bound["error"])

        ideal_prob = np.abs(ideal_statevector(target)) ** 2
        factors = ideal_factors(ideal_prob)
        model_bounds = bounds_from(eps_qcap, factors)

        jobs = [
            (
                target,
                NOISE,
                ideal_prob,
                N_RC_REALIZATIONS,
                N_TRAJ_PER_RC,
                SEED + 100_000_000 * n + 1_000_000 * repetition + 10_000 * j,
            )
            for j in range(N_TVD_POINTS)
        ]
        tvd = np.asarray(pmap(rc_averaged_tvd, jobs, n_workers=n_workers), dtype=float)
        tvd_columns.append(tvd)

        twoq_gates = repetition * twoq_per_rep
        cycle_layers = sum(entry["count"] for entry in census)
        rows.append({
            "repetition": repetition,
            "twoq_gates": twoq_gates,
            "cycle_layers": cycle_layers,
            "qcap": eps_qcap,
            "qcap_std": float(bound["std"]),
            **model_bounds,
            **factors,
        })

        # The trajectory estimator resolves TVD only to about
        # 1 / (N_RC_REALIZATIONS * N_TRAJ_PER_RC): most trajectories carry no error at
        # all, so a single errored one moves the estimate by one quantum.  Individual
        # scatter points can therefore sit above a bound that is itself smaller than
        # that resolution.  Judge the bound on the mean, and say how coarse it is.
        resolution = 1.0 / (N_RC_REALIZATIONS * N_TRAJ_PER_RC)
        sem = float(tvd.std(ddof=1) / np.sqrt(tvd.size)) if tvd.size > 1 else 0.0
        if tvd.mean() - eps_qcap > max(2.0 * sem, TOL):
            print(
                f"    WARNING: N={n} d={repetition}: mean TVD {tvd.mean():.5f} "
                f"+/- {sem:.5f} exceeds the QCAP bound {eps_qcap:.5f} by more than "
                f"two standard errors -- the bound or the noise model is wrong"
            )
        elif tvd.max() > eps_qcap + TOL:
            above = int(np.count_nonzero(tvd > eps_qcap + TOL))
            note = " (below estimator resolution)" if eps_qcap < resolution else ""
            print(
                f"    note: {above}/{len(tvd)} TVD samples above QCAP={eps_qcap:.5f}; "
                f"mean {tvd.mean():.5f} is below it. "
                f"estimator resolution ~{resolution:.5f}{note}"
            )

        print(
            f"  d={repetition:>2} gates={twoq_gates:>5} layers={cycle_layers:>4} "
            f"S2={factors['s2']:.4g} Dtv(p,u)={factors['uniform_tvd']:.4f} "
            f"QCAP={eps_qcap:.4f} Renyi={model_bounds['renyi_bound']:.4f} "
            f"WN={model_bounds['wn_bound']:.4f} TVD={scatter_summary(tvd)}",
            flush=True,
        )

    tvd_path = plot_one_size(n, rows, np.asarray(tvd_columns, dtype=float))

    return {
        "n": n,
        "twoq_per_rep": twoq_per_rep,
        "n_distinct_cycles": len(distinct),
        "ro_fid": ro_fid,
        "ro_std": ro_std,
        "e_f_mean": float(e_f_values.mean()),
        "rows": rows,
        "tvd": np.asarray(tvd_columns, dtype=float),
        "tvd_plot": tvd_path,
    }


# =============================================================================
# Figures
# =============================================================================

_BOUND_STYLE = {
    "qcap": ("#0072B2", "-", "raw QCAP bound"),
    "renyi_bound": ("#CC79A7", "--", r"Renyi-2 bound  $\epsilon\,\min(1,S_2)$"),
    "wn_bound": ("#009E73", ":", r"white-noise model  $\epsilon\,D_{TV}(p,u)$"),
}


def plot_one_size(n: int, rows, tvd_columns) -> str:
    """TVD scatter against the three bounds, versus applied two-qubit gates."""
    x = np.asarray([row["twoq_gates"] for row in rows], dtype=int)
    qcap = np.asarray([row["qcap"] for row in rows], dtype=float)
    qcap_std = np.asarray([row["qcap_std"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for index, gates in enumerate(x):
        ax.plot(
            np.full(tvd_columns.shape[1], gates),
            tvd_columns[index],
            "o",
            ms=5.0,
            alpha=0.60,
            color="#333333",
            label="explicit-RC Z-basis TVD" if index == 0 else None,
        )
    for key, (color, style, label) in _BOUND_STYLE.items():
        ax.plot(
            x,
            [row[key] for row in rows],
            style,
            color=color,
            lw=2.2,
            marker="o",
            ms=4,
            label=label,
        )
    ax.fill_between(
        x,
        np.clip(qcap - QCAP_Z * qcap_std, 0.0, 1.0),
        np.clip(qcap + QCAP_Z * qcap_std, 0.0, 1.0),
        color=_BOUND_STYLE["qcap"][0],
        alpha=0.15,
        linewidth=0,
        label=f"QCAP CB uncertainty ({QCAP_Z:g}$\\sigma$)",
    )

    ax.set_xlabel("number of applied two-qubit gates")
    ax.set_ylabel("total variation distance / bound")
    ax.set_title(
        f"{ALGORITHM.upper()}-{n}: explicit-RC TVD versus the three bounds\n"
        "fixed noise; per-cycle CB census; every two-qubit gate independently dressed"
    )
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.15, which="both")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()

    path = OUT_TVD_TEMPLATE.format(algorithm=ALGORITHM, n=n)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _panel_grid(count, width=13.0, height=4.5):
    ncols = 2
    nrows = int(np.ceil(count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height * nrows))
    axes = np.atleast_1d(axes).ravel()
    return fig, axes


def plot_tvd_vs_n(ns, repetitions, tvd, qcap, qcap_std, renyi, wn):
    """One panel per repetition: measured TVD and the three bounds versus N."""
    fig, axes = _panel_grid(len(repetitions))

    for index, repetition in enumerate(repetitions):
        ax = axes[index]
        column = tvd[index]                              # (n_sizes, N_TVD_POINTS)
        for ni, n in enumerate(ns):
            ax.plot(
                np.full(column.shape[1], n),
                column[ni],
                "o",
                ms=3.5,
                alpha=0.45,
                color="#333333",
                label="explicit-RC TVD" if (ni == 0 and index == 0) else None,
            )
        ax.plot(ns, column.mean(axis=1), "-", lw=1.6, color="#333333",
                label="TVD mean" if index == 0 else None)

        for key, values in (
            ("qcap", qcap[index]),
            ("renyi_bound", renyi[index]),
            ("wn_bound", wn[index]),
        ):
            color, style, label = _BOUND_STYLE[key]
            ax.plot(ns, values, style, color=color, lw=2.0, marker="o", ms=4,
                    label=label if index == 0 else None)
        ax.fill_between(
            ns,
            np.clip(qcap[index] - QCAP_Z * qcap_std[index], 0.0, 1.0),
            np.clip(qcap[index] + QCAP_Z * qcap_std[index], 0.0, 1.0),
            color=_BOUND_STYLE["qcap"][0], alpha=0.15, linewidth=0,
        )

        ax.set_title(f"complete-circuit repetitions d={int(repetition)}")
        ax.set_xlabel("number of qubits N")
        ax.set_ylabel("TVD / bound")
        ax.set_xticks(list(ns))
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.2, which="both")

    for ax in axes[len(repetitions):]:
        ax.axis("off")
    fig.suptitle(
        f"{ALGORITHM.upper()} family: explicit-RC TVD and bounds versus width\n"
        "symlog vertical axis; bounds share one CB census per width"
    )
    # tight_layout is blind to figure-level legends, so reserve the strip first and
    # place the legend into it afterwards -- otherwise it lands on the x-axis label.
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.97))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9,
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005))
    fig.savefig(OUT_TVD_VS_N, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_s2_scaling(ns, repetitions, s2, fits):
    """Dedicated S_2-only scaling panels: exact ideal S_2 versus N, one panel per d."""
    fig, axes = _panel_grid(len(repetitions))
    grid = np.linspace(float(ns.min()), float(ns.max()), 400)

    for index, repetition in enumerate(repetitions):
        ax = axes[index]
        values = s2[index]
        positive = values > 0.0

        ax.plot(ns, values, "o", ms=6, color="#333333", label=r"exact ideal $S_2$")

        power = fits[int(repetition)]["power2"]
        if power["ok"]:
            ax.plot(
                grid,
                power["A"] * 2.0 ** (power["alpha"] * grid),
                "--", lw=1.8, color="#CC79A7",
                label=(
                    fr"$A2^{{\alpha N}}$: $\alpha={power['alpha']:.3f}"
                    fr"{_pm(power['alpha_err'])}$, $R^2={power['r2']:.3f}$"
                    + (" [flat]" if power["degenerate"] else "")
                ),
            )
        else:
            ax.text(0.03, 0.94, f"no power-law fit:\n{power['reason']}",
                    transform=ax.transAxes, va="top", fontsize=7.5, color="#B00020")

        entropy = fits[int(repetition)]["entropy"]
        if entropy["ok"]:
            ax.plot(
                grid,
                reconstruct_s2_from_delta_fit(grid, entropy),
                ":", lw=2.0, color="#0072B2",
                label=(
                    fr"$\Delta_2=aN+c$: $a={entropy['a']:.3f}"
                    fr"{_pm(entropy['a_err'])}$, $R^2={entropy['r2']:.3f}$"
                    + (" [flat]" if entropy["degenerate"] else "")
                ),
            )

        ax.plot(grid, deterministic_s2(grid), "-.", lw=1.3, color="#D55E00",
                label="deterministic-output reference")
        ax.plot(grid, haar_s2(grid), linestyle=(0, (4, 2, 1, 2)), lw=1.3,
                color="#009E73", label="finite-N Haar reference")

        ax.set_title(f"complete-circuit repetitions d={int(repetition)}")
        ax.set_xlabel("number of qubits N")
        ax.set_ylabel(r"unclipped Renyi factor $S_2$")
        ax.set_xticks(list(ns))
        # The deterministic reference reaches 2^(N/2), so a linear axis crushes every
        # real point against zero.  Log where the data allow it, symlog where S_2 = 0.
        if np.all(positive):
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.2, which="both")
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom, top * 4.0 if ax.get_yscale() == "log" else top * 1.6)
        ax.legend(fontsize=7.0, loc="upper left", framealpha=0.88, borderpad=0.5)

    for ax in axes[len(repetitions):]:
        ax.axis("off")
    fig.suptitle(
        f"{ALGORITHM.upper()} family: ideal-output Renyi-2 factor versus N\n"
        "S2 data are exact; fits use only the ideal algorithm distributions"
    )
    fig.tight_layout()
    fig.savefig(OUT_S2, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_fit_parameters(repetitions, fits):
    """Fitted S_2 growth rates versus repetition."""
    reps = np.asarray(repetitions, dtype=float)

    def column(kind, key):
        return np.asarray([
            fits[int(d)][kind].get(key, np.nan) for d in repetitions
        ], dtype=float)

    alpha, alpha_err = column("power2", "alpha"), column("power2", "alpha_err")
    a, a_err = column("entropy", "a"), column("entropy", "a_err")
    # A degenerate fit reports an infinite parameter error; errorbar cannot draw it.
    alpha_err = np.where(np.isfinite(alpha_err), alpha_err, 0.0)
    a_err = np.where(np.isfinite(a_err), a_err, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].errorbar(reps, alpha, yerr=alpha_err, marker="o", capsize=3, color="#CC79A7")
    axes[0].axhline(0.5, ls=":", lw=1.2, color="#D55E00",
                    label=r"deterministic $2^{N/2}$ rate")
    axes[0].set_ylabel(r"direct $S_2$ exponent $\alpha$")
    missing = ~np.isfinite(alpha)
    if missing.any():
        axes[0].text(
            0.03, 0.94,
            "no fit at d=" + ",".join(str(int(d)) for d in reps[missing]),
            transform=axes[0].transAxes, va="top", fontsize=8, color="#B00020",
        )

    axes[1].errorbar(reps, a, yerr=a_err, marker="o", capsize=3, color="#0072B2")
    axes[1].axhline(1.0, ls=":", lw=1.2, color="#D55E00",
                    label="zero collision-entropy density")
    axes[1].set_ylabel(r"collision-entropy deficit density $a$")

    for ax in axes:
        ax.set_xlabel("complete-circuit repetitions d")
        ax.set_xticks(list(repetitions))
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle(f"{ALGORITHM.upper()} $S_2$ fit parameters versus repetition")
    fig.tight_layout()
    fig.savefig(OUT_FIT_PARAMETERS, dpi=170, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

_ROW_FLOATS = (
    "qcap", "qcap_std", "wn_bound", "renyi_bound",
    "s2", "s2_clipped", "uniform_tvd", "delta2", "h2", "h2_density",
)
_ROW_INTS = ("twoq_gates", "cycle_layers", "support")


def main():
    results = [run_one_size(int(n)) for n in QUBIT_SIZES]
    ns = np.asarray([item["n"] for item in results], dtype=int)
    repetitions = np.asarray(ALGORITHM_REPETITIONS, dtype=int)
    shape = (len(repetitions), len(ns))

    grids = {key: np.empty(shape, dtype=float) for key in _ROW_FLOATS}
    grids.update({key: np.empty(shape, dtype=int) for key in _ROW_INTS})
    uniform = np.empty(shape, dtype=bool)
    tvd = np.empty((*shape, N_TVD_POINTS), dtype=float)

    for ni, item in enumerate(results):
        for di, row in enumerate(item["rows"]):
            for key in _ROW_FLOATS:
                grids[key][di, ni] = row[key]
            for key in _ROW_INTS:
                grids[key][di, ni] = row[key]
            uniform[di, ni] = row["uniform"]
            tvd[di, ni] = item["tvd"][di]

    fits = {
        int(repetition): {
            "power2": fit_s2_power2(ns, grids["s2"][di]),
            "entropy": fit_entropy_deficit(ns, grids["delta2"][di]),
        }
        for di, repetition in enumerate(repetitions)
    }

    print("\n=== S2 fits across N ===")
    for repetition in repetitions:
        power = fits[int(repetition)]["power2"]
        entropy = fits[int(repetition)]["entropy"]
        print(f"d={int(repetition)}:")
        if power["ok"]:
            flat = "  [S2 flat in N]" if power["degenerate"] else ""
            print(
                f"  direct S2: alpha={power['alpha']:.6g} +/- "
                f"{power['alpha_err']:.3g}, R2={power['r2']:.6f}{flat}"
            )
        else:
            print(f"  direct S2 fit skipped: {power['reason']}")
        if entropy["ok"]:
            flat = "  [Delta2 flat in N]" if entropy["degenerate"] else ""
            print(
                f"  Delta2: a={entropy['a']:.6g} +/- {entropy['a_err']:.3g}, "
                f"H2/N -> {entropy['h2_density']:.6g}, R2={entropy['r2']:.6f}{flat}"
            )
        else:
            print(f"  Delta2 fit skipped: {entropy['reason']}")

    print("\n=== bound tightness (mean over N) ===")
    print(f"{'d':>3}{'TVD':>12}{'WN':>12}{'Renyi':>12}{'QCAP':>12}{'QCAP/TVD':>11}")
    for di, repetition in enumerate(repetitions):
        tvd_mean = float(tvd[di].mean())
        qcap_mean = float(grids["qcap"][di].mean())
        # TVD is identically zero whenever the ideal output is uniform (plain QFT at
        # odd d), so the ratio is unbounded; keep it out of the column rather than
        # letting a 17-digit number run over the field width.
        ratio = qcap_mean / tvd_mean if tvd_mean > 0.0 else float("inf")
        ratio_text = (
            f"{ratio:>11.1f}" if np.isfinite(ratio) and ratio < 1e5 else f"{'--':>11}"
        )
        print(
            f"{int(repetition):>3}{tvd_mean:>12.3e}"
            f"{float(grids['wn_bound'][di].mean()):>12.3e}"
            f"{float(grids['renyi_bound'][di].mean()):>12.3e}"
            f"{qcap_mean:>12.3e}{ratio_text}"
        )

    plot_tvd_vs_n(
        ns, repetitions, tvd,
        grids["qcap"], grids["qcap_std"], grids["renyi_bound"], grids["wn_bound"],
    )
    plot_s2_scaling(ns, repetitions, grids["s2"], fits)
    plot_fit_parameters(repetitions, fits)

    save_results(
        OUT_FAMILY_DATA,
        algorithm=np.asarray(ALGORITHM),
        qubit_sizes=ns,
        repetitions=repetitions,
        s2=grids["s2"],
        s2_clipped=grids["s2_clipped"],
        uniform_tvd=grids["uniform_tvd"],
        collision_entropy_deficit=grids["delta2"],
        collision_entropy=grids["h2"],
        collision_entropy_density=grids["h2_density"],
        ideal_support=grids["support"],
        ideal_is_uniform=uniform,
        twoq_gates=grids["twoq_gates"],
        cycle_layers=grids["cycle_layers"],
        qcap_bound=grids["qcap"],
        qcap_bound_std=grids["qcap_std"],
        white_noise_bound=grids["wn_bound"],
        renyi_bound=grids["renyi_bound"],
        tvd_points=tvd,
        readout_fidelity=np.asarray([item["ro_fid"] for item in results]),
        n_distinct_cycles=np.asarray([item["n_distinct_cycles"] for item in results]),
        n_tvd_points=N_TVD_POINTS,
        n_rc_realizations=N_RC_REALIZATIONS,
        n_traj_per_rc=N_TRAJ_PER_RC,
    )

    print(f"\nWrote {OUT_TVD_VS_N}")
    print(f"Wrote {OUT_S2}")
    print(f"Wrote {OUT_FIT_PARAMETERS}")
    print(f"Wrote {OUT_FAMILY_DATA}")
    for item in results:
        print(f"Wrote {item['tvd_plot']}")


if __name__ == "__main__":
    main()
