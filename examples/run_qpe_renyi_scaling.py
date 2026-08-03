"""Renyi-2 scaling of programmatically generated N-qubit quantum phase estimation.

This script adapts the fixed-depth-versus-N analysis of
``examples/run_bounding_renyi_focused.py`` (which is NOT modified) from randomly
generated brickwork ansaetze to a family of QPE circuits built here in code, and
borrows the RC / cycle-benchmarking / QCAP conventions of
``examples/run_qasm_bounding_renyi.py`` and ``examples/run_qpe_bounding.py`` for the
optional noisy analysis. No QASM file is read.

THE CENTRAL QUANTITY
--------------------
For a QPE circuit with ``n_counting`` counting qubits, ``n_system`` system qubits and
repetition depth ``d``, with ``m`` MEASURED qubits and measured dimension D = 2^m,

    S2(N, d) = 0.5 * sqrt( 2^m * sum_x p_{N,d}(x)^2 - 1 )

where ``p_{N,d}`` is the exact ideal measured-register distribution of the repeated
target ``U_QPE^d``. Every collision / Renyi quantity uses the MEASURED dimension
D = 2^m, never 2^n_total, unless every qubit is measured.

WHAT IS DETERMINISTIC AND WHAT IS NOT
-------------------------------------
In fixed-phase mode there is exactly ONE circuit per ``(N, depth)``, so
every ideal-distribution quantity is a deterministic number: no instance standard
deviation, no error bars, no SEM-weighted fitting. Error bars appear only in two
places, and they mean different things:

  1. ``PHASE_ENSEMBLE = True`` draws ``N_PHASE_INSTANCES`` independent phases per
     point, which turns the ideal quantities into a genuine circuit ensemble with a
     standard deviation, a SEM, and percentile bands.
  2. ``RUN_NOISY_ANALYSIS = True`` adds explicit-RC replicate scatter of the MEASURED
     TVD, the QCAP parameter uncertainty, and the Monte Carlo trajectory floor. Those
     are noise-estimator quantities and are kept strictly separate from the ideal
     ones.

THE THREE MODELS FITTED AGAINST N AT EVERY FIXED DEPTH
------------------------------------------------------
    exponential       S2(N)     = A * 2^(alpha N)            phenomenological
    entropy deficit   Delta2(N) = a N + c,  Delta2 = log2(1 + 4 S2^2) = m - H2
                                                             exact identity, preferred
    Haar crossover    S2(N)     = S2_Haar(m(N)) + A exp(-b N)
                      with the finite-N Haar prediction S2_Haar = 0.5 sqrt((D-1)/(D+1))

The Haar crossover is an empirical crossover model, not a theorem, and a QPE circuit
whose S2 approaches 1/2 is NOT thereby Haar-random -- S2 is one statistic of the
output distribution, and the script reports several others (collision ratio to the
Haar value, KS distance of z = D p(x) from Exp(1), Var[z]) precisely so that no single
number is over-read.

Terminology used throughout: "Renyi factor" for S2, "counting-register size" for the
measured QPE width, "total circuit qubits" for counting + system, "repetition depth"
for repeated applications of the complete QPE circuit, "finite-N Haar prediction" for
the Porter-Thomas collision benchmark, and "exact under the global white-noise model"
for eps * D_TV(p, u).

Run:  python examples/run_qpe_renyi_scaling.py
"""

from __future__ import annotations

import math
import os
import random
import warnings
from collections import Counter
from typing import Dict, List, Sequence, Tuple

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector
from scipy.optimize import curve_fit, minimize_scalar
from scipy.stats import kstest

import _bootstrap  # noqa: F401
from proxysim import NoiseModel, save_results
from proxysim.backends import StatevectorBackend
from proxysim.backends.stabilizer import StabilizerBackend
from proxysim.benchmarking import cycle_benchmark, qcap_bound, readout_fidelity
from proxysim.circuit import Circuit
from proxysim.mqtbank import repeat
from proxysim.noise import apply_readout_to_distribution, sample_trajectory
from proxysim.parallel import pmap
from proxysim.rc import pauli_twirl


# =============================================================================
# Configuration
# =============================================================================

# -- QPE circuit family -------------------------------------------------------

# Eigenphase of the controlled unitary, in units of a full turn: U = P(2 pi phase).
# Deliberately NOT a dyadic rational such as 1/4: a dyadic phase is exactly
# representable by every large enough counting register and collapses the ideal
# distribution to a delta, which would make the Renyi factor trivial. pi/10 rounded to
# double precision is representable by no counting-register size used here, so the
# ideal distribution keeps a nontrivial finite-width structure at every N.
QPE_PHASE = 0.3141592653589793

# Number of counting (measured) qubits swept on the x axis.
N_COUNTING_SCALING = [2, 3, 4, 5, 6, 7, 8, 9]

# System/eigenstate qubits. The first one carries the controlled phase; any further
# system qubits are prepared in |1> as well and act as idle spectators, so they widen
# the register (and the noise budget) without changing the counting statistics.
N_SYSTEM_QUBITS = 1

# Repetition depths: the complete QPE circuit applied d times.
SCALING_DEPTHS = [1, 2, 4, 8, 16, 32, 64, 128]

# Which register is measured: "counting" (the QPE convention) or "all".
MEASURED_REGISTER = "counting"

# Append the inverse QFT to the counting register. False leaves the register in the
# QFT basis and is available as a control, not as the default.
INVERSE_QFT = True

# What "repetition depth" repeats.
#   "full" -- repeat the COMPLETE QPE circuit, eigenstate preparation included, which
#             is the literal target U_QPE^d. The preparation X gates are re-applied on
#             every repetition, so the ideal distribution of U^d is genuinely a new
#             distribution at every depth (and is recomputed as such).
#   "body" -- prepare the eigenstate once, then repeat only the algorithm body
#             (Hadamards, controlled powers, inverse QFT).
REPEAT_MODE = "full"

# X axis of the scaling plots: "counting" (default; the plotted N is the number of
# measured counting qubits) or "total" (counting + system).
N_AXIS = "counting"

# -- Phase ensemble (optional) ------------------------------------------------

# False -> one fixed phase, every ideal quantity deterministic, unweighted fits, no
# instance error bars anywhere. True (the default here) -> an ensemble of independently sampled phases
# per point, which is the ONLY mode in which instance error bars are drawn.
PHASE_ENSEMBLE = True
SCALING_INSTANCES = 40
N_PHASE_INSTANCES = SCALING_INSTANCES
SCALING_SEED = 90000
PHASE_SEED = SCALING_SEED
PHASE_RANGE = (0.0, 1.0)

# -- Noisy RC / CB / QCAP analysis (optional) ---------------------------------

RUN_NOISY_ANALYSIS = True

# The noisy analysis is orders of magnitude more expensive than the ideal sweep
# (every replicate is N_RC_REALIZATIONS * N_TRAJ statevector simulations of a
# depth-d circuit), so it runs on an explicit sub-grid of the ideal sweep. Points
# outside the sub-grid are stored as NaN, never silently dropped.
NOISY_N_COUNTING = list(N_COUNTING_SCALING)
NOISY_DEPTHS = list(SCALING_DEPTHS)

# Independent explicit-RC / trajectory replicates per noisy point.
N_INSTANCES = SCALING_INSTANCES

# Independently dressed (Pauli-twirled) circuits averaged inside ONE replicate.
N_RC_REALIZATIONS = 20

# Stochastic noise trajectories per dressed circuit.
N_TRAJ = 150

# Independent noisy-ensemble PAIRS used to estimate the Monte Carlo TVD floor.
FLOOR_PAIRS = 2

# Cycle-benchmark statistics.
CB_DEPTHS = [1, 2, 4, 8, 16]
CB_SHOTS = 4000
CB_DECAYS = 100

# Coherent residual-ZZ angle, rad. 0.0 -> purely Pauli-stochastic noise.
THETA_ZZ = 0.00

NOISE = NoiseModel(
    enabled=True,
    p1=1e-3,
    p2=1e-2,
    p_readout=1e-2,
    p_idle=1e-3,
    p_z1=0.0,
    p_zz=0.0,
    theta_1q=0.0,
    theta_zz=THETA_ZZ,
)

SEED = 42

# -- Analysis and resources ---------------------------------------------------

# Depths for which the Porter-Thomas summary figure is written.
PORTER_THOMAS_DEPTHS = [1, 4, 32]

# Leave-largest-N-out generalization test: how many of the largest N values to hold
# out. 0 disables the test.
HOLDOUT_SIZES = [1, 2]

# Skip any size whose dense statevector readout would exceed this budget.
SCALING_MEM_BUDGET_GB = 2.0

TOL = 1e-9  # tolerance for the bound-ordering assertions

COLORS = {
    "measured": "#0072B2",
    "qcap": "#D55E00",
    "exact": "#009E73",
    "renyi": "#CC79A7",
    "floor": "#7A5195",
    "haar": "0.35",
    "entropy": "#7A5195",
}

# -- Derived constants --------------------------------------------------------

if N_AXIS not in ("counting", "total"):
    raise ValueError(f"N_AXIS must be 'counting' or 'total', got {N_AXIS!r}")

if MEASURED_REGISTER not in ("counting", "all"):
    raise ValueError(
        f"MEASURED_REGISTER must be 'counting' or 'all', got {MEASURED_REGISTER!r}"
    )

if REPEAT_MODE not in ("full", "body"):
    raise ValueError(f"REPEAT_MODE must be 'full' or 'body', got {REPEAT_MODE!r}")

# m = x + M_OFFSET, where x is the plotted axis value. The measured dimension is
# always D = 2^m, so every Haar/collision expression evaluated on the fit axis has to
# go through this offset rather than assuming m == N.
if N_AXIS == "counting":
    M_OFFSET = 0 if MEASURED_REGISTER == "counting" else N_SYSTEM_QUBITS
else:
    M_OFFSET = -N_SYSTEM_QUBITS if MEASURED_REGISTER == "counting" else 0

N_AXIS_LABEL = (
    "counting-register size $N$" if N_AXIS == "counting"
    else "total circuit qubits $N$"
)

OUTPUT_SUFFIX = "_focused_sem"

OUT_DATA = f"{_bootstrap.RESULTS_DIR}/qpe_renyi_scaling_data{OUTPUT_SUFFIX}.npz"
OUT_ALL_DEPTHS = f"{_bootstrap.RESULTS_DIR}/qpe_renyi_factors_vs_N_all_depths{OUTPUT_SUFFIX}.png"
OUT_MODEL_COMPARISON = (
    f"{_bootstrap.RESULTS_DIR}/qpe_renyi_model_comparison_vs_depth{OUTPUT_SUFFIX}.png"
)


def renyi_factor_path(depth: int) -> str:
    """Figure A: Renyi factor versus N at one fixed repetition depth."""
    return f"{_bootstrap.RESULTS_DIR}/qpe_renyi_factors_vs_N_depth{int(depth)}{OUTPUT_SUFFIX}.png"


def collision_entropy_path(depth: int) -> str:
    """Figure B: collision entropy versus N at one fixed repetition depth."""
    return f"{_bootstrap.RESULTS_DIR}/qpe_collision_entropy_vs_N_depth{int(depth)}{OUTPUT_SUFFIX}.png"


def noisy_bounds_path(depth: int) -> str:
    """Figure E: measured TVD and the bounds versus N at one fixed depth."""
    return f"{_bootstrap.RESULTS_DIR}/qpe_tvd_exact_renyi_vs_N_depth{int(depth)}{OUTPUT_SUFFIX}.png"


def white_noise_path(depth: int) -> str:
    """Figure F: white-noise-model residuals versus N at one fixed depth."""
    return (
        f"{_bootstrap.RESULTS_DIR}/qpe_white_noise_residual_vs_N_depth{int(depth)}{OUTPUT_SUFFIX}.png"
    )


def porter_thomas_path(depth: int) -> str:
    """Porter-Thomas diagnostic summary versus N at one fixed depth."""
    return f"{_bootstrap.RESULTS_DIR}/qpe_porter_thomas_summary_depth{int(depth)}{OUTPUT_SUFFIX}.png"


_SB, _SV = StabilizerBackend(), StatevectorBackend()

CHECKED = {"factors": 0, "bounds": 0, "normalization": 0}


# =============================================================================
# 1. QPE circuit generation
# =============================================================================

def _wrapped_angle(turns: float) -> float:
    """Map a rotation given in turns onto the radian angle in [0, 2 pi).

    ``cp`` is 2 pi periodic, so reducing before multiplying keeps the controlled
    power ``U^(2^k)`` accurate at large k instead of forming ``2 pi phase 2^k``
    directly and losing the low bits to floating point.
    """
    return 2.0 * math.pi * (turns - math.floor(turns))


def inverse_qft_gates(circ: Circuit, qubits: Sequence[int]) -> Circuit:
    """Append the inverse QFT on ``qubits`` (textbook construction, with swaps).

    Conventions, since every downstream index depends on them: the forward QFT this
    inverts is

        QFT |x> = 2^{-m/2} sum_y exp(2 pi i x y / 2^m) |y>,

    with qubit ``k`` of the register holding bit ``k`` of both ``x`` and ``y`` (qubit 0
    is the least significant bit). The QPE phase-kickback state produced by the
    controlled powers below is exactly ``QFT |x>`` with ``x = phase * 2^m``, so the
    inverse QFT sends it to the computational-basis integer ``x``.
    """
    m = len(qubits)

    for i in range(m // 2):
        circ.swap(qubits[i], qubits[m - 1 - i])

    for j in range(m):
        for k in range(j):
            circ.add(
                "cp",
                qubits[k],
                qubits[j],
                params=(-math.pi / float(2 ** (j - k)),),
            )
        circ.h(qubits[j])

    return circ


def qpe_parts(
    n_counting_qubits: int,
    n_system_qubits: int = N_SYSTEM_QUBITS,
    phase: float = QPE_PHASE,
    measured_register: str = MEASURED_REGISTER,
    inverse_qft: bool = INVERSE_QFT,
):
    """Return ``(preparation, body, measured_qubits)`` for one QPE circuit.

    The split exists only so that ``REPEAT_MODE`` can choose whether eigenstate
    preparation is repeated along with the algorithm body. ``build_qpe_circuit``
    concatenates the two.

    Register layout: counting qubits ``0 .. n_counting-1``, system qubits
    ``n_counting .. n_counting+n_system-1``. The eigenunitary is the phase gate
    ``U = P(2 pi phase)`` acting on the FIRST system qubit, whose ``|1>`` eigenstate is
    prepared with an X. Counting qubit ``k`` applies the controlled power
    ``U^(2^k)`` as a single ``cp(2 pi phase 2^k)``, which is exactly ``2^k``
    repetitions of the controlled unitary and is the decomposition proxysim already
    supports natively.
    """
    if n_counting_qubits < 1:
        raise ValueError("n_counting_qubits must be at least 1")

    if n_system_qubits < 1:
        raise ValueError("n_system_qubits must be at least 1")

    n_total = n_counting_qubits + n_system_qubits
    counting = tuple(range(n_counting_qubits))
    system = tuple(range(n_counting_qubits, n_total))

    prep = Circuit(n_total, name=f"qpe{n_counting_qubits}_prep")

    # |1...1> on the system register: an eigenstate of P(theta) on every system qubit.
    for qubit in system:
        prep.x(qubit)

    body = Circuit(n_total, name=f"qpe{n_counting_qubits}")

    for qubit in counting:
        body.h(qubit)

    for k in counting:
        body.add(
            "cp",
            k,
            system[0],
            params=(_wrapped_angle(phase * (2 ** k)),),
        )

    if inverse_qft:
        inverse_qft_gates(body, counting)

    if measured_register == "counting":
        measured = counting
    elif measured_register == "all":
        measured = tuple(range(n_total))
    else:
        raise ValueError(
            f"measured_register must be 'counting' or 'all', got {measured_register!r}"
        )

    return prep, body, measured


def build_qpe_circuit(
    n_counting_qubits: int,
    n_system_qubits: int = N_SYSTEM_QUBITS,
    phase: float = QPE_PHASE,
    measured_register: str = MEASURED_REGISTER,
    inverse_qft: bool = INVERSE_QFT,
):
    """Return ``(circuit, measured_qubits)`` for one complete QPE circuit.

    ``circuit`` is eigenstate preparation followed by Hadamards on the counting
    register, the controlled powers of ``U = P(2 pi phase)``, and (by default) the
    inverse QFT on the counting register. Measurements are not part of the IR;
    ``measured_qubits`` names the register that is read out and marginalized onto.
    """
    prep, body, measured = qpe_parts(
        n_counting_qubits,
        n_system_qubits,
        phase,
        measured_register,
        inverse_qft,
    )

    circ = Circuit(prep.n_qubits, name=f"qpe{n_counting_qubits}")
    circ.gates = list(prep.gates) + list(body.gates)

    return circ, measured


def repeated_target(prep: Circuit, body: Circuit, depth: int) -> Circuit:
    """The depth-``d`` target circuit, per ``REPEAT_MODE``.

    ``"full"`` reproduces ``repeat(build_qpe_circuit(...), depth)`` exactly -- the
    complete QPE circuit, preparation included, applied ``depth`` times. ``"body"``
    prepares the eigenstate once and repeats only the algorithm body. Either way the
    ideal distribution is recomputed from the returned target; the depth-1
    distribution is never reused.
    """
    if REPEAT_MODE == "full":
        whole = Circuit(prep.n_qubits, name=body.name)
        whole.gates = list(prep.gates) + list(body.gates)
        return repeat(whole, int(depth))

    out = Circuit(prep.n_qubits, name=f"{body.name}_d{int(depth)}")
    out.gates = list(prep.gates) + list(body.gates) * int(depth)
    return out
# =============================================================================
# 2. Ideal distribution  [helpers adapted from run_bounding_renyi_focused.py]
# =============================================================================

def exact_probability_vector(circ: Circuit) -> np.ndarray:
    """Exact ideal full-register distribution on the cheapest capable backend.

    A Clifford circuit goes through the stabilizer backend, anything else through the
    dense statevector backend. Both return the sparse ``{bitstring: p}`` form with
    qubit 0 leftmost above their amplitude cutoff, EXCEPT that the statevector path is
    read directly out of the amplitude array here to avoid building 2^n label strings.
    """
    if circ.is_clifford:
        return dense_probs(_SB.exact_distribution(circ), circ.n_qubits)

    return np.asarray(
        Statevector(_SV._build(circ)).probabilities(),
        dtype=float,
    )


def dense_probs(dist: dict, n: int) -> np.ndarray:
    """Sparse ``{bitstring: p}`` (qubit 0 leftmost) -> dense vector of length 2^n.

    Outcomes missing from the dict are genuine zeros and must be filled in as such:
    both ``sum_x |p(x) - 1/D|`` and ``sum_x p(x)^2`` are sums over ALL 2^n outcomes.
    The bit at string position ``i`` belongs to qubit ``i``, and the flat index puts
    qubit 0 in the least significant bit, matching the little-endian statevector
    layout that :func:`marginalize_distribution` assumes.
    """
    p = np.zeros(2 ** n, dtype=float)

    for key, value in dist.items():
        if len(key) != n:
            raise ValueError(f"bitstring {key!r} is not {n} bits wide")

        p[int(key[::-1], 2)] = value

    total = float(p.sum())

    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"ideal distribution sums to {total:.9f}, not 1")

    return p / total


def marginalize_distribution(
    probability: np.ndarray,
    n_qubits: int,
    measured_qubits: Sequence[int],
) -> np.ndarray:
    """Marginalize a full distribution onto ``measured_qubits``.

    The flat vector follows the little-endian statevector layout, so reshaping to
    ``[2] * n`` maps array axis ``k`` to qubit ``n - 1 - k``. After the unmeasured
    axes are summed out, the surviving axes are ordered by decreasing qubit index, so
    the flat measured index carries measured qubit ``q`` in bit position ``q`` when
    the measured register is ``0 .. m-1``.
    """
    measured_set = set(int(q) for q in measured_qubits)

    drop_axes = tuple(
        n_qubits - 1 - qubit
        for qubit in range(n_qubits)
        if qubit not in measured_set
    )

    tensor = np.asarray(probability, dtype=float).reshape([2] * n_qubits)

    if drop_axes:
        tensor = tensor.sum(axis=drop_axes)

    return tensor.reshape(-1)


def check_normalized(
    probability: np.ndarray,
    where: str,
    atol: float = 1e-6,
) -> np.ndarray:
    """Validate that a probability vector is non-negative and sums to one."""
    probability = np.asarray(probability, dtype=float)

    total = float(probability.sum())

    if not np.isfinite(total) or abs(total - 1.0) > atol:
        raise ValueError(
            f"{where}: probability vector sums to {total:.12f}, not 1."
        )

    if probability.min() < -atol:
        raise ValueError(
            f"{where}: probability vector has a negative entry "
            f"{probability.min():.3e}."
        )

    CHECKED["normalization"] += 1

    return probability


def ideal_distribution(circ: Circuit, measured_qubits: Sequence[int]) -> np.ndarray:
    """Ideal measured-register distribution of ``circ``, as a dense 2^m vector.

    Always computed from the circuit passed in -- at every depth this is the repeated
    target, never the depth-1 circuit -- and marginalized onto the measured register
    so that every outcome of the measured subsystem is present, including the ones
    with probability zero.
    """
    full = exact_probability_vector(circ)

    return check_normalized(
        marginalize_distribution(full, circ.n_qubits, measured_qubits),
        f"ideal measured distribution of {circ.name}",
    )


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two dense probability vectors."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if p.shape != q.shape:
        raise ValueError(
            f"TVD between vectors of different dimension: {p.shape} vs {q.shape}."
        )

    return 0.5 * float(np.abs(p - q).sum())


def uniform_vector(m: int) -> np.ndarray:
    """The uniform distribution over the 2^m measured outcomes."""
    d = 2 ** m
    return np.full(d, 1.0 / d, dtype=float)


# =============================================================================
# 3. Renyi and collision-entropy quantities
# =============================================================================

def haar_collision_probability(m):
    """Finite-D Haar/Porter-Thomas mean collision probability, 2 / (D + 1)."""
    d = np.exp2(np.asarray(m, dtype=float))
    return 2.0 / (d + 1.0)


def haar_renyi_factor(m):
    """Finite-D Haar prediction for the unclipped Renyi factor.

    For a Haar-random pure state in dimension D = 2^m, ``E[sum_x p(x)^2] = 2/(D+1)``,
    so the collision-based factor is ``0.5 sqrt((D-1)/(D+1))``, which approaches 1/2
    from below as D grows. It is a prediction for one statistic of the output
    distribution, not a statement that any particular circuit is Haar-random.
    """
    d = np.exp2(np.asarray(m, dtype=float))
    return 0.5 * np.sqrt(np.maximum((d - 1.0) / (d + 1.0), 0.0))


def haar_renyi_factor_on_axis(x):
    """Finite-N Haar prediction evaluated on the plotted axis.

    The Haar value depends on the MEASURED dimension, so the axis coordinate is first
    mapped to ``m = x + M_OFFSET``. With ``N_AXIS = "total"`` and only the counting
    register measured this subtracts the system qubits, which is exactly the
    correction that makes the fixed-Haar curve comparable to the data.
    """
    return haar_renyi_factor(np.asarray(x, dtype=float) + M_OFFSET)


def ideal_factors(p: np.ndarray, m: int) -> dict:
    """Every ideal-distribution quantity for one measured distribution.

    ``m`` is the number of MEASURED qubits and D = 2^m throughout. The radicand
    ``D sum_x p(x)^2 - 1`` is clamped at zero before the square root: it is
    algebraically zero for a uniform p and floating point can put it a few ulps below.
    """
    p = np.asarray(p, dtype=float)

    d = 2 ** m
    u = 1.0 / d

    if p.size != d:
        raise ValueError(
            f"measured distribution has {p.size} entries, expected D = 2^{m} = {d}."
        )

    uniform_tvd = 0.5 * float(np.abs(p - u).sum())
    collision = float((p ** 2).sum())

    exp_d2 = d * collision                       # e^{D_2(p||u)} = D sum_x p(x)^2
    d2 = math.log(exp_d2) if exp_d2 > 0 else -math.inf
    s2 = 0.5 * math.sqrt(max(exp_d2 - 1.0, 0.0))

    # Exact identities from the collision probability:
    #   H_2 = -log2(sum_x p(x)^2),  Delta_2 = m - H_2 = log2(D sum_x p(x)^2).
    collision_entropy = -math.log2(collision) if collision > 0 else math.inf
    deficit = m - collision_entropy

    haar_collision = float(haar_collision_probability(m))
    haar_factor = float(haar_renyi_factor(m))

    # Scaled probabilities z = D p(x). Porter-Thomas predicts z ~ Exp(1), hence
    # mean 1 and variance 1; the KS statistic measures the distance from Exp(1).
    z = d * p
    ks = kstest(z, "expon")

    return {
        "uniform_tvd": uniform_tvd,
        "collision_probability": collision,
        "renyi2_divergence": d2,
        "renyi_sqrt_factor_unclipped": s2,
        "renyi_sqrt_factor_clipped": min(1.0, s2),
        "collision_entropy": collision_entropy,
        "collision_entropy_deficit": deficit,
        "collision_entropy_density": collision_entropy / m,
        "haar_collision_probability": haar_collision,
        "haar_renyi_factor": haar_factor,
        "collision_ratio_to_haar": collision / haar_collision,
        "renyi_factor_minus_haar": s2 - haar_factor,
        "mean_scaled_probability": float(z.mean()),
        "variance_scaled_probability": float(z.var()),
        "porter_thomas_ks_statistic": float(ks.statistic),
        "porter_thomas_ks_pvalue": float(ks.pvalue),
    }


def check_factors(factors: dict, where: str = "") -> None:
    """D_TV(p,u) <= min(1, S_2) -- the chi-squared / Renyi inequality."""
    lhs = factors["uniform_tvd"]
    rhs = factors["renyi_sqrt_factor_clipped"]

    if lhs > rhs + TOL:
        raise AssertionError(
            f"{where}: D_TV(p,u)={lhs:.12f} exceeds min(1, S_2)={rhs:.12f}"
        )

    CHECKED["factors"] += 1


# =============================================================================
# 4. White-noise output model
# =============================================================================

def white_noise_mixture(p: np.ndarray, epsilon: float, u: np.ndarray) -> np.ndarray:
    """q = (1 - epsilon) p + epsilon u, the global white-noise model."""
    return (1.0 - epsilon) * p + epsilon * u


def fit_white_noise_epsilon(
    q: np.ndarray,
    p: np.ndarray,
    u: np.ndarray,
) -> Tuple[float, float]:
    """Best-fit white-noise parameter and its residual.

    Returns ``(epsilon, D_TV(q, (1-epsilon) p + epsilon u))`` with epsilon minimizing
    the residual over [0, 1]. The objective is convex in epsilon (a TVD over an affine
    family), so the bounded Brent search is reliable; the endpoints are checked too.
    """
    def objective(epsilon: float) -> float:
        return tvd(q, white_noise_mixture(p, epsilon, u))

    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")

    best_eps = float(np.clip(result.x, 0.0, 1.0))
    best_res = float(objective(best_eps))

    for endpoint in (0.0, 1.0):
        value = objective(endpoint)

        if value < best_res:
            best_eps, best_res = endpoint, float(value)

    return best_eps, best_res


# =============================================================================
# 5. Deterministic self-tests
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    """Closed-form distributions, the finite-D Haar identity, and a QPE ground truth.

        uniform p:        D_TV(p,u) = 0,          S_2 = 0,                H_2 = m
        deterministic p:  D_TV(p,u) = 1 - 1/D,    S_2 = 0.5 sqrt(D - 1),  H_2 = 0
        Haar:             S_2^Haar  = 0.5 sqrt((D-1)/(D+1))

    The deterministic S_2 exceeds one for m >= 3 -- expected, and exactly why the
    bound uses the CLIPPED factor.

    The last block is the deterministic-phase test mode the fixed default phase
    deliberately avoids: at ``phase = 1/4`` (and 1/2, 3/8) the QPE output is exactly
    representable by a large enough counting register, so the ideal distribution must
    be a delta at the integer ``phase * 2^m``. That simultaneously validates the
    circuit construction, the inverse-QFT convention, and the marginalization index
    order -- none of which the closed-form checks above can see.
    """
    lines = []

    for m in (1, 2, 3, 5, 8):
        d = 2 ** m

        uni = uniform_vector(m)
        f = ideal_factors(check_normalized(uni, f"self-test uniform m={m}"), m)

        assert abs(f["uniform_tvd"]) < 1e-12, f
        assert abs(f["renyi_sqrt_factor_unclipped"]) < 1e-12, f
        assert abs(f["renyi2_divergence"]) < 1e-12, f
        assert abs(f["collision_entropy"] - m) < 1e-12, f
        assert abs(f["collision_probability"] - 1.0 / d) < 1e-15, f
        check_factors(f, f"uniform m={m}")

        det = np.zeros(d, dtype=float)
        det[0] = 1.0
        g = ideal_factors(check_normalized(det, f"self-test delta m={m}"), m)

        want_tvd = 1.0 - 1.0 / d
        want_s2 = 0.5 * math.sqrt(d - 1.0)

        assert abs(g["uniform_tvd"] - want_tvd) < 1e-12, (g, want_tvd)
        assert abs(g["renyi_sqrt_factor_unclipped"] - want_s2) < 1e-12, (g, want_s2)
        assert abs(g["renyi2_divergence"] - math.log(d)) < 1e-12, g
        assert abs(g["collision_entropy"]) < 1e-12, g
        check_factors(g, f"deterministic m={m}")

        # Finite-D Haar consistency, and the exact identity Delta_2 = log2(1 + 4 S2^2).
        assert abs(
            float(haar_renyi_factor(m)) - 0.5 * math.sqrt((d - 1.0) / (d + 1.0))
        ) < 1e-15, m
        assert abs(
            g["collision_entropy_deficit"]
            - math.log2(1.0 + 4.0 * g["renyi_sqrt_factor_unclipped"] ** 2)
        ) < 1e-12, g

        # The white-noise fit must recover eps on an exact synthetic mixture. The
        # tolerances are set by the bounded Brent search, not by the algebra.
        u = uniform_vector(m)
        eps_true = 0.3
        q = white_noise_mixture(det, eps_true, u)
        eps_fit, residual = fit_white_noise_epsilon(q, det, u)

        assert abs(eps_fit - eps_true) < 1e-4, (eps_fit, eps_true)
        assert residual < 1e-6, residual
        assert abs(tvd(q, det) - eps_true * want_tvd) < 1e-12, (q, det)

        lines.append(
            f"    m={m}: uniform -> D_TV=0, S_2=0, H_2={f['collision_entropy']:.3f}"
            f"   |   deterministic -> D_TV={g['uniform_tvd']:.6f} (=1-1/D), "
            f"S_2={g['renyi_sqrt_factor_unclipped']:.6f} (=0.5*sqrt(D-1)), "
            f"H_2={g['collision_entropy']:.3f}   |   "
            f"S_2^Haar={float(haar_renyi_factor(m)):.6f}"
        )

    qpe_lines = []

    for exact_phase in (0.25, 0.5, 0.375):
        for n_counting in (2, 3, 4):
            if not float(exact_phase * 2 ** n_counting).is_integer():
                continue

            circ, measured = build_qpe_circuit(
                n_counting,
                n_system_qubits=1,
                phase=exact_phase,
                measured_register="counting",
                inverse_qft=True,
            )

            p = ideal_distribution(circ, measured)
            want_index = int(round(exact_phase * 2 ** n_counting))

            assert int(np.argmax(p)) == want_index, (
                exact_phase, n_counting, int(np.argmax(p)), want_index)
            assert p[want_index] > 1.0 - 1e-9, (exact_phase, n_counting, p[want_index])

            f = ideal_factors(p, n_counting)
            check_factors(f, f"deterministic QPE phase={exact_phase} m={n_counting}")

            assert abs(
                f["renyi_sqrt_factor_unclipped"]
                - 0.5 * math.sqrt(2 ** n_counting - 1.0)
            ) < 1e-6, f

            qpe_lines.append(
                f"    phase={exact_phase}: m={n_counting} -> delta at index "
                f"{want_index} (= phase * 2^m), p={p[want_index]:.12f}"
            )

    if verbose:
        print("self-test: closed-form uniform / deterministic distributions")
        print("\n".join(lines))
        print("self-test: deterministic-phase QPE ground truth")
        print("\n".join(qpe_lines))
        print("    all assertions passed\n")

    return True
# =============================================================================
# 6. Fits versus N at fixed depth  [models from run_bounding_renyi_focused.py]
# =============================================================================

def _r2(y: np.ndarray, f: np.ndarray) -> float:
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((y - f) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _weighted_r2(y: np.ndarray, f: np.ndarray, sem: np.ndarray) -> float:
    """R^2 weighted by 1/sigma^2. NaN when no uncertainties are available.

    In the default fixed-phase mode the ideal values are deterministic, so there are
    no uncertainties to weight by and this is deliberately not computed.
    """
    if sem is None:
        return float("nan")

    w = 1.0 / np.maximum(np.asarray(sem, dtype=float), 1e-12) ** 2
    mean = float(np.sum(w * y) / np.sum(w))
    ss_tot = float(np.sum(w * (y - mean) ** 2))
    ss_res = float(np.sum(w * (y - f) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _information_criteria(y: np.ndarray, f: np.ndarray, k: int) -> Tuple[float, float]:
    """Gaussian-likelihood AIC and AICc for a least-squares fit with k parameters.

    AICc is undefined when ``n - k - 1 <= 0`` and is returned as NaN there rather than
    silently falling back to AIC.
    """
    n = int(y.size)
    rss = float(np.sum((y - f) ** 2))

    if n == 0 or rss <= 0.0:
        return float("nan"), float("nan")

    aic = n * math.log(rss / n) + 2.0 * k
    aicc = (
        aic + 2.0 * k * (k + 1.0) / (n - k - 1.0)
        if n - k - 1 > 0
        else float("nan")
    )

    return aic, aicc


def fit_exponential_renyi(x, y, sem=None) -> dict:
    """FIT 1: the phenomenological exponential Renyi-factor model S2(N) = A 2^(alpha N).

    Fitted in log space, exactly as ``fit_scaling`` in run_bounding_renyi_focused.py
    does. When ``sem`` is supplied (phase-ensemble mode only) the log-space sigma is
    propagated as ``sem / mean`` and the fit is weighted by it.

    This is a finite-N description of the sizes actually simulated. If the data
    flatten -- which they must, since S2 of a distribution over D outcomes is bounded
    by 0.5 sqrt(D-1) and the Haar value saturates at 1/2 -- a fitted alpha says
    nothing about the asymptotic behaviour, and the model comparison below exists
    precisely to expose that.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weighted = sem is not None
    sem = np.ones_like(y) if sem is None else np.asarray(sem, dtype=float)

    ok = np.isfinite(y) & (y > 0.0) & np.isfinite(sem) & (sem >= 0.0)
    xu, yu, su = x[ok], y[ok], sem[ok]

    fit = {
        "ok": bool(ok.sum() >= 3),
        "model": "exponential",
        "n_used": int(ok.sum()),
        "N_used": xu.astype(int),
        "dropped_N": x[~ok].astype(int).tolist(),
        "weighted_by_sem": weighted,
        "n_params": 2,
    }

    if not fit["ok"]:
        return fit

    ly = np.log(yu)
    sigma_log = np.maximum(su / yu, 1e-12) if weighted else None
    slope, intercept = np.polyfit(xu, ly, 1)

    par, cov = curve_fit(
        lambda z, log_a, alpha: log_a + alpha * math.log(2.0) * z,
        xu,
        ly,
        p0=[intercept, slope / math.log(2.0)],
        sigma=sigma_log,
        absolute_sigma=weighted,
        maxfev=20000,
    )

    err = np.sqrt(np.diag(cov))
    amplitude = math.exp(par[0])
    pred = amplitude * 2.0 ** (par[1] * xu)

    fit.update({
        "A": amplitude,
        "A_err": amplitude * float(err[0]),
        "alpha": float(par[1]),
        "alpha_err": float(err[1]),
        "r2": _r2(yu, pred),
        "r2_log": _r2(ly, np.log(pred)),
        "rmse": float(np.sqrt(np.mean((yu - pred) ** 2))),
        "cov": cov,
    })

    return fit


def predict_exponential(fit: dict, x) -> np.ndarray:
    """S2 predicted by the exponential fit at arbitrary axis values."""
    return fit["A"] * 2.0 ** (fit["alpha"] * np.asarray(x, dtype=float))


def fit_entropy_deficit(x, deficit, sem=None) -> dict:
    """FIT 2: the collision-entropy-deficit model Delta2(N) = a N + c.

    This is the preferred information-theoretic fit because

        Delta2 = log2(1 + 4 S2^2) = m - H2

    is an exact identity for every distribution, so fitting Delta2 and reconstructing

        S2(N) = 0.5 sqrt(2^(a N + c) - 1)

    involves no modelling beyond linearity of the deficit in N.

    ``h2_density = 1 - a`` is the projected collision-entropy density ONLY when the
    fit axis is the measured-register size. With ``N_AXIS = "total"`` and only the
    counting register measured, ``a`` is a deficit slope per TOTAL qubit and
    ``1 - a`` is not the measured collision-entropy density; the flag
    ``h2_density_valid`` records which case applies and the printed report repeats the
    caveat.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(deficit, dtype=float)
    weighted = sem is not None
    sem = np.ones_like(y) if sem is None else np.asarray(sem, dtype=float)

    ok = np.isfinite(y) & np.isfinite(sem) & (sem >= 0.0)
    xu, yu, su = x[ok], y[ok], sem[ok]

    fit = {
        "ok": bool(ok.sum() >= 3),
        "model": "entropy_deficit",
        "n_used": int(ok.sum()),
        "N_used": xu.astype(int),
        "dropped_N": x[~ok].astype(int).tolist(),
        "weighted_by_sem": weighted,
        "n_params": 2,
        "h2_density_valid": bool(M_OFFSET == 0),
    }

    if not fit["ok"]:
        return fit

    sigma = np.maximum(su, 1e-12) if weighted else None

    par, cov = curve_fit(
        lambda z, a, c: a * z + c,
        xu,
        yu,
        sigma=sigma,
        absolute_sigma=weighted,
        maxfev=20000,
    )

    err = np.sqrt(np.diag(cov))
    pred = par[0] * xu + par[1]

    fit.update({
        "a": float(par[0]),
        "a_err": float(err[0]),
        "c": float(par[1]),
        "c_err": float(err[1]),
        "h2_density": 1.0 - float(par[0]),
        "h2_density_err": float(err[0]),
        "r2": _r2(yu, pred),
        "rmse": float(np.sqrt(np.mean((yu - pred) ** 2))),
        "cov": cov,
    })

    return fit


def predict_entropy_deficit(fit: dict, x) -> np.ndarray:
    """S2 reconstructed from the fitted deficit: 0.5 sqrt(2^(aN+c) - 1)."""
    z = np.clip(fit["a"] * np.asarray(x, dtype=float) + fit["c"], -1024.0, 1024.0)
    return 0.5 * np.sqrt(np.maximum(np.exp2(z) - 1.0, 0.0))


def fit_haar_crossover(x, y, sem=None) -> dict:
    """FIT 3: approach to the finite-N Haar prediction, S2(N) = S2_Haar(N) + A e^{-bN}.

    ``A`` may be positive or negative so the data may approach the Haar curve from
    either side; ``b >= 0`` is enforced. The fixed Haar curve itself has no fitted
    parameters and is scored separately as ``fixed_haar_r2`` / ``fixed_haar_rmse``.

    This is an empirical crossover description, not a theorem, and agreement with it
    is not evidence that the circuit is Haar-random.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weighted = sem is not None
    sem = np.ones_like(y) if sem is None else np.asarray(sem, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(sem) & (sem >= 0.0)
    xu, yu, su = x[ok], y[ok], sem[ok]

    fit = {
        "ok": bool(ok.sum() >= 3),
        "model": "haar_crossover",
        "n_used": int(ok.sum()),
        "N_used": xu.astype(int),
        "dropped_N": x[~ok].astype(int).tolist(),
        "weighted_by_sem": weighted,
        "n_params": 2,
    }

    if not fit["ok"]:
        return fit

    haar = haar_renyi_factor_on_axis(xu)

    fit["fixed_haar_r2"] = _r2(yu, haar)
    fit["fixed_haar_rmse"] = float(np.sqrt(np.mean((yu - haar) ** 2)))
    fit["fixed_haar_weighted_r2"] = _weighted_r2(
        yu, haar, su if weighted else None)
    fit["fixed_haar_aic"], fit["fixed_haar_aicc"] = _information_criteria(
        yu, haar, 0)

    sigma = np.maximum(su, 1e-12) if weighted else None

    def model(z, amplitude, rate):
        return haar_renyi_factor_on_axis(z) + amplitude * np.exp(-rate * z)

    try:
        par, cov = curve_fit(
            model,
            xu,
            yu,
            p0=[float(yu[0] - haar[0]), 0.2],
            sigma=sigma,
            absolute_sigma=weighted,
            bounds=([-1e4, 0.0], [1e4, 10.0]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        fit["ok"] = False
        return fit

    err = np.sqrt(np.diag(cov))
    pred = model(xu, *par)

    fit.update({
        "A": float(par[0]),
        "A_err": float(err[0]),
        "b": float(par[1]),
        "b_err": float(err[1]),
        "r2": _r2(yu, pred),
        "rmse": float(np.sqrt(np.mean((yu - pred) ** 2))),
        "cov": cov,
    })

    return fit


def predict_haar_crossover(fit: dict, x) -> np.ndarray:
    """S2 predicted by the Haar-crossover fit at arbitrary axis values."""
    x = np.asarray(x, dtype=float)
    return haar_renyi_factor_on_axis(x) + fit["A"] * np.exp(-fit["b"] * x)


# Every model is fitted on its own natural coordinate but SCORED on the common S2
# scale, so that R^2 / RMSE / AIC are comparable across models.
MODEL_ORDER = ("exponential", "entropy_deficit", "haar_crossover", "fixed_haar")

MODEL_LABEL = {
    "exponential": r"exponential $A\,2^{\alpha N}$",
    "entropy_deficit": r"entropy deficit $\frac{1}{2}\sqrt{2^{aN+c}-1}$",
    "haar_crossover": r"Haar crossover $S_2^{\rm Haar}+Ae^{-bN}$",
    "fixed_haar": r"fixed finite-$N$ Haar (no free parameters)",
}


def fit_all_models(x, s2, deficit, s2_sem=None, deficit_sem=None) -> dict:
    """Fit every model versus N at one fixed depth and score them on the S2 scale."""
    fits = {
        "exponential": fit_exponential_renyi(x, s2, s2_sem),
        "entropy_deficit": fit_entropy_deficit(x, deficit, deficit_sem),
        "haar_crossover": fit_haar_crossover(x, s2, s2_sem),
    }

    fits["fixed_haar"] = {
        "ok": True,
        "model": "fixed_haar",
        "n_params": 0,
        "weighted_by_sem": s2_sem is not None,
    }

    return fits


def model_prediction(name: str, fits: dict, x) -> np.ndarray:
    """S2 predicted at ``x`` by one named model, or NaN if that fit failed."""
    fit = fits.get(name)

    if fit is None or not fit.get("ok"):
        return np.full(np.shape(x), np.nan, dtype=float)

    if name == "exponential":
        return predict_exponential(fit, x)

    if name == "entropy_deficit":
        return predict_entropy_deficit(fit, x)

    if name == "haar_crossover":
        return predict_haar_crossover(fit, x)

    if name == "fixed_haar":
        return np.asarray(haar_renyi_factor_on_axis(x), dtype=float)

    raise ValueError(f"unknown model {name!r}")


def score_models(x, s2, fits: dict, s2_sem=None) -> dict:
    """R^2, weighted R^2, RMSE, AIC and AICc for every model, on the S2 scale."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(s2, dtype=float)

    scores = {}

    for name in MODEL_ORDER:
        pred = model_prediction(name, fits, x)

        if not np.all(np.isfinite(pred)):
            scores[name] = {
                "r2": float("nan"),
                "weighted_r2": float("nan"),
                "rmse": float("nan"),
                "aic": float("nan"),
                "aicc": float("nan"),
            }
            continue

        k = int(fits[name].get("n_params", 0))
        aic, aicc = _information_criteria(y, pred, k)

        scores[name] = {
            "r2": _r2(y, pred),
            "weighted_r2": _weighted_r2(y, pred, s2_sem),
            "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
            "aic": aic,
            "aicc": aicc,
        }

    return scores


def holdout_test(x, s2, deficit, n_hold: int, s2_sem=None, deficit_sem=None) -> dict:
    """Leave-largest-N-out generalization test.

    Refits every model on all but the ``n_hold`` largest sizes and reports the RMSE of
    its prediction on the held-out points. In-sample R^2 is not sufficient for judging
    large-N predictability: a model can track the simulated range closely and still
    extrapolate badly, and this is the check that exposes it.
    """
    x = np.asarray(x, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    deficit = np.asarray(deficit, dtype=float)

    order = np.argsort(x)
    x, s2, deficit = x[order], s2[order], deficit[order]

    if s2_sem is not None:
        s2_sem = np.asarray(s2_sem, dtype=float)[order]

    if deficit_sem is not None:
        deficit_sem = np.asarray(deficit_sem, dtype=float)[order]

    out = {"n_hold": int(n_hold), "ok": False}

    if n_hold <= 0 or x.size - n_hold < 3:
        return out

    cut = x.size - n_hold
    train = slice(0, cut)
    test = slice(cut, x.size)

    fits = fit_all_models(
        x[train],
        s2[train],
        deficit[train],
        None if s2_sem is None else s2_sem[train],
        None if deficit_sem is None else deficit_sem[train],
    )

    out["ok"] = True
    out["train_N"] = x[train].astype(int)
    out["test_N"] = x[test].astype(int)

    for name in MODEL_ORDER:
        pred = model_prediction(name, fits, x[test])
        out[f"{name}_test_rmse"] = float(np.sqrt(np.mean((s2[test] - pred) ** 2)))
        out[f"{name}_test_pred"] = np.asarray(pred, dtype=float)

    return out
# =============================================================================
# 7. Resource planning  [spec section 19]
# =============================================================================

def dense_state_gb(n_qubits: int) -> float:
    """Peak memory of one complex128 statevector plus working copies, in GiB.

    ``2^n`` amplitudes x 16 bytes, with a factor 4 for the transient copies made
    while building and probability-squaring the state.
    """
    return (2 ** n_qubits) * 16 * 4 / 2 ** 30


def plan_sizes(counting_sizes, budget_gb: float, what: str) -> Tuple[List[int], List[int]]:
    """Split requested counting-register sizes into runnable and skipped.

    A size is skipped when its dense statevector would exceed ``budget_gb`` or the
    backend's exact-simulation limit. Skipped sizes are reported rather than silently
    dropped, so a truncated N range can never be mistaken for a converged sweep.
    """
    runnable, skipped = [], []

    for n_counting in counting_sizes:
        n_total = n_counting + N_SYSTEM_QUBITS
        need = dense_state_gb(n_total)
        limit = getattr(_SV, "max_exact_qubits", 30)

        if need > budget_gb or n_total > limit:
            skipped.append(int(n_counting))
            warnings.warn(
                f"skipping {what} n_counting={n_counting} "
                f"(n_total={n_total}, {need:.2f} GiB > budget {budget_gb:.2f} GiB "
                f"or beyond the backend limit of {limit} qubits).",
                RuntimeWarning,
            )
        else:
            runnable.append(int(n_counting))

    return runnable, skipped


# =============================================================================
# 8. Explicit randomized compiling  [source: run_qasm_bounding_renyi.py]
# =============================================================================

def explicitly_randomized_compile(circ: Circuit, seed: int):
    """Return ``(dressed circuit, virtual dressing-gate indices)``.

    ``mark_virtual=True`` is essential. The inserted Pauli frame operations must stay
    part of the circuit unitary but must NOT receive physical one-qubit noise, add
    physical layers, or change spectator-idle accounting -- on hardware they are
    compiled into the neighbouring single-qubit pulse.
    """
    result = pauli_twirl(circ, random.Random(seed), mark_virtual=True)

    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError(
            "pauli_twirl(mark_virtual=True) did not return (circuit, virtual); "
            f"got {type(result).__name__}."
        )

    dressed, virtual = result

    if not hasattr(dressed, "gates"):
        raise TypeError("pauli_twirl did not return a Circuit as its first element.")

    return dressed, tuple(sorted(int(index) for index in virtual))


def ideal_statevector(circ: Circuit) -> np.ndarray:
    """Noiseless full-register statevector amplitudes."""
    return np.asarray(Statevector(_SV._build(circ)).data, dtype=complex)


def state_fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """|<psi|phi>|^2, used to confirm the RC dressing is logically transparent."""
    return float(abs(np.vdot(psi, phi)) ** 2)


# =============================================================================
# 9. Cycle structure of the generated QPE circuits
# =============================================================================

# Algorithmic two-qubit gate -> Clifford entangler used to benchmark the noise it
# carries (the proxy convention of proxysim.cb_emit.PROXY_OF). The QPE circuits built
# here contain exactly two families: cp (controlled phase, from both the controlled-U
# ladder and the inverse QFT) and swap (from the inverse QFT bit reversal).
PROXY_ENTANGLER = {
    "cx": "CX",
    "cnot": "CX",
    "cz": "CZ",
    "cy": "CY",
    "swap": "SWAP",
    "cp": "CZ",
}

# How completely proxysim.rc.pauli_twirl can twirl each two-qubit gate family.
#   "full"       -- Clifford entangler, full 16-element two-qubit Pauli group.
#   "restricted" -- diagonal cp(theta) at a non-Clifford angle: only the commuting
#                   {I,Z}x{I,Z} subgroup, so the twirl tailors the error toward a
#                   Z-diagonal (dephasing) channel rather than a full Pauli channel.
TWIRL_CLASS = {
    "cx": "full",
    "cnot": "full",
    "cz": "full",
    "cy": "full",
    "swap": "full",
    "cp": "restricted",
}


def twoq_cycle_patterns(circ: Circuit) -> Counter:
    """Return ``Counter{cycle pattern: occurrences}`` for one circuit repetition.

    A cycle is one ASAP-parallel layer of two-qubit gates. The pattern key is the
    frozenset of ``(gate name, qubit pair)`` entries in that layer, so the QPE
    controlled-phase ladder and the inverse-QFT layers are benchmarked as the distinct
    cycles they are, never collapsed into a generic brickwork layer.
    """
    patterns: Counter = Counter()

    for layer in circ.layers():
        pairs = frozenset(
            (gate.name.lower(), tuple(gate.qubits))
            for gate in layer
            if len(gate.qubits) == 2
        )

        if pairs:
            patterns[pairs] += 1

    return patterns


def pattern_proxy(pattern) -> str:
    """Clifford proxy entangler used to benchmark one cycle pattern."""
    families = Counter(name for name, _ in pattern)
    unknown = sorted(set(families) - set(PROXY_ENTANGLER))

    if unknown:
        raise ValueError(
            f"No cycle-benchmark proxy is defined for two-qubit gate(s) {unknown}."
        )

    return PROXY_ENTANGLER[families.most_common(1)[0][0]]


def twirl_report(circ: Circuit):
    """Report how completely each two-qubit family in ``circ`` can be twirled.

    Returns ``(counts_by_family, restricted_families)``. In the QPE circuits the swap
    layer of the inverse QFT gets the full Pauli twirl the QCAP bound assumes, while
    every ``cp(theta)`` at a non-Clifford angle is twirled only over ``{I,Z}x{I,Z}``.
    """
    counts = Counter(
        gate.name.lower() for gate in circ.gates if len(gate.qubits) == 2
    )

    unknown = sorted(set(counts) - set(TWIRL_CLASS))

    if unknown:
        raise ValueError(
            f"pauli_twirl has no twirl construction registered for {unknown}."
        )

    restricted = sorted(
        name for name in counts if TWIRL_CLASS[name] == "restricted"
    )

    return counts, restricted


def clifford_cp_fraction(circ: Circuit) -> float:
    """Fraction of ``cp`` gates whose angle is a multiple of pi (hence Clifford).

    ``cp(k pi)`` is Clifford and does receive the full twirl, so this quantifies how
    much of the restricted-twirl caveat actually bites for a given circuit.
    """
    angles = [
        float(gate.params[0])
        for gate in circ.gates
        if gate.name.lower() == "cp" and gate.params
    ]

    if not angles:
        return float("nan")

    clifford = sum(
        1 for theta in angles
        if abs(theta / math.pi - round(theta / math.pi)) < 1e-12
    )

    return clifford / len(angles)


# =============================================================================
# 10. Noisy pipeline: CB, readout, QCAP  [source: run_qasm_bounding_renyi.py]
# =============================================================================

def measured_readout_noise(
    full_probability: np.ndarray,
    noise: NoiseModel,
    n_qubits: int,
    measured_qubits: Sequence[int],
) -> np.ndarray:
    """Apply readout flips only to the qubits that are actually measured.

    ``apply_readout_to_distribution`` flips every qubit of its input, so marginalize to
    the measured register first and apply the channel on the reduced vector. With the
    counting register measured and the system qubit traced out, charging readout error
    to the system qubit would inflate the measured-register noise.
    """
    measured_probability = marginalize_distribution(
        full_probability, n_qubits, measured_qubits
    )

    return apply_readout_to_distribution(
        measured_probability, noise, len(measured_qubits)
    )


def cb_cycle_worker(args):
    """Benchmark one two-qubit-layer proxy cycle (top level so pmap can pickle it)."""
    pairs, proxy, n_qubits, seed, n_decays = args

    return cycle_benchmark(
        [tuple(pair) for pair in pairs],
        n_qubits,
        CB_DEPTHS,
        NOISE,
        twoq=proxy,
        n_decays=n_decays,
        shots=CB_SHOTS,
        seed=seed,
    )


# Cycle benchmarking is the dominant cost of the noisy sub-grid, and under the
# homogeneous synthetic noise model e_F depends only on (proxy entangler, number of
# simultaneous two-qubit gates, register width). Both caches are therefore keyed on
# exactly those quantities and shared across depths and sizes; nothing about the
# per-pattern occurrence counts is cached, and those are always recounted from the
# actual target circuit.
_EF_CACHE: Dict[tuple, Tuple[float, float]] = {}
_RO_CACHE: Dict[int, Tuple[float, float]] = {}


def benchmark_cycle_patterns(circ: Circuit, patterns: Counter, n_measured: int,
                             n_workers: int):
    """Benchmark every distinct two-qubit cycle class the QCAP bound needs.

    Each distinct cycle pattern of ``circ`` is counted separately -- the QPE
    controlled-phase ladder, the QFT phase layers and the bit-reversal swap layer are
    different cycles and are benchmarked as such. Only the CB runs are cached (see
    ``_EF_CACHE``); the occurrence counts are exact for the circuit passed in and go
    to ``qcap_bound`` unchanged.
    """
    if n_measured not in _RO_CACHE:
        _RO_CACHE[n_measured] = readout_fidelity(n_measured, NOISE, seed=3)

    readout_fid, readout_std = _RO_CACHE[n_measured]

    representative = {}

    for pattern in patterns:
        key = (pattern_proxy(pattern), len(pattern), circ.n_qubits)

        if key not in _EF_CACHE:
            representative.setdefault(key, pattern)

    keys = sorted(representative)

    jobs = [
        (
            tuple(pair for _, pair in representative[key]),
            key[0],
            circ.n_qubits,
            100 + 7 * index,
            CB_DECAYS,
        )
        for index, key in enumerate(keys)
    ]

    if jobs:
        results = pmap(cb_cycle_worker, jobs, n_workers=n_workers)

        for key, result in zip(keys, results):
            _EF_CACHE[key] = (float(result["e_F"]), float(result["e_F_std"]))

    counts_one_rep: Dict[str, int] = {}
    cycle_efs: Dict[str, Tuple[float, float]] = {}
    cycle_labels: Dict[str, str] = {}

    for index, (pattern, count) in enumerate(patterns.items()):
        name = f"cycle_{index}"
        key = (pattern_proxy(pattern), len(pattern), circ.n_qubits)

        counts_one_rep[name] = int(count)
        cycle_efs[name] = _EF_CACHE[key]
        cycle_labels[name] = f"{key[1]}x{key[0]} " + ", ".join(
            f"{gate}{pair}" for gate, pair in sorted(pattern)
        )

    return counts_one_rep, cycle_efs, cycle_labels, readout_fid, readout_std


# =============================================================================
# 11. Explicit-RC replicate workers  [source: run_qasm_bounding_renyi.py]
# =============================================================================

def _noisy_measured_distribution(
    target: Circuit,
    noise: NoiseModel,
    measured_qubits: Sequence[int],
    n_rc_realizations: int,
    n_traj: int,
    seed: int,
    explicit_rc: bool,
) -> np.ndarray:
    """One noisy measured-register distribution.

    Averaging order:

        trajectory mean within each RC realization
        then mean over RC realizations
        then readout applied analytically on the measured register

    With ``explicit_rc=True`` a FRESH Pauli twirl is drawn per realization and the
    dressed circuit is simulated under the ORIGINAL physical noise model with the
    twirl gates marked virtual.
    """
    n_qubits = target.n_qubits
    average = np.zeros(2 ** len(measured_qubits), dtype=float)

    for rc_index in range(n_rc_realizations):
        rc_seed = seed + 1_000_003 * rc_index

        if explicit_rc:
            dressed, virtual_indices = explicitly_randomized_compile(target, rc_seed)
        else:
            dressed, virtual_indices = target, ()

        trajectory_rng = random.Random(rc_seed + 1)
        full_probability = np.zeros(2 ** n_qubits, dtype=float)

        for _ in range(n_traj):
            noisy_dressed = sample_trajectory(
                dressed, noise, trajectory_rng, virtual=virtual_indices
            )

            state = np.asarray(
                Statevector(_SV._build(noisy_dressed)).data, dtype=complex
            )

            full_probability += np.abs(state) ** 2

            del state

        full_probability /= n_traj

        average += measured_readout_noise(
            full_probability, noise, n_qubits, measured_qubits
        )

        del full_probability

    average /= n_rc_realizations

    return average


def rc_replicate_worker(args):
    """One RC/trajectory replicate: measured TVD plus the white-noise-model tests.

    Returns ``(measured_tvd, epsilon_white_noise_fit, residual_at_best_fit_eps,
    residual_at_eps_qcap)``. The last two test the FULL distributional white-noise
    model, not only its scalar TVD prediction.
    """
    (
        target,
        noise,
        ideal_measured,
        measured_qubits,
        n_rc_realizations,
        n_traj,
        seed,
        eps_qcap,
        explicit_rc,
    ) = args

    noisy = _noisy_measured_distribution(
        target, noise, measured_qubits, n_rc_realizations, n_traj, seed, explicit_rc
    )

    check_normalized(noisy, "noisy measured distribution", atol=1e-6)

    if noisy.shape != np.shape(ideal_measured):
        raise ValueError(
            "ideal and noisy measured-register dimensions differ: "
            f"{np.shape(ideal_measured)} vs {noisy.shape}."
        )

    u = uniform_vector(len(measured_qubits))

    residual_qcap = tvd(noisy, white_noise_mixture(ideal_measured, eps_qcap, u))
    eps_fit, residual_fit = fit_white_noise_epsilon(noisy, ideal_measured, u)

    return tvd(noisy, ideal_measured), eps_fit, residual_fit, residual_qcap


def floor_pair_worker(args):
    """TVD between two INDEPENDENT noisy ensembles of the same target circuit.

    This is the Monte Carlo estimator floor: with a finite number of RC realizations
    and trajectories, two independent estimates of the same noisy distribution already
    differ by this much. It is kept strictly separate from the CB/readout parameter
    uncertainty and from the replicate scatter of the measured TVD.
    """
    (
        target,
        noise,
        measured_qubits,
        n_rc_realizations,
        n_traj,
        seed_a,
        seed_b,
        explicit_rc,
    ) = args

    first = _noisy_measured_distribution(
        target, noise, measured_qubits, n_rc_realizations, n_traj, seed_a, explicit_rc
    )

    second = _noisy_measured_distribution(
        target, noise, measured_qubits, n_rc_realizations, n_traj, seed_b, explicit_rc
    )

    return tvd(first, second)


def scatter_summary(values) -> str:
    """Mean and range of one raw scatter column."""
    values = np.asarray(values, dtype=float)

    return f"{values.mean():.5f} [{values.min():.5f}, {values.max():.5f}]"


def validate_one_rc_realization(base_circ: Circuit) -> None:
    """Check RC logical equivalence and virtual-gate reporting on one QPE circuit."""
    dressed, virtual_indices = explicitly_randomized_compile(base_circ, SEED)

    fidelity = state_fidelity(
        ideal_statevector(base_circ), ideal_statevector(dressed)
    )

    original_twoq = sum(len(gate.qubits) == 2 for gate in base_circ.gates)
    dressed_twoq = sum(len(gate.qubits) == 2 for gate in dressed.gates)

    print(
        "explicit-RC smoke test:\n"
        f"    original gates:       {len(base_circ.gates)}\n"
        f"    dressed gates:        {len(dressed.gates)}\n"
        f"    original 2q gates:    {original_twoq}\n"
        f"    dressed 2q gates:     {dressed_twoq}\n"
        f"    virtual RC gates:     {len(virtual_indices)}\n"
        f"    ideal-state fidelity: {fidelity:.12f}\n"
    )

    if not virtual_indices:
        warnings.warn(
            "pauli_twirl reported no virtual-gate indices. Inserted Pauli dressing "
            "gates may be incorrectly charged physical one-qubit and idle noise.",
            RuntimeWarning,
        )

    if not np.isclose(fidelity, 1.0, atol=1e-9, rtol=1e-9):
        raise RuntimeError(
            "The RC-dressed circuit is not logically equivalent to the original QPE "
            "circuit; the twirl frame may not be closing."
        )
# =============================================================================
# 12. Ideal scaling sweep  [spec sections 3 and 16]
# =============================================================================

# Deterministic per-point ideal quantities, one value per (depth, N) -- or one value
# per (depth, N, phase instance) when the phase ensemble is enabled.
_IDEAL_KEYS = (
    "uniform_tvd",
    "collision_probability",
    "renyi2_divergence",
    "renyi_sqrt_factor_unclipped",
    "renyi_sqrt_factor_clipped",
    "collision_entropy",
    "collision_entropy_deficit",
    "collision_entropy_density",
    "haar_collision_probability",
    "haar_renyi_factor",
    "collision_ratio_to_haar",
    "renyi_factor_minus_haar",
    "mean_scaled_probability",
    "variance_scaled_probability",
    "porter_thomas_ks_statistic",
    "porter_thomas_ks_pvalue",
)


def axis_value(n_counting: int) -> int:
    """Plotted N for a given counting-register size."""
    return n_counting + (N_SYSTEM_QUBITS if N_AXIS == "total" else 0)


def measured_size(n_counting: int) -> int:
    """Number of MEASURED qubits m for a given counting-register size."""
    return n_counting + (N_SYSTEM_QUBITS if MEASURED_REGISTER == "all" else 0)


def phase_instances() -> np.ndarray:
    """The eigenphases to average over: one fixed phase, or the sampled ensemble."""
    if not PHASE_ENSEMBLE:
        return np.array([QPE_PHASE], dtype=float)

    rng = np.random.default_rng(PHASE_SEED)

    return rng.uniform(PHASE_RANGE[0], PHASE_RANGE[1], size=N_PHASE_INSTANCES)


def scaling_study():
    """The central sweep: ideal Renyi quantities versus N at each fixed depth.

    Returns ``(arrays, ideal_by_depth, meta)``. ``arrays`` holds ``(n_depths, n_sizes)``
    matrices for every quantity in ``_IDEAL_KEYS``; with the phase ensemble enabled it
    additionally holds the ``(n_depths, n_sizes, n_phase_instances)`` raw instance
    arrays and their standard errors. In the default fixed-phase mode there is exactly
    one instance per point and the values are deterministic, so no error bars exist.
    """
    sizes, skipped = plan_sizes(
        N_COUNTING_SCALING, SCALING_MEM_BUDGET_GB, "ideal sweep"
    )

    if not sizes:
        raise RuntimeError(
            "every requested counting-register size was skipped; raise "
            "SCALING_MEM_BUDGET_GB or lower N_COUNTING_SCALING."
        )

    depths = [int(d) for d in SCALING_DEPTHS]
    phases = phase_instances()

    n_depths, n_sizes, n_instances = len(depths), len(sizes), phases.size

    ensemble = {
        key: np.full((n_depths, n_sizes, n_instances), np.nan)
        for key in _IDEAL_KEYS
    }

    # Measured distribution at the largest simulated size, kept for the
    # Porter-Thomas panels. Only the first phase instance is retained.
    ideal_by_depth = {}

    print(
        f"ideal sweep: {n_depths} depths x {n_sizes} sizes x {n_instances} "
        f"phase instance(s)\n"
    )

    for i, depth in enumerate(depths):
        for j, n_counting in enumerate(sizes):
            m = measured_size(n_counting)

            for k, phase in enumerate(phases):
                prep, body, measured = qpe_parts(n_counting, phase=float(phase))
                target = repeated_target(prep, body, depth)

                probability = ideal_distribution(target, measured)

                factors = ideal_factors(probability, m)
                check_factors(factors, f"depth {depth}, n_counting {n_counting}")

                for key in _IDEAL_KEYS:
                    ensemble[key][i, j, k] = factors[key]

                if k == 0 and n_counting == sizes[-1] and depth in PORTER_THOMAS_DEPTHS:
                    ideal_by_depth[depth] = (n_counting, m, probability.copy())

                del probability, target

            print(
                f"    depth {depth:>3d}  N={axis_value(n_counting):>2d}  "
                f"m={m:>2d}  "
                f"S2={np.nanmean(ensemble['renyi_sqrt_factor_unclipped'][i, j]):.6f}  "
                f"D_TV(p,u)={np.nanmean(ensemble['uniform_tvd'][i, j]):.6f}  "
                f"H2={np.nanmean(ensemble['collision_entropy'][i, j]):.6f}"
            )

    arrays = {
        "depths": np.array(depths, dtype=int),
        "counting_sizes": np.array(sizes, dtype=int),
        "measured_sizes": np.array([measured_size(n) for n in sizes], dtype=int),
        "N_axis": np.array([axis_value(n) for n in sizes], dtype=float),
        "skipped_counting_sizes": np.array(skipped, dtype=int),
        "n_phase_instances": np.array(n_instances, dtype=int),
    }

    for key in _IDEAL_KEYS:
        arrays[key] = ensemble[key].mean(axis=2)

        if PHASE_ENSEMBLE:
            arrays[f"{key}_instances"] = ensemble[key]
            arrays[f"{key}_sem"] = (
                ensemble[key].std(axis=2, ddof=1) / math.sqrt(n_instances)
            )

    meta = {
        "sizes": sizes,
        "skipped": skipped,
        "depths": depths,
        "phases": phases,
        "n_instances": n_instances,
    }

    return arrays, ideal_by_depth, meta


def sem_of(arrays: dict, key: str):
    """Standard error for one quantity, or None in the deterministic fixed-phase mode.

    Fits and error bars consult this rather than fabricating uncertainties: with a
    single fixed phase the ideal values carry no sampling error at all.
    """
    return arrays.get(f"{key}_sem") if PHASE_ENSEMBLE else None


def fit_every_depth(arrays: dict, meta: dict) -> dict:
    """Fit and score all three models (plus fixed Haar) versus N at every depth."""
    x = arrays["N_axis"]
    results = {}

    for i, depth in enumerate(meta["depths"]):
        s2 = arrays["renyi_sqrt_factor_unclipped"][i]
        deficit = arrays["collision_entropy_deficit"][i]

        s2_sem = None if sem_of(arrays, "renyi_sqrt_factor_unclipped") is None else (
            arrays["renyi_sqrt_factor_unclipped_sem"][i]
        )
        deficit_sem = None if sem_of(arrays, "collision_entropy_deficit") is None else (
            arrays["collision_entropy_deficit_sem"][i]
        )

        fits = fit_all_models(x, s2, deficit, s2_sem, deficit_sem)
        scores = score_models(x, s2, fits, s2_sem)

        holdouts = {
            int(n_hold): holdout_test(x, s2, deficit, int(n_hold), s2_sem, deficit_sem)
            for n_hold in HOLDOUT_SIZES
        }

        results[depth] = {"fits": fits, "scores": scores, "holdouts": holdouts}

    return results


# =============================================================================
# 13. Noisy sub-grid: measured TVD against the QCAP and Renyi bounds
# =============================================================================

_NOISY_POINT_KEYS = (
    "qcap_bound",
    "qcap_bound_std",
    "qcap_fidelity",
    "measured_tvd",
    "measured_tvd_sem",
    "trajectory_floor",
    "exact_white_noise_prediction",
    "renyi_bound",
    "white_noise_residual_qcap",
    "white_noise_residual_fit",
    "white_noise_epsilon_fit",
    "twoq_cycles",
)


def noisy_study(arrays: dict, meta: dict, n_workers=None) -> dict:
    """Explicit-RC noisy analysis on the ``NOISY_DEPTHS`` x ``NOISY_N_COUNTING`` grid.

    Every quantity keeps the full ``(n_depths, n_sizes, ...)`` shape of the ideal
    sweep; points outside the sub-grid stay NaN so a partially covered grid can never
    be mistaken for a complete one.
    """
    depths, sizes = meta["depths"], meta["sizes"]
    n_depths, n_sizes = len(depths), len(sizes)

    noisy_sizes, noisy_skipped = plan_sizes(
        [n for n in NOISY_N_COUNTING if n in sizes],
        SCALING_MEM_BUDGET_GB,
        "noisy sub-grid",
    )

    out = {key: np.full((n_depths, n_sizes), np.nan) for key in _NOISY_POINT_KEYS}

    out["measured_tvd_points"] = np.full((n_depths, n_sizes, N_INSTANCES), np.nan)
    out["white_noise_epsilon_points"] = np.full(
        (n_depths, n_sizes, N_INSTANCES), np.nan)
    out["white_noise_residual_fit_points"] = np.full(
        (n_depths, n_sizes, N_INSTANCES), np.nan)
    out["white_noise_residual_qcap_points"] = np.full(
        (n_depths, n_sizes, N_INSTANCES), np.nan)
    out["trajectory_floor_points"] = np.full((n_depths, n_sizes, FLOOR_PAIRS), np.nan)
    out["noisy_counting_sizes"] = np.array(noisy_sizes, dtype=int)
    out["noisy_depths"] = np.array(
        [d for d in depths if d in NOISY_DEPTHS], dtype=int)
    out["noisy_skipped_counting_sizes"] = np.array(noisy_skipped, dtype=int)

    if not noisy_sizes or not out["noisy_depths"].size:
        warnings.warn(
            "the noisy sub-grid is empty; every noisy quantity stays NaN.",
            RuntimeWarning,
        )
        return out

    cycle_labels_seen: Dict[str, str] = {}

    print(
        f"\nnoisy sub-grid: depths {[int(d) for d in out['noisy_depths']]} x "
        f"counting sizes {noisy_sizes}\n"
    )

    for i, depth in enumerate(depths):
        if depth not in NOISY_DEPTHS:
            continue

        for j, n_counting in enumerate(sizes):
            if n_counting not in noisy_sizes:
                continue

            m = measured_size(n_counting)
            prep, body, measured = qpe_parts(n_counting)
            target = repeated_target(prep, body, depth)

            ideal_measured = ideal_distribution(target, measured)
            factors = ideal_factors(ideal_measured, m)

            patterns = twoq_cycle_patterns(target)

            counts, cycle_efs, labels, ro_fid, ro_std = benchmark_cycle_patterns(
                target, patterns, m, n_workers
            )

            cycle_labels_seen.update(labels)

            bound = qcap_bound(counts, cycle_efs, ro_fid, ro_std)
            eps = float(bound["error"])

            jobs = [
                (
                    target,
                    NOISE,
                    ideal_measured,
                    measured,
                    N_RC_REALIZATIONS,
                    N_TRAJ,
                    SEED + 10_007 * instance + 101 * depth + n_counting,
                    eps,
                    True,  # explicit randomized compiling
                )
                for instance in range(N_INSTANCES)
            ]

            replicates = pmap(rc_replicate_worker, jobs, n_workers=n_workers)

            measured_tvd = np.array([r[0] for r in replicates], dtype=float)
            eps_fit = np.array([r[1] for r in replicates], dtype=float)
            residual_fit = np.array([r[2] for r in replicates], dtype=float)
            residual_qcap = np.array([r[3] for r in replicates], dtype=float)

            floor_jobs = [
                (
                    target,
                    NOISE,
                    measured,
                    N_RC_REALIZATIONS,
                    N_TRAJ,
                    SEED + 500_003 * pair + 17 * depth + n_counting,
                    SEED + 900_007 * pair + 19 * depth + n_counting,
                    True,
                )
                for pair in range(FLOOR_PAIRS)
            ]

            floors = np.array(
                pmap(floor_pair_worker, floor_jobs, n_workers=n_workers), dtype=float
            )

            exact_prediction = eps * factors["uniform_tvd"]
            renyi_prediction = eps * factors["renyi_sqrt_factor_clipped"]

            # The two predictions must bracket in the order the theory gives, at every
            # single point: D_TV(p,u) <= min(1, S2) <= 1, all scaled by the same eps.
            if not (exact_prediction <= renyi_prediction + TOL <= eps + 2 * TOL):
                raise AssertionError(
                    f"bound ordering violated at depth {depth}, "
                    f"n_counting {n_counting}: exact={exact_prediction:.9f}, "
                    f"renyi={renyi_prediction:.9f}, qcap={eps:.9f}"
                )

            CHECKED["bounds"] += 1

            out["qcap_bound"][i, j] = eps
            out["qcap_bound_std"][i, j] = float(bound["std"])
            out["qcap_fidelity"][i, j] = float(bound["fidelity"])
            out["measured_tvd"][i, j] = measured_tvd.mean()
            out["measured_tvd_sem"][i, j] = (
                measured_tvd.std(ddof=1) / math.sqrt(measured_tvd.size)
                if measured_tvd.size > 1 else 0.0
            )
            out["trajectory_floor"][i, j] = floors.mean()
            out["exact_white_noise_prediction"][i, j] = exact_prediction
            out["renyi_bound"][i, j] = renyi_prediction
            out["white_noise_residual_qcap"][i, j] = residual_qcap.mean()
            out["white_noise_residual_fit"][i, j] = residual_fit.mean()
            out["white_noise_epsilon_fit"][i, j] = eps_fit.mean()
            out["twoq_cycles"][i, j] = float(sum(counts.values()))

            out["measured_tvd_points"][i, j] = measured_tvd
            out["white_noise_epsilon_points"][i, j] = eps_fit
            out["white_noise_residual_fit_points"][i, j] = residual_fit
            out["white_noise_residual_qcap_points"][i, j] = residual_qcap
            out["trajectory_floor_points"][i, j] = floors

            print(
                f"    depth {depth:>3d}  N={axis_value(n_counting):>2d}  "
                f"cycles={int(sum(counts.values())):>4d}  "
                f"eps_qcap={eps:.5f}  "
                f"TVD={measured_tvd.mean():.5f}  "
                f"exact={exact_prediction:.5f}  "
                f"renyi={renyi_prediction:.5f}  "
                f"floor={floors.mean():.5f}"
            )

            del ideal_measured, target

    out["cycle_labels"] = np.array(
        [f"{name}: {label}" for name, label in sorted(cycle_labels_seen.items())],
        dtype=object,
    )

    return out
# =============================================================================
# 14. Figures  [spec section 14]
# =============================================================================

def _fine_axis(x: np.ndarray) -> np.ndarray:
    """Dense axis grid for drawing fitted curves."""
    return np.linspace(float(np.min(x)), float(np.max(x)), 300)


def _errorbar_kwargs(sem):
    """Error bars only exist in phase-ensemble mode; otherwise draw plain markers."""
    return {} if sem is None else {"yerr": sem, "capsize": 3}


def _fit_caption(fits: dict, scores: dict) -> str:
    """Compact annotation of the three fits and their scores."""
    lines = []

    exp_fit = fits["exponential"]

    if exp_fit.get("ok"):
        lines.append(
            rf"exp: $\alpha$={exp_fit['alpha']:.4f}$\pm${exp_fit['alpha_err']:.4f}, "
            rf"$A$={exp_fit['A']:.4f}, $R^2$={scores['exponential']['r2']:.5f}"
        )

    def_fit = fits["entropy_deficit"]

    if def_fit.get("ok"):
        lines.append(
            rf"deficit: $a$={def_fit['a']:.4f}$\pm${def_fit['a_err']:.4f}, "
            rf"$c$={def_fit['c']:.4f}, "
            rf"$R^2$={scores['entropy_deficit']['r2']:.5f}"
        )

    haar_fit = fits["haar_crossover"]

    if haar_fit.get("ok"):
        lines.append(
            rf"Haar xover: $A$={haar_fit['A']:.4f}, "
            rf"$b$={haar_fit['b']:.4f}$\pm${haar_fit['b_err']:.4f}, "
            rf"$R^2$={scores['haar_crossover']['r2']:.5f}"
        )

    lines.append(
        rf"fixed Haar: $R^2$={scores['fixed_haar']['r2']:.5f} (0 parameters)"
    )

    return "\n".join(lines)


def figure_renyi_factor(depth, index, arrays, meta, result):
    """Figure A: the Renyi factor versus N at one fixed repetition depth."""
    x = arrays["N_axis"]
    xs = _fine_axis(x)

    s2 = arrays["renyi_sqrt_factor_unclipped"][index]
    dtv = arrays["uniform_tvd"][index]

    s2_sem = None if not PHASE_ENSEMBLE else (
        arrays["renyi_sqrt_factor_unclipped_sem"][index])
    dtv_sem = None if not PHASE_ENSEMBLE else arrays["uniform_tvd_sem"][index]

    fits, scores = result["fits"], result["scores"]

    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    ax.errorbar(
        x, s2, marker="o", ls="none", color=COLORS["renyi"],
        label=r"$S_2=\frac{1}{2}\sqrt{D\sum_x p(x)^2-1}$ (Renyi factor; ensemble mean $\pm$ SEM)",
        **_errorbar_kwargs(s2_sem),
    )

    ax.errorbar(
        x, dtv, marker="s", ls="none", color=COLORS["exact"], mfc="none",
        label=r"$D_{TV}(p,u)$ ensemble mean $\pm$ SEM", **_errorbar_kwargs(dtv_sem),
    )

    ax.plot(
        xs, haar_renyi_factor_on_axis(xs), color=COLORS["haar"], ls=":",
        label=r"finite-$N$ Haar $\frac{1}{2}\sqrt{(D-1)/(D+1)}$",
    )

    for name, style in (
        ("exponential", "-"),
        ("entropy_deficit", "--"),
        ("haar_crossover", "-."),
    ):
        if fits[name].get("ok"):
            ax.plot(
                xs, model_prediction(name, fits, xs), ls=style, lw=1.4,
                label=MODEL_LABEL[name],
            )

    ax.set_ylim(bottom=0)
    ax.set_xlabel(N_AXIS_LABEL)
    ax.set_ylabel(r"$S_2$ and $D_{TV}(p,u)$")
    ax.set_title(
        f"QPE Renyi factor versus size at fixed depth $d$={depth}\n"
        f"{N_PHASE_INSTANCES} phase instances per point; error bars are SEM"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="upper left")

    ax.text(
        0.98, 0.02, _fit_caption(fits, scores), transform=ax.transAxes,
        ha="right", va="bottom", fontsize=7,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )

    fig.tight_layout()
    path = renyi_factor_path(depth)
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return path


def figure_collision_entropy(depth, index, arrays, meta, result):
    """Figure B: collision entropy, its density, and the fitted deficit."""
    x = arrays["N_axis"]
    xs = _fine_axis(x)
    m = arrays["measured_sizes"].astype(float)

    h2 = arrays["collision_entropy"][index]
    density = arrays["collision_entropy_density"][index]
    deficit = arrays["collision_entropy_deficit"][index]

    fit = result["fits"]["entropy_deficit"]

    h2_sem = None if not PHASE_ENSEMBLE else arrays["collision_entropy_sem"][index]
    density_sem = None if not PHASE_ENSEMBLE else arrays["collision_entropy_density_sem"][index]
    deficit_sem = None if not PHASE_ENSEMBLE else arrays["collision_entropy_deficit_sem"][index]

    # Haar prediction for the collision entropy: H2 = -log2(2/(D+1)) = log2((D+1)/2).
    haar_h2 = -np.log2(haar_collision_probability(m))
    haar_h2_fine = -np.log2(haar_collision_probability(xs + M_OFFSET))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    left = axes[0]
    left.errorbar(x, h2, marker="o", color=COLORS["entropy"],
                  label=r"$H_2$ ensemble mean $\pm$ SEM",
                  **_errorbar_kwargs(h2_sem))
    left.plot(x, m, color="0.2", ls="--", lw=1.0, label=r"$m$ (maximum, uniform $p$)")
    left.plot(xs, haar_h2_fine, color=COLORS["haar"], ls=":",
              label=r"Haar $\log_2\frac{D+1}{2}$")
    left.set_xlabel(N_AXIS_LABEL)
    left.set_ylabel("collision entropy (bits)")
    left.set_title(f"$H_2$ versus size, depth $d$={depth}")
    left.grid(alpha=0.3)
    left.legend(fontsize=8)

    twin = left.twinx()
    twin.errorbar(x, density, marker="^", ms=4, ls="none", color=COLORS["measured"],
                  label=r"$H_2/m$ ensemble mean $\pm$ SEM",
                  **_errorbar_kwargs(density_sem))
    twin.plot(x, haar_h2 / m, ls=":", lw=1.0, color=COLORS["measured"], alpha=0.6)
    twin.set_ylabel(r"$H_2/m$ (density)", color=COLORS["measured"])
    twin.tick_params(axis="y", labelcolor=COLORS["measured"])
    twin.legend(fontsize=8, loc="lower right")

    right = axes[1]
    right.errorbar(x, deficit, marker="o", ls="none", color=COLORS["entropy"],
                   label=r"$\Delta_2=m-H_2$ ensemble mean $\pm$ SEM",
                   **_errorbar_kwargs(deficit_sem))

    if fit.get("ok"):
        right.plot(
            xs, fit["a"] * xs + fit["c"], color=COLORS["renyi"], lw=1.4,
            label=(
                rf"linear fit $a$={fit['a']:.4f}$\pm${fit['a_err']:.4f}, "
                rf"$c$={fit['c']:.4f}$\pm${fit['c_err']:.4f}"
            ),
        )

        note = (
            rf"$h_2 = 1-a$ = {fit['h2_density']:.4f}"
            if fit["h2_density_valid"]
            else "1 - a is NOT the collision-entropy density:\n"
                 f"the fit axis is total qubits, the deficit is per measured qubit"
        )

        right.text(
            0.02, 0.98, note, transform=right.transAxes, ha="left", va="top",
            fontsize=7.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
        )

    right.plot(x, m - haar_h2, color=COLORS["haar"], ls=":", label=r"Haar $\Delta_2$")
    right.set_xlabel(N_AXIS_LABEL)
    right.set_ylabel(r"$\Delta_2$ (bits)")
    right.set_title(f"collision-entropy deficit, depth $d$={depth}")
    right.grid(alpha=0.3)
    right.legend(fontsize=8)

    fig.tight_layout()
    path = collision_entropy_path(depth)
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return path
def _depth_axis(ax, depths) -> None:
    """Put a base-2 log depth axis on ``ax`` with explicit, always-positive limits.

    Setting the limits first matters: when every fit on a panel failed (too few sizes
    to fit at all) the plotted y values are all NaN, matplotlib finds no finite data
    limits, and a log axis over the default (0, 1) range raises.
    """
    depths = np.asarray(depths, dtype=float)

    ax.set_xscale("log", base=2)
    ax.set_xlim(float(depths.min()) * 0.8, float(depths.max()) * 1.25)


def figure_model_comparison(arrays, meta, results):
    """Figure C: fitted rates, fit quality, and held-out RMSE versus depth."""
    depths = np.array(meta["depths"], dtype=float)

    def collect(model, key):
        return np.array([
            results[int(d)]["fits"][model].get(key, np.nan)
            if results[int(d)]["fits"][model].get("ok") else np.nan
            for d in depths
        ], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))

    rates = axes[0, 0]
    rates.errorbar(depths, collect("exponential", "alpha"),
                   yerr=collect("exponential", "alpha_err"),
                   marker="o", capsize=3, color=COLORS["renyi"],
                   label=r"exponential $\alpha$")
    rates.errorbar(depths, collect("entropy_deficit", "a"),
                   yerr=collect("entropy_deficit", "a_err"),
                   marker="s", capsize=3, color=COLORS["entropy"],
                   label=r"deficit slope $a$")
    rates.errorbar(depths, collect("haar_crossover", "b"),
                   yerr=collect("haar_crossover", "b_err"),
                   marker="^", capsize=3, color=COLORS["measured"],
                   label=r"Haar-crossover rate $b$")
    rates.axhline(0.5, color="0.5", ls=":", lw=1.0,
                  label=r"$\alpha=1/2$ (delta-like $p$)")
    _depth_axis(rates, depths)
    rates.set_xlabel("repetition depth $d$")
    rates.set_ylabel("fitted rate")
    rates.set_title("fitted rates versus depth")
    rates.grid(alpha=0.3, which="both")
    rates.legend(fontsize=8)

    quality = axes[0, 1]

    for model, marker in zip(MODEL_ORDER, "os^v"):
        quality.plot(
            depths,
            [results[int(d)]["scores"][model]["r2"] for d in depths],
            marker=marker, label=MODEL_LABEL[model],
        )

    _depth_axis(quality, depths)
    quality.set_xlabel("repetition depth $d$")
    quality.set_ylabel(r"$R^2$ on the $S_2$ scale")
    quality.set_title(r"in-sample $R^2$ (all models scored on $S_2$)")
    quality.grid(alpha=0.3, which="both")
    quality.legend(fontsize=7.5)

    info = axes[1, 0]
    aicc = np.array([
        [results[int(d)]["scores"][model]["aicc"] for model in MODEL_ORDER]
        for d in depths
    ], dtype=float)

    best = np.nanmin(aicc, axis=1, keepdims=True)

    for column, (model, marker) in enumerate(zip(MODEL_ORDER, "os^v")):
        info.plot(depths, (aicc - best)[:, column], marker=marker,
                  label=MODEL_LABEL[model])

    _depth_axis(info, depths)
    info.set_yscale("symlog", linthresh=1.0)
    info.set_xlabel("repetition depth $d$")
    info.set_ylabel(r"$\Delta$AICc from the best model")
    info.set_title("AICc difference (lower is better; 0 marks the best model)")
    info.grid(alpha=0.3, which="both")
    info.legend(fontsize=7.5)

    heldout = axes[1, 1]
    drawn = False

    for n_hold, style in zip(HOLDOUT_SIZES, ("-", "--", ":")):
        for model, marker in zip(MODEL_ORDER, "os^v"):
            values = np.array([
                results[int(d)]["holdouts"][int(n_hold)].get(
                    f"{model}_test_rmse", np.nan)
                for d in depths
            ], dtype=float)

            if not np.any(np.isfinite(values) & (values > 0.0)):
                continue

            drawn = True
            heldout.plot(depths, values, ls=style, marker=marker, ms=4,
                         label=f"{model}, hold {n_hold}")

    _depth_axis(heldout, depths)

    if drawn:
        # A log axis with nothing positive on it raises rather than drawing an empty
        # panel, and an empty panel is exactly what a sweep with too few sizes to hold
        # any out produces.
        heldout.set_yscale("log")

    heldout.set_xlabel("repetition depth $d$")
    heldout.set_ylabel(r"test RMSE on held-out $S_2$")
    heldout.set_title("leave-largest-$N$-out prediction error")
    heldout.grid(alpha=0.3, which="both")

    if drawn:
        heldout.legend(fontsize=6.5, ncol=2)
    else:
        heldout.text(0.5, 0.5, "holdout disabled or too few sizes",
                     transform=heldout.transAxes, ha="center", va="center")

    fig.suptitle(
        "QPE Renyi-factor model comparison versus repetition depth\n"
        "all models scored on the $S_2$ scale; "
        f"({'SEM-weighted phase ensemble' if PHASE_ENSEMBLE else 'deterministic fixed phase'})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_MODEL_COMPARISON, dpi=160)
    plt.close(fig)

    return OUT_MODEL_COMPARISON


def figure_all_depths(arrays, meta):
    """Figure D: one Renyi-factor curve per depth on a single axis."""
    x = arrays["N_axis"]
    xs = _fine_axis(x)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    colormap = plt.get_cmap("viridis")
    n_depths = len(meta["depths"])

    for index, depth in enumerate(meta["depths"]):
        color = colormap(index / max(n_depths - 1, 1))

        s2_sem = None if not PHASE_ENSEMBLE else arrays["renyi_sqrt_factor_unclipped_sem"][index]
        deficit_sem = None if not PHASE_ENSEMBLE else arrays["collision_entropy_deficit_sem"][index]

        axes[0].errorbar(
            x, arrays["renyi_sqrt_factor_unclipped"][index],
            yerr=s2_sem, marker="o", ms=4, capsize=2, color=color,
            label=f"$d$={depth}",
        )

        axes[1].errorbar(
            x, arrays["collision_entropy_deficit"][index],
            yerr=deficit_sem, marker="o", ms=4, capsize=2, color=color,
            label=f"$d$={depth}",
        )

    axes[0].plot(xs, haar_renyi_factor_on_axis(xs), color="k", ls=":",
                 label="finite-$N$ Haar")
    axes[0].set_ylim(bottom=0)
    axes[0].set_xlabel(N_AXIS_LABEL)
    axes[0].set_ylabel(r"$S_2$")
    axes[0].set_title("Renyi factor versus size, every depth")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7.5, ncol=2)

    axes[1].plot(
        xs,
        np.log2(1.0 + 4.0 * haar_renyi_factor_on_axis(xs) ** 2),
        color="k", ls=":", label="finite-$N$ Haar",
    )
    axes[1].set_xlabel(N_AXIS_LABEL)
    axes[1].set_ylabel(r"$\Delta_2 = m - H_2$ (bits)")
    axes[1].set_title("collision-entropy deficit versus size, every depth")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7.5, ncol=2)

    fig.suptitle("QPE Rényi scaling across depths; error bars are SEM of the phase-ensemble mean")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_ALL_DEPTHS, dpi=160)
    plt.close(fig)

    return OUT_ALL_DEPTHS
def figure_noisy_bounds(depth, index, arrays, noisy):
    """Figure E: measured TVD against the QCAP and Renyi bounds, versus N."""
    x = arrays["N_axis"]

    measured = noisy["measured_tvd"][index]

    if np.all(np.isnan(measured)):
        return None

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    ax.errorbar(
        x, measured, yerr=noisy["measured_tvd_sem"][index],
        marker="o", capsize=3, color=COLORS["measured"],
        label=r"measured TVD ensemble mean $\pm$ SEM (explicit RC)",
    )

    ax.errorbar(
        x, noisy["qcap_bound"][index], yerr=noisy["qcap_bound_std"][index],
        marker="s", capsize=3, color=COLORS["qcap"],
        label=r"raw QCAP $\epsilon$",
    )

    ax.plot(x, noisy["renyi_bound"][index], marker="^", color=COLORS["renyi"],
            label=r"Renyi bound $\epsilon\,\min(1,S_2)$")

    ax.plot(x, noisy["exact_white_noise_prediction"][index], marker="v",
            color=COLORS["exact"],
            label=r"exact white-noise prediction $\epsilon\,D_{TV}(p,u)$")

    ax.plot(x, noisy["trajectory_floor"][index], marker="x", ls="--", ms=5,
            color=COLORS["floor"], label="Monte Carlo estimator floor")

    ax.set_ylim(bottom=0)
    ax.set_xlabel(N_AXIS_LABEL)
    ax.set_ylabel("total variation distance")
    ax.set_title(
        f"QPE noisy TVD and bounds versus size, depth $d$={depth}\n"
        r"ordering $\epsilon D_{TV}(p,u) \leq \epsilon\min(1,S_2) \leq \epsilon$ "
        "is asserted at every point"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = noisy_bounds_path(depth)
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return path


def figure_white_noise(depth, index, arrays, noisy):
    """Figure F: how well the white-noise output model describes the noisy data."""
    x = arrays["N_axis"]

    if np.all(np.isnan(noisy["white_noise_residual_qcap"][index])):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    left = axes[0]
    residual_qcap_sem = np.nanstd(
        noisy["white_noise_residual_qcap_points"][index], axis=-1, ddof=1
    ) / math.sqrt(N_INSTANCES)
    residual_fit_sem = np.nanstd(
        noisy["white_noise_residual_fit_points"][index], axis=-1, ddof=1
    ) / math.sqrt(N_INSTANCES)

    left.errorbar(x, noisy["white_noise_residual_qcap"][index],
                  yerr=residual_qcap_sem, marker="o", capsize=3,
                  color=COLORS["qcap"],
                  label=r"residual at $\epsilon_{QCAP}$ mean $\pm$ SEM")
    left.errorbar(x, noisy["white_noise_residual_fit"][index],
                  yerr=residual_fit_sem, marker="s", capsize=3,
                  color=COLORS["measured"],
                  label=r"residual at best-fit $\epsilon$ mean $\pm$ SEM")
    left.plot(x, noisy["trajectory_floor"][index], marker="x", ls="--", ms=5,
              color=COLORS["floor"], label="Monte Carlo estimator floor")
    left.set_yscale("log")
    left.set_xlabel(N_AXIS_LABEL)
    left.set_ylabel(r"$D_{TV}(q_{noisy},\,(1-\epsilon)p+\epsilon u)$")
    left.set_title(f"white-noise model residual, depth $d$={depth}")
    left.grid(alpha=0.3, which="both")
    left.legend(fontsize=8)

    right = axes[1]
    right.errorbar(
        x, noisy["qcap_bound"][index], yerr=noisy["qcap_bound_std"][index],
        marker="s", capsize=3, color=COLORS["qcap"], label=r"$\epsilon_{QCAP}$",
    )
    right.errorbar(
        x, noisy["white_noise_epsilon_fit"][index],
        yerr=np.nanstd(noisy["white_noise_epsilon_points"][index], axis=-1, ddof=1) / math.sqrt(N_INSTANCES),
        marker="o", capsize=3, color=COLORS["measured"],
        label=r"best-fit $\epsilon$ ensemble mean $\pm$ SEM",
    )
    right.set_xlabel(N_AXIS_LABEL)
    right.set_ylabel(r"$\epsilon$")
    right.set_title("fitted versus predicted white-noise weight")
    right.grid(alpha=0.3)
    right.legend(fontsize=8)

    fig.tight_layout()
    path = white_noise_path(depth)
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return path


def figure_porter_thomas(depth, index, arrays, ideal_by_depth):
    """Porter-Thomas diagnostics versus N at one depth, plus the z histogram.

    Agreement of any one of these statistics with its Haar value does NOT make the
    QPE output Haar-random; the panels are shown together for exactly that reason.
    """
    x = arrays["N_axis"]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))

    ratio = axes[0, 0]
    ratio.plot(x, arrays["collision_ratio_to_haar"][index], marker="o",
               color=COLORS["renyi"])
    ratio.axhline(1.0, color=COLORS["haar"], ls=":", label="Haar value")
    ratio.set_yscale("log")
    ratio.set_xlabel(N_AXIS_LABEL)
    ratio.set_ylabel(r"$\sum_x p^2 \;/\; \frac{2}{D+1}$")
    ratio.set_title("collision probability relative to Haar")
    ratio.grid(alpha=0.3, which="both")
    ratio.legend(fontsize=8)

    excess = axes[0, 1]
    excess.plot(x, arrays["renyi_factor_minus_haar"][index], marker="o",
                color=COLORS["measured"])
    excess.axhline(0.0, color=COLORS["haar"], ls=":", label="Haar value")
    excess.set_xlabel(N_AXIS_LABEL)
    excess.set_ylabel(r"$S_2 - S_2^{\rm Haar}$")
    excess.set_title("Renyi factor minus the finite-$N$ Haar prediction")
    excess.grid(alpha=0.3)
    excess.legend(fontsize=8)

    ks = axes[1, 0]
    ks.plot(x, arrays["porter_thomas_ks_statistic"][index], marker="o",
            color=COLORS["entropy"], label="KS statistic")
    ks.plot(x, arrays["porter_thomas_ks_pvalue"][index], marker="s", ls="--",
            color=COLORS["exact"], label="KS $p$-value")
    ks.set_yscale("log")
    ks.set_xlabel(N_AXIS_LABEL)
    ks.set_ylabel("KS against $\\mathrm{Exp}(1)$")
    ks.set_title(r"distance of $z=Dp(x)$ from Porter-Thomas")
    ks.grid(alpha=0.3, which="both")
    ks.legend(fontsize=8)

    spread = axes[1, 1]
    spread.plot(x, arrays["variance_scaled_probability"][index], marker="o",
                color=COLORS["renyi"], label=r"$\mathrm{Var}(z)$")
    spread.plot(x, arrays["mean_scaled_probability"][index], marker="s", ls="--",
                color=COLORS["measured"], label=r"$\langle z \rangle$ (exactly 1)")
    spread.axhline(1.0, color=COLORS["haar"], ls=":",
                   label="Porter-Thomas prediction")
    spread.set_yscale("log")
    spread.set_xlabel(N_AXIS_LABEL)
    spread.set_ylabel(r"moments of $z = D\,p(x)$")
    spread.set_title(r"mean and variance of the scaled probabilities")
    spread.grid(alpha=0.3, which="both")
    spread.legend(fontsize=8)

    entry = ideal_by_depth.get(depth)

    if entry is not None:
        n_counting, m, probability = entry
        z = (2 ** m) * probability

        inset = spread.inset_axes([0.55, 0.55, 0.42, 0.40])
        inset.hist(z, bins=40, density=True, color=COLORS["renyi"], alpha=0.6)
        grid = np.linspace(0.0, max(float(z.max()), 1e-12), 200)
        inset.plot(grid, np.exp(-grid), color="k", lw=1.0)
        inset.set_yscale("log")
        inset.set_title(f"$z$ at $N$={axis_value(n_counting)}", fontsize=7)
        inset.tick_params(labelsize=6)

    fig.suptitle(
        f"QPE Porter-Thomas diagnostics, depth $d$={depth} -- proximity of one "
        "statistic to its Haar value is not evidence of Haar randomness",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = porter_thomas_path(depth)
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return path
# =============================================================================
# 15. Saved output  [spec section 17]
# =============================================================================

def save_all(arrays, meta, results, noisy, ideal_by_depth):
    """Write results/qpe_renyi_scaling_data.npz."""
    sizes = np.array(meta["sizes"], dtype=int)

    payload = {
        "n_counting_values": sizes,
        "n_total_values": sizes + N_SYSTEM_QUBITS,
        "n_system_qubits": np.array(N_SYSTEM_QUBITS, dtype=int),
        "depths": arrays["depths"],
        "phase": np.array(QPE_PHASE, dtype=float),
        "phases": np.asarray(meta["phases"], dtype=float),
        "measured_qubit_counts": arrays["measured_sizes"],
        "measured_dimensions": (2 ** arrays["measured_sizes"]).astype(np.int64),
        "N_axis": arrays["N_axis"],
        "N_axis_kind": np.array(N_AXIS),
        "measured_register": np.array(MEASURED_REGISTER),
        "repeat_mode": np.array(REPEAT_MODE),
        "inverse_qft": np.array(INVERSE_QFT),
        "phase_ensemble": np.array(PHASE_ENSEMBLE),
        "n_phase_instances": arrays["n_phase_instances"],
        "skipped_counting_sizes": arrays["skipped_counting_sizes"],
        # ideal_ prefixes kept for the two keys the spec names that way; the
        # unprefixed spellings are stored as well so neither name is a surprise.
        "ideal_uniform_tvd": arrays["uniform_tvd"],
        "ideal_collision_probability": arrays["collision_probability"],
    }

    for key in _IDEAL_KEYS:
        payload[key] = arrays[key]

        if PHASE_ENSEMBLE:
            payload[f"{key}_instances"] = arrays[f"{key}_instances"]
            payload[f"{key}_sem"] = arrays[f"{key}_sem"]

    # Fit parameters and fit-quality metrics, tagged with their depth.
    for depth, result in results.items():
        for model, fit in result["fits"].items():
            for name in (
                "A", "A_err", "alpha", "alpha_err", "a", "a_err", "c", "c_err",
                "b", "b_err", "r2", "r2_log", "rmse", "h2_density",
                "fixed_haar_r2", "fixed_haar_rmse",
            ):
                if name in fit:
                    payload[f"fit_{model}_{name}_depth{depth}"] = np.array(
                        fit[name], dtype=float)

            payload[f"fit_{model}_ok_depth{depth}"] = np.array(bool(fit.get("ok")))

        for model, score in result["scores"].items():
            for name, value in score.items():
                payload[f"score_{model}_{name}_depth{depth}"] = np.array(
                    value, dtype=float)

        for n_hold, holdout in result["holdouts"].items():
            if not holdout.get("ok"):
                continue

            payload[f"holdout{n_hold}_test_N_depth{depth}"] = holdout["test_N"]

            for model in MODEL_ORDER:
                payload[f"holdout{n_hold}_{model}_test_rmse_depth{depth}"] = np.array(
                    holdout[f"{model}_test_rmse"], dtype=float)
                payload[f"holdout{n_hold}_{model}_test_pred_depth{depth}"] = holdout[
                    f"{model}_test_pred"]

    for depth, (n_counting, m, probability) in ideal_by_depth.items():
        payload[f"porter_thomas_probabilities_depth{depth}"] = probability
        payload[f"porter_thomas_n_counting_depth{depth}"] = np.array(
            n_counting, dtype=int)

    if noisy is not None:
        for key, value in noisy.items():
            payload[key] = value

        # The spec's names for the three replicate-resolved white-noise arrays.
        payload["epsilon_white_noise_fit"] = noisy["white_noise_epsilon_points"]
        payload["white_noise_model_residual_qcap"] = noisy[
            "white_noise_residual_qcap_points"]
        payload["white_noise_model_residual_fit"] = noisy[
            "white_noise_residual_fit_points"]

    save_results(OUT_DATA, **payload)

    return OUT_DATA


# =============================================================================
# 16. Diagnostics report
# =============================================================================

def report_diagnostics(arrays, meta, results, noisy):
    """Print the fit table, the model ranking, and the caveats that go with them."""
    print("\n" + "=" * 79)
    print("FIT SUMMARY: S2 versus N at fixed depth")
    print("=" * 79)

    header = (
        f"{'depth':>6} {'alpha':>9} {'alpha_err':>10} {'A':>10} "
        f"{'R2_exp':>9} {'a':>9} {'c':>9} {'h2_dens':>9} "
        f"{'b':>9} {'R2_haar':>9} {'R2_fixed':>9}"
    )
    print(header)
    print("-" * len(header))

    for depth in meta["depths"]:
        fits = results[depth]["fits"]
        scores = results[depth]["scores"]

        exp_fit = fits["exponential"]
        def_fit = fits["entropy_deficit"]
        haar_fit = fits["haar_crossover"]

        def value(fit, key):
            return fit.get(key, float("nan")) if fit.get("ok") else float("nan")

        print(
            f"{depth:>6d} "
            f"{value(exp_fit, 'alpha'):>9.4f} "
            f"{value(exp_fit, 'alpha_err'):>10.4f} "
            f"{value(exp_fit, 'A'):>10.4f} "
            f"{scores['exponential']['r2']:>9.5f} "
            f"{value(def_fit, 'a'):>9.4f} "
            f"{value(def_fit, 'c'):>9.4f} "
            f"{value(def_fit, 'h2_density'):>9.4f} "
            f"{value(haar_fit, 'b'):>9.4f} "
            f"{scores['haar_crossover']['r2']:>9.5f} "
            f"{scores['fixed_haar']['r2']:>9.5f}"
        )

    if not results[meta["depths"][0]]["fits"]["entropy_deficit"].get(
        "h2_density_valid", True
    ):
        print(
            "\nNOTE: N_AXIS is the TOTAL qubit count while only the counting register\n"
            "      is measured, so the deficit slope a is per total qubit and\n"
            "      h2_density = 1 - a is NOT the measured collision-entropy density."
        )

    print("\n" + "=" * 79)
    print("MODEL COMPARISON (scored on the S2 scale, unweighted"
          f"{', ensemble SEM available' if PHASE_ENSEMBLE else ''})")
    print("=" * 79)

    columns = (
        f"{'depth':>6} {'model':>16} {'R2':>9} {'wR2':>9} {'RMSE':>10} "
        f"{'AIC':>10} {'AICc':>10} {'dAICc':>9}"
    )
    print(columns)
    print("-" * len(columns))

    for depth in meta["depths"]:
        scores = results[depth]["scores"]
        aicc = [scores[model]["aicc"] for model in MODEL_ORDER]
        best = np.nanmin(aicc) if np.any(np.isfinite(aicc)) else float("nan")

        for model in MODEL_ORDER:
            score = scores[model]
            print(
                f"{depth:>6d} {model:>16} "
                f"{score['r2']:>9.5f} {score['weighted_r2']:>9.5f} "
                f"{score['rmse']:>10.3e} {score['aic']:>10.3f} "
                f"{score['aicc']:>10.3f} {score['aicc'] - best:>9.3f}"
            )

    for n_hold in HOLDOUT_SIZES:
        available = [
            depth for depth in meta["depths"]
            if results[depth]["holdouts"][int(n_hold)].get("ok")
        ]

        if not available:
            print(f"\nleave-largest-{n_hold}-out: not enough sizes to run the test.")
            continue

        held = [
            int(value)
            for value in results[available[0]]["holdouts"][int(n_hold)]["test_N"]
        ]

        print(f"\nLEAVE-LARGEST-{n_hold}-OUT TEST RMSE (held-out N = {held})")

        line = f"{'depth':>6} " + " ".join(f"{model:>16}" for model in MODEL_ORDER)
        print(line)
        print("-" * len(line))

        for depth in available:
            holdout = results[depth]["holdouts"][int(n_hold)]
            print(
                f"{depth:>6d} " + " ".join(
                    f"{holdout[f'{model}_test_rmse']:>16.3e}"
                    for model in MODEL_ORDER
                )
            )

    if noisy is not None and np.any(np.isfinite(noisy["measured_tvd"])):
        print("\n" + "=" * 79)
        print("NOISY SUB-GRID")
        print("=" * 79)

        line = (
            f"{'depth':>6} {'N':>4} {'cycles':>7} {'eps_qcap':>10} {'TVD':>10} "
            f"{'exact':>10} {'renyi':>10} {'floor':>10} {'res_qcap':>10} "
            f"{'res_fit':>10} {'eps_fit':>9}"
        )
        print(line)
        print("-" * len(line))

        for i, depth in enumerate(meta["depths"]):
            for j, n_counting in enumerate(meta["sizes"]):
                if not np.isfinite(noisy["measured_tvd"][i, j]):
                    continue

                print(
                    f"{depth:>6d} {axis_value(n_counting):>4d} "
                    f"{noisy['twoq_cycles'][i, j]:>7.0f} "
                    f"{noisy['qcap_bound'][i, j]:>10.5f} "
                    f"{noisy['measured_tvd'][i, j]:>10.5f} "
                    f"{noisy['exact_white_noise_prediction'][i, j]:>10.5f} "
                    f"{noisy['renyi_bound'][i, j]:>10.5f} "
                    f"{noisy['trajectory_floor'][i, j]:>10.5f} "
                    f"{noisy['white_noise_residual_qcap'][i, j]:>10.5f} "
                    f"{noisy['white_noise_residual_fit'][i, j]:>10.5f} "
                    f"{noisy['white_noise_epsilon_fit'][i, j]:>9.5f}"
                )

        print(
            "\nOrdering asserted at every noisy point: "
            "eps*D_TV(p,u) <= eps*min(1,S2) <= eps."
        )

    print(
        f"\nchecks passed: {CHECKED['factors']} Renyi-inequality, "
        f"{CHECKED['bounds']} bound-ordering, "
        f"{CHECKED['normalization']} normalization."
    )

    if meta["skipped"]:
        print(f"skipped counting sizes (memory budget): {meta['skipped']}")
# =============================================================================
# 17. Entry point
# =============================================================================

def describe_circuit_family() -> None:
    """Print the structure of the generated QPE circuits and the twirl caveats."""
    n_probe = max(N_COUNTING_SCALING)
    circ, measured = build_qpe_circuit(n_probe)

    counts, restricted = twirl_report(circ)
    patterns = twoq_cycle_patterns(circ)

    print(
        f"QPE family: phase={QPE_PHASE:.12f}, "
        f"n_system={N_SYSTEM_QUBITS}, inverse_qft={INVERSE_QFT}, "
        f"measured register={MEASURED_REGISTER}, repeat mode={REPEAT_MODE}\n"
        f"largest base circuit: {circ.summary()}\n"
        f"    measured qubits:    {list(measured)}\n"
        f"    2q gate families:   {dict(counts)}\n"
        f"    distinct 2q cycles: {len(patterns)} "
        f"({sum(patterns.values())} occurrences per repetition)\n"
        f"    Clifford cp share:  {clifford_cp_fraction(circ):.3f}"
    )

    if restricted:
        print(
            f"    NOTE: {restricted} are diagonal non-Clifford gates. pauli_twirl can\n"
            "          only twirl them over the commuting {I,Z}x{I,Z} subgroup, so the\n"
            "          RC channel is tailored toward Z-diagonal noise rather than the\n"
            "          full Pauli channel the QCAP bound assumes."
        )

    print()



# =============================================================================
# Focused SEM analysis: direct QPE analogue of run_bounding_renyi_focused_sem.py
# =============================================================================

PROJECTION_N_MAX = 32


def _mean_sd_sem(values):
    """Mean, instance standard deviation, and SEM along the last axis."""
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=-1)
    if values.shape[-1] <= 1:
        sd = np.zeros_like(mean)
        sem = np.zeros_like(mean)
    else:
        sd = np.nanstd(values, axis=-1, ddof=1)
        sem = sd / math.sqrt(values.shape[-1])
    return mean, sd, sem


def fit_scaling(ns, means, sem=None):
    """Focused-script exponential fits of ensemble-mean unclipped S2 versus N."""
    ns, means = np.asarray(ns, float), np.asarray(means, float)
    weighted = sem is not None
    sem = np.ones_like(means) if sem is None else np.asarray(sem, float)
    ok = np.isfinite(means) & (means > 0) & np.isfinite(sem) & (sem >= 0)
    x, y, sy = ns[ok], means[ok], sem[ok]
    out = {"ok": bool(ok.sum() >= 3), "N_used": x.astype(int),
           "dropped_N": ns[~ok].astype(int).tolist(), "weighted_by_sem": weighted}
    if not out["ok"]:
        return out
    ly = np.log(y)
    sigma_log = np.maximum(sy / y, 1e-12) if weighted else None
    slope, intercept = np.polyfit(x, ly, 1)

    def do_fit(model, p0):
        p, cov = curve_fit(model, x, ly, p0=p0, sigma=sigma_log,
                           absolute_sigma=weighted, maxfev=20000)
        e = np.sqrt(np.diag(cov))
        pred = np.exp(model(x, *p))
        return p, e, cov, _r2(y, pred), _r2(ly, model(x, *p))

    p, e, cov, r2, r2log = do_fit(
        lambda z, log_a, alpha: log_a + alpha * math.log(2.0) * z,
        [intercept, slope / math.log(2.0)],
    )
    out["pow2"] = {"A": math.exp(p[0]), "A_err": math.exp(p[0]) * e[0],
                   "alpha": p[1], "alpha_err": e[1], "r2": r2,
                   "r2_log": r2log, "cov": cov}
    return out


def fit_exact_scaling(ns, means, sem=None):
    """Focused-script saturation fit T(N)=L-A exp(-bN) for D_TV(p,u)."""
    ns, means = np.asarray(ns, float), np.asarray(means, float)
    weighted = sem is not None
    sem = np.ones_like(means) if sem is None else np.asarray(sem, float)
    ok = (np.isfinite(means) & (means >= 0) & (means <= 1)
          & np.isfinite(sem) & (sem >= 0))
    x, y, sy = ns[ok], means[ok], sem[ok]
    out = {"ok": bool(ok.sum() >= 3), "N_used": x.astype(int),
           "dropped_N": ns[~ok].astype(int).tolist(), "weighted_by_sem": weighted}
    if not out["ok"]:
        return out
    sigma = np.maximum(sy, 1e-12) if weighted else None

    def model(z, L, A, b):
        return L - A * np.exp(-b * z)

    L0 = min(1.0, max(float(np.nanmax(y)), 0.5))
    try:
        p, cov = curve_fit(model, x, y,
                           p0=[L0, max(L0 - y[0], 1e-3), 0.3],
                           sigma=sigma, absolute_sigma=weighted,
                           bounds=([0.0, 0.0, 0.0], [1.0, 2.0, 10.0]),
                           maxfev=20000)
    except (RuntimeError, ValueError):
        out["ok"] = False
        return out
    e = np.sqrt(np.diag(cov))
    out["satL"] = {"L": p[0], "L_err": e[0], "A": p[1], "A_err": e[1],
                   "b": p[2], "b_err": e[2], "r2": _r2(y, model(x, *p)),
                   "cov": cov}
    return out


def fit_collision_entropy_deficit_focused(ns, means, sem=None):
    """Fit Delta2=N-H2 to aN+c, weighted by SEM of the ensemble mean."""
    ns, means = np.asarray(ns, float), np.asarray(means, float)
    weighted = sem is not None
    sem = np.ones_like(means) if sem is None else np.asarray(sem, float)
    ok = np.isfinite(means) & np.isfinite(sem) & (sem >= 0)
    x, y, sy = ns[ok], means[ok], sem[ok]
    out = {"ok": bool(ok.sum() >= 3), "N_used": x.astype(int),
           "dropped_N": ns[~ok].astype(int).tolist(), "weighted_by_sem": weighted}
    if not out["ok"]:
        return out
    sigma = np.maximum(sy, 1e-12) if weighted else None
    p, cov = curve_fit(lambda z, a, c: a*z + c, x, y,
                       sigma=sigma, absolute_sigma=weighted, maxfev=20000)
    e = np.sqrt(np.diag(cov))
    pred = p[0]*x + p[1]
    out.update({"a": p[0], "a_err": e[0], "c": p[1], "c_err": e[1],
                "h2_density": 1.0-p[0], "h2_density_err": e[0],
                "r2": _r2(y, pred), "cov": cov})
    return out


def focused_fits(arrays, meta):
    """Return the same fit families used by the focused SEM brickwork analysis."""
    x = np.asarray(arrays["N_axis"], float)
    out = {}
    for di, depth in enumerate(meta["depths"]):
        s2 = arrays["renyi_sqrt_factor_unclipped"][di]
        exact = arrays["uniform_tvd"][di]
        deficit = arrays["collision_entropy_deficit"][di]
        s2sem = arrays.get("renyi_sqrt_factor_unclipped_sem")
        exsem = arrays.get("uniform_tvd_sem")
        desem = arrays.get("collision_entropy_deficit_sem")
        out[int(depth)] = {
            "s2": fit_scaling(x, s2, None if s2sem is None else s2sem[di]),
            "haar": fit_haar_crossover(x, s2, None if s2sem is None else s2sem[di]),
            "exact": fit_exact_scaling(x, exact, None if exsem is None else exsem[di]),
            "entropy": fit_collision_entropy_deficit_focused(
                x, deficit, None if desem is None else desem[di]),
        }
    return out


def noisy_study_focused(arrays, meta, n_workers=None):
    """QPE analogue of the focused per-instance depth-by-N noisy sweep.

    Each phase is one circuit instance.  Ideal factors, exact-WN predictions, Renyi
    bounds, and measured explicit-RC TVDs are all retained with shape
    (depth, N, SCALING_INSTANCES).  Consequently every plotted error bar is the SEM
    across the same QPE phase ensemble, exactly matching the focused script's
    statistical convention.  Raw QCAP remains one value (plus CB/readout uncertainty)
    per (depth,N).
    """
    depths, sizes = meta["depths"], meta["sizes"]
    phases = np.asarray(meta["phases"], float)
    nd, nn, ni = len(depths), len(sizes), len(phases)
    keys = ("measured_tvd", "exact_white_noise_bound", "renyi_bound",
            "uniform_tvd", "renyi_sqrt_factor_unclipped",
            "renyi_sqrt_factor_clipped", "renyi2_divergence",
            "collision_entropy_deficit", "collision_entropy",
            "collision_entropy_density")
    out = {k: np.full((nd, nn, ni), np.nan) for k in keys}
    out["bound"] = np.full((nd, nn), np.nan)
    out["bound_std"] = np.full((nd, nn), np.nan)
    out["trajectory_floor"] = np.full((nd, nn), np.nan)

    for di, depth in enumerate(depths):
        for nj, n_counting in enumerate(sizes):
            m = measured_size(n_counting)
            # Cycle structure is phase independent, so benchmark once per point.
            prep0, body0, measured0 = qpe_parts(n_counting, phase=float(phases[0]))
            target0 = repeated_target(prep0, body0, depth)
            patterns = twoq_cycle_patterns(target0)
            counts, efs, _, ro_fid, ro_std = benchmark_cycle_patterns(
                target0, patterns, m, n_workers)
            bound = qcap_bound(counts, efs, ro_fid, ro_std)
            eps = float(bound["error"])
            out["bound"][di, nj] = eps
            out["bound_std"][di, nj] = float(bound["std"])

            floor_vals = []
            for ii, phase in enumerate(phases):
                prep, body, measured = qpe_parts(n_counting, phase=float(phase))
                target = repeated_target(prep, body, depth)
                ideal = ideal_distribution(target, measured)
                factors = ideal_factors(ideal, m)
                seed = SCALING_SEED + 1_000_000*n_counting + 10_000*depth + ii
                tvd_value, _, _, _ = rc_replicate_worker((
                    target, NOISE, ideal, measured, N_RC_REALIZATIONS, N_TRAJ,
                    seed, eps, True))
                out["measured_tvd"][di, nj, ii] = tvd_value
                out["exact_white_noise_bound"][di, nj, ii] = eps*factors["uniform_tvd"]
                out["renyi_bound"][di, nj, ii] = eps*factors["renyi_sqrt_factor_clipped"]
                for key in ("uniform_tvd", "renyi_sqrt_factor_unclipped",
                            "renyi_sqrt_factor_clipped", "renyi2_divergence",
                            "collision_entropy_deficit", "collision_entropy",
                            "collision_entropy_density"):
                    out[key][di, nj, ii] = factors[key]
                if ii < FLOOR_PAIRS:
                    floor_vals.append(floor_pair_worker((
                        target, NOISE, measured, N_RC_REALIZATIONS, N_TRAJ,
                        seed + 50_000_003, seed + 90_000_007, True)))
            out["trajectory_floor"][di, nj] = np.mean(floor_vals) if floor_vals else np.nan
            print(f"  d={depth:>3} N={axis_value(n_counting):>2}: "
                  f"QCAP={eps:.4f}, TVD={np.nanmean(out['measured_tvd'][di,nj]):.4f}, "
                  f"exact={np.nanmean(out['exact_white_noise_bound'][di,nj]):.4f}, "
                  f"Renyi={np.nanmean(out['renyi_bound'][di,nj]):.4f}", flush=True)
    return out


def focused_bounds_figure(depths, ns, sc, fits):
    """One four-panel focused QPE analysis figure for every fixed depth."""
    outputs = []
    x = np.asarray(ns, float)
    for di, depth in enumerate(depths):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)

        ax = axes[0, 0]
        for key, label, marker, color in (
            ("measured_tvd", "Measured TVD", "o", COLORS["measured"]),
            ("exact_white_noise_bound", "Exact-WN prediction", "^", COLORS["exact"]),
            ("renyi_bound", "Renyi-2 bound", "s", COLORS["renyi"]),
        ):
            mean, _, sem = _mean_sd_sem(sc[key][di])
            ax.errorbar(x, mean, yerr=sem, fmt=marker+"-", color=color,
                        capsize=3, lw=1.8, ms=5,
                        label=label + r" ensemble mean $\pm$ SEM")
        ax.errorbar(x, sc["bound"][di], yerr=sc["bound_std"][di], fmt="--",
                    color=COLORS["qcap"], capsize=3, lw=1.7,
                    label=r"Raw QCAP ($\pm1\sigma$ CB/readout)")
        ax.plot(x, sc["trajectory_floor"][di], ":", color=COLORS["floor"],
                label="trajectory estimator floor")
        ax.set_title("QPE phase ensemble: TVD and bounds")
        ax.set_ylabel("TVD / bound")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, fontsize=7.8)

        ax = axes[0, 1]
        s2m, _, s2sem = _mean_sd_sem(sc["renyi_sqrt_factor_unclipped"][di])
        exm, _, exsem = _mean_sd_sem(sc["uniform_tvd"][di])
        ax.errorbar(x, s2m, yerr=s2sem, fmt="o", color=COLORS["renyi"], capsize=3,
                    label=r"$S_2$ ensemble mean $\pm$ SEM")
        ax.errorbar(x, exm, yerr=exsem, fmt="^", color=COLORS["exact"], capsize=3,
                    label=r"$D_{TV}(p,u)$ ensemble mean $\pm$ SEM")
        g = np.linspace(x.min(), x.max(), 300)
        fs2 = fits[int(depth)]["s2"]
        if fs2.get("ok"):
            f = fs2["pow2"]
            ax.plot(g, f["A"]*2.0**(f["alpha"]*g), "--", color=COLORS["renyi"],
                    label=fr"$A2^{{\alpha N}}$: $\alpha={f['alpha']:.3f}\pm{f['alpha_err']:.3f}$, $R^2={f['r2']:.3f}$")
        ax.plot(g, haar_renyi_factor_on_axis(g), "-.", color=COLORS["haar"],
                label=r"finite-$N$ Haar prediction")
        fh = fits[int(depth)]["haar"]
        if fh.get("ok"):
            ax.plot(g, predict_haar_crossover(fh, g), linestyle=(0,(5,2,1,2)),
                    color=COLORS["entropy"],
                    label=fr"Haar crossover: $b={fh['b']:.3f}$, $R^2={fh['r2']:.3f}$")
        fe = fits[int(depth)]["exact"]
        if fe.get("ok"):
            f = fe["satL"]
            ax.plot(g, f["L"]-f["A"]*np.exp(-f["b"]*g), ":", color=COLORS["exact"],
                    label=fr"$L-Ae^{{-bN}}$: $L={f['L']:.2f}$, $R^2={f['r2']:.3f}$")
        ax.set_title("Rényi and exact factors with fits")
        ax.set_ylabel("factor")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, fontsize=7.3)

        ax = axes[1, 0]
        dm, _, dsem = _mean_sd_sem(sc["collision_entropy_deficit"][di])
        ax.errorbar(x, dm, yerr=dsem, fmt="o", color=COLORS["renyi"], capsize=3,
                    label=r"$\Delta_2=N-H_2$ ensemble mean $\pm$ SEM")
        fent = fits[int(depth)]["entropy"]
        nproj = np.linspace(x.min(), PROJECTION_N_MAX, 500)
        if fent.get("ok"):
            ax.plot(nproj, fent["a"]*nproj+fent["c"], "--", color=COLORS["renyi"],
                    label=fr"$\Delta_2=aN+c$: $a={fent['a']:.3f}\pm{fent['a_err']:.3f}$")
            ax.axvspan(x.max(), PROJECTION_N_MAX, color="0.5", alpha=0.08,
                       label="extrapolation region")
        ax.set_title("Collision-entropy-deficit projection")
        ax.set_xlabel(N_AXIS_LABEL)
        ax.set_ylabel(r"$\Delta_2$ (bits)")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, fontsize=7.8)

        ax = axes[1, 1]
        ax.errorbar(x, exm, yerr=exsem, fmt="^", color=COLORS["exact"], capsize=3,
                    label=r"$D_{TV}(p,u)$ ensemble mean $\pm$ SEM")
        if fe.get("ok"):
            f = fe["satL"]
            ax.plot(nproj, f["L"]-f["A"]*np.exp(-f["b"]*nproj), ":",
                    color=COLORS["exact"],
                    label=fr"$L-Ae^{{-bN}}$: $L={f['L']:.2f}\pm{f['L_err']:.2f}$")
            ax.axvspan(x.max(), PROJECTION_N_MAX, color="0.5", alpha=0.08,
                       label="extrapolation region")
        ax.set_title("Exact white-noise-factor projection")
        ax.set_xlabel(N_AXIS_LABEL)
        ax.set_ylabel(r"$D_{TV}(p,u)$")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, fontsize=7.8)

        fig.suptitle(f"QPE fixed repetition depth {int(depth)}: ensemble means versus N\n"
                     f"{SCALING_INSTANCES} phase instances per point; error bars = SEM")
        fig.tight_layout()
        path = (_bootstrap.RESULTS_DIR +
                f"/qpe_rc_bound_scaling_vs_N_depth{int(depth)}{OUTPUT_SUFFIX}.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)
    return outputs


def focused_renyi_figures(depths, ns, sc, fits):
    """Dedicated focused Rényi-factor figure at every fixed repetition depth."""
    outputs = []
    x = np.asarray(ns, float)
    for di, depth in enumerate(depths):
        fig, ax = plt.subplots(figsize=(7.8, 5.2))
        s2m, _, s2sem = _mean_sd_sem(sc["renyi_sqrt_factor_unclipped"][di])
        exm, _, exsem = _mean_sd_sem(sc["uniform_tvd"][di])
        ax.errorbar(x, s2m, yerr=s2sem, fmt="o", color=COLORS["renyi"], capsize=3,
                    label=r"Rényi factor $S_2$ ensemble mean $\pm$ SEM")
        ax.errorbar(x, exm, yerr=exsem, fmt="^", color=COLORS["exact"], capsize=3,
                    label=r"Exact factor $D_{TV}(p,u)$ ensemble mean $\pm$ SEM")
        g = np.linspace(x.min(), x.max(), 300)
        fs = fits[int(depth)]["s2"]
        if fs.get("ok"):
            f = fs["pow2"]
            ax.plot(g, f["A"]*2**(f["alpha"]*g), "--", color=COLORS["renyi"],
                    label=fr"$S_2=A2^{{\alpha N}}$: $\alpha={f['alpha']:.3f}\pm{f['alpha_err']:.3f}$, $R^2={f['r2']:.3f}$")
        ax.plot(g, haar_renyi_factor_on_axis(g), "-.", color=COLORS["haar"],
                label=r"finite-$N$ Haar prediction $S_2^{\rm Haar}$")
        fh = fits[int(depth)]["haar"]
        if fh.get("ok"):
            ax.plot(g, predict_haar_crossover(fh, g), linestyle=(0,(5,2,1,2)),
                    color=COLORS["entropy"],
                    label=fr"Haar crossover: $A={fh['A']:.2f}$, $b={fh['b']:.3f}$, $R^2={fh['r2']:.3f}$")
        fe = fits[int(depth)]["exact"]
        if fe.get("ok"):
            f = fe["satL"]
            ax.plot(g, f["L"]-f["A"]*np.exp(-f["b"]*g), ":", color=COLORS["exact"],
                    label=fr"$T=L-Ae^{{-bN}}$: $L={f['L']:.2f}$, $b={f['b']:.3f}$, $R^2={f['r2']:.3f}$")
        ax.set_xlabel(N_AXIS_LABEL)
        ax.set_ylabel("Rényi / exact TVD factor")
        ax.set_xticks(list(x.astype(int)))
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, fontsize=7.4)
        fig.suptitle(f"QPE Rényi factors versus N at fixed depth {int(depth)}\n"
                     "error bars: SEM of phase-ensemble mean; fits weighted by SEM")
        fig.tight_layout()
        path = (_bootstrap.RESULTS_DIR +
                f"/qpe_renyi_factors_vs_N_depth{int(depth)}{OUTPUT_SUFFIX}.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)
    return outputs


def focused_entropy_heatmap(depths, ns, sc):
    """Focused depth-by-N collision-entropy-density heatmap."""
    density, _, _ = _mean_sd_sem(sc["collision_entropy_density"])
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    image = ax.imshow(density, aspect="auto", origin="lower",
                      extent=[float(np.min(ns))-0.5, float(np.max(ns))+0.5,
                              -0.5, len(depths)-0.5])
    ax.set_yticks(np.arange(len(depths)))
    ax.set_yticklabels([str(int(d)) for d in depths])
    ax.set_xticks(list(ns))
    ax.set_xlabel(N_AXIS_LABEL)
    ax.set_ylabel("repetition depth")
    ax.set_title(r"QPE collision-entropy density $H_2/N$ (phase-ensemble mean)")
    fig.colorbar(image, ax=ax, label=r"$H_2/N$")
    fig.tight_layout()
    path = (_bootstrap.RESULTS_DIR +
            f"/qpe_rc_collision_entropy_density_grid{OUTPUT_SUFFIX}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_focused(arrays, meta, fits, sc):
    """Save the focused-script-compatible QPE scaling payload."""
    payload = {
        "depths": np.asarray(meta["depths"], int),
        "n_counting_values": np.asarray(meta["sizes"], int),
        "n_total_values": np.asarray(meta["sizes"], int) + N_SYSTEM_QUBITS,
        "N_axis": np.asarray(arrays["N_axis"], float),
        "phases": np.asarray(meta["phases"], float),
        "scaling_instances": np.array(SCALING_INSTANCES, int),
        "measured_qubit_counts": np.asarray(arrays["measured_sizes"], int),
        "bound": sc["bound"], "bound_std": sc["bound_std"],
        "trajectory_floor": sc["trajectory_floor"],
    }
    for key, value in sc.items():
        if key not in payload:
            payload[key] = value
    for depth, by_name in fits.items():
        for family, fit in by_name.items():
            for key, value in fit.items():
                if np.isscalar(value) and isinstance(value, (int, float, bool, np.number)):
                    payload[f"fit_{family}_{key}_depth{depth}"] = np.array(value)
            if family == "s2" and fit.get("ok"):
                for key, value in fit["pow2"].items():
                    if np.isscalar(value):
                        payload[f"fit_s2_pow2_{key}_depth{depth}"] = np.array(value)
            if family == "exact" and fit.get("ok"):
                for key, value in fit["satL"].items():
                    if np.isscalar(value):
                        payload[f"fit_exact_satL_{key}_depth{depth}"] = np.array(value)
    save_results(OUT_DATA, **payload)
    return OUT_DATA


def main():
    if not self_test():
        raise SystemExit("self-tests failed")
    describe_circuit_family()

    arrays, _, meta = scaling_study()
    fits = focused_fits(arrays, meta)

    if not RUN_NOISY_ANALYSIS:
        raise RuntimeError(
            "RUN_NOISY_ANALYSIS must be True for the focused analysis because the "
            "primary figures reproduce measured TVD, QCAP, exact-WN, and Renyi bounds."
        )
    sc = noisy_study_focused(arrays, meta)

    depths = np.asarray(meta["depths"], int)
    ns = np.asarray(arrays["N_axis"], float)
    bound_figs = focused_bounds_figure(depths, ns, sc, fits)
    renyi_figs = focused_renyi_figures(depths, ns, sc, fits)
    heatmap = focused_entropy_heatmap(depths, ns, sc)
    data_file = save_focused(arrays, meta, fits, sc)

    print("\nwrote:")
    for path in (data_file, *bound_figs, *renyi_figs, heatmap):
        print(f"    {path}")
    print(f"\nchecks passed: {CHECKED['factors']} Renyi inequalities, "
          f"{CHECKED['bounds']} bound-ordering checks, "
          f"{CHECKED['normalization']} normalization checks.")


if __name__ == "__main__":
    main()
