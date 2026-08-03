"""Explicit-RC QCAP, exact-white-noise, and Renyi-2 analysis of ONE fixed QASM circuit.

This script combines two existing examples, neither of which is modified:

  * ``examples/run_qpe_bounding.py`` supplies the QASM loading, the ASAP two-qubit
    cycle extraction, the explicit randomized-compiling (RC) path, the cycle
    benchmarking / QCAP construction, the repeated-circuit depth semantics, and the
    parallel trajectory simulation.
  * ``examples/run_bounding_renyi_focused.py`` supplies the ideal-distribution
    quantities (uniform TVD, collision probability, Renyi-2 divergence, the Renyi
    factor S2, collision entropy), the exact-white-noise prediction, the Renyi-2
    bound, the deterministic self-tests, the bound-ordering assertions, and the
    finite-D Haar comparison.

Statistical interpretation (IMPORTANT, and different from the focused script)
-----------------------------------------------------------------------------
There is ONE fixed QASM circuit here, not an ensemble of randomly generated ideal
circuits. At a given repetition depth ``d`` the ideal target is exactly ``U^d`` and
every ideal-distribution quantity is therefore DETERMINISTIC: one number per depth,
with no circuit-instance spread and no instance error bars.

The scatter that does exist comes from three separate sources, kept separate
throughout:

  1. RC/trajectory replicate scatter -- independent Pauli-twirl realizations and
     independent trajectory seeds produce independent measured TVDs, fitted
     white-noise parameters, and white-noise-model residuals.
  2. QCAP parameter uncertainty -- the cycle-benchmark and readout fits.
  3. The trajectory (Monte Carlo) TVD floor -- estimated at each depth from
     ``FLOOR_PAIRS`` independent noisy-ensemble pairs. It is a property of the
     estimator, not of the circuit or the noise.

What is measured at each depth
------------------------------
    target       = repeat(base_circuit, depth)          # U^d, exactly
    p            = ideal measured-register distribution of THAT target
    eps_qcap     = raw QCAP error parameter from CB + readout
    measured TVD = D_TV(p, explicit-RC noisy distribution)
    exact-WN     = eps_qcap * D_TV(p, u)                # EXACT only under the
                                                        # global white-noise mixture
                                                        # q = (1-eps) p + eps u
    Renyi bound  = eps_qcap * min(1, S2),  S2 = 0.5 sqrt(D sum_x p(x)^2 - 1)

plus a direct test of the white-noise MODEL itself (not only its scalar TVD): the
residual D_TV(q, (1-eps) p + eps u) both at ``eps = eps_qcap`` and at the best-fit
``eps``, and Porter-Thomas/Haar-like diagnostics of the ideal output probabilities.

Terminology used consistently below:
    "Renyi factor"                   -- S2, the square-root collision factor
    "raw QCAP bound"                 -- eps_qcap
    "exact under the global white-noise model" -- eps_qcap * D_TV(p, u)
    "Porter-Thomas/Haar-like output probabilities" -- a statement about p(x),
        NOT a claim that the circuit unitary is Haar-random
    "RC/trajectory replicate scatter" -- the spread over replicates
    "white-noise-model residual"     -- D_TV(actual noisy q, white-noise mixture)

Run:
    python examples/run_qasm_bounding_renyi.py
"""

from __future__ import annotations

import inspect
import math
import os
import random
import warnings
from collections import Counter
from typing import Any, Collection, Tuple

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize_scalar
from scipy.stats import kstest

import _bootstrap  # noqa: F401
from proxysim import NoiseModel, save_results
from proxysim.backends import StatevectorBackend
from proxysim.benchmarking import (
    cycle_benchmark,
    qcap_bound,
    readout_fidelity,
)
from proxysim.circuit import circuit_from_qasm
from proxysim.mqtbank import repeat
from proxysim.noise import (
    apply_readout_to_distribution,
    sample_trajectory,
)
from proxysim.parallel import pmap
from proxysim.rc import pauli_twirl


# =============================================================================
# Configuration  (runtime controls carried over from run_qpe_bounding.py)
# =============================================================================

# Any QASM file the proxysim IR can parse. Nothing below is specialized to QPE.
QASM = os.path.join(
    os.path.dirname(__file__),
    "qpe11.qasm",
)

# Complete repetitions of the imported QASM circuit: U, U^2, U^4, ...
DEPTHS = [1, 2, 4, 8, 16]

# Independent explicit-RC / trajectory replicates per depth. Each one yields one
# measured TVD, one fitted white-noise epsilon, and one white-noise residual.
N_INSTANCES = 30

# Independently dressed (Pauli-twirled) circuits averaged inside ONE replicate.
N_RC_REALIZATIONS = 24

# Stochastic noise trajectories per dressed circuit. Total statevector trajectories
# per replicate: N_RC_REALIZATIONS * N_TRAJ.
N_TRAJ = 12

# Independent noisy-ensemble PAIRS used to estimate the Monte Carlo TVD floor.
FLOOR_PAIRS = 4

# Sequence lengths and statistics used inside cycle benchmarking.
CB_DEPTHS = [1, 2, 4, 8]
CB_SHOTS = 800
CB_DECAYS = 20

# Coherent residual-ZZ angle, rad. 0.0 -> purely Pauli-stochastic noise, in which
# case explicit RC and simulation under the already-twirled model coincide and the
# redundant comparison series is skipped (see RUN_TWIRLED_SERIES below).
THETA_ZZ = 0.00

SEED = 42

# Depths at which a Porter-Thomas histogram of z = D * p(x) is written.
PORTER_THOMAS_DEPTHS = [1, 4, 16]

# Fixed physical noise throughout the depth sweep (from run_qpe_bounding.py).
NOISE = NoiseModel(
    enabled=True,
    p1=1e-5,
    p2=2.5e-4,
    p_readout=2.5e-4,
    p_idle=5e-5,
    p_z1=0.0,
    p_zz=0.0,
    theta_1q=0.0,
    theta_zz=THETA_ZZ,
)

# Simulating under NOISE.twirled() is what RC is SUPPOSED to reproduce. With
# THETA_ZZ == 0 the physical model is already Pauli-stochastic, so that series would
# duplicate the explicit-RC series and is not run. With THETA_ZZ != 0 it is run as a
# separately labeled comparison.
RUN_TWIRLED_SERIES = THETA_ZZ != 0.0

TOL = 1e-9  # tolerance for the bound-ordering assertions

COLORS = {
    "measured": "#0072B2",
    "qcap": "#D55E00",
    "exact": "#009E73",
    "renyi": "#CC79A7",
    "floor": "#7A5195",
    "haar": "0.35",
}

STEM = os.path.splitext(os.path.basename(QASM))[0]

OUT_MAIN = f"{_bootstrap.RESULTS_DIR}/{STEM}_bounding_renyi.png"
OUT_DATA = f"{_bootstrap.RESULTS_DIR}/{STEM}_bounding_renyi_data.npz"
OUT_FACTORS = f"{_bootstrap.RESULTS_DIR}/{STEM}_renyi_factors_vs_depth.png"
OUT_STATS = f"{_bootstrap.RESULTS_DIR}/{STEM}_output_statistics_vs_depth.png"
OUT_WHITE_NOISE = f"{_bootstrap.RESULTS_DIR}/{STEM}_white_noise_diagnostics.png"


def porter_thomas_path(depth: int) -> str:
    """Per-depth Porter-Thomas histogram path."""
    return (
        f"{_bootstrap.RESULTS_DIR}/{STEM}"
        f"_porter_thomas_diagnostics_depth{int(depth)}.png"
    )


_SV = StatevectorBackend()

CHECKED = {"factors": 0, "bounds": 0}  # how many depths passed each check


# =============================================================================
# RC return-value compatibility  [source: run_qpe_bounding.py]
# =============================================================================

def _looks_like_circuit(obj: Any) -> bool:
    """Return whether obj resembles a proxysim Circuit."""
    return (
        obj is not None
        and hasattr(obj, "gates")
        and hasattr(obj, "n_qubits")
    )


def _normalize_virtual_indices(
    virtual: Any,
) -> Tuple[int, ...]:
    """Normalize virtual-gate indices returned by pauli_twirl."""
    if virtual is None:
        return ()

    if isinstance(virtual, dict):
        return tuple(
            sorted(int(index) for index in virtual)
        )

    if isinstance(virtual, np.ndarray):
        return tuple(
            int(index)
            for index in virtual.tolist()
        )

    if (
        isinstance(virtual, Collection)
        and not isinstance(virtual, (str, bytes))
    ):
        return tuple(
            int(index)
            for index in virtual
        )

    raise TypeError(
        "Could not interpret pauli_twirl's virtual-gate report. "
        f"Received {type(virtual).__name__}."
    )


def _extract_dressed_and_virtual(
    result: Any,
) -> Tuple[Any, Tuple[int, ...]]:
    """Extract dressed circuit and virtual indices from pauli_twirl output."""
    if result is None:
        raise RuntimeError(
            "pauli_twirl returned None."
        )

    if isinstance(result, dict):
        circuit = None

        for key in (
            "circuit",
            "compiled_circuit",
            "dressed_circuit",
            "twirled_circuit",
        ):
            if key in result:
                circuit = result[key]
                break

        if circuit is None:
            raise TypeError(
                "pauli_twirl returned a dictionary without a recognized "
                f"circuit key. Keys: {list(result)}"
            )

        virtual = ()

        for key in (
            "virtual",
            "virtual_indices",
            "virtual_gates",
            "frame_indices",
        ):
            if key in result:
                virtual = _normalize_virtual_indices(
                    result[key]
                )
                break

        return circuit, virtual

    if isinstance(result, tuple):
        if not result:
            raise TypeError(
                "pauli_twirl returned an empty tuple."
            )

        if len(result) == 1:
            return result[0], ()

        first, second = result[0], result[1]

        if _looks_like_circuit(first):
            return (
                first,
                _normalize_virtual_indices(second),
            )

        if _looks_like_circuit(second):
            return (
                second,
                _normalize_virtual_indices(first),
            )

        raise TypeError(
            "Could not identify a circuit in pauli_twirl's tuple return. "
            f"Types: {[type(item).__name__ for item in result]}"
        )

    if hasattr(result, "circuit"):
        circuit = result.circuit
        virtual = ()

        for attr in (
            "virtual",
            "virtual_indices",
            "virtual_gates",
            "frame_indices",
        ):
            if hasattr(result, attr):
                virtual = _normalize_virtual_indices(
                    getattr(result, attr)
                )
                break

        return circuit, virtual

    if _looks_like_circuit(result):
        return result, ()

    raise TypeError(
        "Could not extract a dressed circuit from pauli_twirl output. "
        f"Received {type(result).__name__}."
    )


def explicitly_randomized_compile(
    circ,
    seed: int,
):
    """Return (dressed circuit, virtual dressing-gate indices).

    mark_virtual=True is essential. The inserted Pauli frame operations must remain
    part of the circuit unitary but must not receive physical p1 noise, create extra
    physical layers, or alter spectator-idle accounting.
    """
    try:
        parameters = inspect.signature(
            pauli_twirl
        ).parameters
    except (TypeError, ValueError):
        parameters = {}

    attempts = []

    if (
        "seed" in parameters
        and "mark_virtual" in parameters
    ):
        attempts.append(
            lambda: pauli_twirl(
                circ,
                seed=seed,
                mark_virtual=True,
            )
        )

    if (
        "rng" in parameters
        and "mark_virtual" in parameters
    ):
        attempts.append(
            lambda: pauli_twirl(
                circ,
                rng=random.Random(seed),
                mark_virtual=True,
            )
        )

    if (
        "random_state" in parameters
        and "mark_virtual" in parameters
    ):
        attempts.append(
            lambda: pauli_twirl(
                circ,
                random_state=random.Random(seed),
                mark_virtual=True,
            )
        )

    attempts.extend(
        [
            lambda: pauli_twirl(
                circ,
                seed=seed,
                mark_virtual=True,
            ),
            lambda: pauli_twirl(
                circ,
                rng=random.Random(seed),
                mark_virtual=True,
            ),
            lambda: pauli_twirl(
                circ,
                random.Random(seed),
                mark_virtual=True,
            ),
            lambda: pauli_twirl(
                circ,
                seed,
                mark_virtual=True,
            ),
        ]
    )

    errors = []

    for attempt in attempts:
        try:
            result = attempt()

            dressed, virtual = (
                _extract_dressed_and_virtual(result)
            )

            if not _looks_like_circuit(dressed):
                raise TypeError(
                    "Extracted object is not a proxysim-like Circuit."
                )

            return dressed, virtual

        except TypeError as exc:
            errors.append(str(exc))

    raise TypeError(
        "Could not call proxysim.rc.pauli_twirl with "
        "mark_virtual=True.\nObserved errors:\n  - "
        + "\n  - ".join(errors)
    )


# =============================================================================
# Circuit and cycle utilities  [source: run_qpe_bounding.py, generalized]
# =============================================================================

# Algorithmic two-qubit gate -> the Clifford entangler used to benchmark the noise it
# carries (the proxy convention of proxysim.cb_emit.PROXY_OF).
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
#   "restricted" -- diagonal cp(theta): only the commuting {I,Z}x{I,Z} subgroup, so
#                   the twirl tailors the error toward a Z-diagonal (dephasing)
#                   channel rather than a full Pauli channel.
TWIRL_CLASS = {
    "cx": "full",
    "cnot": "full",
    "cz": "full",
    "cy": "full",
    "swap": "full",
    "cp": "restricted",
}


def twoq_cycle_patterns(circ):
    """Return Counter{cycle pattern: occurrences} for one circuit repetition.

    A cycle is one ASAP-parallel layer of two-qubit gates, together with the nearby
    one-qubit operations that the ASAP partition places alongside it -- the same
    convention run_qpe_bounding.py uses. The pattern key is the frozenset of
    ``(gate name, qubit pair)`` entries in that layer, so distinct entangling
    patterns are benchmarked separately and are NOT approximated as generic
    brickwork layers.
    """
    patterns = Counter()

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
    """The Clifford proxy entangler used to benchmark one cycle pattern.

    A layer that mixes gate families is benchmarked with the proxy of its most
    common family; under the homogeneous synthetic noise model every proxy carries
    the same channel, so this only affects the label, but it is reported anyway.
    """
    families = Counter(name for name, _ in pattern)

    unknown = sorted(set(families) - set(PROXY_ENTANGLER))

    if unknown:
        raise ValueError(
            "No cycle-benchmark proxy is defined for two-qubit gate(s) "
            f"{unknown} found in the QASM circuit."
        )

    return PROXY_ENTANGLER[families.most_common(1)[0][0]]


def twirl_report(circ):
    """Report how completely each two-qubit gate family in circ can be twirled.

    Returns ``(counts_by_family, restricted_families)``. The distinction matters:
    ``cx`` (and every other Clifford entangler) receives the full Pauli twirl the
    QCAP bound assumes, while ``cp(theta)`` is only twirled over the commuting
    ``{I,Z} x {I,Z}`` subgroup.
    """
    counts = Counter(
        gate.name.lower()
        for gate in circ.gates
        if len(gate.qubits) == 2
    )

    unknown = sorted(set(counts) - set(TWIRL_CLASS))

    if unknown:
        raise ValueError(
            f"pauli_twirl has no twirl construction registered for {unknown}."
        )

    restricted = sorted(
        name
        for name in counts
        if TWIRL_CLASS[name] == "restricted"
    )

    return counts, restricted


def exact_probability_vector(circ) -> np.ndarray:
    """Return the ideal full-register computational-basis distribution."""
    return np.asarray(
        Statevector(
            _SV._build(circ)
        ).probabilities(),
        dtype=float,
    )


def marginalize_distribution(
    probability: np.ndarray,
    n_qubits: int,
    measured_qubits,
) -> np.ndarray:
    """Marginalize a full distribution onto measured_qubits.

    The flat vector follows Qiskit's little-endian statevector indexing. Reshaping
    to [2] * n maps array axis k to qubit n - 1 - k.
    """
    measured_set = set(measured_qubits)

    drop_axes = tuple(
        n_qubits - 1 - qubit
        for qubit in range(n_qubits)
        if qubit not in measured_set
    )

    tensor = np.asarray(
        probability,
        dtype=float,
    ).reshape([2] * n_qubits)

    if drop_axes:
        tensor = tensor.sum(
            axis=drop_axes
        )

    return tensor.reshape(-1)


def measured_readout_noise(
    full_probability: np.ndarray,
    noise: NoiseModel,
    n_qubits: int,
    measured_qubits,
) -> np.ndarray:
    """Apply readout flips only to the qubits the QASM circuit measures.

    apply_readout_to_distribution flips every qubit of its input, so marginalize to
    the measured subsystem first and then apply the channel on the reduced vector.
    """
    measured_probability = marginalize_distribution(
        full_probability,
        n_qubits,
        measured_qubits,
    )

    return apply_readout_to_distribution(
        measured_probability,
        noise,
        len(measured_qubits),
    )


def ideal_distribution(circ, measured_qubits) -> np.ndarray:
    """Ideal measured-register distribution of ``circ`` as a dense vector.

    Computed from the exact repeated target that is passed in -- never reused across
    depths -- and marginalized onto the measured qubits exactly as the QASM
    implementation in run_qpe_bounding.py does.
    """
    full = exact_probability_vector(circ)

    return check_normalized(
        marginalize_distribution(
            full,
            circ.n_qubits,
            measured_qubits,
        ),
        "ideal measured distribution",
    )


def ideal_statevector(circ) -> np.ndarray:
    """Return the ideal pure state of a proxysim circuit."""
    return np.asarray(
        Statevector(
            _SV._build(circ)
        ).data,
        dtype=complex,
    )


def state_fidelity(
    psi: np.ndarray,
    phi: np.ndarray,
) -> float:
    """Pure-state fidelity, insensitive to global phase."""
    return float(
        abs(np.vdot(psi, phi)) ** 2
    )


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

    return probability


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two dense probability vectors."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if p.shape != q.shape:
        raise ValueError(
            f"TVD between vectors of different dimension: {p.shape} vs {q.shape}."
        )

    return 0.5 * float(np.abs(p - q).sum())


# =============================================================================
# Ideal-distribution quantities  [source: run_bounding_renyi_focused.py]
# =============================================================================

def ideal_factors(p: np.ndarray, m: int) -> dict:
    """Every ideal-distribution-derived quantity, for one measured distribution.

    ``m`` is the number of MEASURED qubits and D = 2^m, never the full register
    width. ``D * sum_x p(x)^2 - 1`` is clamped at zero before the square root: it is
    algebraically zero for a uniform p and floating point can put it a few ulps
    below.
    """
    p = np.asarray(p, dtype=float)

    d = 2 ** m
    u = 1.0 / d

    uniform_tvd = 0.5 * float(np.abs(p - u).sum())
    collision = float((p ** 2).sum())

    exp_d2 = d * collision                      # e^{D_2(p||u)} = D sum_x p(x)^2
    d2 = math.log(exp_d2) if exp_d2 > 0 else -math.inf
    s2 = 0.5 * math.sqrt(max(exp_d2 - 1.0, 0.0))

    # Exact identities from the collision probability:
    #   H_2 = -log2(sum_x p(x)^2),  deficit = m - H_2 = log2(D sum_x p(x)^2).
    collision_entropy = (
        -math.log2(collision) if collision > 0 else math.inf
    )
    deficit = m - collision_entropy

    # Finite-D Haar / Porter-Thomas comparison.
    haar_collision = 2.0 / (d + 1.0)
    haar_factor = 0.5 * math.sqrt(max((d - 1.0) / (d + 1.0), 0.0))

    # Scaled probabilities z = D p(x). Porter-Thomas => z ~ Exp(1).
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


def uniform_vector(m: int) -> np.ndarray:
    """The uniform distribution over the 2^m measured outcomes."""
    d = 2 ** m
    return np.full(d, 1.0 / d, dtype=float)


def white_noise_mixture(
    p: np.ndarray,
    epsilon: float,
    u: np.ndarray,
) -> np.ndarray:
    """q = (1 - epsilon) p + epsilon u, the global white-noise model."""
    return (1.0 - epsilon) * p + epsilon * u


def fit_white_noise_epsilon(
    q: np.ndarray,
    p: np.ndarray,
    u: np.ndarray,
) -> Tuple[float, float]:
    """Best-fit white-noise parameter and its residual.

    Returns ``(epsilon, D_TV(q, (1-epsilon) p + epsilon u))`` with epsilon minimizing
    the residual over [0, 1]. The objective is convex in epsilon (a TVD of an affine
    family), so the bounded Brent search is reliable; the endpoints are checked too.
    """
    def objective(epsilon: float) -> float:
        return tvd(q, white_noise_mixture(p, epsilon, u))

    result = minimize_scalar(
        objective,
        bounds=(0.0, 1.0),
        method="bounded",
    )

    best_eps = float(np.clip(result.x, 0.0, 1.0))
    best_res = float(objective(best_eps))

    for endpoint in (0.0, 1.0):
        value = objective(endpoint)

        if value < best_res:
            best_eps, best_res = endpoint, float(value)

    return best_eps, best_res


# =============================================================================
# Deterministic self-test  [source: run_bounding_renyi_focused.py]
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    """Closed-form uniform and deterministic distributions, where every factor is known.

        uniform p:        D_TV(p,u) = 0,          S_2 = 0,   H_2 = m
        deterministic p:  D_TV(p,u) = 1 - 1/D,    S_2 = 0.5 sqrt(D - 1),  H_2 = 0

    The deterministic S_2 exceeds one for m >= 3 -- expected, and exactly why the
    bound uses the CLIPPED factor.
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

        for eps in (0.0, 0.37, 1.0):
            assert abs(eps * f["uniform_tvd"]) < 1e-12
            assert abs(eps * f["renyi_sqrt_factor_clipped"]) < 1e-12

        det = np.zeros(d, dtype=float)
        det[0] = 1.0
        g = ideal_factors(check_normalized(det, f"self-test delta m={m}"), m)

        want_tvd = 1.0 - 1.0 / d
        want_s2 = 0.5 * math.sqrt(d - 1.0)

        assert abs(g["uniform_tvd"] - want_tvd) < 1e-12, (g, want_tvd)
        assert abs(g["renyi_sqrt_factor_unclipped"] - want_s2) < 1e-12, (g, want_s2)
        assert abs(g["renyi2_divergence"] - math.log(d)) < 1e-12, g
        assert abs(g["renyi_sqrt_factor_clipped"] - min(1.0, want_s2)) < 1e-12, g
        assert abs(g["collision_entropy"]) < 1e-12, g
        check_factors(g, f"deterministic m={m}")

        # The white-noise fit must recover eps exactly on a synthetic mixture.
        u = uniform_vector(m)
        eps_true = 0.3
        q = white_noise_mixture(det, eps_true, u)
        eps_fit, residual = fit_white_noise_epsilon(q, det, u)

        # Tolerances here are set by the bounded Brent search, not by the algebra.
        assert abs(eps_fit - eps_true) < 1e-4, (eps_fit, eps_true)
        assert residual < 1e-6, residual
        assert abs(tvd(q, det) - eps_true * want_tvd) < 1e-12, (q, det)

        lines.append(
            f"    m={m}: uniform -> D_TV=0, S_2=0, H_2={f['collision_entropy']:.3f}"
            f"   |   deterministic -> D_TV={g['uniform_tvd']:.6f} (=1-1/D), "
            f"S_2={g['renyi_sqrt_factor_unclipped']:.6f} (=0.5*sqrt(D-1)), "
            f"H_2={g['collision_entropy']:.3f}, "
            f"clipped={g['renyi_sqrt_factor_clipped']:.6f}"
        )

    if verbose:
        print("self-test: closed-form uniform / deterministic distributions")
        print("\n".join(lines))
        print(
            "    white-noise fit recovers eps on an exact mixture; "
            "D_TV(q,p) = eps * D_TV(p,u) verified"
        )
        print("    all assertions passed\n")

    return True


# =============================================================================
# Cycle benchmarking  [source: run_qpe_bounding.py]
# =============================================================================

def cb_cycle_worker(args):
    """Benchmark one two-qubit-layer proxy cycle."""
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


def benchmark_cycle_patterns(circ, patterns, n_measured: int, n_workers: int):
    """Benchmark every distinct two-qubit cycle class the QCAP bound needs.

    Every distinct cycle pattern in the parsed QASM circuit is counted separately.
    Under the homogeneous synthetic noise model e_F depends only on the proxy
    entangler and the number of simultaneous two-qubit gates in the layer, so the
    CB run itself is cached on ``(proxy, layer size)``; the per-pattern occurrence
    counts are retained exactly and fed to qcap_bound unchanged.
    """
    readout_fid, readout_std = readout_fidelity(
        n_measured,
        NOISE,
        seed=3,
    )

    representative = {}

    for pattern in patterns:
        key = (pattern_proxy(pattern), len(pattern))

        if key not in representative:
            representative[key] = pattern

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

    results = pmap(
        cb_cycle_worker,
        jobs,
        n_workers=n_workers,
    )

    ef_by_class = {
        key: (
            float(result["e_F"]),
            float(result["e_F_std"]),
        )
        for key, result in zip(keys, results)
    }

    counts_one_rep = {}
    cycle_efs = {}
    cycle_labels = {}

    for index, (pattern, count) in enumerate(patterns.items()):
        name = f"cycle_{index}"
        key = (pattern_proxy(pattern), len(pattern))

        counts_one_rep[name] = int(count)
        cycle_efs[name] = ef_by_class[key]
        cycle_labels[name] = (
            f"{key[1]}x{key[0]} "
            + ", ".join(
                f"{gate}{pair}"
                for gate, pair in sorted(pattern)
            )
        )

    return (
        counts_one_rep,
        cycle_efs,
        cycle_labels,
        readout_fid,
        readout_std,
        ef_by_class,
    )


# =============================================================================
# Explicit-RC replicate worker  [source: run_qpe_bounding.py, extended]
# =============================================================================

def _noisy_measured_distribution(
    target,
    noise,
    measured_qubits,
    n_rc_realizations,
    n_traj,
    seed,
    explicit_rc: bool,
) -> np.ndarray:
    """One noisy measured-register distribution.

    Averaging order (unchanged from run_qpe_bounding.py):

        trajectory mean within each RC realization
        then mean over RC realizations
        then readout applied analytically on the measured register

    With ``explicit_rc=True`` a FRESH Pauli twirl is generated per realization and
    the dressed circuit is simulated under the ORIGINAL physical noise model, with
    the twirl gates marked virtual. With ``explicit_rc=False`` the undressed circuit
    is simulated under the already-Pauli-twirled model -- the separately labeled
    comparison series, never a replacement for explicit RC.
    """
    n_qubits = target.n_qubits
    d_measured = 2 ** len(measured_qubits)

    average = np.zeros(d_measured, dtype=float)

    for rc_index in range(n_rc_realizations):
        rc_seed = seed + 1_000_003 * rc_index

        if explicit_rc:
            dressed, virtual_indices = explicitly_randomized_compile(
                target,
                seed=rc_seed,
            )
        else:
            dressed, virtual_indices = target, ()

        trajectory_rng = random.Random(rc_seed + 1)

        full_probability = np.zeros(2 ** n_qubits, dtype=float)

        for _ in range(n_traj):
            noisy_dressed = sample_trajectory(
                dressed,
                noise,
                trajectory_rng,
                virtual=virtual_indices,
            )

            state = np.asarray(
                Statevector(
                    _SV._build(noisy_dressed)
                ).data,
                dtype=complex,
            )

            full_probability += np.abs(state) ** 2

        full_probability /= n_traj

        average += measured_readout_noise(
            full_probability,
            noise,
            n_qubits,
            measured_qubits,
        )

    average /= n_rc_realizations

    return average


def rc_replicate_worker(args):
    """One RC/trajectory replicate: measured TVD plus the white-noise-model tests.

    Returns ``(measured_tvd, epsilon_white_noise_fit, white_noise_model_residual_fit,
    white_noise_model_residual_qcap)``. The last two test the FULL distributional
    white-noise model, not only its scalar TVD.
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
        target,
        noise,
        measured_qubits,
        n_rc_realizations,
        n_traj,
        seed,
        explicit_rc,
    )

    check_normalized(noisy, "noisy measured distribution", atol=1e-6)

    if noisy.shape != np.shape(ideal_measured):
        raise ValueError(
            "ideal and noisy measured-register dimensions differ: "
            f"{np.shape(ideal_measured)} vs {noisy.shape}."
        )

    u = uniform_vector(len(measured_qubits))

    residual_qcap = tvd(
        noisy,
        white_noise_mixture(ideal_measured, eps_qcap, u),
    )

    eps_fit, residual_fit = fit_white_noise_epsilon(noisy, ideal_measured, u)

    return (
        tvd(noisy, ideal_measured),
        eps_fit,
        residual_fit,
        residual_qcap,
    )


def floor_pair_worker(args):
    """TVD between two INDEPENDENT noisy ensembles of the same target circuit.

    This is the Monte Carlo estimator floor: with a finite number of RC realizations
    and trajectories, two independent estimates of the same noisy distribution
    already differ by this much. It is kept strictly separate from the CB/readout
    parameter uncertainty, from the RC/trajectory replicate scatter of the measured
    TVD, and from the (deterministic) ideal-distribution quantities.
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
        target,
        noise,
        measured_qubits,
        n_rc_realizations,
        n_traj,
        seed_a,
        explicit_rc,
    )

    second = _noisy_measured_distribution(
        target,
        noise,
        measured_qubits,
        n_rc_realizations,
        n_traj,
        seed_b,
        explicit_rc,
    )

    return tvd(first, second)


def scatter_summary(values) -> str:
    """Return mean and range of one raw-TVD scatter column."""
    values = np.asarray(values, dtype=float)

    return (
        f"{values.mean():.5f} "
        f"[{values.min():.5f}, {values.max():.5f}]"
    )


# =============================================================================
# Validation  [source: run_qpe_bounding.py]
# =============================================================================

def validate_one_rc_realization(base_circ):
    """Check RC logical equivalence and virtual-gate reporting."""
    dressed, virtual_indices = explicitly_randomized_compile(
        base_circ,
        seed=SEED,
    )

    fidelity = state_fidelity(
        ideal_statevector(base_circ),
        ideal_statevector(dressed),
    )

    original_twoq = sum(
        len(gate.qubits) == 2
        for gate in base_circ.gates
    )

    dressed_twoq = sum(
        len(gate.qubits) == 2
        for gate in dressed.gates
    )

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
            "pauli_twirl reported no virtual-gate indices. If Pauli dressing gates "
            "were inserted, they may be incorrectly charged physical one-qubit and "
            "idle noise.",
            RuntimeWarning,
        )

    if not np.isclose(fidelity, 1.0, atol=1e-9, rtol=1e-9):
        raise RuntimeError(
            "The RC-dressed circuit is not logically equivalent to the original "
            "QASM circuit. A returned frame correction may not have been applied, "
            "or the pauli_twirl return contract differs from the forms handled by "
            "this script."
        )


# =============================================================================
# Main sweep
# =============================================================================

# Per-depth ideal-distribution keys (deterministic: ONE value per depth).
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


def run_sweep():
    """Load the QASM circuit, benchmark its cycles, and sweep the repetition depth."""
    with open(QASM, encoding="utf-8") as qasm_file:
        base_circ, measured = circuit_from_qasm(qasm_file.read())

    measured = tuple(sorted(int(qubit) for qubit in measured))

    if not measured:
        raise ValueError(f"{QASM} contains no measurements.")

    n_qubits = base_circ.n_qubits
    n_measured = len(measured)
    d_measured = 2 ** n_measured

    ancillas = [
        qubit
        for qubit in range(n_qubits)
        if qubit not in measured
    ]

    patterns = twoq_cycle_patterns(base_circ)
    cycles_per_rep = sum(patterns.values())

    if cycles_per_rep == 0:
        raise ValueError(
            f"No ASAP two-qubit cycle layers were found in {os.path.basename(QASM)}."
        )

    twoq_counts, restricted_families = twirl_report(base_circ)
    n_twoq = sum(twoq_counts.values())

    print(base_circ.summary())

    print(
        f"QASM: {QASM}\n"
        f"register size: {n_qubits} qubits\n"
        f"measured qubits: {n_measured} {list(measured)}  ->  D = 2^m = {d_measured}\n"
        f"unmeasured ancilla qubits: {ancillas}\n"
        f"two-qubit gates per repetition: {n_twoq} "
        f"({', '.join(f'{k}x{v}' for k, v in sorted(twoq_counts.items()))})\n"
        f"ASAP two-qubit cycles per repetition: {cycles_per_rep}\n"
        f"distinct cycle patterns: {len(patterns)}\n"
        f"simultaneous-2q layer sizes: "
        f"{sorted(set(len(pattern) for pattern in patterns))}"
    )

    print(
        "cycle convention: each ASAP-parallel two-qubit layer, together with its "
        "nearby one-qubit operations, is one CB/RC cycle"
    )

    print(
        "TVD protocol: explicitly Pauli-dress the two-qubit gates, simulate the "
        "dressed circuit under the ORIGINAL noise model, average the output "
        "distributions, then marginalize to the measured register"
    )

    if restricted_families:
        warnings.warn(
            "Restricted Pauli twirl: "
            + ", ".join(restricted_families)
            + " is diagonal but non-Clifford, so pauli_twirl only twirls over the "
            "commuting {I,Z}x{I,Z} subgroup. The error is tailored toward a "
            "Z-diagonal (dephasing) channel rather than a full Pauli channel, and "
            "the QCAP bound's full-Pauli-twirl assumption is only approximate for "
            "those gates.",
            RuntimeWarning,
        )
        print(
            "WARNING: restricted twirl on "
            + ", ".join(restricted_families)
            + " (Z-subgroup only); fully twirled Clifford entanglers: "
            + ", ".join(
                sorted(
                    name
                    for name in twoq_counts
                    if TWIRL_CLASS[name] == "full"
                )
            )
            or "(none)"
        )
    else:
        print(
            "twirl coverage: every two-qubit gate family "
            f"({', '.join(sorted(twoq_counts))}) is Clifford and receives the FULL "
            "two-qubit Pauli twirl"
        )

    print(
        f"fixed noise:\n"
        f"    p1={NOISE.p1:.3e}\n"
        f"    p2={NOISE.p2:.3e}\n"
        f"    p_readout={NOISE.p_readout:.3e}\n"
        f"    p_idle={NOISE.p_idle:.3e}\n"
        f"    p_z1={NOISE.p_z1:.3e}\n"
        f"    p_zz={NOISE.p_zz:.3e}\n"
        f"    theta_1q={NOISE.theta_1q:.3e}\n"
        f"    theta_zz={NOISE.theta_zz:.3e}\n"
    )

    print(
        'note: "exact white-noise prediction" is exact ONLY under the global '
        "white-noise mixture q = (1-eps) p + eps u; it is a model, not a guarantee.\n"
    )

    n_workers = max(1, (os.cpu_count() or 2) - 2)

    validate_one_rc_realization(base_circ)

    (
        counts_one_rep,
        cycle_efs,
        cycle_labels,
        readout_fid,
        readout_std,
        ef_by_class,
    ) = benchmark_cycle_patterns(base_circ, patterns, n_measured, n_workers)

    print("cycle-benchmark results (one e_F per proxy / layer-size class):")

    for (proxy, size), (ef, ef_std) in sorted(ef_by_class.items()):
        print(
            f"    {size} simultaneous {proxy}: e_F={ef:.6f} +/- {ef_std:.6f}"
        )

    print(
        f"    measured-register readout fidelity: "
        f"{readout_fid:.6f} +/- {readout_std:.6f}\n"
    )

    results = {key: [] for key in _IDEAL_KEYS}

    results.update(
        {
            "cycle_depths": [],
            "twoq_gates": [],
            "ideal_support": [],
            "qcap_bound": [],
            "qcap_bound_std": [],
            "exact_white_noise_prediction": [],
            "renyi_bound": [],
            "tvd_points": [],
            "epsilon_white_noise_fit": [],
            "white_noise_model_residual_fit": [],
            "white_noise_model_residual_qcap": [],
            "trajectory_floor_points": [],
            "twirled_tvd_points": [],
        }
    )

    ideal_by_depth = {}

    print(
        f"{'depth':>6}"
        f"{'cycles':>9}"
        f"{'2q':>8}"
        f"{'QCAP':>10}"
        f"{'exact-WN':>10}"
        f"{'Renyi':>9}"
        f"{'S2':>9}"
        f"{'D_TV(p,u)':>11}"
        f"{'floor':>9}"
        f"{'RC TVD mean [min,max]':>31}"
    )

    for depth in DEPTHS:
        target = repeat(base_circ, depth)

        cycle_depth = depth * cycles_per_rep
        expected_twoq = depth * n_twoq

        actual_twoq = sum(
            len(gate.qubits) == 2
            for gate in target.gates
        )

        if actual_twoq != expected_twoq:
            raise RuntimeError(
                "Unexpected two-qubit-gate count after circuit repetition: "
                f"expected {expected_twoq}, found {actual_twoq}."
            )

        # The ideal distribution is recomputed from THIS repeated target; the
        # depth-1 distribution is never reused.
        ideal_measured = ideal_distribution(target, measured)
        ideal_by_depth[int(depth)] = ideal_measured

        factors = ideal_factors(ideal_measured, n_measured)
        check_factors(factors, f"{STEM} depth={depth}")

        # QCAP over the ACTUAL per-pattern cycle census of the repeated circuit.
        repetition_counts = {
            name: count * depth
            for name, count in counts_one_rep.items()
        }

        bound = qcap_bound(
            repetition_counts,
            cycle_efs,
            readout_fid,
            readout_std,
        )

        eps_qcap = float(bound["error"])

        exact_wn = eps_qcap * factors["uniform_tvd"]
        renyi_bound = eps_qcap * factors["renyi_sqrt_factor_clipped"]

        # Bound ordering and range [source: run_bounding_renyi_focused.py].
        assert -TOL <= eps_qcap <= 1.0 + TOL, eps_qcap
        assert -TOL <= exact_wn <= 1.0 + TOL, exact_wn
        assert -TOL <= renyi_bound <= 1.0 + TOL, renyi_bound
        assert exact_wn <= renyi_bound + TOL, (exact_wn, renyi_bound)
        assert renyi_bound <= eps_qcap + TOL, (renyi_bound, eps_qcap)
        CHECKED["bounds"] += 1

        jobs = [
            (
                target,
                NOISE,
                ideal_measured,
                measured,
                N_RC_REALIZATIONS,
                N_TRAJ,
                SEED + 100_000_000 * depth + 100_000 * index,
                eps_qcap,
                True,
            )
            for index in range(N_INSTANCES)
        ]

        replicates = pmap(rc_replicate_worker, jobs, n_workers=n_workers)

        tvd_values = np.asarray([row[0] for row in replicates], dtype=float)
        eps_fit_values = np.asarray([row[1] for row in replicates], dtype=float)
        residual_fit_values = np.asarray([row[2] for row in replicates], dtype=float)
        residual_qcap_values = np.asarray([row[3] for row in replicates], dtype=float)

        floor_jobs = [
            (
                target,
                NOISE,
                measured,
                N_RC_REALIZATIONS,
                N_TRAJ,
                SEED + 700_000_000 + 100_000_000 * depth + 200_000 * index,
                SEED + 900_000_000 + 100_000_000 * depth + 200_000 * index,
                True,
            )
            for index in range(FLOOR_PAIRS)
        ]

        floor_values = np.asarray(
            pmap(floor_pair_worker, floor_jobs, n_workers=n_workers),
            dtype=float,
        )

        if RUN_TWIRLED_SERIES:
            twirled_jobs = [
                (
                    target,
                    NOISE.twirled(),
                    ideal_measured,
                    measured,
                    N_RC_REALIZATIONS,
                    N_TRAJ,
                    SEED + 500_000_000 + 100_000_000 * depth + 100_000 * index,
                    eps_qcap,
                    False,
                )
                for index in range(N_INSTANCES)
            ]

            twirled_values = np.asarray(
                [row[0] for row in pmap(
                    rc_replicate_worker, twirled_jobs, n_workers=n_workers)],
                dtype=float,
            )
        else:
            twirled_values = np.full(N_INSTANCES, np.nan, dtype=float)

        for key in _IDEAL_KEYS:
            results[key].append(factors[key])

        results["cycle_depths"].append(cycle_depth)
        results["twoq_gates"].append(actual_twoq)
        results["ideal_support"].append(
            int(np.count_nonzero(ideal_measured > 1e-12))
        )
        results["qcap_bound"].append(eps_qcap)
        results["qcap_bound_std"].append(float(bound["std"]))
        results["exact_white_noise_prediction"].append(exact_wn)
        results["renyi_bound"].append(renyi_bound)
        results["tvd_points"].append(tvd_values)
        results["epsilon_white_noise_fit"].append(eps_fit_values)
        results["white_noise_model_residual_fit"].append(residual_fit_values)
        results["white_noise_model_residual_qcap"].append(residual_qcap_values)
        results["trajectory_floor_points"].append(floor_values)
        results["twirled_tvd_points"].append(twirled_values)

        print(
            f"{depth:>6}"
            f"{cycle_depth:>9}"
            f"{actual_twoq:>8}"
            f"{eps_qcap:>10.5f}"
            f"{exact_wn:>10.5f}"
            f"{renyi_bound:>9.5f}"
            f"{factors['renyi_sqrt_factor_unclipped']:>9.4f}"
            f"{factors['uniform_tvd']:>11.5f}"
            f"{floor_values.mean():>9.5f}"
            f"{scatter_summary(tvd_values):>31}",
            flush=True,
        )

    arrays = {
        key: np.asarray(value, dtype=float)
        for key, value in results.items()
    }

    arrays["cycle_depths"] = np.asarray(results["cycle_depths"], dtype=int)
    arrays["twoq_gates"] = np.asarray(results["twoq_gates"], dtype=int)
    arrays["ideal_support"] = np.asarray(results["ideal_support"], dtype=int)

    meta = {
        "measured": measured,
        "n_measured": n_measured,
        "cycles_per_rep": cycles_per_rep,
        "twoq_per_rep": n_twoq,
        "readout_fid": readout_fid,
        "readout_std": readout_std,
        "cycle_labels": cycle_labels,
        "counts_one_rep": counts_one_rep,
        "cycle_efs": cycle_efs,
        "restricted_families": restricted_families,
        "twoq_counts": dict(twoq_counts),
    }

    return arrays, ideal_by_depth, meta


def white_noise_summary(arrays: dict) -> dict:
    """Per-depth means and RC/trajectory replicate standard deviations."""
    eps_fit = arrays["epsilon_white_noise_fit"]
    res_fit = arrays["white_noise_model_residual_fit"]
    res_qcap = arrays["white_noise_model_residual_qcap"]

    ddof = 1 if eps_fit.shape[1] > 1 else 0

    return {
        "mean_fitted_epsilon": eps_fit.mean(axis=1),
        "std_fitted_epsilon": eps_fit.std(axis=1, ddof=ddof),
        "mean_white_noise_residual_fit": res_fit.mean(axis=1),
        "std_white_noise_residual_fit": res_fit.std(axis=1, ddof=ddof),
        "mean_white_noise_residual_qcap": res_qcap.mean(axis=1),
        "std_white_noise_residual_qcap": res_qcap.std(axis=1, ddof=ddof),
        "qcap_minus_fitted_epsilon": (
            arrays["qcap_bound"] - eps_fit.mean(axis=1)
        ),
        "trajectory_floor": arrays["trajectory_floor_points"].mean(axis=1),
        "trajectory_floor_std": arrays["trajectory_floor_points"].std(
            axis=1,
            ddof=1 if arrays["trajectory_floor_points"].shape[1] > 1 else 0,
        ),
    }


# =============================================================================
# Figures
# =============================================================================

def _depth_axis(ax, depths):
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(depths))
    ax.set_xticklabels([str(int(d)) for d in depths])
    ax.set_xlabel("repeated-circuit depth $d$ (target $U^d$)")
    ax.grid(True, alpha=0.15)


def figure_main(depths, arrays, summary, meta):
    """A. Measured explicit-RC TVD against every bound, versus repetition depth."""
    fig, ax = plt.subplots(figsize=(9.6, 6.0))

    points = arrays["tvd_points"]

    for index, depth in enumerate(depths):
        # Raw replicate points, deliberately NOT connected: they are independent
        # RC/trajectory realizations of ONE fixed circuit, not a trend line.
        ax.plot(
            np.full(points.shape[1], depth),
            points[index],
            "o",
            color=COLORS["measured"],
            ms=5.0,
            alpha=0.55,
            zorder=3,
            label="explicit-RC measured TVD (replicates)" if index == 0 else None,
        )

    if RUN_TWIRLED_SERIES:
        twirled = arrays["twirled_tvd_points"]

        for index, depth in enumerate(depths):
            ax.plot(
                np.full(twirled.shape[1], depth),
                twirled[index],
                "s",
                color="#666666",
                ms=4.0,
                alpha=0.45,
                zorder=2,
                label=(
                    "simulated under NOISE.twirled() (comparison series)"
                    if index == 0
                    else None
                ),
            )

    qcap = arrays["qcap_bound"]
    qcap_std = arrays["qcap_bound_std"]

    ax.plot(
        depths,
        qcap,
        "-",
        color=COLORS["qcap"],
        lw=2.6,
        zorder=5,
        label="raw QCAP bound $\\epsilon_{\\rm QCAP}$ (CB + readout)",
    )

    ax.fill_between(
        depths,
        np.clip(qcap - 2.96 * qcap_std, 0.0, 1.0),
        np.clip(qcap + 2.96 * qcap_std, 0.0, 1.0),
        color=COLORS["qcap"],
        alpha=0.18,
        lw=0,
        zorder=1,
        label="QCAP CB/readout parameter uncertainty",
    )

    ax.plot(
        depths,
        arrays["renyi_bound"],
        "-",
        color=COLORS["renyi"],
        lw=2.2,
        zorder=5,
        label=r"Rényi-2 bound $\epsilon_{\rm QCAP}\min(1,S_2)$",
    )

    ax.plot(
        depths,
        arrays["exact_white_noise_prediction"],
        "-",
        color=COLORS["exact"],
        lw=2.2,
        zorder=5,
        label=(
            r"exact under the global white-noise model "
            r"$\epsilon_{\rm QCAP}D_{TV}(p,u)$"
        ),
    )

    ax.plot(
        depths,
        summary["trajectory_floor"],
        ":",
        color=COLORS["floor"],
        lw=2.0,
        marker="d",
        ms=5,
        zorder=4,
        label=f"trajectory TVD floor ({FLOOR_PAIRS} independent ensemble pairs)",
    )

    ax.axhline(1.0, color="0.5", lw=0.8, ls=":")

    _depth_axis(ax, depths)
    ax.set_yscale("log")
    ax.set_ylabel("total variation distance / bound")

    ax.set_title(
        f"{STEM}: explicit-RC TVD versus QCAP, exact-white-noise, and Rényi-2\n"
        f"{meta['n_measured']} measured qubits; one fixed QASM circuit; "
        f"{N_INSTANCES} RC/trajectory replicates per depth"
    )

    ax.legend(frameon=False, fontsize=8.2, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_MAIN, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return OUT_MAIN


def figure_renyi_factors(depths, arrays, meta):
    """B. Rényi factor and collision-entropy diagnostics.

    One deterministic ideal value per depth, so no circuit-instance error bars.
    """
    m = meta["n_measured"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    ax = axes[0]
    ax.plot(
        depths,
        arrays["renyi_sqrt_factor_unclipped"],
        "o-",
        color=COLORS["renyi"],
        lw=1.8,
        label=r"Rényi factor $S_2$ (unclipped)",
    )
    ax.plot(
        depths,
        arrays["haar_renyi_factor"],
        "-.",
        color=COLORS["haar"],
        lw=1.7,
        label=r"finite-$D$ Haar Rényi factor $\frac{1}{2}\sqrt{(D-1)/(D+1)}$",
    )
    ax.plot(
        depths,
        arrays["uniform_tvd"],
        "^-",
        color=COLORS["exact"],
        lw=1.8,
        label=r"$D_{TV}(p,u)$",
    )
    ax.axhline(0.5, color="0.7", lw=0.8, ls=":")
    ax.set_ylabel("ideal-distribution factor")
    ax.set_title("Rényi factor and uniform TVD (deterministic per depth)")
    _depth_axis(ax, depths)
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    ax.plot(
        depths,
        arrays["collision_entropy"],
        "o-",
        color=COLORS["renyi"],
        lw=1.8,
        label=r"collision entropy $H_2$ (bits)",
    )
    ax.plot(
        depths,
        arrays["collision_entropy_deficit"],
        "s-",
        color=COLORS["qcap"],
        lw=1.8,
        label=r"deficit $m-H_2$ (bits)",
    )
    ax.axhline(m, color="0.7", lw=0.8, ls=":", label=f"$m={m}$ bits")

    density = ax.twinx()
    density.plot(
        depths,
        arrays["collision_entropy_density"],
        "d--",
        color=COLORS["exact"],
        lw=1.6,
        label=r"density $H_2/m$",
    )
    density.set_ylabel(r"collision-entropy density $H_2/m$")
    density.set_ylim(0, 1.05)

    ax.set_ylabel("bits")
    ax.set_title("Collision entropy, deficit, and density")
    _depth_axis(ax, depths)

    handles, labels = ax.get_legend_handles_labels()
    extra_handles, extra_labels = density.get_legend_handles_labels()
    ax.legend(
        handles + extra_handles,
        labels + extra_labels,
        frameon=False,
        fontsize=8.5,
    )

    fig.suptitle(
        f"{STEM}: ideal-distribution Rényi and entropy diagnostics versus depth\n"
        "one fixed QASM circuit -> one deterministic ideal value per depth",
        fontsize=11.5,
    )
    fig.tight_layout()
    fig.savefig(OUT_FACTORS, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return OUT_FACTORS


def figure_white_noise(depths, arrays, summary):
    """C. Direct test of the global white-noise MODEL, in two separate panels."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    ax = axes[0]
    ax.plot(
        depths,
        arrays["qcap_bound"],
        "-",
        color=COLORS["qcap"],
        lw=2.2,
        label=r"$\epsilon_{\rm QCAP}$",
    )
    ax.errorbar(
        depths,
        summary["mean_fitted_epsilon"],
        yerr=summary["std_fitted_epsilon"],
        fmt="o-",
        color=COLORS["measured"],
        capsize=3,
        lw=1.8,
        label=(
            r"mean best-fit $\epsilon$ $\pm$ 1 s.d. "
            "(RC/trajectory replicate scatter)"
        ),
    )
    ax.set_ylabel(r"white-noise parameter $\epsilon$")
    ax.set_title("Fitted white-noise parameter versus QCAP")
    _depth_axis(ax, depths)
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    ax.errorbar(
        depths,
        summary["mean_white_noise_residual_qcap"],
        yerr=summary["std_white_noise_residual_qcap"],
        fmt="s-",
        color=COLORS["qcap"],
        capsize=3,
        lw=1.8,
        label=r"residual at $\epsilon_{\rm QCAP}$: $D_{TV}(q, q_{\rm WN})$",
    )
    ax.errorbar(
        depths,
        summary["mean_white_noise_residual_fit"],
        yerr=summary["std_white_noise_residual_fit"],
        fmt="o-",
        color=COLORS["measured"],
        capsize=3,
        lw=1.8,
        label="residual at the best-fit $\\epsilon$",
    )
    ax.plot(
        depths,
        summary["trajectory_floor"],
        ":",
        color=COLORS["floor"],
        lw=2.0,
        marker="d",
        ms=5,
        label="trajectory TVD floor (estimator context)",
    )
    ax.set_yscale("log")
    ax.set_ylabel("white-noise-model residual (TVD)")
    ax.set_title("How well the white-noise MIXTURE describes the noisy output")
    _depth_axis(ax, depths)
    ax.legend(frameon=False, fontsize=8.5)

    fig.suptitle(
        f"{STEM}: direct global-white-noise-model test\n"
        "epsilon values and residual TVDs are on separate axes; residuals compare "
        "the full distributions, not only their scalar TVD",
        fontsize=11.5,
    )
    fig.tight_layout()
    fig.savefig(OUT_WHITE_NOISE, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return OUT_WHITE_NOISE


def figure_haar_summary(depths, arrays):
    """E. Haar-comparison summary: three quantities on three panels."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    axes[0].plot(
        depths,
        arrays["collision_ratio_to_haar"],
        "o-",
        color=COLORS["renyi"],
        lw=1.8,
    )
    axes[0].axhline(1.0, color=COLORS["haar"], ls="-.", lw=1.5, label="Haar value 1")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\sum_x p(x)^2 \, / \, [2/(D+1)]$")
    axes[0].set_title("Collision probability relative to Haar")
    axes[0].legend(frameon=False, fontsize=8.5)

    axes[1].plot(
        depths,
        arrays["renyi_factor_minus_haar"],
        "s-",
        color=COLORS["exact"],
        lw=1.8,
    )
    axes[1].axhline(0.0, color=COLORS["haar"], ls="-.", lw=1.5, label="Haar value 0")
    axes[1].set_ylabel(r"$S_2 - S_2^{\rm Haar}$")
    axes[1].set_title("Rényi factor minus finite-$D$ Haar")
    axes[1].legend(frameon=False, fontsize=8.5)

    axes[2].plot(
        depths,
        arrays["porter_thomas_ks_statistic"],
        "d-",
        color=COLORS["measured"],
        lw=1.8,
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("KS statistic vs Exp(1)")
    axes[2].set_title(r"Porter-Thomas KS statistic for $z=Dp(x)$")

    for ax in axes:
        _depth_axis(ax, depths)

    fig.suptitle(
        f"{STEM}: Porter-Thomas / Haar-like output-probability diagnostics\n"
        "these describe the ideal OUTPUT PROBABILITIES; they are not a claim that "
        "the circuit unitary is Haar-random",
        fontsize=11.5,
    )
    fig.tight_layout()
    fig.savefig(OUT_STATS, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return OUT_STATS


def figure_porter_thomas(depths, arrays, ideal_by_depth, meta):
    """D. One histogram of z = D p(x) per selected depth, against Exp(1)."""
    outputs = []

    m = meta["n_measured"]
    d = 2 ** m

    depth_index = {int(depth): index for index, depth in enumerate(depths)}

    for depth in PORTER_THOMAS_DEPTHS:
        if int(depth) not in depth_index:
            print(
                f"  Porter-Thomas: depth {depth} is not in DEPTHS, skipping "
                "(no ideal distribution was computed for it)"
            )
            continue

        index = depth_index[int(depth)]
        z = d * ideal_by_depth[int(depth)]

        fig, ax = plt.subplots(figsize=(8.0, 5.2))

        upper = float(max(np.percentile(z, 99.5), 5.0))

        ax.hist(
            z,
            bins=60,
            range=(0.0, upper),
            density=True,
            color=COLORS["measured"],
            alpha=0.55,
            label=r"ideal $z=D\,p(x)$",
        )

        grid = np.linspace(0.0, upper, 400)
        ax.plot(
            grid,
            np.exp(-grid),
            "-",
            color=COLORS["exact"],
            lw=2.2,
            label=r"Porter-Thomas density $e^{-z}$",
        )

        ax.set_yscale("log")
        ax.set_xlabel(r"$z = D\,p(x)$")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.15)

        ax.set_title(
            f"{STEM}: Porter-Thomas/Haar-like output probabilities, depth {int(depth)}\n"
            f"KS = {arrays['porter_thomas_ks_statistic'][index]:.4f} "
            f"(p = {arrays['porter_thomas_ks_pvalue'][index]:.3g}), "
            f"collision/Haar = {arrays['collision_ratio_to_haar'][index]:.3f}, "
            f"$S_2$ = {arrays['renyi_sqrt_factor_unclipped'][index]:.4f} "
            f"($S_2^{{\\rm Haar}}$ = {arrays['haar_renyi_factor'][index]:.4f})",
            fontsize=10.5,
        )

        ax.annotate(
            f"mean $z$ = {arrays['mean_scaled_probability'][index]:.4f}\n"
            f"var $z$ = {arrays['variance_scaled_probability'][index]:.4f}\n"
            "Exp(1): mean 1, var 1",
            xy=(0.62, 0.72),
            xycoords="axes fraction",
            fontsize=9,
        )

        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()

        path = porter_thomas_path(int(depth))
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)

    return outputs


# =============================================================================
# Saved data
# =============================================================================

def save_all(depths, arrays, summary, meta):
    """Write the .npz. Per-depth ideal quantities are 1-D; replicate arrays are 2-D."""
    kw = {
        # --- axes and circuit identity -------------------------------------
        "depths": np.asarray(depths, dtype=int),
        "cycle_depths": arrays["cycle_depths"],
        "twoq_gates": arrays["twoq_gates"],
        "measured_qubits": np.asarray(meta["measured"], dtype=int),
        "n_measured_qubits": int(meta["n_measured"]),
        "ideal_support": arrays["ideal_support"],
        # --- bounds ---------------------------------------------------------
        "qcap_bound": arrays["qcap_bound"],
        "qcap_bound_std": arrays["qcap_bound_std"],
        "exact_white_noise_prediction": arrays["exact_white_noise_prediction"],
        "renyi_bound": arrays["renyi_bound"],
        # --- measured TVD and estimator floor --------------------------------
        "tvd_points": arrays["tvd_points"],
        "tvd_mean": arrays["tvd_points"].mean(axis=1),
        "tvd_std": arrays["tvd_points"].std(
            axis=1, ddof=1 if arrays["tvd_points"].shape[1] > 1 else 0),
        "trajectory_floor": summary["trajectory_floor"],
        "trajectory_floor_std": summary["trajectory_floor_std"],
        "trajectory_floor_points": arrays["trajectory_floor_points"],
        "twirled_tvd_points": arrays["twirled_tvd_points"],
        "ran_twirled_series": bool(RUN_TWIRLED_SERIES),
        # --- ideal-distribution quantities (deterministic per depth) ---------
        "ideal_uniform_tvd": arrays["uniform_tvd"],
        "ideal_collision_probability": arrays["collision_probability"],
        "renyi2_divergence": arrays["renyi2_divergence"],
        "renyi_sqrt_factor_unclipped": arrays["renyi_sqrt_factor_unclipped"],
        "renyi_sqrt_factor_clipped": arrays["renyi_sqrt_factor_clipped"],
        "collision_entropy": arrays["collision_entropy"],
        "collision_entropy_density": arrays["collision_entropy_density"],
        "collision_entropy_deficit": arrays["collision_entropy_deficit"],
        # --- Haar / Porter-Thomas -------------------------------------------
        "haar_collision_probability": arrays["haar_collision_probability"],
        "haar_renyi_factor": arrays["haar_renyi_factor"],
        "collision_ratio_to_haar": arrays["collision_ratio_to_haar"],
        "renyi_factor_minus_haar": arrays["renyi_factor_minus_haar"],
        "porter_thomas_ks_statistic": arrays["porter_thomas_ks_statistic"],
        "porter_thomas_ks_pvalue": arrays["porter_thomas_ks_pvalue"],
        "mean_scaled_probability": arrays["mean_scaled_probability"],
        "variance_scaled_probability": arrays["variance_scaled_probability"],
        # --- white-noise-model test (per replicate, plus summaries) ----------
        "epsilon_white_noise_fit": arrays["epsilon_white_noise_fit"],
        "white_noise_model_residual_fit": arrays["white_noise_model_residual_fit"],
        "white_noise_model_residual_qcap": arrays["white_noise_model_residual_qcap"],
        # --- settings, seeds, and noise-model parameters ---------------------
        "qasm_path": QASM,
        "qasm_stem": STEM,
        "n_instances": N_INSTANCES,
        "n_rc_realizations": N_RC_REALIZATIONS,
        "n_traj": N_TRAJ,
        "total_trajectories_per_replicate": N_RC_REALIZATIONS * N_TRAJ,
        "floor_pairs": FLOOR_PAIRS,
        "cb_depths": np.asarray(CB_DEPTHS, dtype=int),
        "cb_shots": CB_SHOTS,
        "cb_decays": CB_DECAYS,
        "porter_thomas_depths": np.asarray(PORTER_THOMAS_DEPTHS, dtype=int),
        "seed": SEED,
        "theta_zz": THETA_ZZ,
        "noise_p1": NOISE.p1,
        "noise_p2": NOISE.p2,
        "noise_p_readout": NOISE.p_readout,
        "noise_p_idle": NOISE.p_idle,
        "noise_p_z1": NOISE.p_z1,
        "noise_p_zz": NOISE.p_zz,
        "noise_theta_1q": NOISE.theta_1q,
        "noise_theta_zz": NOISE.theta_zz,
        "readout_fidelity": meta["readout_fid"],
        "readout_fidelity_std": meta["readout_std"],
        "cycles_per_repetition": meta["cycles_per_rep"],
        "twoq_gates_per_repetition": meta["twoq_per_rep"],
        "cycle_names": np.asarray(sorted(meta["counts_one_rep"]), dtype=object),
        "cycle_counts_one_repetition": np.asarray(
            [meta["counts_one_rep"][k] for k in sorted(meta["counts_one_rep"])],
            dtype=int,
        ),
        "cycle_e_F": np.asarray(
            [meta["cycle_efs"][k][0] for k in sorted(meta["counts_one_rep"])],
            dtype=float,
        ),
        "cycle_e_F_std": np.asarray(
            [meta["cycle_efs"][k][1] for k in sorted(meta["counts_one_rep"])],
            dtype=float,
        ),
        "cycle_labels": np.asarray(
            [meta["cycle_labels"][k] for k in sorted(meta["counts_one_rep"])],
            dtype=object,
        ),
        "restricted_twirl_families": np.asarray(
            meta["restricted_families"], dtype=object
        ),
    }

    kw.update(
        {
            key: summary[key]
            for key in (
                "mean_fitted_epsilon",
                "std_fitted_epsilon",
                "mean_white_noise_residual_fit",
                "std_white_noise_residual_fit",
                "mean_white_noise_residual_qcap",
                "std_white_noise_residual_qcap",
                "qcap_minus_fitted_epsilon",
            )
        }
    )

    return save_results(OUT_DATA, **kw)


def report_diagnostics(depths, arrays, summary, meta):
    """Print the scrambling / white-noise diagnostics as two DISTINCT claims."""
    print(
        "\n=== A. are the IDEAL output probabilities Porter-Thomas/Haar-like? ==="
    )
    print(
        f"{'depth':>6}{'S2':>10}{'S2_Haar':>10}{'S2-Haar':>10}"
        f"{'coll/Haar':>13}{'KS':>9}{'KS p':>10}{'mean z':>10}{'var z':>14}"
    )

    for index, depth in enumerate(depths):
        print(
            f"{int(depth):>6}"
            f"{arrays['renyi_sqrt_factor_unclipped'][index]:>10.4f}"
            f"{arrays['haar_renyi_factor'][index]:>10.4f}"
            f"{arrays['renyi_factor_minus_haar'][index]:>10.4f}"
            f"{arrays['collision_ratio_to_haar'][index]:>13.4f}"
            f"{arrays['porter_thomas_ks_statistic'][index]:>9.4f}"
            f"{arrays['porter_thomas_ks_pvalue'][index]:>10.3g}"
            f"{arrays['mean_scaled_probability'][index]:>10.4f}"
            f"{arrays['variance_scaled_probability'][index]:>14.4f}"
        )

    print(
        "\n=== B. is the NOISY output well modeled by global white noise? ==="
    )
    print(
        f"{'depth':>6}{'eps_QCAP':>11}{'eps_fit':>11}{'sd eps_fit':>12}"
        f"{'QCAP-fit':>11}{'res(QCAP)':>12}{'res(fit)':>11}{'floor':>10}"
    )

    for index, depth in enumerate(depths):
        print(
            f"{int(depth):>6}"
            f"{arrays['qcap_bound'][index]:>11.5f}"
            f"{summary['mean_fitted_epsilon'][index]:>11.5f}"
            f"{summary['std_fitted_epsilon'][index]:>12.2e}"
            f"{summary['qcap_minus_fitted_epsilon'][index]:>11.5f}"
            f"{summary['mean_white_noise_residual_qcap'][index]:>12.5f}"
            f"{summary['mean_white_noise_residual_fit'][index]:>11.5f}"
            f"{summary['trajectory_floor'][index]:>10.5f}"
        )

    print(
        "\nA and B test DIFFERENT properties. S2 close to 1/2 (or to the finite-D "
        "Haar value) says the ideal output probabilities are Porter-Thomas-like; it "
        "does NOT by itself imply a small white-noise residual, and a small residual "
        "does not imply Haar-like ideal probabilities."
    )
    print(
        "Residuals are meaningful only down to the trajectory floor: a residual at "
        "or below the floor column is consistent with an exact white-noise mixture "
        "at this Monte Carlo sample size."
    )


def main():
    self_test()

    if RUN_TWIRLED_SERIES:
        print(
            f"theta_zz = {THETA_ZZ} rad is non-zero, so the already-Pauli-twirled "
            "series is run as a separately labeled comparison alongside explicit RC.\n"
        )
    else:
        print(
            "theta_zz = 0, so the physical noise model is already Pauli-stochastic "
            "and simulation under NOISE.twirled() would duplicate the explicit-RC "
            "series; that redundant series is skipped.\n"
        )

    arrays, ideal_by_depth, meta = run_sweep()

    depths = np.asarray(DEPTHS, dtype=int)
    summary = white_noise_summary(arrays)

    report_diagnostics(depths, arrays, summary, meta)

    out_data = save_all(depths, arrays, summary, meta)

    figures = [
        figure_main(depths, arrays, summary, meta),
        figure_renyi_factors(depths, arrays, meta),
        figure_white_noise(depths, arrays, summary),
        figure_haar_summary(depths, arrays),
    ]

    figures.extend(
        figure_porter_thomas(depths, arrays, ideal_by_depth, meta)
    )

    print(
        f"\nvalidation: D_TV(p,u) <= min(1, S_2) verified on {CHECKED['factors']} "
        f"distributions; exact-WN <= Rényi <= raw QCAP and 0 <= bound <= 1 verified "
        f"on {CHECKED['bounds']} depths; closed-form self-test passed "
        f"(tolerance {TOL:g})"
    )

    if meta["restricted_families"]:
        print(
            "caveat: restricted (Z-subgroup) Pauli twirl on "
            + ", ".join(meta["restricted_families"])
            + " -- the full-Pauli-twirl assumption behind QCAP holds only "
            "approximately for those gates."
        )

    print("\nWrote:")

    for path in (out_data, *figures):
        print(f"  {path}")


if __name__ == "__main__":
    main()
