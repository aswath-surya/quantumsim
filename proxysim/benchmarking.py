"""Statevector/trajectory benchmarking and QCAP utilities.

This module is a replacement for the Stim-specific ``benchmarking.py``.  It is
designed to work with the generalized ``proxysim.noise.NoiseModel`` and the
statevector trajectory path.

What changes relative to the old file
-------------------------------------
* No import of Stim and no call to ``to_stim_noisy``.
* The same ``sample_trajectory`` noise injector used by target-circuit
  simulations is also used during benchmarking.
* Coherent, gate-dependent, pair-dependent, spectator, and stochastic Pauli
  errors can therefore contribute to the fitted decay.
* Asymmetric readout fidelity is handled directly.
* The public functions retain the old names:
      cycle_benchmark
      readout_fidelity
      qcap_bound
      randomly_compile

Important methodological note
-----------------------------
The statevector implementation below estimates a decay from the fidelity
between ideal and noisy outputs, averaged over random product-Pauli input
states and noise trajectories.  For a Pauli-twirled/approximately depolarizing
cycle this has the expected exponential form and its fitted polarization can
be converted to process infidelity.  For strongly gate-dependent, non-Markovian,
or coherent noise the decay can be non-exponential; the returned RMSE and
``fit_ok`` flag should then be inspected.

The circuit IR is assumed to provide:
    Circuit(n_qubits, name=...)
    circuit.add(name, *qubits, *params)
    circuit.gates
    circuit.layers()

The statevector backend is assumed to provide ``_build(circuit)``, returning a
Qiskit-compatible circuit.  If your backend has a public method for this, edit
only ``_statevector``.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from qiskit.quantum_info import Statevector

from .backends import StatevectorBackend
from .noise import apply_readout, sample_trajectory

_ONEQ_RANDOM = ("h", "s", "sdg", "x", "y", "z", "sx", "sxdg")
_ONEQ_STRUCTURED = ("i", "h", "x")
_PAULIS = ("I", "X", "Y", "Z")

_SV = StatevectorBackend()


# ---------------------------------------------------------------------------
# Circuit helpers
# ---------------------------------------------------------------------------
def _new_circuit(n: int, name: str):
    from .circuit import Circuit

    return Circuit(n, name=name)


def _append_gate(circuit, name: str, qubits: Sequence[int], params: Sequence[float] = ()):
    """Single adapter point for the local circuit IR."""
    circuit.add(name, *qubits, *params)


def _append_circuit(destination, source):
    for gate in source.gates:
        destination.gates.append(gate)


def _copy_circuit(circuit, name_suffix: str = "_copy"):
    copied = _new_circuit(circuit.n_qubits, circuit.name + name_suffix)
    _append_circuit(copied, circuit)
    return copied


def _statevector(circuit) -> np.ndarray:
    """Return a normalized complex statevector for a proxysim Circuit."""
    qiskit_circuit = _SV._build(circuit)
    psi = np.asarray(Statevector(qiskit_circuit).data, dtype=complex)
    norm = float(np.linalg.norm(psi))
    if norm <= 0.0:
        raise ValueError("Statevector has zero norm.")
    return psi / norm


# ---------------------------------------------------------------------------
# Random product-Pauli eigenstate preparation
# ---------------------------------------------------------------------------
def _random_product_pauli_state(n: int, rng: random.Random):
    """Return a prep circuit for a random product eigenstate of X, Y, or Z.

    The eigenvalue sign is randomized as well.  This gives a broad, inexpensive
    set of stabilizer inputs without requiring Stim.
    """
    prep = _new_circuit(n, "cb_prep")

    for qubit in range(n):
        axis = rng.choice(("X", "Y", "Z"))
        negative = bool(rng.getrandbits(1))

        if axis == "Z":
            if negative:
                _append_gate(prep, "x", [qubit])

        elif axis == "X":
            if negative:
                _append_gate(prep, "x", [qubit])
            _append_gate(prep, "h", [qubit])

        elif axis == "Y":
            if negative:
                _append_gate(prep, "x", [qubit])
            _append_gate(prep, "h", [qubit])
            _append_gate(prep, "s", [qubit])

    return prep


def _build_round(
    n: int,
    oneq: Sequence[str],
    pairs: Sequence[Tuple[int, int]],
    twoq: str,
):
    round_circuit = _new_circuit(n, "cb_round")

    for qubit, gate_name in enumerate(oneq):
        if gate_name != "i":
            _append_gate(round_circuit, gate_name, [qubit])

    for a, b in pairs:
        _append_gate(round_circuit, twoq.lower(), [a, b])

    return round_circuit


def _build_sequence(
    n: int,
    rounds: Sequence[Sequence[str]],
    pairs: Sequence[Tuple[int, int]],
    twoq: str,
):
    sequence = _new_circuit(n, "cb_sequence")
    for oneq in rounds:
        _append_circuit(sequence, _build_round(n, oneq, pairs, twoq))
    return sequence


# ---------------------------------------------------------------------------
# Cycle benchmarking using statevector trajectories
# ---------------------------------------------------------------------------
def cycle_benchmark(
    pairs,
    n: int,
    depths,
    noise,
    mode: str = "random",
    n_decays: int = 30,
    shots: int = 1500,
    seed: int = 0,
    twoq: str = "CZ",
    n_trajectories: int | None = None,
    fit_rmse_tolerance: float = 0.05,
) -> dict:
    """Estimate the process infidelity of one dressed cycle.

    Parameters
    ----------
    pairs:
        Two-qubit pairs in the entangling layer.
    n:
        Number of qubits.
    depths:
        Sequence lengths used in the decay fit.
    noise:
        Generalized NoiseModel consumed by ``sample_trajectory``.
    mode:
        ``"random"`` draws one-qubit gates from a Clifford-like set;
        ``"structured"`` draws from ``{I,H,X}``.
    n_decays:
        Number of random input/sequence realizations per depth.
    shots:
        Retained for API compatibility.  In the statevector path this controls
        the default number of Monte-Carlo trajectories if ``n_trajectories`` is
        not supplied.  It is intentionally capped to avoid accidental enormous
        runs.
    n_trajectories:
        Noise trajectories per random sequence.  Defaults to
        ``max(8, min(64, shots // 100))``.
    """
    if mode not in {"random", "structured"}:
        raise ValueError("mode must be 'random' or 'structured'.")

    if n_trajectories is None:
        n_trajectories = max(8, min(64, int(shots) // 100))

    rng = random.Random(seed)
    oneq_set = _ONEQ_RANDOM if mode == "random" else _ONEQ_STRUCTURED

    decay: List[float] = []
    decay_std: List[float] = []

    for depth in depths:
        sequence_values: List[float] = []

        for _ in range(n_decays):
            prep = _random_product_pauli_state(n, rng)
            rounds = [
                [rng.choice(oneq_set) for _ in range(n)]
                for _ in range(int(depth))
            ]
            ideal_sequence = _build_sequence(n, rounds, pairs, twoq)

            ideal_full = _new_circuit(n, "cb_ideal")
            _append_circuit(ideal_full, prep)
            _append_circuit(ideal_full, ideal_sequence)
            ideal_psi = _statevector(ideal_full)

            trajectory_fidelities = []
            for _traj in range(n_trajectories):
                noisy_sequence = sample_trajectory(ideal_sequence, noise, rng)

                noisy_full = _new_circuit(n, "cb_noisy")
                _append_circuit(noisy_full, prep)
                _append_circuit(noisy_full, noisy_sequence)

                noisy_psi = _statevector(noisy_full)
                fidelity = float(abs(np.vdot(ideal_psi, noisy_psi)) ** 2)
                trajectory_fidelities.append(fidelity)

            sequence_values.append(float(np.mean(trajectory_fidelities)))

        decay.append(float(np.mean(sequence_values)))
        decay_std.append(
            float(np.std(sequence_values, ddof=1))
            if len(sequence_values) > 1
            else 0.0
        )

    depth_array = np.asarray(depths, dtype=float)
    decay_array = np.asarray(decay, dtype=float)
    std_array = np.asarray(decay_std, dtype=float)

    f, f_std, amplitude, rmse = _fit_decay_with_floor(
        depth_array,
        decay_array,
        dimension=2**n,
        sigma=std_array,
    )

    # Same process-polarization conversion used by the previous implementation.
    factor = 1.0 - 4.0 ** (-n)
    e_F = factor * (1.0 - f)
    e_F_std = factor * f_std

    hilbert_dimension = 2.0**n
    r = hilbert_dimension / (hilbert_dimension + 1.0) * e_F

    return {
        "depths": list(depths),
        "decay": decay,
        "decay_std": decay_std,
        "f": f,
        "f_std": f_std,
        "A": amplitude,
        "fit_rmse": rmse,
        "fit_ok": bool(np.isfinite(rmse) and rmse <= fit_rmse_tolerance),
        "e_F": e_F,
        "e_F_std": e_F_std,
        "F_e": 1.0 - e_F,
        "r": r,
        "F_avg": 1.0 - r,
        "n_trajectories": int(n_trajectories),
        "method": "statevector trajectory survival",
    }


def _fit_decay_with_floor(m, y, dimension: int, sigma=None):
    """Fit y(m) = floor + A f^m.

    For ideal-state survival under a depolarizing channel, the asymptotic floor
    is 1/d.  Fixing the floor stabilizes the fit and separates complete mixing
    from a zero-valued Pauli expectation decay.
    """
    from scipy.optimize import curve_fit

    floor = 1.0 / float(dimension)

    def model(depth, amplitude, polarization):
        return floor + amplitude * polarization**depth

    y = np.asarray(y, dtype=float)
    m = np.asarray(m, dtype=float)

    sigma_arg = None
    absolute_sigma = False
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.shape == y.shape and np.any(sigma > 0.0):
            positive = sigma[sigma > 0.0]
            replacement = float(np.median(positive))
            sigma_arg = np.where(sigma > 0.0, sigma, replacement)
            absolute_sigma = True

    amplitude0 = float(np.clip(y[0] - floor, 1e-6, 1.2))
    try:
        popt, pcov = curve_fit(
            model,
            m,
            y,
            p0=[amplitude0, 0.99],
            bounds=([0.0, 0.0], [1.2, 1.0]),
            sigma=sigma_arg,
            absolute_sigma=absolute_sigma,
            maxfev=20_000,
        )
        amplitude, f = map(float, popt)
        f_std = float(np.sqrt(max(float(pcov[1, 1]), 0.0)))
    except Exception:
        shifted = y - floor
        mask = shifted > 1e-8
        if np.count_nonzero(mask) < 2:
            amplitude = amplitude0
            f = 1.0
            f_std = 0.0
        else:
            slope, intercept = np.polyfit(m[mask], np.log(shifted[mask]), 1)
            f = float(np.clip(np.exp(slope), 0.0, 1.0))
            amplitude = float(np.exp(intercept))
            f_std = 0.0

    residual = y - model(m, amplitude, f)
    rmse = float(np.sqrt(np.mean(residual**2)))

    return f, f_std, amplitude, rmse


# ---------------------------------------------------------------------------
# Readout/SPAM fidelity
# ---------------------------------------------------------------------------
def readout_fidelity(
    n: int,
    noise,
    shots: int = 4000,
    seed: int = 0,
    max_states: int = 128,
):
    """Estimate mean diagonal readout fidelity with asymmetric assignment error."""
    rng = random.Random(seed)

    if 2**n <= max_states:
        states = list(range(2**n))
    else:
        states = [rng.randrange(2**n) for _ in range(max_states)]

    diagonal = []

    for basis_index in states:
        ideal_bits = format(basis_index, f"0{n}b")
        correct = 0

        for _ in range(shots):
            measured = apply_readout(ideal_bits, noise, rng)
            correct += int(measured == ideal_bits)

        diagonal.append(correct / shots)

    return float(np.mean(diagonal)), float(np.std(diagonal, ddof=0))


# ---------------------------------------------------------------------------
# QCAP bound and corrected uncertainty propagation
# ---------------------------------------------------------------------------
def qcap_bound(
    cycle_counts: Dict[str, int],
    cycle_efs: Dict[str, Tuple[float, float]],
    ro_fid: float,
    ro_std: float,
) -> dict:
    """Compute 1 - F_RO prod_c (1-e_F,c)^n_c.

    Independent first-order uncertainties are propagated through the product.
    """
    fidelity = float(ro_fid)
    variance = float(ro_std) ** 2

    for cycle_name, count in cycle_counts.items():
        e_F, e_F_std = cycle_efs[cycle_name]
        count = int(count)

        base = 1.0 - float(e_F)
        factor = base**count

        derivative = -count * base ** (count - 1) if count > 0 else 0.0
        factor_variance = derivative**2 * float(e_F_std) ** 2

        # Var(XY) to first order for independent X,Y.
        variance = factor**2 * variance + fidelity**2 * factor_variance
        fidelity *= factor

    return {
        "error": 1.0 - fidelity,
        "std": math.sqrt(max(variance, 0.0)),
        "fidelity": fidelity,
    }


# ---------------------------------------------------------------------------
# Randomized compiling
# ---------------------------------------------------------------------------
_CZ_CONJUGATION = {
    ("I", "I"): ("I", "I"),
    ("I", "X"): ("Z", "X"),
    ("I", "Y"): ("Z", "Y"),
    ("I", "Z"): ("I", "Z"),
    ("X", "I"): ("X", "Z"),
    ("X", "X"): ("Y", "Y"),
    ("X", "Y"): ("Y", "X"),
    ("X", "Z"): ("X", "I"),
    ("Y", "I"): ("Y", "Z"),
    ("Y", "X"): ("X", "Y"),
    ("Y", "Y"): ("X", "X"),
    ("Y", "Z"): ("Y", "I"),
    ("Z", "I"): ("Z", "I"),
    ("Z", "X"): ("I", "X"),
    ("Z", "Y"): ("I", "Y"),
    ("Z", "Z"): ("Z", "Z"),
}


def _append_pauli(circuit, label: str, qubit: int):
    if label != "I":
        _append_gate(circuit, label.lower(), [qubit])


def randomly_compile(circuit, n_compilations: int, seed: int = 0):
    """Return CZ-Pauli-twirled circuit copies.

    This implementation inserts an independently sampled Pauli pair before each
    CZ and its propagated correction after the CZ, preserving the ideal circuit
    action up to an irrelevant global phase.

    Other two-qubit gates are copied unchanged.  Add a conjugation rule before
    claiming RC support for them.
    """
    rng = random.Random(seed)
    compiled = []

    for compilation_index in range(n_compilations):
        rc = _new_circuit(
            circuit.n_qubits,
            f"{circuit.name}_rc_{compilation_index}",
        )

        for gate in circuit.gates:
            if gate.name.lower() == "cz" and len(gate.qubits) == 2:
                q0, q1 = gate.qubits
                before = (rng.choice(_PAULIS), rng.choice(_PAULIS))
                after = _CZ_CONJUGATION[before]

                _append_pauli(rc, before[0], q0)
                _append_pauli(rc, before[1], q1)
                rc.gates.append(gate)
                _append_pauli(rc, after[0], q0)
                _append_pauli(rc, after[1], q1)
            else:
                rc.gates.append(gate)

        compiled.append(rc)

    return compiled

# """Cycle benchmarking, randomized compiling, and the QCAP performance bound.

# A hardware-free reimplementation of the pipeline in akelhashim/qcal's
# `circuit_bounding` notebook, which itself orchestrates Keysight True-Q
# (`trueq.make_cb`, `trueq.randomly_compile`, True-Q's `qcap_bound`). True-Q is
# closed source, so this is the protocol rebuilt from first principles and driven
# by a `proxysim.noise.NoiseModel` instead of a QPU. Because cycle benchmarking
# targets Clifford cycles under Pauli noise, stim is an exact simulator for it.

# The three pieces mirror the notebook:

#   cycle_benchmark(cycle, noise)   -> e_F, the process infidelity of one dressed
#                                      cycle (a single-qubit layer + an entangling
#                                      layer), from an exponential fit that is
#                                      robust to SPAM.
#   readout_fidelity(n, noise)      -> the SPAM (readout) fidelity, from a synthetic
#                                      confusion matrix.
#   qcap_bound(counts, e_Fs, ro)    -> an UPPER bound on circuit error:
#                                      1 - ro_fid * prod_c (1 - e_F_c)^(times c used)
#                                      (a straight port of the notebook's function).

# Note on twirling: on hardware you Pauli-twirl (randomly compile) to turn coherent
# error into a Pauli-stochastic channel so this all holds. Here the noise model is
# already Pauli-stochastic, so twirling is unnecessary for correctness; `randomly_compile`
# is provided for completeness and to mirror the workflow.
# """

# from __future__ import annotations

# import math
# import random
# from typing import Dict, List, Tuple

# import numpy as np
# import stim

# from .backends.stabilizer import _SIMPLE  # IR gate name -> stim name

# _ONEQ_RANDOM = ["h", "s", "sdg", "x", "y", "z", "sx", "sxdg"]
# _ONEQ_STRUCTURED = ["i", "h", "x"]


# # ---------------------------------------------------------------------------
# # Pauli eigenstate prep / measurement rotation
# # ---------------------------------------------------------------------------
# def _letters(p: stim.PauliString) -> str:
#     return "".join("_XYZ"[p[i]] for i in range(len(p)))


# def _rand_pauli(n: int, rng: random.Random) -> str:
#     while True:
#         s = "".join(rng.choice("_XYZ") for _ in range(n))
#         if s.strip("_"):  # reject the all-identity Pauli
#             return s


# def _prep_eigenstate(c: stim.Circuit, letters: str):
#     """Append gates preparing the +1 eigenstate of the Pauli `letters` from |0..0>."""
#     for i, ch in enumerate(letters):
#         if ch == "X":
#             c.append("H", [i])
#         elif ch == "Y":
#             c.append("H", [i])
#             c.append("S", [i])          # S H |0> = |+i>, the +1 eigenstate of Y


# def _measure_rotation(c: stim.Circuit, letters: str):
#     """Append gates rotating each Pauli factor to Z, so a Z-basis readout of the
#     support measures the Pauli."""
#     for i, ch in enumerate(letters):
#         if ch == "X":
#             c.append("H", [i])
#         elif ch == "Y":
#             c.append("S_DAG", [i])
#             c.append("H", [i])


# # ---------------------------------------------------------------------------
# # One dressed-cycle round (single-qubit layer + entangling layer), with noise
# # ---------------------------------------------------------------------------
# def _apply_round(c: stim.Circuit, oneq: List[str], pairs: List[Tuple[int, int]],
#                  n: int, noise, ideal: bool, twoq: str = "CZ"):
#     touched = set()
#     for q, g in enumerate(oneq):                    # single-qubit layer on every qubit
#         c.append(_SIMPLE[g], [q])
#         touched.add(q)
#         if not ideal and noise.p1 > 0:
#             c.append("DEPOLARIZE1", [q], noise.p1)
#     for a, b in pairs:                              # entangling layer (CZ or CX)
#         c.append(twoq, [a, b])
#         touched.update((a, b))
#         if not ideal and noise.p2 > 0:
#             c.append("DEPOLARIZE2", [a, b], noise.p2)
#     if not ideal and noise.p_idle > 0:              # idling dephasing on the rest
#         idle = [q for q in range(n) if q not in {q for pr in pairs for q in pr}]
#         if idle:
#             c.append("Z_ERROR", idle, noise.p_idle)


# # ---------------------------------------------------------------------------
# # Cycle benchmarking
# # ---------------------------------------------------------------------------
# def cycle_benchmark(pairs, n: int, depths, noise, mode: str = "random",
#                     n_decays: int = 30, shots: int = 1500, seed: int = 0,
#                     twoq: str = "CZ") -> dict:
#     """Estimate the process infidelity e_F of the dressed cycle whose entangling
#     layer is ``pairs`` (a list of qubit pairs), under ``noise``.

#     Protocol per sequence length m: for each of ``n_decays`` random Paulis, prepare
#     its +1 eigenstate, apply m noisy rounds (single-qubit twirl layer + the
#     entangling layer), and measure the Pauli it ideally maps to. Averaging the
#     signed survival over the sampled Paulis is a Monte-Carlo estimate of the
#     twirled-cycle survival -- the full Pauli group has d^2 = 4^n elements, far too
#     many to enumerate -- which is fit to  survival(m) = A * f^m.

#     The fitted ``f`` is the process *polarization*, not the infidelity. The process
#     fidelity is the average Pauli fidelity F_e = (1/d^2) sum_P f_P, so with d = 2^n
#     (Hashim et al. 2408.12064, Table II):

#         e_F = (d^2 - 1)/d^2 * (1 - f),   r = d/(d+1) * e_F,   F_avg = 1 - r.

#     SPAM robustness: state-prep and readout errors scale the amplitude A but not the
#     decay rate f, so e_F is SPAM-independent (readout enters the bound separately,
#     via ``readout_fidelity``).
#     """
#     rng = random.Random(seed)
#     oneq_set = _ONEQ_RANDOM if mode == "random" else _ONEQ_STRUCTURED
#     decay = []
#     for m in depths:
#         vals = []
#         for _ in range(n_decays):
#             P = _rand_pauli(n, rng)
#             rounds = [[rng.choice(oneq_set) for _ in range(n)] for _ in range(m)]

#             ideal = stim.Circuit()
#             for oneq in rounds:
#                 _apply_round(ideal, oneq, pairs, n, noise, ideal=True, twoq=twoq)
#             P_m = stim.PauliString(P).after(ideal.to_tableau(),
#                                             targets=range(n))   # C^m P C^-m (+/- sign)

#             noisy = stim.Circuit()
#             _prep_eigenstate(noisy, P)
#             for oneq in rounds:
#                 _apply_round(noisy, oneq, pairs, n, noise, ideal=False, twoq=twoq)
#             _measure_rotation(noisy, _letters(P_m))
#             if noise.p_readout > 0:
#                 noisy.append("X_ERROR", list(range(n)), noise.p_readout)
#             noisy.append("M", list(range(n)))

#             arr = noisy.compile_sampler(seed=rng.randint(0, 2**31 - 1)).sample(shots)
#             support = [i for i, ch in enumerate(_letters(P_m)) if ch != "_"]
#             parity = arr[:, support].sum(axis=1) % 2
#             exp = float(np.mean(1 - 2 * parity))          # <unsigned P_m>
#             vals.append(exp * (1 if P_m.sign.real > 0 else -1))  # sign-correct: ideal = +1
#         decay.append(float(np.mean(vals)))

#     f, f_std = _fit_decay(np.array(depths, float), np.array(decay))
#     factor = 1.0 - 4.0 ** (-n)                 # (d^2 - 1)/d^2 with d = 2^n
#     e_F = factor * (1.0 - f)
#     e_F_std = factor * f_std
#     d = 2.0 ** n
#     r = d / (d + 1.0) * e_F                     # average gate infidelity (Table II)
#     return {"depths": list(depths), "decay": decay, "f": f,
#             "e_F": e_F, "e_F_std": e_F_std, "F_e": 1.0 - e_F,
#             "r": r, "F_avg": 1.0 - r}


# def _fit_decay(m, y):
#     """Fit y = A * f^m for f in (0,1], A in (0,1.2]. Returns (f, std_f)."""
#     from scipy.optimize import curve_fit

#     def model(m, A, f):
#         return A * f ** m

#     try:
#         popt, pcov = curve_fit(model, m, y, p0=[max(y[0], 0.5), 0.99],
#                                bounds=([0.0, 0.0], [1.2, 1.0]), maxfev=10000)
#         return float(popt[1]), float(np.sqrt(pcov[1, 1]))
#     except Exception:
#         # fall back to a log-linear fit over the positive points
#         mask = y > 1e-3
#         slope = np.polyfit(m[mask], np.log(y[mask]), 1)[0]
#         return float(np.exp(slope)), 0.0


# # ---------------------------------------------------------------------------
# # Readout (SPAM) fidelity from a synthetic confusion matrix
# # ---------------------------------------------------------------------------
# def readout_fidelity(n: int, noise, shots: int = 4000, seed: int = 0,
#                      max_states: int = 128):
#     """Prepare computational basis states, read them out through the readout error,
#     and return (mean diagonal, std diagonal) of the confusion matrix.

#     For n small enough, every basis state is used; beyond ``max_states`` a random
#     sample of states is taken (enumerating 2^n is infeasible past ~20 qubits, and
#     for independent bit-flip readout every diagonal element is the same anyway)."""
#     rng = random.Random(seed)
#     if 2 ** n <= max_states:
#         states = range(2 ** n)
#     else:
#         states = [rng.randrange(2 ** n) for _ in range(max_states)]
#     diag = []
#     for i in states:
#         bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
#         c = stim.Circuit()
#         for k, b in enumerate(bits):
#             if b:
#                 c.append("X", [k])
#         if noise.enabled and noise.p_readout > 0:
#             c.append("X_ERROR", list(range(n)), noise.p_readout)
#         c.append("M", list(range(n)))
#         arr = c.compile_sampler(seed=rng.randint(0, 2**31 - 1)).sample(shots)
#         correct = np.all(arr == np.array(bits), axis=1).mean()
#         diag.append(float(correct))
#     return float(np.mean(diag)), float(np.std(diag))


# # ---------------------------------------------------------------------------
# # QCAP bound (port of the notebook's qcap_bound)
# # ---------------------------------------------------------------------------
# def qcap_bound(cycle_counts: Dict[str, int], cycle_efs: Dict[str, Tuple[float, float]],
#                ro_fid: float, ro_std: float) -> dict:
#     """Upper bound on circuit error from per-cycle process fidelities.

#         fidelity_bound = ro_fid * prod_c (1 - e_F_c) ^ (n_c)
#         error_bound    = 1 - fidelity_bound

#     ``cycle_counts[c]`` is how many times cycle ``c`` appears in the circuit;
#     ``cycle_efs[c] = (e_F, e_F_std)``. Uncertainty is propagated exactly as in the
#     notebook (Var(Y^n) ~ n Y^(n-1) Var(Y), then the product rule).
#     """
#     p = ro_fid
#     pvar = ro_std ** 2
#     for cyc, n in cycle_counts.items():
#         e_F, std = cycle_efs[cyc]
#         y = (1 - e_F) ** n
#         yvar = n * (1 - e_F) ** (n - 1) * std ** 2
#         pvar = pvar * (yvar + y ** 2) + yvar * p ** 2
#         p *= y
#     return {"error": 1 - p, "std": math.sqrt(max(pvar, 0.0)), "fidelity": p}


# # ---------------------------------------------------------------------------
# # Randomized compiling (Pauli twirl) -- provided to mirror the workflow.
# # ---------------------------------------------------------------------------
# def randomly_compile(circuit, n_compilations: int, seed: int = 0):
#     """Return ``n_compilations`` Pauli-twirled copies of ``circuit``.

#     A twirl inserts random Paulis around each entangling layer and corrects them
#     on the neighbouring single-qubit layers, leaving the ideal action unchanged
#     while tailoring the physical error toward Pauli-stochastic. Our NoiseModel is
#     ALREADY Pauli-stochastic (depolarizing + dephasing + bit-flip readout), so the
#     output is already twirl-averaged and a real twirl would be a no-op on the
#     statistics. This function therefore returns verbatim copies -- it documents
#     where RC sits in the workflow but does not synthesize the twirl. (A functional
#     twirl only earns its keep once coherent errors are added to the noise model.)
#     The twirling CB actually relies on -- random single-qubit layers around each
#     cycle -- IS implemented, inside ``cycle_benchmark``.
#     """
#     from .circuit import Circuit

#     out = []
#     for _ in range(n_compilations):
#         nc = Circuit(circuit.n_qubits, name=circuit.name + "_rc")
#         for g in circuit.gates:
#             nc.gates.append(g)
#         out.append(nc)
#     return out
