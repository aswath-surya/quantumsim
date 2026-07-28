"""Cycle benchmarking, randomized compiling, and the QCAP performance bound.

A hardware-free reimplementation of the pipeline in akelhashim/qcal's
`circuit_bounding` notebook, which itself orchestrates Keysight True-Q
(`trueq.make_cb`, `trueq.randomly_compile`, True-Q's `qcap_bound`). True-Q is
closed source, so this is the protocol rebuilt from first principles and driven
by a `proxysim.noise.NoiseModel` instead of a QPU. Because cycle benchmarking
targets Clifford cycles under Pauli noise, stim is an exact simulator for it.

The three pieces mirror the notebook:

  cycle_benchmark(cycle, noise)   -> e_F, the process infidelity of one dressed
                                     cycle (a single-qubit layer + an entangling
                                     layer), from an exponential fit that is
                                     robust to SPAM.
  readout_fidelity(n, noise)      -> the SPAM (readout) fidelity, from a synthetic
                                     confusion matrix.
  qcap_bound(counts, e_Fs, ro)    -> an UPPER bound on circuit error:
                                     1 - ro_fid * prod_c (1 - e_F_c)^(times c used)
                                     (a straight port of the notebook's function).

Note on twirling: on hardware you Pauli-twirl (randomly compile) to turn coherent
error into a Pauli-stochastic channel so this all holds. Here the noise model is
already Pauli-stochastic, so twirling is unnecessary for correctness; `randomly_compile`
is provided for completeness and to mirror the workflow.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import numpy as np
import stim

from .backends.stabilizer import _SIMPLE  # IR gate name -> stim name

_ONEQ_RANDOM = ["h", "s", "sdg", "x", "y", "z", "sx", "sxdg"]
_ONEQ_STRUCTURED = ["i", "h", "x"]


# ---------------------------------------------------------------------------
# Pauli eigenstate prep / measurement rotation
# ---------------------------------------------------------------------------
def _letters(p: stim.PauliString) -> str:
    return "".join("_XYZ"[p[i]] for i in range(len(p)))


def _rand_pauli(n: int, rng: random.Random) -> str:
    while True:
        s = "".join(rng.choice("_XYZ") for _ in range(n))
        if s.strip("_"):  # reject the all-identity Pauli
            return s


def _prep_eigenstate(c: stim.Circuit, letters: str):
    """Append gates preparing the +1 eigenstate of the Pauli `letters` from |0..0>."""
    for i, ch in enumerate(letters):
        if ch == "X":
            c.append("H", [i])
        elif ch == "Y":
            c.append("H", [i])
            c.append("S", [i])          # S H |0> = |+i>, the +1 eigenstate of Y


def _measure_rotation(c: stim.Circuit, letters: str):
    """Append gates rotating each Pauli factor to Z, so a Z-basis readout of the
    support measures the Pauli."""
    for i, ch in enumerate(letters):
        if ch == "X":
            c.append("H", [i])
        elif ch == "Y":
            c.append("S_DAG", [i])
            c.append("H", [i])


# ---------------------------------------------------------------------------
# One dressed-cycle round (single-qubit layer + entangling layer), with noise
# ---------------------------------------------------------------------------
def _apply_round(c: stim.Circuit, oneq: List[str], pairs: List[Tuple[int, int]],
                 n: int, noise, ideal: bool, twoq: str = "CZ"):
    touched = set()
    for q, g in enumerate(oneq):                    # single-qubit layer on every qubit
        c.append(_SIMPLE[g], [q])
        touched.add(q)
        if not ideal and noise.p1 > 0:
            c.append("DEPOLARIZE1", [q], noise.p1)
    for a, b in pairs:                              # entangling layer (CZ or CX)
        c.append(twoq, [a, b])
        touched.update((a, b))
        if not ideal and noise.p2 > 0:
            c.append("DEPOLARIZE2", [a, b], noise.p2)
    if not ideal and noise.p_idle > 0:              # idling dephasing on the rest
        idle = [q for q in range(n) if q not in {q for pr in pairs for q in pr}]
        if idle:
            c.append("Z_ERROR", idle, noise.p_idle)


# ---------------------------------------------------------------------------
# Cycle benchmarking
# ---------------------------------------------------------------------------
def cycle_benchmark(pairs, n: int, depths, noise, mode: str = "random",
                    n_decays: int = 30, shots: int = 1500, seed: int = 0,
                    twoq: str = "CZ") -> dict:
    """Estimate the process infidelity e_F of the dressed cycle whose entangling
    layer is ``pairs`` (a list of qubit pairs), under ``noise``.

    Protocol per sequence length m: for each of ``n_decays`` random Paulis, prepare
    its +1 eigenstate, apply m noisy rounds (single-qubit twirl layer + the
    entangling layer), and measure the Pauli it ideally maps to. Averaging the
    signed survival over the sampled Paulis is a Monte-Carlo estimate of the
    twirled-cycle survival -- the full Pauli group has d^2 = 4^n elements, far too
    many to enumerate -- which is fit to  survival(m) = A * f^m.

    The fitted ``f`` is the process *polarization*, not the infidelity. The process
    fidelity is the average Pauli fidelity F_e = (1/d^2) sum_P f_P, so with d = 2^n
    (Hashim et al. 2408.12064, Table II):

        e_F = (d^2 - 1)/d^2 * (1 - f),   r = d/(d+1) * e_F,   F_avg = 1 - r.

    SPAM robustness: state-prep and readout errors scale the amplitude A but not the
    decay rate f, so e_F is SPAM-independent (readout enters the bound separately,
    via ``readout_fidelity``).
    """
    rng = random.Random(seed)
    oneq_set = _ONEQ_RANDOM if mode == "random" else _ONEQ_STRUCTURED
    decay = []
    for m in depths:
        vals = []
        for _ in range(n_decays):
            P = _rand_pauli(n, rng)
            rounds = [[rng.choice(oneq_set) for _ in range(n)] for _ in range(m)]

            ideal = stim.Circuit()
            for oneq in rounds:
                _apply_round(ideal, oneq, pairs, n, noise, ideal=True, twoq=twoq)
            P_m = stim.PauliString(P).after(ideal.to_tableau(),
                                            targets=range(n))   # C^m P C^-m (+/- sign)

            noisy = stim.Circuit()
            _prep_eigenstate(noisy, P)
            for oneq in rounds:
                _apply_round(noisy, oneq, pairs, n, noise, ideal=False, twoq=twoq)
            _measure_rotation(noisy, _letters(P_m))
            if noise.p_readout > 0:
                noisy.append("X_ERROR", list(range(n)), noise.p_readout)
            noisy.append("M", list(range(n)))

            arr = noisy.compile_sampler(seed=rng.randint(0, 2**31 - 1)).sample(shots)
            support = [i for i, ch in enumerate(_letters(P_m)) if ch != "_"]
            parity = arr[:, support].sum(axis=1) % 2
            exp = float(np.mean(1 - 2 * parity))          # <unsigned P_m>
            vals.append(exp * (1 if P_m.sign.real > 0 else -1))  # sign-correct: ideal = +1
        decay.append(float(np.mean(vals)))

    f, f_std = _fit_decay(np.array(depths, float), np.array(decay))
    factor = 1.0 - 4.0 ** (-n)                 # (d^2 - 1)/d^2 with d = 2^n
    e_F = factor * (1.0 - f)
    e_F_std = factor * f_std
    d = 2.0 ** n
    r = d / (d + 1.0) * e_F                     # average gate infidelity (Table II)
    return {"depths": list(depths), "decay": decay, "f": f,
            "e_F": e_F, "e_F_std": e_F_std, "F_e": 1.0 - e_F,
            "r": r, "F_avg": 1.0 - r}


def _fit_decay(m, y):
    """Fit y = A * f^m for f in (0,1], A in (0,1.2]. Returns (f, std_f)."""
    from scipy.optimize import curve_fit

    def model(m, A, f):
        return A * f ** m

    try:
        popt, pcov = curve_fit(model, m, y, p0=[max(y[0], 0.5), 0.99],
                               bounds=([0.0, 0.0], [1.2, 1.0]), maxfev=10000)
        return float(popt[1]), float(np.sqrt(pcov[1, 1]))
    except Exception:
        # fall back to a log-linear fit over the positive points
        mask = y > 1e-3
        slope = np.polyfit(m[mask], np.log(y[mask]), 1)[0]
        return float(np.exp(slope)), 0.0


# ---------------------------------------------------------------------------
# Readout (SPAM) fidelity from a synthetic confusion matrix
# ---------------------------------------------------------------------------
def readout_fidelity(n: int, noise, shots: int = 4000, seed: int = 0,
                     max_states: int = 128):
    """Prepare computational basis states, read them out through the readout error,
    and return (mean diagonal, std diagonal) of the confusion matrix.

    For n small enough, every basis state is used; beyond ``max_states`` a random
    sample of states is taken (enumerating 2^n is infeasible past ~20 qubits, and
    for independent bit-flip readout every diagonal element is the same anyway)."""
    rng = random.Random(seed)
    if 2 ** n <= max_states:
        states = range(2 ** n)
    else:
        states = [rng.randrange(2 ** n) for _ in range(max_states)]
    diag = []
    for i in states:
        bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
        c = stim.Circuit()
        for k, b in enumerate(bits):
            if b:
                c.append("X", [k])
        if noise.enabled and noise.p_readout > 0:
            c.append("X_ERROR", list(range(n)), noise.p_readout)
        c.append("M", list(range(n)))
        arr = c.compile_sampler(seed=rng.randint(0, 2**31 - 1)).sample(shots)
        correct = np.all(arr == np.array(bits), axis=1).mean()
        diag.append(float(correct))
    return float(np.mean(diag)), float(np.std(diag))


# ---------------------------------------------------------------------------
# QCAP bound (port of the notebook's qcap_bound)
# ---------------------------------------------------------------------------
def qcap_bound(cycle_counts: Dict[str, int], cycle_efs: Dict[str, Tuple[float, float]],
               ro_fid: float, ro_std: float) -> dict:
    """Upper bound on circuit error from per-cycle process fidelities.

        fidelity_bound = ro_fid * prod_c (1 - e_F_c) ^ (n_c)
        error_bound    = 1 - fidelity_bound

    ``cycle_counts[c]`` is how many times cycle ``c`` appears in the circuit;
    ``cycle_efs[c] = (e_F, e_F_std)``. Uncertainty is propagated exactly as in the
    notebook (Var(Y^n) ~ n Y^(n-1) Var(Y), then the product rule).
    """
    p = ro_fid
    pvar = ro_std ** 2
    for cyc, n in cycle_counts.items():
        e_F, std = cycle_efs[cyc]
        y = (1 - e_F) ** n
        yvar = n * (1 - e_F) ** (n - 1) * std ** 2
        pvar = pvar * (yvar + y ** 2) + yvar * p ** 2
        p *= y
    return {"error": 1 - p, "std": math.sqrt(max(pvar, 0.0)), "fidelity": p}


# ---------------------------------------------------------------------------
# Randomized compiling (Pauli twirl) -- provided to mirror the workflow.
# ---------------------------------------------------------------------------
def randomly_compile(circuit, n_compilations: int, seed: int = 0):
    """Return ``n_compilations`` Pauli-twirled copies of ``circuit``.

    A twirl inserts random Paulis around each entangling layer and corrects them
    on the neighbouring single-qubit layers, leaving the ideal action unchanged
    while tailoring the physical error toward Pauli-stochastic. Our NoiseModel is
    ALREADY Pauli-stochastic (depolarizing + dephasing + bit-flip readout), so the
    output is already twirl-averaged and a real twirl would be a no-op on the
    statistics. This function therefore returns verbatim copies -- it documents
    where RC sits in the workflow but does not synthesize the twirl. (A functional
    twirl only earns its keep once coherent errors are added to the noise model.)
    The twirling CB actually relies on -- random single-qubit layers around each
    cycle -- IS implemented, inside ``cycle_benchmark``.
    """
    from .circuit import Circuit

    out = []
    for _ in range(n_compilations):
        nc = Circuit(circuit.n_qubits, name=circuit.name + "_rc")
        for g in circuit.gates:
            nc.gates.append(g)
        out.append(nc)
    return out
