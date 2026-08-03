"""Cycle-benchmarking circuits emitted as standalone circuits (rather than run in-process).

:func:`proxysim.benchmarking.cycle_benchmark` runs the whole CB protocol against a
:class:`~proxysim.noise.NoiseModel` and hands back a fitted ``e_F``. This module splits
that in half: it *builds* the same circuits in the backend-agnostic IR so they can be
written to QASM and executed elsewhere, and provides :func:`analyze_cb` to turn the
resulting shot data back into ``e_F`` using the identical fit.

The circuit for one (cycle, depth m, randomization r):

  1. prepare the +1 eigenstate of a random n-qubit Pauli ``P``
  2. ``m`` dressed rounds of ``[random 1q Clifford layer on all n qubits] + [entangler]``
  3. rotate the propagated Pauli ``P_m = C^m P C^-m`` into the Z basis
  4. measure every qubit

Steps 1-3 mirror ``_prep_eigenstate`` / ``_apply_round`` / ``_measure_rotation`` in
:mod:`proxysim.benchmarking`; those emit into a ``stim.Circuit``, so the IR versions
here are separate builders rather than shared code, but the random-Pauli draw, the 1q
gate set, and the decay fit are all imported from there so the two paths cannot drift
on the parts that matter.

**The metadata is not optional.** A CB QASM file is not analyzable on its own: recovering
the decay point needs the propagated Pauli's support and sign, since the estimator is

    survival = sign * mean(1 - 2 * parity(shot bits restricted to support))

which is +1 in the noiseless limit by construction. :func:`cb_circuit` returns that
alongside the circuit, and the bank writes it to a sidecar ``meta.json``.

Non-Clifford targets and the proxy
----------------------------------
CB is a Clifford protocol -- a Pauli conjugated through ``cp(theta)`` is a *sum* of
Paulis, so CB cannot run on the algorithmic gate directly. Following Merkel et al.
(arXiv:2503.05943), as :mod:`examples.run_qft_bounding` already does, the noise a gate
carries is benchmarked with a Clifford *proxy* entangler: under gate-independent Pauli
noise ``e_F`` is a property of the error channel, not the gate angle, so the proxy's
``e_F`` is the target's. Hence ``cp(theta) -> CZ``; ``cx``/``cz``/``swap`` are already
Clifford and are their own proxies.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .benchmarking import (_ONEQ_RANDOM, _ONEQ_STRUCTURED, _fit_decay, _letters,
                           _rand_pauli)
from .circuit import Circuit, Gate

# Algorithmic 2q gate -> the Clifford entangler used to benchmark the noise it carries.
PROXY_OF = {"cp": "cz", "cz": "cz", "cx": "cx", "cy": "cy", "swap": "swap"}


def proxy_name(twoq: str, pair: Tuple[int, int]) -> str:
    """Directory-safe label for a proxy cycle, e.g. ``cz_q0q1``."""
    return f"{twoq}_q{pair[0]}q{pair[1]}"


# ---------------------------------------------------------------------------
# IR builders for Pauli eigenstate prep / measurement rotation
# ---------------------------------------------------------------------------
def _prep_eigenstate(circ: Circuit, letters: str) -> None:
    """Prepare the +1 eigenstate of Pauli ``letters`` from |0...0>."""
    for i, ch in enumerate(letters):
        if ch == "X":
            circ.h(i)
        elif ch == "Y":
            circ.h(i)
            circ.s(i)               # S H |0> = |+i>, the +1 eigenstate of Y


def _measure_rotation(circ: Circuit, letters: str) -> None:
    """Rotate each Pauli factor to Z so a Z-basis readout measures the Pauli."""
    for i, ch in enumerate(letters):
        if ch == "X":
            circ.h(i)
        elif ch == "Y":
            circ.sdg(i)
            circ.h(i)


def _to_stim(circ: Circuit):
    """The IR circuit as a ``stim.Circuit``. CB circuits are Clifford by construction,
    so the simple name map covers every gate they contain."""
    import stim

    from .backends.stabilizer import _SIMPLE

    out = stim.Circuit()
    for g in circ.gates:
        out.append(_SIMPLE[g.name], list(g.qubits))
    return out


# ---------------------------------------------------------------------------
# One CB circuit
# ---------------------------------------------------------------------------
def cb_circuit(n: int, pairs: Sequence[Tuple[int, int]], depth: int,
               rng: random.Random, twoq: str = "cz",
               oneq: Sequence[str] = _ONEQ_RANDOM) -> Tuple[Circuit, dict]:
    """Build one cycle-benchmarking circuit and the metadata needed to analyze it.

    Returns ``(circuit, meta)`` with ``meta`` carrying ``prep_pauli``, ``meas_pauli``,
    ``sign`` and ``support`` -- see the module docstring for why the QASM alone is not
    enough.

    ``oneq`` is the single-qubit dressing set, matching ``cycle_benchmark``'s ``mode``:
    ``_ONEQ_RANDOM`` (the default) is ``mode="random"``, ``_ONEQ_STRUCTURED`` is
    ``mode="structured"``. It must stay Clifford -- the twirl the CB protocol relies on
    is a twirl over the Clifford group, and the propagated Pauli below is computed with
    a stim tableau.
    """
    P = _rand_pauli(n, rng)
    rounds = [[rng.choice(oneq) for _ in range(n)] for _ in range(depth)]

    # The ideal (noiseless) Clifford C^m, used only to propagate P -> P_m.
    ideal = Circuit(n, name="cb_ideal")
    for oneq in rounds:
        for q, g in enumerate(oneq):
            ideal.add(g, q)
        for a, b in pairs:
            ideal.add(twoq, a, b)

    import stim
    P_m = stim.PauliString(P).after(_to_stim(ideal).to_tableau(), targets=range(n))
    letters_m = _letters(P_m)

    circ = Circuit(n, name=f"cb_{twoq}_n{n}_d{depth}")
    _prep_eigenstate(circ, P)
    circ.gates.extend(ideal.gates)
    _measure_rotation(circ, letters_m)

    meta = {
        "n_qubits": n,
        "depth": depth,
        "twoq": twoq,
        "oneq": list(oneq),
        "pairs": [list(p) for p in pairs],
        "prep_pauli": P,
        "meas_pauli": letters_m,
        "sign": 1 if P_m.sign.real > 0 else -1,
        "support": [i for i, ch in enumerate(letters_m) if ch != "_"],
    }
    return circ, meta


# ---------------------------------------------------------------------------
# Turning shot data back into e_F
# ---------------------------------------------------------------------------
def survival(bits, meta: dict) -> float:
    """Signed survival of one CB circuit from its shots.

    ``bits`` is a ``(shots, n_qubits)`` array of 0/1 measurement outcomes ordered so
    that column ``i`` is qubit ``i`` (the convention used throughout this package).
    Noiseless execution gives exactly ``1.0``.
    """
    arr = np.asarray(bits)
    parity = arr[:, meta["support"]].sum(axis=1) % 2
    return float(np.mean(1 - 2 * parity)) * meta["sign"]


def analyze_cb(survivals_by_depth: Dict[int, Sequence[float]], n: int) -> dict:
    """Fit ``A * f^m`` to the per-depth mean survival and convert to a process infidelity.

    Uses the same fit and the same conversion as
    :func:`proxysim.benchmarking.cycle_benchmark`:
    ``e_F = (1 - 4^-n) * (1 - f)`` and ``r = d/(d+1) * e_F`` with ``d = 2^n``.
    """
    depths = sorted(survivals_by_depth)
    decay = [float(np.mean(survivals_by_depth[m])) for m in depths]
    f, f_std = _fit_decay(np.array(depths, float), np.array(decay))
    factor = 1.0 - 4.0 ** (-n)
    e_F = factor * (1.0 - f)
    d = 2.0 ** n
    return {"depths": depths, "decay": decay, "f": f, "e_F": e_F,
            "e_F_std": factor * f_std, "F_e": 1.0 - e_F,
            "r": d / (d + 1.0) * e_F, "F_avg": 1.0 - d / (d + 1.0) * e_F}
