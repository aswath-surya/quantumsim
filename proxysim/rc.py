"""Randomized compiling (Pauli twirling) as a circuit-to-circuit transform.

Randomized compiling tailors coherent error into Pauli-stochastic error by wrapping
each "hard" (entangling) gate in random Paulis and undoing them on the far side:

    P' G P   with   P' = G P G^-1     =>     P' G P = G  (up to a global sign)

so the *logical* action is untouched while the *physical* error the gate carries gets
conjugated by a fresh random Pauli on every randomization. Averaging over many
randomizations projects the error channel onto its Pauli-diagonal part, which is the
assumption cycle benchmarking and the QCAP bound both rest on.

Relation to :func:`proxysim.benchmarking.randomly_compile`: that function was a
documented no-op and now delegates here. The distinction it used to rest on still
matters as a test. Against a purely stochastic :class:`~proxysim.noise.NoiseModel` the
channel is already twirl-averaged, so twirling must leave the statistics unchanged.
Against a model with a coherent term (``theta_1q``/``theta_zz``) it must instead move
them onto :meth:`~proxysim.noise.NoiseModel.twirled`, the exact Pauli channel that
model becomes under RC -- and that is the channel ``cycle_benchmark`` reports an
``e_F`` for, so a QCAP bound is only valid for a circuit that was actually twirled.

Twirling a non-Clifford gate
----------------------------
The construction above needs ``G P G^-1`` to be a Pauli, i.e. ``G`` Clifford. MQT Bench
circuits at the algorithmic level contain ``cp(theta)``, which is not. But ``cp(theta)``
is *diagonal*, so the Z-type Paulis commute with it exactly, and twirling over the
subgroup ``{I,Z} x {I,Z}`` is valid with no modification to the gate. That is what this
module does: the full 16-element Pauli group on Clifford entanglers, the 4-element
commuting subgroup on ``cp(theta)``.

The restriction is real and worth stating: a Z-only twirl tailors the error toward a
Z-diagonal (dephasing) channel rather than a full Pauli channel. A complete twirl over
``cp(theta)`` *is* possible -- pushing an X through flips the angle,
``X_a cp(theta) X_a = cp(-theta) p_b(theta)`` -- but it rewrites the gate itself rather
than just wrapping it, so it is deliberately out of scope here.

The emitted twirl is *uncompiled*: the Paulis appear as explicit ``x``/``y``/``z`` gates
rather than being absorbed into neighbouring single-qubit gates (True-Q compiles them
into the surrounding 1q layer). Adjacent Paulis on the same qubit are merged, which
removes most of the overhead; a consumer that cares about the rest can run its own
single-qubit optimization pass.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from .circuit import Circuit, Gate, _is_multiple_of_pi

# Pauli letters, and their product ignoring phase (phases are global, hence irrelevant).
_PAULIS = ("I", "X", "Y", "Z")
_MUL = {
    ("I", "I"): "I", ("I", "X"): "X", ("I", "Y"): "Y", ("I", "Z"): "Z",
    ("X", "I"): "X", ("X", "X"): "I", ("X", "Y"): "Z", ("X", "Z"): "Y",
    ("Y", "I"): "Y", ("Y", "X"): "Z", ("Y", "Y"): "I", ("Y", "Z"): "X",
    ("Z", "I"): "Z", ("Z", "X"): "Y", ("Z", "Y"): "X", ("Z", "Z"): "I",
}
_PAULI_GATES = {"x": "X", "y": "Y", "z": "Z", "i": "I"}
_GATE_OF = {"X": "x", "Y": "y", "Z": "z"}

# The full Pauli group on two qubits, and the diagonal-commuting subgroup.
_FULL_2Q = tuple(a + b for a in _PAULIS for b in _PAULIS)
_DIAG_2Q = ("II", "IZ", "ZI", "ZZ")


# ---------------------------------------------------------------------------
# Pauli conjugation through a two-qubit Clifford, via stim
# ---------------------------------------------------------------------------
def _stim_name(gate: Gate) -> Optional[str]:
    """The stim gate whose Clifford action equals ``gate``, or None if not Clifford.

    ``cp`` is Clifford only at multiples of pi: an odd multiple is CZ, an even multiple
    is the identity (every Pauli passes through untouched)."""
    n = gate.name
    if n in ("cx", "cz", "cy", "swap"):
        return {"cx": "CX", "cz": "CZ", "cy": "CY", "swap": "SWAP"}[n]
    if n == "cp" and _is_multiple_of_pi(gate.params[0]):
        return "CZ" if round(gate.params[0] / math.pi) % 2 else "I"
    return None


_CONJ_CACHE: Dict[str, Dict[str, str]] = {}


def _conjugation_table(stim_gate: str) -> Dict[str, str]:
    """Map each 2-qubit Pauli string P -> the letters of ``G P G^-1``, for Clifford G.

    Signs are dropped: they contribute only a global phase to ``P' G P = +-G``."""
    if stim_gate in _CONJ_CACHE:
        return _CONJ_CACHE[stim_gate]
    if stim_gate == "I":                       # cp(2k*pi): everything commutes
        _CONJ_CACHE[stim_gate] = {p: p for p in _FULL_2Q}
        return _CONJ_CACHE[stim_gate]
    import stim

    circ = stim.Circuit()
    circ.append(stim_gate, [0, 1])
    tab = circ.to_tableau()
    table = {}
    for p in _FULL_2Q:
        out = stim.PauliString(p).after(tab, targets=[0, 1])
        table[p] = "".join("IXYZ"[out[i]] for i in range(2))
    _CONJ_CACHE[stim_gate] = table
    return table


def twirl_options(gate: Gate) -> Tuple[Tuple[str, ...], Optional[Dict[str, str]]]:
    """The Paulis this two-qubit gate may be twirled with, and its conjugation table.

    Returns ``(paulis, table)``; ``table is None`` means the Paulis commute through the
    gate unchanged (the diagonal ``cp(theta)`` case)."""
    stim_gate = _stim_name(gate)
    if stim_gate is not None:
        return _FULL_2Q, _conjugation_table(stim_gate)
    if gate.name == "cp":            # non-Clifford but diagonal: Z-subgroup commutes
        return _DIAG_2Q, None
    raise ValueError(f"pauli_twirl: don't know how to twirl 2-qubit gate '{gate.name}'")


# ---------------------------------------------------------------------------
# The twirl
# ---------------------------------------------------------------------------
def pauli_twirl(circuit: Circuit, rng: random.Random,
                name_suffix: str = "_rc") -> Circuit:
    """Return one Pauli-twirled randomization of ``circuit``.

    Every two-qubit gate is independently wrapped in a random Pauli and its conjugate.
    The result implements the same unitary as ``circuit`` up to a global phase.
    Single-qubit gates are left alone; adjacent Paulis (including ones already present
    in the source circuit) are merged by :func:`_merge_paulis`.
    """
    out: List[Gate] = []
    for g in circuit.gates:
        if len(g.qubits) != 2:
            out.append(g)
            continue
        paulis, table = twirl_options(g)
        p = rng.choice(paulis)
        p_after = p if table is None else table[p]
        a, b = g.qubits
        for q, letter in ((a, p[0]), (b, p[1])):
            if letter != "I":
                out.append(Gate(_GATE_OF[letter], (q,)))
        out.append(g)
        for q, letter in ((a, p_after[0]), (b, p_after[1])):
            if letter != "I":
                out.append(Gate(_GATE_OF[letter], (q,)))

    twirled = Circuit(circuit.n_qubits, name=circuit.name + name_suffix)
    twirled.gates = _merge_paulis(out, circuit.n_qubits)
    return twirled


def _merge_paulis(gates: List[Gate], n_qubits: int) -> List[Gate]:
    """Collapse runs of single-qubit Paulis on the same qubit into one gate.

    A Pauli is deferred on its qubit and flushed just before the next gate touching
    that qubit, so per-qubit ordering is preserved exactly; gates on disjoint qubits
    commute with it. This is what removes most of the twirl's gate-count overhead,
    since the correction ``P'`` of one entangler and the fresh ``P`` of the next
    entangler on the same qubit land side by side.
    """
    pending = ["I"] * n_qubits
    out: List[Gate] = []

    def flush(qubits):
        for q in qubits:
            if pending[q] != "I":
                out.append(Gate(_GATE_OF[pending[q]], (q,)))
                pending[q] = "I"

    for g in gates:
        if len(g.qubits) == 1 and g.name in _PAULI_GATES:
            q = g.qubits[0]
            pending[q] = _MUL[(pending[q], _PAULI_GATES[g.name])]
            continue
        flush(g.qubits)
        out.append(g)
    flush(range(n_qubits))
    return out


def randomly_compile(circuit: Circuit, n_compilations: int, seed: int = 0,
                     twirl: bool = True) -> List[Circuit]:
    """``n_compilations`` independent Pauli-twirled randomizations of ``circuit``.

    With ``twirl=False`` this returns verbatim copies -- the old behaviour, which is
    statistically equivalent whenever the noise is already Pauli-stochastic (as
    :class:`~proxysim.noise.NoiseModel` is), and is kept for that comparison.
    """
    rng = random.Random(seed)
    if not twirl:
        out = []
        for _ in range(n_compilations):
            c = Circuit(circuit.n_qubits, name=circuit.name + "_rc")
            c.gates = list(circuit.gates)
            out.append(c)
        return out
    return [pauli_twirl(circuit, rng) for _ in range(n_compilations)]
