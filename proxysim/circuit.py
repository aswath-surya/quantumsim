"""Backend-agnostic circuit intermediate representation (IR) and builders.

A :class:`Circuit` is just an ordered list of :class:`Gate` operations on
``n_qubits`` qubits.  Each backend (quimb / qiskit / stim) knows how to
translate this IR into its own native circuit object, so the *same* logical
circuit can be run on a tensor-network, statevector, or stabilizer simulator
and the results compared directly.

Bitstring convention used everywhere downstream:
    string index ``i`` corresponds to qubit ``i``; qubit 0 is the LEFTMOST
    character.  (quimb and stim already do this; the qiskit backend reverses.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Gate metadata
# ---------------------------------------------------------------------------
# Single-qubit Clifford gates (fixed, non-parametrised).
_CLIFFORD_1Q = {"i", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg"}
# Two-qubit Clifford gates.
_CLIFFORD_2Q = {"cx", "cnot", "cz", "cy", "swap"}
# Parametrised rotations: Clifford only when the angle is a multiple of pi/2.
_PARAM_1Q = {"rx", "ry", "rz", "p"}
# Explicitly non-Clifford fixed gates.
_NON_CLIFFORD = {"t", "tdg"}

_HALF_PI = math.pi / 2.0


def _is_multiple_of_half_pi(theta: float, tol: float = 1e-9) -> bool:
    r = theta / _HALF_PI
    return abs(r - round(r)) < tol


@dataclass(frozen=True)
class Gate:
    """A single gate application: name + target qubits (+ optional params)."""

    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()

    @property
    def is_clifford(self) -> bool:
        n = self.name
        if n in _CLIFFORD_1Q or n in _CLIFFORD_2Q:
            return True
        if n in _NON_CLIFFORD:
            return False
        if n in _PARAM_1Q:
            return all(_is_multiple_of_half_pi(p) for p in self.params)
        # Unknown gate: be conservative.
        return False

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        p = f"({', '.join(f'{x:.4g}' for x in self.params)})" if self.params else ""
        return f"{self.name}{p} {list(self.qubits)}"


@dataclass
class Circuit:
    """An ordered list of gates on ``n_qubits`` qubits."""

    n_qubits: int
    gates: List[Gate] = field(default_factory=list)
    name: str = "circuit"

    # -- generic add -------------------------------------------------------
    def add(self, name: str, *qubits: int, params: Tuple[float, ...] = ()) -> "Circuit":
        self.gates.append(Gate(name.lower(), tuple(qubits), tuple(params)))
        return self

    # -- convenience one-qubit gates --------------------------------------
    def h(self, q):
        return self.add("h", q)

    def x(self, q):
        return self.add("x", q)

    def y(self, q):
        return self.add("y", q)

    def z(self, q):
        return self.add("z", q)

    def s(self, q):
        return self.add("s", q)

    def sdg(self, q):
        return self.add("sdg", q)

    def sx(self, q):
        return self.add("sx", q)

    def sxdg(self, q):
        return self.add("sxdg", q)

    def t(self, q):
        return self.add("t", q)

    def rz(self, theta, q):
        return self.add("rz", q, params=(theta,))

    def rx(self, theta, q):
        return self.add("rx", q, params=(theta,))

    def ry(self, theta, q):
        return self.add("ry", q, params=(theta,))

    # -- convenience two-qubit gates --------------------------------------
    def cz(self, a, b):
        return self.add("cz", a, b)

    def cx(self, a, b):
        return self.add("cx", a, b)

    def cnot(self, a, b):
        return self.add("cx", a, b)

    def swap(self, a, b):
        return self.add("swap", a, b)

    # -- introspection -----------------------------------------------------
    @property
    def is_clifford(self) -> bool:
        return all(g.is_clifford for g in self.gates)

    @property
    def n_two_qubit(self) -> int:
        return sum(1 for g in self.gates if len(g.qubits) == 2)

    @property
    def depth(self) -> int:
        """Greedy ASAP depth (number of parallel layers)."""
        frontier = [0] * self.n_qubits
        for g in self.gates:
            t = max(frontier[q] for q in g.qubits) + 1
            for q in g.qubits:
                frontier[q] = t
        return max(frontier) if frontier else 0

    def inverse(self) -> "Circuit":
        """Return the inverse circuit (gates reversed and individually inverted).

        Appending a circuit's inverse gives a 'mirror' circuit whose ideal output
        is |0...0>, which makes the error resolvable (1 - P(0...0)) at any width.
        """
        inv_name = {
            "h": "h", "x": "x", "y": "y", "z": "z", "i": "i",
            "s": "sdg", "sdg": "s", "sx": "sxdg", "sxdg": "sx", "t": "tdg", "tdg": "t",
            "cz": "cz", "cx": "cx", "cnot": "cnot", "cy": "cy", "swap": "swap",
        }
        out = Circuit(self.n_qubits, name=self.name + "_inv")
        for g in reversed(self.gates):
            if g.name in ("rx", "ry", "rz", "p"):
                out.gates.append(Gate(g.name, g.qubits, tuple(-p for p in g.params)))
            else:
                out.gates.append(Gate(inv_name[g.name], g.qubits, g.params))
        return out

    def mirror(self) -> "Circuit":
        """This circuit followed by its inverse (U.U^-1). Ideal output is |0...0>,
        so the error is 1 - P(0...0) and stays resolvable at any width."""
        out = Circuit(self.n_qubits, name=self.name + "_mirror")
        out.gates = list(self.gates) + self.inverse().gates
        return out

    def layers(self):
        """Greedy ASAP partition into parallel layers (each a list of gates).

        Used e.g. to find which qubits are *idle* in a given layer (for idling
        noise): a qubit is idle in layer L if no gate in that layer touches it.
        """
        frontier = [0] * self.n_qubits
        out = []
        for g in self.gates:
            t = max(frontier[q] for q in g.qubits)
            for q in g.qubits:
                frontier[q] = t + 1
            while len(out) <= t:
                out.append([])
            out[t].append(g)
        return out

    def __len__(self) -> int:
        return len(self.gates)

    def summary(self) -> str:
        return (
            f"{self.name}: n={self.n_qubits}, gates={len(self)}, "
            f"2q-gates={self.n_two_qubit}, depth={self.depth}, "
            f"clifford={self.is_clifford}"
        )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
# Single-qubit Clifford gates used to "scramble" between entangling layers.
_RANDOM_CLIFFORDS = ["h", "s", "sdg", "x", "y", "z", "sx", "sxdg"]


def _even_pairs(n: int):
    return [(i, i + 1) for i in range(0, n - 1, 2)]


def _odd_pairs(n: int):
    return [(i, i + 1) for i in range(1, n - 1, 2)]


def lnn_brickwork(
    n: int,
    n_cycles: int,
    twoq: str = "cz",
    mode: str = "clifford",
    seed: int = 0,
    initial_h: bool = True,
) -> Circuit:
    """Build a linear-nearest-neighbour (LNN) fully-entangling brickwork circuit.

    Structure (one *cycle* = two 2-qubit layers, à la Merkel et al. Fig. 3):

        [optional initial Hadamard layer]
        repeat ``n_cycles`` times:
            even-odd 2q layer :  (0,1) (2,3) ...
            single-qubit layer
            odd-even 2q layer :  (1,2) (3,4) ...
            single-qubit layer

    Cycling ``n_cycles = n - 1`` times fully entangles the whole chain.

    Parameters
    ----------
    n          : number of qubits.
    n_cycles   : how many (even-odd, odd-even) cycles to apply.
    twoq       : 'cz' or 'cx' -- the entangling gate.
    mode       : 'clifford'  -> random single-qubit Clifford gates (stim-friendly);
                 'haar'      -> random Z-SX-Z-SX-Z Euler decomposition with Haar
                               angles (non-Clifford, mirrors the paper's
                               Z(phi1) X_{pi/2} Z(phi2) X_{pi/2} Z(phi3) target).
    seed       : RNG seed for the single-qubit gate choices/angles.
    initial_h  : prepend a layer of Hadamards (creates the initial superposition).
    """
    import random

    rng = random.Random(seed)
    twoq = twoq.lower()
    if twoq not in ("cz", "cx"):
        raise ValueError("twoq must be 'cz' or 'cx'")
    if mode not in ("clifford", "haar"):
        raise ValueError("mode must be 'clifford' or 'haar'")

    circ = Circuit(n, name=f"lnn_brickwork_{mode}_n{n}_c{n_cycles}_{twoq}")

    def one_qubit_layer():
        for q in range(n):
            if mode == "clifford":
                circ.add(rng.choice(_RANDOM_CLIFFORDS), q)
            else:  # haar single-qubit gate as Z-SX-Z-SX-Z Euler sequence
                phi1 = rng.uniform(0, 2 * math.pi)
                phi2 = rng.uniform(0, 2 * math.pi)
                phi3 = rng.uniform(0, 2 * math.pi)
                circ.rz(phi1, q)
                circ.sx(q)
                circ.rz(phi2, q)
                circ.sx(q)
                circ.rz(phi3, q)

    if initial_h:
        for q in range(n):
            circ.h(q)

    for c in range(n_cycles):
        for a, b in _even_pairs(n):
            circ.add(twoq, a, b)
        one_qubit_layer()
        for a, b in _odd_pairs(n):
            circ.add(twoq, a, b)
        one_qubit_layer()

    return circ


def clifford_entropy_circuit(n: int, entropy_bits: int, depth: int = 6,
                             seed: int = 0) -> Circuit:
    """A Clifford circuit whose ideal Z-basis distribution has EXACTLY
    ``entropy_bits`` bits of Shannon entropy (uniform over 2^entropy_bits
    correlated bitstrings).  Not a mirror/echo -- the output is genuinely spread.

    How the entropy is fixed: ``entropy_bits`` Hadamards on |0> qubits inject that
    many free bits.  Everything after that preserves the Z-basis support
    dimension, so the entropy can't drift:
      * CNOT -- a linear bijection on bitstrings; spreads/correlates the free bits
        across qubits without changing the coset dimension.
      * CZ, S, S-dagger, Z -- diagonal, so the Z-basis probabilities are untouched.
      * X -- just shifts the coset (an affine offset), dimension unchanged.
    """
    import random

    rng = random.Random(seed)
    k = max(0, min(entropy_bits, n))
    circ = Circuit(n, name=f"clifford_entropy_n{n}_H{k}_d{depth}")

    for q in rng.sample(range(n), k):          # inject exactly k free bits
        circ.h(q)

    dim_preserving = ["i", "s", "sdg", "z", "x"]
    for d in range(depth):
        pairs = _even_pairs(n) if d % 2 == 0 else _odd_pairs(n)
        for a, b in pairs:
            circ.cx(a, b)                      # spread the free bits (bijective)
            circ.cz(a, b)                      # phases only (no effect on Z dist)
        for q in range(n):
            circ.add(rng.choice(dim_preserving), q)
    return circ


def brickwork_magic(n: int, n_cycles: int, n_t: int = 0, seed: int = 0,
                    twoq: str = "cz", initial_h: bool = True) -> Circuit:
    """A Clifford brickwork with ``n_t`` injected "magic" gates.

    The magic gates are pi/4 rotations about a random axis (T-like: T = RZ(pi/4)).
    ``n_t = 0`` is a pure Clifford circuit -- Pauli propagation keeps the
    observable a single Pauli string.  Each magic gate that anticommutes with the
    propagated observable *branches* it into two terms scaled by cos/sin(pi/4)=
    1/sqrt(2), so the number of Pauli strings grows and their coefficients spread
    over the values 2^(-k/2). More magic => more non-stabilizerness => a broader
    coefficient distribution.
    """
    import random

    rng = random.Random(seed)
    circ = Circuit(n, name=f"brickwork_magic_n{n}_c{n_cycles}_t{n_t}")
    oneq_slots = []

    def one_qubit_layer():
        for q in range(n):
            circ.add(rng.choice(_RANDOM_CLIFFORDS), q)
            oneq_slots.append(len(circ.gates) - 1)

    if initial_h:
        for q in range(n):
            circ.h(q)
    for _ in range(n_cycles):
        for a, b in _even_pairs(n):
            circ.add(twoq, a, b)
        one_qubit_layer()
        for a, b in _odd_pairs(n):
            circ.add(twoq, a, b)
        one_qubit_layer()

    n_t = min(n_t, len(oneq_slots))
    for idx in rng.sample(oneq_slots, n_t):
        q = circ.gates[idx].qubits[0]
        axis = rng.choice(["rx", "ry", "rz"])
        circ.gates[idx] = Gate(axis, (q,), (math.pi / 4,))
    return circ


# ---------------------------------------------------------------------------
# Canonical benchmarking / bounding brickwork (shared by the example scripts)
# ---------------------------------------------------------------------------
# Named single-qubit gate sets.
GATE_SETS = {
    "clifford": ["h", "s", "sdg", "x", "y", "z", "sx", "sxdg"],  # random Clifford
    "structured": ["i", "h", "x"],                                # restricted set
    "dim_preserving": ["i", "s", "sdg", "z", "x"],                # keep Z-basis support dim
}


def even_pairs(n):
    """Even nearest-neighbour pairs: (0,1) (2,3) ..."""
    return _even_pairs(n)


def odd_pairs(n):
    """Odd nearest-neighbour pairs: (1,2) (3,4) ..."""
    return _odd_pairs(n)


def bench_brickwork(n, n_cycles, cycles, oneq="clifford", twoq="cz", seed=0,
                    final_oneq=False, n_hadamards=0):
    """Canonical benchmarking / bounding brickwork.

    Repeats ``n_cycles`` times: for each entangling layer ``pairs`` in ``cycles``,
    a random single-qubit layer (drawn from the ``oneq`` gate set) then that layer's
    two-qubit gates (``twoq``). Every entangling layer in ``cycles`` is a distinct
    'cycle' that recurs ``n_cycles`` times -- matching what cycle benchmarking sees.

    ``oneq``          : name in GATE_SETS ('clifford'/'structured'/'dim_preserving')
                        or an explicit list of gate names.
    ``n_hadamards``   : prepend H on this many random qubits (inject exactly this many
                        bits of output entropy when oneq='dim_preserving', twoq='cx').
    ``final_oneq``    : add a trailing single-qubit layer (non-mirror test circuits).
    """
    import random

    rng = random.Random(seed)
    gates = GATE_SETS[oneq] if isinstance(oneq, str) else list(oneq)
    tag = oneq if isinstance(oneq, str) else "custom"
    c = Circuit(n, name=f"brickwork_{tag}_n{n}_c{n_cycles}_{twoq}")

    for q in rng.sample(range(n), min(n_hadamards, n)):
        c.h(q)

    def layer():
        for q in range(n):
            c.add(rng.choice(gates), q)

    for _ in range(n_cycles):
        for pairs in cycles:
            layer()
            for a, b in pairs:
                c.add(twoq, a, b)
    if final_oneq:
        layer()
    return c
