"""Circuit-level noise model with a single on/off toggle.

Channels (all Pauli-stochastic, matching stim's conventions so the stim-native
and trajectory paths agree exactly):

  * 1-qubit depolarizing after every single-qubit gate      (p1)   -> DEPOLARIZE1
  * 2-qubit depolarizing after every two-qubit gate         (p2)   -> DEPOLARIZE2
  * idling dephasing (Z) on qubits idle during a layer      (p_idle) -> Z_ERROR
  * measurement bit-flip (readout error)                    (p_readout) -> X_ERROR before M

`NoiseModel.enabled` is the toggle: when False, every channel is skipped and the
simulation is ideal.

Two application paths:
  * :func:`to_stim_noisy` -- inserts native stim noise ops (exact, scalable) for
    the stabilizer backend. Clifford circuits only.
  * :func:`sample_trajectory` + :func:`apply_readout` -- Monte-Carlo trajectory:
    sample which Pauli errors fire and splice them into the circuit as gates, for
    pure-state backends (statevector / MPS). Averaging trajectories reproduces the
    mixed-state distribution.

This is the setting where finite-shot sampling matters (a mixed state can't be
read off as a single |<x|psi>|^2), i.e. the PTA regime of Merkel et al.

Fidelity of a noisy vs ideal distribution: classical (Hellinger) fidelity
F = (sum_x sqrt(p q))^2 in [0, 1].
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict

import stim

from .backends.stabilizer import _PARAM, _SIMPLE

# 15 non-identity two-qubit Pauli labels, e.g. "IX", "XZ", ...
_TWOQ_PAULIS = ["".join(p) for p in itertools.product("IXYZ", repeat=2) if "".join(p) != "II"]
_ONEQ_PAULIS = ["x", "y", "z"]


@dataclass
class NoiseModel:
    enabled: bool = False
    p1: float = 1e-3          # 1-qubit depolarizing after each single-qubit gate
    p2: float = 1e-2          # 2-qubit depolarizing after each two-qubit gate
    p_readout: float = 1e-2   # measurement bit-flip probability
    p_idle: float = 1e-3      # idling dephasing (Z) per idle qubit per layer

    def scaled(self, s: float) -> "NoiseModel":
        """Same model with every rate multiplied by ``s`` (and enabled)."""
        return NoiseModel(True, self.p1 * s, self.p2 * s, self.p_readout * s, self.p_idle * s)


# ---------------------------------------------------------------------------
# stim-native application (exact, scalable; Clifford circuits only)
# ---------------------------------------------------------------------------
def _emit_stim(c: stim.Circuit, g):
    if g.name in _SIMPLE:
        c.append(_SIMPLE[g.name], list(g.qubits))
        return
    if g.name in _PARAM:
        k = round(g.params[0] / (math.pi / 2)) % 4
        gate = _PARAM[g.name][k]
        if gate is not None:
            c.append(gate, list(g.qubits))
        return
    raise ValueError(f"noise/stim: non-Clifford gate '{g.name}'")


def to_stim_noisy(circuit, noise: NoiseModel) -> stim.Circuit:
    """Build a measured stim circuit with native noise ops inserted."""
    c = stim.Circuit()
    n = circuit.n_qubits
    for layer in circuit.layers():
        touched = set()
        for g in layer:
            _emit_stim(c, g)
            touched.update(g.qubits)
            if noise.enabled:
                if len(g.qubits) == 2 and noise.p2 > 0:
                    c.append("DEPOLARIZE2", list(g.qubits), noise.p2)
                elif len(g.qubits) == 1 and noise.p1 > 0:
                    c.append("DEPOLARIZE1", list(g.qubits), noise.p1)
        if noise.enabled and noise.p_idle > 0:
            idle = [q for q in range(n) if q not in touched]
            if idle:
                c.append("Z_ERROR", idle, noise.p_idle)
    if noise.enabled and noise.p_readout > 0:
        c.append("X_ERROR", list(range(n)), noise.p_readout)
    c.append("M", list(range(n)))
    return c


# ---------------------------------------------------------------------------
# Trajectory application (for pure-state backends: statevector / MPS)
# ---------------------------------------------------------------------------
def sample_trajectory(circuit, noise: NoiseModel, rng):
    """Return a NEW Circuit with one sampled Pauli-error realisation spliced in.

    Readout error is NOT applied here -- apply :func:`apply_readout` to the
    measured bitstring afterwards.
    """
    from .circuit import Circuit

    nc = Circuit(circuit.n_qubits, name=circuit.name + "_traj")
    n = circuit.n_qubits
    for layer in circuit.layers():
        touched = set()
        for g in layer:
            nc.gates.append(g)
            touched.update(g.qubits)
            if not noise.enabled:
                continue
            if len(g.qubits) == 2 and noise.p2 > 0 and rng.random() < noise.p2:
                pa, pb = rng.choice(_TWOQ_PAULIS)
                if pa != "I":
                    nc.add(pa.lower(), g.qubits[0])
                if pb != "I":
                    nc.add(pb.lower(), g.qubits[1])
            elif len(g.qubits) == 1 and noise.p1 > 0 and rng.random() < noise.p1:
                nc.add(rng.choice(_ONEQ_PAULIS), g.qubits[0])
        if noise.enabled and noise.p_idle > 0:
            for q in range(n):
                if q not in touched and rng.random() < noise.p_idle:
                    nc.add("z", q)
    return nc


def apply_readout(bitstring: str, noise: NoiseModel, rng) -> str:
    if not noise.enabled or noise.p_readout <= 0:
        return bitstring
    return "".join(
        (("1" if b == "0" else "0") if rng.random() < noise.p_readout else b)
        for b in bitstring
    )


# ---------------------------------------------------------------------------
def classical_fidelity(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Hellinger / classical fidelity  ( sum_x sqrt(p(x) q(x)) )^2  in [0, 1]."""
    keys = set(p) | set(q)
    return sum(math.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in keys) ** 2
