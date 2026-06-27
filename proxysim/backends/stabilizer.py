"""Stabilizer backend (stim).  Clifford circuits only.

Exact distribution: the noiseless Clifford output is a stabilizer state, computed
with ``stim.TableauSimulator`` (polynomial-size tableau) and read out densely via
``state_vector`` for small N.  stim is little-endian; we map amplitude index i to
the canonical q0-leftmost bitstring directly.

Finite-shot sampling overrides the base multinomial with stim's NATIVE
``compile_sampler`` -- a genuinely scalable CHP stabilizer sampler that works for
thousands of qubits, i.e. even when the full 2^N distribution cannot be stored.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from .base import Backend

try:
    import stim

    _AVAILABLE = True
    _IMPORT_ERR = None
except Exception as e:  # pragma: no cover
    _AVAILABLE = False
    _IMPORT_ERR = e

_HALF_PI = math.pi / 2.0

_SIMPLE = {
    "i": "I", "x": "X", "y": "Y", "z": "Z", "h": "H",
    "s": "S", "sdg": "S_DAG", "sx": "SQRT_X", "sxdg": "SQRT_X_DAG",
    "cz": "CZ", "cx": "CX", "cnot": "CX", "cy": "CY", "swap": "SWAP",
}
_PARAM = {
    "rz": [None, "S", "Z", "S_DAG"],
    "rx": [None, "SQRT_X", "X", "SQRT_X_DAG"],
    "ry": [None, "SQRT_Y", "Y", "SQRT_Y_DAG"],
}


class StabilizerBackend(Backend):
    name = "stabilizer(stim)"
    available = _AVAILABLE
    import_error = _IMPORT_ERR
    _method = "stabilizer tableau (exact) / native CHP sampler"

    def supports(self, circuit) -> bool:
        return _AVAILABLE and circuit.is_clifford

    def _emit(self, append, g):
        if g.name in _SIMPLE:
            append(_SIMPLE[g.name], list(g.qubits))
            return
        if g.name in _PARAM:
            k = round(g.params[0] / _HALF_PI) % 4
            gate = _PARAM[g.name][k]
            if gate is not None:
                append(gate, list(g.qubits))
            return
        raise ValueError(f"stim backend: non-Clifford / unsupported gate '{g.name}'")

    def _unitary_circuit(self, circuit):
        c = stim.Circuit()
        for g in circuit.gates:
            self._emit(c.append, g)
        return c

    def exact_distribution(self, circuit, cutoff: float = 1e-12) -> Dict[str, float]:
        if not self.supports(circuit):
            raise RuntimeError("stim backend only supports fully-Clifford circuits")
        if circuit.n_qubits > self.max_exact_qubits:
            raise ValueError(
                f"stim full-distribution readout: n={circuit.n_qubits} exceeds 2^N "
                f"limit. (Native sampling via run(shots=...) still scales.)"
            )
        import numpy as np

        sim = stim.TableauSimulator()
        sim.do_circuit(self._unitary_circuit(circuit))
        sv = np.asarray(sim.state_vector())  # little-endian: qubit j is bit j of i
        p = np.abs(sv) ** 2
        n = circuit.n_qubits
        out = {}
        for i in range(len(p)):
            if p[i] > cutoff:
                key = "".join(str((i >> j) & 1) for j in range(n))  # q0 leftmost
                out[key] = float(p[i])
        return out

    def _sample(self, circuit, shots: int, seed, distribution) -> Dict[str, int]:
        """Native stim sampler -- scales beyond the exact-readout limit."""
        if not self.supports(circuit):
            raise RuntimeError("stim backend only supports fully-Clifford circuits")
        c = self._unitary_circuit(circuit)
        c.append("M", list(range(circuit.n_qubits)))
        arr = c.compile_sampler(seed=seed).sample(shots)  # col j == qubit j
        counts: Dict[str, int] = {}
        for row in arr:
            key = "".join("1" if b else "0" for b in row)
            counts[key] = counts.get(key, 0) + 1
        return counts
