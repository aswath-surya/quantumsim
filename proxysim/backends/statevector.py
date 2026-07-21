"""Statevector backend (qiskit).

Evolves |0...0> exactly with ``qiskit.quantum_info.Statevector`` (no Aer needed)
and reads off the exact computational-basis distribution.  qiskit is little-endian
(qubit 0 rightmost); keys are reversed to the canonical q0-leftmost convention.
"""

from __future__ import annotations

from typing import Dict

from .base import Backend

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    _AVAILABLE = True
    _IMPORT_ERR = None
except Exception as e:  # pragma: no cover
    _AVAILABLE = False
    _IMPORT_ERR = e

_METHOD = {"i": "id", "cnot": "cx"}


class StatevectorBackend(Backend):
    name = "statevector(qiskit)"
    available = _AVAILABLE
    import_error = _IMPORT_ERR
    _method = "exact dense statevector evolution"

    def supports(self, circuit) -> bool:
        return _AVAILABLE

    def _build(self, circuit):
        qc = QuantumCircuit(circuit.n_qubits)
        for g in circuit.gates:
            getattr(qc, _METHOD.get(g.name, g.name))(*g.params, *g.qubits)
        return qc

    def amplitude(self, circuit, bitstring: str = None) -> complex:
        """<x|psi> for a computational-basis string x (default |0...0>)."""
        x = bitstring if bitstring is not None else "0" * circuit.n_qubits
        idx = sum(int(b) << j for j, b in enumerate(x))     # q0 = LSB (qiskit)
        return complex(Statevector(self._build(circuit)).data[idx])

    def prob0(self, circuit) -> float:
        """|<0...0|psi>|^2 -- survival probability of a mirror circuit."""
        return abs(self.amplitude(circuit)) ** 2

    def exact_distribution(self, circuit, cutoff: float = 1e-12) -> Dict[str, float]:
        if circuit.n_qubits > self.max_exact_qubits:
            raise ValueError(f"statevector: n={circuit.n_qubits} exceeds 2^N readout limit")
        import numpy as np

        # .probabilities() is a numpy array (qiskit little-endian: bit j of index
        # i is qubit j). Avoid probabilities_dict(), which builds a giant string
        # label array for every basis state.
        p = np.asarray(Statevector(self._build(circuit)).probabilities())
        n = circuit.n_qubits
        out = {}
        for i in np.nonzero(p > cutoff)[0]:
            key = "".join(str((int(i) >> j) & 1) for j in range(n))  # q0 leftmost
            out[key] = float(p[i])
        return out
