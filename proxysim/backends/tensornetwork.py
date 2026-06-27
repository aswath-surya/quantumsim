"""Tensor-network backend (quimb), MPS / "swap+split".

For the 1D nearest-neighbour circuits here, the right quimb tool is
``CircuitMPS`` with ``gate_contract='swap+split'``: gates are applied to a matrix
product state, two-qubit gates on non-adjacent sites are handled by SWAP-ing
qubits adjacent, contracting, and splitting (SVD) back into MPS form with bond
compression.  The MPS is built ONCE; the exact output distribution is then read
off from it.  Bond dimension stays small while entanglement is bounded (e.g. a
Clifford brickwork has max_bond ~ 2), so this is far cheaper than a dense state.

Why not draw shots by re-contracting the network per shot?  Because the circuit
is noiseless and unitary -> the distribution is deterministic, so we compute it
once.  Finite-shot sampling (``shots>0``) just resamples that exact distribution.
(quimb's native per-shot perfect sampler ``CircuitMPS.sample`` exists for the
large-N regime where the full 2^N distribution cannot be stored; it is not used
on the default path.)

Might need to move off CircuitMPS for mid-circuit measurement / reset / leakage
and other exotic channels; quimb's circuit classes are limited there.
"""

from __future__ import annotations

from typing import Dict, Optional

from .base import Backend

try:
    import quimb.tensor as qtn

    _AVAILABLE = True
    _IMPORT_ERR = None
except Exception as e:  # pragma: no cover
    _AVAILABLE = False
    _IMPORT_ERR = e

# IR gate name -> quimb gate id
_NAME = {
    "i": "IDEN", "x": "X", "y": "Y", "z": "Z", "h": "H",
    "s": "S", "sdg": "SDG", "sx": "SX", "sxdg": "SXDG",
    "t": "T", "tdg": "TDG",
    "cx": "CX", "cnot": "CX", "cz": "CZ", "cy": "CY", "swap": "SWAP",
    "rx": "RX", "ry": "RY", "rz": "RZ", "p": "PHASE",
}


class TensorNetworkBackend(Backend):
    name = "tensornetwork(quimb-MPS)"
    available = _AVAILABLE
    import_error = _IMPORT_ERR
    _method = "MPS (swap+split), exact readout"

    def __init__(
        self,
        max_bond: Optional[int] = None,   # None -> exact (no bond truncation)
        cutoff: float = 1e-12,            # SVD truncation threshold
        dtype: str = "complex128",
        contract_backend: Optional[str] = None,  # 'cupy' for GPU contractions
    ):
        self.max_bond = max_bond
        self.cutoff = cutoff
        self.dtype = dtype
        self.contract_backend = contract_backend
        self.init_kwargs = dict(
            max_bond=max_bond, cutoff=cutoff, dtype=dtype,
            contract_backend=contract_backend,
        )

    def supports(self, circuit) -> bool:
        return _AVAILABLE

    def _build(self, circuit):
        """Apply gates to an MPS with swap+split -- built once, no dense vector."""
        C = qtn.CircuitMPS(
            circuit.n_qubits,
            gate_contract="swap+split",
            max_bond=self.max_bond,
            cutoff=self.cutoff,
            dtype=self.dtype,
        )
        for g in circuit.gates:
            gid = _NAME.get(g.name)
            if gid is None:
                raise ValueError(f"quimb backend: unsupported gate '{g.name}'")
            C.apply_gate(gid, *g.params, *g.qubits)
        return C

    def exact_distribution(self, circuit, cutoff: float = 1e-12) -> Dict[str, float]:
        if circuit.n_qubits > self.max_exact_qubits:
            raise ValueError(
                f"MPS full-distribution readout: n={circuit.n_qubits} exceeds 2^N "
                f"limit. (Per-amplitude / expectation queries stay scalable; the "
                f"FULL distribution does not.)"
            )
        import numpy as np

        C = self._build(circuit)
        psi = np.asarray(C.psi.to_dense()).reshape(-1)  # q0 = most significant
        p = np.abs(psi) ** 2
        n = circuit.n_qubits
        return {format(i, f"0{n}b"): float(p[i]) for i in range(len(p)) if p[i] > cutoff}

    def max_bond_of(self, circuit) -> int:
        """Largest MPS bond dimension reached (a proxy for entanglement)."""
        return int(self._build(circuit).psi.max_bond())
