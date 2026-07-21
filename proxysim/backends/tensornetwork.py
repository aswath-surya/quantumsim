"""Tensor-network backend (quimb), MPS / "swap+split", with optional GPU.

For the 1D nearest-neighbour circuits here the right quimb tool is ``CircuitMPS``
with ``gate_contract='swap+split'``: gates are applied to a matrix product state,
two-qubit gates on non-adjacent sites are handled by SWAP-ing qubits adjacent,
contracting, and splitting (SVD) back into MPS form with bond compression.

GPU: pass ``gpu=True`` (or ``to_backend='cupy'``). quimb stores the MPS tensors as
cupy arrays and dispatches every contraction to the GPU through autoray -- the
same code path, just a different array backend. Needs a CUDA GPU + ``cupy``.

Readouts available from the MPS, cheapest first:
  * ``amplitude(circuit, bitstring)``  -- <x|psi>/||psi||, O(n D^2), no dense vector
    (this is what mirror/survival P(0...0) estimates use).
  * ``exact_distribution(circuit)``    -- full 2^N distribution (small N only).
  * ``max_bond_of(circuit)``           -- largest bond dimension (entanglement proxy).
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


def gpu_available() -> bool:
    """True if a working cupy / CUDA GPU is importable."""
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


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
        gpu: bool = False,                # run contractions on GPU via cupy
        to_backend: Optional[str] = None,  # explicit array backend ('cupy', 'jax', ...)
    ):
        self.max_bond = max_bond
        self.cutoff = cutoff
        self.dtype = dtype
        self.to_backend = "cupy" if gpu and to_backend is None else to_backend
        self.gpu = gpu or self.to_backend == "cupy"
        self.init_kwargs = dict(max_bond=max_bond, cutoff=cutoff, dtype=dtype,
                                to_backend=self.to_backend)

    def supports(self, circuit) -> bool:
        return _AVAILABLE

    def _build(self, circuit):
        """Apply gates to an MPS with swap+split -- built once, no dense vector.

        ``to_backend='cupy'`` puts the tensors on the GPU."""
        kw = dict(gate_contract="swap+split", max_bond=self.max_bond,
                  cutoff=self.cutoff, dtype=self.dtype)
        if self.to_backend:
            kw["to_backend"] = self.to_backend
        C = qtn.CircuitMPS(circuit.n_qubits, **kw)
        for g in circuit.gates:
            gid = _NAME.get(g.name)
            if gid is None:
                raise ValueError(f"quimb backend: unsupported gate '{g.name}'")
            C.apply_gate(gid, *g.params, *g.qubits)
        return C

    def amplitude(self, circuit, bitstring: Optional[str] = None) -> complex:
        """<x|psi> / ||psi|| for a computational-basis string x (default |0...0>).

        Contracts the MPS against a product state -- O(n * bond^2), so it scales
        to large n even where the full distribution does not. Normalised, so it is
        a valid amplitude even when a finite ``max_bond`` truncated the state."""
        import numpy as np

        x = bitstring if bitstring is not None else "0" * circuit.n_qubits
        psi = self._build(circuit).psi
        amp = psi.isel({f"k{i}": int(b) for i, b in enumerate(x)}).contract()
        amp = complex(np.asarray(amp))
        norm = float(psi.norm())
        return amp / norm if norm else 0.0

    def prob0(self, circuit) -> float:
        """|<0...0|psi>|^2 -- the survival probability of a mirror circuit."""
        return abs(self.amplitude(circuit)) ** 2

    def exact_distribution(self, circuit, cutoff: float = 1e-12) -> Dict[str, float]:
        if circuit.n_qubits > self.max_exact_qubits:
            raise ValueError(
                f"MPS full-distribution readout: n={circuit.n_qubits} exceeds 2^N "
                f"limit. (amplitude()/expectation queries stay scalable; the FULL "
                f"distribution does not.)"
            )
        import numpy as np

        psi = np.asarray(self._build(circuit).psi.to_dense()).reshape(-1)
        p = np.abs(psi) ** 2
        n = circuit.n_qubits
        return {format(i, f"0{n}b"): float(p[i]) for i in range(len(p)) if p[i] > cutoff}

    def max_bond_of(self, circuit) -> int:
        """Largest MPS bond dimension reached (a proxy for entanglement)."""
        return int(self._build(circuit).psi.max_bond())
