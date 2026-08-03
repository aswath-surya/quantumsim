"""Python wrapper around the real PauliPropagation.jl (Julia), via juliacall.

Lazily boots a Julia session, translates a proxysim `Circuit` into
PauliPropagation.jl gates, and calls `propagate` + `overlapwithzero` to get the
Heisenberg-picture expectation value <psi|O|psi> (a scalar -- NOT a statevector
or bitstring distribution).

Validated: matches qiskit statevector expectation values to ~1e-16, and agrees
with the Julia-free reference in `proxysim.pauliprop_validator`.

Enable it:
    pip install juliacall
    python -c "from proxysim.pauliprop import ensure_installed; ensure_installed()"  # once

Best practices for wrapping a Julia package from Python (what this module does):
  * Use juliacall / PythonCall.jl (NOT the older PyJulia) -- it manages a private,
    reproducible Julia depot/environment.
  * Boot Julia LAZILY and ONCE; the first call pays a JIT/precompile cost, later
    calls are fast. Cache the `Main` handle (see `_boot`).
  * Marshal only cheap plain data (ints, floats, small arrays); build the heavy
    objects (gates, PauliSums) on the Julia side.
  * Julia is 1-INDEXED: qubit q (python) -> q+1 (julia).
  * `!`-suffixed Julia functions (push!) are exposed by PythonCall as `push_b`.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Union

ObservableLike = Union[str, Dict[str, complex], Dict[int, str]]

# Robust IR -> PauliPropagation.jl mapping.  Single-qubit gates become Pauli
# rotations exp(-i theta/2 P) where possible (the most portable primitive); only
# H and the 2-qubit entanglers use CliffordGate.
_ROT = {
    "rz": ("Z", None), "rx": ("X", None), "ry": ("Y", None),
    "s": ("Z", math.pi / 2), "sdg": ("Z", -math.pi / 2),
    "sx": ("X", math.pi / 2), "sxdg": ("X", -math.pi / 2),
    "x": ("X", math.pi), "y": ("Y", math.pi), "z": ("Z", math.pi),
}
_CLIFFORD = {"h": "H", "cz": "CZ", "cx": "CNOT", "cnot": "CNOT", "cy": "CY", "swap": "SWAP"}

_JL = None  # cached juliacall.Main


def available() -> bool:
    try:
        import juliacall  # noqa: F401

        return True
    except Exception:
        return False


def ensure_installed():
    """Install PauliPropagation.jl into juliacall's Julia environment (run once)."""
    import juliacall

    jl = juliacall.Main
    jl.seval('import Pkg; Pkg.add("PauliPropagation")')
    jl.seval("using PauliPropagation")
    global _JL
    _JL = jl


def _boot():
    """Lazily import juliacall and `using PauliPropagation` (cached)."""
    global _JL
    if _JL is not None:
        return _JL
    import juliacall

    jl = juliacall.Main
    jl.seval("using PauliPropagation")
    _JL = jl
    return jl


def normalize_observable(observable: ObservableLike, n: int) -> Dict[str, complex]:
    """Coerce an observable spec into {letters: coeff} (q0 leftmost, '_'=I)."""
    if isinstance(observable, str):
        if len(observable) != n:
            raise ValueError(f"observable string length {len(observable)} != n={n}")
        return {observable.upper(): 1.0 + 0j}
    if isinstance(observable, dict):
        if observable and all(isinstance(k, int) for k in observable):
            letters = ["_"] * n
            for q, p in observable.items():
                letters[q] = p.upper()
            return {"".join(letters): 1.0 + 0j}
        return {k.upper(): complex(v) for k, v in observable.items()}
    raise TypeError("observable must be a letter string, {qubit:pauli}, or {letters:coeff}")


class JuliaPauliPropagator:
    """Expectation values via PauliPropagation.jl (Heisenberg picture)."""

    name = "pauli-propagation(PauliPropagation.jl)"

    def __init__(self, max_weight: Optional[int] = None, min_abs_coeff: float = 0.0):
        self.max_weight = max_weight
        self.min_abs_coeff = min_abs_coeff

    def _build_circuit(self, jl, circuit):
        gates = jl.seval("PauliPropagation.Gate[]")
        thetas = []
        for g in circuit.gates:
            if g.name in _ROT:
                pauli, fixed = _ROT[g.name]
                theta = fixed if fixed is not None else g.params[0]
                jl.push_b(gates, jl.PauliRotation(jl.Symbol(pauli), g.qubits[0] + 1))
                thetas.append(float(theta))
            elif g.name in _CLIFFORD:
                sym = jl.Symbol(_CLIFFORD[g.name])
                if len(g.qubits) == 1:
                    jl.push_b(gates, jl.CliffordGate(sym, g.qubits[0] + 1))
                else:
                    jl.push_b(gates, jl.CliffordGate(sym, [q + 1 for q in g.qubits]))
            else:
                raise ValueError(f"pauliprop wrapper: unsupported gate '{g.name}'")
        return gates, thetas

    def _build_observable(self, jl, letters: str):
        syms = [(l, i + 1) for i, l in enumerate(letters) if l != "_"]
        if not syms:
            return None  # identity
        return jl.PauliString(len(letters), [jl.Symbol(l) for l, _ in syms],
                              [q for _, q in syms])

    def expectation(self, circuit, observable: ObservableLike) -> float:
        terms = normalize_observable(observable, circuit.n_qubits)
        if len(terms) != 1:
            raise NotImplementedError("wrapper takes a single Pauli observable; "
                                      "sum expectations for a general PauliSum.")
        letters, coeff = next(iter(terms.items()))

        jl = _boot()
        obs = self._build_observable(jl, letters)
        if obs is None:  # identity observable
            return float(coeff.real)
        gates, thetas = self._build_circuit(jl, circuit)

        kwargs = {}
        if self.max_weight is not None:
            kwargs["max_weight"] = self.max_weight
        if self.min_abs_coeff:
            kwargs["min_abs_coeff"] = self.min_abs_coeff
        psum = jl.propagate(gates, obs, thetas, **kwargs)
        return float(coeff.real * jl.overlapwithzero(psum))
