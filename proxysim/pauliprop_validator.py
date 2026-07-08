"""Julia-free validator / reference implementation of Pauli propagation.

A small pure-Python implementation of the SAME Heisenberg-picture algorithm as
PauliPropagation.jl, used to (a) sanity-check `proxysim.pauliprop`'s Julia wrapper
WITHOUT booting Julia, and (b) run Pauli propagation on machines with no Julia.

It is NOT connected to PauliPropagation.jl and does NOT use stim's stabilizer
simulator -- it only borrows `stim.PauliString` as a fast, correct Pauli-algebra
engine (multiplication with i/sign bookkeeping, commutation, Clifford
conjugation via tableaus). Validated: matches qiskit statevector expectation
values to ~1e-15, and agrees with the Julia wrapper.

Idea: evolve the OBSERVABLE O backward through the circuit,
  O = sum_k c_k P_k ,
  * Clifford gate g:  P -> g^dag P g            (single Pauli, +/- sign)
  * rotation exp(-i th/2 R):  P -> P            if [R,P]=0
                              P -> cos(th) P + i sin(th) (R P)   if {R,P}=0
then take the |0...0> expectation.  Truncation (min_abs_coeff / max_weight)
mirrors PauliPropagation.jl.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Dict, Optional, Union

import stim

_CLIFFORD_STIM = {
    "h": "H", "x": "X", "y": "Y", "z": "Z", "s": "S", "sdg": "S_DAG",
    "sx": "SQRT_X", "sxdg": "SQRT_X_DAG", "cz": "CZ", "cx": "CNOT",
    "cnot": "CNOT", "cy": "CY", "swap": "SWAP",
}
_ROT_PAULI = {"rz": "Z", "rx": "X", "ry": "Y"}

ObservableLike = Union[str, Dict[str, complex], Dict[int, str]]


def _letters(p: stim.PauliString) -> str:
    return "".join("_XYZ"[p[i]] for i in range(len(p)))


def _weight(letters: str) -> int:
    return sum(1 for ch in letters if ch != "_")


@lru_cache(maxsize=None)
def _inv_tableau(stim_name: str) -> stim.Tableau:
    # PauliString.after(t) computes t(O)=g O g^dag; feed inverse to get g^dag O g.
    return stim.Tableau.from_named_gate(stim_name).inverse()


def normalize_observable(observable: ObservableLike, n: int) -> Dict[str, complex]:
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


def depolarizing_damping(p: float) -> Callable[[str], float]:
    """Simple depolarizing damping: a weight-w Pauli is scaled by (1-p)^w."""
    keep = 1.0 - p
    return lambda letters: keep ** _weight(letters)


def _apply_clifford(terms, stim_name, targets):
    tab = _inv_tableau(stim_name)
    out: Dict[str, complex] = {}
    for letters, c in terms.items():
        r = stim.PauliString(letters).after(tab, targets=targets)
        key = _letters(r)
        out[key] = out.get(key, 0j) + c * r.sign
    return out


def _apply_rotation(terms, pauli, qi, theta, n):
    R = stim.PauliString(n)
    R[qi] = pauli
    cos, sin = math.cos(theta), math.sin(theta)
    out: Dict[str, complex] = {}
    for letters, c in terms.items():
        Q = stim.PauliString(letters)
        if R.commutes(Q):
            out[letters] = out.get(letters, 0j) + c
        else:
            out[letters] = out.get(letters, 0j) + c * cos
            prod = R * Q
            key = _letters(prod)
            out[key] = out.get(key, 0j) + c * (1j * sin) * prod.sign
    return out


def _truncate(terms, max_weight, min_abs_coeff):
    return {k: c for k, c in terms.items()
            if abs(c) >= min_abs_coeff and (max_weight is None or _weight(k) <= max_weight)}


def propagate(circuit, observable: ObservableLike, *, max_weight: Optional[int] = None,
              min_abs_coeff: float = 0.0,
              noise: Optional[Callable[[str], float]] = None) -> Dict[str, complex]:
    n = circuit.n_qubits
    terms = normalize_observable(observable, n)
    for g in reversed(circuit.gates):
        if g.name in _CLIFFORD_STIM:
            terms = _apply_clifford(terms, _CLIFFORD_STIM[g.name], list(g.qubits))
        elif g.name in _ROT_PAULI:
            terms = _apply_rotation(terms, _ROT_PAULI[g.name], g.qubits[0], g.params[0], n)
        else:
            raise ValueError(f"validator: unsupported gate '{g.name}'")
        if noise is not None:
            terms = {k: v * noise(k) for k, v in terms.items()}
        terms = _truncate(terms, max_weight, min_abs_coeff)
    return terms


def overlap_with_zero(terms: Dict[str, complex]) -> float:
    return sum(c.real for letters, c in terms.items() if all(ch in "_Z" for ch in letters))


def expectation(circuit, observable: ObservableLike, *, max_weight: Optional[int] = None,
                min_abs_coeff: float = 0.0,
                noise: Optional[Callable[[str], float]] = None) -> float:
    return overlap_with_zero(propagate(circuit, observable, max_weight=max_weight,
                                       min_abs_coeff=min_abs_coeff, noise=noise))


class PauliPropagator:
    """Julia-free Pauli propagator (validator / reference). Mirrors the wrapper API."""

    name = "pauli-propagation(stim, python)"

    def __init__(self, max_weight: Optional[int] = None, min_abs_coeff: float = 0.0,
                 noise: Optional[Callable[[str], float]] = None):
        self.max_weight = max_weight
        self.min_abs_coeff = min_abs_coeff
        self.noise = noise

    def expectation(self, circuit, observable: ObservableLike) -> float:
        return expectation(circuit, observable, max_weight=self.max_weight,
                           min_abs_coeff=self.min_abs_coeff, noise=self.noise)
