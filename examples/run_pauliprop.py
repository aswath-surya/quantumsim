"""Pauli propagation (Heisenberg picture): expectation values <psi|O|psi>.

Uses the wrapper around PauliPropagation.jl (proxysim.pauliprop) when Julia is
available, and always cross-checks against the Julia-free validator
(proxysim.pauliprop_validator) and exact qiskit statevector.

Also demonstrates the core truncation knob (max_weight): with a small max_weight
the propagated observable is truncated and the expectation is approximate; the
error shrinks as max_weight grows. (This is the lever that makes the method scale
-- especially under noise, where high-weight Paulis decay and truncation becomes
nearly lossless.)

Run:  python examples/run_pauliprop.py
      # first time only, to install the Julia package into juliacall's env:
      #   python -c "from proxysim.pauliprop import ensure_installed; ensure_installed()"
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import _bootstrap  # noqa: F401
from proxysim import lnn_brickwork
from proxysim.pauliprop import JuliaPauliPropagator, available
from proxysim.pauliprop_validator import PauliPropagator, depolarizing_damping
from proxysim.backends.statevector import StatevectorBackend
from qiskit.quantum_info import Statevector, Pauli

N = 6
N_CYCLES = 3
SEED = 3
OBS = "Z" + "_" * (N - 1)        # <Z_0>


def sv_expect(circ, letters):
    psi = Statevector(StatevectorBackend()._build(circ))
    return psi.expectation_value(Pauli(letters.replace("_", "I")[::-1])).real


def main():
    circ = lnn_brickwork(N, N_CYCLES, twoq="cz", mode="haar", seed=SEED)
    print(circ.summary())
    print(f"observable O = {OBS}  (i.e. <Z_0>)\n")

    exact = sv_expect(circ, OBS)
    val = PauliPropagator().expectation(circ, OBS)          # Julia-free reference
    print(f"exact statevector      <O> = {exact:+.10f}")
    print(f"validator (stim,python) <O> = {val:+.10f}   err={abs(val-exact):.1e}")

    if available():
        jl = JuliaPauliPropagator().expectation(circ, OBS)  # the real .jl wrapper
        print(f"PauliPropagation.jl     <O> = {jl:+.10f}   err={abs(jl-exact):.1e}")
    else:
        print("PauliPropagation.jl     <O> = (juliacall not installed -- pip install juliacall)")

    # --- truncation knob: error vs max_weight (validator; the wrapper takes the
    #     same max_weight kwarg) ---
    print("\ntruncation by max_weight (no noise):")
    print(f"  {'max_weight':>11}{'<O>':>16}{'|error|':>12}")
    for mw in [1, 2, 3, 4, None]:
        e = PauliPropagator(max_weight=mw).expectation(circ, OBS)
        label = "inf" if mw is None else str(mw)
        print(f"  {label:>11}{e:>16.10f}{abs(e-exact):>12.1e}")

    # --- with depolarizing noise, high-weight Paulis decay -> aggressive
    #     truncation becomes nearly lossless (the method's real strength) ---
    print("\nwith depolarizing damping p=0.05, small max_weight stays accurate:")
    noisy_full = PauliPropagator(noise=depolarizing_damping(0.05)).expectation(circ, OBS)
    noisy_trunc = PauliPropagator(max_weight=2, noise=depolarizing_damping(0.05)).expectation(circ, OBS)
    print(f"  noisy <O>, max_weight=inf : {noisy_full:+.10f}")
    print(f"  noisy <O>, max_weight=2   : {noisy_trunc:+.10f}"
          f"   (diff {abs(noisy_full-noisy_trunc):.1e})")


if __name__ == "__main__":
    main()
