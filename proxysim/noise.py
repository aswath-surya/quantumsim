"""Noise models -- APPROACH STUB (not yet implemented).

Right now the simulators are NOISELESS and unitary, so the output distribution is
deterministic and we compute it exactly (no shots needed -- shots only add
Monte-Carlo noise to an exact answer).

Noise is where finite-shot sampling becomes essential, and is the whole point of
the Merkel et al. "Clifford benchmarks" setting (Pauli Twirling Assumption):

  * Under the PTA each layer's error is a Pauli-stochastic channel
    E(rho) = sum_j p_j  P_j rho P_j.  A noisy circuit is then a mixed state; you
    can no longer read off a single |<x|psi>|^2.

  * Two standard ways to get a distribution from a noisy circuit:
      1. Density-matrix / superoperator simulation -> exact noisy distribution
         (cost ~ (2^N)^2; or an MPO / noisy-MPS for bounded correlations).
      2. Monte-Carlo TRAJECTORIES: for each shot, sample which Pauli error fires
         per layer (per p_j), apply it to the pure state, then measure. Averaging
         many shots reproduces the mixed-state distribution. THIS is where "many
         shots" earns its keep, and where parallel.sample_parallel / trajectory
         parallelism pays off.

  * stim handles Pauli noise natively (DEPOLARIZE1/2, X_ERROR, ...), so the
    stabilizer backend can sample noisy Clifford circuits directly and at scale.

Planned API (sketch):

    @dataclass
    class PauliNoise:
        p1: float    # 1-qubit depolarizing rate (per single-qubit gate)
        p2: float    # 2-qubit depolarizing rate (per entangling gate)
        # ... readout error, biased Pauli channels, per-gate overrides ...

    def apply(circuit, noise) -> NoisyCircuit: ...

    # backends then expose noisy sampling:
    #   stim     : insert DEPOLARIZE1/2 after gates -> native noisy CHP sampling
    #   quimb    : MPO evolution, or pure-state trajectories + averaging
    #   qiskit   : density_matrix (small N) or trajectories
"""

from __future__ import annotations


class PauliNoise:  # placeholder
    def __init__(self, p1: float = 0.0, p2: float = 0.0, readout: float = 0.0):
        self.p1, self.p2, self.readout = p1, p2, readout

    def __repr__(self):
        return f"PauliNoise(p1={self.p1}, p2={self.p2}, readout={self.readout})"


def apply(circuit, noise: "PauliNoise"):
    raise NotImplementedError(
        "Noise is not implemented yet. See this module's docstring for the planned "
        "PTA / trajectory approach -- that is where finite-shot sampling matters."
    )
