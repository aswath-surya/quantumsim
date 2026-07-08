"""proxysim -- a tiny general-purpose multi-backend quantum-circuit simulator.

Run the same backend-agnostic :class:`~proxysim.circuit.Circuit` on:
  * a tensor-network simulator (quimb, native ``Circuit.sample`` contraction),
  * an exact statevector simulator (qiskit), and
  * a stabilizer simulator (stim, Clifford circuits only),
each reporting the wall-clock time taken and the output bitstring distribution.

Motivated by Merkel et al., "When Clifford benchmarks are sufficient"
(arXiv:2503.05943): Clifford "proxy" circuits are efficiently simulable
(stabilizer), while their non-Clifford targets need TN/statevector methods.
"""

from .circuit import Circuit, Gate, lnn_brickwork, brickwork_magic
from .runner import run_all, ComparisonReport
from .parallel import sample_parallel
from . import noise
from .noise import NoiseModel
from .pauliprop import JuliaPauliPropagator
from .pauliprop_validator import PauliPropagator, expectation as pp_expectation
from .backends import (
    TensorNetworkBackend,
    StatevectorBackend,
    StabilizerBackend,
    GPUStatevectorBackend,
    GPUTensorNetworkBackend,
    SimResult,
    total_variation_distance,
)

__all__ = [
    "Circuit",
    "Gate",
    "lnn_brickwork",
    "brickwork_magic",
    "run_all",
    "ComparisonReport",
    "sample_parallel",
    "noise",
    "NoiseModel",
    "JuliaPauliPropagator",
    "PauliPropagator",
    "pp_expectation",
    "TensorNetworkBackend",
    "StatevectorBackend",
    "StabilizerBackend",
    "GPUStatevectorBackend",
    "GPUTensorNetworkBackend",
    "SimResult",
    "total_variation_distance",
]

__version__ = "0.1.0"
