"""proxysim -- a multi-backend quantum-circuit simulator + benchmarking toolkit.

Run the same backend-agnostic :class:`~proxysim.circuit.Circuit` on a
tensor-network (quimb MPS, optional GPU), exact statevector (qiskit), or stabilizer
(stim) backend, plus Pauli-propagation expectation values -- and bound circuit
error from cycle benchmarking. Everything is reachable from one dispatcher,
:func:`proxysim.simulate.simulate`.

Motivated by Merkel et al., "When Clifford benchmarks are sufficient"
(arXiv:2503.05943): Clifford "proxy" circuits are efficiently simulable
(stabilizer), while their non-Clifford targets need TN / statevector methods.
"""

# circuits
from .circuit import (Circuit, Gate, lnn_brickwork, brickwork_magic,
                      clifford_entropy_circuit, bench_brickwork, even_pairs,
                      odd_pairs, GATE_SETS)
# unified entry point
from .simulate import simulate, make_backend, auto_simulator, noisy_survival
# distribution comparison + metrics + I/O
from .runner import run_all, ComparisonReport
from . import metrics
from .metrics import (total_variation_distance, classical_fidelity,
                      hellinger_fidelity, shannon_entropy, tvd_to_ideal_support,
                      save_results, load_results)
# parallelism
from .parallel import pmap, parallel_trajectories, sample_parallel
# noise + benchmarking
from . import noise
from .noise import NoiseModel
from . import benchmarking
from .benchmarking import cycle_benchmark, qcap_bound, readout_fidelity
# randomized compiling + QASM emission (building circuit banks for external runners)
from . import rc, cb_emit, mqtbank
from .rc import pauli_twirl, randomly_compile
from .cb_emit import cb_circuit, survival, analyze_cb
from .qasm import circuit_to_qasm, write_qasm
# Pauli propagation
from .pauliprop import JuliaPauliPropagator
from .pauliprop_validator import PauliPropagator, expectation as pp_expectation
# backends
from .backends import (TensorNetworkBackend, StatevectorBackend, StabilizerBackend,
                       GPUStatevectorBackend, GPUTensorNetworkBackend, SimResult)

__all__ = [
    # circuits
    "Circuit", "Gate", "lnn_brickwork", "brickwork_magic", "clifford_entropy_circuit",
    "bench_brickwork", "even_pairs", "odd_pairs", "GATE_SETS",
    # dispatcher
    "simulate", "make_backend", "auto_simulator", "noisy_survival",
    # comparison / metrics / io
    "run_all", "ComparisonReport", "metrics", "total_variation_distance",
    "classical_fidelity", "hellinger_fidelity", "shannon_entropy",
    "tvd_to_ideal_support", "save_results", "load_results",
    # parallel
    "pmap", "parallel_trajectories", "sample_parallel",
    # noise / benchmarking
    "noise", "NoiseModel", "benchmarking", "cycle_benchmark", "qcap_bound",
    "readout_fidelity",
    # randomized compiling / QASM emission
    "rc", "cb_emit", "mqtbank", "pauli_twirl", "randomly_compile", "cb_circuit",
    "survival", "analyze_cb", "circuit_to_qasm", "write_qasm",
    # pauli propagation
    "JuliaPauliPropagator", "PauliPropagator", "pp_expectation",
    # backends
    "TensorNetworkBackend", "StatevectorBackend", "StabilizerBackend",
    "GPUStatevectorBackend", "GPUTensorNetworkBackend", "SimResult",
]

__version__ = "0.2.0"
