"""GPU backends -- APPROACH STUBS (not yet implemented).

These are intentionally empty scaffolds capturing *how* to add GPU-accelerated
statevector and tensor-network simulation. They raise NotImplementedError so the
intent is explicit; fill in ``run`` when a CUDA toolchain is available.

Hardware/toolkit: NVIDIA GPU + CUDA 12.x.  NVIDIA's cuQuantum SDK provides
``cuStateVec`` (statevector) and ``cuTensorNet`` (tensor-network contraction);
both are reachable from Python via cupy and the simulators below.
"""

from __future__ import annotations

from typing import Optional

from .base import Backend, SimResult


class GPUStatevectorBackend(Backend):
    """GPU statevector via cuStateVec.  STUB.

    Approach
    --------
    Option A -- qiskit-aer GPU:
        from qiskit_aer import AerSimulator
        sim = AerSimulator(method="statevector", device="GPU")  # uses cuStateVec Probably can use this directly
        # transpile the IR-built QuantumCircuit, add measure_all(), run(shots),
        # then .get_counts(); reverse keys to canonical q0-leftmost.
        # install: pip install qiskit-aer-gpu  (CUDA 12.x wheels)

    Option B -- qulacs GPU:
        from qulacs import QuantumStateGpu, QuantumCircuit as QC
        state = QuantumStateGpu(n); ...; state.sampling(shots)
        # install: pip install qulacs-gpu

    Option C -- cuQuantum cuStateVec directly (cupy arrays) for full control.

    Memory wall is unchanged: a dense GPU statevector is still 2^N amplitudes,
    so GPU buys speed (and a bit more headroom) but not exponential scaling.
    """

    name = "gpu-statevector(cuStateVec)"
    available = False  # flip to True once a GPU build is wired up

    def run(self, circuit, shots: int, seed: Optional[int] = None) -> SimResult:
        raise NotImplementedError(
            "GPU statevector backend is a stub. See the docstring for the "
            "qiskit-aer-gpu / qulacs-gpu / cuStateVec approach, then implement run()."
        )


class GPUTensorNetworkBackend(Backend):
    """GPU tensor-network via cuTensorNet (quimb + cupy).  STUB.

    Approach
    --------
    quimb already supports non-numpy array backends through autoray, so the
    *same* ``Circuit.sample`` path can contract on the GPU:

        from proxysim.backends import TensorNetworkBackend
        tn = TensorNetworkBackend(contract_backend="cupy", dtype="complex64")
        # quimb dispatches the contractions to cupy (GPU). For best paths use
        # cotengra; cuTensorNet can execute the contraction tree on-device.

    So in practice GPU-TN is mostly a configuration of the existing CPU backend
    (``contract_backend='cupy'``) plus a cotengra/cuTensorNet path optimiser --
    this class exists to make that an explicit, named backend and to host any
    GPU-specific batching of the marginal chain.

    install: pip install cupy-cuda12x cotengra  (+ optional cuquantum-python)

    Unlike statevector, TN contraction memory scales with the contraction
    *width* (max bond dimension), not 2^N, so GPU-TN is the path to genuinely
    larger systems when entanglement is bounded.
    """

    name = "gpu-tensornetwork(cuTensorNet)"
    available = False  # flip to True once cupy/cuTensorNet is wired up

    def run(self, circuit, shots: int, seed: Optional[int] = None) -> SimResult:
        raise NotImplementedError(
            "GPU tensor-network backend is a stub. The near-term route is "
            "TensorNetworkBackend(contract_backend='cupy'); implement run() here "
            "to add cuTensorNet path execution and marginal batching."
        )
