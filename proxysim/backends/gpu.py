"""GPU backends.

GPUTensorNetworkBackend is REAL: it is the quimb MPS backend with the array
backend set to cupy, so every contraction runs on the GPU. It just needs a CUDA
GPU + ``cupy`` (``pip install cupy-cuda12x``); with no GPU it reports
``available == False`` and you fall back to the CPU TensorNetworkBackend.

GPUStatevectorBackend is still a STUB -- a dense GPU statevector is a bigger lift
(qiskit-aer-gpu / qulacs-gpu / cuStateVec) and, unlike the TN case, does not change
the 2^N memory wall, only the speed. The approach is documented below.
"""

from __future__ import annotations

from typing import Optional

from .base import Backend, SimResult
from .tensornetwork import TensorNetworkBackend, gpu_available


class GPUTensorNetworkBackend(TensorNetworkBackend):
    """quimb MPS contractions on the GPU (cupy). Same API as the CPU backend."""

    name = "gpu-tensornetwork(quimb+cupy)"
    available = gpu_available()

    def __init__(self, max_bond: Optional[int] = None, cutoff: float = 1e-12,
                 dtype: str = "complex64"):
        # complex64 is usually the sweet spot on GPU (2x throughput, plenty for shots)
        super().__init__(max_bond=max_bond, cutoff=cutoff, dtype=dtype, gpu=True)


class GPUStatevectorBackend(Backend):
    """GPU statevector via cuStateVec -- STUB.

    Approach (fill in run()):
      * qiskit-aer GPU:
            from qiskit_aer import AerSimulator
            sim = AerSimulator(method="statevector", device="GPU")   # cuStateVec
        # transpile the IR circuit, measure_all(), run(shots), reverse the keys.
        # install: pip install qiskit-aer-gpu   (CUDA 12.x wheels)
      * qulacs GPU:  from qulacs import QuantumStateGpu; state.sampling(shots)
        # install: pip install qulacs-gpu
    The 2^N memory wall is unchanged -- GPU buys speed, not scaling.
    """

    name = "gpu-statevector(cuStateVec)"
    available = False

    def exact_distribution(self, circuit, cutoff: float = 1e-12):
        raise NotImplementedError(
            "GPU statevector is a stub -- see the docstring for the qiskit-aer-gpu / "
            "qulacs-gpu route, then implement exact_distribution()."
        )
