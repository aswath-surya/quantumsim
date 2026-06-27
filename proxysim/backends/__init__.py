"""Simulation backends for proxysim."""

from .base import Backend, SimResult, total_variation_distance
from .tensornetwork import TensorNetworkBackend
from .statevector import StatevectorBackend
from .stabilizer import StabilizerBackend
from .gpu import GPUStatevectorBackend, GPUTensorNetworkBackend

__all__ = [
    "Backend",
    "SimResult",
    "total_variation_distance",
    "TensorNetworkBackend",
    "StatevectorBackend",
    "StabilizerBackend",
    "GPUStatevectorBackend",
    "GPUTensorNetworkBackend",
    "all_backends",
]


def all_backends():
    """Return one instance of each CPU backend (available or not)."""
    return [
        TensorNetworkBackend(),
        StatevectorBackend(),
        StabilizerBackend(),
    ]
