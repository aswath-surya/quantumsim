"""Backend base class, result container, and distribution helpers.

Design: the circuits here are (for now) **noiseless and unitary**, so the output
is a *deterministic* probability distribution P(x) = |<x|psi>|^2.  The primary
thing each backend produces is therefore that EXACT distribution, computed once
-- not a pile of Monte-Carlo shots.  Finite-shot sampling is an optional extra
(``shots > 0``): it just resamples the exact distribution to emulate finite
hardware statistics, and is the natural place noise will plug in later (a mixed
state must be sampled / averaged over trajectories).

Bitstring convention everywhere: index ``i`` is qubit ``i``, qubit 0 leftmost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


def multinomial_counts(distribution: Dict[str, float], shots: int, seed=None) -> Dict[str, int]:
    """Draw ``shots`` finite samples from an exact distribution (no re-simulation)."""
    import numpy as np

    keys = list(distribution.keys())
    p = np.array([distribution[k] for k in keys], dtype=float)
    p = p / p.sum()
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(shots, p)
    return {k: int(c) for k, c in zip(keys, draws) if c > 0}


@dataclass
class SimResult:
    """Result of simulating one circuit on one backend.

    ``distribution`` is the EXACT output distribution (the deterministic answer).
    ``counts`` is an optional finite-shot sample of it (None unless shots > 0).
    """

    backend: str
    n_qubits: int
    distribution: Dict[str, float]
    t_compute: float                  # seconds to compute the exact distribution
    clifford: bool
    method: str = ""
    shots: int = 0
    counts: Optional[Dict[str, int]] = None
    t_sample: float = 0.0             # seconds to draw the optional finite sample
    extra: dict = field(default_factory=dict)

    def probs(self) -> Dict[str, float]:
        """The exact distribution."""
        return self.distribution

    def sampled_probs(self) -> Optional[Dict[str, float]]:
        if not self.counts:
            return None
        tot = sum(self.counts.values()) or 1
        return {k: v / tot for k, v in self.counts.items()}

    def support(self) -> int:
        return len(self.distribution)

    def top(self, k: int = 8):
        return sorted(self.distribution.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


class Backend:
    """Abstract simulation backend."""

    name = "base"
    available = False
    import_error: Optional[BaseException] = None
    init_kwargs: dict = {}            # kwargs to rebuild an identical instance
    max_exact_qubits = 26             # full 2^N distribution readout ceiling

    def supports(self, circuit) -> bool:
        return self.available

    def exact_distribution(self, circuit, cutoff: float = 1e-12) -> Dict[str, float]:
        """Deterministic output distribution P(x)=|<x|psi>|^2 (canonical order)."""
        raise NotImplementedError

    def _sample(self, circuit, shots: int, seed, distribution) -> Dict[str, int]:
        """Default finite-shot sampler: multinomial resample of the exact dist.

        Backends with a genuinely scalable native sampler (e.g. stim) override
        this so that ``shots`` works even when the full distribution is too big
        to enumerate.
        """
        return multinomial_counts(distribution, shots, seed)

    def run(self, circuit, shots: int = 0, seed: Optional[int] = None,
            want_exact: bool = True) -> SimResult:
        import time

        dist: Dict[str, float] = {}
        t_compute = 0.0
        if want_exact:
            try:
                t0 = time.perf_counter()
                dist = self.exact_distribution(circuit)
                t_compute = time.perf_counter() - t0
            except (ValueError, MemoryError):
                # full 2^N distribution not computable at this size; fall back to
                # native sampling only (e.g. stim at thousands of qubits).
                dist = {}

        counts, t_sample = None, 0.0
        if shots:
            t0 = time.perf_counter()
            counts = self._sample(circuit, shots, seed, dist)
            t_sample = time.perf_counter() - t0

        return SimResult(
            backend=self.name,
            n_qubits=circuit.n_qubits,
            distribution=dist,
            t_compute=t_compute,
            clifford=getattr(circuit, "is_clifford", False),
            method=getattr(self, "_method", ""),
            shots=shots,
            counts=counts,
            t_sample=t_sample,
        )


# ---------------------------------------------------------------------------
# Distribution comparison helpers
# ---------------------------------------------------------------------------
def total_variation_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    """TVD = 1/2 sum_x |p(x) - q(x)|.  0 = identical, 1 = disjoint support."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)
