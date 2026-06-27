"""Run a circuit across all backends and compare their EXACT output distributions.

Because the circuits are noiseless/unitary, the output is deterministic: each
backend computes the exact distribution P(x)=|<x|psi>|^2 once, by a completely
different method (dense statevector / MPS contraction / stabilizer tableau).
They should agree to floating-point precision -- that cross-method agreement is
the headline result.  Finite-shot sampling is optional (``shots>0``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .backends import (
    SimResult,
    StabilizerBackend,
    StatevectorBackend,
    TensorNetworkBackend,
    total_variation_distance,
)


@dataclass
class ComparisonReport:
    circuit_summary: str
    shots: int
    results: List[SimResult]
    skipped: Dict[str, str]
    reference: str
    tvd_vs_ref: Dict[str, float]          # exact-dist TVD to reference (~1e-15)
    sample_tvd: Dict[str, float] = field(default_factory=dict)  # sampled vs exact

    def to_text(self, top_k: int = 10) -> str:
        L = []
        L.append("=" * 74)
        L.append(self.circuit_summary)
        L.append(f"output is deterministic (no noise yet); shots = {self.shots or 0}"
                 + ("  [exact only]" if not self.shots else "  [+ finite sampling]"))
        L.append("=" * 74)

        # Timing / agreement table
        sample_col = self.shots and any(r.counts for r in self.results)
        head = f"{'backend':<26}{'exact (ms)':>12}"
        if sample_col:
            head += f"{'sample (ms)':>13}"
        head += f"{'support':>9}{'TVD@ref':>11}"
        L.append(head)
        L.append("-" * len(head))
        for r in self.results:
            row = f"{r.backend:<26}{r.t_compute * 1e3:>12.3f}"
            if sample_col:
                row += f"{r.t_sample * 1e3:>13.3f}"
            row += f"{r.support():>9}{self.tvd_vs_ref.get(r.backend, float('nan')):>11.1e}"
            L.append(row)
        for name, reason in self.skipped.items():
            L.append(f"{name:<26} -- skipped: {reason}")
        L.append(f"(TVD@ref = exact-distribution distance to '{self.reference}'; "
                 f"~1e-15 means identical)")
        L.append("")

        # Aligned EXACT distribution
        ref_dist = next((r.distribution for r in self.results if r.backend == self.reference),
                        self.results[0].distribution if self.results else {})
        rows = [bs for bs, _ in sorted(ref_dist.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]]
        short = {r.backend: r.backend.split("(")[-1].rstrip(")") for r in self.results}
        head = f"{'bitstring':<14}" + "".join(f"{short[r.backend]:>14}" for r in self.results)
        L.append("exact P(x) per outcome (q0 leftmost):")
        L.append(head)
        L.append("-" * len(head))
        for bs in rows:
            line = f"|{bs}>".ljust(14)
            for r in self.results:
                line += f"{r.distribution.get(bs, 0.0):>14.5f}"
            L.append(line)
        L.append("")
        if sample_col:
            L.append("finite-shot check: TVD(sampled, exact) per backend "
                     "(shot noise only):")
            for r in self.results:
                if r.counts is not None:
                    L.append(f"    {r.backend:<26} {self.sample_tvd.get(r.backend, float('nan')):.3e}")
            L.append("")
        return "\n".join(L)


def run_all(circuit, shots: int = 0, seed: Optional[int] = 1234) -> ComparisonReport:
    """Compute the exact output distribution on every supporting backend and
    compare them.  If ``shots>0`` also draw a finite sample from each (the hook
    for finite-hardware statistics / future noise)."""
    backends = [TensorNetworkBackend(), StatevectorBackend(), StabilizerBackend()]

    results: List[SimResult] = []
    skipped: Dict[str, str] = {}
    for b in backends:
        if not b.available:
            skipped[b.name] = f"not installed ({b.import_error})"
            continue
        if not b.supports(circuit):
            skipped[b.name] = "circuit not supported (non-Clifford)"
            continue
        results.append(b.run(circuit, shots=shots, seed=seed))

    # Pick a reference for the exact-distribution agreement check.
    ref = next((r.backend for r in results if r.backend.startswith("statevector")),
               results[0].backend if results else "")
    ref_dist = next((r.distribution for r in results if r.backend == ref), {})
    tvd_vs_ref = {r.backend: total_variation_distance(r.distribution, ref_dist) for r in results}

    sample_tvd = {}
    if shots:
        for r in results:
            sp = r.sampled_probs()
            if sp is not None:
                sample_tvd[r.backend] = total_variation_distance(sp, r.distribution)

    return ComparisonReport(
        circuit_summary=circuit.summary(),
        shots=shots or 0,
        results=results,
        skipped=skipped,
        reference=ref,
        tvd_vs_ref=tvd_vs_ref,
        sample_tvd=sample_tvd,
    )
