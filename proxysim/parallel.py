"""Parallel sampling -- SCAFFOLD (toggle-able, opt-in).

Splits a backend's shots across worker processes (or threads), runs each chunk
with a *distinct* seed, and merges the bitstring counts.

NOTE on when this actually helps: for the current NOISELESS circuits the output
is deterministic, so we compute the exact distribution once and sampling is a
cheap multinomial -- there is nothing worth parallelising, and naive shot-split
even *loses* (each worker re-does the per-call setup; process spawn re-imports
quimb). This module is kept as a toggle-able scaffold for the regimes where shot
parallelism is the right tool:
  * NOISE trajectories -- each shot samples a different Pauli-error realisation
    (see proxysim.noise); thousands of independent trajectories parallelise well.
  * MANY circuits / randomisations -- e.g. averaging over Cliffordizations.

Process workers require ``proxysim`` importable in the child (``pip install -e .``);
the thread executor needs no install. GPU backends are single-device -- prefer one
process with a GPU contraction backend over many CPU processes.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Optional

from .backends.base import SimResult


def _chunk_shots(shots: int, n: int):
    base, rem = divmod(shots, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def _worker(payload):
    """Top-level so it is picklable for ProcessPoolExecutor (Windows spawn)."""
    backend_cls, init_kwargs, circuit, shots, seed = payload
    backend = backend_cls(**init_kwargs)
    r = backend.run(circuit, shots=shots, seed=seed)
    return r.counts, r.t_run


def sample_parallel(
    backend,
    circuit,
    shots: int,
    n_workers: Optional[int] = None,
    base_seed: int = 0,
    executor: str = "process",
) -> SimResult:
    """Run ``shots`` across ``n_workers`` workers and merge the counts.

    Parameters
    ----------
    backend    : a Backend instance (its class + ``init_kwargs`` are rebuilt in
                 each worker).
    n_workers  : worker count (default: os.cpu_count()).
    base_seed  : worker ``i`` uses seed ``base_seed + i + 1`` (distinct streams).
    executor   : 'process' (CPU-bound, needs install) or 'thread' (no install).
    """
    n_workers = n_workers or os.cpu_count() or 1
    chunks = [c for c in _chunk_shots(shots, n_workers) if c > 0]
    payloads = [
        (type(backend), getattr(backend, "init_kwargs", {}), circuit, c, base_seed + i + 1)
        for i, c in enumerate(chunks)
    ]
    Exec = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor

    counts: dict = {}
    worker_times = []
    t0 = time.perf_counter()
    with Exec(max_workers=len(chunks)) as ex:
        for c_counts, t_run in ex.map(_worker, payloads):
            for k, v in c_counts.items():
                counts[k] = counts.get(k, 0) + v
            worker_times.append(t_run)
    wall = time.perf_counter() - t0

    return SimResult(
        backend=f"{backend.name} x{len(chunks)} ({executor})",
        n_qubits=circuit.n_qubits,
        shots=sum(chunks),
        counts=counts,
        t_build=0.0,
        t_run=wall,
        clifford=circuit.is_clifford,
        extra={
            "workers": len(chunks),
            "executor": executor,
            "worker_t_run_sum": sum(worker_times),
            "worker_t_run_max": max(worker_times) if worker_times else 0.0,
            # serial-equivalent work / wall-clock ~ achieved parallel speedup
            "speedup_vs_serial_est": (sum(worker_times) / wall) if wall > 0 else float("nan"),
        },
    )
