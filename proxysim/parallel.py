"""Single-node parallelism for HPC runs.

The workhorse is :func:`pmap` -- a parallel ``map`` over items (noise trajectories,
circuit instances, parameter points) with one crucial detail for HPC: each worker
pins its BLAS/OpenMP thread count (default 1) so ``n_workers`` processes don't each
spawn a full thread pool and thrash the cores. That oversubscription is why naive
process parallelism looked *slower* before; with ``threads_per_worker=1`` it scales.

Rules of thumb on a single node with C cores:
  * statevector / tensor-network trajectories are CPU-bound and embarrassingly
    parallel  ->  n_workers = C, threads_per_worker = 1.
  * one big contraction that already uses threaded BLAS  ->  n_workers small,
    threads_per_worker large (let BLAS use the cores).
  * GPU tensor networks are single-device: use ONE worker with a GPU backend
    (proxysim.backends.gpu), not many CPU processes.

The mapped function must be importable (top-level, not a lambda/closure) because
process workers pickle it by reference -- so ``pip install -e .`` the package.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Iterable, List, Optional

_TP_LIMITER = None  # keep the threadpoolctl limiter alive for the worker's lifetime


def _init_worker(threads: int):
    global _TP_LIMITER
    try:
        import threadpoolctl

        _TP_LIMITER = threadpoolctl.threadpool_limits(threads)
    except Exception:  # threadpoolctl missing -> fall back to env vars (spawn only)
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v] = str(threads)


def pmap(func: Callable, items: Iterable, n_workers: Optional[int] = None,
         executor: str = "process", threads_per_worker: int = 1,
         chunksize: int = 1) -> List:
    """Parallel ``map``: return ``[func(x) for x in items]``, computed in parallel.

    executor : 'process' (CPU-bound, needs the package importable), 'thread'
               (GIL-bound but no pickling), or 'serial' (a plain loop).
    """
    items = list(items)
    n_workers = n_workers or os.cpu_count() or 1
    if executor == "serial" or n_workers <= 1 or len(items) <= 1:
        return [func(x) for x in items]

    if executor == "process":
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                 initargs=(threads_per_worker,)) as ex:
            return list(ex.map(func, items, chunksize=chunksize))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(func, items, chunksize=chunksize))


def parallel_trajectories(worker: Callable[[int], object], n_traj: int,
                          n_workers: Optional[int] = None, base_seed: int = 0,
                          executor: str = "process", threads_per_worker: int = 1):
    """Run ``worker(seed)`` for ``n_traj`` distinct seeds in parallel and return the
    list of results (aggregate them yourself). ``worker`` must be top-level."""
    seeds = [base_seed + i + 1 for i in range(n_traj)]
    return pmap(worker, seeds, n_workers=n_workers, executor=executor,
                threads_per_worker=threads_per_worker)


# ---------------------------------------------------------------------------
# Back-compat: split a backend's shots across workers and merge the counts.
# ---------------------------------------------------------------------------
def _shot_worker(payload):
    backend_cls, init_kwargs, circuit, shots, seed = payload
    r = backend_cls(**init_kwargs).run(circuit, shots=shots, seed=seed)
    return r.counts


def sample_parallel(backend, circuit, shots: int, n_workers: Optional[int] = None,
                    base_seed: int = 0, executor: str = "process"):
    """Split ``shots`` across workers and merge the bitstring counts (rebuilds the
    backend in each worker from its class + init_kwargs)."""
    n_workers = n_workers or os.cpu_count() or 1
    base, rem = divmod(shots, n_workers)
    chunks = [base + (1 if i < rem else 0) for i in range(n_workers)]
    payloads = [(type(backend), getattr(backend, "init_kwargs", {}), circuit, c,
                 base_seed + i + 1) for i, c in enumerate(chunks) if c > 0]
    merged: dict = {}
    for counts in pmap(_shot_worker, payloads, n_workers=n_workers, executor=executor):
        for k, v in (counts or {}).items():
            merged[k] = merged.get(k, 0) + v
    return merged
