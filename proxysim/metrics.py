"""Distribution metrics and result I/O.

One home for every "compare two distributions" and "save/load results" helper, so
nothing gets redefined per example script. Pure functions (no backends, no noise);
the simulation that produces the distributions lives in proxysim.simulate.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Distribution metrics (inputs are {bitstring: probability} dicts)
# ---------------------------------------------------------------------------
def total_variation_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    """TVD = 1/2 sum_x |p(x) - q(x)|.  0 = identical, 1 = disjoint support."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def classical_fidelity(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Hellinger / classical fidelity  ( sum_x sqrt(p q) )^2  in [0, 1]."""
    keys = set(p) | set(q)
    return sum(math.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in keys) ** 2


hellinger_fidelity = classical_fidelity


def shannon_entropy(dist: Dict[str, float]) -> float:
    """Shannon entropy of a distribution, in bits."""
    return -sum(pr * math.log2(pr) for pr in dist.values() if pr > 0)


def tvd_to_ideal_support(counts: Dict, shots: int, ideal_support: Iterable,
                         p_ideal: float | None = None) -> float:
    """TVD(sampled, ideal) using only the ideal's (small) support plus total leakage.

    Resolvable even when the noisy distribution spreads over 2^n outcomes: we only
    need the sampled counts on the known ideal outcomes and the leftover mass.

    ``counts`` maps an outcome key (bitstring or packed int) to a shot count;
    ``ideal_support`` lists the ideal outcomes in the same key type (assumed
    uniform on the support unless ``p_ideal`` is given).
    """
    supp = list(ideal_support)
    if p_ideal is None:
        p_ideal = 1.0 / len(supp)
    in_supp = np.array([counts.get(x, 0) for x in supp], dtype=float) / shots
    leak = 1.0 - in_supp.sum()
    return 0.5 * (float(np.abs(in_supp - p_ideal).sum()) + leak)


# ---------------------------------------------------------------------------
# Result persistence -- ALWAYS save what you generate, then plot from disk.
# ---------------------------------------------------------------------------
def save_results(path: str, **arrays) -> str:
    """Save named arrays/scalars to a .npz so figures can be re-rendered without
    re-running the simulation."""
    if not path.endswith(".npz"):
        path += ".npz"
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    return path


def load_results(path: str) -> Dict[str, np.ndarray]:
    """Load a .npz saved by :func:`save_results` into a plain dict."""
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}
