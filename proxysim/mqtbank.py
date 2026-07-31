"""MQT Bench adapter: fetch algorithmic benchmark circuits into the proxysim IR.

MQT Bench 2.x's algorithmic level (``BenchmarkLevel.ALG``) is the abstract, pre-hardware
form of each benchmark -- ``h``/``cp``/``swap`` for a QFT rather than a device's native
gate set. That is the level this module targets, so the emitted bank stays independent
of any particular backend's basis or connectivity.

One wrinkle makes a transpile pass mandatory rather than cosmetic: at ALG level MQT
emits *opaque composite* instructions (a single ``qft`` block, a ``gate_oracle`` for
Deutsch-Jozsa), which ``circuit_from_qasm`` rejects outright. Running qiskit's transpiler
at ``optimization_level=0`` against an explicit abstract basis flattens those blocks
without otherwise rewriting the circuit -- no routing, no gate-set change beyond
unrolling. Skipping it makes ``qft``, ``qftentangled`` and ``dj`` fail to parse.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple

from .cb_emit import PROXY_OF, proxy_name
from .circuit import Circuit, circuit_from_qasm

#: Abstract basis the composite blocks are unrolled into. Every entry is a gate the IR
#: understands (see ``circuit._QASM_GATE``); ``cp`` and ``swap`` are kept so the QFT
#: keeps its algorithmic shape instead of being decomposed into a hardware basis.
BASIS = ["id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "t", "tdg",
         "rx", "ry", "rz", "p", "cx", "cz", "swap", "cp"]

#: The six algorithms the bank covers. All are verified to build for n = 4..16.
ALGORITHMS = ["qft", "qftentangled", "ghz", "graphstate", "dj", "wstate"]

_CACHE: Dict[Tuple[str, int], Tuple[Circuit, List[int]]] = {}


def fetch(name: str, n: int) -> Tuple[Circuit, List[int]]:
    """``(circuit, measured_qubits)`` for one MQT Bench algorithm at width ``n``.

    ``measured_qubits`` is not always ``range(n)`` -- ``dj`` leaves its ancilla
    unmeasured -- so it is threaded through to the QASM writer rather than assumed.
    """
    key = (name, n)
    if key in _CACHE:
        return _CACHE[key]

    from mqt.bench import BenchmarkLevel, get_benchmark
    from qiskit import qasm2, transpile

    qc = get_benchmark(name, BenchmarkLevel.ALG, n)
    flat = transpile(qc, basis_gates=BASIS, optimization_level=0)
    circ, measured = circuit_from_qasm(qasm2.dumps(flat))
    circ.name = f"{name}_n{n}"
    _CACHE[key] = (circ, measured)
    return circ, measured


def repeat(circ: Circuit, d: int) -> Circuit:
    """``U^d`` -- the circuit's gate list concatenated ``d`` times.

    The IR carries no measurements (``circuit_from_qasm`` strips them out and reports
    them separately), so repetition is unitary and the single measurement layer is
    appended once, at write time.
    """
    out = Circuit(circ.n_qubits, name=f"{circ.name}_d{d}")
    out.gates = list(circ.gates) * d
    return out


def two_qubit_families(circ: Circuit) -> List[str]:
    """Sorted distinct two-qubit gate names appearing in ``circ``."""
    return sorted({g.name for g in circ.gates if len(g.qubits) == 2})


def proxies_for(circ: Circuit, pair: Tuple[int, int] = (0, 1)) -> List[str]:
    """The Clifford proxy cycles needed to benchmark ``circ``'s entangling gates.

    Deduplicated, because several algorithmic gates map to the same proxy: ``cp(theta)``
    and ``cz`` are both benchmarked with a CZ carrier (see :mod:`proxysim.cb_emit`).
    """
    seen = {PROXY_OF[fam] for fam in two_qubit_families(circ)}
    return sorted(proxy_name(t, pair) for t in seen)


def cycle_census(circ: Circuit) -> List[dict]:
    """Distinct entangling ASAP layers of ``circ``, with their multiplicities.

    CB is only run on a handful of proxy cycles, but the QCAP bound is a product over
    *every* cycle the circuit actually executes,
    ``1 - F_RO * prod_c (1 - e_F_c)^{n_c}`` (see
    :func:`proxysim.benchmarking.qcap_bound`). Recording the full census in the manifest
    is what lets a consumer form that product from a proxy ``e_F``: the counts are here
    even though the benchmarks are shared.

    Each entry is ``{"pairs", "count", "families"}``, ordered by descending count.
    """
    counts: Counter = Counter()
    families: Dict[frozenset, set] = {}
    for layer in circ.layers():
        twoq = [g for g in layer if len(g.qubits) == 2]
        if not twoq:
            continue
        key = frozenset(g.qubits for g in twoq)
        counts[key] += 1
        families.setdefault(key, set()).update(g.name for g in twoq)

    out = [{"pairs": sorted([list(p) for p in key]),
            "count": c,
            "families": sorted(families[key])}
           for key, c in counts.most_common()]
    return out


def summarize(name: str, n: int) -> dict:
    """Everything the manifest records about one (algorithm, width)."""
    circ, measured = fetch(name, n)
    census = cycle_census(circ)
    return {
        "algorithm": name,
        "n_qubits": circ.n_qubits,
        "measured_qubits": measured,
        "gates": len(circ.gates),
        "two_qubit_gates": circ.n_two_qubit,
        "depth": circ.depth,
        "is_clifford": circ.is_clifford,
        "gate_counts": dict(Counter(g.name for g in circ.gates)),
        "two_qubit_families": two_qubit_families(circ),
        "cb_proxies": proxies_for(circ),
        "n_distinct_cycles": len(census),
        "cycle_census": census,
    }
