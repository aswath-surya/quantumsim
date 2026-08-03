"""Bridge from the proxysim IR to C-3PQ, the code-generating state-vector simulator.

C-3PQ (``~/Documents/Code/quantum``) does not interpret a circuit -- it *compiles* one,
emitting bespoke C++ in which every gate is its own inlined loop nest over the state
vector. That makes it fast and makes its cost per circuit a codegen-plus-compile cost
that is almost independent of qubit count, which is the trade this pipeline is built
around: a bank of many small-to-medium circuits is dominated by compilation, not by
simulation, so the pipeline batches many circuits into one binary.

Nothing here modifies the C-3PQ tree. Everything below is either a translation applied
on our side or a post-processing pass over the C++ it generates, which is why the module
reads as a list of workarounds. Each one was reproduced before it was worked around.

What C-3PQ actually accepts
---------------------------
1. **``s``/``sdg``/``t``/``tdg`` crash the parser.** ``gates.py`` names the patterns for
   these backwards (``s_pattern`` matches ``sdg``) and both branches index
   ``line_split[3]`` on a line that splits into two tokens, so ``sdg q[2];`` raises
   ``IndexError``; past the crash the rewrite produces ``pdg(pi/2)``, matching no
   pattern, and the gate would be dropped silently. :func:`lower_for_c3pq` rewrites all
   four into ``rz``, which differs only by a global phase and cannot change a
   probability. ``sx``/``sxdg`` share one pattern and are lowered to ``rx`` for the same
   reason.
2. **A circuit with no ``measure`` statements never finishes generating.**
   ``partitioner.partitionCircuit`` loops on ``while current != end_nodes``, and
   ``end_nodes`` is populated only by ``measure`` lines, so a unitary-only file spins
   forever appending partitions until the machine runs out of memory. :func:`c3pq_qasm`
   always emits a full measurement round, in qubit order (the comparison is elementwise
   against a list indexed by declaration order, so the order matters too).
3. **``runs=1`` is a division by zero.** The generated ``apply_l0_c0`` weights each timing
   sample by ``(i / (runs / 2))`` in integer arithmetic. :data:`RUNS` is 2. ``runs`` is
   purely a timing repetition -- every iteration re-``memcpy``s the input -- so the
   physics is unaffected.
4. **Every circuit generates the same symbol names.** ``apply_l0_c0``,
   ``apply_l1_c0x0``, ``data_packing_l1_c0_c1`` are named from level and cluster indices,
   never from the circuit, and the per-gate helpers are ``inline`` with external linkage.
   Two circuits therefore cannot be linked into one binary. :func:`prefix_symbols`
   rewrites them, which is what makes batching possible at all.
5. **The stock makefile drops any ``.cpp`` whose name contains ``_m`` or ``_l``** (it uses
   that to separate generated module sources from mains). A circuit staged under a name
   containing either substring silently builds nothing. :func:`stage_name` refuses to
   produce such a name. This pipeline compiles directly rather than through that
   makefile, but the constraint is free to respect and keeps the makefile usable.
6. **The stock harness is a timing harness.** ``executor.generateHarness`` initialises the
   state to all-ones-normalised rather than |0...0> and prints only amplitude 0 and three
   timings. :func:`emit_batch_harness` replaces it.

Index convention
----------------
Verified, not assumed: qubit ``i`` is bit ``i`` of the global state-vector index, the same
little-endian convention proxysim's statevector backend uses. The generated code makes
this visible -- ``h q[0]`` strides by 1, ``ry q[3]`` strides by 8. Since it is documented
nowhere upstream, the parity estimator is emitted for *both* conventions
(:func:`parity_masks`) and :func:`examples.c3pq_analyze` asserts which one holds against a
calibration circuit rather than trusting this paragraph.

Ranks
-----
Single-rank only. Multi-rank runs shuffle the state vector through ``MPI_Alltoall`` and
the global index a rank's slice corresponds to afterwards is not documented; every
estimator here depends on the global index, so the harness aborts if it is launched with
more than one rank. Set ``--distributed_memory`` equal to the qubit count. On one node
that caps out near 28 qubits (2^28 x 16 B = 4 GB, twice over for the two buffers).
"""

from __future__ import annotations

import glob
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .circuit import Circuit, Gate
from .qasm import circuit_to_qasm

# Timing repetitions passed to apply_l0_c0. Must be even and >= 2 (workaround 3).
RUNS = 2

# Gates whose QASM form C-3PQ's pattern table parses correctly, with or without a
# leading control chain. Anything outside this set must be lowered.
C3PQ_GATES = frozenset({"h", "x", "y", "z", "rx", "ry", "rz",
                        "cx", "cy", "cz", "cp", "swap"})

# IR gate -> (replacement name, angle). None means "drop the gate entirely".
_LOWERING: Dict[str, Optional[Tuple[str, float]]] = {
    "s": ("rz", math.pi / 2),
    "sdg": ("rz", -math.pi / 2),
    "t": ("rz", math.pi / 4),
    "tdg": ("rz", -math.pi / 4),
    "sx": ("rx", math.pi / 2),
    "sxdg": ("rx", -math.pi / 2),
    "i": None,
}


# ---------------------------------------------------------------------------
# Lowering
# ---------------------------------------------------------------------------
def lower_for_c3pq(circuit: Circuit) -> Circuit:
    """Rewrite ``circuit`` into the subset of QASM C-3PQ parses correctly.

    Every substitution is exact up to a global phase, so the measurement distribution is
    unchanged bit for bit -- ``s`` and ``rz(pi/2)`` differ by ``e^{-i pi/4}``, ``sx`` and
    ``rx(pi/2)`` by ``e^{i pi/4}``, and identity gates contribute nothing. The
    replacements are all uncontrolled, which is what makes the global phase harmless;
    a controlled ``s`` would *not* be a controlled ``rz`` and is deliberately not handled.
    """
    out = Circuit(circuit.n_qubits, name=circuit.name)
    for g in circuit.gates:
        if g.name in C3PQ_GATES:
            out.gates.append(g)
            continue
        if g.name not in _LOWERING:
            raise ValueError(
                f"lower_for_c3pq: no C-3PQ form for gate '{g.name}'. Extend _LOWERING "
                f"only with a substitution that is exact up to a global phase.")
        repl = _LOWERING[g.name]
        if repl is None:
            continue
        if len(g.qubits) != 1:
            raise ValueError(f"lower_for_c3pq: '{g.name}' lowering assumes one qubit")
        name, theta = repl
        out.gates.append(Gate(name, g.qubits, (theta,)))
    return out


def c3pq_qasm(circuit: Circuit, header_lines: Iterable[str] = ()) -> str:
    """``circuit`` lowered and serialised as QASM C-3PQ can consume.

    The measurement round is not optional -- without it the partitioner never terminates
    (workaround 2) -- and it must cover every qubit in declaration order.
    """
    return circuit_to_qasm(lower_for_c3pq(circuit), measured=None,
                           header_lines=header_lines)


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------
_NAME_OK = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def stage_name(*parts: object, qubits: Optional[int] = None) -> str:
    """Join ``parts`` into a staging stem that is safe for every consumer downstream.

    The generated sources, the object symbols and (optionally) the stock makefile all key
    off this string, so it is validated rather than trusted:

    * no ``_m`` or ``_l`` substring -- the makefile would filter the main out and build
      nothing at all, with no error (workaround 5);
    * no ``apply_l`` / ``data_packing_l`` / ``data_unpacking_l`` substring -- those are
      the tokens :func:`prefix_symbols` rewrites, and a stem containing one would have
      its own ``#include`` lines corrupted;
    * a plain C identifier, since it becomes part of a symbol name.

    Failing loudly here is the whole point: every one of these produces a silent wrong
    answer rather than an error if it slips through.
    """
    stem = "_".join(str(p) for p in parts)
    if qubits is not None:
        stem = f"{stem}_q{int(qubits)}"
    if not _NAME_OK.match(stem):
        raise ValueError(f"stage_name: '{stem}' is not a valid C identifier")
    for bad in ("_m", "_l"):
        if bad in stem:
            raise ValueError(
                f"stage_name: '{stem}' contains '{bad}'; the C-3PQ makefile filters out "
                f"any main whose filename contains it and would build nothing silently")
    for tok in _SYMBOL_TOKENS:
        if tok in stem:
            raise ValueError(f"stage_name: '{stem}' contains the renamed symbol '{tok}'")
    return stem


# ---------------------------------------------------------------------------
# Symbol renaming (what makes batching possible)
# ---------------------------------------------------------------------------
# Renaming the token 'apply_l' also catches 'local_apply_l...' -> 'local_<p>_apply_l...',
# which is ugly but consistent, and consistency is all the linker needs.
_SYMBOL_TOKENS = ("apply_l", "data_packing_l", "data_unpacking_l")


def prefix_symbols(directory: str, stem: str, prefix: str) -> int:
    """Give one circuit's generated sources a private symbol namespace.

    Rewrites the C-3PQ symbol tokens in every ``<stem>*.cpp`` / ``<stem>*.hpp`` under
    ``directory`` to carry ``prefix``, and deletes the stock single-circuit harness
    ``<stem>.cpp`` so it cannot contribute a second ``main``. Returns the number of files
    rewritten. ``#include`` lines name files, not symbols, so they are untouched -- which
    :func:`stage_name` guarantees by rejecting stems containing the tokens.
    """
    if not _NAME_OK.match(prefix):
        raise ValueError(f"prefix_symbols: '{prefix}' is not a valid C identifier")

    harness = os.path.join(directory, f"{stem}.cpp")
    if os.path.exists(harness):
        os.remove(harness)

    touched = 0
    for path in sorted(glob.glob(os.path.join(directory, f"{stem}*.cpp"))
                       + glob.glob(os.path.join(directory, f"{stem}*.hpp"))):
        with open(path) as fh:
            text = fh.read()
        new = text
        for tok in _SYMBOL_TOKENS:
            new = new.replace(tok, f"{prefix}_{tok}")
        if new != text:
            with open(path, "w") as fh:
                fh.write(new)
            touched += 1
    if touched == 0:
        raise RuntimeError(
            f"prefix_symbols: no generated sources for '{stem}' in {directory} -- "
            f"codegen produced nothing, which C-3PQ does not report as an error")
    return touched


def entry_symbol(prefix: str) -> str:
    """The renamed top-level entry point for a circuit staged with ``prefix``."""
    return f"{prefix}_apply_l0_c0"


# ---------------------------------------------------------------------------
# Parity masks (the CB estimator, in global-index bits)
# ---------------------------------------------------------------------------
def parity_masks(support: Sequence[int], n: int) -> Tuple[int, int]:
    """``(mask_lsb, mask_msb)`` selecting ``support`` under either index convention.

    The CB survival estimator is a parity over a subset of qubits, and turning that into
    a mask over state-vector indices needs to know whether qubit ``i`` is bit ``i`` or bit
    ``n-1-i``. Measurement says the former, but nothing upstream documents it, so the
    harness evaluates both and the analyzer keeps whichever the calibration circuit
    confirms. Two extra accumulator updates is a cheap price for not having to trust it.
    """
    lsb = msb = 0
    for q in support:
        if not 0 <= q < n:
            raise ValueError(f"parity_masks: qubit {q} out of range for n={n}")
        lsb |= 1 << q
        msb |= 1 << (n - 1 - q)
    return lsb, msb


# ---------------------------------------------------------------------------
# The batch harness
# ---------------------------------------------------------------------------
MODES = ("probs", "parity", "topk")

_HARNESS = r'''// Generated by proxysim.c3pq.emit_batch_harness -- do not edit.
//
// One binary per batch. Each entry is an independently generated circuit whose symbols
// were given a private prefix by prefix_symbols(); entries sharing a group index are
// accumulated together with their weights, so the K Monte-Carlo trajectories of one
// noise realisation collapse to a single output before anything is written to disk.
#include <algorithm>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <mpi.h>
#include <omp.h>

{includes}

typedef void (*apply_fn)(std::complex<double> *, std::complex<double> *, double *, int);

static const size_t DIM = {dim}ull;
static const int N_QUBITS = {n_qubits};
static const int N_ENTRIES = {n_entries};
static const int N_GROUPS = {n_groups};

static apply_fn const FN[] = {{{fn_table}}};
static const int ENTRY_GROUP[] = {{{entry_group}}};
static const double ENTRY_WEIGHT[] = {{{entry_weight}}};
static const char *const ENTRY_NAME[] = {{{entry_name}}};
static const char *const GROUP_NAME[] = {{{group_name}}};
{group_extra}

int main(int argc, char **argv) {{
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    // Every estimator below indexes the state vector globally. Under more than one rank
    // the layout after MPI_Alltoall is undocumented upstream, so refuse rather than
    // report a plausible wrong number.
    if (world != 1) {{
        if (rank == 0)
            fprintf(stderr, "c3pq batch harness: single-rank only, launched with %d\n",
                    world);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }}

    int threads = (argc > 1) ? atoi(argv[1]) : 1;
    const char *outdir = (argc > 2) ? argv[2] : ".";
    omp_set_dynamic(0);
    omp_set_num_threads(threads > 0 ? threads : 1);

    std::vector<std::complex<double> > sv(DIM), tmp(DIM);
{alloc}

    for (int e = 0; e < N_ENTRIES; ++e) {{
        std::fill(sv.begin(), sv.end(), std::complex<double>(0.0, 0.0));
        std::fill(tmp.begin(), tmp.end(), std::complex<double>(0.0, 0.0));
        sv[0] = std::complex<double>(1.0, 0.0);   // |0...0>, not the stock all-ones state

        double timing[3] = {{0.0, 0.0, 0.0}};
        double t0 = MPI_Wtime();
        FN[e](sv.data(), tmp.data(), timing, {runs});   // result lands back in sv
        double t1 = MPI_Wtime();

        const int g = ENTRY_GROUP[e];
        const double w = ENTRY_WEIGHT[e];
{accumulate}
        printf("entry\t%s\t%s\t%.6f\n", ENTRY_NAME[e], GROUP_NAME[g], t1 - t0);
        fflush(stdout);
    }}

{write_out}
    MPI_Finalize();
    return 0;
}}
'''

_ALLOC_VECTOR = """    std::vector<std::vector<double> > acc(
        N_GROUPS, std::vector<double>(DIM, 0.0));
    std::vector<double> norm(N_GROUPS, 0.0);"""

_ALLOC_PARITY = """    std::vector<double> par_lsb(N_GROUPS, 0.0), par_msb(N_GROUPS, 0.0);
    std::vector<double> norm(N_GROUPS, 0.0);"""

_ACC_VECTOR = """        {
            double *a = acc[g].data();
            double s = 0.0;
#pragma omp parallel for reduction(+ : s) schedule(static)
            for (size_t i = 0; i < DIM; ++i) {
                double p = std::norm(sv[i]);
                a[i] += w * p;
                s += p;
            }
            norm[g] += w * s;
        }"""

_ACC_PARITY = """        {
            const unsigned long long m0 = MASK_LSB[g], m1 = MASK_MSB[g];
            double s0 = 0.0, s1 = 0.0, s = 0.0;
#pragma omp parallel for reduction(+ : s0, s1, s) schedule(static)
            for (size_t i = 0; i < DIM; ++i) {
                double p = std::norm(sv[i]);
                s += p;
                s0 += (__builtin_parityll(i & m0) ? -p : p);
                s1 += (__builtin_parityll(i & m1) ? -p : p);
            }
            par_lsb[g] += w * s0;
            par_msb[g] += w * s1;
            norm[g] += w * s;
        }"""

_WRITE_PARITY = """    {
        std::string path = std::string(outdir) + "/" + "__BATCH__" + ".parity.tsv";
        FILE *f = fopen(path.c_str(), "w");
        if (!f) { fprintf(stderr, "cannot write %s\\n", path.c_str()); MPI_Abort(MPI_COMM_WORLD, 3); }
        fprintf(f, "group\\tsign\\tparity_lsb\\tparity_msb\\tnorm\\n");
        for (int g = 0; g < N_GROUPS; ++g)
            fprintf(f, "%s\\t%d\\t%.17g\\t%.17g\\t%.17g\\n", GROUP_NAME[g], SIGN[g],
                    SIGN[g] * par_lsb[g], SIGN[g] * par_msb[g], norm[g]);
        fclose(f);
    }"""

_WRITE_PROBS = """    for (int g = 0; g < N_GROUPS; ++g) {
        std::string path = std::string(outdir) + "/" + GROUP_NAME[g] + ".f64";
        FILE *f = fopen(path.c_str(), "wb");
        if (!f) { fprintf(stderr, "cannot write %s\\n", path.c_str()); MPI_Abort(MPI_COMM_WORLD, 3); }
        fwrite(acc[g].data(), sizeof(double), DIM, f);
        fclose(f);
    }
    {
        std::string path = std::string(outdir) + "/" + "__BATCH__" + ".norm.tsv";
        FILE *f = fopen(path.c_str(), "w");
        fprintf(f, "group\\tnorm\\n");
        for (int g = 0; g < N_GROUPS; ++g)
            fprintf(f, "%s\\t%.17g\\n", GROUP_NAME[g], norm[g]);
        fclose(f);
    }"""

_WRITE_TOPK = """    {
        std::string path = std::string(outdir) + "/" + "__BATCH__" + ".topk.tsv";
        FILE *f = fopen(path.c_str(), "w");
        if (!f) { fprintf(stderr, "cannot write %s\\n", path.c_str()); MPI_Abort(MPI_COMM_WORLD, 3); }
        fprintf(f, "group\\tindex\\tprob\\tnorm\\tretained\\n");
        std::vector<size_t> order(DIM);
        for (int g = 0; g < N_GROUPS; ++g) {
            for (size_t i = 0; i < DIM; ++i) order[i] = i;
            const double *a = acc[g].data();
            size_t k = (TOPK < DIM) ? TOPK : DIM;
            std::partial_sort(order.begin(), order.begin() + k, order.end(),
                              [a](size_t x, size_t y) { return a[x] > a[y]; });
            double retained = 0.0;
            for (size_t i = 0; i < k; ++i) retained += a[order[i]];
            for (size_t i = 0; i < k; ++i)
                fprintf(f, "%s\\t%zu\\t%.17g\\t%.17g\\t%.17g\\n", GROUP_NAME[g],
                        order[i], a[order[i]], norm[g], retained);
        }
        fclose(f);
    }"""


def _c_list(values, fmt=str) -> str:
    return ", ".join(fmt(v) for v in values)


def _c_str_list(values) -> str:
    return ", ".join('"%s"' % str(v).replace('"', '\\"') for v in values)


def emit_batch_harness(path: str, batch: str, n_qubits: int, headers: Sequence[str],
                       entries: Sequence[dict], groups: Sequence[dict],
                       mode: str = "probs", topk: int = 1024) -> str:
    """Write the ``main.cpp`` that runs a whole batch of circuits in one process.

    ``entries`` are dicts with ``symbol``, ``name``, ``group`` (index into ``groups``) and
    ``weight``; ``groups`` are dicts with ``name`` and, for ``mode="parity"``, ``sign``
    and ``support``. Entries in the same group are summed with their weights *inside the
    binary*, which is the whole reason this is worth generating: the K trajectories of one
    arm reduce to one probability vector before any of them reach the filesystem.

    ``mode`` selects the estimator, trading output size against generality:

    * ``parity`` -- one scalar per group, the CB survival, independent of ``n``. Both
      index conventions are emitted (see :func:`parity_masks`).
    * ``probs`` -- the full 2^n float64 probability vector per group.
    * ``topk`` -- the ``topk`` largest probabilities with their indices and the mass they
      retain, for widths where a full vector per group is too much to keep.

    Returns the generated source.
    """
    if mode not in MODES:
        raise ValueError(f"emit_batch_harness: mode must be one of {MODES}, got {mode!r}")
    if not entries:
        raise ValueError("emit_batch_harness: empty batch")
    n_groups = len(groups)
    for e in entries:
        if not 0 <= e["group"] < n_groups:
            raise ValueError(f"emit_batch_harness: entry {e['name']} names group "
                             f"{e['group']} of {n_groups}")

    group_extra = ""
    if mode == "parity":
        lsb, msb, sign = [], [], []
        for g in groups:
            a, b = parity_masks(g["support"], n_qubits)
            lsb.append(a)
            msb.append(b)
            sign.append(int(g.get("sign", 1)))
        group_extra = (
            f"static const unsigned long long MASK_LSB[] = {{{_c_list(lsb)}}};\n"
            f"static const unsigned long long MASK_MSB[] = {{{_c_list(msb)}}};\n"
            f"static const int SIGN[] = {{{_c_list(sign)}}};")
        alloc, acc = _ALLOC_PARITY, _ACC_PARITY
        write_out = _WRITE_PARITY.replace("__BATCH__", batch)
    else:
        alloc, acc = _ALLOC_VECTOR, _ACC_VECTOR
        tmpl = _WRITE_PROBS if mode == "probs" else _WRITE_TOPK
        write_out = tmpl.replace("__BATCH__", batch)
        if mode == "topk":
            group_extra = f"static const size_t TOPK = {int(topk)}ull;"

    src = _HARNESS.format(
        includes="\n".join(f'#include "{h}"' for h in headers),
        dim=1 << n_qubits,
        n_qubits=n_qubits,
        n_entries=len(entries),
        n_groups=n_groups,
        fn_table=_c_list(e["symbol"] for e in entries),
        entry_group=_c_list(e["group"] for e in entries),
        entry_weight=_c_list(("%.17g" % float(e.get("weight", 1.0))) for e in entries),
        entry_name=_c_str_list(e["name"] for e in entries),
        group_name=_c_str_list(g["name"] for g in groups),
        group_extra=group_extra,
        alloc=alloc,
        accumulate=acc,
        write_out=write_out,
        runs=RUNS,
    )
    with open(path, "w") as fh:
        fh.write(src)
    return src


# ---------------------------------------------------------------------------
# Reading what the harness wrote
# ---------------------------------------------------------------------------
def read_probs(path: str, n_qubits: int) -> np.ndarray:
    """One group's probability vector, indexed with qubit ``j`` in bit ``j``.

    That is the same convention :func:`proxysim.noise.apply_readout_to_distribution` and
    the statevector backend use, so the result drops straight into the existing analysis
    with no reindexing -- provided the calibration circuit confirmed it.
    """
    vec = np.fromfile(path, dtype=np.float64)
    if vec.size != 1 << n_qubits:
        raise ValueError(f"read_probs: {path} holds {vec.size} values, expected "
                         f"{1 << n_qubits} for n={n_qubits}")
    return vec


def read_parity(path: str) -> Dict[str, dict]:
    """Parse a ``.parity.tsv`` into ``{group: {survival, survival_msb, norm, sign, ...}}``.

    **The sign is already applied.** The harness multiplies by ``SIGN[g]`` on the way out,
    so the ``parity_lsb`` column *is* the CB survival and multiplying by ``sign`` again
    flips half the sequences to -1 -- which does not look like a bug, it looks like a
    plausible decay curve with a wrong ``e_F``. ``survival`` is provided as the name to
    reach for; ``sign`` is carried through for the record, not to be re-applied.

    Both index conventions are carried through unresolved; picking one is the analyzer's
    job (see :func:`parity_masks`).
    """
    out: Dict[str, dict] = {}
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        want = ["group", "sign", "parity_lsb", "parity_msb", "norm"]
        if header != want:
            raise ValueError(f"read_parity: unexpected header in {path}: {header}")
        for line in fh:
            if not line.strip():
                continue
            group, sign, lsb, msb, norm = line.rstrip("\n").split("\t")
            out[group] = {"sign": int(sign), "survival": float(lsb),
                          "survival_msb": float(msb), "parity_lsb": float(lsb),
                          "parity_msb": float(msb), "norm": float(norm)}
    return out


def read_norms(path: str) -> Dict[str, float]:
    """Parse a ``.norm.tsv`` into ``{group: norm}``, the cheap did-it-run sanity check."""
    out: Dict[str, float] = {}
    with open(path) as fh:
        fh.readline()
        for line in fh:
            if line.strip():
                group, norm = line.rstrip("\n").split("\t")
                out[group] = float(norm)
    return out


def read_topk(path: str) -> Dict[str, dict]:
    """Parse a ``.topk.tsv`` into ``{group: {index, prob, norm, retained}}``."""
    out: Dict[str, dict] = {}
    with open(path) as fh:
        fh.readline()
        for line in fh:
            if not line.strip():
                continue
            group, idx, prob, norm, retained = line.rstrip("\n").split("\t")
            rec = out.setdefault(group, {"index": [], "prob": [],
                                         "norm": float(norm),
                                         "retained": float(retained)})
            rec["index"].append(int(idx))
            rec["prob"].append(float(prob))
    for rec in out.values():
        rec["index"] = np.array(rec["index"], dtype=np.int64)
        rec["prob"] = np.array(rec["prob"], dtype=np.float64)
    return out
