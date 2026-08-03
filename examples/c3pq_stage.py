"""Turn the QASM bank into C-3PQ build units: lower, batch, name, and emit the harness.

Two subcommands, run either side of code generation:

    c3pq_stage.py stage    ...   bank -> staging/<batch>/qasm/*.qasm + jobs.json
    (c3pq_build.sh runs C-3PQ codegen on each staged file)
    c3pq_stage.py harness  ...   rename symbols, write staging/<batch>/main.cpp

Everything C-3PQ-specific that is *not* a shell command lives here rather than in the
build script, because all of it is a correctness constraint rather than a convenience.

WHY BATCH AT ALL
----------------
C-3PQ generates C++ and compiles it, so its cost is codegen plus compile *per circuit* and
is almost independent of qubit count. A trajectory study is thousands of circuits, and one
binary per circuit means one process launch, one MPI init and one link per circuit. So
circuits are batched: many staged circuits, one binary, one launch.

The obstacle is that C-3PQ names its symbols after level and cluster indices --
``apply_l0_c0``, ``data_packing_l1_c0_c1`` -- never after the circuit, so every circuit it
generates exports the same names and no two can be linked together. `c3pq.prefix_symbols`
gives each circuit a private namespace; this script assigns the prefixes and records them.

A group is never split across batches. Groups are the accumulation unit -- the harness
weight-sums a group's K trajectories internally, so they must all be in the same binary --
and that is also what keeps the output one vector per group instead of one per trajectory.

BATCH SIZING
------------
Two independent caps, because two different things run out:

* ``--batch`` bounds circuits per binary, which bounds compile time and link size.
* ``--max-accum-mb`` bounds ``n_groups * 2^n * 8`` bytes, the accumulators the harness
  holds live. At n=2 this never binds; at n=24 one group's accumulator is 128 MB and it
  is the only thing that matters.

Above ``--probs-max-qubits`` a ``probs`` group is demoted to ``topk``: a full probability
vector per group stops being writable long before the simulation stops being runnable, and
`proxysim.metrics.tvd_to_ideal_support` already knows how to work from a truncated one.
The demotion is recorded in ``jobs.json`` so the analyzer never has to infer it.

Examples
--------
    python examples/c3pq_stage.py stage --dry-run
    python examples/c3pq_stage.py stage --widths 2 --batch 64
    python examples/c3pq_stage.py harness --staging staging --batch b0000_n02_probs
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import _bootstrap  # noqa: F401
from proxysim import c3pq
from proxysim.circuit import circuit_from_qasm

DEFAULT_BANK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qasm_bank_bounding")
DEFAULT_STAGING = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "staging")

BATCH = 64                # circuits per binary
MAX_ACCUM_MB = 2048       # live accumulator bytes per binary
# Above this a probs group is written as topk instead -- which costs real information:
# a truncated distribution gives `c3pq_analyze.tvd_from_topk` a LOWER BOUND on the TVD
# and not the TVD. So this is set where a single 2^n float64 vector stops being sane to
# hold and write (2^26 * 8 B = 537 MB), not lower: batching is already bounded separately
# by --max-accum-mb, which splits a width across binaries instead of degrading it.
PROBS_MAX_QUBITS = 26
TOPK = 1024


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------
def estimator_for(group: dict, probs_max_qubits: int) -> str:
    """The harness mode this group's estimator maps to at this width."""
    est = group["estimator"]
    if est == "probs" and group["n_qubits"] > probs_max_qubits:
        return "topk"
    return est


def plan_batches(groups, batch: int, max_accum_mb: int, probs_max_qubits: int):
    """Pack groups into batches, one batch per (width, harness mode).

    Greedy and order-preserving: a batch is sealed as soon as either cap would be
    exceeded. Nothing here tries to balance batch sizes -- the caps are the point, and an
    unbalanced last batch costs one short compile.
    """
    open_batches, out = {}, []
    n_opened = 0                 # monotonic: sealing a batch must not free its name

    def seal(key):
        b = open_batches.pop(key, None)
        if b:
            out.append(b)

    for g in groups:
        n, mode = g["n_qubits"], estimator_for(g, probs_max_qubits)
        key = (n, mode)
        b = open_batches.get(key)
        per_group_mb = (8 * (1 << n) / 1e6) if mode in ("probs", "topk") else 0.0
        if b is not None and (len(b["_circuits"]) + len(g["files"]) > batch
                              or (len(b["groups"]) + 1) * per_group_mb > max_accum_mb):
            seal(key)
            b = None
        if b is None:
            b = {"name": c3pq.stage_name(f"b{n_opened:04d}", f"n{n:02d}", mode),
                 "n_qubits": n, "mode": mode, "topk": TOPK,
                 "groups": [], "entries": [], "_circuits": []}
            open_batches[key] = b
            n_opened += 1
        gid = len(b["groups"])
        rec = {"name": g["name"], "kind": g["kind"], "arm": g["arm"],
               "estimator": g["estimator"], "demoted": mode != g["estimator"]}
        if mode == "parity":
            rec["sign"], rec["support"] = g["sign"], g["support"]
        b["groups"].append(rec)
        for f in g["files"]:
            b["entries"].append({"stem": f["stem"], "group": gid,
                                 "weight": f["weight"], "name": f["stem"],
                                 "source": os.path.join(g["dir"], f["file"])})
            b["_circuits"].append(f["stem"])
    for key in list(open_batches):
        seal(key)
    for b in out:
        b.pop("_circuits")
        for i, e in enumerate(b["entries"]):
            e["prefix"] = f"s{i:05d}"
            e["symbol"] = c3pq.entry_symbol(e["prefix"])
            e["qasm"] = os.path.join("qasm", f"{e['stem']}.qasm")
    return out


def cmd_stage(args) -> int:
    manifest = json.load(open(os.path.join(args.bank, "manifest.json")))
    groups = [g for g in manifest["groups"]
              if (not args.widths or g["n_qubits"] in args.widths)
              and (not args.kinds or g["kind"] in args.kinds)]
    if not groups:
        print("nothing selected", file=sys.stderr)
        return 1

    batches = plan_batches(groups, args.batch, args.max_accum_mb,
                           args.probs_max_qubits)
    print(f"{len(groups)} groups -> {len(batches)} batches")
    print(f"{'batch':>22}{'n':>4}{'mode':>8}{'groups':>8}{'circuits':>10}{'accum MB':>10}")
    n_files = 0
    for b in batches:
        acc = (0.0 if b["mode"] == "parity"
               else len(b["groups"]) * 8 * (1 << b["n_qubits"]) / 1e6)
        print(f"{b['name']:>22}{b['n_qubits']:>4}{b['mode']:>8}{len(b['groups']):>8}"
              f"{len(b['entries']):>10}{acc:>10.1f}")
        if args.dry_run:
            n_files += len(b["entries"])
            continue
        qdir = os.path.join(args.staging, b["name"], "qasm")
        os.makedirs(qdir, exist_ok=True)
        for e in b["entries"]:
            circ, _ = circuit_from_qasm(open(os.path.join(args.bank, e["source"])).read())
            if circ.n_qubits != b["n_qubits"]:
                raise ValueError(f"{e['source']} is {circ.n_qubits} qubits, batch is "
                                 f"{b['n_qubits']}")
            with open(os.path.join(qdir, f"{e['stem']}.qasm"), "w") as fh:
                fh.write(c3pq.c3pq_qasm(circ, header_lines=[
                    f"lowered for C-3PQ from {e['source']}",
                    f"batch={b['name']} prefix={e['prefix']} group={e['group']}"]))
            n_files += 1

    jobs = {"generator": "examples/c3pq_stage.py",
            "bank": os.path.abspath(args.bank),
            "staging": os.path.abspath(args.staging),
            "codegen": {"architecture": args.architecture, "threads": args.threads,
                        "ranks": args.ranks},
            "runs": c3pq.RUNS,
            "batches": batches}
    if not args.dry_run:
        os.makedirs(args.staging, exist_ok=True)
        with open(os.path.join(args.staging, "jobs.json"), "w") as fh:
            json.dump(jobs, fh, indent=1)
        print(f"\nstaged {n_files} QASM files under {args.staging}, "
              f"index at {args.staging}/jobs.json")
    else:
        print(f"\nwould stage {n_files} QASM files")
    return 0


# ---------------------------------------------------------------------------
# harness -- runs after codegen
# ---------------------------------------------------------------------------
def cmd_harness(args) -> int:
    jobs = json.load(open(os.path.join(args.staging, "jobs.json")))
    batches = [b for b in jobs["batches"] if not args.batch or b["name"] in args.batch]
    if not batches:
        print("no matching batch", file=sys.stderr)
        return 1

    for b in batches:
        root = os.path.join(args.staging, b["name"])
        gen = os.path.join(root, "gen")
        headers, missing = [], []
        for e in b["entries"]:
            hdr = os.path.join(gen, f"{e['stem']}_m.hpp")
            if not os.path.exists(hdr):
                missing.append(e["stem"])
                continue
            # Renaming is idempotent only in the sense that a second run finds nothing
            # left to rewrite, so guard on the marker rather than on file mtimes.
            if e["symbol"] not in open(hdr).read():
                c3pq.prefix_symbols(gen, e["stem"], e["prefix"])
            headers.append(f"{e['stem']}_m.hpp")
        if missing:
            raise SystemExit(f"{b['name']}: codegen produced nothing for "
                             f"{len(missing)} circuits, first {missing[:3]}")
        src = c3pq.emit_batch_harness(
            os.path.join(root, "main.cpp"), b["name"], b["n_qubits"], headers,
            b["entries"], b["groups"], mode=b["mode"], topk=b.get("topk", TOPK))
        srcs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(gen, "*.cpp")))
        with open(os.path.join(root, "sources.txt"), "w") as fh:
            fh.write("\n".join(srcs) + "\n")
        print(f"{b['name']}: {len(b['entries'])} circuits, {len(srcs)} generated .cpp, "
              f"main.cpp {len(src)} bytes, mode={b['mode']}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stage", help="lower the bank, plan batches, write jobs.json")
    s.add_argument("--bank", default=DEFAULT_BANK)
    s.add_argument("--staging", default=DEFAULT_STAGING)
    s.add_argument("--widths", nargs="*", type=int, default=[])
    s.add_argument("--kinds", nargs="*", default=[],
                   help="restrict to tvd / cb / cal (default: all)")
    s.add_argument("--batch", type=int, default=BATCH,
                   help=f"max circuits per binary (default {BATCH})")
    s.add_argument("--max-accum-mb", type=int, default=MAX_ACCUM_MB,
                   help=f"max live accumulator MB per binary (default {MAX_ACCUM_MB})")
    s.add_argument("--probs-max-qubits", type=int, default=PROBS_MAX_QUBITS,
                   help=f"above this width a probs group becomes topk "
                        f"(default {PROBS_MAX_QUBITS})")
    s.add_argument("--architecture", default="cpu", choices=("cpu", "gpu"))
    s.add_argument("--threads", type=int, default=8,
                   help="OpenMP threads C-3PQ generates code for")
    s.add_argument("--ranks", type=int, default=1,
                   help="MPI ranks. The batch harness asserts 1: with "
                        "--distributed_memory == --qubits there is a single partition "
                        "and no Alltoall, which is what makes a per-circuit reset to "
                        "|0...0> well defined")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_stage)

    h = sub.add_parser("harness", help="rename symbols and emit main.cpp (post-codegen)")
    h.add_argument("--staging", default=DEFAULT_STAGING)
    h.add_argument("--batch", nargs="*", default=[],
                   help="batch names (default: all in jobs.json)")
    h.set_defaults(func=cmd_harness)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
