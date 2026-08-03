"""Build a bank of RC and CB circuits from MQT Bench, as standalone OpenQASM 2.0 files.

The bank is a wire format for *another* repo: it reads the .qasm files in, runs them at
each depth, and analyzes the results. Nothing here executes a circuit.

For every (algorithm, qubit count, depth) it writes:

  * **RC** -- the algorithm repeated ``d`` times, ``U^d``, with every two-qubit gate
    independently Pauli-twirled (:mod:`proxysim.rc`). ``--randomizations`` independent
    twirls per depth. The twirl leaves the logical unitary unchanged up to a global
    phase, so all randomizations of a given depth are the same computation with
    differently-conjugated noise.
  * **CB** -- cycle-benchmarking sequences of ``d`` dressed cycles on a Clifford proxy
    entangler (:mod:`proxysim.cb_emit`), for extracting the per-cycle process infidelity
    ``e_F`` that feeds :func:`proxysim.benchmarking.qcap_bound`.

CB circuits depend only on ``(n_qubits, entangler, pair)`` and not on the algorithm, so
they are generated once per width and shared -- ``manifest.json`` records which proxies
each algorithm needs. That is the difference between ~3k CB files and ~50k.

Each CB directory carries a ``meta.json``: a CB QASM file cannot be analyzed without the
propagated Pauli's support and sign. See :func:`proxysim.cb_emit.survival`.

Examples
--------
    # what would be written, without writing it
    python examples/build_qasm_bank.py --dry-run

    # the full default sweep
    python examples/build_qasm_bank.py

    # a quick subset
    python examples/build_qasm_bank.py --algos ghz qft --widths 4 6 --depths 1 2 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
import zlib

warnings.filterwarnings("ignore")

import _bootstrap  # noqa: F401
from proxysim import cb_emit, mqtbank
from proxysim.qasm import circuit_to_qasm
from proxysim.rc import pauli_twirl

DEPTHS = [1, 2, 4, 8, 16, 32, 64, 128]
WIDTHS = [4, 6, 8, 10, 12, 14, 16]
N_RANDOMIZATIONS = 20
CB_PAIR = (0, 1)


def _seed(*parts) -> int:
    """A deterministic seed from the identity of a file, so the bank is reproducible
    and resumable regardless of the order things are generated in."""
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


class Writer:
    """Writes files, or just counts them under ``--dry-run``."""

    def __init__(self, dry_run: bool, force: bool):
        self.dry_run, self.force = dry_run, force
        self.files = self.bytes = self.skipped = 0

    def __call__(self, path, circuit, measured=None, header_lines=()):
        text = circuit_to_qasm(circuit, measured=measured, header_lines=header_lines)
        self.files += 1
        self.bytes += len(text)
        if self.dry_run:
            return
        if os.path.exists(path) and not self.force:
            self.skipped += 1
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)

    def json(self, path, obj):
        if self.dry_run:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=1)


def build_rc(out, algo, n, depths, n_rand, writer, base_seed):
    """RC files for one (algorithm, width): ``U^d`` twirled, for each depth and rand."""
    circ, measured = mqtbank.fetch(algo, n)
    writer(os.path.join(out, "base", algo, f"n{n:02d}.qasm"), circ, measured,
           header_lines=[f"MQT Bench {algo}, n={n}, level=alg (untwirled source)",
                         f"gates={len(circ.gates)} 2q={circ.n_two_qubit} "
                         f"depth={circ.depth} clifford={circ.is_clifford}"])
    for d in depths:
        rep = mqtbank.repeat(circ, d)
        for r in range(n_rand):
            rng = random.Random(_seed(base_seed, "rc", algo, n, d, r))
            twirled = pauli_twirl(rep, rng)
            path = os.path.join(out, "rc", algo, f"n{n:02d}", f"d{d:03d}", f"r{r:02d}.qasm")
            writer(path, twirled, measured, header_lines=[
                f"RC | {algo} n={n} depth={d} (U^{d}) randomization={r}",
                f"MQT Bench level=alg | Pauli-twirled per 2q gate; logical unitary "
                f"unchanged up to global phase",
                f"gates={len(twirled.gates)} 2q={twirled.n_two_qubit} "
                f"measured={measured}",
            ])


def build_cb(out, n, proxies, depths, n_rand, writer, base_seed):
    """CB files for one width, shared across every algorithm that needs the proxy."""
    for proxy in proxies:
        twoq = proxy.split("_")[0]
        for d in depths:
            metas = []
            for r in range(n_rand):
                rng = random.Random(_seed(base_seed, "cb", n, proxy, d, r))
                circ, meta = cb_emit.cb_circuit(n, [CB_PAIR], d, rng, twoq=twoq)
                meta["file"] = f"r{r:02d}.qasm"
                meta["randomization"] = r
                metas.append(meta)
                path = os.path.join(out, "cb", f"n{n:02d}", proxy, f"d{d:03d}",
                                    f"r{r:02d}.qasm")
                writer(path, circ, None, header_lines=[
                    f"CB | n={n} cycle={proxy} depth={d} randomization={r}",
                    f"dressed cycle = [random 1q Clifford layer on all {n} qubits] "
                    f"+ [{twoq} on {list(CB_PAIR)}], repeated {d}x",
                    f"prep_pauli={meta['prep_pauli']} meas_pauli={meta['meas_pauli']} "
                    f"sign={meta['sign']:+d} support={meta['support']}",
                    "survival = sign * mean(1 - 2*parity(bits[support])); "
                    "noiseless value is exactly +1",
                ])
            writer.json(os.path.join(out, "cb", f"n{n:02d}", proxy, f"d{d:03d}",
                                     "meta.json"),
                        {"n_qubits": n, "cycle": proxy, "twoq": twoq,
                         "pair": list(CB_PAIR), "depth": d, "randomizations": metas})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algos", nargs="+", default=mqtbank.ALGORITHMS)
    ap.add_argument("--widths", nargs="+", type=int, default=WIDTHS)
    ap.add_argument("--depths", nargs="+", type=int, default=DEPTHS)
    ap.add_argument("--randomizations", type=int, default=N_RANDOMIZATIONS)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qasm_bank"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="report file count and size without writing anything")
    ap.add_argument("--force", action="store_true",
                    help="rewrite files that already exist (default: skip)")
    args = ap.parse_args()

    writer = Writer(args.dry_run, args.force)
    manifest = {
        "generator": "examples/build_qasm_bank.py",
        "source": "MQT Bench, BenchmarkLevel.ALG",
        "mqt_bench_version": _mqt_version(),
        "qasm_version": "2.0",
        "depths": args.depths,
        "widths": args.widths,
        "n_randomizations": args.randomizations,
        "seed": args.seed,
        "cb_pair": list(CB_PAIR),
        "rc_semantics": "U^d, every 2q gate independently Pauli-twirled",
        "cb_semantics": "d repetitions of a dressed Clifford-proxy cycle",
        "paths": {
            "base": "base/{algorithm}/n{n:02d}.qasm",
            "rc": "rc/{algorithm}/n{n:02d}/d{depth:03d}/r{r:02d}.qasm",
            "cb": "cb/n{n:02d}/{cycle}/d{depth:03d}/r{r:02d}.qasm",
            "cb_meta": "cb/n{n:02d}/{cycle}/d{depth:03d}/meta.json",
        },
        "algorithms": {},
        "cb_cycles": {},
    }

    print(f"{'algorithm':>14}{'n':>4}{'gates':>7}{'2q':>6}{'depth':>7}"
          f"{'cycles':>8}  proxies")
    proxies_by_width = {}
    for algo in args.algos:
        for n in args.widths:
            try:
                info = mqtbank.summarize(algo, n)
            except Exception as exc:
                print(f"{algo:>14}{n:>4}   SKIPPED: {type(exc).__name__}: {exc}")
                continue
            manifest["algorithms"].setdefault(algo, {})[str(n)] = info
            proxies_by_width.setdefault(n, set()).update(info["cb_proxies"])
            build_rc(args.out, algo, n, args.depths, args.randomizations,
                     writer, args.seed)
            print(f"{algo:>14}{n:>4}{info['gates']:>7}{info['two_qubit_gates']:>6}"
                  f"{info['depth']:>7}{info['n_distinct_cycles']:>8}  "
                  f"{','.join(info['cb_proxies'])}", flush=True)

    rc_files = writer.files
    for n in sorted(proxies_by_width):
        proxies = sorted(proxies_by_width[n])
        manifest["cb_cycles"][str(n)] = proxies
        build_cb(args.out, n, proxies, args.depths, args.randomizations,
                 writer, args.seed)
    print(f"\nCB cycles per width: "
          f"{ {n: sorted(p) for n, p in sorted(proxies_by_width.items())} }")

    writer.json(os.path.join(args.out, "manifest.json"), manifest)

    verb = "would write" if args.dry_run else "bank contains"
    print(f"\n{verb} {writer.files} files ({rc_files} RC/base, "
          f"{writer.files - rc_files} CB), {writer.bytes / 1e6:.1f} MB total")
    if writer.skipped:
        print(f"  {writer.files - writer.skipped} written this run; {writer.skipped} "
              f"already existed and were left alone (--force to rewrite)")
    if not args.dry_run:
        print(f"  bank at {args.out}, index at {args.out}/manifest.json")


def _mqt_version():
    try:
        from importlib.metadata import version
        return version("mqt.bench")
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
