import argparse
import glob
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from proxysim.backends.exatn_backend import ExaTNBackend, SimulationResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_exatn_bank")

@dataclass
class RunManifest:
    """Tracks the state of a bank run."""
    results_dir: str
    qasm_bank: str
    shots: int
    visitor: str
    compiler: Optional[str] = None
    lowered: bool = True
    completed: List[str] = None # List of qasm relative paths
    failed: Dict[str, str] = None # qasm path -> error
    skipped: List[str] = None

    def __post_init__(self):
        self.completed = self.completed or []
        self.failed = self.failed or {}
        self.skipped = self.skipped or []

    def to_json(self):
        return asdict(self)

def load_manifest(path: Path) -> Optional[RunManifest]:
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return RunManifest(**data)
    except Exception as e:
        logger.warning(f"Could not load manifest from {path}: {e}")
        return None

def save_manifest(path: Path, manifest: RunManifest):
    with open(path, "w") as f:
        json.dump(manifest.to_json(), f, indent=2)

def parse_shard(spec: str) -> Tuple[int, int]:
    """``"3/8"`` -> ``(3, 8)``. Raises ValueError with a usable message otherwise."""
    try:
        index, count = (int(part) for part in spec.split("/"))
    except Exception:
        raise ValueError(f"--shard must look like I/N (e.g. 0/8), got {spec!r}")
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"--shard {spec}: need 0 <= I < N and N >= 1")
    return index, count


def merge_shards(out_dir: Path, manifest_name: str) -> int:
    """Fold every ``manifest.shard*of*.json`` in ``out_dir`` into one manifest.

    Shards write separate manifests because a single file cannot be appended to
    concurrently without losing entries. The analyzer reads one manifest, so they have to
    be recombined -- and the recombination is also where a mismatch shows up: if two shards
    disagree on shots, visitor or lowering, their ``.f64`` files are not one population and
    merging them would hide that behind whichever value happened to be written last.
    """
    shards = sorted(out_dir.glob("manifest.shard*of*.json"))
    if not shards:
        logger.error(f"no manifest.shard*of*.json in {out_dir}")
        return 1

    merged: Optional[RunManifest] = None
    settings: Dict[str, Any] = {}
    for path in shards:
        m = load_manifest(path)
        if m is None:
            logger.error(f"could not read {path}")
            return 1
        this = {"shots": m.shots, "visitor": m.visitor, "lowered": m.lowered,
                "compiler": m.compiler, "qasm_bank": m.qasm_bank}
        if merged is None:
            merged, settings = m, this
            merged.results_dir = str(out_dir)
            continue
        differing = {k: (settings[k], v) for k, v in this.items() if settings[k] != v}
        if differing:
            logger.error(
                f"{path.name} disagrees with {shards[0].name} on "
                f"{', '.join(f'{k} {a!r} vs {b!r}' for k, (a, b) in differing.items())}. "
                "These shards did not run the same way; not merging."
            )
            return 1
        merged.completed.extend(m.completed)
        merged.failed.update(m.failed)
        merged.skipped.extend(m.skipped)

    # A file can legitimately appear in two shards only if the shard specs overlapped;
    # dedupe rather than inflate the count, but say so, because it means some work was
    # done twice and the shard specs were wrong.
    before = len(merged.completed)
    merged.completed = sorted(set(merged.completed))
    if len(merged.completed) != before:
        logger.warning(f"{before - len(merged.completed)} files were completed by more "
                       f"than one shard -- check the I/N values used")
    merged.skipped = sorted(set(merged.skipped))

    save_manifest(out_dir / manifest_name, merged)
    print(f"merged {len(shards)} shard manifests -> {out_dir / manifest_name}")
    print(f"  completed {len(merged.completed)}   failed {len(merged.failed)}")
    if merged.failed:
        print(f"  first failures: {list(merged.failed)[:3]}")
    return 1 if merged.failed else 0


def parse_qasm_metadata(filename: str) -> Dict[str, Any]:
    """
    Infers metadata from the C3PQ filename convention:
    tvd_n02_d001_i000_ideal.qasm
    """
    # Pattern: (type)_n(qubits)_d(depth)_i(instance)_(arm).qasm
    pattern = r"([^_]+)_n(\d+)_d(\d+)_i(\d+)_([^_.]+)\.qasm"
    match = re.search(pattern, filename)
    if not match:
        # Fallback for other conventions or simpler names
        return {"filename": filename}
    
    return {
        "type": match.group(1),
        "n_qubits": int(match.group(2)),
        "depth": int(match.group(3)),
        "instance": int(match.group(4)),
        "arm": match.group(5),
        "filename": filename
    }

def write_c3pq_probs(path: Path, counts: Dict[str, int], n_qubits: int):
    """
    Writes a C3PQ-compatible .f64 file.
    A .f64 file is a binary dump of 2^n float64 probabilities.
    """
    dim = 1 << n_qubits
    probs = np.zeros(dim, dtype=np.float64)
    
    total_shots = sum(counts.values())
    if total_shots == 0:
        # Write zeros or handle as error; here we write zeros
        pass
    else:
        # C3PQ indexes states with qubit 0 as the LSB, and ExaTNBackend.normalize_counts
        # already emits MSB-first keys, so int(bitstring, 2) is the state index directly.
        for bitstring, count in counts.items():
            try:
                idx = int(bitstring, 2)
                if 0 <= idx < dim:
                    probs[idx] = count / total_shots
            except ValueError:
                logger.error(f"Invalid bitstring key: {bitstring}")

    # Atomic write
    tmp_path = path.with_suffix(".tmp")
    probs.tofile(tmp_path)
    tmp_path.rename(path)

def check_environment():
    """Reports XACC/ExaTN environment status."""
    print("--- XACC/ExaTN Environment Check ---")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    try:
        import xacc

        print("xacc: IMPORTABLE")

        # Initialize XACC
        xacc.Initialize()

        # Check TNQVM
        try:
            accelerator = xacc.getAccelerator("tnqvm")
            print(f"TNQVM accelerator: AVAILABLE ({accelerator.name()})")
        except Exception as exc:
            print(f"TNQVM accelerator: NOT FOUND ({exc})")

        # Check which compilers can actually parse OpenQASM 2.0. Resolving the service is
        # not sufficient -- 'xasm' resolves everywhere and parses XACC's own DSL, not QASM.
        probe = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
            "h q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
        )
        for name in ["staq", "openqasm", "qasm", "xasm"]:
            try:
                xacc.getCompiler(name).compile(probe)
                print(f"Compiler '{name}': PARSES OpenQASM 2.0")
            except Exception as exc:
                first = str(exc).strip().splitlines()[0] if str(exc) else repr(exc)
                print(f"Compiler '{name}': unusable ({first})")

        # Check ExaTN visitor
        try:
            accelerator = xacc.getAccelerator(
                "tnqvm",
                {"tnqvm-visitor": "exatn"},
            )
            print(
                f"ExaTN visitor: SUCCESSFUL instantiation ({accelerator.name()})"
            )
        except Exception as exc:
            print(f"ExaTN visitor: FAILED instantiation ({exc})")

        # Bit-order probe: X on qubit 0 only. XACC's convention puts qubit 0 leftmost, so
        # the raw key should be '10'; the normalized key must be '01' (index 1).
        try:
            from proxysim.backends.exatn_backend import ExaTNBackend

            backend = ExaTNBackend(shots=64)
            circ = backend._extract_composite(
                backend.compiler.compile(
                    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
                    "x q[0];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
                )
            )
            buf = xacc.qalloc(2)
            backend.qpu.execute(buf, circ)
            raw = dict(buf.getMeasurementCounts())
            normalized = backend.normalize_counts(raw, 2, backend.reverse_bits)
            print(f"Bit-order probe (x q[0]): raw={raw} normalized={normalized}")
            if set(normalized) == {"01"}:
                print("Bit order: OK (qubit 0 is the LSB of the C3PQ state index)")
            else:
                print(
                    "Bit order: MISMATCH -- expected {'01'}. "
                    f"Re-run with reverse_bits={not backend.reverse_bits}."
                )
        except Exception as exc:
            print(f"Bit-order probe: FAILED ({exc})")

    except ImportError as exc:
        print(f"xacc: NOT IMPORTABLE ({exc})")
    except Exception as exc:
        print(f"XACC initialization error: {exc}")

    print("------------------------------------")

def main():
    parser = argparse.ArgumentParser(description="Run XACC/ExaTN simulation bank.")
    parser.add_argument("--check-environment", action="store_true", help="Check XACC/ExaTN installation and exit.")
    parser.add_argument("--qasm-bank", "--bank", type=str, required=False, help="Path to compiled QASM bank")
    parser.add_argument("--output-dir", "--out", type=str, required=False, help="Directory for result files")
    parser.add_argument("--shots", type=int, default=10000, help="Number of shots per circuit")
    parser.add_argument("--visitor", type=str, default="exatn", help="TNQVM visitor name")
    parser.add_argument("--pattern", type=str, default="*.qasm", help="Glob pattern for QASM files")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--resume", action="store_true", help="Skip completed files")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--no-reverse-bits",
        action="store_true",
        help="Do not reverse XACC bitstrings. XACC reports qubit 0 leftmost while C3PQ "
             "indexes qubit 0 as the LSB, so reversing is the default; use this only if "
             "--check-environment reports a bit-order mismatch.",
    )
    parser.add_argument(
        "--no-lower",
        action="store_true",
        help="Hand XACC the QASM exactly as written. By default each circuit is rewritten "
             "into the gate subset qelib1.inc declares (sx -> rx(pi/2) and so on, each "
             "exact up to a global phase), because the banks here are unlowered proxysim "
             "IR and staq rejects sx/sxdg/cp.",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        metavar="I/N",
        help="Process only the files whose position in the sorted list is congruent to I "
             "mod N. Circuits are independent, so N of these run concurrently on one node "
             "for a near-linear speedup. Each shard keeps its own manifest; combine them "
             "with --merge-shards once they finish.",
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Combine every manifest.shard*of*.json in --output-dir into manifest.json "
             "and exit. The analyzer reads that one file, so run this after a sharded run.",
    )
    parser.add_argument("--manifest", type=str, default="manifest.json", help="Manifest filename relative to output-dir")
    parser.add_argument("--fail-fast", action="store_true", help="Exit immediately on first failure")
    parser.add_argument("--dry-run", action="store_true", help="List files to be processed without executing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()

    if args.check_environment:
        check_environment()
        return

    if not args.qasm_bank or not args.output_dir:
        parser.error("--qasm-bank and --output-dir are required unless using --check-environment")

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    bank_path = Path(args.qasm_bank)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_shards:
        sys.exit(merge_shards(out_dir, args.manifest))

    shard = None
    if args.shard:
        try:
            shard = parse_shard(args.shard)
        except ValueError as exc:
            parser.error(str(exc))
        # Concurrent shards cannot share one manifest file -- the last writer would drop
        # everyone else's entries. Give each its own unless the caller named one.
        if args.manifest == "manifest.json":
            args.manifest = f"manifest.shard{shard[0]}of{shard[1]}.json"

    manifest_path = out_dir / args.manifest
    manifest = load_manifest(manifest_path)
    if manifest is None:
        manifest = RunManifest(
            results_dir=str(out_dir),
            qasm_bank=str(bank_path),
            shots=args.shots,
            visitor=args.visitor,
            lowered=not args.no_lower,
        )
    else:
        # Resuming into a directory whose .f64 files were produced under different
        # settings silently mixes two populations: the manifest can only record one shot
        # count, and exatn_analyze.py derives every statistical tolerance from it, so half
        # the results would be judged against the wrong noise floor. The circuits are
        # deterministic in their seeds, so growing a bank and resuming is the intended
        # workflow -- changing how the circuits are *run* mid-directory is not.
        conflicts = []
        if manifest.shots != args.shots:
            conflicts.append(f"shots {manifest.shots} -> {args.shots}")
        if manifest.lowered != (not args.no_lower):
            conflicts.append(f"lowered {manifest.lowered} -> {not args.no_lower}")
        if manifest.visitor != args.visitor:
            conflicts.append(f"visitor '{manifest.visitor}' -> '{args.visitor}'")
        if conflicts:
            logger.error(
                f"{manifest_path} already describes a run with different settings "
                f"({'; '.join(conflicts)}). Existing .f64 files in {out_dir} were produced "
                f"the old way. Either keep the original settings and --resume, or write to "
                f"a fresh --output-dir."
            )
            sys.exit(1)
        if manifest.qasm_bank != str(bank_path):
            logger.warning(
                f"{manifest_path} records bank '{manifest.qasm_bank}' but this run uses "
                f"'{bank_path}'; updating the record. Stems from two different banks in "
                f"one directory will not collide, but the provenance will be ambiguous."
            )
            manifest.qasm_bank = str(bank_path)

    # Discover files
    qasm_files = sorted(list(bank_path.rglob(args.pattern)))
    discovered = len(qasm_files)
    if shard:
        # Stride, not contiguous blocks. The sorted order groups by width, then kind, then
        # mode, and the kinds differ in cost (a CB sequence of length 24 is not a depth-1
        # TVD circuit), so contiguous blocks would hand one process all the expensive work.
        # Taking every Nth file interleaves the kinds and keeps the shards balanced.
        qasm_files = qasm_files[shard[0]::shard[1]]
    if args.limit:
        qasm_files = qasm_files[:args.limit]

    if shard:
        logger.info(f"Discovered {discovered} QASM files in {bank_path}; shard "
                    f"{shard[0]}/{shard[1]} takes {len(qasm_files)}")
    else:
        logger.info(f"Discovered {discovered} QASM files in {bank_path}")

    # Initialize Backend
    try:
        backend = ExaTNBackend(
            shots=args.shots,
            visitor=args.visitor,
            seed=args.seed,
            reverse_bits=not args.no_reverse_bits,
            lower=not args.no_lower,
        )
        manifest.compiler = backend.compiler_name
        manifest.lowered = backend.lower
    except Exception as e:
        logger.error(f"Backend initialization failed: {e}")
        sys.exit(1)

    # Processing loop
    processed = 0
    skipped = 0
    failed = 0
    start_time = time.perf_counter()

    for qasm_path in qasm_files:
        rel_path = str(qasm_path.relative_to(bank_path))
        result_path = out_dir / f"{qasm_path.stem}.f64"

        # Two independent records of "already done". The manifest is the bookkeeping; the
        # .f64 on disk is the artefact, and it is the one that survives sharding -- each
        # shard keeps its own manifest, so work finished by an earlier run or by a
        # differently-sharded one is visible only on disk. Checking both means --resume
        # does the right thing when the shard count changes between runs.
        if not args.overwrite and args.resume and (
                rel_path in manifest.completed or result_path.exists()):
            skipped += 1
            continue

        if args.dry_run:
            logger.info(f"Dry-run: would process {rel_path}")
            continue

        try:
            logger.info(f"Processing {rel_path}...")
            res = backend.run_qasm(qasm_path)

            # C3PQ expects results named after the group/stem, and the bank's stems are
            # unique across widths, kinds and trajectories -- so the flat layout here is
            # collision-free and exatn_analyze.py can find a group's members by name.
            # Filename: tvd_n02_d001_i000_ideal_k000.qasm -> ..._k000.f64
            write_c3pq_probs(result_path, res.counts, res.num_qubits)
            
            manifest.completed.append(rel_path)
            processed += 1
            
        except Exception as e:
            logger.error(f"Failed to process {rel_path}: {e}")
            manifest.failed[rel_path] = str(e)
            failed += 1
            if args.fail_fast:
                break
        
        # Periodic manifest save
        if processed % 10 == 0:
            save_manifest(manifest_path, manifest)

    save_manifest(manifest_path, manifest)
    elapsed = time.perf_counter() - start_time

    # Summary
    print("\n--- Execution Summary ---")
    if shard:
        print(f"Shard:            {shard[0]}/{shard[1]} -> {args.manifest}")
    print(f"Files discovered: {len(qasm_files)}"
          + (f" (of {discovered} in the bank)" if shard else ""))
    print(f"Completed:        {processed}")
    print(f"Skipped:         {skipped}")
    print(f"Failed:          {failed}")
    print(f"Total shots/file: {args.shots}")
    print(f"Elapsed runtime: {elapsed:.2f}s")
    if processed:
        print(f"Per circuit:      {elapsed / processed:.3f}s"
              f"  -> {discovered * elapsed / processed / 60:.1f} min for the whole bank"
              + (f" / {shard[1]} shards = "
                 f"{discovered * elapsed / processed / 60 / shard[1]:.1f} min"
                 if shard else ""))
    print("-------------------------")
    if shard:
        print("Run --merge-shards once every shard has finished:")
        print(f"  python examples/run_exatn_bank.py --out {out_dir} --bank {bank_path} "
              f"--merge-shards")

    if failed > 0 and not args.fail_fast:
        # Exit nonzero if any failed
        sys.exit(1)

if __name__ == "__main__":
    main()
