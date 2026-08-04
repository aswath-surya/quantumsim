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
    
    manifest_path = out_dir / args.manifest
    manifest = load_manifest(manifest_path) or RunManifest(
        results_dir=str(out_dir),
        qasm_bank=str(bank_path),
        shots=args.shots,
        visitor=args.visitor
    )

    # Discover files
    qasm_files = sorted(list(bank_path.rglob(args.pattern)))
    if args.limit:
        qasm_files = qasm_files[:args.limit]
    
    logger.info(f"Discovered {len(qasm_files)} QASM files in {bank_path}")

    # Initialize Backend
    try:
        backend = ExaTNBackend(
            shots=args.shots,
            visitor=args.visitor,
            seed=args.seed,
            reverse_bits=not args.no_reverse_bits,
        )
        manifest.compiler = backend.compiler_name
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
        
        if args.resume and rel_path in manifest.completed:
            skipped += 1
            continue
        if not args.overwrite and (out_dir / f"{Path(qasm_path.stem)}.f64").exists():
            # If we are not resuming and not overwriting, but the file exists, we might skip
            # But resume is the explicit flag for this.
            pass

        if args.dry_run:
            logger.info(f"Dry-run: would process {rel_path}")
            continue

        try:
            logger.info(f"Processing {rel_path}...")
            res = backend.run_qasm(qasm_path)
            
            # C3PQ expects results named after the group/stem.
            # Filename: tdv_n02_d001_i000_ideal.qasm -> tdv_n02_d001_i000_ideal.f64
            result_filename = f"{qasm_path.stem}.f64"
            result_path = out_dir / result_filename
            
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
    print(f"Files discovered: {len(qasm_files)}")
    print(f"Completed:        {processed}")
    print(f"Skipped:         {skipped}")
    print(f"Failed:          {failed}")
    print(f"Total shots/file: {args.shots}")
    print(f"Elapsed runtime: {elapsed:.2f}s")
    print("-------------------------")

    if failed > 0 and not args.fail_fast:
        # Exit nonzero if any failed
        sys.exit(1)

if __name__ == "__main__":
    main()
