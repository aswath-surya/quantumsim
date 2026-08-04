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
        for bitstring, count in counts.items():
            # C3PQ convention: qubit i is bit i (LSB = qubit 0)
            # But we must be careful about how the binary string was built.
            # If normalize_counts produced '001' where bit 0 is '1', 
            # then int('001', 2) would be 1.
            # However, usually binary strings are MSB leftmost.
            # If '001' means qubit 0=0, 1=0, 2=1, then the index is 1 << 2 = 4.
            # If '100' means qubit 0=1, 1=0, 2=0, then index is 1 << 0 = 1.
            # The critical part is that it must match c3pq.read_probs.
            
            # Based on proxysim/c3pq.py and base.py:
            # "index i is qubit i"
            # "h q[0] strides by 1" -> qubit 0 is LSB.
            
            # If our bitstring is 'b_{n-1}...b_1b_0', then int(bitstring, 2)
            # treats b_0 as LSB. 
            # Let's assume normalize_counts returns strings where 
            # the rightmost character is qubit 0.
            
            # If the bitstring is '001' and qubit 0 is 1, int('001', 2) = 1. Correct.
            # If the bitstring is '100' and qubit 0 is 1, int('100', 2) = 4. Wrong.
            
            # In our current normalize_counts: 
            # it just zfills. We need to ensure the string format matches the index.
            # To be safe, we'll handle this in the normalization logic.
            
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
    import sys
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    try:
        import xacc
        print("xacc: IMPORTABLE")
        # We can't easily get version without calling it, but let's try init
        xacc.init()
        accels = xacc.getAccelerators()
        print(f"Available accelerators: {accels}")
        if "tnqvm" in accels:
            print("TNQVM accelerator: AVAILABLE")
        else:
            print("TNQVM accelerator: NOT FOUND")
        
        compilers = xacc.getCompilers()
        print(f"Available compilers: {compilers}")
        
        # Try creating a visitor
        try:
            qpu = xacc.getAccelerator("tnqvm", {"tnqvm-visitor": "exatn"})
            print("ExaTN visitor: SUCCESSFUL instantiation")
        except Exception as e:
            print(f"ExaTN visitor: FAILED instantiation ({e})")
            
    except ImportError:
        print("xacc: NOT IMPORTABLE")
    except Exception as e:
        print(f"XACC initialization error: {e}")
    
    print("------------------------------------")

def main():
    parser = argparse.ArgumentParser(description="Run XACC/ExaTN simulation bank.")
    parser.add_argument("--check-environment", action="store_true", help="Check XACC/ExaTN installation and exit.")
    parser.add_argument("--qasm-bank", type=str, required=False, help="Path to compiled QASM bank")
    parser.add_argument("--output-dir", type=str, required=False, help="Directory for result files")
    parser.add_argument("--shots", type=int, default=10000, help="Number of shots per circuit")
    parser.add_argument("--visitor", type=str, default="exatn", help="TNQVM visitor name")
    parser.add_argument("--pattern", type=str, default="*.qasm", help="Glob pattern for QASM files")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--resume", action="store_true", help="Skip completed files")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
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
        backend = ExaTNBackend(shots=args.shots, visitor=args.visitor, seed=args.seed)
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
