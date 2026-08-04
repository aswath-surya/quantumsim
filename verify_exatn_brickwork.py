import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from proxysim.backends.exatn_backend import ExaTNBackend
from proxysim.backends.base import total_variation_distance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("verify_exatn_brickwork")

def get_reference_distribution(ref_path: Path, n_qubits: int) -> Dict[str, float]:
    """Reads a C3PQ .f64 file and returns it as a dictionary {bitstring: prob}."""
    dim = 1 << n_qubits
    probs = np.fromfile(ref_path, dtype=np.float64)
    if probs.size != dim:
        raise ValueError(f"Reference file {ref_path} has size {probs.size}, expected {dim}")
    
    dist = {}
    for i in range(dim):
        if probs[i] > 0:
            # Convert index to bitstring. 
            # C3PQ convention: index i is qubit i, so i=1 is '001' (LSB = qubit 0)
            bitstring = bin(i)[2:].zfill(n_qubits)
            dist[bitstring] = float(probs[i])
    return dist

def run_validation():
    parser = argparse.ArgumentParser(description="Validate ExaTN brickwork outputs against reference.")
    parser.add_argument("--qasm-bank", type=str, required=True, help="Path to the test QASM bank")
    parser.add_argument("--exatn-results", type=str, required=True, help="Path to ExaTN result files (.f64)")
    parser.add_argument("--reference-results", type=str, required=True, help="Path to reference results (.f64)")
    parser.add_argument("--shots", type=int, default=100000, help="Shots used for ExaTN run")
    parser.add_argument("--report", type=str, default="exatn_brickwork_validation.json", help="JSON report output path")
    
    args = parser.parse_args()

    bank_path = Path(args.qasm_bank)
    exatn_dir = Path(args.exatn_results)
    ref_dir = Path(args.reference_results)
    
    # We want to find the smallest brickwork circuits.
    # Assuming filenames like 'brickwork_N2_depth1_instance0.qasm'
    qasm_files = sorted(list(bank_path.glob("*.qasm")))
    
    report_data = {}
    summary_table = []

    for qasm_path in qasm_files:
        logger.info(f"Validating {qasm_path.name}...")
        
        # Infer metadata from filename
        # brickwork_N{N}_depth{D}_instance{I}.qasm
        import re
        match = re.search(r"brickwork_N(\d+)_depth(\d+)_instance(\d+)", qasm_path.name)
        if not match:
            logger.warning(f"Skipping {qasm_path.name} as it doesn't match expected brickwork naming.")
            continue
            
        n_qubits = int(match.group(1))
        depth = int(match.group(2))
        instance = int(match.group(3))
        
        # Expected result filenames
        exatn_res_path = exatn_dir / f"{qasm_path.stem}.f64"
        ref_res_path = ref_dir / f"{qasm_path.stem}.f64"
        
        if not exatn_res_path.exists() or not ref_res_path.exists():
            logger.error(f"Missing result files for {qasm_path.name}")
            continue
            
        # Read distributions
        # ExaTN result is a .f64 file (binary numpy array)
        # Reference result is also a .f64 file
        
        # For ExaTN, we can just use numpy since run_exatn_bank.py writes it as .tofile()
        exatn_probs = np.fromfile(exatn_res_path, dtype=np.float64)
        ref_probs = np.fromfile(ref_res_path, dtype=np.float64)
        
        if exatn_probs.size != (1 << n_qubits) or ref_probs.size != (1 << n_qubits):
            logger.error(f"Dimension mismatch for {qasm_path.name}")
            continue
            
        tvd = 0.5 * np.sum(np.abs(exatn_probs - ref_probs))
        
        # Collision Probability: sum(p_i^2)
        exatn_coll = np.sum(exatn_probs**2)
        ref_coll = np.sum(ref_probs**2)
        
        passed = tvd <= 0.02 # Initial threshold
        
        res_entry = {
            "qasm_path": str(qasm_path),
            "num_qubits": n_qubits,
            "depth": depth,
            "instance": instance,
            "shots": args.shots,
            "tvd": float(tvd),
            "reference_collision_probability": float(ref_coll),
            "exatn_collision_probability": float(exatn_coll),
            "bit_order": "C3PQ-compatible",
            "passed": bool(passed)
        }
        report_data[qasm_path.name] = res_entry
        summary_table.append((qasm_path.name, n_qubits, depth, tvd, passed))

    # Write JSON report
    with open(args.report, "w") as f:
        json.dump(report_data, f, indent=2)
    
    # Print Table
    print("\n" + "="*80)
    print(f"{'Circuit':<40} {'N':<4} {'D':<4} {'TVD':<10} {'Status'}")
    print("-" * 80)
    for name, n, d, tvd, passed in summary_table:
        status = "PASS" if passed else "FAIL"
        print(f"{name:<40} {n:<4} {d:<4} {tvd:<10.4f} {status}")
    print("="*80)

if __name__ == "__main__":
    run_validation()
