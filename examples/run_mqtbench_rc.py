"""General purpose Randomized Compilation (RC) analysis script for MQTBench algorithms.

This script mirrors the logic in `examples/run_qft_cycle_sweep.py` but allows
specifying the target algorithm/QASM and noise parameters via the CLI.

Protocol:
1. Load logical target circuit (from QASM or MQTBank).
2. Generate N_RC independently dressed (twirled) versions of the circuit.
3. For each dressed circuit, sample N_TRAJ noisy trajectories.
4. Simulate each trajectory using the selected backend.
5. Average distributions over trajectories, then over RC realizations.
6. Apply readout noise and compute TVD against the ideal distribution.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from typing import Any, Tuple, List, Dict

warnings.filterwarnings("ignore")

import numpy as np
from qiskit.quantum_info import Statevector

import _bootstrap
from proxysim import NoiseModel, save_results
from proxysim.backends import StatevectorBackend, ExaTNBackend
from proxysim.circuit import circuit_from_qasm
from proxysim.mqtbank import repeat
from proxysim.noise import (
    apply_readout_to_distribution,
    sample_trajectory,
)
from proxysim.parallel import pmap
from proxysim.rc import pauli_twirl

# =============================================================================
# Workers
# =============================================================================

def sv_worker(args):
    """Statevector simulation worker for a single trajectory."""
    (target, noise, ideal_prob, seed, rc_index, traj_index) = args
    
    # Deterministic seed for this specific RC realization and trajectory
    rc_seed = seed + 1_000_003 * rc_index
    traj_seed = rc_seed + 1_000_000 * traj_index
    
    # 1. RC Dressing (Logical equivalence check should be done in main)
    # We use pauli_twirl(..., mark_virtual=True) to ensure twirl Paulis
    # are treated as frame changes and don't add to the noise budget.
    dressed, virtual = pauli_twirl(target, random.Random(rc_seed), mark_virtual=True)
    
    # 2. Noise Trajectory Sampling
    noisy_dressed = sample_trajectory(
        dressed, 
        noise, 
        random.Random(traj_seed), 
        virtual=virtual
    )
    
    # 3. Simulation
    backend = StatevectorBackend()
    psi = np.asarray(Statevector(backend._build(noisy_dressed)).data, dtype=complex)
    prob = np.abs(psi)**2
    
    return prob

def run_sv_analysis(target, noise, ideal_prob, n_rc, n_traj, seed, n_workers):
    """Perform the full RC-averaged TVD calculation using Statevector backend."""
    n_qubits = target.n_qubits
    
    # Prepare jobs for pmap: (target, noise, ideal_prob, seed, rc_index, traj_index)
    jobs = []
    for rc in range(n_rc):
        for traj in range(n_traj):
            jobs.append((target, noise, ideal_prob, seed, rc, traj))
            
    # Run parallel simulations
    all_probs = pmap(sv_worker, jobs, n_workers=n_workers)
    
    # Average over trajectories and RC realizations
    # all_probs is a list of 2^n arrays.
    combined_prob = np.mean(all_probs, axis=0)
    
    # Apply readout noise analytically
    final_prob = apply_readout_to_distribution(combined_prob, noise, n_qubits)
    
    # Compute TVD
    tvd = 0.5 * float(np.abs(final_prob - ideal_prob).sum())
    return tvd

# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="RC Noise Analysis for MQTBench Algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    src = ap.add_argument_group("source circuit")
    src.add_argument("--algo", help="MQT Bench algorithm name, e.g. qft")
    src.add_argument("--n", type=int, help="qubit count")
    src.add_argument("--qasm", help="path to QASM file")
    src.add_argument("--reps", type=int, default=1, help="repetitions of the circuit (U^d)")
    
    noise_group = ap.add_argument_group("noise parameters")
    noise_group.add_argument("--p1", type=float, default=5e-5)
    noise_group.add_argument("--p2", type=float, default=2.5e-4)
    noise_group.add_argument("--p-readout", type=float, default=2.5e-4)
    noise_group.add_argument("--p-idle", type=float, default=5e-5)
    noise_group.add_argument("--p-z1", type=float, default=0.0)
    noise_group.add_argument("--p-zz", type=float, default=0.0)
    noise_group.add_argument("--theta-1q", type=float, default=0.0)
    noise_group.add_argument("--theta-zz", type=float, default=0.0)
    
    exec_group = ap.add_argument_group("execution")
    exec_group.add_argument("--backend", default="statevector", choices=["statevector", "exatn", "c3pq"],
                            help="simulation backend to use")
    exec_group.add_argument("--n-rc", type=int, default=24, help="number of RC realizations")
    exec_group.add_argument("--n-traj", type=int, default=12, help="trajectories per RC realization")
    exec_group.add_argument("--seed", type=int, default=42)
    exec_group.add_argument("--n-workers", type=int, default=None)
    
    args = ap.parse_args()
    
    # 1. Load Circuit
    if args.qasm:
        with open(args.qasm, encoding="utf-8") as fh:
            target, measured = circuit_from_qasm(fh.read())
        label = os.path.splitext(os.path.basename(args.qasm))[0]
    elif args.algo and args.n is not None:
        from proxysim import mqtbank
        target, measured = mqtbank.fetch(args.algo, args.n)
        label = args.algo
    else:
        ap.error("Provide either --qasm or both --algo and --n")

    # Repeat target if requested (U^d)
    if args.reps > 1:
        target = repeat(target, args.reps)
    
    n_qubits = target.n_qubits
    print(f"Target: {label} (n={n_qubits}, reps={args.reps})")
    print(target.summary())
    
    # 2. Define Noise Model
    noise = NoiseModel(
        enabled=True, 
        p1=args.p1, p2=args.p2, p_readout=args.p_readout, p_idle=args.p_idle,
        p_z1=args.p_z1, p_zz=args.p_zz, theta_1q=args.theta_1q, theta_zz=args.theta_zz
    )
    
    # 3. Compute Ideal Distribution
    print("Computing ideal distribution...")
    sv = StatevectorBackend()
    ideal_psi = np.asarray(Statevector(sv._build(target)).data, dtype=complex)
    ideal_prob = np.abs(ideal_psi)**2
    
    # 4. Run Analysis
    if args.backend == "statevector":
        n_workers = args.n_workers or max(1, (os.cpu_count() or 2) - 2)
        print(f"Running RC analysis (SV) with {n_workers} workers, "
              f"n_rc={args.n_rc}, n_traj={args.n_traj}...")
        
        tvd = run_sv_analysis(target, noise, ideal_prob, args.n_rc, args.n_traj, args.seed, n_workers)
        print(f"Resulting TVD: {tvd:.6f}")
        
    elif args.backend == "exatn":
        print("ExaTN backend is supported via bank generation. "
              "Please use examples/generate_qasm_mqtbench_rc.py and examples/run_exatn_bank.py")
        sys.exit(1)
        
    elif args.backend == "c3pq":
        print("C-3PQ backend is supported via bank generation. "
              "Please use examples/generate_qasm_mqtbench_rc.py and C-3PQ pipeline.")
        sys.exit(1)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
