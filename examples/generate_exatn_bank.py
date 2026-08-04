"""Generate a bank of brickwork QASM circuits for ExaTN validation.
This script creates a directory of .qasm files that can be consumed by
`examples/run_exatn_bank.py`.
"""

import argparse
import os
from proxysim.circuit import bench_brickwork, even_pairs, odd_pairs
from proxysim.qasm import write_qasm

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="qasm_bank_exatn", help="Output directory")
    ap.add_argument("--widths", nargs="+", type=int, default=[4, 6, 8, 10])
    ap.add_argument("--cycles", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Generating QASM bank at: {args.out}")
    
    # We use a standard LNN brickwork structure: [even pairs, odd pairs]
    # This matches the intended use of 'cycles' in bench_brickwork.
    # Each 'cycle' in the list passed to bench_brickwork is a layer of 2q gates.
    # To simulate a standard brickwork cycle (Even then Odd), we provide:
    # cycles = [even_pairs(n), odd_pairs(n)]
    
    count = 0
    for n in args.widths:
        # The 'cycles' argument to bench_brickwork is the sequence of 2q layers
        # that is repeated n_cycles times.
        cycle_layers = [even_pairs(n), odd_pairs(n)]
        
        for c in args.cycles:
            # Generate a brickwork circuit
            # n_hadamards=n ensures we start in a superposition (essential for entropy/complexity)
            circ = bench_brickwork(
                n=n, 
                n_cycles=c, 
                cycles=cycle_layers, 
                seed=args.seed, 
                n_hadamards=n
            )
            
            # Path: qasm_bank_exatn/n{n}/c{c}.qasm
            path = os.path.join(args.out, f"n{n:02d}", f"c{c:03d}.qasm")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Write to file
            write_qasm(
                path, 
                circ, 
                header_lines=[
                    f"Brickwork | n={n} cycles={c} seed={args.seed}",
                    f"Structure: LNN [Even, Odd] repeated {c} times",
                    f"Initial state: Hadamards on all qubits"
                ]
            )
            count += 1
            print(f"Wrote {path}")

    print(f"\nSuccessfully generated {count} QASM files in {args.out}")

if __name__ == "__main__":
    main()
