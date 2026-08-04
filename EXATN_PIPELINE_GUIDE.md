# ExaTN Simulation Pipeline Guide

This guide describes how to execute the XACC/TNQVM/ExaTN simulation pipeline on Perlmutter to generate result files compatible with the C3PQ analysis tools.

## 1. Environment Setup

### Request an Interactive Node
Request a GPU node via `salloc` to ensure hardware compatibility:

```bash
salloc -N 1 -C gpu -q interactive -t 01:00:00 -A <your_account>
```

### Activate the Stack
Navigate to the repository and source the activation script. This configures the XACC/TNQVM paths, loads necessary MKL libraries, and activates the Python virtual environment.

```bash
cd /pscratch/sd/n/nparr1/quantumsim
source scripts/activate_tnqvm.sh
```

---

## 2. Pipeline Execution Flow

The pipeline consists of three main stages: **Generation**, **Simulation**, and **Analysis**.

### Stage A: Generate the QASM Bank
Create the set of OpenQASM 2.0 circuits to be simulated. The `generate_exatn_bank.py` script creates LNN brickwork circuits.

```bash
# Example: Generate widths 4, 6, 8 with 1, 2, 4 cycles
python examples/generate_exatn_bank.py --out qasm_bank_exatn --widths 4 6 8 --cycles 1 2 4
```

### Stage B: Run the Simulations
Execute the circuits using the ExaTN backend. This script iterates through the bank and writes probability distributions as binary `.f64` files.

```bash
# --bank / --qasm-bank: path to the generated QASM files
# --out / --output-dir: directory to save the .f64 results
python examples/run_exatn_bank.py --bank qasm_bank_exatn --out results_exatn
```

**Tip:** To verify your XACC environment is correctly configured before running a full bank, use:
```bash
python examples/run_exatn_bank.py --check-environment
```

### Stage C: Analyze Results
Because the output format mirrors the C3PQ simulator exactly, you can use the existing analysis tool to generate TVD plots and compute bounds.

```bash
# --bank: the QASM bank used for simulation
# --results: the directory containing .f64 files
# --staging: a label for the run (used for folder organization in results)
python examples/c3pq_analyze.py --bank qasm_bank_exatn --results results_exatn --staging exatn_run
```

---

## 3. Quick-Start Summary

For a rapid end-to-end test, run these commands in sequence:

```bash
# 1. Setup
source scripts/activate_tnqvm.sh

# 2. Generate
python examples/generate_exatn_bank.py --out test_bank --widths 4 6 --cycles 1 2

# 3. Simulate
python examples/run_exatn_bank.py --bank test_bank --out test_results

# 4. Analyze
python examples/c3pq_analyze.py --bank test_bank --results test_results --staging test_run
```

## 4. Technical Details

- **Result Format**: Results are stored as raw `float64` binary arrays (`numpy.tofile`).
- **Bit Order**: The pipeline implements LSB (Least Significant Bit) convention where qubit $i$ corresponds to bit $i$.
- **Compatibility**: Output files are 100% compatible with `proxysim.c3pq.read_probs`.
