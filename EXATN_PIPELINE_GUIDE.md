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

Three stages: **Generation**, **Simulation**, and **Analysis**. ExaTN stands in for the
C-3PQ binary as the simulation engine; the bank and the analysis are the same physics
either way.

### Stage A: Build the bounding bank
`build_bounding_bank.py` writes the circuits the analysis needs — the `ideal`/`noisy`/`rc`
TVD arms, the cycle-benchmarking decays, and the calibration probe — plus the
`manifest.json` that says what each one is for. The defaults are a smoke config
(`--widths 2`, K=4, 2 instances, 4 decays; ~490 circuits, 181 groups).

```bash
python examples/build_bounding_bank.py --out qasm_bank_bounding
# --widths / --depths / --cb-depths / --trajectories / --cb-decays to scale it up
```

### Stage B: Run the simulations
One `.f64` per QASM file, named after the file's stem, written flat into the output
directory. Trajectories are *not* averaged here — `exatn_analyze.py` does that from the
per-file weights in the bank manifest.

```bash
# --bank / --qasm-bank: the bank from stage A
# --out / --output-dir: directory to save the .f64 results
python examples/run_exatn_bank.py --bank qasm_bank_bounding --out results_exatn \
    --shots 100000
```

Shots matter: every tolerance in stage C is derived from this number, and at the smoke
config the sampling bias of the TVD is the same order as the TVD itself. 100k is a
reasonable floor; the analyzer prints the bias next to each point so it stays visible.

**Tip:** To verify your XACC environment before running a full bank, use:
```bash
python examples/run_exatn_bank.py --check-environment
```
This reports which compilers actually parse OpenQASM 2.0 (not merely resolve) and runs an
`x q[0]` probe that confirms the bit order.

### Stage C: Analyze Results
`exatn_analyze.py` is `c3pq_analyze.py` with the engine swapped: it imports every check,
the CB fit, the bound and the figure from it, and changes only where the vectors come from
and how the tolerances are set. `--shots` defaults to the value in the run manifest.

```bash
# --bank: the bank from stage A (its manifest says what each circuit means)
# --results: the directory of .f64 files from stage B
python examples/exatn_analyze.py --bank qasm_bank_bounding --results results_exatn
```

It writes `results/exatn_bounding_<n>qubits*.png` and `results/exatn_bounding_data*.npz`,
and exits nonzero if a check fails (`--no-assert` reports without failing).

**Note on `generate_exatn_bank.py`.** That script writes plain LNN brickwork circuits with
no arms, no CB decays and no manifest. It is a backend smoke test — useful for confirming
XACC/TNQVM/ExaTN runs at all and how it scales with width — and there is no analyze stage
for it. It is not an input to `exatn_analyze.py` or `c3pq_analyze.py`; pointing either at
it fails with `FileNotFoundError: <bank>/manifest.json`.

```bash
python examples/generate_exatn_bank.py --out qasm_bank_exatn --widths 4 6 8 --cycles 1 2 4
python examples/run_exatn_bank.py --bank qasm_bank_exatn --out results_exatn_smoke
```

---

## 3. Quick-Start Summary

For a rapid end-to-end test, run these commands in sequence:

```bash
# 1. Setup
source scripts/activate_tnqvm.sh
python examples/run_exatn_bank.py --check-environment

# 2. Build the bank (defaults: n=2 smoke config, ~490 circuits)
python examples/build_bounding_bank.py --out qasm_bank_bounding

# 3. Simulate
python examples/run_exatn_bank.py --bank qasm_bank_bounding --out results_exatn \
    --shots 100000

# 4. Analyze
python examples/exatn_analyze.py --bank qasm_bank_bounding --results results_exatn
```

## 4. Technical Details

- **Result Format**: Results are stored as raw `float64` binary arrays (`numpy.tofile`).
- **Bit Order**: The pipeline implements LSB (Least Significant Bit) convention where qubit $i$ corresponds to bit $i$. XACC reports measurement bitstrings with qubit 0 *leftmost*, so the backend reverses them; `--check-environment` runs an `x q[0]` probe that verifies this, and `--no-reverse-bits` overrides it if a build differs.
- **QASM Compiler**: XACC must parse the bank with `staq` (its OpenQASM 2.0 front end), *not* `xasm`. `xasm` is XACC's own DSL and is always registered, so the backend probe-compiles each candidate rather than trusting the service lookup. A `mismatched input '// ...' expecting {'__qpu__', '['}` error means `xasm` was selected.
- **Gate Lowering**: The bounding bank writes unlowered proxysim IR, which includes `sx`, `sxdg` and (at `--theta-zz != 0`) `cp` — none of which standard `qelib1.inc` declares, so staq rejects them. The backend therefore rewrites each circuit into the qelib1 subset before compiling (`sx` → `rx(pi/2)`, `s` → `rz(pi/2)`, `cp` → `cu1`, and so on, each exact up to a global phase), via `proxysim.c3pq.lower_for_c3pq`. `--no-lower` hands XACC the file as written. This requires qiskit, which `circuit_from_qasm` uses to read the bank; without it the backend warns once and falls back to the raw text.
- **Shot Noise**: ExaTN samples, where C-3PQ returns exact probabilities. `exatn_analyze.py` therefore replaces `c3pq_analyze.py`'s exact tolerances (1e-9, 1e-12) with statistical ones derived from `--shots` and from each group's effective sample count (`shots x K`, since a group averages K trajectories), and prints each noise floor next to the number it governs. The TVD is biased *upward* by roughly `sum_i sqrt(p_i(1-p_i)/(2 pi S))`; that bias is reported per point and stored in the `.npz` but never subtracted, since the QCAP bound is an upper bound and deflating the measurement would manufacture agreement.
- **Compatibility**: Output files are 100% compatible with `proxysim.c3pq.read_probs`.
