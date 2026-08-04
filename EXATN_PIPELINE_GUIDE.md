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

Run them in order. Stage A is not optional: the *bank* is the circuits **plus** the
manifest saying which of them form a group, with what weights and what estimator, and none
of that is recoverable from the `.qasm` files. Stage C reads the bank and the results
directory as two separate inputs — `--bank` is stage A's output, `--results` is stage B's.

### Stage A: Build the bounding bank
`build_bounding_bank.py` writes the circuits the analysis needs — the `ideal`/`noisy`/`rc`
TVD arms, the cycle-benchmarking decays, and the calibration probe — plus the
`manifest.json` that says what each one is for. The defaults are a smoke config
(`--widths 2`, K=4, 2 instances, 4 decays; ~490 circuits, 181 groups).

```bash
python examples/build_bounding_bank.py --out qasm_bank_bounding
# --widths / --depths / --cb-depths / --trajectories / --cb-decays to scale it up
# --dry-run reports the circuit and byte count without writing anything -- use it to size
# a scaled-up config before committing to it
```

**The defaults are a smoke config, not a measurement.** Two of them dominate the figure:

- `--trajectories` (K, default 4). The noisy and rc arms average K sampled Pauli-error
  realisations, and at these error rates most realisations carry no error at all — so one
  bad draw out of K shifts the averaged distribution by 1/K of its own distance and the
  measured TVDs come out **quantised** in a comb at multiples of ~1/K. On the plot that
  reads as wild outliers at 0.25 and 0.5; it is not scatter, and more shots will not
  touch it. `run_bounding` uses 150. The analyzer prints a warning below 32.
- `--instances` (default 2). This is literally the number of markers per depth per arm.
  Five or more before the spread means anything.

Trajectory noise is ~1/√K and shot noise ~1/√S, so at K=4 / 100k shots the first is ~150×
the second. Shots are not what to buy.

#### Sizing it

Per width, the bank is

```
circuits = 1                                                   # calibration probe
         + modes x depths x instances x (1 + 2K)               # TVD arms
         + modes x cycles x cb_depths x cb_decays x (1 + K)    # CB decays
```

At n=2 (`modes=2, depths=7, cycles=1, cb_depths=6`):

| `--trajectories` | `--instances` | `--cb-decays` | TVD | CB | total | vs. default |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 2 | 4 | 252 | 240 | **493** | 1× (the default smoke config) |
| 50 | 5 | 4 | 7,070 | 2,448 | **9,519** | 19× |
| 150 | 5 | 4 | 21,070 | 7,248 | **28,319** | 57× |
| 150 | 5 | 30 | 21,070 | 54,360 | **75,431** | 153× |

Scale from the elapsed time your last run printed — at n=2 the cost is XACC compile plus
sampling, roughly linear in circuit count.

**Leave `--cb-decays` alone.** It is the most expensive knob and the least useful one here:
at `theta_zz == 0` the bound takes its e_F from the Clifford (stim) path, so the CB arms
only sharpen cross-check [5], and raising decays from 4 to 30 nearly triples the bank to
buy that alone. `--cb-decays-ref` on the analyzer controls the path that *does* feed the
bound, and costs seconds.

#### Growing an existing bank

Circuit seeds are deterministic in `(depth, instance, arm, k)`, so raising `--trajectories`
or `--instances` regenerates the existing files byte-identically — the builder skips them
and writes only what is new, and `run_exatn_bank.py --resume` skips the `.f64` you already
have. You pay for the increment:

```bash
python examples/build_bounding_bank.py --out qasm_bank_bounding \
    --trajectories 150 --instances 5 --dry-run     # check the count, then drop --dry-run

python examples/run_exatn_bank.py --bank qasm_bank_bounding --out results_exatn \
    --shots 100000 --resume                        # same --shots as the original run

python examples/exatn_analyze.py --bank qasm_bank_bounding --results results_exatn
```

Keep `--shots` at whatever the directory was first run with. A results directory records
one shot count and stage C derives every tolerance from it, so mixing two would judge half
the files against the wrong noise floor; `run_exatn_bank.py` refuses the run rather than
letting that happen. To change shots, use a fresh `--output-dir`.

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

Give each bank its own output directory. The run manifest that records `shots`, the
compiler and the lowering flag is per-directory, and stage C reads `--shots` from it — so
mixing two banks' results in one directory leaves the second run's manifest describing
both. Stems will not collide, but the provenance will be wrong.

**Watch the first few circuits.** Let a couple complete before walking away:

- `Could not lower ... check that qiskit is importable` — lowering has fallen back to the
  raw source, and staq will then reject `sx`/`sxdg`. This warning is printed **once** per
  run, so it is easy to miss in the scrollback. Fix by installing qiskit into the venv; see
  *Gate Lowering* below.
- `XACC compiler '...' failed to parse QASM file ...` — the compile itself failed. The
  message says whether lowering was applied, which separates a gate-set problem from a
  compiler-selection one.

#### Running it in parallel on one node

Circuits are independent and each writes its own `.f64`, so the run shards cleanly.
`--shard I/N` takes every Nth file from the sorted list — a stride rather than a
contiguous block, so the expensive CB sequences and the cheap short-depth TVD circuits
spread evenly across shards.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1     # each shard is single-threaded; see below
N=16
for i in $(seq 0 $((N-1))); do
  python examples/run_exatn_bank.py --bank qasm_bank_bounding --out results_exatn \
      --shots 100000 --resume --shard $i/$N > shard$i.log 2>&1 &
done
wait

python examples/run_exatn_bank.py --bank qasm_bank_bounding --out results_exatn \
    --merge-shards
```

At n=2 nearly all the per-circuit cost is single-threaded Python — qiskit parse, lowering,
staq compile — not tensor contraction, so this scales close to linearly and the two
`*_NUM_THREADS` exports matter: without them each shard's MKL will try to grab every core
and the shards will fight.

Each shard keeps its own `manifest.shardIofN.json`, because concurrent processes appending
to one file would lose entries. `--merge-shards` folds them into the `manifest.json` that
stage C reads, and refuses to merge shards that disagree on shots, visitor or lowering.

Use plain background processes inside your existing `salloc`, not `srun -n $N` — ExaTN may
initialise MPI, and launching the shards as MPI ranks makes them one communicating job
rather than N independent ones. For the same reason, don't reach for Python
`multiprocessing` inside the script: `xacc.Initialize()` sets up global state that does not
survive a fork.

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

By default the figure plots only the randomly-compiled arm, since the QCAP bound is stated
for the RC'd circuit — the un-compiled series shares the axis without being what the curve
bounds. `--arms both` draws both, `--arms noisy` only the un-compiled one. This is a
presentation choice: both arms are always written to the `.npz` (`*_tvd` and
`*_tvd_nonrc`) and check [4] compares them either way.

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
python -c "import qiskit"                      # lowering needs it; see Gate Lowering
python examples/run_exatn_bank.py --check-environment

# 2. Build the bank (defaults: n=2 smoke config, ~490 circuits)
python examples/build_bounding_bank.py --out qasm_bank_bounding

# 3. Simulate
python examples/run_exatn_bank.py --bank qasm_bank_bounding --out results_exatn \
    --shots 100000

# 4. Analyze
python examples/exatn_analyze.py --bank qasm_bank_bounding --results results_exatn
```

## 4. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `no bank manifest at qasm_bank_bounding/manifest.json` | Stage A was skipped, or `--bank` was pointed at the results directory. Run `build_bounding_bank.py --out qasm_bank_bounding` first. `--bank` is the circuits, `--results` is the `.f64` files. |
| `FileNotFoundError: <bank>/manifest.json` from `c3pq_analyze.py` | Same cause, older message. Also what you get from pointing either analyzer at a `generate_exatn_bank.py` brickwork bank, which has no manifest and no groups. |
| `Invalid xasm source: ... mismatched input '// ...' expecting {'__qpu__', '['}` | The `xasm` compiler was selected. It is XACC's own DSL, always registered, and cannot parse OpenQASM. The backend now probe-compiles candidates instead of trusting the lookup; if you still see this, `--check-environment` will show whether *any* compiler parses OpenQASM in this build. |
| `Could not lower ... passing the original source to XACC` | qiskit is not importable, so `circuit_from_qasm` cannot read the bank. Circuits containing `sx`/`sxdg`/`cp` will then fail to compile. Printed once per run. |
| `no shot count: .../manifest.json is missing or has no 'shots'` | Stage B wrote no run manifest (interrupted before the first save), or `--results` is wrong. Pass `--shots N` matching the run, or re-run stage B. |
| `N groups have no results at all` | Stage B has not covered this bank. Check its summary for failures; `--resume` continues a partial run. |
| `M groups incomplete` and they are skipped | Some trajectories of a group are missing. Re-run stage B with `--resume`, or accept a reweighted estimate with `exatn_analyze.py --allow-partial`. |
| `calibration=FAIL` with `both conventions fit` | Not a bit-order failure — the probe cannot discriminate at this width and shot count. Raise `--shots`. |
| `calibration=FAIL` with the transposed convention winning | A real bit-order problem. Re-run `--check-environment` and, if its `x q[0]` probe disagrees with the default, re-run stage B with `--no-reverse-bits`. |
| TVD points flagged as `within 2x its sampling bias` | Not an error. The measurement is swamped by shot noise at those depths; raise `--shots` before reading them as physics. |
| Only a couple of markers per depth | That is `--instances` from stage A (default 2), one marker per instance per arm. Rebuild the bank with more. |
| TVD points sitting at 0.25 / 0.5 with nothing in between | Trajectory quantisation at small K, not outliers — see *The defaults are a smoke config* under stage A. Rebuild with `--trajectories 150`. More shots will not help. |
| Green "NOT compiled" points you don't want | `--arms rc` is the default now; `--arms both` restores them. |
| `manifest.json already describes a run with different settings` | You changed `--shots`, `--no-lower` or `--visitor` while resuming into a directory that already holds results from the old settings. Keep the original values, or use a fresh `--output-dir`. |
| `shardXofN.json disagrees with shard0ofN.json` | The shards were not all launched with the same flags. Fix the command and re-run the odd one out; `--resume` makes that cheap. |
| `no bank manifest` / `no shot count` after a sharded run | You forgot `--merge-shards`. Stage C reads `manifest.json`, which only exists once the shard manifests are folded together. |
| Sharded run no faster than serial | `OMP_NUM_THREADS`/`MKL_NUM_THREADS` are unset, so every shard is trying to use every core. Export both as 1. |

## 5. Technical Details

- **Result Format**: Results are stored as raw `float64` binary arrays (`numpy.tofile`).
- **Bit Order**: The pipeline implements LSB (Least Significant Bit) convention where qubit $i$ corresponds to bit $i$. XACC reports measurement bitstrings with qubit 0 *leftmost*, so the backend reverses them; `--check-environment` runs an `x q[0]` probe that verifies this, and `--no-reverse-bits` overrides it if a build differs.
- **QASM Compiler**: XACC must parse the bank with `staq` (its OpenQASM 2.0 front end), *not* `xasm`. `xasm` is XACC's own DSL and is always registered, so the backend probe-compiles each candidate rather than trusting the service lookup. A `mismatched input '// ...' expecting {'__qpu__', '['}` error means `xasm` was selected.
- **Gate Lowering**: The bounding bank writes unlowered proxysim IR, which includes `sx`, `sxdg` and (at `--theta-zz != 0`) `cp` — none of which standard `qelib1.inc` declares, so staq rejects them. The backend therefore rewrites each circuit into the qelib1 subset before compiling (`sx` → `rx(pi/2)`, `s` → `rz(pi/2)`, `cp` → `cu1`, and so on, each exact up to a global phase), via `proxysim.c3pq.lower_for_c3pq`. `--no-lower` hands XACC the file as written. This requires **qiskit**, which `circuit_from_qasm` uses to read the bank — it is a declared dependency (`requirements.txt`) but is easy to miss in a hand-built venv, and without it the backend warns once and falls back to the raw text, after which every circuit containing `sx` fails to compile. Confirm with `python -c "import qiskit"` before a long run.
- **Shot Noise**: ExaTN samples, where C-3PQ returns exact probabilities. `exatn_analyze.py` therefore replaces `c3pq_analyze.py`'s exact tolerances (1e-9, 1e-12) with statistical ones derived from `--shots` and from each group's effective sample count (`shots x K`, since a group averages K trajectories), and prints each noise floor next to the number it governs. The TVD is biased *upward* by roughly `sum_i sqrt(p_i(1-p_i)/(2 pi S))`; that bias is reported per point and stored in the `.npz` but never subtracted, since the QCAP bound is an upper bound and deflating the measurement would manufacture agreement.
- **Compatibility**: Output files are 100% compatible with `proxysim.c3pq.read_probs`.
