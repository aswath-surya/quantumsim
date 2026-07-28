# proxysim

A multi-backend quantum-circuit simulator and benchmarking toolkit. The same
backend-agnostic circuit runs on a tensor-network (quimb MPS, optionally on GPU),
an exact statevector (qiskit), or a stabilizer (stim) backend, plus a
Pauli-propagation expectation-value engine — and on top of that it can bound
circuit error from cycle benchmarking. It grew out of Merkel et al., ["When
Clifford benchmarks are sufficient"](https://arxiv.org/abs/2503.05943): Clifford
"proxy" circuits are cheap to simulate (stabilizer), while their non-Clifford
targets need tensor-network or statevector methods.

Package name: `proxysim`. Repository: `quantumsim`.

## One entry point

Everything is reachable from `simulate(circuit, simulator=..., output=...)`, which
dispatches over two axes:

| axis | values |
|---|---|
| `simulator` | `statevector` · `tensornetwork` · `stabilizer` · `pauliprop` · `auto` |
| `output` | `distribution` · `samples` · `survival` · `expectation` |

with flags `noise` (a `NoiseModel`), `gpu` (tensor network on cupy), `max_bond`
(MPS truncation), and `parallel` / `n_workers`. `auto` picks the stabilizer backend
for Clifford circuits and the tensor-network backend otherwise.

```python
from proxysim import bench_brickwork, even_pairs, odd_pairs, NoiseModel, simulate

c = bench_brickwork(12, 3, [even_pairs(12), odd_pairs(12)], oneq="clifford").mirror()
simulate(c, "stabilizer", "distribution")                 # exact |<x|psi>|^2
simulate(c, "stabilizer", "survival", noise=NoiseModel(True, p2=5e-3))   # P(0...0) under noise
simulate(c, "tensornetwork", "survival", noise=..., max_bond=8, parallel=True)  # MPS trajectories, fanned out
simulate(c, output="expectation", observable="Z___________")            # Pauli propagation
```

## Install

```bash
git clone https://github.com/aswath-surya/quantumsim.git && cd quantumsim
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
```

A plain `pip install -e .` pulls quimb, qiskit, stim, matplotlib, pylatexenc — the
full CPU toolkit including plots. Optional extras: `.[fast]` (kahypar/optuna for
better quimb paths), `.[pauliprop]` (juliacall for the PauliPropagation.jl wrapper).
GPU needs `cupy-cuda12x`; multi-worker BLAS pinning uses `threadpoolctl` (both
optional). Developed on Python 3.12 with quimb 1.12, qiskit 2.4, stim 1.15.

## What every file is

**`proxysim/` — the package**

| file | role |
|---|---|
| `__init__.py` | public API surface; re-exports the whole toolkit |
| `simulate.py` | **the dispatcher** — `simulate()`, `make_backend`, `auto_simulator`, `noisy_survival`, `noisy_tvd_vs_support` |
| `circuit.py` | backend-agnostic `Circuit`/`Gate` IR; builders (`lnn_brickwork`, `brickwork_magic`, `clifford_entropy_circuit`, `bench_brickwork`); `mirror()`/`inverse()`; `even_pairs`/`odd_pairs`; `GATE_SETS` |
| `metrics.py` | distribution metrics (`total_variation_distance`, `classical_fidelity`, `shannon_entropy`, `tvd_to_ideal_support`) and result I/O (`save_results`/`load_results`, npz) |
| `noise.py` | `NoiseModel` (one toggle, 4 Pauli channels: 2q/1q depolarizing, idle dephasing, readout); `to_stim_noisy` (native stim noise) and `sample_trajectory`/`apply_readout` (trajectories) |
| `parallel.py` | single-node parallelism: `pmap`, `parallel_trajectories`, `sample_parallel`, with per-worker BLAS/thread pinning |
| `runner.py` | `run_all()` — compares the exact output distribution across all backends (`ComparisonReport`) |
| `benchmarking.py` | synthetic cycle benchmarking (`cycle_benchmark` via stim), `readout_fidelity`, the `qcap_bound`, `randomly_compile` |
| `pauliprop.py` | `JuliaPauliPropagator` — wrapper around the real PauliPropagation.jl via juliacall; **expectation values only** |
| `pauliprop_validator.py` | `PauliPropagator` — Julia-free reference for the same algorithm (uses `stim.PauliString` as a Pauli-algebra engine); **expectation values only** |
| `viz.py` | matplotlib drawings: qiskit circuit, quimb tensor network, distribution bars, Pauli-prop coefficient panel |

**`proxysim/backends/`**

| file | role |
|---|---|
| `base.py` | `Backend` ABC + `SimResult`; the exact-distribution / optional-sampling split |
| `statevector.py` | qiskit `Statevector` backend — `exact_distribution`, `amplitude`, `prob0` |
| `tensornetwork.py` | quimb `CircuitMPS` (swap+split) — `exact_distribution`, `amplitude`/`prob0` (O(n·D²), no dense vector), `max_bond_of`, GPU via `gpu=True` |
| `stabilizer.py` | stim backend (Clifford only) — exact tableau readout + native CHP sampler |
| `gpu.py` | `GPUTensorNetworkBackend` (real: quimb + cupy), `GPUStatevectorBackend` (stub: cuStateVec / qulacs-gpu) |

**`examples/`** — each is a thin experiment script (circuits + metrics come from the package); each saves a figure and, where relevant, an `.npz` of its results

| file | what it does |
|---|---|
| `_bootstrap.py` | puts the repo root on `sys.path`; defines `RESULTS_DIR` |
| `run_5q_lnn.py` | the preliminary 5-qubit MPS run |
| `run_compare.py` | the three backends agreeing on the exact distribution |
| `run_depth_sweep.py` | how depth drives entanglement / entropy / cost |
| `run_scaling.py` | cost vs qubit count; stim sampling to 1000 qubits |
| `visualize.py` | the 4-panel figure (circuit + TN + distribution + Pauli-prop) |
| `run_pauliprop.py` | Pauli-propagation expectation values (wrapper vs validator) |
| `run_magic_coeffs.py` | magic vs the Pauli-coefficient distribution |
| `run_noise_compare.py` | noise: fidelity and time across simulators (GHZ) |
| `run_bounding.py` | n=4 cycle-benchmark bound vs measured TVD (random + structured) |
| `run_bounding_clifford.py` | 20-qubit Clifford mirror circuits: bound vs error |
| `run_bound_vs_entropy.py` | bound looseness vs output Shannon entropy (fixed depth) |
| `run_bound_vs_cycles.py` | bound vs TVD vs number of cycles at fixed entropy |
| `run_qpe_bounding.py` | real MQT QPE circuit (`qpe11.qasm`): CB bound vs full-distribution TVD, swept over noise strength |
| `run_qft_bounding.py` | real MQT QFT circuit (`qft14.qasm`, non-Clifford): per-2q-gate dressed cycles, CB via Clifford proxy, summed-infidelity bound; shows the bound going vacuous when the ideal is a noise fixed point |
| `run_qec_vs_random.py` | bound vs actual performance for two families as noise grows, all on one shared log axis: a random moderate-entropy circuit (bound vs output-distribution TVD) and a rotated surface code (AKN bound vs the actual full-record TVD vs the pymatching-decoded LER). The bound is *tight* on the full-record TVD in both cases; the surface code's orders-of-magnitude drop to the LER is entirely decoding, not bound looseness |
| `run_parallel_hpc.py` | sample single-node parallel run (trajectories fanned across cores) |

`qpe11.qasm` is the 12-qubit MQT-Bench QPE circuit loaded via
`proxysim.circuit.circuit_from_qasm`. Its ideal output is a delta at the correct
phase, so the full-distribution TVD collapses to `1 - P(peak)`; the bound is
correspondingly tight (a delta ideal is maximally far from uniform). The ideal is
read exactly off the MPS backend (bond 1 — a product state), but the noisy sweep
runs on the statevector backend: at n=12 QPE's all-to-all inverse-QFT is SWAP-bound
and MPS is ~25x slower, since MPS only pays off for local, bounded-entanglement
circuits.

`qft14.qasm` (MQT QFT-14) is **non-Clifford** -- its `cp(pi/2^k)` gates are
controlled-phases that stim cannot simulate and that standard Clifford cycle
benchmarking cannot be run on directly (a Pauli conjugated through `cp` is a sum of
Paulis, not one Pauli). `run_qft_bounding.py` handles this with the **Clifford
proxy** (Merkel et al. 2503.05943): each 2q gate is its own dressed cycle
(single-qubit layer, which absorbs the Hadamards + Pauli twirl, plus the entangling
gate plus idle dephasing), and CB benchmarks the *noise* that gate carries via a
Clifford entangler (CZ = `cp(pi)`); under gate-independent Pauli noise the proxy
`e_F` equals the real gate's. The example also illustrates the looseness extreme:
`QFT|0...0> = |+>^n` is a product state whose uniform Z-basis readout is a
**Pauli-noise fixed point**, so the actual Z-basis TVD is identically 0 while the
bound climbs to 1 -- the opposite of QPE's tight delta-ideal bound. This is NOT "no
noise": the plot also shows the state infidelity `1 - <psi|rho|psi>` rising to ~0.5,
and an exact density-matrix simulation confirms the state goes nearly maximally mixed
(fidelity ~0.26, purity ~0.08 at p1=2e-2, p2=5e-2) while the TVD stays 0 to machine
precision. The readout basis is simply blind to the damage -- coherent (non-Pauli)
noise would show up; Pauli noise on `|+>^n` cannot.

**root**: `pyproject.toml`, `requirements.txt`, `LICENSE`, `.gitignore`; `docs/`
(committed figures for this README), `results/` (generated outputs, gitignored).

## Running on an HPC node (parallel)

The workhorse is `proxysim.parallel.pmap` — a parallel `map` over independent items
(noise trajectories, circuit instances, parameter points). The one detail that
matters: each worker pins its BLAS/OpenMP threads (default 1) so `n_workers`
processes don't each spawn a full thread pool and thrash the cores.

- statevector / tensor-network **trajectories** are CPU-bound and embarrassingly
  parallel → `n_workers = cores`, `threads_per_worker = 1`.
- one big contraction that already uses threaded BLAS → few workers, many threads.
- GPU tensor networks are single-device → one worker with `gpu=True`, not many
  CPU processes.

```python
simulate(circ, "tensornetwork", "survival", noise=noise, n_traj=4000,
         parallel=True, n_workers=None)          # None -> os.cpu_count()
```

`examples/run_parallel_hpc.py` is a sample run (same seeds serial vs parallel, so
the answers must match — that is the correctness check — while the wall-clock shows
the speedup). Process workers pickle the worker by reference, so the package must be
importable in the child (`pip install -e .`). On Linux the default `fork` start
method inherits everything cleanly; validate the multi-worker path on the target
node.

## GPU (tensor networks)

The tensor-network backend runs its contractions on the GPU by moving the MPS
tensors to cupy — the same code path, a different array backend:

```python
simulate(circ, "tensornetwork", "distribution", gpu=True)     # needs cupy + CUDA
# or:  from proxysim import TensorNetworkBackend; TensorNetworkBackend(gpu=True, max_bond=16)
```

Unlike statevector, tensor-network memory scales with the bond dimension, not `2^N`,
so GPU-TN is the path to genuinely larger systems when entanglement is bounded. A
dense GPU statevector (`GPUStatevectorBackend`) is a documented stub — it would buy
speed via cuStateVec/qulacs-gpu but not change the `2^N` wall.

## The science, briefly

**Exact by default.** These circuits are noiseless and unitary, so the output is
deterministic. `run_all` computes the exact distribution once per backend; the three
methods agree to ~1e-14. Shots are optional and only matter with noise.

**Pauli propagation** ([run_pauliprop.py](examples/run_pauliprop.py),
[run_magic_coeffs.py](examples/run_magic_coeffs.py)) evolves an *observable* backward
(Heisenberg picture) and returns `⟨ψ|O|ψ⟩`. Note it is an **expectation-value tool,
not a distribution tool**: a bitstring distribution `P(x)` is the Walsh–Hadamard
transform of all `2^n` diagonal-Pauli expectations, so you cannot read TVD off a few
`⟨Z_i⟩` (those give only the marginals). It is really the non-Clifford
generalization of stim's Pauli-frame tracking — a Clifford gate maps one Pauli to
one Pauli (so stim tracks a single Pauli, exactly and in poly time), whereas a
T/rotation splits it into a *sum* of Paulis (which Pauli propagation tracks, growing
~`2^t`, controlled by truncation). Accordingly `simulate()` sends `expectation` to
Pauli propagation and everything distributional (`distribution`/`survival`) to the
sampling backends.

**Noise + bounding.** `NoiseModel` adds Pauli-stochastic channels; cycle
benchmarking measures each entangling cycle's infidelity `e_F`, and the QCAP bound
`1 - F_RO·∏(1-e_F)^n` upper-bounds the circuit error (the AKN diamond-norm chain,
valid because the noise is Pauli). The bound is tight for low-entropy (mirror /
Loschmidt-echo) outputs and loosens as the output entropy grows.

![bounding vs entropy](docs/bound_vs_entropy.png)

## Convention

Bitstrings are written with qubit 0 on the left throughout; the qiskit and stim
backends convert their native little-endian for you.
