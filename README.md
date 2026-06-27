# proxysim: multi-backend proxy-circuit simulator

Run the **same** quantum circuit on three completely different classical
simulators and compare them - the deterministic output distribution and the time
each method takes.

| backend | engine | scales to | restriction |
|---|---|---|---|
| `tensornetwork(quimb-MPS)` | matrix-product state, `swap+split` contraction | bounded-entanglement (shallow / structured) | none (any gate) |
| `statevector(qiskit)` | exact dense statevector | ~24 qubits (2ᴺ memory) | none (any gate) |
| `stabilizer(stim)` | stabilizer tableau (exact) / native CHP sampler | exact ~24 q; **sampling: thousands** | **Clifford only** |

Motivation: Merkel et al., *"When Clifford benchmarks are sufficient"*
([arXiv:2503.05943](https://arxiv.org/abs/2503.05943)) — Clifford **proxy**
circuits are efficiently simulable (stabilizer); their non-Clifford **targets**
need tensor-network / statevector methods. proxysim lets you run both and check
they line up.

As noise hasn't been inserted yet (these are preliminary runs), these circuits are noiseless and unitary, so the output is a *deterministic*
distribution `P(x) = |⟨x|ψ⟩|²`. proxysim computes that **exactly, once**, per
backend — it does **not** Monte-Carlo shots by re-simulating per shot (that would
only add sampling noise to an answer we can get exactly; quimb's per-shot
sampler, for instance, re-contracts the network every shot). But there is facility to do this as this is the natural way to proceed.

The three methods agree to floating-point precision — that cross-method
agreement (TVD ≈ 1e-15) is the headline check:

```
backend                     exact (ms)  support    TVD@ref
tensornetwork(quimb-MPS)       123.779       16    7.4e-16
statevector(qiskit)              5.141       16    0.0e+00
stabilizer(stim)                 9.021       16    1.1e-15
```

**Finite-shot sampling is optional** (`run_all(circuit, shots=N)`): it resamples
the exact distribution to emulate finite hardware statistics. Shots become
*essential* only once **noise** is added (a mixed state must be sampled /
averaged over trajectories) — see [proxysim/noise.py](proxysim/noise.py). That is
the regime this is building toward (the PTA setting of the Merkel paper).

## Layout

```
bounding-performance-qc-repo/        # repo root (pip install -e . here)
├── pyproject.toml / requirements.txt / LICENSE / .gitignore
├── proxysim/                       # the installable package
│   ├── circuit.py                  # backend-agnostic IR + lnn_brickwork() builder
│   ├── runner.py                   # run_all(): exact-distribution comparison
│   ├── viz.py                      # qiskit + quimb drawings, distribution bars
│   ├── parallel.py                 # parallel-sampling scaffold (for future noise traj.)
│   ├── noise.py                    # noise roadmap STUB (where shots matter)
│   └── backends/
│       ├── base.py                 # Backend ABC, SimResult, exact/sample split
│       ├── tensornetwork.py        # quimb CircuitMPS (swap+split)
│       ├── statevector.py          # qiskit Statevector
│       ├── stabilizer.py           # stim (Clifford only)
│       └── gpu.py                  # GPU backends STUB (cuStateVec / cuTensorNet)
├── examples/
│   ├── run_5q_lnn.py               # preliminary 5-qubit MPS sim
│   ├── run_compare.py              # all backends, exact agreement
│   ├── run_depth_sweep.py          # depth -> entanglement / entropy / cost
│   ├── run_scaling.py              # exact-distribution cost vs N; stim sampling to 1000 q
│   └── visualize.py                # generate the images below
└── results/                        # generated outputs (gitignored)
```

## Install (portable - no hard-coded environment)

```bash
git clone <your-repo-url> && cd bounding-performance-qc
python -m venv .venv
# Windows:  .venv\Scripts\activate     Linux/macOS:  source .venv/bin/activate
pip install -e .            # core deps incl. quimb, qiskit, stim, matplotlib, pylatexenc
```

A plain `pip install -e .` gives a fully working install, **including the
visualization**. Optional: `pip install -e ".[fast]"` adds `kahypar`/`optuna`
for better quimb contraction paths.

> Developed against an existing `venv_quantum` (Python 3.12: quimb 1.12, qiskit
> 2.4, stim 1.15). Any fresh venv from the steps above works the same.

## Run

```bash
python examples/visualize.py        # circuit + tensor-network + distribution images
python examples/run_5q_lnn.py       # preliminary MPS sim
python examples/run_compare.py      # exact agreement across all 3 backends
python examples/run_depth_sweep.py  # how depth drives entanglement/scrambling
python examples/run_scaling.py      # cost vs N; stim sampling to 1000 qubits
```

## Library use

```python
from proxysim import Circuit, lnn_brickwork, run_all

c = lnn_brickwork(n=5, n_cycles=4, twoq="cz", mode="clifford")  # or mode="haar"
report = run_all(c)                 # exact distributions + agreement check
print(report.to_text())
report = run_all(c, shots=8192)     # also draw a finite hardware-style sample

# build any circuit from the IR:
c = Circuit(3); c.h(0).cx(0, 1).cz(1, 2)
```

## The circuit - `lnn_brickwork`

Linear-nearest-neighbour fully-entangling **brickwork** (Merkel et al. Fig. 3):

```
[initial Hadamard layer]
repeat n_cycles times:
    even-odd 2q layer :  (0,1) (2,3) ...        # CZ or CX
    single-qubit layer
    odd-even 2q layer :  (1,2) (3,4) ...
    single-qubit layer
```

`n_cycles = n-1` fully entangles the chain. `mode="clifford"` (random single-qubit
Cliffords → all three backends) or `mode="haar"` (`Z(φ₁)·√X·Z(φ₂)·√X·Z(φ₃)`,
non-Clifford → statevector + MPS only, stim skipped).

## Visuals (`results/`)

`visualize.py` writes `viz_circuit_qiskit.png` (qiskit gate diagram),
`viz_tn_quimb.png` (quimb gate-level tensor network via native `draw`),
`viz_distribution.png` (exact P(x) per backend — they coincide), and a stitched
`viz_panel.png`.

## What the results show

- **Exact agreement** (`run_compare`): MPS contraction, dense statevector, and
  stabilizer tableau give identical P(x) (TVD ≲ 1e-14). Clifford proxy → uniform
  over a 2ᵏ stabilizer coset; Haar target → non-uniform, stim skipped.
- **Depth** (`run_depth_sweep`): MPS max-bond and output entropy grow with depth
  then saturate; for Clifford brickwork the support is exactly 2^entropy.
- **Scaling** (`run_scaling`): the *full* distribution is 2ᴺ for everyone (all top
  out ~20–24 q); for low-entanglement circuits MPS gets there fastest. The
  genuinely scalable Clifford operation is **sampling** — stim does 1000 qubits /
  ~2000 entanglers in well under a second.

## Roadmap

- **Noise** ([noise.py](proxysim/noise.py)) - Pauli/depolarizing channels (the
  PTA setting). Makes the state mixed ⇒ finite-shot sampling / trajectories
  become necessary. stim samples noisy Clifford circuits natively at scale.
- **GPU** ([backends/gpu.py](proxysim/backends/gpu.py)) - cuStateVec
  (qiskit-aer-gpu / qulacs-gpu) and cuTensorNet (quimb `contract_backend='cupy'`).
- **Parallel** ([parallel.py](proxysim/parallel.py)) - toggle-able scaffold; the
  real payoff is parallel **noise trajectories** / **many randomizations**, not
  shots of a single noiseless circuit.
- **Multiple nodes** - Use MPI to perform large scale computations across nodes.

Additionally, implement cupy backends for quimb, consider PEPS?

Bitstring convention everywhere: string index `i` = qubit `i`, **qubit 0
leftmost** (the qiskit/stim backends convert their native little-endian for you).
