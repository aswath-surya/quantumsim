"""Per-2q-gate cycle benchmarking + error bounds versus applied cycle count
for a non-Clifford QASM circuit: MQT QFT-14 (qft14.qasm).

This keeps the logistical setup of the original fixed-QFT noise-scaling script:

  * load one QASM circuit;
  * treat each individual two-qubit gate as one dressed cycle;
  * estimate one per-cycle process infidelity e_F using a Clifford proxy;
  * measure readout fidelity once at a fixed NoiseModel;
  * simulate the actual non-Clifford target circuit with statevector trajectories;
  * compare the measured Z-basis TVD and state infidelity with
      - the additive summed-infidelity bound, and
      - the multiplicative QCAP bound.

The sweep variable is now the ACTUAL NUMBER OF APPLIED TWO-QUBIT CYCLES rather
than a noise scale or a full-circuit repetition count.  For target C, the script
forms complete repetitions of the QASM circuit plus the shortest gate-stream
prefix that ends immediately after the C-th two-qubit gate.

Important modeling assumption
-----------------------------
The QFT controlled-phase gates are non-Clifford, so standard Clifford CB cannot
benchmark them directly.  The script uses a CZ Clifford proxy and assumes the
synthetic NoiseModel is gate independent, so e_F is a property of the attached
error channel rather than the controlled-phase angle.  This is a proxy-CB/QCAP
analysis, not direct CB of the non-Clifford controlled-phase gates.

Run:
    python examples/run_qft_cycle_sweep.py
"""

from __future__ import annotations

import dataclasses
import os
import random
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector

import _bootstrap  # noqa: F401
from proxysim import NoiseModel, save_results
from proxysim.circuit import circuit_from_qasm
from proxysim.noise import sample_trajectory
from proxysim.parallel import pmap
from proxysim.backends import StatevectorBackend, TensorNetworkBackend
from proxysim.benchmarking import cycle_benchmark, qcap_bound, readout_fidelity

QASM = os.path.join(os.path.dirname(__file__), "qft14.qasm")

# Fixed physical noise throughout the cycle-count sweep.
NOISE = NoiseModel(
    enabled=True,
    p1=2e-4,
    p2=1e-3,
    p_readout=1e-3,
    p_idle=2e-4,
)

# Sequence lengths used INSIDE cycle benchmarking.  These are unrelated to the
# target-circuit cycle counts below.
CB_DEPTHS = [1, 2, 4, 8]

# Exact numbers of individual two-qubit gates included in each target circuit.
# QFT-14 has 98 two-qubit gates, so values above 98 use complete QFT repetitions
# followed by a prefix of the next repetition.
TARGET_CYCLES = [8, 16, 24, 32, 48, 64, 80, 98, 128, 160, 196, 256, 294]

# Independent trajectory ensembles at each target cycle count.  Their mean and
# standard error are plotted.  These are estimator replicates of one target
# circuit, not different random circuit instances.
N_ENSEMBLES = 12
N_TRAJ = 160
SEED = 42

OUT = _bootstrap.RESULTS_DIR + "/qft_cycle_sweep.png"
OUT_DATA = _bootstrap.RESULTS_DIR + "/qft_cycle_sweep_data.npz"

_SV = StatevectorBackend()


def cycle_ef(args):
    """Benchmark one dressed Clifford-proxy cycle.

    Tuple-taking and top-level so it can be sent to process workers by pmap.
    """
    pair, n, noise, twoq, seed, n_decays = args
    return cycle_benchmark(
        [pair],
        n,
        CB_DEPTHS,
        noise,
        twoq=twoq,
        n_decays=n_decays,
        shots=800,
        seed=seed,
    )


def build_to_twoq_count(base_circ, target_cycles):
    """Return a circuit containing exactly ``target_cycles`` two-qubit gates.

    The operation stream is repeated as needed.  The final partial repetition is
    truncated immediately after its last requested two-qubit gate.  All one-qubit
    gates preceding that gate are retained, so the result is a valid prefix of the
    repeated QASM operation stream.

    This function assumes the proxysim Circuit object is a dataclass, as in the
    current codebase.  If Circuit later changes, replace the final
    ``dataclasses.replace`` with the corresponding circuit constructor/copy API.
    """
    if target_cycles < 1:
        raise ValueError("target_cycles must be positive")

    base_twoq = sum(len(g.qubits) == 2 for g in base_circ.gates)
    if base_twoq == 0:
        raise ValueError("QASM circuit contains no two-qubit gates")

    full_reps, remainder = divmod(target_cycles, base_twoq)
    gates = []

    for _ in range(full_reps):
        gates.extend(base_circ.gates)

    if remainder:
        seen = 0
        for gate in base_circ.gates:
            gates.append(gate)
            if len(gate.qubits) == 2:
                seen += 1
                if seen == remainder:
                    break

        if seen != remainder:
            raise RuntimeError(
                f"Could not construct a prefix with {remainder} two-qubit gates"
            )

    target = dataclasses.replace(base_circ, gates=list(gates))
    actual = sum(len(g.qubits) == 2 for g in target.gates)
    if actual != target_cycles:
        raise RuntimeError(
            f"cycle-count construction failed: requested {target_cycles}, got {actual}"
        )
    return target


def trajectory_metrics(args):
    """Estimate TVD and state infidelity from one trajectory ensemble."""
    circ, noise, ideal_amp, ideal_prob, n_traj, seed = args
    rng = random.Random(seed)
    dist = np.zeros_like(ideal_prob, dtype=float)
    fidelity = 0.0

    for _ in range(n_traj):
        noisy_circ = sample_trajectory(circ, noise, rng)
        psi = np.asarray(Statevector(_SV._build(noisy_circ)).data)
        dist += np.abs(psi) ** 2
        fidelity += abs(np.vdot(ideal_amp, psi)) ** 2

    dist /= n_traj
    fidelity /= n_traj

    measured_tvd = 0.5 * float(np.abs(dist - ideal_prob).sum())
    state_infidelity = 1.0 - float(fidelity)
    return measured_tvd, state_infidelity


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, 0.0
    sem = float(values.std(ddof=1) / np.sqrt(len(values)))
    return mean, sem


def main():
    with open(QASM) as f:
        base_circ, measured = circuit_from_qasm(f.read())

    n = base_circ.n_qubits
    twoq_gates = [g for g in base_circ.gates if len(g.qubits) == 2]
    n_cp = sum(g.name == "cp" for g in twoq_gates)
    n_swap = sum(g.name == "swap" for g in twoq_gates)
    cycles_per_rep = len(twoq_gates)

    print(base_circ.summary())
    print(
        f"non-Clifford: {not base_circ.is_clifford}  |  "
        f"cycles per full QASM repetition = {cycles_per_rep} "
        f"({n_cp} cp + {n_swap} swap)"
    )
    print(
        "cycle convention: one individual 2q gate = one dressed cycle; "
        "the sweep truncates only at 2q-gate boundaries"
    )
    print(
        f"fixed noise: p1={NOISE.p1:.2e}, p2={NOISE.p2:.2e}, "
        f"p_ro={NOISE.p_readout:.2e}, p_idle={NOISE.p_idle:.2e}"
    )
    print(
        f"TN max bond of one full QASM repetition: "
        f"{TensorNetworkBackend().max_bond_of(base_circ)}\n"
    )

    n_workers = max(1, (os.cpu_count() or 2) - 2)

    # Proxy-agreement diagnostics plus the higher-statistics CZ value used below.
    cb_jobs = [
        ((0, 1), n, NOISE, "CZ", 11, 40),
        ((6, 7), n, NOISE, "CZ", 12, 40),
        ((0, 13), n, NOISE, "SWAP", 13, 40),
        ((0, 1), n, NOISE, "CZ", 100, 25),
    ]
    cbs = pmap(cycle_ef, cb_jobs, n_workers=n_workers)

    labels = [
        "CZ proxy, pair (0,1)",
        "CZ proxy, pair (6,7)",
        "SWAP proxy, pair (0,13)",
    ]
    print("proxy check (agreement is expected for this homogeneous noise model):")
    for label, cb in zip(labels, cbs[:3]):
        print(f"    e_F = {cb['e_F']:.6f}   {label}")

    eF = float(cbs[-1]["e_F"])
    eF_std = float(cbs[-1]["e_F_std"])
    ro_fid, ro_std = readout_fidelity(n, NOISE, seed=3)

    print(
        f"\nvalue used in bounds: e_F={eF:.6f} ({eF_std:.6f}), "
        f"readout fidelity={ro_fid:.6f} ({ro_std:.6f})\n"
    )

    additive_bounds = []
    qcap_bounds = []
    qcap_stds = []
    tvd_means = []
    tvd_sems = []
    infid_means = []
    infid_sems = []
    total_gates = []
    supports = []

    header = (
        f"{'cycles':>8}{'gates':>9}{'support':>10}{'additive':>12}"
        f"{'QCAP':>12}{'TVD mean':>12}{'TVD SEM':>11}"
        f"{'infid mean':>13}{'infid SEM':>12}"
    )
    print(header)

    for target_cycles in TARGET_CYCLES:
        target = build_to_twoq_count(base_circ, target_cycles)
        total_gates.append(len(target.gates))

        ideal_amp = np.asarray(Statevector(_SV._build(target)).data)
        ideal_prob = np.abs(ideal_amp) ** 2
        support = int(np.count_nonzero(ideal_prob > 1e-12))
        supports.append(support)

        # All cycles share the same proxy e_F under the homogeneous-noise model.
        counts = {"twoq_proxy": target_cycles}
        cycle_efs = {"twoq_proxy": (eF, eF_std)}
        qcap = qcap_bound(counts, cycle_efs, ro_fid, ro_std)

        additive = min(1.0, (1.0 - ro_fid) + target_cycles * eF)
        additive_bounds.append(additive)
        qcap_bounds.append(float(qcap["error"]))
        qcap_stds.append(float(qcap["std"]))

        jobs = [
            (
                target,
                NOISE,
                ideal_amp,
                ideal_prob,
                N_TRAJ,
                SEED + 100_000 * j + target_cycles,
            )
            for j in range(N_ENSEMBLES)
        ]
        metrics = pmap(trajectory_metrics, jobs, n_workers=n_workers)
        tvd_values = [x[0] for x in metrics]
        infid_values = [x[1] for x in metrics]

        tvd_mean, tvd_sem = mean_sem(tvd_values)
        infid_mean, infid_sem = mean_sem(infid_values)
        tvd_means.append(tvd_mean)
        tvd_sems.append(tvd_sem)
        infid_means.append(infid_mean)
        infid_sems.append(infid_sem)

        print(
            f"{target_cycles:>8}{len(target.gates):>9}{support:>10}"
            f"{additive:>12.5f}{qcap['error']:>12.5f}"
            f"{tvd_mean:>12.5f}{tvd_sem:>11.5f}"
            f"{infid_mean:>13.5f}{infid_sem:>12.5f}",
            flush=True,
        )

    save_results(
        OUT_DATA,
        target_cycles=np.asarray(TARGET_CYCLES),
        total_gates=np.asarray(total_gates),
        ideal_support=np.asarray(supports),
        additive_bound=np.asarray(additive_bounds),
        qcap_bound=np.asarray(qcap_bounds),
        qcap_bound_std=np.asarray(qcap_stds),
        tvd_mean=np.asarray(tvd_means),
        tvd_sem=np.asarray(tvd_sems),
        state_infidelity_mean=np.asarray(infid_means),
        state_infidelity_sem=np.asarray(infid_sems),
        e_F=eF,
        e_F_std=eF_std,
        ro_fid=ro_fid,
        ro_fid_std=ro_std,
        cycles_per_full_repetition=cycles_per_rep,
        n_ensembles=N_ENSEMBLES,
        n_traj=N_TRAJ,
    )

    x = np.asarray(TARGET_CYCLES)
    qcap_values = np.asarray(qcap_bounds)
    qcap_sigma = np.asarray(qcap_stds)

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.axhline(1.0, lw=0.8, ls=":")

    ax.plot(
        x,
        additive_bounds,
        "--",
        lw=2.0,
        label=r"additive bound: $(1-F_{\rm RO})+N_{\rm cyc}e_F$",
    )
    ax.plot(
        x,
        qcap_values,
        "-",
        lw=2.5,
        label=r"QCAP: $1-F_{\rm RO}(1-e_F)^{N_{\rm cyc}}$",
    )
    ax.fill_between(
        x,
        np.maximum(0.0, qcap_values - 2.96 * qcap_sigma),
        np.minimum(1.0, qcap_values + 2.96 * qcap_sigma),
        alpha=0.2,
    )

    ax.errorbar(
        x,
        tvd_means,
        yerr=tvd_sems,
        fmt="o-",
        capsize=3,
        lw=1.6,
        ms=6,
        label="actual Z-basis TVD (mean ± SEM)",
    )
    ax.errorbar(
        x,
        infid_means,
        yerr=infid_sems,
        fmt="^-",
        capsize=3,
        lw=1.6,
        ms=6,
        label=r"actual state infidelity $1-\langle\psi|\rho|\psi\rangle$",
    )

    for multiple in range(cycles_per_rep, max(TARGET_CYCLES) + 1, cycles_per_rep):
        ax.axvline(multiple, lw=0.7, ls=":", alpha=0.35)

    ax.set_xlabel("number of applied two-qubit cycles")
    ax.set_ylabel("error probability")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(
        "QFT-14 proxy-CB bounds versus exact two-qubit cycle count\n"
        "fixed noise; complete QASM repetitions plus a final gate-stream prefix"
    )
    ax.grid(True, alpha=0.15)
    ax.legend(frameon=False, fontsize=8.7)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)

    print(f"\nWrote {OUT}")
    print(f"Wrote {OUT_DATA}")


if __name__ == "__main__":
    main()
