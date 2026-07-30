"""Per-2q-gate cycle benchmarking and QCAP bounds for MQT QFT-14.

The target circuit contains non-Clifford controlled-phase gates. Cycle
benchmarking is therefore performed using Clifford proxy cycles, while the
actual target circuit is simulated using statevector Monte-Carlo trajectories
under the generalized NoiseModel.

Important distinction
---------------------
For purely stochastic Pauli noise, QFT|0...0> has a uniform computational-basis
distribution that can remain invariant even while the quantum state fidelity
decreases substantially.

That statement does NOT generally hold once coherent errors, controlled-phase
offsets, crosstalk, or other non-Pauli errors are enabled. Under the generalized
noise model, the measured TVD may therefore become nonzero.

The script reports:

    1. QCAP multiplicative error bound,
    2. additive summed-infidelity bound,
    3. actual computational-basis TVD,
    4. actual state infidelity.

Run:
    python examples/run_qft_bounding.py
"""

from __future__ import annotations

import os
import random
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector

import _bootstrap  # noqa: F401

from proxysim import NoiseModel, save_results
from proxysim.backends import StatevectorBackend, TensorNetworkBackend
from proxysim.benchmarking import (
    cycle_benchmark,
    qcap_bound,
    readout_fidelity,
)
from proxysim.circuit import circuit_from_qasm
from proxysim.noise import sample_trajectory


QASM = os.path.join(
    os.path.dirname(__file__),
    "qft14.qasm",
)

# ---------------------------------------------------------------------------
# Generalized noise model
# ---------------------------------------------------------------------------
#
# The exact field names must agree with your current NoiseModel dataclass.
# The values below preserve approximately the old stochastic error scale while
# also demonstrating weak coherent and spectator errors.
#
BASE = NoiseModel(
    enabled=True,

    # baseline stochastic noise
    p1=2e-4,
    p2=1e-3,
    p_idle_z=2e-4,

    # asymmetric readout
    p_readout_01=1e-3,
    p_readout_10=1e-3,

    # coherent errors
    oneq_overrotation=2e-4,
    oneq_axis="z",

    twoq_zz_overrotation=5e-4,
    controlled_phase_offset=2e-4,
    idle_z_angle=1e-4,

    # spectator/crosstalk
    spectator_z_probability=1e-4,
    spectator_z_angle=1e-4,

    # optional drift
    stochastic_rate_drift_std=0.0,
    coherent_angle_drift_std=0.0,
)

SCALES = [0.5, 1.0, 2.0, 4.0, 8.0]
DEPTHS = [1, 2, 4, 8]

CB_DECAYS = 15
CB_TRAJECTORIES = 16
ACTUAL_TRAJECTORIES = 120
READOUT_SHOTS = 4000

OUT = _bootstrap.RESULTS_DIR + "/qft_bounding.png"
OUT_DATA = _bootstrap.RESULTS_DIR + "/qft_bounding_data.npz"

_SV = StatevectorBackend()


# ---------------------------------------------------------------------------
# Statevector and distribution helpers
# ---------------------------------------------------------------------------
def statevector(circuit) -> np.ndarray:
    """Return the exact statevector produced by a proxysim circuit."""
    return np.asarray(
        Statevector(_SV._build(circuit)).data,
        dtype=complex,
    )


def normalized_probabilities(psi: np.ndarray) -> np.ndarray:
    """Convert a statevector into a normalized dense probability vector."""
    probability = np.abs(psi) ** 2
    total = float(np.sum(probability))

    if total <= 0.0:
        raise ValueError("Statevector produced zero probability mass.")

    return probability / total


def readout_rates(noise) -> tuple[float, float]:
    """Return P(1|0) and P(0|1), supporting old and new noise APIs."""
    if not noise.enabled:
        return 0.0, 0.0

    p01 = float(
        getattr(
            noise,
            "p_readout_01",
            getattr(noise, "p_readout", 0.0),
        )
    )

    p10 = float(
        getattr(
            noise,
            "p_readout_10",
            getattr(noise, "p_readout", 0.0),
        )
    )

    return p01, p10


def apply_readout_channel(
    distribution: np.ndarray,
    n_qubits: int,
    noise,
) -> np.ndarray:
    """Apply independent asymmetric classical readout error exactly.

    The assignment matrix is

        columns: true bit
        rows:    measured bit

              true 0       true 1
        0   [ 1-p01          p10 ]
        1   [   p01        1-p10 ]
    """
    p01, p10 = readout_rates(noise)

    if p01 <= 0.0 and p10 <= 0.0:
        return np.asarray(distribution, dtype=float)

    assignment = np.array(
        [
            [1.0 - p01, p10],
            [p01, 1.0 - p10],
        ],
        dtype=float,
    )

    tensor = np.asarray(
        distribution,
        dtype=float,
    ).reshape((2,) * n_qubits)

    for axis in range(n_qubits):
        tensor = np.tensordot(
            assignment,
            tensor,
            axes=([1], [axis]),
        )
        tensor = np.moveaxis(tensor, 0, axis)

    result = tensor.reshape(-1)
    result = np.clip(result, 0.0, None)

    total = float(np.sum(result))
    if total > 0.0:
        result /= total

    return result


# ---------------------------------------------------------------------------
# Cycle benchmarking
# ---------------------------------------------------------------------------
def cycle_ef(
    pair,
    n,
    noise,
    twoq,
    seed,
    n_decays=CB_DECAYS,
):
    """Estimate e_F for a dressed Clifford proxy cycle.

    ``twoq`` should be a Clifford proxy supported by the benchmarking circuit
    generator, such as CZ or SWAP.
    """
    return cycle_benchmark(
        [pair],
        n,
        DEPTHS,
        noise,
        twoq=twoq,
        n_decays=n_decays,
        shots=800,
        n_trajectories=CB_TRAJECTORIES,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Actual target-circuit metrics
# ---------------------------------------------------------------------------
def actual_tvd_and_infidelity(
    circuit,
    noise,
    ideal_amplitude,
    ideal_distribution,
    n_trajectories,
    seed,
):
    """Estimate output TVD and state infidelity from the same trajectories.

    TVD includes the classical readout channel.

    State fidelity excludes readout error because readout is not a quantum
    channel acting on the premeasurement state.
    """
    rng = random.Random(seed)

    dimension = 2**circuit.n_qubits
    average_distribution = np.zeros(
        dimension,
        dtype=float,
    )

    average_fidelity = 0.0

    for _ in range(n_trajectories):
        noisy_circuit = sample_trajectory(
            circuit,
            noise,
            rng,
        )

        noisy_amplitude = statevector(noisy_circuit)

        average_distribution += normalized_probabilities(
            noisy_amplitude
        )

        average_fidelity += float(
            abs(
                np.vdot(
                    ideal_amplitude,
                    noisy_amplitude,
                )
            )
            ** 2
        )

    average_distribution /= n_trajectories
    average_fidelity /= n_trajectories

    measured_distribution = apply_readout_channel(
        average_distribution,
        circuit.n_qubits,
        noise,
    )

    tvd = 0.5 * float(
        np.sum(
            np.abs(
                measured_distribution
                - ideal_distribution
            )
        )
    )

    state_infidelity = 1.0 - average_fidelity

    return tvd, state_infidelity


# ---------------------------------------------------------------------------
# Proxy grouping
# ---------------------------------------------------------------------------
def classify_two_qubit_gate(gate) -> str:
    """Map a target 2Q gate to its Clifford proxy class."""
    gate_name = gate.name.lower()

    if gate_name == "cp":
        return "cp_proxy"

    if gate_name == "swap":
        return "swap_proxy"

    raise ValueError(
        f"No proxy class defined for two-qubit gate '{gate.name}'."
    )


def proxy_specification(proxy_class: str):
    """Return proxy entangler and representative pair."""
    if proxy_class == "cp_proxy":
        return {
            "twoq": "CZ",
            "label": "controlled-phase gates via CZ proxy",
        }

    if proxy_class == "swap_proxy":
        return {
            "twoq": "SWAP",
            "label": "SWAP gates via SWAP proxy",
        }

    raise ValueError(
        f"Unknown proxy class '{proxy_class}'."
    )


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    with open(QASM, "r", encoding="utf-8") as handle:
        circuit, measured = circuit_from_qasm(
            handle.read()
        )

    n = circuit.n_qubits

    two_qubit_gates = [
        gate
        for gate in circuit.gates
        if len(gate.qubits) == 2
    ]

    proxy_counts = Counter(
        classify_two_qubit_gate(gate)
        for gate in two_qubit_gates
    )

    n_cp = sum(
        gate.name.lower() == "cp"
        for gate in two_qubit_gates
    )

    n_swap = sum(
        gate.name.lower() == "swap"
        for gate in two_qubit_gates
    )

    n_cycles = len(two_qubit_gates)

    print(circuit.summary())

    print(
        f"non-Clifford: {not circuit.is_clifford}"
        f"  |  cycles = {n_cycles}"
        f"  ({n_cp} cp + {n_swap} swap)"
    )

    print(
        "proxy classes: "
        + ", ".join(
            f"{name}={count}"
            for name, count in proxy_counts.items()
        )
    )

    # ------------------------------------------------------------------
    # Ideal target
    # ------------------------------------------------------------------
    ideal_amplitude = statevector(circuit)
    ideal_distribution = normalized_probabilities(
        ideal_amplitude
    )

    support_size = int(
        np.count_nonzero(
            ideal_distribution > 1e-12
        )
    )

    uniform = np.full(
        2**n,
        1.0 / 2**n,
        dtype=float,
    )

    ideal_uniform_tvd = 0.5 * float(
        np.sum(
            np.abs(
                ideal_distribution - uniform
            )
        )
    )

    print(
        f"ideal support: {support_size} outcomes"
        f"  |  max probability = "
        f"{ideal_distribution.max():.3e}"
    )

    print(
        f"ideal TVD to uniform = "
        f"{ideal_uniform_tvd:.3e}"
    )

    print(
        "TN max bond = "
        f"{TensorNetworkBackend().max_bond_of(circuit)}\n"
    )

    # ------------------------------------------------------------------
    # Demonstrate whether proxy errors are pair independent
    # ------------------------------------------------------------------
    noise_scale_one = BASE.scaled(1.0)

    demo = [
        (
            "CZ proxy, pair (0,1)",
            cycle_ef(
                (0, 1),
                n,
                noise_scale_one,
                "CZ",
                seed=11,
                n_decays=40,
            )["e_F"],
        ),
        (
            "CZ proxy, pair (6,7)",
            cycle_ef(
                (6, 7),
                n,
                noise_scale_one,
                "CZ",
                seed=12,
                n_decays=40,
            )["e_F"],
        ),
        (
            "SWAP proxy, pair (0,13)",
            cycle_ef(
                (0, 13),
                n,
                noise_scale_one,
                "SWAP",
                seed=13,
                n_decays=40,
            )["e_F"],
        ),
    ]

    print(
        "proxy check:"
    )

    for label, process_infidelity in demo:
        print(
            f"    e_F = {process_infidelity:.5f}"
            f"   {label}"
        )

    print(
        "\nUnder a gate- or pair-dependent noise model, "
        "these values are not expected to agree exactly.\n"
    )

    # ------------------------------------------------------------------
    # Noise sweep
    # ------------------------------------------------------------------
    qcap_bounds = []
    additive_bounds = []
    tvds = []
    state_infidelities = []

    cp_proxy_efs = []
    swap_proxy_efs = []

    print(
        f"{'scale':>7}"
        f"{'eF(CZ)':>11}"
        f"{'eF(SWAP)':>12}"
        f"{'QCAP':>11}"
        f"{'additive':>12}"
        f"{'TVD':>12}"
        f"{'state infid.':>15}"
    )

    for scale_index, scale in enumerate(SCALES):
        noise = BASE.scaled(scale)

        readout_fid, readout_std = readout_fidelity(
            n,
            noise,
            shots=READOUT_SHOTS,
            seed=300 + scale_index,
        )

        # Benchmark separate proxy classes.
        cp_cb = cycle_ef(
            (0, 1),
            n,
            noise,
            "CZ",
            seed=1000 + scale_index,
        )

        swap_cb = cycle_ef(
            (0, n - 1),
            n,
            noise,
            "SWAP",
            seed=2000 + scale_index,
        )

        cp_eF = cp_cb["e_F"]
        cp_eF_std = cp_cb["e_F_std"]

        swap_eF = swap_cb["e_F"]
        swap_eF_std = swap_cb["e_F_std"]

        cycle_counts = {
            "cp_proxy": proxy_counts.get(
                "cp_proxy",
                0,
            ),
            "swap_proxy": proxy_counts.get(
                "swap_proxy",
                0,
            ),
        }

        cycle_efs = {
            "cp_proxy": (
                cp_eF,
                cp_eF_std,
            ),
            "swap_proxy": (
                swap_eF,
                swap_eF_std,
            ),
        }

        qcap_result = qcap_bound(
            cycle_counts,
            cycle_efs,
            readout_fid,
            readout_std,
        )

        qcap_error = min(
            1.0,
            qcap_result["error"],
        )

        # Additive AKN-style chain.
        additive_error = (
            1.0
            - readout_fid
            + cycle_counts["cp_proxy"] * cp_eF
            + cycle_counts["swap_proxy"] * swap_eF
        )

        additive_error = min(
            1.0,
            additive_error,
        )

        tvd, state_infidelity = actual_tvd_and_infidelity(
            circuit,
            noise,
            ideal_amplitude,
            ideal_distribution,
            n_trajectories=ACTUAL_TRAJECTORIES,
            seed=4000 + scale_index,
        )

        qcap_bounds.append(qcap_error)
        additive_bounds.append(additive_error)
        tvds.append(tvd)
        state_infidelities.append(
            state_infidelity
        )

        cp_proxy_efs.append(cp_eF)
        swap_proxy_efs.append(swap_eF)

        save_results(
            OUT_DATA,
            scales=np.asarray(
                SCALES[: len(qcap_bounds)]
            ),
            cp_proxy_e_F=np.asarray(
                cp_proxy_efs
            ),
            swap_proxy_e_F=np.asarray(
                swap_proxy_efs
            ),
            qcap_bound=np.asarray(
                qcap_bounds
            ),
            additive_bound=np.asarray(
                additive_bounds
            ),
            tvd=np.asarray(tvds),
            state_infidelity=np.asarray(
                state_infidelities
            ),
            n_cycles=n_cycles,
            n_cp=n_cp,
            n_swap=n_swap,
        )

        print(
            f"{scale:>7.1f}"
            f"{cp_eF:>11.5f}"
            f"{swap_eF:>12.5f}"
            f"{qcap_error:>11.4f}"
            f"{additive_error:>12.4f}"
            f"{tvd:>12.4e}"
            f"{state_infidelity:>15.4f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    figure, axis = plt.subplots(
        figsize=(9.2, 5.8)
    )

    axis.axhline(
        1.0,
        color="0.6",
        linewidth=0.8,
        linestyle=":",
    )

    axis.plot(
        SCALES,
        qcap_bounds,
        "-",
        linewidth=2.5,
        label="multiplicative QCAP bound",
    )

    axis.plot(
        SCALES,
        additive_bounds,
        "--",
        linewidth=2.0,
        label="additive summed-infidelity bound",
    )

    axis.plot(
        SCALES,
        state_infidelities,
        "^-",
        linewidth=2.0,
        markersize=7,
        label=(
            r"actual state infidelity "
            r"$1-\langle\psi|\rho|\psi\rangle$"
        ),
    )

    axis.plot(
        SCALES,
        tvds,
        "o-",
        linewidth=2.0,
        markersize=7,
        label="actual computational-basis TVD",
    )

    axis.set_xlabel(
        "noise scale factor"
    )

    axis.set_ylabel(
        "error"
    )

    axis.set_title(
        "QFT-14 with generalized stochastic and coherent noise\n"
        "Clifford proxy CB bounds versus actual output TVD and state infidelity",
        fontsize=10.5,
    )

    axis.set_ylim(
        -0.03,
        1.1,
    )

    axis.legend(
        frameon=False,
        fontsize=9.0,
        loc="best",
    )

    axis.grid(
        True,
        alpha=0.15,
    )

    figure.tight_layout()
    figure.savefig(
        OUT,
        dpi=150,
    )

    print(
        f"\nWrote {OUT}, {OUT_DATA}"
    )


if __name__ == "__main__":
    main()

# """Per-2q-gate cycle benchmarking + summed-infidelity bound for a non-Clifford
# circuit: MQT QFT-14 (qft14.qasm).

# Cycle definition (different from run_qpe_bounding's ASAP layers): EACH individual
# two-qubit gate is its own dressed cycle, so there are as many cycles as 2q gates
# (98 here: 91 cp + 7 swap). The bound is the summed process infidelity of those
# cycles (the additive AKN chain, TVD <= readout_err + sum_c e_F), alongside the
# tighter multiplicative QCAP form 1 - F_RO * prod_c (1 - e_F).

# Non-Clifford handling -- the important part:
#   The cp(pi/2^k) gates are NON-Clifford, so stim cannot simulate them and standard
#   (Clifford) cycle benchmarking cannot be run on them directly: a Pauli conjugated
#   through cp is a SUM of Paulis, not one Pauli. The fix is the Merkel "Clifford
#   proxy" (2503.05943): benchmark the NOISE the gate carries using a Clifford
#   entangler (CZ = cp(pi)) as the carrier. Under a gate-independent Pauli noise
#   model, e_F is a property of the error channel, not the gate angle, so the proxy
#   e_F equals the real one. The tensor network is used for the actual (non-Clifford)
#   simulation; CB itself runs on the Clifford proxy in stim.

# A striking consequence for THIS circuit: QFT|0...0> = |+>^14, a product state whose
# Z-basis readout is uniform -- and uniform is a fixed point of Pauli noise here
# (depolarizing -> uniform; dephasing is Z-diagonal; bit-flip readout permutes
# uniform -> uniform). So the actual Z-basis TVD is identically 0 no matter how strong
# the noise. This is NOT "no noise": the state infidelity 1 - <psi|rho|psi> rises the
# whole time (plotted here), and an exact density-matrix simulation confirms that at
# p1=2e-2, p2=5e-2 the state fidelity falls to ~0.26 and the purity to ~0.08 (nearly
# maximally mixed) while TVD-to-uniform stays 0 to machine precision. The reason is
# structural: QFT is built only from Clifford non-diagonal gates (H, swap) and
# non-Clifford DIAGONAL gates (cp), so a Pauli error commuted to the output becomes
# (Pauli) x (diagonal unitary) acting on |+>^14 -- both preserve the uniform amplitude
# magnitudes, hence the uniform Z-distribution. Coherent (non-Pauli) noise WOULD show
# up; Pauli noise cannot. The TVD is the maximally-loose corner (opposite of QPE's
# delta ideal) precisely because the readout basis is blind to the damage.

# Run:  python examples/run_qft_bounding.py
# """

# from __future__ import annotations

# import dataclasses
# import os
# import random
# import warnings

# warnings.filterwarnings("ignore")

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import numpy as np
# from qiskit.quantum_info import Statevector

# import _bootstrap  # noqa: F401
# from proxysim import NoiseModel, save_results
# from proxysim.circuit import circuit_from_qasm
# from proxysim.noise import sample_trajectory
# from proxysim.backends import StatevectorBackend, TensorNetworkBackend
# from proxysim.benchmarking import cycle_benchmark, qcap_bound, readout_fidelity

# QASM = os.path.join(os.path.dirname(__file__), "qft14.qasm")
# BASE = NoiseModel(enabled=True, p1=2e-4, p2=1e-3, p_readout=1e-3, p_idle=2e-4)
# SCALES = [0.5, 1.0, 2.0, 4.0, 8.0]
# DEPTHS = [1, 2, 4, 8]
# OUT = _bootstrap.RESULTS_DIR + "/qft_bounding.png"
# OUT_DATA = _bootstrap.RESULTS_DIR + "/qft_bounding_data.npz"

# _SV = StatevectorBackend()


# def cycle_ef(pair, n, noise, twoq, seed, n_decays=15):
#     """process infidelity e_F of the single-2q-gate dressed cycle on `pair`,
#     benchmarked with a Clifford proxy entangler (`twoq`)."""
#     cb = cycle_benchmark([pair], n, DEPTHS, noise, twoq=twoq,
#                          n_decays=n_decays, shots=800, seed=seed)
#     return cb


# def actual_tvd_and_infidelity(circ, noise, ideal_amp, n_traj, seed):
#     """Trajectory-averaged over the SAME noisy trajectories, two quantities:
#       * TVD of the noisy Z-basis distribution to the uniform ideal -- provably 0 for
#         QFT|0>, and independently confirmed by an exact density-matrix simulation.
#       * state infidelity 1 - <ideal|rho|ideal> = 1 - mean_traj |<ideal|psi_traj>|^2,
#         which does NOT vanish -- it shows the noise really is corrupting the state.
#     The gap between them is the whole point: the noise damages the state (infidelity
#     up) but a Z-basis readout of |+>^n is a fixed point of Pauli noise (TVD == 0)."""
#     rng = random.Random(seed)
#     N = 2 ** circ.n_qubits
#     dist = np.zeros(N)
#     fid = 0.0
#     for _ in range(n_traj):
#         psi = np.asarray(Statevector(_SV._build(sample_trajectory(circ, noise, rng))).data)
#         dist += np.abs(psi) ** 2
#         fid += abs(np.vdot(ideal_amp, psi)) ** 2
#     dist /= n_traj
#     return 0.5 * np.abs(dist - 1.0 / N).sum(), 1.0 - fid / n_traj


# def main():
#     circ, measured = circuit_from_qasm(open(QASM).read())
#     n = circ.n_qubits
#     two_qubit_gates = [g for g in circ.gates if len(g.qubits) == 2]
#     n_cp = sum(1 for g in two_qubit_gates if g.name == "cp")
#     n_swap = sum(1 for g in two_qubit_gates if g.name == "swap")
#     n_cycles = len(two_qubit_gates)
#     n_h = sum(1 for g in circ.gates if len(g.qubits) == 1)
#     print(circ.summary())
#     print(f"non-Clifford: {not circ.is_clifford}  |  cycles (one per 2q gate) = "
#           f"{n_cycles}  ({n_cp} cp + {n_swap} swap)")
#     print(f"each cycle is DRESSED: [single-qubit layer on all {n} qubits, p1 noise -- "
#           f"this is where the {n_h} Hadamards + Pauli twirl live] + [entangling gate, "
#           f"p2] + [idle dephasing on the {n-2} spectators]")

#     # ideal = |+>^n (product state -> uniform readout)
#     p_ideal = np.asarray(Statevector(_SV._build(circ)).probabilities())
#     print(f"ideal: {int((p_ideal > 1e-12).sum())} outcomes, all = {p_ideal.max():.3e} "
#           f"(uniform 1/2^{n})   TN max bond = {TensorNetworkBackend().max_bond_of(circ)}\n")

#     # --- demonstrate the proxy is gate/pair-independent (Merkel) at scale 1 -----
#     noise1 = BASE.scaled(1.0)
#     demo = [
#         ("CZ-proxy, pair (0,1)", cycle_ef((0, 1), n, noise1, "CZ", 11, n_decays=60)["e_F"]),
#         ("CZ-proxy, pair (6,7)", cycle_ef((6, 7), n, noise1, "CZ", 12, n_decays=60)["e_F"]),
#         ("SWAP-proxy, pair (0,13)", cycle_ef((0, 13), n, noise1, "SWAP", 13, n_decays=60)["e_F"]),
#     ]
#     print("proxy check (should agree -- e_F is the noise channel, not the gate):")
#     for label, ef in demo:
#         print(f"    e_F = {ef:.5f}   {label}")
#     print()

#     # --- sweep noise strength: bound vs actual Z-TVD vs actual state infidelity --
#     ideal_amp = np.asarray(Statevector(_SV._build(circ)).data)
#     prod_bounds, tvds, infids, efs = [], [], [], []
#     print(f"{'scale':>6}{'e_F/cycle':>11}{'QCAP bound':>12}{'actual TVD':>12}"
#           f"{'state infidelity':>18}")
#     for s in SCALES:
#         noise = BASE.scaled(s)
#         ro_fid, ro_std = readout_fidelity(n, noise, seed=3)
#         cb = cycle_ef((0, 1), n, noise, "CZ", seed=100)
#         eF, eF_std = cb["e_F"], cb["e_F_std"]
#         # every single-2q-gate cycle has identical structure -> identical e_F
#         counts = {f"c{i}": 1 for i in range(n_cycles)}
#         cyc_efs = {f"c{i}": (eF, eF_std) for i in range(n_cycles)}
#         prod = qcap_bound(counts, cyc_efs, ro_fid, ro_std)["error"]
#         tvd, infid = actual_tvd_and_infidelity(circ, noise, ideal_amp, n_traj=120, seed=7)
#         prod_bounds.append(prod); tvds.append(tvd); infids.append(infid); efs.append(eF)
#         save_results(OUT_DATA, scales=np.array(SCALES[:len(efs)]),
#                      e_F=np.array(efs), qcap_bound=np.array(prod_bounds),
#                      tvd=np.array(tvds), state_infidelity=np.array(infids),
#                      n_cycles=n_cycles)
#         print(f"{s:>6.1f}{eF:>11.5f}{prod:>12.4f}{tvd:>12.2e}{infid:>18.4f}", flush=True)

#     fig, ax = plt.subplots(figsize=(9.0, 5.6))
#     ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
#     ax.plot(SCALES, prod_bounds, "-", color="#D55E00", lw=2.5, label="QCAP bound (on the TVD)")
#     ax.plot(SCALES, infids, "^-", color="#CC79A7", lw=2, ms=7,
#             label=r"actual state infidelity $1-\langle\psi|\rho|\psi\rangle$ (noise IS real)")
#     ax.plot(SCALES, tvds, "o-", color="#0072B2", ms=7, label="actual Z-basis TVD (= 0)")
#     ax.set_xlabel("noise scale factor")
#     ax.set_ylabel("probability of an error")
#     ax.set_title(f"QFT-14 (non-Clifford): the noise IS applied (state infidelity rises),\n"
#                  f"but a Z-readout of QFT|0> = |+>$^{{14}}$ is a Pauli-noise fixed point "
#                  f"so the TVD == 0", fontsize=10.5)
#     ax.set_ylim(-0.03, 1.1)
#     ax.annotate("state genuinely corrupted:\nfidelity falls, purity -> mixed\n"
#                 "(confirmed by exact density matrix)",
#                 xy=(8.0, infids[-1]), xytext=(2.2, 0.72), fontsize=8.3, color="#8E4A78",
#                 arrowprops=dict(arrowstyle="->", color="#8E4A78", lw=0.8))
#     ax.annotate("Z-basis TVD is ZERO at every noise level:\n"
#                 r"|+>$^{14}$ has uniform readout, a Pauli-noise fixed point",
#                 xy=(4.0, 0.0), xytext=(1.7, 0.14), fontsize=8.3, color="#0072B2",
#                 arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.8))
#     ax.legend(frameon=False, fontsize=9.0, loc="center left")
#     ax.grid(True, alpha=0.15)
#     fig.tight_layout()
#     fig.savefig(OUT, dpi=150)
#     print(f"\nWrote {OUT}, {OUT_DATA}")


# if __name__ == "__main__":
#     main()
