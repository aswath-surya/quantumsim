"""Explicit-RC QCAP bound and raw TVD scatter versus 2q-cycle depth for QFT-14.

For each complete-circuit repetition count d:

    logical target = U^d
    number of two-qubit cycles = d * cycles_per_repetition

Each individual two-qubit gate is treated as one cycle, together with its nearby
single-qubit operations. Every supported two-qubit cycle is independently dressed
by proxysim.rc.pauli_twirl.

Each plotted TVD point is produced by:

    1. Constructing U^d.
    2. Generating N_RC_REALIZATIONS independently dressed versions of U^d.
    3. Simulating N_TRAJ_PER_RC noisy trajectories for every dressed circuit.
    4. Treating the inserted RC Paulis as virtual frame changes:
           sample_trajectory(..., virtual=virtual_indices)
    5. Applying readout noise analytically to each trajectory-averaged distribution.
    6. Averaging the distributions over RC realizations.
    7. Computing TVD against the ideal distribution of U^d.

The QCAP bound is obtained from Clifford-proxy cycle benchmarking. Under the
homogeneous gate-independent synthetic noise model, every individual two-qubit
cycle is assigned the same proxy-derived process infidelity e_F.

Important:
    QFT-14 contains generic non-Clifford cp(theta) gates. The actual tailoring
    performed by pauli_twirl is determined by the compiling group supported by
    proxysim.rc for those gates. This script performs the package's explicit RC
    operation; it does not replace it with NoiseModel.twirled().

Run:
    python examples/run_qft_explicit_rc_bound_tvd.py
"""

from __future__ import annotations

import inspect
import os
import random
import warnings
from typing import Any, Collection, Tuple

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
from proxysim.mqtbank import repeat
from proxysim.noise import (
    apply_readout_to_distribution,
    sample_trajectory,
)
from proxysim.parallel import pmap
from proxysim.rc import pauli_twirl


# =============================================================================
# Configuration
# =============================================================================

QASM = os.path.join(
    os.path.dirname(__file__),
    "qft14.qasm",
)

# Complete repetitions of the imported QASM circuit:
#
#     target = U^d
#
# If the QASM contains 98 individual two-qubit gates, the plotted cycle counts
# are 98 * d.
REPETITIONS = [1, 2, 3, 4, 6, 8]

# Sequence lengths used internally by cycle benchmarking.
# These are unrelated to the target-circuit repetition counts above.
CB_DEPTHS = [1, 2, 4, 8]

# Fixed physical noise model throughout the target-depth sweep.
NOISE = NoiseModel(
    enabled=True,
    p1=5e-5,
    p2=2.5e-4,
    p_readout=2.5e-4,
    p_idle=5e-5,
    p_z1=0.0,
    p_zz=0.0,
    theta_1q=0.0,
    theta_zz=0.0,
)

# Number of independently estimated TVD values shown at every cycle depth.
N_TVD_POINTS = 30

# Number of independently dressed circuits averaged to produce one TVD point.
N_RC_REALIZATIONS = 24

# Number of stochastic noise trajectories per dressed RC realization.
#
# Total trajectories per plotted TVD point:
#
#     N_RC_REALIZATIONS * N_TRAJ_PER_RC
N_TRAJ_PER_RC = 12

# Cycle-benchmarking statistics.
CB_SHOTS = 800
CB_DECAYS = 20

SEED = 42

OUT = (
    _bootstrap.RESULTS_DIR
    + "/qft_explicit_rc_bound_tvd.png"
)

OUT_DATA = (
    _bootstrap.RESULTS_DIR
    + "/qft_explicit_rc_bound_tvd_data.npz"
)

_SV = StatevectorBackend()


# =============================================================================
# RC compatibility helpers
# =============================================================================

def _looks_like_circuit(obj: Any) -> bool:
    """Return True when an object resembles a proxysim Circuit."""
    return (
        obj is not None
        and hasattr(obj, "gates")
        and hasattr(obj, "n_qubits")
    )


def _normalize_virtual_indices(
    virtual: Any,
) -> Tuple[int, ...]:
    """Convert the virtual-gate report into a stable tuple of integer indices."""
    if virtual is None:
        return ()

    if isinstance(virtual, dict):
        # Some APIs may return a map from gate index to metadata.
        return tuple(sorted(int(index) for index in virtual))

    if isinstance(virtual, np.ndarray):
        return tuple(int(index) for index in virtual.tolist())

    if isinstance(virtual, Collection) and not isinstance(
        virtual,
        (str, bytes),
    ):
        return tuple(int(index) for index in virtual)

    raise TypeError(
        "Could not interpret pauli_twirl's virtual-gate report. "
        f"Received object of type {type(virtual).__name__}."
    )


def _extract_dressed_and_virtual(
    result: Any,
) -> Tuple[Any, Tuple[int, ...]]:
    """Extract a dressed circuit and virtual gate indices from pauli_twirl.

    Supported return forms include:

        dressed_circuit

        (dressed_circuit, virtual_indices)

        {
            "circuit": dressed_circuit,
            "virtual": virtual_indices,
        }

        object.circuit
        object.virtual
    """
    if result is None:
        raise RuntimeError(
            "pauli_twirl returned None."
        )

    # Dictionary-style return.
    if isinstance(result, dict):
        circuit = None

        for key in (
            "circuit",
            "compiled_circuit",
            "dressed_circuit",
            "twirled_circuit",
        ):
            if key in result:
                circuit = result[key]
                break

        if circuit is None:
            raise TypeError(
                "pauli_twirl returned a dictionary without a recognized "
                f"circuit key. Available keys: {list(result)}"
            )

        virtual = ()

        for key in (
            "virtual",
            "virtual_indices",
            "virtual_gates",
            "frame_indices",
        ):
            if key in result:
                virtual = _normalize_virtual_indices(result[key])
                break

        return circuit, virtual

    # Tuple-style return.
    if isinstance(result, tuple):
        if len(result) == 0:
            raise TypeError(
                "pauli_twirl returned an empty tuple."
            )

        if len(result) == 1:
            circuit = result[0]
            virtual = ()
        else:
            first, second = result[0], result[1]

            if _looks_like_circuit(first):
                circuit = first
                virtual = _normalize_virtual_indices(second)
            elif _looks_like_circuit(second):
                circuit = second
                virtual = _normalize_virtual_indices(first)
            else:
                raise TypeError(
                    "Could not identify the circuit in pauli_twirl's tuple "
                    f"return value: {[type(x).__name__ for x in result]}"
                )

        return circuit, virtual

    # Object-style return.
    if hasattr(result, "circuit"):
        circuit = result.circuit

        virtual = ()

        for attr in (
            "virtual",
            "virtual_indices",
            "virtual_gates",
            "frame_indices",
        ):
            if hasattr(result, attr):
                virtual = _normalize_virtual_indices(
                    getattr(result, attr)
                )
                break

        return circuit, virtual

    # Bare-circuit return.
    if _looks_like_circuit(result):
        return result, ()

    raise TypeError(
        "Could not extract a dressed circuit from pauli_twirl's return value. "
        f"Received type {type(result).__name__}."
    )


def explicitly_randomized_compile(
    circ,
    seed: int,
):
    """Run pauli_twirl and return (dressed_circuit, virtual_gate_indices).

    The wrapper accommodates likely pauli_twirl signatures while requiring
    mark_virtual=True. The latter is essential: the twirling Paulis are frame
    changes and must not receive p1 or alter the idle-layer accounting.
    """
    rng = random.Random(seed)

    try:
        signature = inspect.signature(pauli_twirl)
        parameters = signature.parameters
    except (TypeError, ValueError):
        parameters = {}

    attempts = []

    # Prefer calls supported explicitly by the inspected function signature.
    if "seed" in parameters and "mark_virtual" in parameters:
        attempts.append(
            lambda: pauli_twirl(
                circ,
                seed=seed,
                mark_virtual=True,
            )
        )

    if "rng" in parameters and "mark_virtual" in parameters:
        attempts.append(
            lambda: pauli_twirl(
                circ,
                rng=random.Random(seed),
                mark_virtual=True,
            )
        )

    if "random_state" in parameters and "mark_virtual" in parameters:
        attempts.append(
            lambda: pauli_twirl(
                circ,
                random_state=random.Random(seed),
                mark_virtual=True,
            )
        )

    # Fallbacks for generic or unavailable Python signatures.
    attempts.extend(
        [
            lambda: pauli_twirl(
                circ,
                seed=seed,
                mark_virtual=True,
            ),
            lambda: pauli_twirl(
                circ,
                rng=random.Random(seed),
                mark_virtual=True,
            ),
            lambda: pauli_twirl(
                circ,
                random.Random(seed),
                mark_virtual=True,
            ),
            lambda: pauli_twirl(
                circ,
                seed,
                mark_virtual=True,
            ),
        ]
    )

    errors = []

    for attempt in attempts:
        try:
            result = attempt()
            dressed, virtual = _extract_dressed_and_virtual(
                result
            )

            if not _looks_like_circuit(dressed):
                raise TypeError(
                    "The object extracted from pauli_twirl does not resemble "
                    "a proxysim Circuit."
                )

            return dressed, virtual

        except TypeError as exc:
            errors.append(str(exc))

    raise TypeError(
        "Could not call proxysim.rc.pauli_twirl using a supported signature "
        "with mark_virtual=True.\nObserved errors:\n  - "
        + "\n  - ".join(errors)
    )


# =============================================================================
# CB worker
# =============================================================================

def cycle_ef(args):
    """Run cycle benchmarking on one Clifford-proxy cycle."""
    pair, n, twoq, seed, n_decays = args

    return cycle_benchmark(
        [pair],
        n,
        CB_DEPTHS,
        NOISE,
        twoq=twoq,
        n_decays=n_decays,
        shots=CB_SHOTS,
        seed=seed,
    )


# =============================================================================
# RC and TVD worker
# =============================================================================

def rc_averaged_tvd(args):
    """Produce one raw TVD estimate for the explicitly RC'd target circuit.

    The hierarchy of averaging is:

        trajectory average within each dressed circuit
        then RC-realization average across dressed circuits
        then TVD against the ideal distribution

    That is:

        TVD(
            mean_rc mean_traj p(noisy dressed circuit),
            p_ideal
        )

    rather than averaging a separate TVD for every RC realization.
    """
    (
        target,
        noise,
        ideal_prob,
        n_rc_realizations,
        n_traj_per_rc,
        seed,
    ) = args

    rc_average_prob = np.zeros_like(
        ideal_prob,
        dtype=float,
    )

    for rc_index in range(n_rc_realizations):
        rc_seed = (
            seed
            + 1_000_003 * rc_index
        )

        dressed, virtual_indices = (
            explicitly_randomized_compile(
                target,
                seed=rc_seed,
            )
        )

        trajectory_rng = random.Random(
            rc_seed + 1
        )

        dressed_prob = np.zeros_like(
            ideal_prob,
            dtype=float,
        )

        for _ in range(n_traj_per_rc):
            noisy_dressed = sample_trajectory(
                dressed,
                noise,
                trajectory_rng,
                virtual=virtual_indices,
            )

            psi = np.asarray(
                Statevector(
                    _SV._build(noisy_dressed)
                ).data,
                dtype=complex,
            )

            dressed_prob += np.abs(psi) ** 2

        dressed_prob /= n_traj_per_rc

        # sample_trajectory does not apply measurement error.
        # Apply the independent bit-flip readout channel exactly.
        dressed_prob = apply_readout_to_distribution(
            dressed_prob,
            noise,
            target.n_qubits,
        )

        rc_average_prob += dressed_prob

    rc_average_prob /= n_rc_realizations

    return 0.5 * float(
        np.abs(
            rc_average_prob - ideal_prob
        ).sum()
    )


def scatter_summary(values):
    """Return mean and range for one TVD scatter column."""
    values = np.asarray(
        values,
        dtype=float,
    )

    return (
        f"{values.mean():.5f} "
        f"[{values.min():.5f}, {values.max():.5f}]"
    )


# =============================================================================
# Validation helpers
# =============================================================================

def ideal_statevector(circ) -> np.ndarray:
    """Return the ideal statevector of a proxysim circuit."""
    return np.asarray(
        Statevector(
            _SV._build(circ)
        ).data,
        dtype=complex,
    )


def state_fidelity(
    psi: np.ndarray,
    phi: np.ndarray,
) -> float:
    """Pure-state fidelity, insensitive to global phase."""
    return float(
        abs(np.vdot(psi, phi)) ** 2
    )


def validate_one_rc_realization(
    base_circ,
):
    """Verify that the explicit RC pass preserves the ideal logical circuit."""
    dressed, virtual_indices = (
        explicitly_randomized_compile(
            base_circ,
            seed=SEED,
        )
    )

    original_state = ideal_statevector(
        base_circ
    )
    dressed_state = ideal_statevector(
        dressed
    )

    fidelity = state_fidelity(
        original_state,
        dressed_state,
    )

    original_twoq = sum(
        len(gate.qubits) == 2
        for gate in base_circ.gates
    )

    dressed_twoq = sum(
        len(gate.qubits) == 2
        for gate in dressed.gates
    )

    print(
        "explicit-RC smoke test:\n"
        f"    original gate count: {len(base_circ.gates)}\n"
        f"    dressed gate count:  {len(dressed.gates)}\n"
        f"    original 2q gates:   {original_twoq}\n"
        f"    dressed 2q gates:    {dressed_twoq}\n"
        f"    virtual RC gates:    {len(virtual_indices)}\n"
        f"    ideal state fidelity: {fidelity:.12f}\n"
    )

    if len(virtual_indices) == 0:
        warnings.warn(
            "pauli_twirl reported no virtual gate indices. If dressing gates "
            "were inserted, they may be incorrectly charged physical p1 and "
            "idle noise. Inspect proxysim.rc.pauli_twirl's return contract.",
            RuntimeWarning,
        )

    if not np.isclose(
        fidelity,
        1.0,
        atol=1e-9,
        rtol=1e-9,
    ):
        raise RuntimeError(
            "The dressed circuit is not logically equivalent to the original "
            "QASM circuit. pauli_twirl may return an unapplied final frame "
            "correction or use a different result structure."
        )


# =============================================================================
# Main analysis
# =============================================================================

def main():
    with open(
        QASM,
        encoding="utf-8",
    ) as qasm_file:
        base_circ, measured = (
            circuit_from_qasm(
                qasm_file.read()
            )
        )

    n = base_circ.n_qubits

    twoq_gates = [
        gate
        for gate in base_circ.gates
        if len(gate.qubits) == 2
    ]

    cycles_per_rep = len(
        twoq_gates
    )

    if cycles_per_rep == 0:
        raise ValueError(
            "The imported QASM circuit contains no two-qubit gates."
        )

    n_cp = sum(
        gate.name == "cp"
        for gate in twoq_gates
    )

    n_swap = sum(
        gate.name == "swap"
        for gate in twoq_gates
    )

    print(base_circ.summary())

    print(
        f"non-Clifford: {not base_circ.is_clifford}\n"
        f"individual 2q cycles per QASM repetition: "
        f"{cycles_per_rep}\n"
        f"gate breakdown: {n_cp} cp + {n_swap} swap"
    )

    print(
        "cycle convention: each individual two-qubit gate, together with "
        "its nearby one-qubit operations, is treated as one cycle"
    )

    print(
        "TVD protocol: explicitly dress every supported two-qubit cycle, "
        "simulate the dressed circuit under the original noise model, "
        "and average probability distributions"
    )

    print(
        f"fixed noise:\n"
        f"    p1={NOISE.p1:.3e}\n"
        f"    p2={NOISE.p2:.3e}\n"
        f"    p_readout={NOISE.p_readout:.3e}\n"
        f"    p_idle={NOISE.p_idle:.3e}\n"
        f"    p_z1={NOISE.p_z1:.3e}\n"
        f"    p_zz={NOISE.p_zz:.3e}\n"
        f"    theta_1q={NOISE.theta_1q:.3e}\n"
        f"    theta_zz={NOISE.theta_zz:.3e}"
    )

    print(
        "maximum tensor-network bond dimension for one repetition: "
        f"{TensorNetworkBackend().max_bond_of(base_circ)}\n"
    )

    n_workers = max(
        1,
        (os.cpu_count() or 2) - 2,
    )

    # Verify the pauli_twirl return contract and logical equivalence before
    # launching the expensive parallel sweep.
    validate_one_rc_realization(
        base_circ
    )

    # -------------------------------------------------------------------------
    # Proxy cycle benchmarking
    # -------------------------------------------------------------------------

    cb_jobs = [
        # Diagnostic comparisons.
        ((0, 1), n, "CZ", 11, 30),
        ((6, 7), n, "CZ", 12, 30),
        ((0, 13), n, "SWAP", 13, 30),

        # Higher-statistics CZ proxy used by the bound.
        ((0, 1), n, "CZ", 100, CB_DECAYS),
    ]

    cbs = pmap(
        cycle_ef,
        cb_jobs,
        n_workers=n_workers,
    )

    labels = [
        "CZ proxy on pair (0,1)",
        "CZ proxy on pair (6,7)",
        "SWAP proxy on pair (0,13)",
    ]

    print(
        "proxy comparison "
        "(agreement expected for homogeneous gate-independent noise):"
    )

    for label, cb in zip(
        labels,
        cbs[:3],
    ):
        print(
            f"    e_F={cb['e_F']:.6f}   {label}"
        )

    eF = float(
        cbs[-1]["e_F"]
    )

    eF_std = float(
        cbs[-1]["e_F_std"]
    )

    ro_fid, ro_std = readout_fidelity(
        n,
        NOISE,
        seed=3,
    )

    print(
        "\nCB values used in QCAP:\n"
        f"    e_F  = {eF:.6f} +/- {eF_std:.6f}\n"
        f"    F_RO = {ro_fid:.6f} +/- {ro_std:.6f}\n"
    )

    # -------------------------------------------------------------------------
    # Repetition / two-qubit-cycle-depth sweep
    # -------------------------------------------------------------------------

    cycle_depths = []
    qcap_bounds = []
    qcap_stds = []
    all_tvd_points = []
    ideal_supports = []
    ideal_is_uniform = []

    print(
        f"{'reps':>6}"
        f"{'2q cycles':>12}"
        f"{'target gates':>14}"
        f"{'support':>12}"
        f"{'QCAP':>12}"
        f"{'RC TVD mean [min,max]':>32}"
    )

    for d in REPETITIONS:
        target = repeat(
            base_circ,
            d,
        )

        expected_cycles = (
            d * cycles_per_rep
        )

        actual_cycles = sum(
            len(gate.qubits) == 2
            for gate in target.gates
        )

        if actual_cycles != expected_cycles:
            raise RuntimeError(
                "Unexpected 2q-cycle count after QASM repetition: "
                f"expected {expected_cycles}, found {actual_cycles}."
            )

        cycle_depths.append(
            actual_cycles
        )

        # Ideal logical output of U^d.
        ideal_state = ideal_statevector(
            target
        )

        ideal_prob = (
            np.abs(ideal_state) ** 2
        )

        support = int(
            np.count_nonzero(
                ideal_prob > 1e-12
            )
        )

        uniform = bool(
            np.allclose(
                ideal_prob,
                1.0 / ideal_prob.size,
                atol=1e-10,
                rtol=1e-10,
            )
        )

        ideal_supports.append(
            support
        )

        ideal_is_uniform.append(
            uniform
        )

        # Homogeneous-noise reduction: one proxy cycle class occurring
        # actual_cycles times.
        counts = {
            "twoq_proxy": actual_cycles,
        }

        cycle_efs = {
            "twoq_proxy": (
                eF,
                eF_std,
            ),
        }

        bound = qcap_bound(
            counts,
            cycle_efs,
            ro_fid,
            ro_std,
        )

        qcap_bounds.append(
            float(bound["error"])
        )

        qcap_stds.append(
            float(bound["std"])
        )

        # Each job produces one independently seeded RC-averaged TVD point.
        jobs = [
            (
                target,
                NOISE,
                ideal_prob,
                N_RC_REALIZATIONS,
                N_TRAJ_PER_RC,
                (
                    SEED
                    + 100_000_000 * d
                    + 100_000 * point_index
                ),
            )
            for point_index in range(
                N_TVD_POINTS
            )
        ]

        tvd_values = pmap(
            rc_averaged_tvd,
            jobs,
            n_workers=n_workers,
        )

        tvd_values = np.asarray(
            tvd_values,
            dtype=float,
        )

        all_tvd_points.append(
            tvd_values
        )

        print(
            f"{d:>6}"
            f"{actual_cycles:>12}"
            f"{len(target.gates):>14}"
            f"{support:>12}"
            f"{bound['error']:>12.5f}"
            f"{scatter_summary(tvd_values):>32}",
            flush=True,
        )

    cycle_depths = np.asarray(
        cycle_depths,
        dtype=int,
    )

    qcap_bounds = np.asarray(
        qcap_bounds,
        dtype=float,
    )

    qcap_stds = np.asarray(
        qcap_stds,
        dtype=float,
    )

    all_tvd_points = np.asarray(
        all_tvd_points,
        dtype=float,
    )

    # -------------------------------------------------------------------------
    # Save numerical results
    # -------------------------------------------------------------------------

    save_results(
        OUT_DATA,
        repetitions=np.asarray(
            REPETITIONS,
            dtype=int,
        ),
        cycle_depths=cycle_depths,
        qcap_bound=qcap_bounds,
        qcap_bound_std=qcap_stds,
        tvd_points=all_tvd_points,
        ideal_support=np.asarray(
            ideal_supports,
            dtype=int,
        ),
        ideal_is_uniform=np.asarray(
            ideal_is_uniform,
            dtype=bool,
        ),
        e_F=eF,
        e_F_std=eF_std,
        readout_fidelity=ro_fid,
        readout_fidelity_std=ro_std,
        cycles_per_rep=cycles_per_rep,
        n_tvd_points=N_TVD_POINTS,
        n_rc_realizations=N_RC_REALIZATIONS,
        n_traj_per_rc=N_TRAJ_PER_RC,
        total_trajectories_per_tvd=(
            N_RC_REALIZATIONS
            * N_TRAJ_PER_RC
        ),
    )

    # -------------------------------------------------------------------------
    # Plot: raw explicit-RC TVD points and QCAP bound
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9.2, 5.8)
    )

    for index, cycle_depth in enumerate(
        cycle_depths
    ):
        ax.plot(
            np.full(
                N_TVD_POINTS,
                cycle_depth,
            ),
            all_tvd_points[index],
            "o",
            color="#0072B2",
            ms=5.5,
            alpha=0.60,
            label=(
                "explicit-RC Z-basis TVD"
                if index == 0
                else None
            ),
        )

    ax.plot(
        cycle_depths,
        qcap_bounds,
        "-",
        color="#D55E00",
        lw=2.6,
        label=(
            r"QCAP bound from proxy CB: "
            r"$1-F_{\mathrm{RO}}"
            r"(1-e_F)^{N_{\mathrm{cyc}}}$"
        ),
    )

    lower = np.clip(
        qcap_bounds
        - 2.96 * qcap_stds,
        0.0,
        1.0,
    )

    upper = np.clip(
        qcap_bounds
        + 2.96 * qcap_stds,
        0.0,
        1.0,
    )

    ax.fill_between(
        cycle_depths,
        lower,
        upper,
        color="#D55E00",
        alpha=0.18,
        linewidth=0,
        label="QCAP fit uncertainty",
    )

    ax.axhline(
        1.0,
        color="0.5",
        lw=0.8,
        ls=":",
    )

    ax.set_xlabel(
        "number of applied two-qubit cycles"
    )

    ax.set_ylabel(
        "total variation distance / bound"
    )

    ax.set_title(
        "QFT-14: explicit-RC TVD versus proxy-CB QCAP bound\n"
        "fixed noise; every two-qubit cycle is independently dressed"
    )

    ax.set_ylim(
        -0.03,
        1.05,
    )

    ax.grid(
        True,
        alpha=0.15,
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    fig.savefig(
        OUT,
        dpi=160,
        bbox_inches="tight",
    )

    print(
        f"\nWrote {OUT}"
    )

    print(
        f"Wrote {OUT_DATA}"
    )


if __name__ == "__main__":
    main()