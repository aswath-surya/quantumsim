"""Plot explicit-RC TVD with QCAP, exact white-noise, and Renyi-2 bounds.

This is a lightweight post-processing script. It reuses the .npz produced by

    run_qft_explicit_rc_bound_tvd.py

and recomputes the ideal measured-register distribution for qft14.qasm.
It does NOT rerun cycle benchmarking or the expensive explicit-RC trajectories.

Under the global white-noise model

    q = (1 - eps) p + eps u,

the circuit-specific exact prediction is

    B_exact(eps) = eps * D_TV(p, u),

and the Renyi-2 upper bound is

    B_Renyi(eps) = eps * min(
        1,
        0.5 * sqrt(D * sum_x p(x)^2 - 1)
    ),

where D = 2^(number of measured qubits).

The script plots B_exact(eps_qcap) and B_Renyi(eps_qcap), together with
the original TVD scatter, QCAP curve, and QCAP uncertainty band.

Run:
    python examples/plot_qft_explicit_rc_exact_renyi.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector

import _bootstrap  # noqa: F401
from proxysim.backends import StatevectorBackend
from proxysim.circuit import circuit_from_qasm


# =============================================================================
# Configuration
# =============================================================================

HERE = os.path.dirname(__file__)

QASM = os.path.join(
    HERE,
    "qft14.qasm",
)

INPUT_DATA = (
    _bootstrap.RESULTS_DIR
    + "/qft_explicit_rc_bound_tvd_data.npz"
)

OUT = (
    _bootstrap.RESULTS_DIR
    + "/qft_explicit_rc_bound_tvd_exact_renyi.png"
)

OUT_DATA = (
    _bootstrap.RESULTS_DIR
    + "/qft_explicit_rc_bound_tvd_exact_renyi_data.npz"
)

# Match the uncertainty multiplier used by the original plotting script.
UNCERTAINTY_SIGMAS = 2.96

_SV = StatevectorBackend()


# =============================================================================
# Distribution helpers
# =============================================================================

def exact_probability_vector(circ) -> np.ndarray:
    """Return the ideal full-register computational-basis distribution."""
    probability = np.asarray(
        Statevector(
            _SV._build(circ)
        ).probabilities(),
        dtype=float,
    )

    total = float(probability.sum())

    if not np.isclose(
        total,
        1.0,
        atol=1e-10,
        rtol=1e-10,
    ):
        raise RuntimeError(
            "Ideal full-register distribution is not normalized: "
            f"sum={total:.12f}."
        )

    return probability / total


def marginalize_distribution(
    probability: np.ndarray,
    n_qubits: int,
    measured_qubits,
) -> np.ndarray:
    """Marginalize a Qiskit little-endian distribution onto measured_qubits."""
    measured_set = set(
        int(qubit)
        for qubit in measured_qubits
    )

    drop_axes = tuple(
        n_qubits - 1 - qubit
        for qubit in range(n_qubits)
        if qubit not in measured_set
    )

    tensor = np.asarray(
        probability,
        dtype=float,
    ).reshape([2] * n_qubits)

    if drop_axes:
        tensor = tensor.sum(
            axis=drop_axes
        )

    marginal = tensor.reshape(-1)
    marginal /= marginal.sum()

    return marginal


def ideal_bound_factors(
    ideal_probability: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return exact and Renyi-2 factors for one ideal output distribution.

    Returns
    -------
    uniform_tvd
        D_TV(p, u), the exact multiplicative white-noise factor.

    collision_probability
        sum_x p(x)^2.

    renyi_factor_unclipped
        0.5 * sqrt(D * sum_x p(x)^2 - 1).

    renyi_factor_clipped
        min(1, renyi_factor_unclipped), since TVD cannot exceed one.
    """
    p = np.asarray(
        ideal_probability,
        dtype=float,
    )

    if p.ndim != 1:
        raise ValueError(
            "ideal_probability must be a one-dimensional vector."
        )

    dimension = p.size
    uniform_probability = 1.0 / dimension

    uniform_tvd = 0.5 * float(
        np.abs(
            p - uniform_probability
        ).sum()
    )

    collision_probability = float(
        np.square(p).sum()
    )

    radicand = max(
        dimension * collision_probability - 1.0,
        0.0,
    )

    renyi_factor_unclipped = 0.5 * np.sqrt(
        radicand
    )

    renyi_factor_clipped = min(
        1.0,
        float(renyi_factor_unclipped),
    )

    if uniform_tvd > renyi_factor_clipped + 1e-10:
        raise AssertionError(
            "The exact white-noise factor exceeds the clipped Renyi-2 "
            "factor, contrary to the chi-squared/Renyi inequality: "
            f"{uniform_tvd:.12f} > {renyi_factor_clipped:.12f}."
        )

    return (
        uniform_tvd,
        collision_probability,
        float(renyi_factor_unclipped),
        renyi_factor_clipped,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    if not os.path.exists(INPUT_DATA):
        raise FileNotFoundError(
            f"Could not find {INPUT_DATA}. Run "
            "run_qft_explicit_rc_bound_tvd.py first."
        )

    with np.load(
        INPUT_DATA,
        allow_pickle=False,
    ) as source:
        repetitions = np.asarray(
            source["repetitions"],
            dtype=int,
        )

        cycle_depths = np.asarray(
            source["cycle_depths"],
            dtype=int,
        )

        qcap_bounds = np.asarray(
            source["qcap_bound"],
            dtype=float,
        )

        qcap_stds = np.asarray(
            source["qcap_bound_std"],
            dtype=float,
        )

        all_tvd_points = np.asarray(
            source["tvd_points"],
            dtype=float,
        )

        # Older archives produced by run_qft_explicit_rc_bound_tvd.py did
        # not save measured_qubits. Treat the QASM file as the source of
        # truth in that case and only perform the consistency check when the
        # field is present.
        if "measured_qubits" in source.files:
            measured_qubits_saved = tuple(
                int(qubit)
                for qubit in np.asarray(
                    source["measured_qubits"],
                    dtype=int,
                )
            )
        else:
            measured_qubits_saved = None

    if all_tvd_points.ndim != 2:
        raise ValueError(
            "Expected tvd_points to have shape "
            "(number of repetitions, number of TVD points). "
            f"Found {all_tvd_points.shape}."
        )

    if all_tvd_points.shape[0] != len(repetitions):
        raise ValueError(
            "The first tvd_points dimension does not match repetitions."
        )

    if not (
        len(cycle_depths)
        == len(qcap_bounds)
        == len(qcap_stds)
        == len(repetitions)
    ):
        raise ValueError(
            "Input arrays do not share the same depth dimension."
        )

    with open(
        QASM,
        encoding="utf-8",
    ) as qasm_file:
        base_circ, measured_qubits_qasm = circuit_from_qasm(
            qasm_file.read()
        )

    measured_qubits_qasm = tuple(
        sorted(
            int(qubit)
            for qubit in measured_qubits_qasm
        )
    )

    if (
        measured_qubits_saved is not None
        and measured_qubits_saved != measured_qubits_qasm
    ):
        raise RuntimeError(
            "Measured-qubit mismatch between the saved result and qft14.qasm: "
            f"saved={measured_qubits_saved}, qasm={measured_qubits_qasm}."
        )

    n_qubits = base_circ.n_qubits

    measured_dimension = 2 ** len(
        measured_qubits_qasm
    )

    # Compute the ideal QFT output distribution once. The ideal algorithm is
    # the same at every RC/CB depth; only the accumulated noise changes.
    ideal_full = exact_probability_vector(
        base_circ
    )

    ideal_measured = marginalize_distribution(
        ideal_full,
        n_qubits,
        measured_qubits_qasm,
    )

    if ideal_measured.size != measured_dimension:
        raise RuntimeError(
            "Unexpected measured-register dimension: "
            f"expected {measured_dimension}, found {ideal_measured.size}."
        )

    (
        uniform_tvd,
        collision_probability,
        renyi_unclipped,
        renyi_clipped,
    ) = ideal_bound_factors(
        ideal_measured
    )

    # Reuse the same ideal-distribution factors at every cycle depth.
    uniform_tvds = np.full(
        len(repetitions),
        uniform_tvd,
        dtype=float,
    )

    collision_probabilities = np.full(
        len(repetitions),
        collision_probability,
        dtype=float,
    )

    renyi_factors_unclipped = np.full(
        len(repetitions),
        renyi_unclipped,
        dtype=float,
    )

    renyi_factors_clipped = np.full(
        len(repetitions),
        renyi_clipped,
        dtype=float,
    )

    # Central exact white-noise prediction.
    exact_bounds = np.clip(
        qcap_bounds * uniform_tvds,
        0.0,
        1.0,
    )

    # Central Renyi-2 white-noise upper bound.
    renyi_bounds = np.clip(
        qcap_bounds * renyi_factors_clipped,
        0.0,
        1.0,
    )

    # Upper and lower edges of the QCAP uncertainty interval.
    qcap_upper = np.clip(
        qcap_bounds
        + UNCERTAINTY_SIGMAS * qcap_stds,
        0.0,
        1.0,
    )

    qcap_lower = np.clip(
        qcap_bounds
        - UNCERTAINTY_SIGMAS * qcap_stds,
        0.0,
        1.0,
    )

    # Optional upper-edge exact and Renyi curves, saved for later analysis.
    max_exact_bounds = np.clip(
        qcap_upper * uniform_tvds,
        0.0,
        1.0,
    )

    max_renyi_bounds = np.clip(
        qcap_upper * renyi_factors_clipped,
        0.0,
        1.0,
    )

    if np.any(
        exact_bounds
        > renyi_bounds + 1e-10
    ):
        raise AssertionError(
            "Exact white-noise prediction exceeds the Renyi-2 bound."
        )

    if np.any(
        renyi_bounds
        > qcap_bounds + 1e-10
    ):
        raise AssertionError(
            "Renyi-2 bound exceeds the QCAP bound."
        )

    np.savez_compressed(
        OUT_DATA,
        repetitions=repetitions,
        cycle_depths=cycle_depths,
        qcap_bound=qcap_bounds,
        qcap_bound_std=qcap_stds,
        qcap_lower=qcap_lower,
        qcap_upper=qcap_upper,
        uncertainty_sigmas=UNCERTAINTY_SIGMAS,
        tvd_points=all_tvd_points,
        measured_qubits=np.asarray(
            measured_qubits_qasm,
            dtype=int,
        ),
        ideal_uniform_tvd=uniform_tvds,
        ideal_collision_probability=collision_probabilities,
        renyi_factor_unclipped=renyi_factors_unclipped,
        renyi_factor_clipped=renyi_factors_clipped,
        exact_bound=exact_bounds,
        renyi_bound=renyi_bounds,
        max_exact_bound=max_exact_bounds,
        max_renyi_bound=max_renyi_bounds,
    )

    print(
        f"{'reps':>6}"
        f"{'CX cycles':>12}"
        f"{'QCAP':>10}"
        f"{'QCAP std':>12}"
        f"{'DTV(p,u)':>12}"
        f"{'R2 factor':>12}"
        f"{'exact':>12}"
        f"{'R2 bound':>12}"
    )

    for index, repetition in enumerate(
        repetitions
    ):
        print(
            f"{repetition:>6d}"
            f"{cycle_depths[index]:>12d}"
            f"{qcap_bounds[index]:>10.5f}"
            f"{qcap_stds[index]:>12.5f}"
            f"{uniform_tvds[index]:>12.5f}"
            f"{renyi_factors_clipped[index]:>12.5f}"
            f"{exact_bounds[index]:>12.5f}"
            f"{renyi_bounds[index]:>12.5f}"
        )

    print(
        "\nIdeal-distribution factors:"
    )

    print(
        f"  D_TV(p, u)                = {uniform_tvd:.12f}"
    )

    print(
        f"  collision probability     = {collision_probability:.12f}"
    )

    print(
        f"  Renyi-2 factor unclipped  = {renyi_unclipped:.12f}"
    )

    print(
        f"  Renyi-2 factor clipped    = {renyi_clipped:.12f}"
    )

    fig, ax = plt.subplots(
        figsize=(9.4, 6.0)
    )

    n_tvd_points = all_tvd_points.shape[1]

    for index, cycle_depth in enumerate(
        cycle_depths
    ):
        ax.plot(
            np.full(
                n_tvd_points,
                cycle_depth,
            ),
            all_tvd_points[index],
            "o",
            color="#0072B2",
            ms=5.5,
            alpha=0.60,
            label=(
                "explicit-RC measured-register TVD"
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
        label="QCAP bound from CX-cycle CB",
    )

    ax.fill_between(
        cycle_depths,
        qcap_lower,
        qcap_upper,
        color="#D55E00",
        alpha=0.16,
        linewidth=0,
        label=(
            f"QCAP fit uncertainty "
            f"($\\pm {UNCERTAINTY_SIGMAS:g}\\sigma$)"
        ),
    )

    ax.plot(
        cycle_depths,
        exact_bounds,
        "--",
        color="#009E73",
        lw=2.2,
        marker="s",
        ms=5.0,
        label="exact white-noise prediction",
    )

    ax.plot(
        cycle_depths,
        renyi_bounds,
        "-.",
        color="#CC79A7",
        lw=2.2,
        marker="^",
        ms=5.2,
        label="Renyi-2 white-noise bound",
    )

    ax.axhline(
        1.0,
        color="0.5",
        lw=0.8,
        ls=":",
    )

    ax.set_xlabel(
        "number of applied ASAP CX cycles"
    )

    ax.set_ylabel(
        "total variation distance / bound"
    )

    ax.set_title(
        "QFT-14: explicit-RC TVD, QCAP, exact, and Renyi-2 bounds\n"
        "14-qubit circuit; TVD evaluated on the 14 measured qubits"
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
        fontsize=9,
    )

    fig.tight_layout()

    fig.savefig(
        OUT,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nWrote {OUT}"
    )

    print(
        f"Wrote {OUT_DATA}"
    )


if __name__ == "__main__":
    main()