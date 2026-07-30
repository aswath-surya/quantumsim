"""Bound circuit error from cycle benchmarking.

This is a hardware-free analogue of the qcal ``circuit_bounding`` workflow.

Pipeline
--------
1. Cycle-benchmark each distinct entangling cycle to estimate e_F.
2. Estimate the readout/SPAM fidelity.
3. Generate target circuits of increasing depth.
4. Form the multiplicative QCAP bound

       error <= 1 - F_RO * product_c (1 - e_F,c) ** n_c

5. Compare the bound with the actual TVD between the ideal and noisy
   computational-basis output distributions.
6. Compare with the global-depolarizing prediction and its Renyi-2 relaxation.

Target ensembles
----------------
``random``:
    Generic non-Clifford single-qubit SU(2) layers, implemented as
    RZ(phi) RY(theta) RZ(lambda), with continuously sampled angles.

``structured``:
    Single-qubit layers drawn from {I, H, X}.

The cycle-benchmarking circuits themselves retain Clifford dressing because
cycle benchmarking is performed on Clifford proxy cycles.

Run
---
    python examples/run_bounding.py
"""

from __future__ import annotations

import random
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401

from proxysim import (
    NoiseModel,
    even_pairs,
    odd_pairs,
    save_results,
    simulate,
    total_variation_distance,
)
from proxysim.backends import StatevectorBackend
from proxysim.benchmarking import (
    cycle_benchmark,
    qcap_bound,
    readout_fidelity,
)
from proxysim.circuit import Circuit


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

# Use at least three qubits if both even and odd brickwork cycles should exist.
# At N=2, odd_pairs(2) is normally empty, so counting cycle B in the QCAP bound
# would overcount the target circuit.
N = 2

CYCLE_A = even_pairs(N)
CYCLE_B = odd_pairs(N)

DEPTHS = [1, 2, 4, 8, 16, 32, 64]
CB_DEPTHS = [1, 2, 4, 8, 16, 24]

N_INSTANCES = 60

# These are Monte-Carlo noise trajectories, not measurement shots.
TVD_TRAJECTORIES = 1000
CB_TRAJECTORIES = 50

SEED = 7

NOISE = NoiseModel(
    enabled=True,
    p1=1e-3,
    p2=1e-2,
    p_idle_z=1e-3,
    p_readout_01=1e-2,
    p_readout_10=1e-2,

    # Keep these zero initially if you want to isolate stochastic Pauli noise.
    oneq_overrotation=0.0,
    twoq_zz_overrotation=0.0,
    controlled_phase_offset=0.0,
    idle_z_angle=0.0,
)

OUT_DATA = _bootstrap.RESULTS_DIR + "/bounding_data.npz"

_SV = StatevectorBackend()


# ---------------------------------------------------------------------------
# Target-circuit construction
# ---------------------------------------------------------------------------

def add_random_su2_layer(circuit, rng):
    """Append one generic random U(2) gate per qubit.

    Each qubit receives one logical 1Q operation, so the target layer has the
    same number of noisy 1Q gate locations as the 1Q layer used in CB.
    """
    for q in range(N):
        theta = rng.uniform(0.0, np.pi)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        lam = rng.uniform(0.0, 2.0 * np.pi)

        circuit.add(
            "u",
            q,
            params=(theta, phi, lam),
        )


def add_structured_layer(circuit: Circuit, rng: random.Random) -> None:
    """Append a structured layer drawn independently from {I, H, X}."""
    for q in range(N):
        gate = rng.choice(("i", "h", "x"))

        if gate != "i":
            circuit.add(gate, q)


def add_oneq_layer(
    circuit: Circuit,
    mode: str,
    rng: random.Random,
) -> None:
    """Append the requested one-qubit layer."""
    if mode == "random":
        add_random_su2_layer(circuit, rng)
        return

    if mode == "structured":
        add_structured_layer(circuit, rng)
        return

    raise ValueError(
        f"Unknown target-circuit mode {mode!r}; "
        "expected 'random' or 'structured'."
    )


def add_entangling_layer(circuit: Circuit, pairs) -> None:
    """Append CZ gates on all pairs in one brickwork cycle."""
    for a, b in pairs:
        circuit.add("cz", a, b)


def test_circuit(
    depth: int,
    mode: str,
    seed: int,
) -> Circuit:
    """Construct one target-circuit instance.

    Each repetition contains

        one-qubit layer
        cycle A
        one-qubit layer
        cycle B

    followed by one final one-qubit layer.
    """
    rng = random.Random(seed)

    circuit = Circuit(
        N,
        name=f"{mode}_depth_{depth}_seed_{seed}",
    )

    for _ in range(depth):
        add_oneq_layer(circuit, mode, rng)
        add_entangling_layer(circuit, CYCLE_A)

        add_oneq_layer(circuit, mode, rng)
        add_entangling_layer(circuit, CYCLE_B)

    add_oneq_layer(circuit, mode, rng)

    return circuit


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def dense_distribution(circuit: Circuit) -> np.ndarray:
    """Return the ideal distribution as a normalized dense vector."""
    distribution = _SV.exact_distribution(circuit)
    dimension = 2**circuit.n_qubits

    if isinstance(distribution, dict):
        dense = np.zeros(
            dimension,
            dtype=float,
        )

        for outcome, probability in distribution.items():
            if isinstance(outcome, str):
                index = int(
                    outcome.replace(" ", ""),
                    2,
                )
            else:
                index = int(outcome)

            dense[index] += float(probability)

    else:
        dense = np.asarray(
            distribution,
            dtype=float,
        ).reshape(-1)

    if dense.size != dimension:
        raise ValueError(
            f"Ideal distribution has length {dense.size}; "
            f"expected {dimension}."
        )

    dense = np.clip(
        dense,
        0.0,
        None,
    )

    total = float(np.sum(dense))

    if total <= 0.0:
        raise ValueError(
            "Ideal distribution has zero total probability."
        )

    return dense / total


def noisy_distribution(
    circuit: Circuit,
    seed: int,
) -> np.ndarray:
    """Return the trajectory-averaged noisy output distribution."""
    distribution = simulate(
        circuit,
        simulator="statevector",
        output="distribution",
        noise=NOISE,

        # In the revised simulate.py, shots=0 allows n_traj to control the
        # number of statevector trajectories directly.
        shots=0,
        n_traj=TVD_TRAJECTORIES,

        seed=seed,
    )

    dense = np.asarray(
        distribution,
        dtype=float,
    ).reshape(-1)

    expected_size = 2**circuit.n_qubits

    if dense.size != expected_size:
        raise ValueError(
            f"Noisy distribution has length {dense.size}; "
            f"expected {expected_size}."
        )

    dense = np.clip(
        dense,
        0.0,
        None,
    )

    total = float(np.sum(dense))

    if total <= 0.0:
        raise ValueError(
            "Noisy distribution has zero total probability."
        )

    return dense / total


def noisy_tvd(
    circuit: Circuit,
    seed: int,
) -> float:
    """Compute TVD between ideal and noisy output distributions."""
    ideal = dense_distribution(circuit)
    noisy = noisy_distribution(circuit, seed)

    return float(
        total_variation_distance(
            ideal,
            noisy,
        )
    )


# ---------------------------------------------------------------------------
# Entropy-dependent predictions
# ---------------------------------------------------------------------------

def entropy_prediction_factors(
    ideal_distribution: np.ndarray,
) -> tuple[float, float]:
    """Return the exact-model and Renyi-2 multiplicative factors.

    Exact global-depolarizing model:

        TVD(p, q) = epsilon * TVD(q, uniform)

    Renyi-2 relaxation:

        TVD(q, uniform)
        <= 1/2 sqrt(D sum_x q_x^2 - 1).
    """
    q = np.asarray(
        ideal_distribution,
        dtype=float,
    ).reshape(-1)

    q /= np.sum(q)

    dimension = q.size

    uniform = np.full(
        dimension,
        1.0 / dimension,
        dtype=float,
    )

    exact_factor = 0.5 * float(
        np.sum(
            np.abs(q - uniform)
        )
    )

    collision_probability = float(
        np.sum(q**2)
    )

    renyi_factor = 0.5 * np.sqrt(
        max(
            dimension * collision_probability - 1.0,
            0.0,
        )
    )

    renyi_factor = min(
        1.0,
        float(renyi_factor),
    )

    return exact_factor, renyi_factor


# ---------------------------------------------------------------------------
# Cycle benchmarking and target-circuit study
# ---------------------------------------------------------------------------

def benchmark_cycles(mode: str):
    """Benchmark the nonempty brickwork cycles."""
    cycle_results = {}

    cycle_results["A"] = cycle_benchmark(
        CYCLE_A,
        N,
        CB_DEPTHS,
        NOISE,
        mode=mode,
        n_trajectories=CB_TRAJECTORIES,
        seed=1,
    )

    if CYCLE_B:
        cycle_results["B"] = cycle_benchmark(
            CYCLE_B,
            N,
            CB_DEPTHS,
            NOISE,
            mode=mode,
            n_trajectories=CB_TRAJECTORIES,
            seed=2,
        )

    return cycle_results


def run(mode: str):
    """Run the depth sweep for one target-circuit ensemble."""
    print(f"\n=== {mode} ansatz ===")

    cycle_results = benchmark_cycles(mode)

    ro_fid, ro_std = readout_fidelity(
        N,
        NOISE,
        seed=3,
    )

    for cycle_name, result in cycle_results.items():
        print(
            f"  cycle {cycle_name}: "
            f"e_F = {result['e_F']:.4f} "
            f"(std = {result['e_F_std']:.4f})"
        )

    print(
        f"  readout fidelity: "
        f"{ro_fid:.4f} ({ro_std:.4f})"
    )

    cycle_efs = {
        name: (
            result["e_F"],
            result["e_F_std"],
        )
        for name, result in cycle_results.items()
    }

    bounds = []
    bound_stds = []

    tvd_points = []
    renyi2_points = []
    exact_points = []

    print(
        f"  {'depth':>6}"
        f"{'bound':>10}"
        f"{'mean TVD':>12}"
        f"{'std TVD':>11}"
    )

    for depth in DEPTHS:
        cycle_counts = {
            name: depth
            for name in cycle_results
        }

        bound_result = qcap_bound(
            cycle_counts,
            cycle_efs,
            ro_fid,
            ro_std,
        )

        bound = float(bound_result["error"])
        bound_std = float(bound_result["std"])

        bounds.append(bound)
        bound_stds.append(bound_std)

        depth_tvds = []
        depth_renyi2 = []
        depth_exact = []

        for instance_index in range(N_INSTANCES):
            # Use depth-dependent circuit seeds so a depth-d circuit is not
            # simply regenerated from exactly the same short seed prefix as all
            # other depths.
            circuit_seed = (
                SEED
                + 100_000 * depth
                + 100 * instance_index
            )

            # Give every circuit instance an independent trajectory stream.
            noise_seed = (
                SEED
                + 1_000_000 * depth
                + instance_index
            )

            circuit = test_circuit(
                depth,
                mode,
                circuit_seed,
            )

            depth_tvds.append(
                noisy_tvd(
                    circuit,
                    noise_seed,
                )
            )

            ideal = dense_distribution(circuit)

            exact_factor, renyi_factor = (
                entropy_prediction_factors(ideal)
            )

            depth_exact.append(
                bound * exact_factor
            )

            depth_renyi2.append(
                bound * renyi_factor
            )

        tvd_points.append(depth_tvds)
        exact_points.append(depth_exact)
        renyi2_points.append(depth_renyi2)

        print(
            f"  {depth:>6}"
            f"{bound:>10.4f}"
            f"{np.mean(depth_tvds):>12.4f}"
            f"{np.std(depth_tvds):>11.4f}"
        )

    return {
        "bounds": np.asarray(
            bounds,
            dtype=float,
        ),
        "bound_stds": np.asarray(
            bound_stds,
            dtype=float,
        ),
        "tvd_points": np.asarray(
            tvd_points,
            dtype=float,
        ),
        "renyi2_points": np.asarray(
            renyi2_points,
            dtype=float,
        ),
        "exact_points": np.asarray(
            exact_points,
            dtype=float,
        ),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(data) -> str:
    """Plot both circuit ensembles."""
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.0),
        sharey=True,
    )

    for axis, mode in zip(
        axes,
        ("random", "structured"),
    ):
        result = data[mode]

        bounds = np.asarray(
            result["bounds"],
            dtype=float,
        )
        bound_stds = np.asarray(
            result["bound_stds"],
            dtype=float,
        )
        tvd_points = np.asarray(
            result["tvd_points"],
            dtype=float,
        )
        renyi2_points = np.asarray(
            result["renyi2_points"],
            dtype=float,
        )
        exact_points = np.asarray(
            result["exact_points"],
            dtype=float,
        )

        # Plot all circuit instances at every depth.
        for depth_index, depth in enumerate(DEPTHS):
            tvds = tvd_points[depth_index]

            axis.plot(
                np.full(
                    len(tvds),
                    depth,
                    dtype=float,
                ),
                tvds,
                "o",
                color="#0072B2",
                markersize=5,
                alpha=0.65,
                linestyle="none",
                label=(
                    "actual TVD"
                    if depth_index == 0
                    else None
                ),
            )

        axis.plot(
            DEPTHS,
            bounds,
            "-",
            color="#D55E00",
            linewidth=2.2,
            label="QCAP bound",
        )

        # Uncomment after the CB uncertainty estimate is stable enough.
        #
        # lower = np.clip(
        #     bounds - 1.96 * bound_stds,
        #     0.0,
        #     1.0,
        # )
        #
        # upper = np.clip(
        #     bounds + 1.96 * bound_stds,
        #     0.0,
        #     1.0,
        # )
        #
        # axis.fill_between(
        #     DEPTHS,
        #     lower,
        #     upper,
        #     color="#D55E00",
        #     alpha=0.18,
        # )

        exact_max = np.max(
            exact_points,
            axis=1,
        )

        renyi2_max = np.max(
            renyi2_points,
            axis=1,
        )

        axis.plot(
            DEPTHS,
            exact_max,
            "-.",
            color="darkgreen",
            linewidth=2.0,
            label="maximum depolarizing-model prediction",
        )

        axis.plot(
            DEPTHS,
            renyi2_max,
            "--",
            color="black",
            linewidth=2.0,
            label="maximum Renyi-2 relaxation",
        )

        axis.set_xlabel(
            "circuit depth"
        )

        axis.set_title(
            f"{mode} ansatz",
            fontsize=11,
        )

        axis.set_xticks(
            DEPTHS
        )

        axis.get_xaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter()
        )

        axis.grid(
            True,
            which="both",
            alpha=0.15,
        )

        axis.legend(
            frameon=False,
            fontsize=8.5,
        )

    axes[0].set_ylabel(
        "output-distribution error (TVD)"
    )

    figure.suptitle(
        "Bounding circuit error from cycle benchmarking "
        f"(n={N}, generalized trajectory noise)",
        fontsize=12,
    )

    figure.tight_layout()

    output_path = (
        _bootstrap.RESULTS_DIR
        + f"/bounding_{N}qubits.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.close(figure)

    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"N = {N}")
    print(f"cycle A = {CYCLE_A}")
    print(f"cycle B = {CYCLE_B}")
    print(
        f"target trajectories per circuit = "
        f"{TVD_TRAJECTORIES}"
    )

    data = {
        mode: run(mode)
        for mode in (
            "random",
            "structured",
        )
    }

    save_results(
        OUT_DATA,
        depths=np.asarray(
            DEPTHS,
            dtype=int,
        ),

        random_bound=data["random"]["bounds"],
        random_bound_std=data["random"]["bound_stds"],
        random_tvd=data["random"]["tvd_points"],
        random_exact=data["random"]["exact_points"],
        random_renyi2=data["random"]["renyi2_points"],

        structured_bound=data["structured"]["bounds"],
        structured_bound_std=data["structured"]["bound_stds"],
        structured_tvd=data["structured"]["tvd_points"],
        structured_exact=data["structured"]["exact_points"],
        structured_renyi2=data["structured"]["renyi2_points"],
    )

    output_path = plot_results(data)

    print(
        f"\nWrote {output_path}"
    )
    print(
        f"Wrote {OUT_DATA}"
    )


if __name__ == "__main__":
    main()