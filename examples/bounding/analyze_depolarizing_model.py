"""Diagnose why an entropy-corrected QCAP curve works for structured circuits.

This script tests the key modeling assumption directly:

    p ~= (1 - alpha) q + alpha u,

where

    q : ideal logical output distribution,
    p : noisy output distribution averaged over randomized-compiled circuits,
    u : uniform distribution.

For every random or structured target circuit, the script:

  1. constructs the ideal logical distribution q;
  2. generates randomized-compiled (RC) copies;
  3. verifies that each RC copy preserves q ideally;
  4. averages noisy output distributions over RC copies and noise trajectories;
  5. fits the best global-depolarizing strength alpha;
  6. records the residual TVD of the best fit;
  7. computes H_2(q), TVD(q,u), and the Renyi-2 square-root factor;
  8. compares actual TVD with:
       - the baseline QCAP curve,
       - the exact depolarizing-model prediction,
       - the Renyi-2 relaxation;
  9. produces plots that separate:
       - circuit-to-circuit entropy variation,
       - depolarizing-fit quality,
       - fitted alpha versus QCAP error,
       - empirical bound coverage.

The main research question is:

    Does the entropy correction work for structured circuits because their noisy
    output distributions are still close to a global white-noise mixture, even
    when their ideal distributions are not Porter-Thomas?

Run from the repository root, for example:

    python examples/bounding/analyze_depolarizing_model.py \
        --qubits 4 \
        --instances 200 \
        --rc-randomizations 16 \
        --trajectories 100

For the full ~1000-circuit study:

    python examples/bounding/analyze_depolarizing_model.py \
        --qubits 4 \
        --instances 1000 \
        --rc-randomizations 32 \
        --trajectories 200

Warning:
    Total statevector calls scale approximately as

        2 modes * len(depths) * instances
        * rc_randomizations * trajectories.

Use a smaller pilot run first.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from proxysim import (
    NoiseModel,
    even_pairs,
    odd_pairs,
    simulate,
    total_variation_distance,
)
from proxysim.backends import StatevectorBackend
from proxysim.benchmarking import (
    cycle_benchmark,
    qcap_bound,
    randomly_compile,
    readout_fidelity,
)
from proxysim.circuit import Circuit


_SV = StatevectorBackend()


# =============================================================================
# Configuration containers
# =============================================================================

@dataclass(frozen=True)
class StudyConfig:
    n_qubits: int
    depths: Tuple[int, ...]
    n_instances: int
    rc_randomizations: int
    trajectories_per_rc: int
    cb_depths: Tuple[int, ...]
    cb_decays: int
    cb_trajectories: int
    seed: int
    output_dir: Path
    ideal_rc_tolerance: float


@dataclass
class CircuitRecord:
    mode: str
    n_qubits: int
    depth: int
    instance: int
    circuit_seed: int
    noise_seed: int

    collision: float
    h2: float
    tvd_q_uniform: float
    renyi_factor: float

    actual_tvd: float
    qcap_error: float
    qcap_std: float
    exact_prediction: float
    renyi_prediction: float

    alpha_fit: float
    fit_residual_tvd: float
    fitted_model_tvd: float
    residual_fraction: float
    alpha_over_qcap: float

    mean_individual_rc_tvd: float
    std_individual_rc_tvd: float
    max_ideal_rc_tvd: float

    qcap_covers: bool
    exact_prediction_covers: bool
    renyi_prediction_covers: bool


# =============================================================================
# Circuit construction
# =============================================================================

def add_random_u_layer(
    circuit: Circuit,
    rng: random.Random,
) -> None:
    """Append one generic continuously distributed U gate per qubit.

    The theta sampling below gives the correct sin(theta) density for a random
    pure-state direction on the Bloch sphere. For this scrambling diagnostic,
    independent U gates plus repeated entangling layers are sufficient.
    """
    for qubit in range(circuit.n_qubits):
        z = rng.uniform(-1.0, 1.0)
        theta = math.acos(z)
        phi = rng.uniform(0.0, 2.0 * math.pi)
        lam = rng.uniform(0.0, 2.0 * math.pi)

        circuit.add(
            "u",
            qubit,
            params=(theta, phi, lam),
        )


def add_structured_layer(
    circuit: Circuit,
    rng: random.Random,
) -> None:
    """Append a structured layer drawn from {I,H,X}."""
    for qubit in range(circuit.n_qubits):
        gate = rng.choice(("i", "h", "x"))
        if gate != "i":
            circuit.add(gate, qubit)


def add_entangling_layer(
    circuit: Circuit,
    pairs: Sequence[Tuple[int, int]],
) -> None:
    for a, b in pairs:
        circuit.add("cz", a, b)


def build_target_circuit(
    n_qubits: int,
    depth: int,
    mode: str,
    seed: int,
) -> Circuit:
    """Build [1Q,A,1Q,B]^depth followed by a final 1Q layer."""
    if mode not in {"random", "structured"}:
        raise ValueError("mode must be 'random' or 'structured'.")

    rng = random.Random(seed)
    circuit = Circuit(
        n_qubits,
        name=f"{mode}_n{n_qubits}_d{depth}_s{seed}",
    )

    cycle_a = even_pairs(n_qubits)
    cycle_b = odd_pairs(n_qubits)

    add_layer = (
        add_random_u_layer
        if mode == "random"
        else add_structured_layer
    )

    for _ in range(depth):
        add_layer(circuit, rng)
        add_entangling_layer(circuit, cycle_a)

        add_layer(circuit, rng)
        add_entangling_layer(circuit, cycle_b)

    add_layer(circuit, rng)

    return circuit


# =============================================================================
# Distribution and entropy helpers
# =============================================================================

def as_dense_distribution(
    distribution,
    n_qubits: int,
) -> np.ndarray:
    """Convert a dict/list/array distribution into a normalized dense vector."""
    dimension = 2**n_qubits

    if isinstance(distribution, dict):
        dense = np.zeros(dimension, dtype=float)

        for outcome, probability in distribution.items():
            if isinstance(outcome, str):
                index = int(outcome.replace(" ", ""), 2)
            else:
                index = int(outcome)

            dense[index] += float(probability)
    else:
        dense = np.asarray(distribution, dtype=float).reshape(-1)

    if dense.size != dimension:
        raise ValueError(
            f"Distribution length {dense.size}; expected {dimension}."
        )

    dense = np.clip(dense, 0.0, None)
    total = float(np.sum(dense))

    if total <= 0.0:
        raise ValueError("Distribution has zero total probability.")

    return dense / total


def ideal_distribution(circuit: Circuit) -> np.ndarray:
    return as_dense_distribution(
        _SV.exact_distribution(circuit),
        circuit.n_qubits,
    )


def noisy_distribution(
    circuit: Circuit,
    noise: NoiseModel,
    trajectories: int,
    seed: int,
) -> np.ndarray:
    distribution = simulate(
        circuit,
        simulator="statevector",
        output="distribution",
        noise=noise,
        shots=0,
        n_traj=trajectories,
        seed=seed,
    )

    return as_dense_distribution(
        distribution,
        circuit.n_qubits,
    )


def collision_probability(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=float)
    return float(np.sum(q**2))


def collision_entropy(q: np.ndarray) -> float:
    return -math.log2(collision_probability(q))


def tvd_to_uniform(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=float)
    uniform = np.full(q.size, 1.0 / q.size)

    return 0.5 * float(np.sum(np.abs(q - uniform)))


def renyi2_factor(q: np.ndarray) -> float:
    """Return min{1, 1/2 sqrt(D sum_x q_x^2 - 1)}."""
    q = np.asarray(q, dtype=float)
    dimension = q.size

    raw = 0.5 * math.sqrt(
        max(
            dimension * collision_probability(q) - 1.0,
            0.0,
        )
    )

    return min(1.0, raw)


# =============================================================================
# Best global-depolarizing fit
# =============================================================================

def depolarized_distribution(
    q: np.ndarray,
    alpha: float,
) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    uniform = np.full(q.size, 1.0 / q.size)

    return (1.0 - alpha) * q + alpha * uniform


def fit_global_depolarizing_model(
    q: np.ndarray,
    p: np.ndarray,
) -> Dict[str, float | np.ndarray]:
    """Fit p ~= (1-alpha)q + alpha u by minimizing TVD over alpha in [0,1]."""
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)

    if q.shape != p.shape:
        raise ValueError(
            f"q and p shapes differ: {q.shape} vs {p.shape}."
        )

    def objective(alpha: float) -> float:
        model = depolarized_distribution(q, alpha)
        return 0.5 * float(np.sum(np.abs(p - model)))

    result = minimize_scalar(
        objective,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-10},
    )

    alpha = float(np.clip(result.x, 0.0, 1.0))
    model = depolarized_distribution(q, alpha)
    residual = objective(alpha)

    actual_tvd = float(
        total_variation_distance(q, p)
    )

    fitted_model_tvd = float(
        total_variation_distance(q, model)
    )

    residual_fraction = (
        residual / actual_tvd
        if actual_tvd > 1e-15
        else 0.0
    )

    return {
        "alpha": alpha,
        "model": model,
        "residual_tvd": residual,
        "actual_tvd": actual_tvd,
        "fitted_model_tvd": fitted_model_tvd,
        "residual_fraction": residual_fraction,
    }


# =============================================================================
# RC analysis of one target circuit
# =============================================================================

def analyze_rc_target(
    circuit: Circuit,
    noise: NoiseModel,
    n_randomizations: int,
    trajectories_per_rc: int,
    seed: int,
    ideal_tolerance: float,
) -> Dict[str, object]:
    """Average noisy randomized-compiled copies and fit the white-noise model."""
    q = ideal_distribution(circuit)

    rc_circuits = randomly_compile(
        circuit,
        n_compilations=n_randomizations,
        seed=seed,
    )

    p_average = np.zeros_like(q)
    ideal_rc_tvds: List[float] = []
    individual_noisy_tvds: List[float] = []

    for rc_index, rc_circuit in enumerate(rc_circuits):
        q_rc = ideal_distribution(rc_circuit)

        ideal_rc_tvd = float(
            total_variation_distance(q, q_rc)
        )
        ideal_rc_tvds.append(ideal_rc_tvd)

        if ideal_rc_tvd > ideal_tolerance:
            raise RuntimeError(
                "RC implementation failed ideal-equivalence check. "
                f"Instance {rc_index}: TVD(q, q_rc)={ideal_rc_tvd:.3e}. "
                "The Pauli correction frame is incorrect or unsupported."
            )

        p_rc = noisy_distribution(
            rc_circuit,
            noise,
            trajectories=trajectories_per_rc,
            seed=seed + 100_000 + rc_index,
        )

        p_average += p_rc

        individual_noisy_tvds.append(
            float(
                total_variation_distance(q, p_rc)
            )
        )

    p_average /= max(len(rc_circuits), 1)

    fit = fit_global_depolarizing_model(
        q,
        p_average,
    )

    return {
        "q": q,
        "p_average": p_average,
        "model": fit["model"],
        "actual_tvd": fit["actual_tvd"],
        "alpha_fit": fit["alpha"],
        "fit_residual_tvd": fit["residual_tvd"],
        "fitted_model_tvd": fit["fitted_model_tvd"],
        "residual_fraction": fit["residual_fraction"],
        "mean_individual_rc_tvd": float(
            np.mean(individual_noisy_tvds)
        ),
        "std_individual_rc_tvd": float(
            np.std(individual_noisy_tvds, ddof=1)
            if len(individual_noisy_tvds) > 1
            else 0.0
        ),
        "max_ideal_rc_tvd": float(
            max(ideal_rc_tvds, default=0.0)
        ),
    }


# =============================================================================
# Cycle benchmarking and QCAP
# =============================================================================

def benchmark_cycle_classes(
    config: StudyConfig,
    noise: NoiseModel,
    mode: str,
) -> Dict[str, dict]:
    cycle_a = even_pairs(config.n_qubits)
    cycle_b = odd_pairs(config.n_qubits)

    results: Dict[str, dict] = {}

    if cycle_a:
        results["A"] = cycle_benchmark(
            cycle_a,
            config.n_qubits,
            config.cb_depths,
            noise,
            mode=mode,
            n_decays=config.cb_decays,
            n_trajectories=config.cb_trajectories,
            seed=config.seed + 11,
        )

    if cycle_b:
        results["B"] = cycle_benchmark(
            cycle_b,
            config.n_qubits,
            config.cb_depths,
            noise,
            mode=mode,
            n_decays=config.cb_decays,
            n_trajectories=config.cb_trajectories,
            seed=config.seed + 29,
        )

    return results


def qcap_at_depth(
    depth: int,
    cycle_results: Mapping[str, dict],
    ro_fid: float,
    ro_std: float,
) -> Dict[str, float]:
    cycle_counts = {
        cycle_name: depth
        for cycle_name in cycle_results
    }

    cycle_efs = {
        cycle_name: (
            float(result["e_F"]),
            float(result["e_F_std"]),
        )
        for cycle_name, result in cycle_results.items()
    }

    result = qcap_bound(
        cycle_counts,
        cycle_efs,
        ro_fid,
        ro_std,
    )

    return {
        "error": float(result["error"]),
        "std": float(result["std"]),
    }


# =============================================================================
# Main simulation
# =============================================================================

def run_study(
    config: StudyConfig,
    noise: NoiseModel,
) -> Tuple[List[CircuitRecord], Dict[str, object]]:
    records: List[CircuitRecord] = []
    metadata: Dict[str, object] = {
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
        },
        "noise": repr(noise),
        "cycle_benchmarks": {},
        "readout": {},
    }

    for mode_index, mode in enumerate(("random", "structured")):
        print(f"\n=== {mode} ensemble ===", flush=True)

        cycle_results = benchmark_cycle_classes(
            config,
            noise,
            mode,
        )

        metadata["cycle_benchmarks"][mode] = {
            name: {
                key: value
                for key, value in result.items()
                if isinstance(
                    value,
                    (int, float, str, bool, list),
                )
            }
            for name, result in cycle_results.items()
        }

        for name, result in cycle_results.items():
            print(
                f"cycle {name}: "
                f"e_F={result['e_F']:.6f}, "
                f"std={result['e_F_std']:.6f}, "
                f"fit_rmse={result.get('fit_rmse', float('nan')):.4g}",
                flush=True,
            )

        ro_fid, ro_std = readout_fidelity(
            config.n_qubits,
            noise,
            seed=config.seed + 101 + mode_index,
        )

        metadata["readout"][mode] = {
            "fidelity": ro_fid,
            "std": ro_std,
        }

        print(
            f"readout fidelity={ro_fid:.6f} ± {ro_std:.6f}",
            flush=True,
        )

        for depth in config.depths:
            qcap = qcap_at_depth(
                depth,
                cycle_results,
                ro_fid,
                ro_std,
            )

            print(
                f"depth={depth:>4}: QCAP={qcap['error']:.5f}",
                flush=True,
            )

            for instance in range(config.n_instances):
                circuit_seed = (
                    config.seed
                    + 10_000_000 * mode_index
                    + 100_000 * depth
                    + instance
                )
                noise_seed = (
                    config.seed
                    + 20_000_000 * mode_index
                    + 1_000_000 * depth
                    + instance
                )

                circuit = build_target_circuit(
                    config.n_qubits,
                    depth,
                    mode,
                    circuit_seed,
                )

                analysis = analyze_rc_target(
                    circuit,
                    noise,
                    n_randomizations=config.rc_randomizations,
                    trajectories_per_rc=config.trajectories_per_rc,
                    seed=noise_seed,
                    ideal_tolerance=config.ideal_rc_tolerance,
                )

                q = np.asarray(analysis["q"], dtype=float)

                collision = collision_probability(q)
                h2 = -math.log2(collision)
                uniform_distance = tvd_to_uniform(q)
                r2 = renyi2_factor(q)

                actual_tvd = float(analysis["actual_tvd"])
                exact_prediction = (
                    qcap["error"] * uniform_distance
                )
                renyi_prediction = (
                    qcap["error"] * r2
                )

                alpha_fit = float(analysis["alpha_fit"])
                alpha_over_qcap = (
                    alpha_fit / qcap["error"]
                    if qcap["error"] > 1e-15
                    else float("nan")
                )

                tolerance = 1e-10

                records.append(
                    CircuitRecord(
                        mode=mode,
                        n_qubits=config.n_qubits,
                        depth=depth,
                        instance=instance,
                        circuit_seed=circuit_seed,
                        noise_seed=noise_seed,
                        collision=collision,
                        h2=h2,
                        tvd_q_uniform=uniform_distance,
                        renyi_factor=r2,
                        actual_tvd=actual_tvd,
                        qcap_error=qcap["error"],
                        qcap_std=qcap["std"],
                        exact_prediction=exact_prediction,
                        renyi_prediction=renyi_prediction,
                        alpha_fit=alpha_fit,
                        fit_residual_tvd=float(
                            analysis["fit_residual_tvd"]
                        ),
                        fitted_model_tvd=float(
                            analysis["fitted_model_tvd"]
                        ),
                        residual_fraction=float(
                            analysis["residual_fraction"]
                        ),
                        alpha_over_qcap=alpha_over_qcap,
                        mean_individual_rc_tvd=float(
                            analysis["mean_individual_rc_tvd"]
                        ),
                        std_individual_rc_tvd=float(
                            analysis["std_individual_rc_tvd"]
                        ),
                        max_ideal_rc_tvd=float(
                            analysis["max_ideal_rc_tvd"]
                        ),
                        qcap_covers=(
                            actual_tvd
                            <= qcap["error"] + tolerance
                        ),
                        exact_prediction_covers=(
                            actual_tvd
                            <= exact_prediction + tolerance
                        ),
                        renyi_prediction_covers=(
                            actual_tvd
                            <= renyi_prediction + tolerance
                        ),
                    )
                )

    return records, metadata


# =============================================================================
# Data export
# =============================================================================

def records_as_columns(
    records: Sequence[CircuitRecord],
) -> Dict[str, np.ndarray]:
    field_names = list(
        CircuitRecord.__dataclass_fields__.keys()
    )

    columns: Dict[str, np.ndarray] = {}

    for field_name in field_names:
        values = [
            getattr(record, field_name)
            for record in records
        ]

        if field_name == "mode":
            columns[field_name] = np.asarray(
                values,
                dtype="U16",
            )
        elif isinstance(values[0], bool):
            columns[field_name] = np.asarray(
                values,
                dtype=bool,
            )
        elif isinstance(values[0], int):
            columns[field_name] = np.asarray(
                values,
                dtype=np.int64,
            )
        else:
            columns[field_name] = np.asarray(
                values,
                dtype=float,
            )

    return columns


def save_records(
    records: Sequence[CircuitRecord],
    metadata: Mapping[str, object],
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "depolarizing_diagnostics.csv"
    npz_path = output_dir / "depolarizing_diagnostics.npz"
    metadata_path = output_dir / "depolarizing_metadata.json"

    field_names = list(
        CircuitRecord.__dataclass_fields__.keys()
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=field_names,
        )
        writer.writeheader()

        for record in records:
            writer.writerow(asdict(record))

    np.savez_compressed(
        npz_path,
        **records_as_columns(records),
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            default=str,
        )

    return {
        "csv": csv_path,
        "npz": npz_path,
        "metadata": metadata_path,
    }


# =============================================================================
# Plotting helpers
# =============================================================================

def subset(
    records: Sequence[CircuitRecord],
    *,
    mode: str | None = None,
    depth: int | None = None,
) -> List[CircuitRecord]:
    selected = list(records)

    if mode is not None:
        selected = [
            record
            for record in selected
            if record.mode == mode
        ]

    if depth is not None:
        selected = [
            record
            for record in selected
            if record.depth == depth
        ]

    return selected


def mean_and_quantiles(
    values: Sequence[float],
) -> Tuple[float, float, float]:
    array = np.asarray(values, dtype=float)

    return (
        float(np.mean(array)),
        float(np.quantile(array, 0.16)),
        float(np.quantile(array, 0.84)),
    )


def save_figure(
    figure,
    path: Path,
) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)
    print(f"Wrote {path}")


def plot_depth_summary(
    records: Sequence[CircuitRecord],
    config: StudyConfig,
) -> Path:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.2),
        sharey=True,
    )

    for axis, mode in zip(
        axes,
        ("random", "structured"),
    ):
        actual_means = []
        actual_low = []
        actual_high = []

        residual_means = []
        qcap_values = []
        exact_means = []
        renyi_means = []

        for depth in config.depths:
            group = subset(
                records,
                mode=mode,
                depth=depth,
            )

            mean, low, high = mean_and_quantiles(
                [record.actual_tvd for record in group]
            )

            actual_means.append(mean)
            actual_low.append(low)
            actual_high.append(high)

            residual_means.append(
                np.mean(
                    [
                        record.fit_residual_tvd
                        for record in group
                    ]
                )
            )

            qcap_values.append(group[0].qcap_error)

            exact_means.append(
                np.mean(
                    [
                        record.exact_prediction
                        for record in group
                    ]
                )
            )

            renyi_means.append(
                np.mean(
                    [
                        record.renyi_prediction
                        for record in group
                    ]
                )
            )

        actual_means = np.asarray(actual_means)
        actual_low = np.asarray(actual_low)
        actual_high = np.asarray(actual_high)

        axis.plot(
            config.depths,
            actual_means,
            "o-",
            label="mean actual RC-averaged TVD",
        )
        axis.fill_between(
            config.depths,
            actual_low,
            actual_high,
            alpha=0.15,
            label="16–84% circuit range",
        )
        axis.plot(
            config.depths,
            qcap_values,
            "-",
            linewidth=2.0,
            label="QCAP",
        )
        axis.plot(
            config.depths,
            exact_means,
            "-.",
            linewidth=2.0,
            label="mean depolarizing prediction",
        )
        axis.plot(
            config.depths,
            renyi_means,
            "--",
            linewidth=2.0,
            label="mean Renyi-2 prediction",
        )
        axis.plot(
            config.depths,
            residual_means,
            ":",
            linewidth=2.0,
            label="mean depolarizing-fit residual",
        )

        axis.set_xlabel("circuit depth")
        axis.set_title(f"{mode} circuits")
        axis.set_xticks(config.depths)
        axis.grid(True, alpha=0.2)
        axis.legend(frameon=False, fontsize=8.2)

    axes[0].set_ylabel("TVD")

    figure.suptitle(
        "Output error, entropy predictions, and white-noise fit residual",
        fontsize=12,
    )

    path = config.output_dir / "depth_summary.png"
    save_figure(figure, path)

    return path


def plot_alpha_vs_qcap(
    records: Sequence[CircuitRecord],
    config: StudyConfig,
) -> Path:
    figure, axis = plt.subplots(figsize=(7.0, 6.2))

    for mode in ("random", "structured"):
        group = subset(records, mode=mode)

        axis.scatter(
            [record.qcap_error for record in group],
            [record.alpha_fit for record in group],
            s=18,
            alpha=0.5,
            label=mode,
        )

    limit = max(
        max(record.qcap_error for record in records),
        max(record.alpha_fit for record in records),
    )

    axis.plot(
        [0.0, limit],
        [0.0, limit],
        "--",
        linewidth=1.4,
        label=r"$\alpha_{\rm fit}=\epsilon_{\rm QCAP}$",
    )

    axis.set_xlabel("QCAP error parameter")
    axis.set_ylabel("best-fit global-depolarizing alpha")
    axis.set_title("Does QCAP estimate the fitted white-noise strength?")
    axis.grid(True, alpha=0.2)
    axis.legend(frameon=False)

    path = config.output_dir / "alpha_fit_vs_qcap.png"
    save_figure(figure, path)

    return path


def plot_residual_vs_entropy(
    records: Sequence[CircuitRecord],
    config: StudyConfig,
) -> Path:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.0),
    )

    for mode in ("random", "structured"):
        group = subset(records, mode=mode)

        axes[0].scatter(
            [record.h2 for record in group],
            [record.fit_residual_tvd for record in group],
            s=18,
            alpha=0.5,
            label=mode,
        )

        axes[1].scatter(
            [record.tvd_q_uniform for record in group],
            [record.fit_residual_tvd for record in group],
            s=18,
            alpha=0.5,
            label=mode,
        )

    axes[0].set_xlabel(r"$H_2(q)$")
    axes[0].set_ylabel("best-fit residual TVD")
    axes[0].set_title("Residual versus collision entropy")

    axes[1].set_xlabel(r"$d_{\rm TV}(q,u)$")
    axes[1].set_ylabel("best-fit residual TVD")
    axes[1].set_title("Residual versus ideal nonuniformity")

    for axis in axes:
        axis.grid(True, alpha=0.2)
        axis.legend(frameon=False)

    path = config.output_dir / "residual_vs_entropy.png"
    save_figure(figure, path)

    return path


def plot_entropy_distributions(
    records: Sequence[CircuitRecord],
    config: StudyConfig,
) -> Path:
    deepest = max(config.depths)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.7),
    )

    for mode in ("random", "structured"):
        group = subset(
            records,
            mode=mode,
            depth=deepest,
        )

        axes[0].hist(
            [record.h2 for record in group],
            bins=28,
            alpha=0.55,
            density=True,
            label=mode,
        )

        axes[1].hist(
            [record.tvd_q_uniform for record in group],
            bins=28,
            alpha=0.55,
            density=True,
            label=mode,
        )

    axes[0].set_xlabel(r"$H_2(q)$")
    axes[0].set_ylabel("density")
    axes[0].set_title(f"Collision entropy at depth {deepest}")

    axes[1].set_xlabel(r"$d_{\rm TV}(q,u)$")
    axes[1].set_ylabel("density")
    axes[1].set_title(f"Ideal distance to uniform at depth {deepest}")

    for axis in axes:
        axis.grid(True, alpha=0.2)
        axis.legend(frameon=False)

    path = config.output_dir / "entropy_distributions.png"
    save_figure(figure, path)

    return path


def plot_coverage(
    records: Sequence[CircuitRecord],
    config: StudyConfig,
) -> Path:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 4.8),
        sharey=True,
    )

    for axis, mode in zip(
        axes,
        ("random", "structured"),
    ):
        qcap_coverage = []
        exact_coverage = []
        renyi_coverage = []

        for depth in config.depths:
            group = subset(
                records,
                mode=mode,
                depth=depth,
            )

            qcap_coverage.append(
                np.mean(
                    [
                        record.qcap_covers
                        for record in group
                    ]
                )
            )
            exact_coverage.append(
                np.mean(
                    [
                        record.exact_prediction_covers
                        for record in group
                    ]
                )
            )
            renyi_coverage.append(
                np.mean(
                    [
                        record.renyi_prediction_covers
                        for record in group
                    ]
                )
            )

        axis.plot(
            config.depths,
            qcap_coverage,
            "o-",
            label="QCAP",
        )
        axis.plot(
            config.depths,
            exact_coverage,
            "s-",
            label="depolarizing prediction",
        )
        axis.plot(
            config.depths,
            renyi_coverage,
            "^-",
            label="Renyi-2 prediction",
        )

        axis.set_xlabel("circuit depth")
        axis.set_title(mode)
        axis.set_xticks(config.depths)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(True, alpha=0.2)
        axis.legend(frameon=False)

    axes[0].set_ylabel("fraction of circuit instances covered")

    figure.suptitle(
        "Empirical coverage of central prediction curves",
        fontsize=12,
    )

    path = config.output_dir / "coverage_fraction.png"
    save_figure(figure, path)

    return path


def plot_sample_distributions(
    config: StudyConfig,
    noise: NoiseModel,
) -> Path:
    cases = [
        ("random", min(config.depths)),
        ("random", max(config.depths)),
        ("structured", min(config.depths)),
        ("structured", max(config.depths)),
    ]

    figure, axes = plt.subplots(
        len(cases),
        1,
        figsize=(12.0, 10.5),
        sharex=True,
    )

    for row, (mode, depth) in enumerate(cases):
        circuit_seed = (
            config.seed
            + (0 if mode == "random" else 10_000_000)
            + 100_000 * depth
        )
        noise_seed = (
            config.seed
            + (0 if mode == "random" else 20_000_000)
            + 1_000_000 * depth
        )

        circuit = build_target_circuit(
            config.n_qubits,
            depth,
            mode,
            circuit_seed,
        )

        analysis = analyze_rc_target(
            circuit,
            noise,
            n_randomizations=config.rc_randomizations,
            trajectories_per_rc=config.trajectories_per_rc,
            seed=noise_seed,
            ideal_tolerance=config.ideal_rc_tolerance,
        )

        q = np.asarray(analysis["q"])
        p = np.asarray(analysis["p_average"])
        model = np.asarray(analysis["model"])

        x = np.arange(q.size)

        axes[row].plot(
            x,
            q,
            linewidth=1.2,
            label="ideal q",
        )
        axes[row].plot(
            x,
            p,
            linewidth=1.2,
            label="noisy RC average p",
        )
        axes[row].plot(
            x,
            model,
            "--",
            linewidth=1.4,
            label="best global-depolarizing fit",
        )

        axes[row].set_ylabel("probability")
        axes[row].set_title(
            f"{mode}, depth={depth}, "
            f"alpha={analysis['alpha_fit']:.3f}, "
            f"residual={analysis['fit_residual_tvd']:.3g}"
        )
        axes[row].grid(True, alpha=0.15)
        axes[row].legend(frameon=False, fontsize=8)

    axes[-1].set_xlabel("bitstring index")

    figure.suptitle(
        "Representative ideal, noisy, and best-fit output distributions",
        fontsize=12,
    )

    path = config.output_dir / "sample_distributions.png"
    save_figure(figure, path)

    return path


# =============================================================================
# Summary report and next-step recommendations
# =============================================================================

def write_summary(
    records: Sequence[CircuitRecord],
    config: StudyConfig,
) -> Path:
    path = config.output_dir / "diagnostic_summary.txt"

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "Global-depolarizing model diagnostic summary\n"
        )
        handle.write("=" * 45 + "\n\n")

        handle.write(
            f"n_qubits: {config.n_qubits}\n"
        )
        handle.write(
            f"depths: {list(config.depths)}\n"
        )
        handle.write(
            f"instances per mode/depth: {config.n_instances}\n"
        )
        handle.write(
            f"RC randomizations per circuit: "
            f"{config.rc_randomizations}\n"
        )
        handle.write(
            f"trajectories per RC circuit: "
            f"{config.trajectories_per_rc}\n\n"
        )

        for mode in ("random", "structured"):
            handle.write(f"{mode.upper()}\n")
            handle.write("-" * len(mode) + "\n")

            for depth in config.depths:
                group = subset(
                    records,
                    mode=mode,
                    depth=depth,
                )

                mean_residual = np.mean(
                    [
                        record.fit_residual_tvd
                        for record in group
                    ]
                )
                median_residual_fraction = np.median(
                    [
                        record.residual_fraction
                        for record in group
                    ]
                )
                mean_alpha_ratio = np.nanmean(
                    [
                        record.alpha_over_qcap
                        for record in group
                    ]
                )
                qcap_coverage = np.mean(
                    [
                        record.qcap_covers
                        for record in group
                    ]
                )
                exact_coverage = np.mean(
                    [
                        record.exact_prediction_covers
                        for record in group
                    ]
                )
                renyi_coverage = np.mean(
                    [
                        record.renyi_prediction_covers
                        for record in group
                    ]
                )

                handle.write(
                    f"depth={depth:>4}: "
                    f"mean residual={mean_residual:.6g}, "
                    f"median residual/actual={median_residual_fraction:.4f}, "
                    f"mean alpha/QCAP={mean_alpha_ratio:.4f}, "
                    f"coverage(QCAP/exact/R2)="
                    f"{qcap_coverage:.3f}/"
                    f"{exact_coverage:.3f}/"
                    f"{renyi_coverage:.3f}\n"
                )

            handle.write("\n")

        handle.write("Interpretation guide\n")
        handle.write("--------------------\n")
        handle.write(
            "1. Small fit residuals for both ensembles support the claim that "
            "the measured output channel is approximately a global mixture with "
            "uniform noise, even when the circuit is structured.\n"
        )
        handle.write(
            "2. If structured residuals are large but the entropy prediction "
            "still covers the TVD, the apparent success is not explained by the "
            "global-depolarizing model and should be treated as empirical.\n"
        )
        handle.write(
            "3. If alpha_fit tracks QCAP error, CB/QCAP is estimating the same "
            "effective white-noise strength used by the entropy correction.\n"
        )
        handle.write(
            "4. If alpha_fit is systematically below QCAP, the baseline remains "
            "conservative and the entropy correction may still work as an upper "
            "prediction. If alpha_fit exceeds QCAP, violations are expected.\n"
        )
        handle.write(
            "5. Repeat with different noise mechanisms separately: stochastic "
            "Pauli only, coherent only, spectator/correlated, and drift. This "
            "identifies which channels preserve or destroy the white-noise fit.\n"
        )

    print(f"Wrote {path}")
    return path


# =============================================================================
# CLI
# =============================================================================

def parse_depths(text: str) -> Tuple[int, ...]:
    values = tuple(
        int(piece.strip())
        for piece in text.split(",")
        if piece.strip()
    )

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one depth is required."
        )

    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a global-depolarizing output model to random and structured "
            "circuits under CB, RC, and generalized trajectory noise."
        )
    )

    parser.add_argument(
        "--qubits",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--depths",
        type=parse_depths,
        default=(1, 2, 4, 8, 16, 32),
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--rc-randomizations",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--trajectories",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--cb-depths",
        type=parse_depths,
        default=(1, 2, 4, 8, 16, 24),
    )
    parser.add_argument(
        "--cb-decays",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--cb-trajectories",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/depolarizing_diagnostics"),
    )
    parser.add_argument(
        "--ideal-rc-tolerance",
        type=float,
        default=1e-9,
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = StudyConfig(
        n_qubits=args.qubits,
        depths=tuple(args.depths),
        n_instances=args.instances,
        rc_randomizations=args.rc_randomizations,
        trajectories_per_rc=args.trajectories,
        cb_depths=tuple(args.cb_depths),
        cb_decays=args.cb_decays,
        cb_trajectories=args.cb_trajectories,
        seed=args.seed,
        output_dir=args.output_dir,
        ideal_rc_tolerance=args.ideal_rc_tolerance,
    )

    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Start with the stochastic Pauli model. Then run controlled ablations by
    # turning on coherent, spectator, correlated, or drift terms one at a time.
    noise = NoiseModel(
        enabled=True,
        p1=1e-3,
        p2=1e-2,
        p_idle_z=1e-3,
        p_readout_01=1e-2,
        p_readout_10=1e-2,
        oneq_overrotation=0.0,
        twoq_zz_overrotation=0.0,
        controlled_phase_offset=0.0,
        idle_z_angle=0.0,
        spectator_z_probability=0.0,
        spectator_z_angle=0.0,
        correlated_fault_probability=0.0,
        stochastic_rate_drift_std=0.0,
        coherent_angle_drift_std=0.0,
    )

    total_noisy_statevectors = (
        2
        * len(config.depths)
        * config.n_instances
        * config.rc_randomizations
        * config.trajectories_per_rc
    )

    print("Starting depolarizing-model diagnostic")
    print(f"qubits: {config.n_qubits}")
    print(f"depths: {config.depths}")
    print(f"instances per mode/depth: {config.n_instances}")
    print(f"RC randomizations: {config.rc_randomizations}")
    print(
        f"trajectories per RC circuit: "
        f"{config.trajectories_per_rc}"
    )
    print(
        f"approximate target noisy statevector calls: "
        f"{total_noisy_statevectors:,}"
    )

    records, metadata = run_study(
        config,
        noise,
    )

    paths = save_records(
        records,
        metadata,
        config.output_dir,
    )

    plot_depth_summary(records, config)
    plot_alpha_vs_qcap(records, config)
    plot_residual_vs_entropy(records, config)
    plot_entropy_distributions(records, config)
    plot_coverage(records, config)
    plot_sample_distributions(config, noise)
    write_summary(records, config)

    print("\nData files:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
