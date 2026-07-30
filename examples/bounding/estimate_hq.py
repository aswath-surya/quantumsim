from __future__ import annotations

from pathlib import Path

import csv
import math
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from proxysim.backends import StatevectorBackend
from proxysim.circuit import Circuit

"""Numerical study of H_2(q) for sufficiently scrambling random circuits.

This script generates ensembles of random brickwork circuits and studies

    C_2(q) = sum_x q(x)^2
    H_2(q) = -log_2 C_2(q)
    R_2(q) = 1/2 * sqrt(2^n C_2(q) - 1)

where q is the ideal computational-basis output distribution.

For Haar-random states in dimension D = 2^n,

    E[C_2] = 2 / (D + 1),

so the Porter-Thomas / Haar reference curves are

    H_2,PT(n) = log_2((D + 1)/2),
    R_2,PT(n) = 1/2 * sqrt((D - 1)/(D + 1)).

The script produces:

  1. sample bitstring distributions for shallow, intermediate, and deep circuits;
  2. mean H_2 versus qubit number, with a finite-size fit and Haar reference;
  3. mean R_2 versus qubit number, with a finite-size fit and Haar reference;
  4. variance of H_2 and R_2 versus qubit number;
  5. depth-convergence plots for C_2, H_2, and R_2;
  6. histograms of R_2 for selected qubit counts;
  7. a CSV/NPZ data export and a text summary of fitted parameters.

The random circuit ensemble uses one Haar-random single-qubit U gate per qubit
per layer, followed by alternating nearest-neighbor CZ brickwork layers.

Run:
    python examples/run_scrambling_h2.py

Important:
    Exact statevector simulation scales exponentially. Start with modest values
    such as n <= 12. The default ensemble size is intentionally smaller than
    1000 so the script is practical on a laptop. Set N_CIRCUITS = 1000 for the
    full study once the workflow is validated.
"""




# =============================================================================
# Configuration
# =============================================================================

# Keep this modest for a first run. Increase to range(2, 15) or similar on a
# workstation.
N_QUBITS = list(range(2, 11))

# Depths count full brickwork repetitions:
#   random U layer + CZ-even + random U layer + CZ-odd
DEPTHS = [1, 2, 4, 8, 16, 32, 64]

# Use 100-250 while debugging. Set to 1000 for the intended ensemble study.
N_CIRCUITS = 200

SEED = 12345

from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURE_DIR = RESULTS_DIR / "scrambling_h2_figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULTS_DIR / "scrambling_h2_data.csv"
NPZ_PATH = RESULTS_DIR / "scrambling_h2_data.npz"
SUMMARY_PATH = RESULTS_DIR / "scrambling_h2_fit_summary.txt"

_SV = StatevectorBackend()


# =============================================================================
# Circuit construction
# =============================================================================

def even_pairs(n: int) -> List[Tuple[int, int]]:
    """Nearest-neighbor pairs (0,1), (2,3), ..."""
    return [(q, q + 1) for q in range(0, n - 1, 2)]


def odd_pairs(n: int) -> List[Tuple[int, int]]:
    """Nearest-neighbor pairs (1,2), (3,4), ..."""
    return [(q, q + 1) for q in range(1, n - 1, 2)]


def add_haar_random_u_layer(
    circuit: Circuit,
    rng: random.Random,
) -> None:
    """Append one Haar-random single-qubit U gate to every qubit.

    For a Haar-random pure qubit state, cos(theta) is uniform on [-1,1].
    The remaining Euler angles are uniform on [0, 2pi).

    The repository's Circuit.add API is assumed to accept:

        circuit.add(name, *qubits, params=(...))
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


def add_cz_layer(
    circuit: Circuit,
    pairs: Sequence[Tuple[int, int]],
) -> None:
    for a, b in pairs:
        circuit.add("cz", a, b)


def random_brickwork_circuit(
    n_qubits: int,
    depth: int,
    seed: int,
) -> Circuit:
    """Build a random non-Clifford brickwork circuit."""
    rng = random.Random(seed)

    circuit = Circuit(
        n_qubits,
        name=f"scrambling_n{n_qubits}_d{depth}_s{seed}",
    )

    cycle_a = even_pairs(n_qubits)
    cycle_b = odd_pairs(n_qubits)

    for _ in range(depth):
        add_haar_random_u_layer(circuit, rng)
        add_cz_layer(circuit, cycle_a)

        add_haar_random_u_layer(circuit, rng)
        add_cz_layer(circuit, cycle_b)

    add_haar_random_u_layer(circuit, rng)

    return circuit


# =============================================================================
# Probability and entropy helpers
# =============================================================================

def dense_distribution(circuit: Circuit) -> np.ndarray:
    """Return the exact ideal output distribution as a dense vector."""
    distribution = _SV.exact_distribution(circuit)
    dimension = 2 ** circuit.n_qubits

    if isinstance(distribution, dict):
        dense = np.zeros(dimension, dtype=float)

        for key, probability in distribution.items():
            if isinstance(key, str):
                index = int(key.replace(" ", ""), 2)
            else:
                index = int(key)

            dense[index] += float(probability)
    else:
        dense = np.asarray(distribution, dtype=float).reshape(-1)

    if dense.size != dimension:
        raise ValueError(
            f"Expected {dimension} probabilities, received {dense.size}."
        )

    dense = np.clip(dense, 0.0, None)
    total = float(np.sum(dense))

    if total <= 0.0:
        raise ValueError("Distribution has zero total probability.")

    return dense / total


def collision_probability(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=float)
    return float(np.sum(q * q))


def renyi2_entropy(q: np.ndarray) -> float:
    c2 = collision_probability(q)
    return -math.log2(c2)


def renyi2_factor(q: np.ndarray) -> float:
    """Return the square-root factor before multiplication by a CB/QCAP error."""
    q = np.asarray(q, dtype=float)
    dimension = q.size
    c2 = collision_probability(q)

    return 0.5 * math.sqrt(
        max(dimension * c2 - 1.0, 0.0)
    )


def tvd_to_uniform(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=float)
    uniform = np.full(q.size, 1.0 / q.size)

    return 0.5 * float(np.sum(np.abs(q - uniform)))


def porter_thomas_reference(n_qubits: int) -> Dict[str, float]:
    """Haar-state reference values for dimension D = 2^n."""
    dimension = 2 ** n_qubits

    mean_c2 = 2.0 / (dimension + 1.0)
    h2_from_mean_c2 = math.log2((dimension + 1.0) / 2.0)
    r2_from_mean_c2 = 0.5 * math.sqrt(
        (dimension - 1.0) / (dimension + 1.0)
    )

    # Exact Haar variance of C2 = sum_i p_i^2.
    variance_c2 = (
        4.0 * (dimension - 1.0)
        / (
            (dimension + 1.0) ** 2
            * (dimension + 2.0)
            * (dimension + 3.0)
        )
    )

    return {
        "mean_c2": mean_c2,
        "h2_from_mean_c2": h2_from_mean_c2,
        "r2_from_mean_c2": r2_from_mean_c2,
        "variance_c2": variance_c2,
    }


# =============================================================================
# Data model and ensemble simulation
# =============================================================================

@dataclass
class Record:
    n_qubits: int
    depth: int
    instance: int
    seed: int
    collision: float
    h2: float
    r2_factor: float
    tvd_uniform: float


def simulate_ensemble() -> List[Record]:
    records: List[Record] = []

    total_groups = len(N_QUBITS) * len(DEPTHS)
    group_index = 0

    for n_qubits in N_QUBITS:
        for depth in DEPTHS:
            group_index += 1
            print(
                f"[{group_index:>3}/{total_groups}] "
                f"n={n_qubits:>2}, depth={depth:>3}",
                flush=True,
            )

            for instance in range(N_CIRCUITS):
                circuit_seed = (
                    SEED
                    + 10_000_000 * n_qubits
                    + 100_000 * depth
                    + instance
                )

                circuit = random_brickwork_circuit(
                    n_qubits,
                    depth,
                    circuit_seed,
                )

                q = dense_distribution(circuit)
                c2 = collision_probability(q)

                records.append(
                    Record(
                        n_qubits=n_qubits,
                        depth=depth,
                        instance=instance,
                        seed=circuit_seed,
                        collision=c2,
                        h2=-math.log2(c2),
                        r2_factor=renyi2_factor(q),
                        tvd_uniform=tvd_to_uniform(q),
                    )
                )

    return records


def records_to_arrays(records: Sequence[Record]) -> Dict[str, np.ndarray]:
    return {
        "n_qubits": np.asarray([r.n_qubits for r in records], dtype=int),
        "depth": np.asarray([r.depth for r in records], dtype=int),
        "instance": np.asarray([r.instance for r in records], dtype=int),
        "seed": np.asarray([r.seed for r in records], dtype=np.int64),
        "collision": np.asarray([r.collision for r in records], dtype=float),
        "h2": np.asarray([r.h2 for r in records], dtype=float),
        "r2_factor": np.asarray([r.r2_factor for r in records], dtype=float),
        "tvd_uniform": np.asarray([r.tvd_uniform for r in records], dtype=float),
    }


def save_data(records: Sequence[Record]) -> None:
    arrays = records_to_arrays(records)

    np.savez_compressed(
        NPZ_PATH,
        **arrays,
        n_circuits=N_CIRCUITS,
        qubit_grid=np.asarray(N_QUBITS, dtype=int),
        depth_grid=np.asarray(DEPTHS, dtype=int),
    )

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "n_qubits",
                "depth",
                "instance",
                "seed",
                "collision",
                "h2",
                "r2_factor",
                "tvd_uniform",
            ]
        )

        for record in records:
            writer.writerow(
                [
                    record.n_qubits,
                    record.depth,
                    record.instance,
                    record.seed,
                    record.collision,
                    record.h2,
                    record.r2_factor,
                    record.tvd_uniform,
                ]
            )


# =============================================================================
# Aggregation and fitting
# =============================================================================

def select(
    arrays: Dict[str, np.ndarray],
    *,
    n_qubits: int | None = None,
    depth: int | None = None,
) -> np.ndarray:
    mask = np.ones(arrays["n_qubits"].shape, dtype=bool)

    if n_qubits is not None:
        mask &= arrays["n_qubits"] == n_qubits

    if depth is not None:
        mask &= arrays["depth"] == depth

    return mask


def group_stats(
    arrays: Dict[str, np.ndarray],
    field: str,
    *,
    depth: int,
) -> Dict[str, np.ndarray]:
    means = []
    stds = []
    variances = []
    medians = []
    q16 = []
    q84 = []

    for n_qubits in N_QUBITS:
        values = arrays[field][
            select(
                arrays,
                n_qubits=n_qubits,
                depth=depth,
            )
        ]

        means.append(float(np.mean(values)))
        stds.append(float(np.std(values, ddof=1)))
        variances.append(float(np.var(values, ddof=1)))
        medians.append(float(np.median(values)))
        q16.append(float(np.quantile(values, 0.16)))
        q84.append(float(np.quantile(values, 0.84)))

    return {
        "mean": np.asarray(means),
        "std": np.asarray(stds),
        "variance": np.asarray(variances),
        "median": np.asarray(medians),
        "q16": np.asarray(q16),
        "q84": np.asarray(q84),
    }


def fit_h2_mean(
    n_values: np.ndarray,
    means: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit H2(n) = n - c + a*2^{-n}."""

    def model(n, c, a):
        return n - c + a * 2.0 ** (-n)

    popt, pcov = curve_fit(
        model,
        n_values,
        means,
        p0=(1.0, 1.0),
        maxfev=20_000,
    )

    return popt, pcov


def fit_r2_mean(
    n_values: np.ndarray,
    means: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit R2(n) = R_inf + a*2^{-n} + b*2^{-2n}."""

    def model(n, r_inf, a, b):
        x = 2.0 ** (-n)
        return r_inf + a * x + b * x * x

    popt, pcov = curve_fit(
        model,
        n_values,
        means,
        p0=(0.5, -0.5, 0.0),
        maxfev=20_000,
    )

    return popt, pcov


def fit_variance_scaling(
    n_values: np.ndarray,
    variances: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit Var(n) = a*2^{-n} + b*2^{-2n}."""

    def model(n, a, b):
        x = 2.0 ** (-n)
        return a * x + b * x * x

    popt, pcov = curve_fit(
        model,
        n_values,
        variances,
        p0=(1.0, 0.0),
        maxfev=20_000,
    )

    return popt, pcov


# =============================================================================
# Plot helpers
# =============================================================================

def save_figure(fig, filename: str) -> None:
    path = FIGURE_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_sample_bitstring_distributions() -> None:
    """Plot representative output distributions at several depths."""
    selected_n = max(N_QUBITS)
    candidate_depths = [
        DEPTHS[0],
        DEPTHS[len(DEPTHS) // 2],
        DEPTHS[-1],
    ]

    fig, axes = plt.subplots(
        len(candidate_depths),
        1,
        figsize=(12.0, 8.5),
        sharex=True,
    )

    if len(candidate_depths) == 1:
        axes = [axes]

    for axis, depth in zip(axes, candidate_depths):
        seed = (
            SEED
            + 10_000_000 * selected_n
            + 100_000 * depth
        )

        circuit = random_brickwork_circuit(
            selected_n,
            depth,
            seed,
        )
        q = dense_distribution(circuit)

        x = np.arange(q.size)

        axis.bar(
            x,
            q,
            width=1.0,
            alpha=0.85,
        )

        axis.axhline(
            1.0 / q.size,
            linestyle="--",
            linewidth=1.2,
            label="uniform probability",
        )

        axis.set_ylabel("q(x)")
        axis.set_title(
            f"n={selected_n}, depth={depth}, "
            f"H2={renyi2_entropy(q):.3f}, "
            f"R2={renyi2_factor(q):.3f}"
        )

        axis.legend(
            frameon=False,
            fontsize=8.5,
        )

    axes[-1].set_xlabel("bitstring index")

    fig.suptitle(
        "Representative ideal output distributions",
        fontsize=12,
    )

    save_figure(
        fig,
        "sample_bitstring_distributions.png",
    )


def plot_mean_h2_and_fit(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    depth = DEPTHS[-1]
    stats = group_stats(arrays, "h2", depth=depth)

    n_values = np.asarray(N_QUBITS, dtype=float)
    pt = np.asarray(
        [
            porter_thomas_reference(n)["h2_from_mean_c2"]
            for n in N_QUBITS
        ]
    )

    popt, pcov = fit_h2_mean(
        n_values,
        stats["mean"],
    )

    c_fit, a_fit = popt

    dense_n = np.linspace(
        min(N_QUBITS),
        max(N_QUBITS),
        300,
    )
    fitted_curve = (
        dense_n
        - c_fit
        + a_fit * 2.0 ** (-dense_n)
    )

    fig, axis = plt.subplots(figsize=(8.2, 5.2))

    axis.errorbar(
        N_QUBITS,
        stats["mean"],
        yerr=stats["std"],
        fmt="o",
        capsize=3,
        label="ensemble mean ± std",
    )

    axis.plot(
        N_QUBITS,
        pt,
        "--",
        linewidth=2.0,
        label="Haar / Porter-Thomas reference",
    )

    axis.plot(
        dense_n,
        fitted_curve,
        "-",
        linewidth=2.0,
        label=(
            r"fit: $H_2=n-c+a2^{-n}$"
            + f"\n"
            + f"c={c_fit:.4f}, a={a_fit:.4f}"
        ),
    )

    axis.set_xlabel("number of qubits n")
    axis.set_ylabel(r"$H_2(q)$")
    axis.set_title(
        f"Collision entropy at depth {depth}"
    )
    axis.grid(True, alpha=0.2)
    axis.legend(frameon=False)

    save_figure(
        fig,
        "mean_h2_vs_qubits.png",
    )

    return {
        "params": popt,
        "covariance": pcov,
        "stats_mean": stats["mean"],
        "stats_std": stats["std"],
    }


def plot_mean_r2_and_fit(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    depth = DEPTHS[-1]
    stats = group_stats(arrays, "r2_factor", depth=depth)

    n_values = np.asarray(N_QUBITS, dtype=float)
    pt = np.asarray(
        [
            porter_thomas_reference(n)["r2_from_mean_c2"]
            for n in N_QUBITS
        ]
    )

    popt, pcov = fit_r2_mean(
        n_values,
        stats["mean"],
    )

    r_inf, a_fit, b_fit = popt

    dense_n = np.linspace(
        min(N_QUBITS),
        max(N_QUBITS),
        300,
    )
    x = 2.0 ** (-dense_n)
    fitted_curve = r_inf + a_fit * x + b_fit * x * x

    fig, axis = plt.subplots(figsize=(8.2, 5.2))

    axis.errorbar(
        N_QUBITS,
        stats["mean"],
        yerr=stats["std"],
        fmt="o",
        capsize=3,
        label="ensemble mean ± std",
    )

    axis.plot(
        N_QUBITS,
        pt,
        "--",
        linewidth=2.0,
        label="Haar / Porter-Thomas reference",
    )

    axis.plot(
        dense_n,
        fitted_curve,
        "-",
        linewidth=2.0,
        label=(
            r"fit: $R_2=R_\infty+a2^{-n}+b2^{-2n}$"
            + "\n"
            + f"R∞={r_inf:.4f}"
        ),
    )

    axis.axhline(
        0.5,
        linestyle=":",
        linewidth=1.4,
        label=r"expected asymptote $R_\infty=1/2$",
    )

    axis.set_xlabel("number of qubits n")
    axis.set_ylabel(r"$R_2(q)=\frac12\sqrt{2^n\sum_xq_x^2-1}$")
    axis.set_title(
        f"Rényi-2 square-root factor at depth {depth}"
    )
    axis.grid(True, alpha=0.2)
    axis.legend(frameon=False)

    save_figure(
        fig,
        "mean_r2_factor_vs_qubits.png",
    )

    return {
        "params": popt,
        "covariance": pcov,
        "stats_mean": stats["mean"],
        "stats_std": stats["std"],
    }


def plot_variance_scaling(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    depth = DEPTHS[-1]
    h2_stats = group_stats(arrays, "h2", depth=depth)
    r2_stats = group_stats(arrays, "r2_factor", depth=depth)

    n_values = np.asarray(N_QUBITS, dtype=float)

    h2_popt, h2_pcov = fit_variance_scaling(
        n_values,
        h2_stats["variance"],
    )
    r2_popt, r2_pcov = fit_variance_scaling(
        n_values,
        r2_stats["variance"],
    )

    dense_n = np.linspace(
        min(N_QUBITS),
        max(N_QUBITS),
        300,
    )
    x = 2.0 ** (-dense_n)

    h2_fit = h2_popt[0] * x + h2_popt[1] * x * x
    r2_fit = r2_popt[0] * x + r2_popt[1] * x * x

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.8),
    )

    axes[0].plot(
        N_QUBITS,
        h2_stats["variance"],
        "o",
        label="measured variance",
    )
    axes[0].plot(
        dense_n,
        h2_fit,
        "-",
        label=r"fit: $a2^{-n}+b2^{-2n}$",
    )
    axes[0].set_title(r"$\mathrm{Var}[H_2]$")
    axes[0].set_xlabel("number of qubits n")
    axes[0].set_ylabel("variance")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(frameon=False)

    axes[1].plot(
        N_QUBITS,
        r2_stats["variance"],
        "o",
        label="measured variance",
    )
    axes[1].plot(
        dense_n,
        r2_fit,
        "-",
        label=r"fit: $a2^{-n}+b2^{-2n}$",
    )
    axes[1].set_title(r"$\mathrm{Var}[R_2]$")
    axes[1].set_xlabel("number of qubits n")
    axes[1].set_ylabel("variance")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(frameon=False)

    fig.suptitle(
        f"Ensemble-variance scaling at depth {depth}",
        fontsize=12,
    )

    save_figure(
        fig,
        "variance_scaling.png",
    )

    return {
        "h2_params": h2_popt,
        "h2_covariance": h2_pcov,
        "r2_params": r2_popt,
        "r2_covariance": r2_pcov,
    }


def plot_depth_convergence(arrays: Dict[str, np.ndarray]) -> None:
    selected_n = [
        N_QUBITS[0],
        N_QUBITS[len(N_QUBITS) // 2],
        N_QUBITS[-1],
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.0, 4.5),
    )

    fields = [
        ("collision", r"$C_2=\sum_xq_x^2$"),
        ("h2", r"$H_2(q)$"),
        ("r2_factor", r"$R_2(q)$"),
    ]

    for n_qubits in selected_n:
        for axis, (field, ylabel) in zip(axes, fields):
            means = []
            stds = []

            for depth in DEPTHS:
                values = arrays[field][
                    select(
                        arrays,
                        n_qubits=n_qubits,
                        depth=depth,
                    )
                ]

                means.append(float(np.mean(values)))
                stds.append(float(np.std(values, ddof=1)))

            means = np.asarray(means)
            stds = np.asarray(stds)

            axis.plot(
                DEPTHS,
                means,
                "o-",
                label=f"n={n_qubits}",
            )
            axis.fill_between(
                DEPTHS,
                means - stds,
                means + stds,
                alpha=0.12,
            )

            axis.set_xscale("log", base=2)
            axis.set_xlabel("circuit depth")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.2)

    for axis in axes:
        axis.legend(frameon=False)

    fig.suptitle(
        "Approach to depth-saturated scrambling statistics",
        fontsize=12,
    )

    save_figure(
        fig,
        "depth_convergence.png",
    )


def plot_r2_histograms(arrays: Dict[str, np.ndarray]) -> None:
    depth = DEPTHS[-1]

    selected_n = [
        N_QUBITS[0],
        N_QUBITS[len(N_QUBITS) // 2],
        N_QUBITS[-1],
    ]

    fig, axes = plt.subplots(
        1,
        len(selected_n),
        figsize=(13.0, 4.2),
        sharey=True,
    )

    if len(selected_n) == 1:
        axes = [axes]

    for axis, n_qubits in zip(axes, selected_n):
        values = arrays["r2_factor"][
            select(
                arrays,
                n_qubits=n_qubits,
                depth=depth,
            )
        ]

        pt_value = porter_thomas_reference(
            n_qubits
        )["r2_from_mean_c2"]

        axis.hist(
            values,
            bins=24,
            density=True,
            alpha=0.8,
        )
        axis.axvline(
            np.mean(values),
            linewidth=2.0,
            label="ensemble mean",
        )
        axis.axvline(
            pt_value,
            linestyle="--",
            linewidth=2.0,
            label="Haar reference",
        )

        axis.set_title(f"n={n_qubits}")
        axis.set_xlabel(r"$R_2(q)$")
        axis.grid(True, alpha=0.2)
        axis.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel("density")

    fig.suptitle(
        f"Distribution of the Rényi-2 factor at depth {depth}",
        fontsize=12,
    )

    save_figure(
        fig,
        "r2_histograms.png",
    )


def plot_tvd_vs_r2(arrays: Dict[str, np.ndarray]) -> None:
    """Compare the exact TVD-to-uniform factor with its R2 upper relaxation."""
    depth = DEPTHS[-1]

    fig, axis = plt.subplots(figsize=(7.0, 6.0))

    for n_qubits in N_QUBITS:
        mask = select(
            arrays,
            n_qubits=n_qubits,
            depth=depth,
        )

        axis.scatter(
            arrays["r2_factor"][mask],
            arrays["tvd_uniform"][mask],
            s=18,
            alpha=0.5,
            label=f"n={n_qubits}",
        )

    max_value = max(
        float(np.max(arrays["r2_factor"])),
        float(np.max(arrays["tvd_uniform"])),
    )

    axis.plot(
        [0.0, max_value],
        [0.0, max_value],
        "--",
        linewidth=1.4,
        label="equality line",
    )

    axis.set_xlabel(r"Rényi-2 relaxation $R_2(q)$")
    axis.set_ylabel(r"exact $d_{\rm TV}(q,u)$")
    axis.set_title(
        f"Exact uniform-distance factor versus Rényi-2 relaxation, depth {depth}"
    )
    axis.grid(True, alpha=0.2)
    axis.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
    )

    save_figure(
        fig,
        "tvd_uniform_vs_r2_factor.png",
    )


# =============================================================================
# Fit summary
# =============================================================================

def write_fit_summary(
    h2_fit: Dict[str, np.ndarray],
    r2_fit: Dict[str, np.ndarray],
    variance_fit: Dict[str, np.ndarray],
) -> None:
    h2_params = h2_fit["params"]
    h2_errors = np.sqrt(np.diag(h2_fit["covariance"]))

    r2_params = r2_fit["params"]
    r2_errors = np.sqrt(np.diag(r2_fit["covariance"]))

    h2_var_params = variance_fit["h2_params"]
    h2_var_errors = np.sqrt(
        np.diag(variance_fit["h2_covariance"])
    )

    r2_var_params = variance_fit["r2_params"]
    r2_var_errors = np.sqrt(
        np.diag(variance_fit["r2_covariance"])
    )

    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        handle.write("Scrambling H2 numerical fit summary\n")
        handle.write("=" * 42 + "\n\n")

        handle.write(
            f"Ensemble size per (n, depth): {N_CIRCUITS}\n"
        )
        handle.write(f"Qubit counts: {N_QUBITS}\n")
        handle.write(f"Depths: {DEPTHS}\n")
        handle.write(
            f"Fits use deepest depth: {DEPTHS[-1]}\n\n"
        )

        handle.write(
            "Mean H2 fit: H2(n) = n - c + a 2^{-n}\n"
        )
        handle.write(
            f"  c = {h2_params[0]:.8g} ± {h2_errors[0]:.3g}\n"
        )
        handle.write(
            f"  a = {h2_params[1]:.8g} ± {h2_errors[1]:.3g}\n"
        )
        handle.write("  Haar expectation: c -> 1\n\n")

        handle.write(
            "Mean R2 fit: R2(n) = R_inf + a 2^{-n} + b 2^{-2n}\n"
        )
        handle.write(
            f"  R_inf = {r2_params[0]:.8g} ± {r2_errors[0]:.3g}\n"
        )
        handle.write(
            f"  a     = {r2_params[1]:.8g} ± {r2_errors[1]:.3g}\n"
        )
        handle.write(
            f"  b     = {r2_params[2]:.8g} ± {r2_errors[2]:.3g}\n"
        )
        handle.write("  Haar expectation: R_inf -> 1/2\n\n")

        handle.write(
            "Var(H2) fit: a 2^{-n} + b 2^{-2n}\n"
        )
        handle.write(
            f"  a = {h2_var_params[0]:.8g} ± {h2_var_errors[0]:.3g}\n"
        )
        handle.write(
            f"  b = {h2_var_params[1]:.8g} ± {h2_var_errors[1]:.3g}\n\n"
        )

        handle.write(
            "Var(R2) fit: a 2^{-n} + b 2^{-2n}\n"
        )
        handle.write(
            f"  a = {r2_var_params[0]:.8g} ± {r2_var_errors[0]:.3g}\n"
        )
        handle.write(
            f"  b = {r2_var_params[1]:.8g} ± {r2_var_errors[1]:.3g}\n"
        )

    print(f"Wrote {SUMMARY_PATH}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("Starting scrambling H2 study")
    print(f"Qubits: {N_QUBITS}")
    print(f"Depths: {DEPTHS}")
    print(f"Circuits per (n, depth): {N_CIRCUITS}")
    print()

    records = simulate_ensemble()
    save_data(records)

    arrays = records_to_arrays(records)

    plot_sample_bitstring_distributions()
    h2_fit = plot_mean_h2_and_fit(arrays)
    r2_fit = plot_mean_r2_and_fit(arrays)
    variance_fit = plot_variance_scaling(arrays)
    plot_depth_convergence(arrays)
    plot_r2_histograms(arrays)
    plot_tvd_vs_r2(arrays)

    write_fit_summary(
        h2_fit,
        r2_fit,
        variance_fit,
    )

    print()
    print(f"Wrote raw CSV: {CSV_PATH}")
    print(f"Wrote compressed data: {NPZ_PATH}")
    print(f"Wrote figures to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()

