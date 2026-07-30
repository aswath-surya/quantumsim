"""Follow-up analyses for the global-depolarizing diagnostic study.

This script consumes ``depolarizing_diagnostics.csv`` produced by
``analyze_depolarizing_model.py`` and separates two questions:

1. Does the global-depolarizing output model work when its strength alpha is
   fitted directly from the noisy distribution?

2. Does CB/QCAP estimate that effective alpha accurately enough for the
   entropy-corrected prediction to remain an upper prediction?

For every circuit instance, the input data already contain

    actual_tvd
    qcap_error
    alpha_fit
    tvd_q_uniform
    renyi_factor
    fit_residual_tvd

This script constructs and compares:

    QCAP baseline:
        B_QCAP = epsilon_QCAP

    QCAP exact depolarizing prediction:
        P_exact,QCAP = epsilon_QCAP * d_TV(q,u)

    QCAP Renyi-2 prediction:
        P_R2,QCAP = epsilon_QCAP * R_2(q)

    Fitted-alpha exact depolarizing prediction:
        P_exact,fit = alpha_fit * d_TV(q,u)

    Fitted-alpha Renyi-2 prediction:
        P_R2,fit = alpha_fit * R_2(q)

where

    R_2(q) = min{1, 1/2 sqrt(D sum_x q_x^2 - 1)}.

The fitted-alpha predictions are diagnostics, not deployable bounds: alpha_fit
is inferred using the noisy output itself. They isolate whether failures come
from the white-noise model, the Renyi relaxation, or the CB/QCAP estimate of the
effective noise strength.

Outputs
-------
    prediction_scatter.png
    prediction_error_by_depth.png
    prediction_coverage.png
    alpha_calibration_by_depth.png
    alpha_ratio_distributions.png
    residual_decomposition.png
    structured_failure_modes.png
    followup_analysis.csv
    followup_summary.txt

Run
---
    python analyze_alpha_entropy_predictions.py

or

    python analyze_alpha_entropy_predictions.py \
            --input results/depolarizing_diagnostics/depolarizing_diagnostics.csv \
            --output-dir results/depolarizing_followup
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Data handling
# =============================================================================

NUMERIC_FIELDS = {
    "n_qubits",
    "depth",
    "instance",
    "circuit_seed",
    "noise_seed",
    "collision",
    "h2",
    "tvd_q_uniform",
    "renyi_factor",
    "actual_tvd",
    "qcap_error",
    "qcap_std",
    "exact_prediction",
    "renyi_prediction",
    "alpha_fit",
    "fit_residual_tvd",
    "fitted_model_tvd",
    "residual_fraction",
    "alpha_over_qcap",
    "mean_individual_rc_tvd",
    "std_individual_rc_tvd",
    "max_ideal_rc_tvd",
}

BOOLEAN_FIELDS = {
    "qcap_covers",
    "exact_prediction_covers",
    "renyi_prediction_covers",
}


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def read_csv(path: Path) -> List[dict]:
    rows: List[dict] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        for source in reader:
            row: dict = {}

            for key, value in source.items():
                if key in BOOLEAN_FIELDS:
                    row[key] = parse_bool(value)
                elif key in NUMERIC_FIELDS:
                    row[key] = float(value)
                else:
                    row[key] = value

            rows.append(row)

    if not rows:
        raise ValueError(f"No records found in {path}.")

    return rows


def enriched_rows(rows: Sequence[dict]) -> List[dict]:
    output: List[dict] = []

    for source in rows:
        row = dict(source)

        actual = float(row["actual_tvd"])
        qcap = float(row["qcap_error"])
        alpha = float(row["alpha_fit"])
        dqu = float(row["tvd_q_uniform"])
        r2 = float(row["renyi_factor"])
        residual = float(row["fit_residual_tvd"])

        row["pred_qcap"] = qcap
        row["pred_exact_qcap"] = qcap * dqu
        row["pred_r2_qcap"] = qcap * r2
        row["pred_exact_fit"] = alpha * dqu
        row["pred_r2_fit"] = alpha * r2

        row["error_qcap"] = row["pred_qcap"] - actual
        row["error_exact_qcap"] = row["pred_exact_qcap"] - actual
        row["error_r2_qcap"] = row["pred_r2_qcap"] - actual
        row["error_exact_fit"] = row["pred_exact_fit"] - actual
        row["error_r2_fit"] = row["pred_r2_fit"] - actual

        row["abs_error_qcap"] = abs(row["error_qcap"])
        row["abs_error_exact_qcap"] = abs(row["error_exact_qcap"])
        row["abs_error_r2_qcap"] = abs(row["error_r2_qcap"])
        row["abs_error_exact_fit"] = abs(row["error_exact_fit"])
        row["abs_error_r2_fit"] = abs(row["error_r2_fit"])

        tolerance = 1e-10
        row["covers_qcap"] = actual <= row["pred_qcap"] + tolerance
        row["covers_exact_qcap"] = actual <= row["pred_exact_qcap"] + tolerance
        row["covers_r2_qcap"] = actual <= row["pred_r2_qcap"] + tolerance
        row["covers_exact_fit"] = actual <= row["pred_exact_fit"] + tolerance
        row["covers_r2_fit"] = actual <= row["pred_r2_fit"] + tolerance

        row["alpha_minus_qcap"] = alpha - qcap
        row["model_fraction_unexplained"] = (
            residual / actual if actual > 1e-15 else 0.0
        )

        # If the best-fit model is accurate, actual TVD should be close to
        # alpha*d_TV(q,u). This difference is bounded in magnitude by the
        # distribution-fit residual via the reverse triangle inequality.
        row["exact_fit_gap"] = actual - row["pred_exact_fit"]
        row["gap_over_residual"] = (
            abs(row["exact_fit_gap"]) / residual
            if residual > 1e-15
            else 0.0
        )

        output.append(row)

    return output


def subset(
    rows: Sequence[dict],
    *,
    mode: str | None = None,
    depth: int | None = None,
) -> List[dict]:
    selected = list(rows)

    if mode is not None:
        selected = [row for row in selected if row["mode"] == mode]

    if depth is not None:
        selected = [
            row for row in selected
            if int(row["depth"]) == int(depth)
        ]

    return selected


def unique_depths(rows: Sequence[dict]) -> List[int]:
    return sorted({int(row["depth"]) for row in rows})


# =============================================================================
# Statistics
# =============================================================================

PREDICTIONS = {
    "QCAP": "pred_qcap",
    "exact(QCAP)": "pred_exact_qcap",
    "R2(QCAP)": "pred_r2_qcap",
    "exact(alpha_fit)": "pred_exact_fit",
    "R2(alpha_fit)": "pred_r2_fit",
}

ERROR_FIELDS = {
    "QCAP": "error_qcap",
    "exact(QCAP)": "error_exact_qcap",
    "R2(QCAP)": "error_r2_qcap",
    "exact(alpha_fit)": "error_exact_fit",
    "R2(alpha_fit)": "error_r2_fit",
}

ABS_ERROR_FIELDS = {
    "QCAP": "abs_error_qcap",
    "exact(QCAP)": "abs_error_exact_qcap",
    "R2(QCAP)": "abs_error_r2_qcap",
    "exact(alpha_fit)": "abs_error_exact_fit",
    "R2(alpha_fit)": "abs_error_r2_fit",
}

COVERAGE_FIELDS = {
    "QCAP": "covers_qcap",
    "exact(QCAP)": "covers_exact_qcap",
    "R2(QCAP)": "covers_r2_qcap",
    "exact(alpha_fit)": "covers_exact_fit",
    "R2(alpha_fit)": "covers_r2_fit",
}


def values(rows: Sequence[dict], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=float)


def mean_quantiles(
    rows: Sequence[dict],
    field: str,
) -> Tuple[float, float, float]:
    array = values(rows, field)

    return (
        float(np.mean(array)),
        float(np.quantile(array, 0.16)),
        float(np.quantile(array, 0.84)),
    )


def linear_calibration(
    rows: Sequence[dict],
) -> Dict[str, float]:
    """Fit alpha_fit = intercept + slope*qcap_error."""
    x = values(rows, "qcap_error")
    y = values(rows, "alpha_fit")

    if len(x) < 2 or np.std(x) <= 1e-15:
        return {
            "intercept": float("nan"),
            "slope": float("nan"),
            "r2": float("nan"),
            "rmse": float("nan"),
        }

    slope, intercept = np.polyfit(x, y, 1)
    prediction = intercept + slope * x

    residual = y - prediction
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    r_squared = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0.0
        else float("nan")
    )

    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "r2": float(r_squared),
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }


# =============================================================================
# Plotting
# =============================================================================

def save_figure(
    figure,
    path: Path,
) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    print(f"Wrote {path}")


def plot_prediction_scatter(
    rows: Sequence[dict],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15.0, 9.5),
        sharex=True,
        sharey=True,
    )

    panels = [
        ("QCAP", "pred_qcap"),
        ("exact(QCAP)", "pred_exact_qcap"),
        ("R2(QCAP)", "pred_r2_qcap"),
        ("exact(alpha_fit)", "pred_exact_fit"),
        ("R2(alpha_fit)", "pred_r2_fit"),
    ]

    for axis, (label, field) in zip(axes.flat, panels):
        for mode in ("random", "structured"):
            group = subset(rows, mode=mode)

            axis.scatter(
                values(group, field),
                values(group, "actual_tvd"),
                s=18,
                alpha=0.5,
                label=mode,
            )

        limit = max(
            float(np.max(values(rows, field))),
            float(np.max(values(rows, "actual_tvd"))),
        )

        axis.plot(
            [0.0, limit],
            [0.0, limit],
            "--",
            linewidth=1.2,
            label="equality",
        )

        axis.set_title(label)
        axis.set_xlabel("prediction")
        axis.set_ylabel("actual TVD")
        axis.grid(True, alpha=0.2)

    axes.flat[-1].axis("off")
    axes.flat[0].legend(frameon=False, fontsize=8)

    figure.suptitle(
        "Actual TVD versus QCAP- and fitted-alpha-based predictions",
        fontsize=13,
    )

    save_figure(
        figure,
        output_dir / "prediction_scatter.png",
    )


def plot_prediction_error_by_depth(
    rows: Sequence[dict],
    output_dir: Path,
) -> None:
    depths = unique_depths(rows)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.2),
        sharey=True,
    )

    for axis, mode in zip(axes, ("random", "structured")):
        for label, field in ERROR_FIELDS.items():
            means = []
            low = []
            high = []

            for depth in depths:
                group = subset(
                    rows,
                    mode=mode,
                    depth=depth,
                )
                mean, q16, q84 = mean_quantiles(group, field)
                means.append(mean)
                low.append(q16)
                high.append(q84)

            means = np.asarray(means)
            low = np.asarray(low)
            high = np.asarray(high)

            axis.plot(
                depths,
                means,
                "o-",
                label=label,
            )

            if label in {"exact(QCAP)", "exact(alpha_fit)"}:
                axis.fill_between(
                    depths,
                    low,
                    high,
                    alpha=0.08,
                )

        axis.axhline(
            0.0,
            linestyle="--",
            linewidth=1.2,
        )
        axis.set_xlabel("circuit depth")
        axis.set_ylabel("prediction - actual TVD")
        axis.set_title(mode)
        axis.set_xticks(depths)
        axis.grid(True, alpha=0.2)
        axis.legend(frameon=False, fontsize=8)

    figure.suptitle(
        "Signed prediction error: positive values are conservative",
        fontsize=13,
    )

    save_figure(
        figure,
        output_dir / "prediction_error_by_depth.png",
    )


def plot_coverage(
    rows: Sequence[dict],
    output_dir: Path,
) -> None:
    depths = unique_depths(rows)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.0),
        sharey=True,
    )

    for axis, mode in zip(axes, ("random", "structured")):
        for label, field in COVERAGE_FIELDS.items():
            coverage = []

            for depth in depths:
                group = subset(
                    rows,
                    mode=mode,
                    depth=depth,
                )

                coverage.append(
                    float(np.mean([bool(row[field]) for row in group]))
                )

            axis.plot(
                depths,
                coverage,
                "o-",
                label=label,
            )

        axis.set_xlabel("circuit depth")
        axis.set_title(mode)
        axis.set_xticks(depths)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(True, alpha=0.2)
        axis.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel("fraction of instances with prediction >= actual TVD")

    figure.suptitle(
        "Empirical coverage: isolating alpha estimation from entropy relaxation",
        fontsize=13,
    )

    save_figure(
        figure,
        output_dir / "prediction_coverage.png",
    )


def plot_alpha_calibration_by_depth(
    rows: Sequence[dict],
    output_dir: Path,
) -> None:
    depths = unique_depths(rows)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.2),
        sharey=True,
    )

    for axis, mode in zip(axes, ("random", "structured")):
        alpha_means = []
        alpha_low = []
        alpha_high = []
        qcap_values = []

        for depth in depths:
            group = subset(rows, mode=mode, depth=depth)

            mean, q16, q84 = mean_quantiles(
                group,
                "alpha_fit",
            )
            alpha_means.append(mean)
            alpha_low.append(q16)
            alpha_high.append(q84)
            qcap_values.append(
                float(np.mean(values(group, "qcap_error")))
            )

        alpha_means = np.asarray(alpha_means)
        alpha_low = np.asarray(alpha_low)
        alpha_high = np.asarray(alpha_high)

        axis.plot(
            depths,
            alpha_means,
            "o-",
            label="mean alpha_fit",
        )
        axis.fill_between(
            depths,
            alpha_low,
            alpha_high,
            alpha=0.15,
            label="16-84% alpha range",
        )
        axis.plot(
            depths,
            qcap_values,
            "s--",
            label="QCAP error",
        )

        axis.set_xlabel("circuit depth")
        axis.set_title(mode)
        axis.set_xticks(depths)
        axis.grid(True, alpha=0.2)
        axis.legend(frameon=False)

    axes[0].set_ylabel("effective noise strength")

    figure.suptitle(
        "Depth-dependent calibration of QCAP to fitted white-noise strength",
        fontsize=13,
    )

    save_figure(
        figure,
        output_dir / "alpha_calibration_by_depth.png",
    )


def plot_alpha_ratio_distributions(
    rows: Sequence[dict],
    output_dir: Path,
) -> None:
    depths = unique_depths(rows)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.0),
        sharey=True,
    )

    positions = np.arange(len(depths))

    for axis, mode in zip(axes, ("random", "structured")):
        datasets = []

        for depth in depths:
            group = subset(rows, mode=mode, depth=depth)

            ratio = np.asarray(
                [
                    row["alpha_fit"] / row["qcap_error"]
                    for row in group
                    if row["qcap_error"] > 1e-15
                ],
                dtype=float,
            )
            datasets.append(ratio)

        axis.boxplot(
            datasets,
            positions=positions,
            widths=0.6,
            showfliers=True,
        )
        axis.axhline(
            1.0,
            linestyle="--",
            linewidth=1.2,
            label="alpha_fit = QCAP",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(depths)
        axis.set_xlabel("circuit depth")
        axis.set_title(mode)
        axis.grid(True, alpha=0.2)
        axis.legend(frameon=False)

    axes[0].set_ylabel(r"$\alpha_{\rm fit}/\epsilon_{\rm QCAP}$")

    figure.suptitle(
        "How strongly does QCAP under- or overestimate fitted alpha?",
        fontsize=13,
    )

    save_figure(
        figure,
        output_dir / "alpha_ratio_distributions.png",
    )


def plot_residual_decomposition(
    rows: Sequence[dict],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.2),
    )

    for mode in ("random", "structured"):
        group = subset(rows, mode=mode)

        axes[0].scatter(
            values(group, "fit_residual_tvd"),
            np.abs(values(group, "exact_fit_gap")),
            s=18,
            alpha=0.5,
            label=mode,
        )

        axes[1].scatter(
            values(group, "model_fraction_unexplained"),
            values(group, "abs_error_r2_fit"),
            s=18,
            alpha=0.5,
            label=mode,
        )

    max_value = max(
        float(np.max(values(rows, "fit_residual_tvd"))),
        float(np.max(np.abs(values(rows, "exact_fit_gap")))),
    )

    axes[0].plot(
        [0.0, max_value],
        [0.0, max_value],
        "--",
        linewidth=1.2,
        label="equality",
    )

    axes[0].set_xlabel("distribution-fit residual TVD")
    axes[0].set_ylabel(
        r"$|d_{\rm TV}(p,q)-\alpha_{\rm fit}d_{\rm TV}(q,u)|$"
    )
    axes[0].set_title("Does the fit residual explain exact-model mismatch?")

    axes[1].set_xlabel("fraction of actual TVD unexplained by fitted model")
    axes[1].set_ylabel(
        r"$|R_2(\alpha_{\rm fit})-\mathrm{actual\ TVD}|$"
    )
    axes[1].set_title("Remaining error after fitting alpha")

    for axis in axes:
        axis.grid(True, alpha=0.2)
        axis.legend(frameon=False, fontsize=8)

    save_figure(
        figure,
        output_dir / "residual_decomposition.png",
    )


def plot_structured_failure_modes(
    rows: Sequence[dict],
    output_dir: Path,
) -> None:
    structured = subset(rows, mode="structured")

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16.0, 4.8),
    )

    scatter_specs = [
        (
            "h2",
            "abs_error_r2_qcap",
            r"$H_2(q)$",
            r"$|R_2(\mathrm{QCAP})-\mathrm{actual}|$",
        ),
        (
            "alpha_over_qcap",
            "abs_error_r2_qcap",
            r"$\alpha_{\rm fit}/\epsilon_{\rm QCAP}$",
            r"$|R_2(\mathrm{QCAP})-\mathrm{actual}|$",
        ),
        (
            "fit_residual_tvd",
            "abs_error_r2_qcap",
            "depolarizing-fit residual",
            r"$|R_2(\mathrm{QCAP})-\mathrm{actual}|$",
        ),
    ]

    depths = unique_depths(structured)

    for axis, (x_field, y_field, xlabel, ylabel) in zip(
        axes,
        scatter_specs,
    ):
        for depth in depths:
            group = subset(
                structured,
                depth=depth,
            )

            axis.scatter(
                values(group, x_field),
                values(group, y_field),
                s=18,
                alpha=0.5,
                label=f"d={depth}",
            )

        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.2)

    axes[0].set_title("Dependence on ideal entropy")
    axes[1].set_title("Dependence on alpha miscalibration")
    axes[2].set_title("Dependence on model failure")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)

    figure.suptitle(
        "What drives failures of the structured-circuit R2(QCAP) prediction?",
        fontsize=13,
    )

    save_figure(
        figure,
        output_dir / "structured_failure_modes.png",
    )


# =============================================================================
# Export and summary
# =============================================================================

EXPORT_FIELDS = [
    "mode",
    "n_qubits",
    "depth",
    "instance",
    "h2",
    "tvd_q_uniform",
    "renyi_factor",
    "actual_tvd",
    "qcap_error",
    "alpha_fit",
    "fit_residual_tvd",
    "pred_qcap",
    "pred_exact_qcap",
    "pred_r2_qcap",
    "pred_exact_fit",
    "pred_r2_fit",
    "error_qcap",
    "error_exact_qcap",
    "error_r2_qcap",
    "error_exact_fit",
    "error_r2_fit",
    "covers_qcap",
    "covers_exact_qcap",
    "covers_r2_qcap",
    "covers_exact_fit",
    "covers_r2_fit",
    "alpha_minus_qcap",
    "alpha_over_qcap",
    "model_fraction_unexplained",
    "exact_fit_gap",
    "gap_over_residual",
]


def write_enriched_csv(
    rows: Sequence[dict],
    path: Path,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=EXPORT_FIELDS,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row[field]
                    for field in EXPORT_FIELDS
                }
            )

    print(f"Wrote {path}")


def write_summary(
    rows: Sequence[dict],
    path: Path,
) -> None:
    depths = unique_depths(rows)

    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "Follow-up alpha/entropy prediction analysis\n"
        )
        handle.write("=" * 44 + "\n\n")

        for mode in ("random", "structured"):
            group_mode = subset(rows, mode=mode)
            calibration = linear_calibration(group_mode)

            handle.write(f"{mode.upper()}\n")
            handle.write("-" * len(mode) + "\n")
            handle.write(
                "Global calibration alpha_fit = intercept + slope*QCAP:\n"
            )
            handle.write(
                f"  intercept = {calibration['intercept']:.6g}\n"
            )
            handle.write(
                f"  slope     = {calibration['slope']:.6g}\n"
            )
            handle.write(
                f"  R^2       = {calibration['r2']:.6g}\n"
            )
            handle.write(
                f"  RMSE      = {calibration['rmse']:.6g}\n\n"
            )

            for depth in depths:
                group = subset(
                    rows,
                    mode=mode,
                    depth=depth,
                )

                handle.write(f"depth={depth}\n")

                for label in PREDICTIONS:
                    mae = float(
                        np.mean(
                            values(
                                group,
                                ABS_ERROR_FIELDS[label],
                            )
                        )
                    )
                    bias = float(
                        np.mean(
                            values(
                                group,
                                ERROR_FIELDS[label],
                            )
                        )
                    )
                    coverage = float(
                        np.mean(
                            [
                                bool(row[COVERAGE_FIELDS[label]])
                                for row in group
                            ]
                        )
                    )

                    handle.write(
                        f"  {label:17s} "
                        f"MAE={mae:.6f}, "
                        f"bias={bias:+.6f}, "
                        f"coverage={coverage:.3f}\n"
                    )

                handle.write(
                    f"  mean alpha/QCAP = "
                    f"{np.mean(values(group, 'alpha_over_qcap')):.4f}\n"
                )
                handle.write(
                    f"  mean fit residual = "
                    f"{np.mean(values(group, 'fit_residual_tvd')):.6f}\n"
                )
                handle.write(
                    f"  median unexplained fraction = "
                    f"{np.median(values(group, 'model_fraction_unexplained')):.4f}\n\n"
                )

            handle.write("\n")

        handle.write("Interpretation\n")
        handle.write("--------------\n")
        handle.write(
            "1. If exact(alpha_fit) has low MAE, the global-depolarizing "
            "output model is accurate after the effective strength is known.\n"
        )
        handle.write(
            "2. If R2(alpha_fit) covers exact(alpha_fit) but R2(QCAP) fails, "
            "the entropy relaxation is not the problem; CB/QCAP is "
            "underestimating alpha.\n"
        )
        handle.write(
            "3. If exact(alpha_fit) remains inaccurate beyond the measured fit "
            "residual, the one-parameter global-depolarizing model is inadequate.\n"
        )
        handle.write(
            "4. The structured_failure_modes plot separates entropy effects, "
            "alpha miscalibration, and genuine white-noise-model failure.\n"
        )

    print(f"Wrote {path}")


# =============================================================================
# CLI and main
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare QCAP- and fitted-alpha-based entropy predictions using "
            "depolarizing_diagnostics.csv."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/depolarizing_diagnostics/"
            "depolarizing_diagnostics.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/depolarizing_followup"
        ),
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = enriched_rows(
        read_csv(args.input)
    )

    write_enriched_csv(
        rows,
        args.output_dir / "followup_analysis.csv",
    )

    plot_prediction_scatter(
        rows,
        args.output_dir,
    )
    plot_prediction_error_by_depth(
        rows,
        args.output_dir,
    )
    plot_coverage(
        rows,
        args.output_dir,
    )
    plot_alpha_calibration_by_depth(
        rows,
        args.output_dir,
    )
    plot_alpha_ratio_distributions(
        rows,
        args.output_dir,
    )
    plot_residual_decomposition(
        rows,
        args.output_dir,
    )
    plot_structured_failure_modes(
        rows,
        args.output_dir,
    )

    write_summary(
        rows,
        args.output_dir / "followup_summary.txt",
    )

    print()
    print(f"Read {len(rows)} circuit records from {args.input}")
    print(f"Wrote follow-up analyses to {args.output_dir}")


if __name__ == "__main__":
    main()
