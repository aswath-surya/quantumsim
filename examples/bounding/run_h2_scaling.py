"""Study H2(q) scaling and noisy-data reconstruction in the RC/CB pipeline.

This script implements two complementary estimators:

1. Ensemble extrapolation:
       compute exact H2(q) and
       R2(q) = 0.5 * sqrt(2**n * 2**(-H2(q)) - 1)
   in the classically simulable regime, fit their dependence on qubit number,
   and use the fitted curve as an extrapolatable ensemble model.

2. Circuit-specific noisy-data inversion:
       p = (1 - alpha) q + alpha u
   with alpha estimated from calibrated QCAP. Reconstruct q directly and also
   infer its collision probability without reconstructing every component.

The exact statevector values are retained only as validation ground truth.
"""

from __future__ import annotations

import csv
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import _bootstrap  # noqa: F401

from proxysim import NoiseModel, even_pairs, odd_pairs, simulate
from proxysim.backends import StatevectorBackend
from proxysim.benchmarking import cycle_benchmark, qcap_bound, readout_fidelity
from proxysim.circuit import Circuit


QUBITS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
DEPTHS = [2, 4, 8, 16]
MODES = ("random", "structured")

N_INSTANCES = 30
TVD_TRAJECTORIES = 1000
CB_TRAJECTORIES = 50
CB_DEPTHS = [1, 2, 4, 8, 16, 24]

ALPHA_INTERCEPT = 0.0
ALPHA_SLOPE = 1.2
SEED = 7

NOISE = NoiseModel(
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
)

OUT_DIR = Path(_bootstrap.RESULTS_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "h2_scaling_records.csv"
OUT_NPZ = OUT_DIR / "h2_scaling_data.npz"
_SV = StatevectorBackend()


def add_random_su2_layer(circuit: Circuit, n: int, rng: random.Random) -> None:
    for q in range(n):
        z = rng.uniform(-1.0, 1.0)
        theta = math.acos(z)
        phi = rng.uniform(0.0, 2.0 * math.pi)
        lam = rng.uniform(0.0, 2.0 * math.pi)
        circuit.add("u", q, params=(theta, phi, lam))


def add_structured_layer(circuit: Circuit, n: int, rng: random.Random) -> None:
    for q in range(n):
        gate = rng.choice(("i", "h", "x"))
        if gate != "i":
            circuit.add(gate, q)


def add_oneq_layer(circuit: Circuit, n: int, mode: str, rng: random.Random) -> None:
    if mode == "random":
        add_random_su2_layer(circuit, n, rng)
    elif mode == "structured":
        add_structured_layer(circuit, n, rng)
    else:
        raise ValueError(f"Unknown mode {mode!r}")


def add_entangling_layer(circuit: Circuit, pairs) -> None:
    for a, b in pairs:
        circuit.add("cz", a, b)


def test_circuit(n: int, depth: int, mode: str, seed: int) -> Circuit:
    rng = random.Random(seed)
    cycle_a = even_pairs(n)
    cycle_b = odd_pairs(n)
    circuit = Circuit(n, name=f"{mode}_n{n}_d{depth}_s{seed}")
    for _ in range(depth):
        add_oneq_layer(circuit, n, mode, rng)
        add_entangling_layer(circuit, cycle_a)
        add_oneq_layer(circuit, n, mode, rng)
        add_entangling_layer(circuit, cycle_b)
    add_oneq_layer(circuit, n, mode, rng)
    return circuit


def normalize_distribution(values: np.ndarray, dimension: int) -> np.ndarray:
    q = np.asarray(values, dtype=float).reshape(-1)
    if q.size != dimension:
        raise ValueError(f"Distribution length {q.size}; expected {dimension}.")
    q = np.clip(q, 0.0, None)
    total = float(q.sum())
    if total <= 0.0:
        raise ValueError("Distribution has zero total probability.")
    return q / total


def dense_distribution(circuit: Circuit) -> np.ndarray:
    distribution = _SV.exact_distribution(circuit)
    dimension = 2**circuit.n_qubits
    if isinstance(distribution, dict):
        dense = np.zeros(dimension, dtype=float)
        for outcome, probability in distribution.items():
            index = int(outcome.replace(" ", ""), 2) if isinstance(outcome, str) else int(outcome)
            dense[index] += float(probability)
    else:
        dense = np.asarray(distribution, dtype=float).reshape(-1)
    return normalize_distribution(dense, dimension)


def noisy_distribution(circuit: Circuit, seed: int) -> np.ndarray:
    distribution = simulate(
        circuit,
        simulator="statevector",
        output="distribution",
        noise=NOISE,
        shots=0,
        n_traj=TVD_TRAJECTORIES,
        seed=seed,
    )
    return normalize_distribution(np.asarray(distribution, dtype=float), 2**circuit.n_qubits)


def collision_probability(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=float)
    return float(np.dot(q, q))


def h2_entropy(q: np.ndarray) -> float:
    return float(-np.log2(max(collision_probability(q), np.finfo(float).tiny)))


def renyi_factor_from_collision(c2: float, dimension: int) -> float:
    value = 0.5 * math.sqrt(max(dimension * c2 - 1.0, 0.0))
    return float(min(1.0, value))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    indices = np.arange(1, v.size + 1)
    positive = u - cssv / indices > 0
    if not np.any(positive):
        return np.full(v.size, 1.0 / v.size)
    rho = int(np.nonzero(positive)[0][-1])
    theta = cssv[rho] / float(rho + 1)
    return np.maximum(v - theta, 0.0)


def calibrated_alpha(qcap_error: float) -> float:
    return float(np.clip(ALPHA_INTERCEPT + ALPHA_SLOPE * qcap_error, 0.0, 1.0 - 1e-9))


def reconstruct_q_from_noisy(noisy: np.ndarray, alpha_hat: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(noisy, dtype=float)
    dimension = p.size
    uniform = np.full(dimension, 1.0 / dimension)
    raw = (p - alpha_hat * uniform) / (1.0 - alpha_hat)
    return raw, project_to_simplex(raw)


def infer_collision_from_noisy(noisy: np.ndarray, alpha_hat: float) -> float:
    p = np.asarray(noisy, dtype=float)
    dimension = p.size
    c2_p = collision_probability(p)
    c2_q = 1.0 / dimension + (c2_p - 1.0 / dimension) / (1.0 - alpha_hat) ** 2
    return float(np.clip(c2_q, 1.0 / dimension, 1.0))


@dataclass
class Characterization:
    cycle_efs: dict
    ro_fid: float
    ro_std: float


def characterize_n(n: int, mode: str) -> Characterization:
    cycle_a = even_pairs(n)
    cycle_b = odd_pairs(n)
    cycle_results = {
        "A": cycle_benchmark(
            cycle_a, n, CB_DEPTHS, NOISE, mode=mode,
            n_trajectories=CB_TRAJECTORIES,
            seed=SEED + 10_000 * n + 1,
        )
    }
    if cycle_b:
        cycle_results["B"] = cycle_benchmark(
            cycle_b, n, CB_DEPTHS, NOISE, mode=mode,
            n_trajectories=CB_TRAJECTORIES,
            seed=SEED + 10_000 * n + 2,
        )
    ro_fid, ro_std = readout_fidelity(n, NOISE, seed=SEED + 10_000 * n + 3)
    cycle_efs = {
        name: (float(result["e_F"]), float(result["e_F_std"]))
        for name, result in cycle_results.items()
    }
    return Characterization(cycle_efs, float(ro_fid), float(ro_std))


def qcap_for_depth(characterization: Characterization, depth: int) -> tuple[float, float]:
    counts = {name: depth for name in characterization.cycle_efs}
    result = qcap_bound(counts, characterization.cycle_efs, characterization.ro_fid, characterization.ro_std)
    return float(result["error"]), float(result["std"])


def h2_model(n, c, a):
    n = np.asarray(n, dtype=float)
    return n - c + a * np.power(2.0, -n)


def r2_model(n, r_inf, a, b):
    n = np.asarray(n, dtype=float)
    x = np.power(2.0, -n)
    return r_inf + a * x + b * x**2


def fit_h2_scaling(n_values: np.ndarray, h_values: np.ndarray):
    return curve_fit(h2_model, n_values, h_values, p0=(1.0, 0.0), maxfev=10000)


def fit_r2_scaling(n_values: np.ndarray, r_values: np.ndarray):
    return curve_fit(r2_model, n_values, r_values, p0=(0.5, 0.0, 0.0), maxfev=10000)


def run_study() -> list[dict]:
    records: list[dict] = []
    for mode in MODES:
        print(f"\n=== {mode} ensemble ===")
        for n in QUBITS:
            print(f"  characterizing n={n}")
            char = characterize_n(n, mode)
            for depth in DEPTHS:
                qcap_error, qcap_std = qcap_for_depth(char, depth)
                alpha_hat = calibrated_alpha(qcap_error)
                for instance in range(N_INSTANCES):
                    circuit_seed = (
                        SEED + 100_000_000 * (0 if mode == "random" else 1)
                        + 1_000_000 * n + 10_000 * depth + instance
                    )
                    noise_seed = circuit_seed + 500_000_000
                    circuit = test_circuit(n, depth, mode, circuit_seed)
                    q_true = dense_distribution(circuit)
                    p_noisy = noisy_distribution(circuit, noise_seed)

                    c2_true = collision_probability(q_true)
                    h2_true = h2_entropy(q_true)
                    r2_true = renyi_factor_from_collision(c2_true, 2**n)

                    raw_q, projected_q = reconstruct_q_from_noisy(p_noisy, alpha_hat)
                    c2_proj = collision_probability(projected_q)
                    h2_proj = h2_entropy(projected_q)
                    r2_proj = renyi_factor_from_collision(c2_proj, 2**n)

                    c2_collision = infer_collision_from_noisy(p_noisy, alpha_hat)
                    h2_collision = float(-np.log2(c2_collision))
                    r2_collision = renyi_factor_from_collision(c2_collision, 2**n)

                    records.append({
                        "mode": mode,
                        "n": n,
                        "depth": depth,
                        "instance": instance,
                        "qcap": qcap_error,
                        "qcap_std": qcap_std,
                        "alpha_hat": alpha_hat,
                        "h2_true": h2_true,
                        "r2_true": r2_true,
                        "c2_true": c2_true,
                        "h2_collision": h2_collision,
                        "r2_collision": r2_collision,
                        "c2_collision": c2_collision,
                        "h2_qproj": h2_proj,
                        "r2_qproj": r2_proj,
                        "c2_qproj": c2_proj,
                        "q_reconstruction_tvd": total_variation(projected_q, q_true),
                        "raw_negative_mass": float(-np.minimum(raw_q, 0.0).sum()),
                        "actual_tvd": total_variation(p_noisy, q_true),
                        "pred_true_h2": alpha_hat * r2_true,
                        "pred_collision_h2": alpha_hat * r2_collision,
                        "pred_qproj_h2": alpha_hat * r2_proj,
                    })
    return records


def grouped_mean(records, mode: str, depth: int, key: str):
    xs, means, stds = [], [], []
    for n in QUBITS:
        values = [float(r[key]) for r in records if r["mode"] == mode and r["depth"] == depth and r["n"] == n]
        xs.append(n)
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))
    return np.asarray(xs), np.asarray(means), np.asarray(stds)


def make_scaling_plots(records: list[dict]) -> list[Path]:
    outputs: list[Path] = []
    for mode in MODES:
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        for depth in DEPTHS:
            n, mean_h, std_h = grouped_mean(records, mode, depth, "h2_true")
            params, _ = fit_h2_scaling(n, mean_h)
            n_dense = np.linspace(min(QUBITS), max(QUBITS) + 6, 300)
            ax.errorbar(n, mean_h, yerr=std_h, marker="o", linestyle="none", capsize=3, label=f"depth {depth}: exact")
            ax.plot(n_dense, h2_model(n_dense, *params), linewidth=1.8, label=f"depth {depth}: c={params[0]:.3f}")
        n_dense = np.linspace(min(QUBITS), max(QUBITS) + 6, 300)
        ax.plot(n_dense, np.log2((2.0**n_dense + 1.0) / 2.0), "k--", linewidth=2.0, label="Porter-Thomas")
        ax.set_xlabel("number of qubits")
        ax.set_ylabel(r"$H_2(q)$")
        ax.set_title(f"{mode}: collision-entropy scaling")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
        path = OUT_DIR / f"h2_vs_qubits_{mode}.png"
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); outputs.append(path)

        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        for depth in DEPTHS:
            n, mean_r, std_r = grouped_mean(records, mode, depth, "r2_true")
            params, _ = fit_r2_scaling(n, mean_r)
            n_dense = np.linspace(min(QUBITS), max(QUBITS) + 12, 400)
            ax.errorbar(n, mean_r, yerr=std_r, marker="o", linestyle="none", capsize=3, label=f"depth {depth}: exact")
            ax.plot(n_dense, r2_model(n_dense, *params), linewidth=1.8, label=f"depth {depth}: R_inf={params[0]:.3f}")
        n_dense = np.linspace(min(QUBITS), max(QUBITS) + 12, 400)
        r_pt = 0.5 * np.sqrt((2.0**n_dense - 1.0) / (2.0**n_dense + 1.0))
        ax.plot(n_dense, r_pt, "k--", linewidth=2.0, label="Porter-Thomas")
        ax.set_xlabel("number of qubits")
        ax.set_ylabel(r"$R_2(q)=\frac{1}{2}\sqrt{2^n2^{-H_2(q)}-1}$")
        ax.set_title(f"{mode}: entropy-factor scaling and extrapolation")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
        path = OUT_DIR / f"r2_vs_qubits_{mode}.png"
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); outputs.append(path)

        fig, ax = plt.subplots(figsize=(6.4, 6.0))
        subset = [r for r in records if r["mode"] == mode]
        true = np.asarray([r["h2_true"] for r in subset])
        collision = np.asarray([r["h2_collision"] for r in subset])
        qproj = np.asarray([r["h2_qproj"] for r in subset])
        ax.scatter(true, collision, alpha=0.45, s=18, label="collision inversion")
        ax.scatter(true, qproj, alpha=0.45, s=18, label="full q inversion + simplex")
        lo = float(min(true.min(), collision.min(), qproj.min()))
        hi = float(max(true.max(), collision.max(), qproj.max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5)
        ax.set_xlabel(r"true $H_2(q)$")
        ax.set_ylabel(r"estimated $H_2(q)$")
        ax.set_title(f"{mode}: circuit-specific entropy estimation")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        path = OUT_DIR / f"h2_estimator_validation_{mode}.png"
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); outputs.append(path)
    return outputs


def save_records(records: list[dict]) -> None:
    fieldnames = list(records[0].keys())
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(records)
    arrays = {key: np.asarray([record[key] for record in records]) for key in fieldnames}
    np.savez_compressed(OUT_NPZ, **arrays)


def main() -> None:
    records = run_study()
    save_records(records)
    paths = make_scaling_plots(records)
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_NPZ}")
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
