"""Exact ideal-output Renyi-2 scaling for the MQT Bench QPE family.

For every requested circuit width N, this script loads an MQT Bench quantum phase
estimation circuit, computes its exact ideal output distribution, and evaluates

    S_2(N) = 1/2 * sqrt(2^N * sum_x p(x)^2 - 1).

Only the ideal S_2 scaling is studied.  There is no noisy simulation, randomized
compiling, cycle benchmarking, QCAP calculation, or TVD plot.

Two complementary fits are reported and plotted:

    S_2(N) = A * 2^(alpha N)

and the exact collision-entropy transformation

    Delta_2(N) = log2(1 + 4 S_2(N)^2) = a N + c.

The second fit reconstructs S_2 without fitting the square root directly and gives
an asymptotic collision-entropy density H_2/N -> 1-a.

By default the non-exact QPE benchmark is used because its output distribution is
normally nontrivial.  Set ALGORITHM = "qpeexact" to study the exactly representable
phase-estimation benchmark instead.

Run:
    python examples/run_qpe_s2_scaling.py
"""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector
from scipy.optimize import curve_fit

import _bootstrap  # noqa: F401
from proxysim import save_results
from proxysim.backends import StatevectorBackend
from proxysim.circuit import circuit_from_qasm
from proxysim.mqtbank import fetch


# =============================================================================
# Configuration
# =============================================================================

# MQT Bench commonly exposes "qpeexact" and "qpeinexact".
# The inexact family is the more informative default for distribution scaling.
ALGORITHM = "qpeinexact"

QUBIT_SIZES = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14]

# A local file such as qpeinexact8.qasm overrides the generated MQT Bench circuit.
QASM_TEMPLATE = "{algorithm}{n}.qasm"

OUT_FIGURE = _bootstrap.RESULTS_DIR + f"/{ALGORITHM}_s2_scaling_vs_N.png"
OUT_DATA = _bootstrap.RESULTS_DIR + f"/{ALGORITHM}_s2_scaling_data.npz"

_SV = StatevectorBackend()


# =============================================================================
# Circuit loading and exact ideal factors
# =============================================================================

def qasm_path(n: int) -> str:
    return os.path.join(
        os.path.dirname(__file__),
        QASM_TEMPLATE.format(algorithm=ALGORITHM, n=n),
    )


def load_algorithm_circuit(n: int):
    """Load a local QASM family member when present, otherwise use MQT Bench."""
    path = qasm_path(n)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            circ, _measured = circuit_from_qasm(handle.read())
        source = os.path.basename(path)
    else:
        circ, _measured = fetch(ALGORITHM, n)
        source = "mqt.bench"

    if circ.n_qubits != n:
        raise ValueError(
            f"{source} returned {circ.n_qubits} qubits for {ALGORITHM}{n}; expected {n}"
        )
    return circ, source


def ideal_probability(circ) -> np.ndarray:
    state = np.asarray(Statevector(_SV._build(circ)).data, dtype=complex)
    prob = np.abs(state) ** 2
    total = float(prob.sum())
    if not np.isclose(total, 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError(f"ideal probabilities sum to {total}, not one")
    return prob / total


def ideal_factors(prob: np.ndarray) -> dict:
    """Return S2 and exact collision-entropy quantities for one distribution."""
    prob = np.asarray(prob, dtype=float)
    dimension = prob.size
    n = int(round(np.log2(dimension)))
    if 2**n != dimension:
        raise ValueError(f"probability vector length {dimension} is not a power of two")

    exp_d2 = dimension * float(np.sum(prob**2))
    s2 = 0.5 * float(np.sqrt(max(exp_d2 - 1.0, 0.0)))
    delta2 = float(np.log2(exp_d2))
    h2 = n - delta2

    return {
        "s2": s2,
        "exp_d2": exp_d2,
        "delta2": delta2,
        "h2": h2,
        "h2_density": h2 / n,
        "support": int(np.count_nonzero(prob > 1e-12)),
        "max_probability": float(prob.max()),
    }


# =============================================================================
# Fits across N
# =============================================================================

def _r2(y, fitted) -> float:
    y = np.asarray(y, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((y - fitted) ** 2))
    scale = max(1.0, float(np.max(y**2))) * y.size
    return float("nan") if ss_tot <= 1e-20 * scale else 1.0 - ss_res / ss_tot


def fit_s2_power2(ns, s2):
    """Fit positive points to S2=A*2^(alpha*N) in log2 space."""
    ns = np.asarray(ns, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    good = np.isfinite(s2) & (s2 > 0.0)
    out = {
        "ok": False,
        "N_used": ns[good].astype(int),
        "dropped_N": ns[~good].astype(int).tolist(),
        "reason": "",
    }
    if np.count_nonzero(good) < 3:
        out["reason"] = "fewer than three strictly positive S2 points"
        return out

    x, y = ns[good], s2[good]
    log2y = np.log2(y)
    alpha0, log2a0 = np.polyfit(x, log2y, 1)
    par, cov = curve_fit(
        lambda z, log2a, alpha: log2a + alpha * z,
        x, log2y, p0=[log2a0, alpha0], maxfev=20000,
    )
    err = np.sqrt(np.diag(cov))
    pred = 2.0 ** (par[0] + par[1] * x)
    out.update({
        "ok": True,
        "A": float(2.0 ** par[0]),
        "A_err": float(np.log(2.0) * 2.0 ** par[0] * err[0]),
        "alpha": float(par[1]),
        "alpha_err": float(err[1]),
        "r2": _r2(y, pred),
    })
    return out


def fit_entropy_deficit(ns, delta2):
    """Fit the exact transformed quantity Delta2=a*N+c."""
    ns = np.asarray(ns, dtype=float)
    delta2 = np.asarray(delta2, dtype=float)
    good = np.isfinite(delta2)
    out = {
        "ok": False,
        "N_used": ns[good].astype(int),
        "dropped_N": ns[~good].astype(int).tolist(),
        "reason": "",
    }
    if np.count_nonzero(good) < 3:
        out["reason"] = "fewer than three finite Delta2 points"
        return out

    x, y = ns[good], delta2[good]
    par, cov = curve_fit(lambda z, a, c: a * z + c, x, y, maxfev=20000)
    err = np.sqrt(np.diag(cov))
    pred = par[0] * x + par[1]
    out.update({
        "ok": True,
        "a": float(par[0]),
        "a_err": float(err[0]),
        "c": float(par[1]),
        "c_err": float(err[1]),
        "h2_density": float(1.0 - par[0]),
        "h2_density_err": float(err[0]),
        "r2": _r2(y, pred),
    })
    return out


def reconstruct_s2_from_delta_fit(n, fit):
    n = np.asarray(n, dtype=float)
    delta = fit["a"] * n + fit["c"]
    return 0.5 * np.sqrt(np.maximum(np.exp2(delta) - 1.0, 0.0))


def deterministic_s2(n):
    n = np.asarray(n, dtype=float)
    return 0.5 * np.sqrt(np.maximum(np.exp2(n) - 1.0, 0.0))


def haar_s2(n):
    n = np.asarray(n, dtype=float)
    d = np.exp2(n)
    return 0.5 * np.sqrt(np.maximum((d - 1.0) / (d + 1.0), 0.0))


# =============================================================================
# Plotting
# =============================================================================

def plot_s2(ns, s2, power_fit, entropy_fit):
    grid = np.linspace(float(ns.min()), float(ns.max()), 500)
    fig, ax = plt.subplots(figsize=(9.4, 6.0))

    ax.plot(ns, s2, "o", ms=7, color="#333333", label=r"exact ideal $S_2$")

    if power_fit["ok"]:
        ax.plot(
            grid,
            power_fit["A"] * 2.0 ** (power_fit["alpha"] * grid),
            "--", lw=2.1, color="#CC79A7",
            label=(
                fr"$A2^{{\alpha N}}$: $\alpha={power_fit['alpha']:.4f}"
                fr"\pm{power_fit['alpha_err']:.4f}$, $R^2={power_fit['r2']:.4f}$"
            ),
        )
    else:
        ax.text(
            0.03, 0.96, f"direct fit skipped: {power_fit['reason']}",
            transform=ax.transAxes, va="top", color="#B00020",
        )

    if entropy_fit["ok"]:
        ax.plot(
            grid,
            reconstruct_s2_from_delta_fit(grid, entropy_fit),
            ":", lw=2.3, color="#0072B2",
            label=(
                fr"$\Delta_2=aN+c$: $a={entropy_fit['a']:.4f}"
                fr"\pm{entropy_fit['a_err']:.4f}$, $R^2={entropy_fit['r2']:.4f}$; "
                fr"$H_2/N\to{entropy_fit['h2_density']:.4f}$"
            ),
        )

    ax.plot(grid, deterministic_s2(grid), "-.", lw=1.4, color="#D55E00",
            label="deterministic-output reference")
    ax.plot(grid, haar_s2(grid), linestyle=(0, (5, 2, 1, 2)), lw=1.4,
            color="#009E73", label="finite-N Haar reference")

    ax.set_xlabel("number of qubits $N$")
    ax.set_ylabel(r"unclipped Renyi factor $S_2$")
    ax.set_title(
        f"{ALGORITHM.upper()} ideal-output Renyi-2 scaling\n"
        "exact statevector data; no noise, TVD, randomized compiling, or QCAP"
    )
    ax.set_xticks(list(ns))
    if np.all(np.asarray(s2) > 0.0):
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.2, which="both")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    rows = []
    for n in QUBIT_SIZES:
        circ, source = load_algorithm_circuit(int(n))
        prob = ideal_probability(circ)
        factors = ideal_factors(prob)
        rows.append({"n": int(n), "source": source, **factors})
        print(
            f"N={n:>2} source={source:<14} gates={len(circ.gates):>6} "
            f"support={factors['support']:>7} max(p)={factors['max_probability']:.6f} "
            f"S2={factors['s2']:.8g} Delta2={factors['delta2']:.8g} "
            f"H2/N={factors['h2_density']:.8g}",
            flush=True,
        )

    ns = np.asarray([row["n"] for row in rows], dtype=int)
    s2 = np.asarray([row["s2"] for row in rows], dtype=float)
    delta2 = np.asarray([row["delta2"] for row in rows], dtype=float)
    h2 = np.asarray([row["h2"] for row in rows], dtype=float)
    h2_density = np.asarray([row["h2_density"] for row in rows], dtype=float)
    support = np.asarray([row["support"] for row in rows], dtype=int)
    max_probability = np.asarray([row["max_probability"] for row in rows], dtype=float)

    power_fit = fit_s2_power2(ns, s2)
    entropy_fit = fit_entropy_deficit(ns, delta2)

    print("\n=== fits across N ===")
    if power_fit["ok"]:
        print(
            f"S2=A*2^(alpha N): A={power_fit['A']:.8g} +/- {power_fit['A_err']:.3g}, "
            f"alpha={power_fit['alpha']:.8g} +/- {power_fit['alpha_err']:.3g}, "
            f"R2={power_fit['r2']:.8g}"
        )
    else:
        print(f"direct S2 fit skipped: {power_fit['reason']}")

    if entropy_fit["ok"]:
        print(
            f"Delta2=aN+c: a={entropy_fit['a']:.8g} +/- {entropy_fit['a_err']:.3g}, "
            f"c={entropy_fit['c']:.8g} +/- {entropy_fit['c_err']:.3g}, "
            f"R2={entropy_fit['r2']:.8g}"
        )
        print(
            f"projected collision-entropy density H2/N -> "
            f"{entropy_fit['h2_density']:.8g} +/- {entropy_fit['h2_density_err']:.3g}"
        )
    else:
        print(f"Delta2 fit skipped: {entropy_fit['reason']}")

    plot_s2(ns, s2, power_fit, entropy_fit)

    save_results(
        OUT_DATA,
        algorithm=np.asarray(ALGORITHM),
        qubit_sizes=ns,
        s2=s2,
        collision_entropy_deficit=delta2,
        collision_entropy=h2,
        collision_entropy_density=h2_density,
        ideal_support=support,
        max_probability=max_probability,
        power_fit_ok=np.asarray(power_fit["ok"]),
        power_fit_A=np.asarray(power_fit.get("A", np.nan)),
        power_fit_A_err=np.asarray(power_fit.get("A_err", np.nan)),
        power_fit_alpha=np.asarray(power_fit.get("alpha", np.nan)),
        power_fit_alpha_err=np.asarray(power_fit.get("alpha_err", np.nan)),
        power_fit_r2=np.asarray(power_fit.get("r2", np.nan)),
        entropy_fit_ok=np.asarray(entropy_fit["ok"]),
        entropy_fit_a=np.asarray(entropy_fit.get("a", np.nan)),
        entropy_fit_a_err=np.asarray(entropy_fit.get("a_err", np.nan)),
        entropy_fit_c=np.asarray(entropy_fit.get("c", np.nan)),
        entropy_fit_c_err=np.asarray(entropy_fit.get("c_err", np.nan)),
        entropy_fit_h2_density=np.asarray(entropy_fit.get("h2_density", np.nan)),
        entropy_fit_r2=np.asarray(entropy_fit.get("r2", np.nan)),
    )

    print(f"\nWrote {OUT_FIGURE}")
    print(f"Wrote {OUT_DATA}")


if __name__ == "__main__":
    main()
