"""Magic vs the distribution of Pauli-string coefficients under propagation.

Take a Clifford brickwork and inject `n_t` "magic" gates (pi/4 rotations). Pauli-
propagate an observable O backward to U^dag O U and look at the resulting sum of
Pauli strings  O = sum_k c_k P_k :

  * n_t = 0 (Clifford): O stays a SINGLE Pauli (|c|=1). No spread.
  * each magic gate that anticommutes with a term BRANCHES it, scaling both
    children by cos/sin(pi/4) = 1/sqrt(2). So coefficients land on 2^(-k/2), the
    number of terms grows, and the |c| distribution spreads toward many small
    coefficients -- which is exactly what truncation (max_weight/min_abs_coeff)
    later exploits.

We use the Julia-free validator (proxysim.pauliprop_validator.propagate) because
it returns the full {PauliString: coeff} dict directly. Produces
results/magic_coeffs.png.

Run:  python examples/run_magic_coeffs.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401
from proxysim import brickwork_magic
from proxysim.pauliprop_validator import propagate

N = 8
N_CYCLES = 4
SEED = 5
OBS = "Z" * N                              # full-support observable <Z_0 Z_1 ... Z_{n-1}>
MAGIC_LEVELS = [4, 8, 16, 24, 32, 48]      # number of injected pi/4 magic gates
FULL_PAULI = 4 ** N                        # size of the n-qubit Pauli space

# Sequential single-hue ramp (light -> dark) for the ORDERED magic level.
RAMP = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]


def collect(n_t):
    circ = brickwork_magic(N, N_CYCLES, n_t=n_t, seed=SEED)
    terms = propagate(circ, OBS)          # no truncation -> full Pauli sum
    mags = np.array([abs(c) for c in terms.values()])
    return mags[mags > 1e-12]             # drop numerically-cancelled terms


def main():
    data = {m: collect(m) for m in MAGIC_LEVELS}
    for m in MAGIC_LEVELS:
        print(f"  magic n_t={m:>2}: {len(data[m]):>6} Pauli terms, "
              f"|c| range [{data[m].min():.2e}, {data[m].max():.2e}]")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (left) distribution of coefficient magnitudes, log-x, one line per magic level
    lo = min(d[d > 0].min() for d in data.values() if len(d))
    bins = np.logspace(np.log10(lo) - 0.1, 0.1, 40)
    for m, col in zip(MAGIC_LEVELS, RAMP):
        d = data[m]
        counts, edges = np.histogram(d, bins=bins)
        centers = np.sqrt(edges[:-1] * edges[1:])
        ax1.plot(centers, counts, color=col, lw=2, label=f"n_t={m}")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=0.8)
    ax1.set_xlabel(r"Pauli-string coefficient magnitude  $|c_k|$")
    ax1.set_ylabel("number of Pauli strings")
    ax1.set_title("coefficient distribution vs injected magic", fontsize=11)
    ax1.legend(title="magic gates", fontsize=8, title_fontsize=8, frameon=False)
    ax1.grid(True, which="both", alpha=0.15)

    # (right) number of Pauli terms vs magic count, saturating at the 4^n ceiling
    xs = np.array(MAGIC_LEVELS)
    nterms = np.array([len(data[m]) for m in MAGIC_LEVELS])
    ax2.plot(xs, nterms, "o-", color="#084594", lw=2, ms=7, label="Pauli strings")
    ax2.axhline(FULL_PAULI, ls="--", color="#d55e00", lw=1.6,
                label=r"$4^{n}$ full Pauli space")
    ax2.set_yscale("log")
    ax2.set_xlabel("number of injected magic gates  $n_t$")
    ax2.set_ylabel("number of Pauli strings in  $U^{\\dagger}OU$")
    ax2.set_title("observable spread grows with magic", fontsize=11)
    ax2.legend(fontsize=9, frameon=False)
    ax2.grid(True, which="both", alpha=0.15)

    fig.suptitle(r"Pauli propagation of $\langle Z^{\otimes n}\rangle$ through a "
                 f"magic brickwork (n={N}, {N_CYCLES} cycles)", fontsize=12)
    fig.tight_layout()
    out = _bootstrap.RESULTS_DIR + "/magic_coeffs.png"
    fig.savefig(out, dpi=150)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
