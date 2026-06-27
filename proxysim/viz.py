"""Visualization helpers: circuit diagrams (qiskit + quimb TN) and output bars.

Uses a non-interactive matplotlib backend so it works headless / saves to file.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from .backends.statevector import StatevectorBackend  # noqa: E402
from .backends.tensornetwork import _NAME  # noqa: E402


def draw_qiskit_circuit(circuit, path: str, fold: int = 28):
    """Render the circuit as a qiskit gate diagram (needs pylatexenc)."""
    qc = StatevectorBackend()._build(circuit)
    fig = qc.draw("mpl", fold=fold)
    fig.suptitle(f"qiskit circuit diagram - {circuit.name}", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_quimb_tn(circuit, path: str):
    """Render the gate-level tensor network via quimb's native draw().

    (This draws the full gate network for illustration; the TN *backend* itself
    simulates with an MPS / swap+split.)
    """
    import quimb.tensor as qtn

    C = qtn.Circuit(circuit.n_qubits)
    for g in circuit.gates:
        C.apply_gate(_NAME[g.name], *g.params, *g.qubits)
    fig = C.psi.draw(
        color=["PSI0", "H", "CZ", "CX", "S", "SDG", "X", "Y", "Z", "SX", "SXDG", "RZ"],
        figsize=(9, 6),
        return_fig=True,
        show_tags=False,
    )
    fig.suptitle(f"quimb tensor network (gate-level) - {circuit.name}", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_distribution(report, path: str, top_k: int = 12):
    """Grouped bar chart of the EXACT P(x) from each backend (they coincide)."""
    ref = next((r.distribution for r in report.results if r.backend == report.reference),
               report.results[0].distribution if report.results else {})
    rows = [bs for bs, _ in sorted(ref.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]]
    series = [(r.backend.split("(")[-1].rstrip(")"), r.distribution) for r in report.results]

    import numpy as np

    x = np.arange(len(rows))
    n = max(len(series), 1)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.8), 4.5))
    for i, (label, dist) in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * w, [dist.get(b, 0.0) for b in rows], w, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"|{b}>" for b in rows], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("exact probability  P(x)")
    title = "exact output distribution (deterministic; backends coincide)"
    if report.shots:
        title += f" -- {report.shots} shots also drawn"
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def panel(circuit, report, outdir: str, prefix: str = "viz"):
    """Make the three images and stitch them into one combined panel PNG."""
    os.makedirs(outdir, exist_ok=True)
    paths = {
        "circuit": draw_qiskit_circuit(circuit, os.path.join(outdir, f"{prefix}_circuit_qiskit.png")),
        "tn": draw_quimb_tn(circuit, os.path.join(outdir, f"{prefix}_tn_quimb.png")),
        "dist": plot_distribution(report, os.path.join(outdir, f"{prefix}_distribution.png")),
    }
    combined = os.path.join(outdir, f"{prefix}_panel.png")
    imgs = [plt.imread(paths[k]) for k in ("circuit", "tn", "dist")]
    titles = ["qiskit circuit", "quimb tensor network", "exact output distribution"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 16))
    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=12)
        ax.axis("off")
    fig.suptitle(f"proxysim - {circuit.name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(combined, dpi=130)
    plt.close(fig)
    paths["panel"] = combined
    return paths
