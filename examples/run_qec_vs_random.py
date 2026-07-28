"""The error bound against ACTUAL performance, for two circuit families as the
physical error rate grows -- a random circuit vs a quantum error-correcting code.
Everything on ONE shared log axis of error probability, so vertical gaps are real
(no twin axes, whose apparent gap is an artifact of independent axis scalings).

Left panel  -- a random, moderate-entropy Clifford circuit (NOT a mirror/Loschmidt
  echo): the CB/QCAP bound vs the actual output-distribution TVD. With no decoding,
  the bound sits just above the real TVD -- a genuine, tight-ish upper bound.

Right panel -- a rotated surface code memory (stim), distance d, d rounds, THREE
  curves, all computed from one noisy circuit so they are perfectly consistent:
  * AKN bound = P(any physical error) = 1 - prod(1 - p_i) over EVERY noise location
    (all ~800 of them: every syndrome measurement each round, resets, 1q + 2q gates
    -- not just the entangling cycles). This is the honest realisation of the AKN
    chain TVD <= sum eps_diamond; CB is how you'd estimate the per-cycle infidelities
    on hardware.
  * actual TVD of the FULL measurement record (data + all ancilla/syndrome bits) ~
    P(>=1 detection event): the ideal record is uniform on the detectors=0 subspace,
    so the deviating mass is the fraction of shots with any detection. THIS is the
    quantity the bound actually bounds -- and the bound is TIGHT on it, because in a
    QEC circuit almost every physical error flips a detector.
  * logical error rate (LER) after pymatching decoding.
  So the bound tightly tracks the full-record TVD (both ~ P(any error)); the orders-
  of-magnitude drop to the LER is NOT bound looseness -- it is entirely DECODING.

Refs: stim surface-code generator + pymatching DEM decoding (Gidney, "Stim"); the
QCAP/AKN bound as in Hashim et al. 2408.12064.

Run:  python examples/run_qec_vs_random.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymatching
import stim

import _bootstrap  # noqa: F401
from proxysim import NoiseModel, even_pairs, odd_pairs, save_results
from proxysim.backends.stabilizer import StabilizerBackend
from proxysim.benchmarking import cycle_benchmark, qcap_bound, readout_fidelity
from proxysim.simulate import noisy_tvd_vs_support

P_VALUES = np.array([1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2])
CB_DEPTHS = [1, 2, 4, 8]
OUT = _bootstrap.RESULTS_DIR + "/qec_vs_random.png"
OUT_DATA = _bootstrap.RESULTS_DIR + "/qec_vs_random_data.npz"

_SB = StabilizerBackend()


def noise_at(p):
    """2-qubit depolarizing on entangling gates + readout flip, at rate p."""
    return NoiseModel(enabled=True, p1=0.0, p2=p, p_readout=p, p_idle=0.0)


# ---------------------------------------------------------------------------
# Left panel: random MODERATE-entropy Clifford circuit
#
# H on K_ENT qubits injects K_ENT free bits (support = 2^K_ENT); CX-only brickwork
# cycles spread them bijectively (dim-preserving 1q gates in between). We keep the
# support modest and the shot count high so the finite-shot TVD "phantom floor"
# ~ sqrt(support / shots) stays negligible -- otherwise it would swamp the true TVD
# at small p and the (true-TVD) bound would look violated.
# ---------------------------------------------------------------------------
N_RAND, K_ENT, DEPTH, N_INST, TVD_SHOTS = 10, 5, 6, 6, 40_000


def moderate_entropy_circuit(n, k, depth, seed):
    import random as _random
    from proxysim.circuit import Circuit

    rng = _random.Random(seed)
    circ = Circuit(n, name=f"modent_n{n}_k{k}_d{depth}")
    for q in rng.sample(range(n), k):              # inject exactly k free bits
        circ.h(q)
    dim_preserving = ["i", "x", "z", "s", "sdg"]   # keep the Z-support dimension
    for d in range(depth):
        for a, b in (even_pairs(n) if d % 2 == 0 else odd_pairs(n)):
            circ.cx(a, b)                          # bijective spread, keeps support = 2^k
        for q in range(n):
            circ.add(rng.choice(dim_preserving), q)
    return circ


def random_panel():
    circ = moderate_entropy_circuit(N_RAND, K_ENT, DEPTH, seed=7)
    support = [int(k, 2) for k in _SB.exact_distribution(circ)]   # q0 = MSB codes
    n_even, n_odd = (DEPTH + 1) // 2, DEPTH // 2
    floor = (len(support) / (2 * np.pi * TVD_SHOTS)) ** 0.5
    print(f"[random] n={N_RAND}, depth={DEPTH}, ideal support = {len(support)} outcomes "
          f"({np.log2(len(support)):.0f} bits); shot-noise TVD floor ~ {floor:.3f}")

    bounds, tvd_mean, tvd_scatter = [], [], []
    print(f"  {'p':>8}{'bound':>10}{'mean TVD':>10}")
    for p in P_VALUES:
        noise = noise_at(p)
        cbE = cycle_benchmark(even_pairs(N_RAND), N_RAND, CB_DEPTHS, noise, twoq="CX", seed=1)
        cbO = cycle_benchmark(odd_pairs(N_RAND), N_RAND, CB_DEPTHS, noise, twoq="CX", seed=2)
        ro_fid, ro_std = readout_fidelity(N_RAND, noise, seed=3)
        efs = {"E": (cbE["e_F"], cbE["e_F_std"]), "O": (cbO["e_F"], cbO["e_F_std"])}
        bnd = qcap_bound({"E": n_even, "O": n_odd}, efs, ro_fid, ro_std)["error"]
        tvds = [noisy_tvd_vs_support(circ, noise, support, TVD_SHOTS, seed=10 + i)
                for i in range(N_INST)]
        bounds.append(bnd); tvd_mean.append(float(np.mean(tvds))); tvd_scatter.append(tvds)
        print(f"  {p:>8.3f}{bnd:>10.4f}{np.mean(tvds):>10.4f}", flush=True)
    return np.array(bounds), np.array(tvd_mean), tvd_scatter


# ---------------------------------------------------------------------------
# Right panel: rotated surface code (stim) -- CB bound (left axis) + LER (right)
# ---------------------------------------------------------------------------
D, ROUNDS, LER_SHOTS = 5, 5, 100_000


_NOISE_INSTR = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR",
                "DEPOLARIZE", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2"}


def akn_union_bound(circuit):
    """AKN/QCAP bound realised exactly: TVD <= sum_c eps_diamond(c) <= P(any error)
    = 1 - prod over every noise location of (1 - p_fire). This counts ALL the
    circuit's noise (every syndrome measurement each round, resets, 1q + 2q gates),
    so it is a genuine upper bound on the full-record TVD -- unlike an entangling-
    only cycle count. On hardware, CB is how you would estimate each cycle's
    infidelity; in simulation we know the channels, so we sum them exactly."""
    logprod, nloc = 0.0, 0
    for inst in circuit.flattened():
        if inst.name in _NOISE_INSTR:
            p = inst.gate_args_copy()[0]
            nt = len(inst.targets_copy())
            k = nt // 2 if inst.name == "DEPOLARIZE2" else nt
            logprod += k * np.log1p(-p)
            nloc += k
    return 1.0 - np.exp(logprod), nloc


def surface_bound_tvd_ler(distance, rounds, p, shots):
    """From ONE noisy circuit (so bound / TVD / LER are perfectly consistent):
      * bound = AKN union bound (upper-bounds the full-record TVD),
      * TVD   = actual TVD of the full measurement record ~ P(>=1 detection event)
                (the ideal record is uniform on the detectors=0 subspace, so the
                deviating mass is the fraction of shots with any detection),
      * LER   = logical error rate after pymatching decoding.
    """
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=distance, rounds=rounds,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p)
    bound, nloc = akn_union_bound(circuit)
    det, obs = circuit.compile_detector_sampler().sample(shots, separate_observables=True)
    tvd = float(np.mean(det.any(axis=1)))                         # full-record TVD
    match = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    ler = float(np.mean(match.decode_batch(det)[:, 0] != obs[:, 0]))
    return bound, tvd, ler, nloc


def surface_panel():
    nloc0 = surface_bound_tvd_ler(D, ROUNDS, P_VALUES[0], 1)[3]
    print(f"[surface] rotated d={D}, {ROUNDS} rounds; {nloc0} noise locations "
          f"(every syndrome measurement, reset, and gate -- not just the CX)")
    bounds, tvds, lers = [], [], []
    print(f"  {'p':>8}{'AKN bound':>11}{'full-record TVD':>17}{'LER':>12}")
    for p in P_VALUES:
        b, tvd, ler, _ = surface_bound_tvd_ler(D, ROUNDS, p, LER_SHOTS)
        bounds.append(b); tvds.append(tvd); lers.append(ler)
        print(f"  {p:>8.3f}{b:>11.4f}{tvd:>17.4f}{ler:>12.2e}", flush=True)
    return np.array(bounds), np.array(tvds), np.array(lers)


def main():
    print("=" * 66)
    rb, rt, rscatter = random_panel()
    print()
    sb, sc_tvd, sl = surface_panel()
    save_results(OUT_DATA, p=P_VALUES, rand_bound=rb, rand_tvd=rt,
                 rand_tvd_scatter=np.array(rscatter), sc_bound=sb, sc_tvd=sc_tvd,
                 sc_ler=sl, distance=D, rounds=ROUNDS)

    # Both panels: ONE shared log axis of "probability of an error". Everything
    # plotted is an error probability, so the vertical distance between the bound
    # and the actual performance is a REAL, readable quantity -- unlike a twin-axis
    # plot, whose apparent gap is just an artifact of two independent axis scalings.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.3), sharey=True)
    ORANGE, BLUE, GREEN = "#D55E00", "#0072B2", "#009E73"

    # -- left: random circuit, bound vs actual TVD --
    for p, tvds in zip(P_VALUES, rscatter):
        axL.plot([p] * len(tvds), tvds, "o", color=BLUE, ms=4, alpha=0.5,
                 label="measured TVD" if p == P_VALUES[0] else None)
    axL.plot(P_VALUES, rt, "-", color=BLUE, lw=1.8, alpha=0.9, label="actual TVD (mean)")
    axL.plot(P_VALUES, rb, "-", color=ORANGE, lw=2.5, label="CB / QCAP bound")
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel("physical error rate  p")
    axL.set_ylabel("probability of an error  (log scale)")
    axL.set_title(f"random moderate-entropy circuit (n={N_RAND})\n"
                  "no decoding: bound sits just above the TVD", fontsize=10.5)
    axL.grid(True, which="both", alpha=0.15)
    axL.legend(frameon=False, fontsize=9, loc="upper left")

    # -- right: surface code -- bound, actual full-record TVD, and LER, one axis --
    axR.plot(P_VALUES, sb, "-", color=ORANGE, lw=2.5, label="AKN bound (P any error)")
    axR.plot(P_VALUES, sc_tvd, "^--", color=BLUE, lw=1.8, ms=6,
             label="actual TVD of full record")
    axR.plot(P_VALUES, sl, "s-", color=GREEN, lw=2, ms=6, label="logical error rate (decoded)")
    axR.set_xscale("log")
    axR.set_xlabel("physical error rate  p")
    axR.set_title(f"surface code d={D}, {ROUNDS} rounds (stim + pymatching)\n"
                  "bound is TIGHT on the full-record TVD; the huge gap is DECODING",
                  fontsize=10.5)
    axR.grid(True, which="both", alpha=0.15)
    # the meaningful gap is TVD (what the bound bounds) vs LER (what decoding leaves)
    p0 = P_VALUES[0]
    axR.annotate("", xy=(p0, sc_tvd[0]), xytext=(p0, sl[0]),
                 arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.3))
    axR.text(p0 * 1.25, (sc_tvd[0] * sl[0]) ** 0.5,
             f"~{np.log10(sc_tvd[0] / sl[0]):.1f} decades\nremoved by decoding",
             fontsize=8.5, color="0.25", va="center")
    axR.legend(frameon=False, fontsize=9, loc="lower right")
    axR.set_ylim(min(sl.min(), rt.min()) * 0.5, 1.7)

    fig.suptitle("Bound vs actual performance (shared log axis): random circuit "
                 "(bound tracks TVD) vs surface code (bound tracks TVD, but decoding "
                 "sinks the LER)", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"\nWrote {OUT}, {OUT_DATA}")


if __name__ == "__main__":
    main()
