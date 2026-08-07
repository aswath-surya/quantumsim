"""Build a QASM bank of randomly-compiled MQT Bench algorithms. Nothing is simulated.

This is the generation half of the pipeline `run_qft_cycle_sweep.py` runs in-process.
That script builds `U^d`, dresses every two-qubit gate with `proxysim.rc.pauli_twirl`,
splices one sampled noise trajectory into each dressed circuit, and simulates the result
on a statevector backend. Here the same circuits are written to disk instead, so the
widths are bounded by the consumer (C-3PQ / ExaTN) rather than by `2^n` in memory.

For a single algorithm at a fixed width `n`, and for each repetition count `d`:

    logical target = U^d
    2q cycles      = d * (two-qubit gates in U)

    for rc in range(--n-rc):
        dressed, virtual = pauli_twirl(U^d, mark_virtual=True)
        for t in range(--n-traj):
            write sample_trajectory(dressed, noise, rng, virtual=virtual)

`mark_virtual=True` is load-bearing: the twirl Paulis are frame changes, and charging
them `p1`/`p_idle` would make an RC-vs-raw comparison a comparison of gate counts. The
noise trajectory is baked into the written circuit because C-3PQ and ExaTN are both
noiseless simulators -- the stochastic Pauli realisation has to be in the QASM or it is
nowhere. Each of the `n_rc * n_traj` files therefore carries weight `1/(n_rc*n_traj)`
within its group, and the group's weighted sum is the RC-averaged distribution.

Readout error is NOT in the circuits. `sample_trajectory` does not apply it and neither
does this script; apply `proxysim.noise.apply_readout_to_distribution` to the recovered
distribution downstream. `p_readout` is recorded in the manifest for that purpose.

QASM is unlowered proxysim IR in the `full` dialect, matching every other bank in this
repo: `c3pq_stage.py` calls `c3pq.lower_for_c3pq` itself, and `run_exatn_bank.py` lowers
by default. Pre-lowering here would double-apply it.

Consumers
---------
    # C-3PQ -- driven off manifest.json
    python examples/c3pq_stage.py --bank rc_bank/qft_n14

    # ExaTN -- rglobs the same tree, ignores the manifest
    python examples/run_exatn_bank.py --qasm-bank rc_bank/qft_n14 --output-dir results/...

Examples
--------
    # what would be written, without writing it
    python examples/run_mqtbench_rc.py --algo qft --n 14 --dry-run

    # fetch straight from MQT Bench
    python examples/run_mqtbench_rc.py --algo qft --n 14 --reps 1 2 4 8

    # or start from a QASM file already on disk
    python examples/run_mqtbench_rc.py --qasm examples/qft14.qasm --reps 1 2 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
import warnings
import zlib

warnings.filterwarnings("ignore")

import _bootstrap  # noqa: F401
from proxysim import NoiseModel
from proxysim.c3pq import stage_name
from proxysim.circuit import circuit_from_qasm
from proxysim.mqtbank import repeat
from proxysim.noise import sample_trajectory
from proxysim.qasm import circuit_to_qasm
from proxysim.rc import pauli_twirl

REPETITIONS = [1, 2, 3, 4, 6, 8]
N_RC_REALIZATIONS = 24
N_TRAJ_PER_RC = 12
SEED = 42

#: Above this width the optional --validate smoke test is skipped: it builds a dense
#: statevector, which is the exact thing this bank exists to avoid.
VALIDATE_MAX_QUBITS = 20


def _seed(*parts) -> int:
    """A deterministic seed from the identity of a file, so the bank is reproducible and
    resumable regardless of the order things are generated in."""
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
class Bank:
    """Accumulates groups and writes their circuits, or just counts under --dry-run.

    Mirrors ``build_bounding_bank.Bank`` so the manifest it emits is the one
    ``c3pq_stage.py`` already knows how to read.
    """

    def __init__(self, root: str, dry_run: bool, force: bool):
        self.root, self.dry_run, self.force = root, dry_run, force
        self.groups: list = []
        self.files = self.bytes = self.skipped = 0

    def group(self, name: str, subdir: str, **fields) -> dict:
        """Open a group. ``name`` is validated as a C-3PQ staging stem here rather than
        in the stager, so a naming problem surfaces now and not thousands of
        codegen-seconds later."""
        g = dict(name=stage_name(name), dir=os.path.join(subdir, name),
                 files=[], **fields)
        self.groups.append(g)
        return g

    def add(self, g: dict, stem: str, circuit, measured, header_lines=(), **fields):
        """Write one member circuit of group ``g``."""
        stem = stage_name(stem)
        text = circuit_to_qasm(circuit, measured=measured, header_lines=header_lines)
        g["files"].append(dict(file=f"{stem}.qasm", stem=stem,
                               gates=len(circuit.gates), **fields))
        self.files += 1
        self.bytes += len(text)
        if self.dry_run:
            return
        path = os.path.join(self.root, g["dir"], f"{stem}.qasm")
        if os.path.exists(path) and not self.force:
            self.skipped += 1
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)

    def close(self, g: dict) -> None:
        """Assign the within-group weights. They are equal by construction -- the
        trajectories are i.i.d. draws from the same channel."""
        w = 1.0 / len(g["files"])
        for f in g["files"]:
            f["weight"] = w

    def json(self, path: str, obj) -> None:
        if self.dry_run:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=1)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
def load_base(args):
    """``(circuit, measured, algo_label)`` for the algorithm this bank is built from.

    ``measured`` is threaded through from the source rather than assumed to be
    ``range(n)`` -- MQT's ``dj`` leaves its ancilla unmeasured, and writing it as
    measured would silently shift every creg bit.
    """
    if args.qasm:
        with open(args.qasm, encoding="utf-8") as fh:
            circ, measured = circuit_from_qasm(fh.read())
        label = args.algo or os.path.splitext(os.path.basename(args.qasm))[0]
        circ.name = f"{label}_n{circ.n_qubits}"
        if args.n is not None and args.n != circ.n_qubits:
            sys.exit(f"--n {args.n} contradicts {args.qasm}, which is "
                     f"{circ.n_qubits} qubits")
        return circ, measured, label

    from proxysim import mqtbank
    circ, measured = mqtbank.fetch(args.algo, args.n)
    return circ, measured, args.algo


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_twirl(base_circ, seed: int) -> None:
    """Check that the explicit RC pass preserves the logical unitary before generating
    thousands of files that inherit any mistake. Dense; small widths only."""
    import numpy as np
    from qiskit.quantum_info import Statevector

    from proxysim.backends import StatevectorBackend

    sv = StatevectorBackend()
    dressed, virtual = pauli_twirl(base_circ, random.Random(seed), mark_virtual=True)

    psi = np.asarray(Statevector(sv._build(base_circ)).data, dtype=complex)
    phi = np.asarray(Statevector(sv._build(dressed)).data, dtype=complex)
    fidelity = float(abs(np.vdot(psi, phi)) ** 2)

    print(f"explicit-RC smoke test:\n"
          f"    original gates: {len(base_circ.gates)}  "
          f"dressed gates: {len(dressed.gates)}\n"
          f"    virtual RC gates: {len(virtual)}\n"
          f"    ideal state fidelity: {fidelity:.12f}\n")

    if not virtual:
        warnings.warn(
            "pauli_twirl reported no virtual gate indices; the twirl Paulis would be "
            "charged physical p1 and idle noise, making this an RC-vs-raw comparison of "
            "gate counts rather than of the channel.", RuntimeWarning)
    if not np.isclose(fidelity, 1.0, atol=1e-9, rtol=1e-9):
        sys.exit("the dressed circuit is not logically equivalent to the source "
                 "circuit; refusing to generate the bank")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def build_depth(bank, base_circ, measured, algo, n, d, noise, args):
    """One group: every RC realization x trajectory of ``U^d``, uniformly weighted."""
    target = repeat(base_circ, d)
    cycles = sum(len(g.qubits) == 2 for g in target.gates)

    where = os.path.join(f"n{n:02d}", algo, f"d{d:03d}")
    common = [f"{algo} n={n} depth={d} (U^{d}) | MQT Bench level=alg",
              f"2q cycles={cycles} measured={measured}",
              "readout error is NOT in this circuit; apply it to the distribution"]

    g = bank.group(f"rc_{algo}_n{n:02d}_d{d:03d}", where,
                   kind="tvd", arm="rc", n_qubits=n, algorithm=algo, depth=d,
                   two_qubit_cycles=cycles, estimator="probs", measured=list(measured),
                   n_rc=args.n_rc, n_traj=args.n_traj)

    for rc in range(args.n_rc):
        rc_seed = _seed(args.seed, "rc", algo, n, d, rc)
        dressed, virtual = pauli_twirl(target, random.Random(rc_seed),
                                       mark_virtual=True)
        for t in range(args.n_traj):
            traj = sample_trajectory(dressed, noise,
                                     random.Random(_seed(rc_seed, "traj", t)),
                                     virtual=virtual)
            bank.add(g, f"rc_{algo}_n{n:02d}_d{d:03d}_r{rc:03d}_t{t:03d}",
                     traj, measured, rc=rc, traj=t, rc_seed=rc_seed,
                     header_lines=common + [
                         f"arm=rc | randomization={rc} trajectory={t} seed={rc_seed}",
                         "every 2q gate Pauli-twirled (mark_virtual=True); one sampled "
                         "noise trajectory spliced in",
                     ])
    bank.close(g)

    if args.with_ideal:
        gi = bank.group(f"ideal_{algo}_n{n:02d}_d{d:03d}", where,
                        kind="tvd", arm="ideal", n_qubits=n, algorithm=algo, depth=d,
                        two_qubit_cycles=cycles, estimator="probs",
                        measured=list(measured))
        bank.add(gi, f"ideal_{algo}_n{n:02d}_d{d:03d}", target, measured,
                 header_lines=common + ["arm=ideal (untwirled, no noise of any kind)"])
        bank.close(gi)

    return cycles


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("source circuit (one of --algo/--n or --qasm)")
    src.add_argument("--algo", help="MQT Bench algorithm name, e.g. qft")
    src.add_argument("--n", type=int, help="qubit count (fixed for the whole bank)")
    src.add_argument("--qasm", help="read the algorithm from this QASM file instead")

    ap.add_argument("--reps", nargs="+", type=int, default=REPETITIONS,
                    help="repetition counts d; the bank holds one group per U^d")
    ap.add_argument("--n-rc", type=int, default=N_RC_REALIZATIONS,
                    help="independently twirled circuits per depth")
    ap.add_argument("--n-traj", type=int, default=N_TRAJ_PER_RC,
                    help="noise trajectories per twirled circuit")

    ap.add_argument("--p1", type=float, default=5e-5)
    ap.add_argument("--p2", type=float, default=2.5e-4)
    ap.add_argument("--p-readout", type=float, default=2.5e-4,
                    help="recorded in the manifest; NOT baked into the circuits")
    ap.add_argument("--p-idle", type=float, default=5e-5)
    ap.add_argument("--p-z1", type=float, default=0.0)
    ap.add_argument("--p-zz", type=float, default=0.0)
    ap.add_argument("--theta-1q", type=float, default=0.0)
    ap.add_argument("--theta-zz", type=float, default=0.0)

    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None,
                    help="bank root (default results/rc_bank/<algo>_n<N>)")
    ap.add_argument("--with-ideal", action="store_true",
                    help="also emit an untwirled noiseless arm per depth, so a TVD can "
                         "be computed downstream without a separate reference run")
    ap.add_argument("--validate", action="store_true",
                    help=f"statevector-check that the twirl preserves the logical "
                         f"unitary before generating (n <= {VALIDATE_MAX_QUBITS})")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written without writing it")
    ap.add_argument("--force", action="store_true",
                    help="rewrite files that already exist")
    args = ap.parse_args()

    if not args.qasm and (not args.algo or args.n is None):
        ap.error("give either --qasm, or both --algo and --n")
    if any(d < 1 for d in args.reps):
        ap.error("--reps must be positive")

    noise = NoiseModel(enabled=True, p1=args.p1, p2=args.p2,
                       p_readout=args.p_readout, p_idle=args.p_idle,
                       p_z1=args.p_z1, p_zz=args.p_zz,
                       theta_1q=args.theta_1q, theta_zz=args.theta_zz)

    base_circ, measured, algo = load_base(args)
    n = base_circ.n_qubits
    cycles_per_rep = sum(len(g.qubits) == 2 for g in base_circ.gates)
    if cycles_per_rep == 0:
        sys.exit("the source circuit contains no two-qubit gates; there is nothing "
                 "for RC to twirl")

    out = args.out or os.path.join(_bootstrap.RESULTS_DIR, "rc_bank", f"{algo}_n{n:02d}")

    print(f"{base_circ.summary()}\n"
          f"algorithm={algo} n={n} measured={measured}\n"
          f"2q cycles per repetition: {cycles_per_rep}  "
          f"non-Clifford: {not base_circ.is_clifford}\n"
          f"noise: p1={noise.p1:.3e} p2={noise.p2:.3e} p_idle={noise.p_idle:.3e} "
          f"p_z1={noise.p_z1:.3e} p_zz={noise.p_zz:.3e}\n"
          f"       theta_1q={noise.theta_1q:.3e} theta_zz={noise.theta_zz:.3e}\n"
          f"       p_readout={noise.p_readout:.3e} (manifest only, not in circuits)\n"
          f"bank root: {out}\n")

    if args.validate:
        if n > VALIDATE_MAX_QUBITS:
            print(f"skipping --validate: n={n} exceeds {VALIDATE_MAX_QUBITS} qubits\n")
        else:
            validate_twirl(base_circ, args.seed)

    bank = Bank(out, args.dry_run, args.force)

    print(f"{'d':>5}{'2q cycles':>12}{'gates':>10}{'files':>9}")
    for d in sorted(set(args.reps)):
        before = bank.files
        cycles = build_depth(bank, base_circ, measured, algo, n, d, noise, args)
        print(f"{d:>5}{cycles:>12}{len(base_circ.gates) * d:>10}"
              f"{bank.files - before:>9}", flush=True)

    bank.json(os.path.join(out, "manifest.json"), {
        "generator": "examples/run_mqtbench_rc.py",
        "mirrors": "examples/run_qft_cycle_sweep.py",
        "qasm_version": "2.0",
        "note": "QASM is UNLOWERED proxysim IR; c3pq_stage.py applies "
                "c3pq.lower_for_c3pq and run_exatn_bank.py lowers by default. Each rc "
                "file has one sampled noise trajectory baked in and the twirl Paulis "
                "marked virtual. Readout error is NOT in the circuits. Each group "
                "records `measured`: c3pq_stage.py discards it and measures every "
                "qubit (its partitioner requires a full round), so a C-3PQ probs "
                "vector must be marginalized onto those qubits, while ExaTN reads "
                "the creg as written. Only algorithms with unmeasured ancillas "
                "(MQT's dj) differ between the two.",
        "config": {"algorithm": algo, "n_qubits": n, "source_qasm": args.qasm,
                   "measured": list(measured) if measured is not None else None,
                   "reps": sorted(set(args.reps)), "n_rc": args.n_rc,
                   "n_traj": args.n_traj, "seed": args.seed,
                   "with_ideal": args.with_ideal,
                   "cycles_per_rep": cycles_per_rep},
        "noise": dataclasses.asdict(noise),
        "groups": bank.groups,
    })

    verb = "would write" if args.dry_run else "bank contains"
    print(f"\n{verb} {bank.files} circuits in {len(bank.groups)} groups, "
          f"{bank.bytes / 1e6:.1f} MB")
    if bank.skipped:
        print(f"  {bank.files - bank.skipped} written this run; {bank.skipped} already "
              f"existed and were left alone (--force to rewrite)")
    if not args.dry_run:
        print(f"  bank at {out}, index at {out}/manifest.json")
        print("  nothing was simulated; run c3pq_stage.py or run_exatn_bank.py next")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
