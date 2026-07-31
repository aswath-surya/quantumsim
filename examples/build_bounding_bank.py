"""Emit every circuit `run_bounding.py` would have simulated in-process, as QASM.

`examples/run_bounding.py` fuses circuit construction, noise, simulation, fitting and
plotting into one process, and every number in it comes from proxysim's own backends.
This script performs the first third of that job only: it *writes the circuits*, so
C-3PQ can do the simulating and `examples/c3pq_analyze.py` can do the analysis.

    build_bounding_bank.py -> qasm_bank_bounding/   (this file)
    c3pq_stage.py          -> staging/<batch>/
    c3pq_build.sh          -> <batch>.x
    c3pq_run.sh            -> results/c3pq/<run>/
    c3pq_analyze.py        -> TVD vs bound

WHAT GETS WRITTEN
-----------------
The headline measurement is the TVD between a noiseless arm and a Pauli-simulated arm,
so both arms are emitted as explicit circuits. Per (width, mode, depth, instance):

  ideal   1 circuit    the test circuit as `run_bounding.test_circuit` builds it
  noisy   K circuits   `noise.sample_trajectory(circ, NOISE, rng)`
  rc      K circuits   a fresh `rc.pauli_twirl` randomization, then a trajectory of it

and per (width, mode, cycle, CB depth m, decay r) the cycle-benchmarking sequences from
`cb_emit.cb_circuit`, again as one noiseless copy plus K trajectories.

The stochastic parts of the noise model (`p1`, `p2`, `p_idle`) are *sampled into the
circuit* -- that is what a trajectory is, and it is the only way a noiseless simulator
can carry them. `theta_zz` is deterministic and is spliced in as `cp(theta)` on every
trajectory, so turning it on costs no extra circuits. `p_readout` is deliberately absent
from the QASM: independent bit-flips factorise, so `noise.apply_readout_to_distribution`
applies them exactly in post-processing without any 2^n circuit machinery.

GROUPS: WHY THE MANIFEST IS THE INTERFACE
-----------------------------------------
K trajectories are not K results. They are one result -- the average of K exact
probability vectors, which is what removes the shot-noise floor `run_bounding` lives with
(`TVD_SHOTS = 50000` puts it at ~0.003) and leaves only ~1/sqrt(K) trajectory noise. So
the unit the pipeline actually cares about is the *group*, and every circuit in a group
carries the weight it contributes. `c3pq.emit_batch_harness` sums a group inside the
binary, so K trajectories collapse to one vector before anything reaches the filesystem.

`manifest.json` records the groups, and everything downstream is driven off it: the
stager reads it to know what to compile, the analyzer reads it to know what the outputs
mean. A CB QASM file in particular is not analyzable on its own -- the survival estimator
needs the propagated Pauli's sign and support (see `proxysim.cb_emit.survival`).

CALIBRATION
-----------
One extra group per width: a product state of per-qubit `ry` rotations chosen so its
largest amplitudes are non-degenerate and strictly ordered. Nothing in C-3PQ documents
whether qubit `i` is bit `i` or bit `n-1-i`, and a silently transposed index convention
would produce a plausible wrong TVD rather than an error. The reference here is a product
of 2x2 factors, so the analyzer can check it at any width without ever forming a 2^n
state vector -- unlike the statevector cross-check, which stops being possible at the
widths this pipeline exists to reach.

Examples
--------
    # counts and sizes, writing nothing
    python examples/build_bounding_bank.py --dry-run

    # the smoke config (the default): validates the chain, measures codegen cost
    python examples/build_bounding_bank.py

    # match run_bounding's statistics, at its width
    python examples/build_bounding_bank.py --widths 2 --instances 80 --trajectories 150

    # push past where the in-process version can go
    python examples/build_bounding_bank.py --widths 20 24 --instances 20 --trajectories 64
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import warnings
import zlib

warnings.filterwarnings("ignore")

import _bootstrap  # noqa: F401
from proxysim import GATE_SETS, NoiseModel, cb_emit, even_pairs, noise as noise_mod, odd_pairs, rc
from proxysim.benchmarking import _ONEQ_RANDOM, _ONEQ_STRUCTURED
from proxysim.c3pq import stage_name
from proxysim.circuit import Circuit, bench_brickwork
from proxysim.qasm import circuit_to_qasm

# --- run_bounding.py's configuration, verbatim where it is a choice and not a limit ---
WIDTHS = [2]                     # run_bounding.py:106 is N = 2; --widths pushes past it
DEPTHS = [1, 2, 4, 8, 16, 32, 64]                          # run_bounding.py:113
CB_DEPTHS = [1, 2, 4, 8, 16, 24]                           # run_bounding.py:186
MODES = ("random", "structured")                           # run_bounding.py:220
THETA_ZZ = 0.00                                            # run_bounding.py:117
ONEQ_SET = "t"                                             # run_bounding.py:119
SEED = 42                                                  # run_bounding.py:130
INSTANCE_STRIDE = 100            # run_bounding.py:206 -- s = SEED + 100 * i

# --- the knobs this pipeline adds, at their smoke values (see --help) ---
N_INSTANCES = 2                  # run_bounding uses 80; the smoke config measures first
N_TRAJ = 4                       # run_bounding uses 150; K sets the ~1/sqrt(K) floor
N_DECAYS = 4                     # cycle_benchmark's n_decays is 30

P1, P2, P_READOUT, P_IDLE = 1e-3, 1e-2, 1e-2, 1e-3         # run_bounding.py:131


def cycles_for(n: int):
    """The non-empty entangling layers at width ``n``, named as run_bounding names them.

    Empty layers are dropped exactly as run_bounding.py:111 does. ``odd_pairs(2)`` has no
    pairs at all, and benchmarking an empty cycle returns a spurious non-zero ``e_F`` (it
    still carries 1q and idle noise) that ``qcap_bound`` would then charge the bound for.
    """
    return {k: v for k, v in (("A", even_pairs(n)), ("B", odd_pairs(n))) if v}


def oneq_sets(oneq_set: str):
    """``{mode: 1q gate set}`` for the TEST circuits -- run_bounding.py:142-147.

    T goes into the random ansatz only, which is what makes the two panels a controlled
    comparison: same cycles, same noise, same bound, differing only in whether the ideal
    output is a stabilizer state. CB is untouched by this and keeps its own Clifford sets
    (`_ONEQ_RANDOM` / `_ONEQ_STRUCTURED`), because the twirl the CB protocol rests on is a
    twirl over the Clifford group.
    """
    sets = {"random": list(GATE_SETS["clifford"]),
            "structured": list(GATE_SETS["structured"])}
    if oneq_set == "t":
        sets["random"].append("t")
    elif oneq_set != "clifford":
        raise ValueError(f"--oneq-set must be 'clifford' or 't', got {oneq_set!r}")
    return sets


def _seed(*parts) -> int:
    """A deterministic seed from a file's identity, so the bank is reproducible and
    resumable no matter what order things are generated in."""
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def calibration_circuit(n: int):
    """A product state whose top ``n+1`` probabilities pin down the index convention.

    Qubit ``j`` gets ``ry(theta_j)`` with ``sin^2(theta_j/2) = eps*(j+1)``, so the
    outcome is a product of independent biased coins with *distinct* biases. Choosing
    ``eps = 0.4/n^2`` keeps every single-excitation probability above every
    double-excitation one by a factor of ~2.5, so the sorted top-k is
    ``|0...0>`` followed by the n single excitations in a known qubit order. Reading which
    bit is set in each of them says directly whether qubit ``j`` landed in bit ``j`` or in
    bit ``n-1-j`` -- and it says so from n+1 numbers rather than from a 2^n vector, which
    is the only reason this check survives to the widths the pipeline is for.
    """
    eps = 0.4 / max(n, 1) ** 2
    thetas = [2.0 * math.asin(math.sqrt(eps * (j + 1))) for j in range(n)]
    circ = Circuit(n, name=f"cal_n{n}")
    for j, theta in enumerate(thetas):
        circ.add("ry", j, params=(theta,))
    return circ, thetas


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
class Bank:
    """Accumulates groups and writes their circuits, or just counts under --dry-run."""

    def __init__(self, root: str, dry_run: bool, force: bool):
        self.root, self.dry_run, self.force = root, dry_run, force
        self.groups: list = []
        self.files = self.bytes = self.skipped = 0

    def group(self, name: str, subdir: str, **fields) -> dict:
        """Open a group. ``name`` is validated as a staging stem here rather than in the
        stager, so a naming problem surfaces while the bank is being written and not
        thousands of codegen-seconds later."""
        g = dict(name=stage_name(name), dir=os.path.join(subdir, name),
                 files=[], **fields)
        self.groups.append(g)
        return g

    def add(self, g: dict, k: int, circuit, header_lines=(), **fields) -> None:
        """Write one member circuit of group ``g``, weighted 1/K within it."""
        stem = stage_name(f"{g['name']}_k{k:03d}")
        text = circuit_to_qasm(circuit, measured=None, header_lines=header_lines)
        g["files"].append(dict(file=f"{stem}.qasm", stem=stem, k=k,
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
        trajectories are i.i.d. draws from the same channel -- and they are recorded
        per file anyway so the harness never has to know K."""
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
# The test-circuit arms: ideal / noisy / randomly-compiled
# ---------------------------------------------------------------------------
def build_tvd(bank, n, mode, oneq, depth, inst, cycle_list, noise, k_traj, base_seed):
    """The three arms of one (width, mode, depth, instance) point.

    The base circuit is `run_bounding.test_circuit` lifted verbatim, including its
    instance seeding (`s = SEED + 100*i`), so a bank instance *is* the circuit
    run_bounding would have built -- which is what makes the N=2 cross-check against
    `results/bounding_data.npz` a check and not a coincidence.
    """
    s = base_seed + INSTANCE_STRIDE * inst
    circ = bench_brickwork(n, depth, cycle_list, oneq=oneq, twoq="cz", seed=s,
                           final_oneq=True)
    where = f"n{n:02d}/tvd/{mode}"
    common = [f"n={n} mode={mode} depth={depth} instance={inst} circuit_seed={s}",
              f"base = bench_brickwork(oneq={oneq}, twoq=cz, final_oneq=True)"]

    g = bank.group(f"tvd_n{n:02d}_{mode}_d{depth:03d}_i{inst:03d}_ideal", where,
                   kind="tvd", arm="ideal", n_qubits=n, mode=mode, depth=depth,
                   instance=inst, circuit_seed=s, estimator="probs")
    bank.add(g, 0, circ, header_lines=common + ["arm=ideal (no noise of any kind)"])
    bank.close(g)

    for arm in ("noisy", "rc"):
        g = bank.group(f"tvd_n{n:02d}_{mode}_d{depth:03d}_i{inst:03d}_{arm}", where,
                       kind="tvd", arm=arm, n_qubits=n, mode=mode, depth=depth,
                       instance=inst, circuit_seed=s, estimator="probs")
        for k in range(k_traj):
            ts = _seed(base_seed, "tvd", n, mode, depth, inst, arm, k)
            rng = random.Random(ts)
            if arm == "noisy":
                traj = noise_mod.sample_trajectory(circ, noise, rng)
                virt = ()
                note = "arm=noisy (one sampled Pauli-error realisation, no twirl)"
            else:
                twirled, virt = rc.pauli_twirl(circ, rng, mark_virtual=True)
                traj = noise_mod.sample_trajectory(twirled, noise, rng, virtual=virt)
                note = (f"arm=rc (fresh Pauli twirl, then one error realisation); "
                        f"{len(virt)} twirl gates charged as frame changes")
            bank.add(g, k, traj, trajectory_seed=ts, n_virtual=len(virt),
                     header_lines=common + [note, f"trajectory={k} seed={ts}"])
        bank.close(g)


# ---------------------------------------------------------------------------
# The CB sequences
# ---------------------------------------------------------------------------
def build_cb(bank, n, mode, pairs, cycle, m, r, noise, k_traj, base_seed):
    """One cycle-benchmarking decay point: (cycle, mode, sequence length m, decay r).

    `cb_circuit` returns the propagated Pauli's sign and support alongside the circuit;
    both go into the group record, because the survival estimator
    `sign * mean(1 - 2*parity(bits[support]))` cannot be reconstructed from the QASM.
    The harness evaluates that parity exactly against the probability vector, so a CB
    group is one scalar per group no matter how wide the circuit is.

    The 1q dressing set is CB's own Clifford set, not the test circuit's: `cycle_benchmark`
    keys it off `mode` the same way (benchmarking.py:142) and T must not appear, since the
    propagated Pauli is computed with a stim tableau.
    """
    cb_oneq = _ONEQ_RANDOM if mode == "random" else _ONEQ_STRUCTURED
    circ, meta = cb_emit.cb_circuit(
        n, pairs, m, random.Random(_seed(base_seed, "cb", n, mode, cycle, m, r)),
        twoq="cz", oneq=cb_oneq)

    where = f"n{n:02d}/cb/{mode}"
    stem = f"cb_n{n:02d}_{mode}_c{cycle}_cbd{m:03d}_r{r:03d}"
    common = [f"CB | n={n} mode={mode} cycle={cycle}{list(map(list, pairs))} "
              f"length={m} decay={r}",
              f"dressed cycle = [1q Clifford layer from {cb_oneq}] + [cz on the cycle]",
              f"prep_pauli={meta['prep_pauli']} meas_pauli={meta['meas_pauli']} "
              f"sign={meta['sign']:+d} support={meta['support']}",
              "survival = sign * mean(1 - 2*parity(bits[support])); noiseless = +1"]
    fields = dict(kind="cb", n_qubits=n, mode=mode, cycle=cycle,
                  pairs=[list(p) for p in pairs], cb_depth=m, decay=r,
                  estimator="parity", sign=meta["sign"], support=meta["support"],
                  prep_pauli=meta["prep_pauli"], meas_pauli=meta["meas_pauli"])

    g = bank.group(f"{stem}_ideal", where, arm="ideal", **fields)
    bank.add(g, 0, circ, header_lines=common + ["arm=ideal -- survival must be exactly +1"])
    bank.close(g)

    g = bank.group(f"{stem}_noisy", where, arm="noisy", **fields)
    for k in range(k_traj):
        ts = _seed(base_seed, "cbtraj", n, mode, cycle, m, r, k)
        traj = noise_mod.sample_trajectory(circ, noise, random.Random(ts))
        bank.add(g, k, traj, trajectory_seed=ts,
                 header_lines=common + [f"arm=noisy trajectory={k} seed={ts}"])
    bank.close(g)


def build_calibration(bank, n):
    """The index-convention probe. See `calibration_circuit`."""
    circ, thetas = calibration_circuit(n)
    g = bank.group(f"cal_n{n:02d}_probe", f"n{n:02d}/cal", kind="cal", arm="ideal",
                   n_qubits=n, estimator="topk", topk=n + 2,
                   thetas=thetas, p_one=[math.sin(t / 2) ** 2 for t in thetas])
    bank.add(g, 0, circ, header_lines=[
        f"CALIBRATION | n={n} -- product state, ry(theta_j) on qubit j",
        "p(qubit j = 1) is distinct and increasing in j, and every single-excitation "
        "probability exceeds every double-excitation one",
        "the top-k indices therefore say whether qubit j is bit j or bit n-1-j"])
    bank.close(g)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--widths", nargs="+", type=int, default=WIDTHS)
    ap.add_argument("--depths", nargs="+", type=int, default=DEPTHS)
    ap.add_argument("--cb-depths", nargs="+", type=int, default=CB_DEPTHS)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--instances", type=int, default=N_INSTANCES,
                    help=f"random circuit instances per depth (run_bounding uses 80; "
                         f"default {N_INSTANCES}, the smoke value)")
    ap.add_argument("--trajectories", type=int, default=N_TRAJ,
                    help=f"K error realisations per noisy arm. Sets the residual noise "
                         f"floor at ~1/sqrt(K) -- the only one left once exact "
                         f"probability vectors replace shots (default {N_TRAJ})")
    ap.add_argument("--cb-decays", type=int, default=N_DECAYS,
                    help=f"random Paulis per CB sequence length; cycle_benchmark's "
                         f"n_decays is 30 (default {N_DECAYS})")
    ap.add_argument("--theta-zz", type=float, default=THETA_ZZ,
                    help="coherent residual-ZZ angle in rad. At 0 the channel is purely "
                         "Pauli-stochastic, so the noisy and rc arms MUST agree to "
                         "within 1/sqrt(K) -- that is the pipeline's unbiasedness test")
    ap.add_argument("--oneq-set", default=ONEQ_SET, choices=("clifford", "t"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "qasm_bank_bounding"))
    ap.add_argument("--dry-run", action="store_true",
                    help="report counts and sizes without writing anything")
    ap.add_argument("--force", action="store_true",
                    help="rewrite files that already exist (default: skip)")
    args = ap.parse_args()

    noise = NoiseModel(enabled=True, p1=P1, p2=P2, p_readout=P_READOUT, p_idle=P_IDLE,
                       theta_zz=args.theta_zz)
    oneq = oneq_sets(args.oneq_set)
    bank = Bank(args.out, args.dry_run, args.force)

    print(f"noise: p1={P1} p2={P2} p_idle={P_IDLE} theta_zz={args.theta_zz} | "
          f"p_readout={P_READOUT} applied analytically in post-processing")
    for m in args.modes:
        print(f"1q set ({m:>10}): {oneq[m]}")
    print(f"K={args.trajectories} trajectories, {args.instances} instances, "
          f"{args.cb_decays} CB decays\n")
    print(f"{'n':>3}{'cycles':>8}{'tvd grp':>9}{'cb grp':>8}{'files':>9}")

    for n in args.widths:
        cycles = cycles_for(n)
        if not cycles:
            print(f"{n:>3}   SKIPPED: no non-empty entangling cycle at this width")
            continue
        before_files, before_groups = bank.files, len(bank.groups)
        build_calibration(bank, n)
        n_tvd = n_cb = 0
        for mode in args.modes:
            for depth in args.depths:
                for inst in range(args.instances):
                    build_tvd(bank, n, mode, oneq[mode], depth, inst,
                              list(cycles.values()), noise, args.trajectories,
                              args.seed)
                    n_tvd += 3
            for cycle, pairs in cycles.items():
                for m in args.cb_depths:
                    for r in range(args.cb_decays):
                        build_cb(bank, n, mode, pairs, cycle, m, r, noise,
                                 args.trajectories, args.seed)
                        n_cb += 2
        print(f"{n:>3}{','.join(cycles):>8}{n_tvd:>9}{n_cb:>8}"
              f"{bank.files - before_files:>9}", flush=True)
        assert len(bank.groups) - before_groups == n_tvd + n_cb + 1

    manifest = {
        "generator": "examples/build_bounding_bank.py",
        "mirrors": "examples/run_bounding.py",
        "qasm_version": "2.0",
        "note": "QASM is UNLOWERED proxysim IR output; c3pq_stage.py applies "
                "c3pq.lower_for_c3pq. Readout error is NOT in the circuits.",
        "config": {"widths": args.widths, "depths": args.depths,
                   "cb_depths": args.cb_depths, "modes": args.modes,
                   "instances": args.instances, "trajectories": args.trajectories,
                   "cb_decays": args.cb_decays, "oneq_set": args.oneq_set,
                   "oneq": oneq, "seed": args.seed,
                   "instance_stride": INSTANCE_STRIDE},
        "noise": dataclasses.asdict(noise),
        "cycles": {str(n): {k: [list(p) for p in v]
                            for k, v in cycles_for(n).items()} for n in args.widths},
        "groups": bank.groups,
    }
    bank.json(os.path.join(args.out, "manifest.json"), manifest)

    verb = "would write" if args.dry_run else "bank contains"
    print(f"\n{verb} {bank.files} circuits in {len(bank.groups)} groups, "
          f"{bank.bytes / 1e6:.1f} MB")
    if bank.skipped:
        print(f"  {bank.files - bank.skipped} written this run; {bank.skipped} already "
              f"existed and were left alone (--force to rewrite)")
    if not args.dry_run:
        print(f"  bank at {args.out}, index at {args.out}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
