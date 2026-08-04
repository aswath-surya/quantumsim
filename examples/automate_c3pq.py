import os
import sys
import json
import random
import inspect
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any, Optional, Sequence, Tuple

import _bootstrap  # noqa: F401  -- puts the repo root on sys.path

from proxysim.circuit import Circuit
from proxysim.qasm import circuit_to_qasm
from proxysim.noise import NoiseModel, sample_trajectory
from proxysim import rc
from proxysim.c3pq import stage_name
from proxysim.benchmarking import cycle_benchmark, qcap_bound, readout_fidelity

# Import analysis tools from the existing pipeline
from examples.c3pq_analyze import Results

class AutomationBank:
    """Helper to create the QASM bank and manifest required by C-3PQ."""
    def __init__(self, root: str, force: bool = False):
        self.root = root
        self.force = force
        self.groups = []
        self.files_count = 0
        self.written_count = 0
        self.skipped_count = 0

    def group(self, name: str, subdir: str, **fields) -> dict:
        g = dict(name=stage_name(name), dir=os.path.join(subdir, name),
                 files=[], **fields)
        self.groups.append(g)
        return g

    def add(self, g: dict, k: int, circuit, header_lines=(), **fields) -> None:
        stem = stage_name(f"{g['name']}_k{k:03d}")
        text = circuit_to_qasm(circuit, measured=None, header_lines=header_lines)
        g["files"].append(dict(file=f"{stem}.qasm", stem=stem, k=k, 
                                gates=len(circuit.gates), **fields))
        self.files_count += 1
        path = os.path.join(self.root, g["dir"], f"{stem}.qasm")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not self.force:
            self.skipped_count += 1
            return
        with open(path, "w") as fh:
            fh.write(text)
        self.written_count += 1

    def close(self, g: dict) -> None:
        if not g["files"]:
            return
        w = 1.0 / len(g["files"])
        for f in g["files"]:
            f["weight"] = w

    def write_manifest(self, path: str, config: Dict[str, Any], noise: NoiseModel):
        manifest = {
            "generator": "automate_c3pq_pipeline",
            "qasm_version": "2.0",
            "config": config,
            "noise": noise.__dict__ if hasattr(noise, '__dict__') else {},
            "groups": self.groups,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=1)

def run_cmd(cmd: List[str], cwd: str = None):
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {result.stderr}")
        raise RuntimeError(f"Command failed with return code {result.returncode}: {result.stderr}")
    return result.stdout

def instance_factory(circuit_fn: Callable) -> Callable[[int, int], Circuit]:
    """Normalise ``circuit_fn`` to ``f(depth, instance) -> Circuit``.

    A one-argument ``circuit_fn(depth)`` is still accepted -- it just returns the same
    circuit for every instance, which makes ``n_instances > 1`` pointless (the spread
    would be trajectory noise alone). Two-argument functions get the instance index and
    are expected to reseed on it, giving one random circuit per point the way
    examples/run_bounding.py does.
    """
    n_params = len(inspect.signature(circuit_fn).parameters)
    if n_params == 1:
        return lambda depth, instance: circuit_fn(depth)
    return circuit_fn


def cycle_counts(circ: Circuit, cycles: Dict[str, List[Tuple[int, int]]]) -> Dict[str, int]:
    """How many times each named cycle is applied in ``circ``.

    A 'cycle' is an entangling layer, named and given as its list of qubit pairs -- the
    object cycle benchmarking characterises. Counting is done by tallying the two-qubit
    gates that land on the cycle's pairs and dividing by the layer width, so it reads
    the count off the circuit instead of assuming ``count == depth``. That assumes the
    cycles' pair sets are disjoint; overlapping cycles cannot be told apart from the
    gate list alone.
    """
    owner = {}
    for name, pairs in cycles.items():
        for a, b in pairs:
            key = (min(a, b), max(a, b))
            if key in owner:
                raise ValueError(f"pair {key} appears in both cycle {owner[key]!r} and "
                                 f"cycle {name!r}; cycles must have disjoint pairs")
            owner[key] = name

    tally = {name: 0 for name in cycles}
    for g in circ.gates:
        if len(g.qubits) != 2:
            continue
        name = owner.get((min(g.qubits), max(g.qubits)))
        if name is not None:
            tally[name] += 1
    return {name: tally[name] // len(pairs) for name, pairs in cycles.items()}


def qcap_bounds(
    make_circuit: Callable[[int, int], Circuit],
    depths: List[int],
    noise_model: NoiseModel,
    cycles: Dict[str, List[Tuple[int, int]]],
    n_qubits: int,
    n_instances: int = 1,
    cb_depths: Sequence[int] = (1, 2, 4, 8, 16, 24),
    cb_mode: str = "random",
    seed: int = 1,
) -> Dict[int, Dict[str, float]]:
    """QCAP upper bound on the circuit error at each depth, from synthetic CB.

    Cycle benchmarking gives a process infidelity e_F per cycle and readout gives a
    fidelity F_RO; the bound is 1 - F_RO * prod_c (1 - e_F,c)^(n_c) with n_c the number
    of applications of cycle c in the circuit at that depth. CB runs in stim and so
    measures the Pauli-TWIRLED channel -- the bound therefore applies to the randomly
    compiled arm. (With a purely stochastic noise model the raw arm sees the same
    channel, so it applies there too.)

    One bound per depth, not per instance: the instances of a depth differ only in
    their single-qubit layers, so they share a cycle census and therefore a bound.
    That is checked rather than assumed.
    """
    cbs = {name: cycle_benchmark(pairs, n_qubits, list(cb_depths), noise_model,
                                 mode=cb_mode, seed=seed + j)
           for j, (name, pairs) in enumerate(cycles.items())}
    ro_fid, ro_std = readout_fidelity(n_qubits, noise_model, seed=seed + 100)
    for name, cb in cbs.items():
        print(f"  cycle {name} {str(cycles[name]):<16} e_F = {cb['e_F']:.4f} "
              f"({cb['e_F_std']:.4f})")
    print(f"  readout fidelity: {ro_fid:.4f} ({ro_std:.4f})")

    efs = {name: (cb["e_F"], cb["e_F_std"]) for name, cb in cbs.items()}
    out = {}
    for depth in depths:
        counts = cycle_counts(make_circuit(depth, 0), cycles)
        for i in range(1, n_instances):
            other = cycle_counts(make_circuit(depth, i), cycles)
            if other != counts:
                raise ValueError(
                    f"instances of depth {depth} disagree on the cycle census "
                    f"({counts} vs {other} for instance {i}); they would not share "
                    "a bound")
        b = qcap_bound(counts, efs, ro_fid, ro_std)
        out[depth] = {"bound": b["error"], "bound_std": b["std"], "counts": counts}
    return out


def automate_c3pq_pipeline(
    circuit_fn: Callable[[int], Circuit],
    depths: List[int],
    noise_model: NoiseModel,
    n_trajectories: int = 4,
    n_instances: int = 1,
    cycles: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    cb_depths: Sequence[int] = (1, 2, 4, 8, 16, 24),
    jobs: Optional[int] = None,
    bank_dir: str = "tmp_bank",
    staging_dir: str = "tmp_staging",
    results_dir: str = "tmp_results",
    seed: int = 42,
    clean: bool = True,
    force_bank: bool = False,
):
    """
    Drives a circuit through the full C3PQ pipeline across multiple depths.

    Args:
        circuit_fn: ``f(depth)`` or ``f(depth, instance)`` -> Circuit. The
            two-argument form is what makes ``n_instances`` meaningful.
        depths: List of depths to simulate.
        noise_model: The noise model to apply.
        n_trajectories: Number of noise trajectories in each RC group.
        n_instances: Random circuit instances per depth. Each becomes its own
            TVD point, so the output is a distribution per depth rather than a
            single number. Cost is linear in instances x trajectories.
        cycles: Named entangling layers of the circuit family, e.g.
            ``{"A": [(0, 1)]}``. When given, a QCAP bound is computed per depth
            from synthetic cycle benchmarking and returned alongside the TVDs.
        cb_depths: Sequence lengths for the cycle-benchmarking decay fits.
        jobs: Parallel codegen/compile/run jobs (-j). Defaults to cores - 2.
            Codegen and compile dominate the wall time, and both parallelise
            across batches, so this is worth roughly its value in speedup.
        bank_dir: Directory to store generated QASM files.
        staging_dir: Directory for C-3PQ build artifacts.
        results_dir: Directory for simulation outputs.
        seed: Random seed for reproducibility.
    """
    # 0. Cleanup. Keep this explicit so stale staging/results cannot be mixed into
    # a new manifest. Set clean=False only when intentionally resuming the same run.
    if clean:
        for d in (bank_dir, staging_dir, results_dir):
            if os.path.exists(d):
                shutil.rmtree(d)
    for d in (bank_dir, staging_dir, results_dir):
        os.makedirs(d, exist_ok=True)

    # 1. Create Bank
    print("Creating compact RC-only QASM bank...")
    make_circuit = instance_factory(circuit_fn)
    bank = AutomationBank(bank_dir, force=force_bank)

    # Every (depth, instance) is its own group pair -- ideal / rc -- because
    # a group is what C-3PQ averages over, and averaging two instances together would
    # collapse the very spread the scatter is there to show.
    for depth in depths:
        for inst in range(n_instances):
            circ = make_circuit(depth, inst)
            n_qubits = circ.n_qubits
            tag = f"tvd_n{n_qubits:02d}_d{depth:03d}_i{inst:03d}"
            common = dict(kind="tvd", n_qubits=n_qubits, depth=depth,
                          instance=inst, estimator="probs")

            # Ideal arm
            g_ideal = bank.group(f"{tag}_ideal", f"n{n_qubits:02d}/tvd",
                                 arm="ideal", **common)
            bank.add(g_ideal, 0, circ,
                     header_lines=[f"Depth {depth} instance {inst} - Ideal"])
            bank.close(g_ideal)

            # RC arm only. The QCAP bound applies to this arm, and removing the unused
            # raw-noisy arm nearly halves code generation and compilation.
            g_rc = bank.group(f"{tag}_rc", f"n{n_qubits:02d}/tvd", arm="rc", **common)
            for k in range(n_trajectories):
                # Include depth in the seed so different depths do not reuse identical
                # twirls/error draws.
                rng = random.Random(seed + 1_000_000 * depth + 10_000 * inst + k)
                twirled, virt = rc.pauli_twirl(circ, rng, mark_virtual=True)
                traj = sample_trajectory(twirled, noise_model, rng, virtual=virt)
                bank.add(
                    g_rc,
                    k,
                    traj,
                    header_lines=[
                        f"Depth {depth} instance {inst} - RC trajectory {k}"
                    ],
                )
            bank.close(g_rc)

    expected = len(depths) * n_instances * (1 + n_trajectories)
    print(f"  {bank.files_count} circuits "
          f"({len(depths)} depths x {n_instances} instances x "
          f"(1 ideal + {n_trajectories} RC trajectories))")
    if bank.files_count != expected:
        raise RuntimeError(f"Expected {expected} bank entries, got {bank.files_count}")
    print(f"  written={bank.written_count}, reused={bank.skipped_count}")
    bank.write_manifest(
        os.path.join(bank_dir, "manifest.json"),
        {"depths": depths, "n_trajectories": n_trajectories,
         "n_instances": n_instances, "seed": seed, "arms": ["ideal", "rc"]},
        noise_model
    )

    # 2. Staging
    print("Staging circuits...")
    run_cmd([sys.executable, "examples/c3pq_stage.py", "stage",
             "--bank", bank_dir, "--staging", staging_dir])

    # 3. Build
    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) - 2)
    print(f"Building C++ binaries (-j {jobs})...")
    run_cmd(["bash", "examples/c3pq_build.sh", "--staging", staging_dir,
             "-j", str(jobs)])

    # 4. Run
    print("Executing simulations...")
    run_cmd(["bash", "examples/c3pq_run.sh", "--staging", staging_dir,
             "--out", results_dir, "-j", str(jobs)])

    # 5. Bound. Independent of the C-3PQ run: it comes from synthetic CB on the same
    # noise model, which is exactly the point -- it is the prediction the measured
    # TVDs are being checked against.
    bounds = {}
    if cycles:
        print("Cycle benchmarking (for the QCAP bound)...")
        bounds = qcap_bounds(make_circuit, depths, noise_model, cycles,
                             make_circuit(depths[0], 0).n_qubits,
                             n_instances=n_instances, cb_depths=cb_depths,
                             seed=seed + 1)

    # 6. Analyze
    print("Analyzing results...")
    res = Results(staging_dir, results_dir)

    from proxysim.noise import apply_readout_to_distribution

    tvd_results = {}
    for depth in depths:
        n = make_circuit(depth, 0).n_qubits
        rc_points = []
        for inst in range(n_instances):
            tag = f"tvd_n{n:02d}_d{depth:03d}_i{inst:03d}"
            try:
                # Raw probability vectors, straight from the C-3PQ output.
                ideal_vec = res.probs(f"{tag}_ideal", n)
                rc_vec_raw = res.probs(f"{tag}_rc", n)

                # Readout is applied analytically to the RC distribution; the C-3PQ
                # output is the pre-measurement probability vector.
                rc_vec = apply_readout_to_distribution(rc_vec_raw, noise_model, n)
                rc_points.append(0.5 * float(np.abs(ideal_vec - rc_vec).sum()))
            except Exception as e:
                print(f"Error analyzing depth {depth} instance {inst}: {e}")

        if not rc_points:
            tvd_results[depth] = None
            continue

        b = bounds.get(depth)
        entry = {
            "tvd_rc_points": rc_points,
            "tvd_rc": float(np.mean(rc_points)),
            **{k: v for k, v in (b or {}).items() if k != "counts"},
        }
        tvd_results[depth] = entry

        extra = ""
        if b is not None:
            over = int(np.sum(np.array(rc_points) > b["bound"]))
            extra = (f", bound={b['bound']:.6f}"
                     + (f"   <-- {over}/{len(rc_points)} RC points over the bound"
                        if over else ""))
        print(f"Depth {depth}: mean TVD_RC={entry['tvd_rc']:.6f} "
              f"(max {max(rc_points):.6f} over {len(rc_points)} instances){extra}")

    return tvd_results

def plot_tvd_results(tvd_results: Dict[int, Dict[str, float]],
                     filename: str = "tvd_vs_depth.png", title: str = "TVD vs Depth",
                     n_trajectories: Optional[int] = None):
    """Plot the measured TVDs against the QCAP bound: one marker per circuit instance.

    The bound is drawn as a line with a +/- 2.96 sigma band (the CB uncertainty
    propagated through the product), matching examples/run_bounding.py. Points above
    the band are the interesting ones: the bound is meant to hold for the RC arm.
    """
    depths = sorted(d for d in tvd_results if tvd_results[d])
    if not depths:
        print("No usable results to plot.")
        return
    have_bound = all("bound" in tvd_results[d] for d in depths)

    plt.figure(figsize=(8, 5))
    if have_bound:
        b = np.array([tvd_results[d]["bound"] for d in depths])
        bs = np.array([tvd_results[d]["bound_std"] for d in depths])
        plt.plot(depths, b, "-", color="#D55E00", lw=2, zorder=3,
                 label="QCAP bound (on the RC'd circuit)")
        plt.fill_between(depths, b - 2.96 * bs, b + 2.96 * bs, color="#D55E00",
                         alpha=0.2, zorder=2)

    # One marker per circuit instance, all instances of a depth stacked on the same
    # abscissa -- the spread IS the result, so it is drawn rather than summarised. The
    # mean is overlaid as a line only to make the trend readable.
    series = (("tvd_rc_points", "tvd_rc", "s", "#0072B2",
               "measured TVD, randomly compiled"),)

    for points_key, mean_key, marker, color, label in series:
        for d in depths:
            pts = tvd_results[d].get(points_key, [tvd_results[d][mean_key]])
            plt.plot([d] * len(pts), pts, marker, color=color, ms=5, alpha=0.55,
                     mew=0, label=label if d == depths[0] else None)


    # A single error-bearing trajectory out of K shifts the TVD by ~1/K, so that is the
    # quantum of this estimator. Where the bound sits near or below the line, a marker
    # above the bound is Monte-Carlo granularity, not a violation.
    #if n_trajectories:
    #    plt.axhline(1.0 / n_trajectories, color="0.4", ls="--", lw=1, zorder=1,
    #                label=f"one trajectory in {n_trajectories} (TVD granularity)")

    plt.xlabel("circuit depth")
    plt.ylabel("probability of an error (TVD)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.15)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Plot saved to {filename}")
    plt.close()

if __name__ == "__main__":
    from proxysim.circuit import bench_brickwork, GATE_SETS

    # The circuit family and its cycles have to agree: `cycles` names the entangling
    # layers `circuit_fn` builds, and is what CB characterises for the bound.
    CYCLES = {"A": [(0, 1)]}

    # The two runtime knobs, and the trade-off between them. Cost is
    # depths x instances x (1 + 2 x trajectories) circuits, each ~0.5 s of wall time
    # here (codegen + a share of one mpicxx invocation; the simulation itself is
    # nothing at n=2). The settings below are ~1850 circuits, ~15 min at -j 8.
    #
    #  * instances buy SPREAD: every instance is an independent random circuit and its
    #    own marker on the plot. This is the distribution the bound is supposed to sit
    #    above, so it is what makes the figure a test rather than an anecdote.
    #  * trajectories buy PRECISION per marker. At the low end the estimator is
    #    useless -- with 4 trajectories a single error-bearing trajectory moves the TVD
    #    by ~0.25, so markers scatter far above the bound for reasons that have nothing
    #    to do with the bound. 16 is about the floor for a legible cloud.
    #
    # examples/run_bounding.py runs 80 instances x 150 trajectories, which is out of
    # reach here only because every trajectory is a separately compiled C++ binary.
    # C-3PQ compiles every trajectory as a circuit implementation, so use modest K.
    # This configuration creates 6 * 12 * (1 + 24) = 1,800 QASM circuits.
    N_INSTANCES = 6
    N_TRAJECTORIES = 400

    # Expect the TVD cloud to reach down to exactly 0 at some depths. Whenever an
    # instance's ideal output is the uniform distribution, no Pauli channel can move it
    # (they are all unital) and neither can symmetric readout: the errors are still
    # there, they are just invisible to this metric. The bound is on the probability of
    # an error, which is why it is an upper bound and not an estimate.

    def my_circuit_fn(depth, instance):
        # 2-qubit simple brickwork: one entangling cycle, the pair (0, 1). The seed
        # moves with the instance, so each instance is a different random circuit of
        # the same depth and the same cycle census.
        oneq = list(GATE_SETS["clifford"]) + ["t"]
        return bench_brickwork(2, depth, list(CYCLES.values()), oneq=oneq,
                                twoq="cz", seed=42 + 1000 * instance)

    noise = NoiseModel(enabled=True, p1=1e-3, p2=1e-2, p_readout=1e-2, p_idle=1e-3)

    depths_to_test = [1, 2, 4, 8]#, 16, 32]
    results = automate_c3pq_pipeline(
        my_circuit_fn,
        depths_to_test,
        noise,
        n_trajectories=N_TRAJECTORIES,
        n_instances=N_INSTANCES,
        cycles=CYCLES,
        jobs=max(1, (os.cpu_count() or 2) - 2),
        clean=True,
    )
    print("\nFinal Results:", results)
    plot_tvd_results(results, n_trajectories=N_TRAJECTORIES,
                     title=f"TVD vs depth (n=2, CZ brickwork, "
                           f"{N_INSTANCES} instances x "
                           f"{N_TRAJECTORIES} trajectories)")
