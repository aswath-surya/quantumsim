import inspect
import json
import os
import random
import shutil
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401 -- puts the repo root on sys.path

from examples.c3pq_analyze import Results
from proxysim import rc
from proxysim.benchmarking import cycle_benchmark, qcap_bound, readout_fidelity
from proxysim.c3pq import stage_name
from proxysim.circuit import Circuit
from proxysim.noise import NoiseModel, sample_trajectory
from proxysim.qasm import circuit_to_qasm


class AutomationBank:
    """Create the QASM bank and manifest required by C-3PQ."""

    def __init__(self, root: str):
        self.root = root
        self.groups = []
        self.files_count = 0

    def group(self, name: str, subdir: str, **fields) -> dict:
        group = dict(
            name=stage_name(name),
            dir=os.path.join(subdir, name),
            files=[],
            **fields,
        )
        self.groups.append(group)
        return group

    def add(
        self,
        group: dict,
        k: int,
        circuit: Circuit,
        header_lines=(),
        **fields,
    ) -> None:
        stem = stage_name(f"{group['name']}_k{k:05d}")
        text = circuit_to_qasm(
            circuit,
            measured=None,
            header_lines=header_lines,
        )
        group["files"].append(
            dict(
                file=f"{stem}.qasm",
                stem=stem,
                k=k,
                gates=len(circuit.gates),
                **fields,
            )
        )
        self.files_count += 1

        path = os.path.join(self.root, group["dir"], f"{stem}.qasm")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    def close(self, group: dict) -> None:
        """Assign equal weights to all files in one C-3PQ average group."""
        if not group["files"]:
            return
        weight = 1.0 / len(group["files"])
        for file_entry in group["files"]:
            file_entry["weight"] = weight

    def write_manifest(
        self,
        path: str,
        config: Dict[str, Any],
        noise: NoiseModel,
    ) -> None:
        manifest = {
            "generator": "automate_c3pq_pipeline",
            "qasm_version": "2.0",
            "config": config,
            "noise": noise.__dict__ if hasattr(noise, "__dict__") else {},
            "groups": self.groups,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=1)


def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> str:
    """Run one pipeline command and raise with useful output on failure."""
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        raise RuntimeError(
            f"Command failed with return code {result.returncode}: "
            f"{' '.join(cmd)}"
        )
    return result.stdout


def instance_factory(
    circuit_fn: Callable,
) -> Callable[[int, int], Circuit]:
    """Normalize ``circuit_fn`` to ``f(depth, instance) -> Circuit``.

    A one-argument function is accepted, but then every instance at a given depth is
    the same logical circuit. In that case any point-to-point spread comes only from
    Monte Carlo noise, not from circuit-instance variation.
    """
    n_params = len(inspect.signature(circuit_fn).parameters)
    if n_params == 1:
        return lambda depth, instance: circuit_fn(depth)
    if n_params == 2:
        return circuit_fn
    raise TypeError(
        "circuit_fn must accept either (depth) or (depth, instance)"
    )


def cycle_counts(
    circ: Circuit,
    cycles: Dict[str, List[Tuple[int, int]]],
) -> Dict[str, int]:
    """Count applications of each named entangling layer in ``circ``.

    Each cycle is represented by a set of disjoint two-qubit pairs. Counting tallies
    two-qubit gates on those pairs and divides by the width of the layer.
    """
    owner: Dict[Tuple[int, int], str] = {}
    for name, pairs in cycles.items():
        if not pairs:
            raise ValueError(f"cycle {name!r} has no qubit pairs")
        for a, b in pairs:
            key = (min(a, b), max(a, b))
            if key in owner:
                raise ValueError(
                    f"pair {key} appears in both cycle {owner[key]!r} "
                    f"and cycle {name!r}; cycles must have disjoint pair sets"
                )
            owner[key] = name

    tally = {name: 0 for name in cycles}
    for gate in circ.gates:
        if len(gate.qubits) != 2:
            continue
        pair = (min(gate.qubits), max(gate.qubits))
        name = owner.get(pair)
        if name is not None:
            tally[name] += 1

    counts = {}
    for name, pairs in cycles.items():
        width = len(pairs)
        if tally[name] % width != 0:
            raise ValueError(
                f"cycle {name!r} has {tally[name]} matching two-qubit gates, "
                f"which is not divisible by its layer width {width}"
            )
        counts[name] = tally[name] // width
    return counts


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
    """Compute one QCAP bound per circuit depth from synthetic CB.

    The bound at a depth is shared by all instances only when those instances have
    the same entangling-cycle census. This condition is checked explicitly.
    """
    cbs = {
        name: cycle_benchmark(
            pairs,
            n_qubits,
            list(cb_depths),
            noise_model,
            mode=cb_mode,
            seed=seed + index,
        )
        for index, (name, pairs) in enumerate(cycles.items())
    }

    ro_fid, ro_std = readout_fidelity(
        n_qubits,
        noise_model,
        seed=seed + 100,
    )

    for name, cb in cbs.items():
        print(
            f"  cycle {name} {str(cycles[name]):<16} "
            f"e_F = {cb['e_F']:.4f} ({cb['e_F_std']:.4f})"
        )
    print(f"  readout fidelity: {ro_fid:.4f} ({ro_std:.4f})")

    efs = {
        name: (cb["e_F"], cb["e_F_std"])
        for name, cb in cbs.items()
    }

    out: Dict[int, Dict[str, float]] = {}
    for depth in depths:
        counts = cycle_counts(make_circuit(depth, 0), cycles)
        for instance in range(1, n_instances):
            other = cycle_counts(make_circuit(depth, instance), cycles)
            if other != counts:
                raise ValueError(
                    f"instances of depth {depth} disagree on the cycle census "
                    f"({counts} vs {other} for instance {instance}); they cannot "
                    "share one QCAP bound"
                )

        bound = qcap_bound(counts, efs, ro_fid, ro_std)
        out[depth] = {
            "bound": float(bound["error"]),
            "bound_std": float(bound["std"]),
            "counts": counts,
        }

    return out


def automate_c3pq_pipeline(
    circuit_fn: Callable,
    depths: List[int],
    noise_model: NoiseModel,
    n_rc_realizations: int = 8,
    n_traj_per_rc: int = 8,
    n_noisy_trajectories: Optional[int] = None,
    n_instances: int = 1,
    cycles: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    cb_depths: Sequence[int] = (1, 2, 4, 8, 16, 24),
    jobs: Optional[int] = None,
    bank_dir: str = "tmp_bank",
    staging_dir: str = "tmp_staging",
    results_dir: str = "tmp_results",
    seed: int = 42,
) -> Dict[int, Optional[Dict[str, Any]]]:
    """Run the complete C-3PQ TVD and QCAP pipeline.

    Averaging hierarchy
    -------------------
    For each logical circuit instance i at a fixed depth:

      1. Generate R independently randomized compilations.
      2. For each fixed randomized compilation, generate M independent stochastic
         noise trajectories.
      3. Average all R*M probability distributions inside that instance's RC group.
      4. Compute one instance-level TVD against that instance's ideal distribution.
      5. Repeat for all logical circuit instances.
      6. Report the mean, standard deviation, and SEM of the instance-level TVDs.

    Different logical instances are never pooled at the probability-vector level,
    because they generally have different ideal target distributions.
    """
    if not depths:
        raise ValueError("depths must contain at least one value")
    if n_instances < 1:
        raise ValueError("n_instances must be at least 1")
    if n_rc_realizations < 1:
        raise ValueError("n_rc_realizations must be at least 1")
    if n_traj_per_rc < 1:
        raise ValueError("n_traj_per_rc must be at least 1")

    total_rc_trajectories = n_rc_realizations * n_traj_per_rc
    if n_noisy_trajectories is None:
        n_noisy_trajectories = total_rc_trajectories
    if n_noisy_trajectories < 1:
        raise ValueError("n_noisy_trajectories must be at least 1")

    # 0. Cleanup.
    for directory in (bank_dir, staging_dir, results_dir):
        if os.path.exists(directory):
            shutil.rmtree(directory)

    # 1. Create QASM bank.
    print("Creating QASM bank...")
    make_circuit = instance_factory(circuit_fn)
    bank = AutomationBank(bank_dir)

    for depth in depths:
        for instance in range(n_instances):
            circ = make_circuit(depth, instance)
            n_qubits = circ.n_qubits
            tag = f"tvd_n{n_qubits:02d}_d{depth:03d}_i{instance:03d}"
            common = dict(
                kind="tvd",
                n_qubits=n_qubits,
                depth=depth,
                instance=instance,
                estimator="probs",
            )

            # Ideal arm: one exact logical target for this instance.
            ideal_group = bank.group(
                f"{tag}_ideal",
                f"n{n_qubits:02d}/tvd",
                arm="ideal",
                **common,
            )
            bank.add(
                ideal_group,
                0,
                circ,
                header_lines=[
                    f"Depth {depth} instance {instance} - Ideal"
                ],
            )
            bank.close(ideal_group)

            # Raw noisy arm: independent stochastic histories of the same circuit.
            noisy_group = bank.group(
                f"{tag}_noisy",
                f"n{n_qubits:02d}/tvd",
                arm="noisy",
                **common,
            )
            for trajectory_index in range(n_noisy_trajectories):
                trajectory_seed = (
                    seed
                    + 1_000_000 * depth
                    + 10_000 * instance
                    + trajectory_index
                )
                trajectory_rng = random.Random(trajectory_seed)
                trajectory = sample_trajectory(
                    circ,
                    noise_model,
                    trajectory_rng,
                )
                bank.add(
                    noisy_group,
                    trajectory_index,
                    trajectory,
                    header_lines=[
                        f"Depth {depth} instance {instance} - "
                        f"Noisy trajectory {trajectory_index}"
                    ],
                    trajectory=trajectory_index,
                    trajectory_seed=trajectory_seed,
                )
            bank.close(noisy_group)

            # RC arm: R independent Pauli dressings, each with M noise trajectories.
            rc_group = bank.group(
                f"{tag}_rc",
                f"n{n_qubits:02d}/tvd",
                arm="rc",
                n_rc_realizations=n_rc_realizations,
                n_traj_per_rc=n_traj_per_rc,
                **common,
            )

            file_index = 0
            for rc_index in range(n_rc_realizations):
                rc_seed = (
                    seed
                    + 100_000_000
                    + 1_000_000 * depth
                    + 10_000 * instance
                    + 100 * rc_index
                )
                twirl_rng = random.Random(rc_seed)
                twirled, virtual_indices = rc.pauli_twirl(
                    circ,
                    twirl_rng,
                    mark_virtual=True,
                )

                for trajectory_index in range(n_traj_per_rc):
                    trajectory_seed = (
                        rc_seed
                        + 10_000_000
                        + trajectory_index
                    )
                    trajectory_rng = random.Random(trajectory_seed)
                    noisy_twirled = sample_trajectory(
                        twirled,
                        noise_model,
                        trajectory_rng,
                        virtual=virtual_indices,
                    )
                    bank.add(
                        rc_group,
                        file_index,
                        noisy_twirled,
                        header_lines=[
                            f"Depth {depth} instance {instance} - "
                            f"RC realization {rc_index}, "
                            f"noise trajectory {trajectory_index}"
                        ],
                        rc_realization=rc_index,
                        trajectory=trajectory_index,
                        rc_seed=rc_seed,
                        trajectory_seed=trajectory_seed,
                    )
                    file_index += 1

            bank.close(rc_group)

    circuits_per_instance = (
        1 + n_noisy_trajectories + total_rc_trajectories
    )
    print(
        f"  {bank.files_count} circuits\n"
        f"    depths:                         {len(depths)}\n"
        f"    instances per depth:            {n_instances}\n"
        f"    noisy trajectories per instance:{n_noisy_trajectories:>6}\n"
        f"    RC realizations per instance:   {n_rc_realizations:>6}\n"
        f"    trajectories per RC:            {n_traj_per_rc:>6}\n"
        f"    total RC trajectories:          {total_rc_trajectories:>6}\n"
        f"    total circuits per instance:    {circuits_per_instance:>6}"
    )

    bank.write_manifest(
        os.path.join(bank_dir, "manifest.json"),
        {
            "depths": depths,
            "n_instances": n_instances,
            "n_rc_realizations": n_rc_realizations,
            "n_traj_per_rc": n_traj_per_rc,
            "n_noisy_trajectories": n_noisy_trajectories,
            "total_rc_trajectories_per_instance": total_rc_trajectories,
            "seed": seed,
        },
        noise_model,
    )

    # 2. Stage.
    print("Staging circuits...")
    run_cmd(
        [
            sys.executable,
            "examples/c3pq_stage.py",
            "stage",
            "--bank",
            bank_dir,
            "--staging",
            staging_dir,
        ]
    )

    # 3. Build.
    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) - 2)
    print(f"Building C++ binaries (-j {jobs})...")
    run_cmd(
        [
            "bash",
            "examples/c3pq_build.sh",
            "--staging",
            staging_dir,
            "-j",
            str(jobs),
        ]
    )

    # 4. Run.
    print("Executing simulations...")
    run_cmd(
        [
            "bash",
            "examples/c3pq_run.sh",
            "--staging",
            staging_dir,
            "--out",
            results_dir,
            "-j",
            str(jobs),
        ]
    )

    # 5. Synthetic CB/QCAP bound.
    bounds: Dict[int, Dict[str, float]] = {}
    if cycles:
        print("Cycle benchmarking (for the QCAP bound)...")
        bounds = qcap_bounds(
            make_circuit,
            depths,
            noise_model,
            cycles,
            make_circuit(depths[0], 0).n_qubits,
            n_instances=n_instances,
            cb_depths=cb_depths,
            seed=seed + 1,
        )

    # 6. Analyze.
    print("Analyzing results...")
    results = Results(staging_dir, results_dir)

    from proxysim.noise import apply_readout_to_distribution

    tvd_results: Dict[int, Optional[Dict[str, Any]]] = {}

    for depth in depths:
        n_qubits = make_circuit(depth, 0).n_qubits
        noisy_points: List[float] = []
        rc_points: List[float] = []

        for instance in range(n_instances):
            tag = f"tvd_n{n_qubits:02d}_d{depth:03d}_i{instance:03d}"
            try:
                # Results.probs returns the weighted group average. Thus the RC vector
                # is already the average over all R*M files for this one instance.
                ideal_vec = results.probs(f"{tag}_ideal", n_qubits)
                noisy_vec_raw = results.probs(f"{tag}_noisy", n_qubits)
                rc_vec_raw = results.probs(f"{tag}_rc", n_qubits)

                noisy_vec = apply_readout_to_distribution(
                    noisy_vec_raw,
                    noise_model,
                    n_qubits,
                )
                rc_vec = apply_readout_to_distribution(
                    rc_vec_raw,
                    noise_model,
                    n_qubits,
                )

                noisy_tvd = 0.5 * float(
                    np.abs(ideal_vec - noisy_vec).sum()
                )
                rc_tvd = 0.5 * float(
                    np.abs(ideal_vec - rc_vec).sum()
                )

                noisy_points.append(noisy_tvd)
                rc_points.append(rc_tvd)

            except Exception as exc:
                print(
                    f"Error analyzing depth {depth} instance {instance}: {exc}"
                )

        if not rc_points:
            tvd_results[depth] = None
            continue

        noisy_array = np.asarray(noisy_points, dtype=float)
        rc_array = np.asarray(rc_points, dtype=float)

        noisy_std = (
            float(noisy_array.std(ddof=1))
            if noisy_array.size > 1
            else 0.0
        )
        rc_std = (
            float(rc_array.std(ddof=1))
            if rc_array.size > 1
            else 0.0
        )

        noisy_sem = noisy_std / np.sqrt(noisy_array.size)
        rc_sem = rc_std / np.sqrt(rc_array.size)

        bound_entry = bounds.get(depth)
        entry: Dict[str, Any] = {
            "tvd_noisy_points": noisy_array.tolist(),
            "tvd_rc_points": rc_array.tolist(),
            "mean_instance_tvd_noisy": float(noisy_array.mean()),
            "mean_instance_tvd_rc": float(rc_array.mean()),
            "std_instance_tvd_noisy": noisy_std,
            "std_instance_tvd_rc": rc_std,
            "sem_instance_tvd_noisy": float(noisy_sem),
            "sem_instance_tvd_rc": float(rc_sem),
            # Backward-compatible names used by the plotting function.
            "tvd_noisy": float(noisy_array.mean()),
            "tvd_rc": float(rc_array.mean()),
            "n_successful_instances": int(rc_array.size),
            "n_rc_realizations": n_rc_realizations,
            "n_traj_per_rc": n_traj_per_rc,
            "total_rc_trajectories_per_instance": total_rc_trajectories,
            **{
                key: value
                for key, value in (bound_entry or {}).items()
                if key != "counts"
            },
        }
        tvd_results[depth] = entry

        extra = ""
        if bound_entry is not None:
            n_over = int(np.sum(rc_array > bound_entry["bound"]))
            extra = f", bound={bound_entry['bound']:.6f}"
            if n_over:
                extra += (
                    f"   <-- {n_over}/{len(rc_array)} RC instance TVDs "
                    "over the bound"
                )

        print(
            f"Depth {depth}: "
            f"mean TVD_Noisy={entry['mean_instance_tvd_noisy']:.6f}, "
            f"mean TVD_RC={entry['mean_instance_tvd_rc']:.6f}, "
            f"RC s.d.={entry['std_instance_tvd_rc']:.6f}, "
            f"RC SEM={entry['sem_instance_tvd_rc']:.6f} "
            f"(max {rc_array.max():.6f} over {len(rc_array)} instances)"
            f"{extra}"
        )

    return tvd_results


def plot_tvd_results(
    tvd_results: Dict[int, Optional[Dict[str, Any]]],
    filename: str = "tvd_vs_depth.png",
    title: str = "TVD vs Depth",
    total_rc_trajectories: Optional[int] = None,
) -> None:
    """Plot one RC TVD marker per logical circuit instance and the mean trend."""
    depths = sorted(
        depth
        for depth, result in tvd_results.items()
        if result is not None
    )
    if not depths:
        print("No usable results to plot.")
        return

    have_bound = all(
        "bound" in tvd_results[depth]
        for depth in depths
    )

    plt.figure(figsize=(8, 5))

    if have_bound:
        bound = np.asarray(
            [tvd_results[depth]["bound"] for depth in depths],
            dtype=float,
        )
        bound_std = np.asarray(
            [tvd_results[depth]["bound_std"] for depth in depths],
            dtype=float,
        )
        plt.plot(
            depths,
            bound,
            "-",
            color="#D55E00",
            lw=2,
            zorder=3,
            label="QCAP bound (on the RC'd circuit)",
        )
        plt.fill_between(
            depths,
            np.clip(bound - 2.96 * bound_std, 0.0, 1.0),
            np.clip(bound + 2.96 * bound_std, 0.0, 1.0),
            color="#D55E00",
            alpha=0.2,
            zorder=2,
            label="QCAP fit uncertainty",
        )

    # Every marker is one separately generated logical circuit instance. Its RC
    # probability vector has already been averaged over all RC/noise samples.
    for index, depth in enumerate(depths):
        points = tvd_results[depth]["tvd_rc_points"]
        plt.plot(
            [depth] * len(points),
            points,
            "s",
            color="#0072B2",
            ms=5,
            alpha=0.55,
            mew=0,
            label=(
                "instance-level TVD, randomly compiled"
                if index == 0
                else None
            ),
        )

    plt.plot(
        depths,
        [tvd_results[depth]["mean_instance_tvd_rc"] for depth in depths],
        "-",
        color="#0072B2",
        lw=1.4,
        alpha=0.95,
        label="mean across circuit instances",
    )

    if total_rc_trajectories:
        plt.axhline(
            1.0 / total_rc_trajectories,
            color="0.4",
            ls="--",
            lw=1,
            zorder=1,
            label=(
                "single-trajectory mixture weight "
                f"$1/{total_rc_trajectories}$"
            ),
        )

    plt.xlabel("circuit depth")
    plt.ylabel("total-variation distance from ideal")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.15)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Plot saved to {filename}")
    plt.close()


if __name__ == "__main__":
    from proxysim.circuit import GATE_SETS, bench_brickwork

    # One two-qubit entangling cycle on qubits (0, 1).
    CYCLES = {"A": [(0, 1)]}

    # Independent logical circuits at every depth. These determine the spread of TVD.
    N_INSTANCES = 16

    # Independent randomized compilations of each fixed logical circuit.
    N_RC_REALIZATIONS = 8

    # Independent stochastic physical-noise histories for each fixed RC realization.
    N_TRAJ_PER_RC = 8

    # Give the raw noisy arm the same total Monte Carlo budget as the RC arm.
    N_NOISY_TRAJECTORIES = N_RC_REALIZATIONS * N_TRAJ_PER_RC

    def my_circuit_fn(depth: int, instance: int) -> Circuit:
        """Two-qubit Clifford+T brickwork circuit with an instance-dependent seed."""
        oneq = list(GATE_SETS["clifford"]) + ["t"]
        return bench_brickwork(
            2,
            depth,
            list(CYCLES.values()),
            oneq=oneq,
            twoq="cz",
            seed=42 + 1000 * instance,
        )

    noise = NoiseModel(
        enabled=True,
        p1=1e-3,
        p2=1e-2,
        p_readout=1e-2,
        p_idle=1e-3,
    )

    depths_to_test = [1, 2, 4, 8, 16]

    results = automate_c3pq_pipeline(
        my_circuit_fn,
        depths_to_test,
        noise,
        n_rc_realizations=N_RC_REALIZATIONS,
        n_traj_per_rc=N_TRAJ_PER_RC,
        n_noisy_trajectories=N_NOISY_TRAJECTORIES,
        n_instances=N_INSTANCES,
        cycles=CYCLES,
    )

    print("\nFinal Results:", results)

    total_rc_trajectories = N_RC_REALIZATIONS * N_TRAJ_PER_RC
    plot_tvd_results(
        results,
        filename="tvd_vs_depth.png",
        total_rc_trajectories=total_rc_trajectories,
        title=(
            "TVD vs depth "
            f"(n=2, CZ brickwork, {N_INSTANCES} instances, "
            f"{N_RC_REALIZATIONS} RC realizations x "
            f"{N_TRAJ_PER_RC} trajectories)"
        ),
    )
