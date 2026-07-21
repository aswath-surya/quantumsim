"""Sample single-node HPC run: noise-trajectory work fanned across the cores.

Computes the noisy survival probability of a non-Clifford mirror circuit (which
needs Monte-Carlo trajectories on the statevector / MPS backend) both serially and
in parallel. Same seeds are used either way, so the two answers must match exactly
-- that is the correctness check -- while the wall-clock shows the speedup.

This is the pattern you scale on an HPC node: set n_workers = cores,
threads_per_worker = 1, and fan the independent trajectories out with
proxysim.parallel. (For GPU tensor networks use ONE worker with gpu=True instead.)

Run:  python examples/run_parallel_hpc.py
"""

from __future__ import annotations

import os
import time
import warnings

warnings.filterwarnings("ignore")

import _bootstrap  # noqa: F401
from proxysim import NoiseModel, brickwork_magic, save_results, simulate

N = 12
N_CYCLES = 4
N_TRAJ = 2000
NOISE = NoiseModel(enabled=True, p1=5e-4, p2=2e-3, p_readout=2e-3, p_idle=5e-4)
OUT = _bootstrap.RESULTS_DIR + "/parallel_hpc.npz"


def main():
    circ = brickwork_magic(N, N_CYCLES, n_t=8, seed=1).mirror()  # non-Clifford
    print(f"n={N}, non-Clifford mirror, trajectories={N_TRAJ}, cpu_count={os.cpu_count()}\n")
    print(f"{'backend':<16}{'serial (s)':>12}{'parallel (s)':>14}{'speedup':>10}"
          f"{'result match':>14}")

    rows = {}
    for sim in ("statevector", "tensornetwork"):
        t0 = time.perf_counter()
        s_serial = simulate(circ, sim, "survival", noise=NOISE, n_traj=N_TRAJ,
                            seed=1, parallel=False)
        t_serial = time.perf_counter() - t0

        t0 = time.perf_counter()
        s_par = simulate(circ, sim, "survival", noise=NOISE, n_traj=N_TRAJ,
                         seed=1, parallel=True)
        t_par = time.perf_counter() - t0

        rows[sim] = (t_serial, t_par, s_serial, s_par)
        print(f"{sim:<16}{t_serial:>12.2f}{t_par:>14.2f}{t_serial / t_par:>9.2f}x"
              f"{'yes' if abs(s_serial - s_par) < 1e-12 else 'NO':>14}")

    save_results(OUT, **{f"{k}_{n}": v for k, (ts, tp, ss, sp) in rows.items()
                         for n, v in [("t_serial", ts), ("t_parallel", tp),
                                      ("survival", ss)]})
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
