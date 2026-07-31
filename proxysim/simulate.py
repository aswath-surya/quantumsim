"""One intuitive entry point: ``simulate(circuit, simulator=..., output=...)``.

Everything the package can do is reachable from here with a readable if-else over
two axes:

  simulator :  'statevector' | 'tensornetwork' | 'stabilizer' | 'pauliprop' | 'auto'
  output    :  'distribution' | 'samples' | 'survival' | 'expectation'

plus flags: ``noise`` (a NoiseModel), ``gpu`` (tensor-network on cupy),
``max_bond`` (MPS truncation), and ``parallel``/``n_workers`` (fan trajectories or
shots across a node). 'auto' picks the stabilizer backend for Clifford circuits and
the tensor-network backend otherwise.

Noisy outputs: a Clifford circuit is sampled natively by stim (exact, scalable);
a non-Clifford circuit is handled by Monte-Carlo trajectories on the statevector /
MPS backend, which parallelise across a node.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Optional

import numpy as np

from . import metrics  # noqa: F401  (kept for convenience re-export)
from .backends import StabilizerBackend, StatevectorBackend, TensorNetworkBackend
from .noise import apply_readout_to_distribution, sample_trajectory, to_stim_noisy

_SV_ALIASES = {"statevector", "sv", "qiskit"}
_TN_ALIASES = {"tensornetwork", "tn", "mps", "quimb"}
_STAB_ALIASES = {"stabilizer", "stim", "clifford"}


def auto_simulator(circuit) -> str:
    return "stabilizer" if circuit.is_clifford else "tensornetwork"


def noisy_tvd_vs_support(circuit, noise, ideal_support, shots, seed):
    """TVD of the stim-sampled noisy Clifford circuit vs a known uniform ideal
    support (the 'coset trick': works at 20 qubits because we only need the sampled
    counts on the ideal outcomes plus the leftover mass). ``ideal_support`` is an
    iterable of integer codes (bit j of the code is qubit j, q0 = MSB)."""
    arr = to_stim_noisy(circuit, noise).compile_sampler(seed=seed).sample(shots)
    n = circuit.n_qubits
    weights = (1 << np.arange(n - 1, -1, -1)).astype(np.int64)   # q0 = MSB
    codes = arr.astype(np.int64) @ weights
    vals, cnts = np.unique(codes, return_counts=True)
    counts = dict(zip(vals.tolist(), cnts.tolist()))
    return metrics.tvd_to_ideal_support(counts, shots, list(ideal_support))


def make_backend(simulator: str, max_bond: Optional[int] = None, gpu: bool = False):
    """Return a backend instance for a simulator name (with GPU / bond options)."""
    s = simulator.lower()
    if s in _SV_ALIASES:
        return StatevectorBackend()
    if s in _TN_ALIASES:
        return TensorNetworkBackend(max_bond=max_bond, gpu=gpu)
    if s in _STAB_ALIASES:
        return StabilizerBackend()
    raise ValueError(f"unknown simulator '{simulator}'")


# ---------------------------------------------------------------------------
# Top-level trajectory workers (must be importable so process workers can pickle).
# ---------------------------------------------------------------------------
def _sv_prob0_worker(args):
    circuit, noise, seed = args
    return StatevectorBackend().prob0(sample_trajectory(circuit, noise, random.Random(seed)))


def _tn_prob0_worker(args):
    circuit, noise, max_bond, gpu, seed = args
    tn = TensorNetworkBackend(max_bond=max_bond, gpu=gpu)
    return tn.prob0(sample_trajectory(circuit, noise, random.Random(seed)))


# ---------------------------------------------------------------------------
# Noisy survival probability P(0...0) -- the mirror-circuit error metric.
# ---------------------------------------------------------------------------
def noisy_survival(circuit, noise, simulator="auto", n_traj=2000, seed=0,
                   max_bond=None, gpu=False, parallel=False, n_workers=None):
    """Mean P(0...0) of the noisy circuit. Stim-native for Clifford circuits, else
    Monte-Carlo trajectories on statevector / MPS (readout handled by the caller)."""
    sim = auto_simulator(circuit) if simulator == "auto" else simulator.lower()
    noise = dataclasses.replace(noise, p_readout=0.0)  # readout folded in via ro_fid

    if sim in _STAB_ALIASES and circuit.is_clifford:
        arr = to_stim_noisy(circuit, noise).compile_sampler(seed=seed).sample(n_traj)
        return float(np.mean(np.all(arr == 0, axis=1)))

    if sim in _SV_ALIASES:
        worker, make_args = _sv_prob0_worker, lambda s: (circuit, noise, s)
    elif sim in _TN_ALIASES:
        worker, make_args = _tn_prob0_worker, lambda s: (circuit, noise, max_bond, gpu, s)
    else:
        raise ValueError(f"noisy_survival: simulator '{simulator}' can't do this circuit")

    arglist = [make_args(seed + i + 1) for i in range(n_traj)]
    if parallel:
        from .parallel import pmap
        vals = pmap(worker, arglist, n_workers=n_workers, executor="process")
    else:
        vals = [worker(a) for a in arglist]
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Trajectory-averaged noisy distribution (non-Clifford and/or coherent noise).
# ---------------------------------------------------------------------------
def _trajectory_distribution(circuit, noise, backend, n_traj, shots, seed):
    """Mean Z-basis distribution over ``n_traj`` noisy trajectories.

    Each trajectory is a pure state, so its exact distribution is read off directly
    and averaged -- far lower variance than sampling shots per trajectory. Readout
    error is then applied analytically, and the result is multinomial-resampled to
    ``shots`` (when ``shots > 0``) so the shot-noise semantics match the stim path.
    """
    n = circuit.n_qubits
    acc = np.zeros(2 ** n)
    rng = random.Random(seed)
    for _ in range(n_traj):
        traj = sample_trajectory(circuit, noise, rng)
        for key, p in backend.exact_distribution(traj).items():
            acc[sum(int(b) << j for j, b in enumerate(key))] += p   # q0 leftmost
    acc /= n_traj
    acc = apply_readout_to_distribution(acc, noise, n)

    if shots and shots > 0:
        acc = np.random.default_rng(seed).multinomial(shots, acc / acc.sum()) / shots
    return {"".join(str((i >> j) & 1) for j in range(n)): float(acc[i])
            for i in np.nonzero(acc > 0)[0]}


# ---------------------------------------------------------------------------
# The unified dispatcher.
# ---------------------------------------------------------------------------
def simulate(circuit, simulator="auto", output="distribution", noise=None,
             shots=0, seed=1234, max_bond=None, gpu=False, observable=None,
             n_traj=2000, parallel=False, n_workers=None):
    """Dispatch by (simulator, output). See module docstring for the axes.

    ``shots`` for a NOISY distribution: the trajectory path treats ``shots=0`` as
    "return the exact trajectory-averaged distribution, no shot noise"; the stim path
    can only sample and raises instead of returning a one-shot histogram.
    """
    sim = auto_simulator(circuit) if simulator == "auto" else simulator.lower()

    # -- expectation values go through Pauli propagation, a separate modality --
    if output == "expectation" or sim == "pauliprop":
        from .pauliprop_validator import PauliPropagator
        if observable is None:
            observable = "Z" + "_" * (circuit.n_qubits - 1)
        return PauliPropagator(max_weight=max_bond).expectation(circuit, observable)

    backend = make_backend(sim, max_bond=max_bond, gpu=gpu)

    if output == "distribution":
        if noise is None:
            return backend.exact_distribution(circuit)
        # Noisy distribution. Stim samples it natively for a Clifford circuit under
        # Pauli noise -- exact and much faster, so it stays the preferred path. Note
        # this branches on the CIRCUIT before the requested simulator, so an explicit
        # simulator='statevector' is ignored for Clifford circuits; that predates
        # coherent noise and is left as-is. A coherent model is non-Clifford by
        # construction and always falls through to trajectories.
        if circuit.is_clifford and not noise.has_coherent:
            if shots <= 0:
                # This path can only SAMPLE, so shots=0 would silently return a
                # one-shot histogram (a delta) rather than the exact distribution the
                # trajectory path below returns for shots=0. Refuse rather than
                # quietly hand back something that looks like a distribution.
                raise ValueError(
                    "simulate(output='distribution', noise=...): the stim path samples, "
                    "so it needs shots > 0. (shots=0 means 'exact' only on the "
                    "trajectory path, which stim-representable circuits do not take.)")
            arr = to_stim_noisy(circuit, noise).compile_sampler(seed=seed).sample(shots)
            counts = {}
            for row in arr:
                k = "".join("1" if b else "0" for b in row)
                counts[k] = counts.get(k, 0) + 1
            tot = sum(counts.values()) or 1
            return {k: v / tot for k, v in counts.items()}
        return _trajectory_distribution(circuit, noise, backend, n_traj=n_traj,
                                        shots=shots, seed=seed)

    if output == "samples":
        return backend.run(circuit, shots=shots, seed=seed).counts

    if output == "survival":
        if noise is None:
            return backend.prob0(circuit)
        return noisy_survival(circuit, noise, simulator=sim, n_traj=n_traj, seed=seed,
                              max_bond=max_bond, gpu=gpu, parallel=parallel,
                              n_workers=n_workers)

    raise ValueError(f"unknown output '{output}'")
