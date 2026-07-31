"""Circuit-level noise model with a single on/off toggle.

Pauli-stochastic channels (matching stim's conventions so the stim-native and
trajectory paths agree exactly):

  * 1-qubit depolarizing after every single-qubit gate      (p1)   -> DEPOLARIZE1
  * 2-qubit depolarizing after every two-qubit gate         (p2)   -> DEPOLARIZE2
  * Z dephasing after every single-qubit gate               (p_z1) -> Z_ERROR
  * correlated ZZ-type Pauli error after every 2q gate      (p_zz) -> PAULI_CHANNEL_2
  * idling dephasing (Z) on qubits idle during a layer      (p_idle) -> Z_ERROR
  * measurement bit-flip (readout error)                    (p_readout) -> X_ERROR before M

COHERENT (non-Pauli, unitary) channels -- off by default:

  * systematic rz(theta_1q) over-rotation after every 1q gate  (phase miscalibration)
  * residual cp(theta_zz) coupling after every 2q gate         (always-on ZZ)

Why they are worth having: every Pauli channel above is *unital*, so the uniform
distribution is a fixed point and any circuit whose ideal output is uniform shows
TVD == 0 no matter how strong the noise. Pauli noise also maps stabilizer states to
stabilizer states, so a Clifford circuit's noisy output stays flat on its coset and
the TVD is quantised by the ideal support size. Coherent errors break both: they add
amplitudes rather than probabilities, and how much they accumulate depends on the
particular circuit. See :meth:`NoiseModel.twirled`.

`NoiseModel.enabled` is the toggle: when False, every channel is skipped and the
simulation is ideal.

Two application paths:
  * :func:`to_stim_noisy` -- inserts native stim noise ops (exact, scalable) for
    the stabilizer backend. Clifford circuits and Pauli noise only: it RAISES on a
    coherent model, since stim cannot represent one (use ``.twirled()`` to get the
    exact Pauli channel that randomized compiling would produce).
  * :func:`sample_trajectory` + :func:`apply_readout` -- Monte-Carlo trajectory:
    sample which Pauli errors fire and splice them into the circuit as gates, plus
    the coherent rotations (which are deterministic, not sampled), for pure-state
    backends (statevector / MPS). Averaging trajectories reproduces the mixed-state
    distribution. Coherent terms make the trajectory circuit non-Clifford.

This is the setting where finite-shot sampling matters (a mixed state can't be
read off as a single |<x|psi>|^2), i.e. the PTA regime of Merkel et al.

Fidelity of a noisy vs ideal distribution: classical (Hellinger) fidelity
F = (sum_x sqrt(p q))^2 in [0, 1].
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import dataclass
from typing import Dict

import stim

from .backends.stabilizer import _PARAM, _SIMPLE

# 15 non-identity two-qubit Pauli labels, e.g. "IX", "XZ", ...
_TWOQ_PAULIS = ["".join(p) for p in itertools.product("IXYZ", repeat=2) if "".join(p) != "II"]
_ONEQ_PAULIS = ["x", "y", "z"]


@dataclass
class NoiseModel:
    enabled: bool = False
    p1: float = 1e-3          # 1-qubit depolarizing after each single-qubit gate
    p2: float = 1e-2          # 2-qubit depolarizing after each two-qubit gate
    p_readout: float = 1e-2   # measurement bit-flip probability
    p_idle: float = 1e-3      # idling dephasing (Z) per idle qubit per layer
    # -- Pauli dephasing / crosstalk (0 = off). Also the target of .twirled(). ----
    p_z1: float = 0.0         # Z error after each single-qubit gate
    p_zz: float = 0.0         # each of IZ, ZI, ZZ after each 2q gate, w.p. p_zz EACH
    # -- coherent (unitary, non-Pauli) errors, in radians (0 = off) --------------
    theta_1q: float = 0.0     # systematic rz(theta) after each single-qubit gate
    theta_zz: float = 0.0     # residual cp(theta) coupling after each two-qubit gate

    @property
    def has_coherent(self) -> bool:
        """True if any coherent term is active. Such a model is non-Clifford, so it
        cannot go through stim -- see :func:`to_stim_noisy` and :meth:`twirled`."""
        return bool(self.enabled and (self.theta_1q or self.theta_zz))

    def twirled(self) -> "NoiseModel":
        """The exact Pauli channel this model becomes under randomized compiling.

        Pauli-twirling a coherent rotation projects it onto its Pauli-diagonal part,
        which for these two channels is exact and closed-form:

            rz(theta)  --twirl-->  Z error with probability sin^2(theta/2)
            cp(theta)  --twirl-->  p_IZ = p_ZI = p_ZZ = sin^2(theta/2) / 4

        (Both verified against an explicit density-matrix twirl over the Pauli group.)

        The stochastic channels pass through untouched, and no merging is needed: for
        a Pauli channel E_s and a coherent channel E_c,

            twirl(E_s . E_c) = E_s . twirl(E_c)

        because a Pauli channel commutes with conjugation by a Pauli. So the twirled
        model is just this model with the angles zeroed and the equivalent Pauli rates
        added on top.

        Use this for anything that must run in stim (``cycle_benchmark`` does it
        internally) and as the reference a randomly-compiled circuit should reproduce.

        Caveat on ``theta_1q``. Both identities above are the correct analytic Pauli
        twirl, but :func:`proxysim.rc.pauli_twirl` only wraps TWO-qubit gates, so it
        does not implement the 1q twirl and will NOT tailor ``theta_1q`` into ``p_z1``.
        Measured on a 3-qubit depth-4 brickwork: RC reproduces the twirled model to
        TVD 0.003 for ``theta_zz`` but is 0.44 away for ``theta_1q``. So
        ``theta_zz`` -> RC is validated end-to-end (``examples/run_coherent_rc.py``);
        ``theta_1q`` -> ``.twirled()`` is the theory only, and a circuit run through
        this package's RC still carries its raw 1q over-rotation.
        """
        return dataclasses.replace(
            self,
            p_z1=self.p_z1 + math.sin(self.theta_1q / 2.0) ** 2,
            p_zz=self.p_zz + math.sin(self.theta_zz / 2.0) ** 2 / 4.0,
            theta_1q=0.0,
            theta_zz=0.0,
        )

    def scaled(self, s: float) -> "NoiseModel":
        """Same model with every rate multiplied by ``s`` (and enabled).

        Coherent *angles* scale linearly, which means their contribution to the error
        grows QUADRATICALLY in ``s`` (the twirled rate is sin^2(s*theta/2) ~ s^2
        theta^2/4) while every Pauli rate grows linearly. That difference is the
        physics -- coherent errors accumulate in amplitude, not in probability.
        """
        return dataclasses.replace(
            self, enabled=True,
            p1=self.p1 * s, p2=self.p2 * s, p_readout=self.p_readout * s,
            p_idle=self.p_idle * s, p_z1=self.p_z1 * s, p_zz=self.p_zz * s,
            theta_1q=self.theta_1q * s, theta_zz=self.theta_zz * s,
        )


# ---------------------------------------------------------------------------
# stim-native application (exact, scalable; Clifford circuits only)
# ---------------------------------------------------------------------------
def _emit_stim(c: stim.Circuit, g):
    if g.name in _SIMPLE:
        c.append(_SIMPLE[g.name], list(g.qubits))
        return
    if g.name in _PARAM:
        k = round(g.params[0] / (math.pi / 2)) % 4
        gate = _PARAM[g.name][k]
        if gate is not None:
            c.append(gate, list(g.qubits))
        return
    raise ValueError(f"noise/stim: non-Clifford gate '{g.name}'")


# PAULI_CHANNEL_2 takes the 15 non-identity 2-qubit Paulis in this order (verified
# against stim directly): IX IY IZ XI XX XY XZ YI YX YY YZ ZI ZX ZY ZZ.
_PC2_IZ, _PC2_ZI, _PC2_ZZ = 2, 11, 14


def _zz_channel_args(p_zz: float):
    """PAULI_CHANNEL_2 probability vector for the correlated ZZ-type Pauli channel:
    each of IZ, ZI and ZZ occurs with probability ``p_zz``."""
    args = [0.0] * 15
    args[_PC2_IZ] = args[_PC2_ZI] = args[_PC2_ZZ] = p_zz
    return args


def _reject_coherent(noise: NoiseModel, where: str):
    if noise.has_coherent:
        bad = "theta_1q" if noise.theta_1q else "theta_zz"
        raise ValueError(
            f"{where}: coherent noise ({bad}={getattr(noise, bad)!r}) is not Clifford, "
            f"so stim cannot represent it. Call noise.twirled() for the exact Pauli "
            f"channel randomized compiling produces, or use a pure-state backend "
            f"(statevector / tensornetwork), which applies the rotation directly."
        )


def to_stim_noisy(circuit, noise: NoiseModel) -> stim.Circuit:
    """Build a measured stim circuit with native noise ops inserted.

    Raises on a coherent model -- see :func:`_reject_coherent`.
    """
    _reject_coherent(noise, "to_stim_noisy")
    c = stim.Circuit()
    n = circuit.n_qubits
    for layer in circuit.layers():
        touched = set()
        for g in layer:
            _emit_stim(c, g)
            touched.update(g.qubits)
            if noise.enabled:
                if len(g.qubits) == 2:
                    if noise.p2 > 0:
                        c.append("DEPOLARIZE2", list(g.qubits), noise.p2)
                    if noise.p_zz > 0:
                        c.append("PAULI_CHANNEL_2", list(g.qubits),
                                 _zz_channel_args(noise.p_zz))
                elif len(g.qubits) == 1:
                    if noise.p1 > 0:
                        c.append("DEPOLARIZE1", list(g.qubits), noise.p1)
                    if noise.p_z1 > 0:
                        c.append("Z_ERROR", list(g.qubits), noise.p_z1)
        if noise.enabled and noise.p_idle > 0:
            idle = [q for q in range(n) if q not in touched]
            if idle:
                c.append("Z_ERROR", idle, noise.p_idle)
    if noise.enabled and noise.p_readout > 0:
        c.append("X_ERROR", list(range(n)), noise.p_readout)
    c.append("M", list(range(n)))
    return c


# ---------------------------------------------------------------------------
# Trajectory application (for pure-state backends: statevector / MPS)
# ---------------------------------------------------------------------------
def sample_trajectory(circuit, noise: NoiseModel, rng):
    """Return a NEW Circuit with one sampled Pauli-error realisation spliced in.

    Coherent terms (``theta_1q`` / ``theta_zz``) are deterministic unitaries, not
    sampled: they are appended after every gate of the matching arity on every
    trajectory. They are emitted as ``rz`` / ``cp`` gates with a non-special angle,
    which makes the returned circuit non-Clifford (``Gate.is_clifford`` already keys
    off the angle), so it must be run on a pure-state backend.

    Order within one gate is: gate, then its coherent error, then its stochastic
    Pauli. The order does not matter for a randomly-compiled (twirled) comparison --
    Pauli channels commute with Pauli conjugation, so twirl(E_s . E_c) = E_s .
    twirl(E_c) either way -- but it does slightly affect the raw coherent statistics.

    Readout error is NOT applied here -- apply :func:`apply_readout` to the
    measured bitstring, or :func:`apply_readout_to_distribution` to a distribution.
    """
    from .circuit import Circuit

    nc = Circuit(circuit.n_qubits, name=circuit.name + "_traj")
    n = circuit.n_qubits
    for layer in circuit.layers():
        touched = set()
        for g in layer:
            nc.gates.append(g)
            touched.update(g.qubits)
            if not noise.enabled:
                continue
            if len(g.qubits) == 2:
                if noise.theta_zz:                       # coherent residual ZZ
                    nc.add("cp", *g.qubits, params=(noise.theta_zz,))
                if noise.p2 > 0 and rng.random() < noise.p2:
                    pa, pb = rng.choice(_TWOQ_PAULIS)
                    if pa != "I":
                        nc.add(pa.lower(), g.qubits[0])
                    if pb != "I":
                        nc.add(pb.lower(), g.qubits[1])
                if noise.p_zz > 0:                       # IZ / ZI / ZZ, p_zz each
                    r = rng.random()
                    if r < 3 * noise.p_zz:
                        which = int(r / noise.p_zz)      # 0 -> IZ, 1 -> ZI, 2 -> ZZ
                        if which != 0:
                            nc.add("z", g.qubits[0])
                        if which != 1:
                            nc.add("z", g.qubits[1])
            elif len(g.qubits) == 1:
                if noise.theta_1q:                       # coherent over-rotation
                    nc.add("rz", g.qubits[0], params=(noise.theta_1q,))
                if noise.p1 > 0 and rng.random() < noise.p1:
                    nc.add(rng.choice(_ONEQ_PAULIS), g.qubits[0])
                if noise.p_z1 > 0 and rng.random() < noise.p_z1:
                    nc.add("z", g.qubits[0])
        if noise.enabled and noise.p_idle > 0:
            for q in range(n):
                if q not in touched and rng.random() < noise.p_idle:
                    nc.add("z", q)
    return nc


def apply_readout(bitstring: str, noise: NoiseModel, rng) -> str:
    if not noise.enabled or noise.p_readout <= 0:
        return bitstring
    return "".join(
        (("1" if b == "0" else "0") if rng.random() < noise.p_readout else b)
        for b in bitstring
    )


def apply_readout_to_distribution(vec, noise: NoiseModel, n: int):
    """Apply the readout bit-flip channel analytically to a probability VECTOR.

    ``vec`` is a length-2^n numpy array indexed with qubit ``j`` in bit ``j`` (the
    qiskit/little-endian convention the statevector backend produces). Independent
    bit-flips factorise, so this is n successive 2x2 mixings rather than a 2^n x 2^n
    matrix -- exact, and equivalent to stim's ``X_ERROR`` before ``M``.

    :func:`apply_readout` does the same thing to one sampled bitstring; a
    trajectory-averaged distribution has no bitstrings to flip, hence this variant.
    """
    import numpy as np

    if not noise.enabled or noise.p_readout <= 0:
        return vec
    p = noise.p_readout
    out = np.asarray(vec, dtype=float).reshape([2] * n)
    for axis in range(n):
        out = np.moveaxis(out, axis, 0)
        out = np.stack(((1 - p) * out[0] + p * out[1],
                        p * out[0] + (1 - p) * out[1]))
        out = np.moveaxis(out, 0, axis)
    return out.reshape(-1)


# classical_fidelity now lives in proxysim.metrics; re-exported here for back-compat.
from .metrics import classical_fidelity  # noqa: E402,F401
