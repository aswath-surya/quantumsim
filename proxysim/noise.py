"""General circuit-level noise model for statevector / tensor-network trajectories.

This module replaces the original Pauli-only v2 noise model. It intentionally
does not preserve Stim compatibility. The trajectory path supports:

  * gate-, qubit-, and pair-dependent stochastic Pauli noise;
  * coherent one-qubit overrotations;
  * coherent two-qubit ZZ overrotations;
  * coherent controlled-phase offsets;
  * idle dephasing and coherent idle-Z drift;
  * correlated spectator/crosstalk Pauli faults;
  * asymmetric readout errors;
  * per-trajectory quasi-static drift.

The same native error model should be used for:
  1. cycle benchmarking,
  2. randomized compiling,
  3. target-circuit simulation.

Important
---------
`sample_trajectory` inserts unitary and Pauli errors into a copied circuit.
It therefore handles unitary/coherent and stochastic-Pauli errors. Genuine
nonunitary channels such as amplitude damping require a state-aware Kraus or
quantum-jump trajectory backend and are not silently approximated here.

The circuit IR is assumed to support:
    circuit.add(name, *qubits, *params)
for the gates:
    x, y, z, rx, ry, rz, rzz, cp

If your Circuit.add signature differs, edit only `_append_gate`.
"""

from __future__ import annotations

import copy
import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

_ONEQ_PAULIS = ("x", "y", "z")
_TWOQ_PAULIS = tuple(
    "".join(p)
    for p in itertools.product("IXYZ", repeat=2)
    if "".join(p) != "II"
)

GateKey1Q = Tuple[str, int]
GateKey2Q = Tuple[str, int, int]
Pair = Tuple[int, int]


def _canonical_pair(a: int, b: int) -> Pair:
    return (a, b) if a <= b else (b, a)


def _clamp_probability(p: float, name: str) -> float:
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {p}.")
    return p


def _validate_distribution(weights: Mapping[str, float], allowed: Sequence[str], name: str):
    unknown = set(weights) - set(allowed)
    if unknown:
        raise ValueError(f"{name} contains unsupported labels: {sorted(unknown)}")
    for label, value in weights.items():
        if value < 0.0:
            raise ValueError(f"{name}[{label!r}] must be nonnegative.")


def _weighted_choice(rng: random.Random, weights: Mapping[str, float]) -> Optional[str]:
    """Choose a label from unnormalized nonnegative weights.

    Returns None when the total weight is zero.
    """
    total = float(sum(weights.values()))
    if total <= 0.0:
        return None

    x = rng.random() * total
    accum = 0.0
    for label, weight in weights.items():
        accum += float(weight)
        if x <= accum:
            return label
    return next(reversed(weights))


def _append_gate(
    circuit,
    name: str,
    qubits: Sequence[int],
    params: Sequence[float] = (),
):
    """Append a gate using explicit qubit and parameter fields."""
    circuit.add(
        name,
        *qubits,
        params=tuple(params),
    )


@dataclass
class NoiseModel:
    enabled: bool = False

    # ------------------------------------------------------------------
    # Baseline stochastic error probabilities.
    # These are total probabilities that a non-identity Pauli fault occurs.
    # ------------------------------------------------------------------
    p1: float = 1e-3
    p2: float = 1e-2
    p_idle_z: float = 1e-3

    # Relative Pauli weights conditioned on an error occurring.
    oneq_pauli_weights: Dict[str, float] = field(
        default_factory=lambda: {"x": 1.0, "y": 1.0, "z": 1.0}
    )
    twoq_pauli_weights: Dict[str, float] = field(
        default_factory=lambda: {label: 1.0 for label in _TWOQ_PAULIS}
    )

    # ------------------------------------------------------------------
    # Coherent errors, in radians.
    # Applied after the corresponding ideal physical gate.
    # ------------------------------------------------------------------
    oneq_overrotation: float = 0.0
    oneq_axis: str = "z"            # x, y, or z
    twoq_zz_overrotation: float = 0.0
    controlled_phase_offset: float = 0.0
    idle_z_angle: float = 0.0

    # ------------------------------------------------------------------
    # Asymmetric classical readout assignment errors.
    # ------------------------------------------------------------------
    p_readout_01: float = 1e-2      # true 0 reported as 1
    p_readout_10: float = 1e-2      # true 1 reported as 0

    # ------------------------------------------------------------------
    # Optional gate-, qubit-, and pair-specific overrides.
    #
    # Examples:
    #   oneq_error_by_gate["h"] = 2e-3
    #   oneq_error_by_gate_qubit[("h", 3)] = 4e-3
    #   twoq_error_by_gate["cz"] = 1.5e-2
    #   twoq_error_by_gate_pair[("cz", 1, 2)] = 2.1e-2
    # ------------------------------------------------------------------
    oneq_error_by_gate: Dict[str, float] = field(default_factory=dict)
    oneq_error_by_gate_qubit: Dict[GateKey1Q, float] = field(default_factory=dict)
    twoq_error_by_gate: Dict[str, float] = field(default_factory=dict)
    twoq_error_by_gate_pair: Dict[GateKey2Q, float] = field(default_factory=dict)

    oneq_angle_by_gate: Dict[str, float] = field(default_factory=dict)
    oneq_angle_by_gate_qubit: Dict[GateKey1Q, float] = field(default_factory=dict)
    zz_angle_by_gate: Dict[str, float] = field(default_factory=dict)
    zz_angle_by_gate_pair: Dict[GateKey2Q, float] = field(default_factory=dict)
    phase_offset_by_gate_pair: Dict[GateKey2Q, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Crosstalk / spectator model.
    # When a 2q gate is applied to pair (a,b), each listed spectator may
    # receive a stochastic Z and/or coherent Z rotation.
    # ------------------------------------------------------------------
    spectator_z_probability: float = 0.0
    spectator_z_angle: float = 0.0
    spectators_by_pair: Dict[Pair, Tuple[int, ...]] = field(default_factory=dict)

    # Optional correlated Pauli error affecting the active pair and a spectator.
    correlated_fault_probability: float = 0.0
    correlated_fault_label: str = "ZZZ"

    # ------------------------------------------------------------------
    # Quasi-static trajectory-to-trajectory drift.
    # One Gaussian multiplier/offset is drawn per trajectory.
    # ------------------------------------------------------------------
    stochastic_rate_drift_std: float = 0.0
    coherent_angle_drift_std: float = 0.0

    def __post_init__(self):
        for name in (
            "p1",
            "p2",
            "p_idle_z",
            "p_readout_01",
            "p_readout_10",
            "spectator_z_probability",
            "correlated_fault_probability",
        ):
            _clamp_probability(getattr(self, name), name)

        if self.oneq_axis.lower() not in {"x", "y", "z"}:
            raise ValueError("oneq_axis must be 'x', 'y', or 'z'.")

        _validate_distribution(
            self.oneq_pauli_weights,
            _ONEQ_PAULIS,
            "oneq_pauli_weights",
        )
        _validate_distribution(
            self.twoq_pauli_weights,
            _TWOQ_PAULIS,
            "twoq_pauli_weights",
        )

        for table_name in (
            "oneq_error_by_gate",
            "oneq_error_by_gate_qubit",
            "twoq_error_by_gate",
            "twoq_error_by_gate_pair",
        ):
            table = getattr(self, table_name)
            for key, value in table.items():
                _clamp_probability(value, f"{table_name}[{key!r}]")

        if len(self.correlated_fault_label) != 3:
            raise ValueError("correlated_fault_label must contain three Pauli letters.")
        if any(ch not in "IXYZ" for ch in self.correlated_fault_label):
            raise ValueError("correlated_fault_label must use only I, X, Y, Z.")

    def scaled(self, scale: float) -> "NoiseModel":
        """Return a copy with stochastic rates and coherent angles scaled."""
        if scale < 0.0:
            raise ValueError("scale must be nonnegative.")

        new = copy.deepcopy(self)
        new.enabled = True

        for name in (
            "p1",
            "p2",
            "p_idle_z",
            "p_readout_01",
            "p_readout_10",
            "spectator_z_probability",
            "correlated_fault_probability",
        ):
            setattr(new, name, min(1.0, float(getattr(new, name)) * scale))

        for name in (
            "oneq_overrotation",
            "twoq_zz_overrotation",
            "controlled_phase_offset",
            "idle_z_angle",
            "spectator_z_angle",
        ):
            setattr(new, name, float(getattr(new, name)) * scale)

        for table_name in (
            "oneq_error_by_gate",
            "oneq_error_by_gate_qubit",
            "twoq_error_by_gate",
            "twoq_error_by_gate_pair",
        ):
            table = getattr(new, table_name)
            for key in table:
                table[key] = min(1.0, float(table[key]) * scale)

        for table_name in (
            "oneq_angle_by_gate",
            "oneq_angle_by_gate_qubit",
            "zz_angle_by_gate",
            "zz_angle_by_gate_pair",
            "phase_offset_by_gate_pair",
        ):
            table = getattr(new, table_name)
            for key in table:
                table[key] = float(table[key]) * scale

        return new

    def oneq_error_probability(self, gate: str, qubit: int) -> float:
        key = (gate, qubit)
        if key in self.oneq_error_by_gate_qubit:
            return self.oneq_error_by_gate_qubit[key]
        if gate in self.oneq_error_by_gate:
            return self.oneq_error_by_gate[gate]
        return self.p1

    def twoq_error_probability(self, gate: str, a: int, b: int) -> float:
        a, b = _canonical_pair(a, b)
        key = (gate, a, b)
        if key in self.twoq_error_by_gate_pair:
            return self.twoq_error_by_gate_pair[key]
        if gate in self.twoq_error_by_gate:
            return self.twoq_error_by_gate[gate]
        return self.p2

    def oneq_coherent_angle(self, gate: str, qubit: int) -> float:
        key = (gate, qubit)
        if key in self.oneq_angle_by_gate_qubit:
            return self.oneq_angle_by_gate_qubit[key]
        if gate in self.oneq_angle_by_gate:
            return self.oneq_angle_by_gate[gate]
        return self.oneq_overrotation

    def zz_coherent_angle(self, gate: str, a: int, b: int) -> float:
        a, b = _canonical_pair(a, b)
        key = (gate, a, b)
        if key in self.zz_angle_by_gate_pair:
            return self.zz_angle_by_gate_pair[key]
        if gate in self.zz_angle_by_gate:
            return self.zz_angle_by_gate[gate]
        return self.twoq_zz_overrotation

    def cp_phase_offset(self, gate: str, a: int, b: int) -> float:
        a, b = _canonical_pair(a, b)
        return self.phase_offset_by_gate_pair.get(
            (gate, a, b),
            self.controlled_phase_offset,
        )


@dataclass(frozen=True)
class _TrajectoryDrift:
    rate_multiplier: float
    angle_offset: float


def _draw_drift(noise: NoiseModel, rng: random.Random) -> _TrajectoryDrift:
    rate_multiplier = 1.0
    if noise.stochastic_rate_drift_std > 0.0:
        rate_multiplier += rng.gauss(0.0, noise.stochastic_rate_drift_std)
        rate_multiplier = max(0.0, rate_multiplier)

    angle_offset = 0.0
    if noise.coherent_angle_drift_std > 0.0:
        angle_offset = rng.gauss(0.0, noise.coherent_angle_drift_std)

    return _TrajectoryDrift(rate_multiplier, angle_offset)


def _sample_probability(base_p: float, drift: _TrajectoryDrift) -> float:
    return min(1.0, max(0.0, base_p * drift.rate_multiplier))


def _apply_pauli_label(circuit, label: str, qubits: Sequence[int]):
    if len(label) != len(qubits):
        raise ValueError("Pauli label length does not match qubit count.")

    for pauli, qubit in zip(label.upper(), qubits):
        if pauli != "I":
            _append_gate(circuit, pauli.lower(), [qubit])


def _apply_oneq_coherent_error(
    circuit,
    qubit: int,
    axis: str,
    angle: float,
):
    if abs(angle) <= 0.0:
        return
    _append_gate(circuit, f"r{axis.lower()}", [qubit], [angle])


def _apply_twoq_coherent_error(
    circuit,
    gate_name: str,
    a: int,
    b: int,
    zz_angle: float,
    phase_offset: float,
):
    if abs(zz_angle) > 0.0:
        _append_gate(circuit, "rzz", [a, b], [zz_angle])

    # This is an additional controlled phase applied after the ideal gate.
    if abs(phase_offset) > 0.0:
        _append_gate(circuit, "cp", [a, b], [phase_offset])


def sample_trajectory(circuit, noise: NoiseModel, rng: random.Random):
    """Return a new circuit with one sampled native-error trajectory.

    Order after each ideal gate:
      1. deterministic/quasi-static coherent error,
      2. sampled stochastic Pauli error,
      3. optional spectator/correlated error.

    Readout error remains classical and is applied by `apply_readout`.
    """
    from .circuit import Circuit

    nc = Circuit(circuit.n_qubits, name=circuit.name + "_traj")
    n = circuit.n_qubits

    if not noise.enabled:
        for gate in circuit.gates:
            nc.gates.append(gate)
        return nc

    drift = _draw_drift(noise, rng)

    for layer in circuit.layers():
        touched = set()

        for gate in layer:
            nc.gates.append(gate)
            qubits = tuple(gate.qubits)
            touched.update(qubits)

            if len(qubits) == 1:
                q = qubits[0]

                angle = (
                    noise.oneq_coherent_angle(gate.name, q)
                    + drift.angle_offset
                )
                _apply_oneq_coherent_error(
                    nc,
                    q,
                    noise.oneq_axis,
                    angle,
                )

                p = _sample_probability(
                    noise.oneq_error_probability(gate.name, q),
                    drift,
                )
                if p > 0.0 and rng.random() < p:
                    label = _weighted_choice(rng, noise.oneq_pauli_weights)
                    if label is not None:
                        _append_gate(nc, label, [q])

            elif len(qubits) == 2:
                a, b = qubits

                zz_angle = (
                    noise.zz_coherent_angle(gate.name, a, b)
                    + drift.angle_offset
                )
                phase_offset = noise.cp_phase_offset(gate.name, a, b)

                _apply_twoq_coherent_error(
                    nc,
                    gate.name,
                    a,
                    b,
                    zz_angle,
                    phase_offset,
                )

                p = _sample_probability(
                    noise.twoq_error_probability(gate.name, a, b),
                    drift,
                )
                if p > 0.0 and rng.random() < p:
                    label = _weighted_choice(rng, noise.twoq_pauli_weights)
                    if label is not None:
                        _apply_pauli_label(nc, label, [a, b])

                pair = _canonical_pair(a, b)
                spectators = noise.spectators_by_pair.get(pair, ())
                for spectator in spectators:
                    if (
                        noise.spectator_z_probability > 0.0
                        and rng.random()
                        < _sample_probability(
                            noise.spectator_z_probability,
                            drift,
                        )
                    ):
                        _append_gate(nc, "z", [spectator])

                    if abs(noise.spectator_z_angle) > 0.0:
                        _append_gate(
                            nc,
                            "rz",
                            [spectator],
                            [noise.spectator_z_angle],
                        )

                    if (
                        noise.correlated_fault_probability > 0.0
                        and rng.random()
                        < _sample_probability(
                            noise.correlated_fault_probability,
                            drift,
                        )
                    ):
                        _apply_pauli_label(
                            nc,
                            noise.correlated_fault_label,
                            [a, b, spectator],
                        )

            else:
                raise ValueError(
                    f"Noise model supports only 1q and 2q gates; "
                    f"got {gate.name!r} on {len(qubits)} qubits."
                )

        # Idle errors after each logical layer.
        for q in range(n):
            if q in touched:
                continue

            if (
                noise.p_idle_z > 0.0
                and rng.random()
                < _sample_probability(noise.p_idle_z, drift)
            ):
                _append_gate(nc, "z", [q])

            if abs(noise.idle_z_angle) > 0.0:
                _append_gate(nc, "rz", [q], [noise.idle_z_angle])

    return nc


def apply_readout(bitstring: str, noise: NoiseModel, rng: random.Random) -> str:
    """Apply independent asymmetric readout assignment errors."""
    if not noise.enabled:
        return bitstring

    out = []
    for bit in bitstring:
        if bit == "0":
            flip = rng.random() < noise.p_readout_01
            out.append("1" if flip else "0")
        elif bit == "1":
            flip = rng.random() < noise.p_readout_10
            out.append("0" if flip else "1")
        else:
            raise ValueError(f"Invalid bit {bit!r} in bitstring.")
    return "".join(out)


def sample_readout_counts(
    probabilities,
    shots: int,
    noise: NoiseModel,
    rng: random.Random,
) -> Dict[str, int]:
    """Sample finite-shot bitstrings and then apply assignment errors."""
    probs = list(float(x) for x in probabilities)
    total = sum(probs)
    if total <= 0.0:
        raise ValueError("Probability vector has zero total mass.")
    probs = [p / total for p in probs]

    n_qubits = round(math.log2(len(probs)))
    if 2**n_qubits != len(probs):
        raise ValueError("Probability vector length must be a power of two.")

    # `random.choices` is adequate for modest shot counts.
    outcomes = rng.choices(range(len(probs)), weights=probs, k=shots)

    counts: Dict[str, int] = {}
    for outcome in outcomes:
        bitstring = format(outcome, f"0{n_qubits}b")
        bitstring = apply_readout(bitstring, noise, rng)
        counts[bitstring] = counts.get(bitstring, 0) + 1

    return counts


# Backward-compatible re-export.
from .metrics import classical_fidelity  # noqa: E402,F401

# """Circuit-level noise model with a single on/off toggle.

# Channels (all Pauli-stochastic, matching stim's conventions so the stim-native
# and trajectory paths agree exactly):

#   * 1-qubit depolarizing after every single-qubit gate      (p1)   -> DEPOLARIZE1
#   * 2-qubit depolarizing after every two-qubit gate         (p2)   -> DEPOLARIZE2
#   * idling dephasing (Z) on qubits idle during a layer      (p_idle) -> Z_ERROR
#   * measurement bit-flip (readout error)                    (p_readout) -> X_ERROR before M

# `NoiseModel.enabled` is the toggle: when False, every channel is skipped and the
# simulation is ideal.

# Two application paths:
#   * :func:`to_stim_noisy` -- inserts native stim noise ops (exact, scalable) for
#     the stabilizer backend. Clifford circuits only.
#   * :func:`sample_trajectory` + :func:`apply_readout` -- Monte-Carlo trajectory:
#     sample which Pauli errors fire and splice them into the circuit as gates, for
#     pure-state backends (statevector / MPS). Averaging trajectories reproduces the
#     mixed-state distribution.

# This is the setting where finite-shot sampling matters (a mixed state can't be
# read off as a single |<x|psi>|^2), i.e. the PTA regime of Merkel et al.

# Fidelity of a noisy vs ideal distribution: classical (Hellinger) fidelity
# F = (sum_x sqrt(p q))^2 in [0, 1].
# """

# from __future__ import annotations

# import itertools
# import math
# from dataclasses import dataclass
# from typing import Dict

# import stim

# from .backends.stabilizer import _PARAM, _SIMPLE

# # 15 non-identity two-qubit Pauli labels, e.g. "IX", "XZ", ...
# _TWOQ_PAULIS = ["".join(p) for p in itertools.product("IXYZ", repeat=2) if "".join(p) != "II"]
# _ONEQ_PAULIS = ["x", "y", "z"]


# @dataclass
# class NoiseModel:
#     enabled: bool = False
#     p1: float = 1e-3          # 1-qubit depolarizing after each single-qubit gate
#     p2: float = 1e-2          # 2-qubit depolarizing after each two-qubit gate
#     p_readout: float = 1e-2   # measurement bit-flip probability
#     p_idle: float = 1e-3      # idling dephasing (Z) per idle qubit per layer

#     def scaled(self, s: float) -> "NoiseModel":
#         """Same model with every rate multiplied by ``s`` (and enabled)."""
#         return NoiseModel(True, self.p1 * s, self.p2 * s, self.p_readout * s, self.p_idle * s)


# # ---------------------------------------------------------------------------
# # stim-native application (exact, scalable; Clifford circuits only)
# # ---------------------------------------------------------------------------
# def _emit_stim(c: stim.Circuit, g):
#     if g.name in _SIMPLE:
#         c.append(_SIMPLE[g.name], list(g.qubits))
#         return
#     if g.name in _PARAM:
#         k = round(g.params[0] / (math.pi / 2)) % 4
#         gate = _PARAM[g.name][k]
#         if gate is not None:
#             c.append(gate, list(g.qubits))
#         return
#     raise ValueError(f"noise/stim: non-Clifford gate '{g.name}'")


def to_stim_noisy(circuit, noise: NoiseModel) -> stim.Circuit:
    """Build a measured stim circuit with native noise ops inserted."""
    c = stim.Circuit()
    n = circuit.n_qubits
    for layer in circuit.layers():
        touched = set()
        for g in layer:
            _emit_stim(c, g)
            touched.update(g.qubits)
            if noise.enabled:
                if len(g.qubits) == 2 and noise.p2 > 0:
                    c.append("DEPOLARIZE2", list(g.qubits), noise.p2)
                elif len(g.qubits) == 1 and noise.p1 > 0:
                    c.append("DEPOLARIZE1", list(g.qubits), noise.p1)
        if noise.enabled and noise.p_idle > 0:
            idle = [q for q in range(n) if q not in touched]
            if idle:
                c.append("Z_ERROR", idle, noise.p_idle)
    if noise.enabled and noise.p_readout > 0:
        c.append("X_ERROR", list(range(n)), noise.p_readout)
    c.append("M", list(range(n)))
    return c


# # ---------------------------------------------------------------------------
# # Trajectory application (for pure-state backends: statevector / MPS)
# # ---------------------------------------------------------------------------
# def sample_trajectory(circuit, noise: NoiseModel, rng):
#     """Return a NEW Circuit with one sampled Pauli-error realisation spliced in.

#     Readout error is NOT applied here -- apply :func:`apply_readout` to the
#     measured bitstring afterwards.
#     """
#     from .circuit import Circuit

#     nc = Circuit(circuit.n_qubits, name=circuit.name + "_traj")
#     n = circuit.n_qubits
#     for layer in circuit.layers():
#         touched = set()
#         for g in layer:
#             nc.gates.append(g)
#             touched.update(g.qubits)
#             if not noise.enabled:
#                 continue
#             if len(g.qubits) == 2 and noise.p2 > 0 and rng.random() < noise.p2:
#                 pa, pb = rng.choice(_TWOQ_PAULIS)
#                 if pa != "I":
#                     nc.add(pa.lower(), g.qubits[0])
#                 if pb != "I":
#                     nc.add(pb.lower(), g.qubits[1])
#             elif len(g.qubits) == 1 and noise.p1 > 0 and rng.random() < noise.p1:
#                 nc.add(rng.choice(_ONEQ_PAULIS), g.qubits[0])
#         if noise.enabled and noise.p_idle > 0:
#             for q in range(n):
#                 if q not in touched and rng.random() < noise.p_idle:
#                     nc.add("z", q)
#     return nc


# def apply_readout(bitstring: str, noise: NoiseModel, rng) -> str:
#     if not noise.enabled or noise.p_readout <= 0:
#         return bitstring
#     return "".join(
#         (("1" if b == "0" else "0") if rng.random() < noise.p_readout else b)
#         for b in bitstring
#     )


# # classical_fidelity now lives in proxysim.metrics; re-exported here for back-compat.
# from .metrics import classical_fidelity  # noqa: E402,F401
