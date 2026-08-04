"""OpenQASM 2.0 emission -- the inverse of :func:`proxysim.circuit.circuit_from_qasm`.

The package could always *read* QASM (``circuit_from_qasm``); this module lets it
*write* QASM too, so a circuit built or transformed in the IR can be handed to an
external tool. Everything round-trips: ``circuit_from_qasm(circuit_to_qasm(c))``
reproduces ``c`` gate for gate.

Only the gates the IR itself supports are emitted, all of which are in ``qelib1.inc``.
Rotation angles are printed with ``repr``-level precision (17 significant digits) so a
round trip is exact to the last bit -- a QASM bank is a wire format, and silently
rounding ``pi/1024`` in a deep QFT would change the circuit.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from .circuit import Circuit

# IR gate name -> QASM 2.0 name. The inverse of circuit.py's _QASM_GATE, except that
# the IR has already collapsed the aliases (cnot -> cx, cu1 -> cp, p -> rz), so this
# map is single-valued in both directions.
_IR_TO_QASM = {
    "i": "id", "x": "x", "y": "y", "z": "z", "h": "h", "s": "s", "sdg": "sdg",
    "sx": "sx", "sxdg": "sxdg", "t": "t", "tdg": "tdg",
    "rx": "rx", "ry": "ry", "rz": "rz",
    "cx": "cx", "cz": "cz", "cy": "cy", "swap": "swap", "cp": "cp",
}

# Overrides for the strict `qelib1.inc` dialect -- the gate list the original OpenQASM 2.0
# standard header declares, and all that a conforming parser (staq, and therefore XACC) is
# obliged to know. `cp` is the one name in _IR_TO_QASM that postdates it; `cu1(lambda)` is
# the same controlled-phase gate under its qelib1 name, and circuit.py's _QASM_GATE already
# maps `cu1` back to `cp` on read, so the round trip is unaffected.
#
# `sx`/`sxdg` are also absent from qelib1 but have no qelib1 spelling at all -- they must be
# rewritten as rotations *before* emission. `proxysim.c3pq.lower_for_c3pq` does exactly
# that, and is what callers targeting this dialect should run first.
_QELIB1_OVERRIDES = {"cp": "cu1"}

DIALECTS = ("full", "qelib1")


def _gate_names(dialect: str) -> dict:
    if dialect == "full":
        return _IR_TO_QASM
    if dialect == "qelib1":
        return {**_IR_TO_QASM, **_QELIB1_OVERRIDES}
    raise ValueError(f"circuit_to_qasm: unknown dialect {dialect!r}, expected one of {DIALECTS}")


def _fmt_angle(theta: float) -> str:
    """Full-precision angle literal. ``repr`` on a float is the shortest string that
    parses back to the identical double, which is exactly what a wire format wants."""
    return repr(float(theta))


def gate_to_qasm(gate, reg: str = "q", dialect: str = "full") -> str:
    """One QASM statement for one IR gate."""
    try:
        name = _gate_names(dialect)[gate.name]
    except KeyError:
        raise ValueError(f"circuit_to_qasm: no QASM 2.0 form for gate '{gate.name}'")
    params = f"({','.join(_fmt_angle(p) for p in gate.params)})" if gate.params else ""
    targets = ",".join(f"{reg}[{q}]" for q in gate.qubits)
    return f"{name}{params} {targets};"


def circuit_to_qasm(circuit: Circuit,
                    measured: Optional[Sequence[int]] = None,
                    qreg: str = "q",
                    creg: str = "c",
                    header_lines: Iterable[str] = (),
                    dialect: str = "full") -> str:
    """Serialize an IR :class:`~proxysim.circuit.Circuit` to an OpenQASM 2.0 string.

    Parameters
    ----------
    circuit      : the circuit to write.
    measured     : qubit indices to measure at the end, in order. ``None`` measures
                   every qubit; ``[]`` emits a unitary-only file with no ``creg``.
                   Bit ``k`` of the creg is ``measured[k]`` -- for circuits with
                   unmeasured ancillas (MQT's ``dj``) that mapping is not the identity,
                   so it is recorded in the header.
    header_lines : lines emitted as leading ``//`` comments. Used to make every file in
                   a generated bank self-describing (algorithm, depth, seed, and for CB
                   circuits the prep/measure Pauli needed to analyze it).
    dialect      : ``'full'`` (default) emits every name in ``_IR_TO_QASM``, which is what
                   qiskit and this package read back. ``'qelib1'`` restricts to the gates
                   the original ``qelib1.inc`` declares, spelling ``cp`` as ``cu1`` -- use
                   it for strict parsers such as staq/XACC, after running
                   :func:`proxysim.c3pq.lower_for_c3pq` to clear ``sx``/``sxdg``/``s``/``t``,
                   which have no qelib1 spelling.
    """
    if measured is None:
        measured = list(range(circuit.n_qubits))
    measured = list(measured)

    out: List[str] = [f"// {line}" for line in header_lines]
    out += ['OPENQASM 2.0;', 'include "qelib1.inc";', f"qreg {qreg}[{circuit.n_qubits}];"]
    if measured:
        out.append(f"creg {creg}[{len(measured)}];")
    out += [gate_to_qasm(g, qreg, dialect) for g in circuit.gates]
    out += [f"measure {qreg}[{q}] -> {creg}[{k}];" for k, q in enumerate(measured)]
    return "\n".join(out) + "\n"


def write_qasm(path, circuit: Circuit, measured=None, header_lines=(),
               dialect: str = "full") -> int:
    """Write ``circuit`` to ``path`` as QASM 2.0; return the number of bytes written."""
    text = circuit_to_qasm(circuit, measured=measured, header_lines=header_lines,
                           dialect=dialect)
    with open(path, "w") as fh:
        fh.write(text)
    return len(text)
