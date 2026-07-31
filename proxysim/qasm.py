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


def _fmt_angle(theta: float) -> str:
    """Full-precision angle literal. ``repr`` on a float is the shortest string that
    parses back to the identical double, which is exactly what a wire format wants."""
    return repr(float(theta))


def gate_to_qasm(gate, reg: str = "q") -> str:
    """One QASM statement for one IR gate."""
    try:
        name = _IR_TO_QASM[gate.name]
    except KeyError:
        raise ValueError(f"circuit_to_qasm: no QASM 2.0 form for gate '{gate.name}'")
    params = f"({','.join(_fmt_angle(p) for p in gate.params)})" if gate.params else ""
    targets = ",".join(f"{reg}[{q}]" for q in gate.qubits)
    return f"{name}{params} {targets};"


def circuit_to_qasm(circuit: Circuit,
                    measured: Optional[Sequence[int]] = None,
                    qreg: str = "q",
                    creg: str = "c",
                    header_lines: Iterable[str] = ()) -> str:
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
    """
    if measured is None:
        measured = list(range(circuit.n_qubits))
    measured = list(measured)

    out: List[str] = [f"// {line}" for line in header_lines]
    out += ['OPENQASM 2.0;', 'include "qelib1.inc";', f"qreg {qreg}[{circuit.n_qubits}];"]
    if measured:
        out.append(f"creg {creg}[{len(measured)}];")
    out += [gate_to_qasm(g, qreg) for g in circuit.gates]
    out += [f"measure {qreg}[{q}] -> {creg}[{k}];" for k, q in enumerate(measured)]
    return "\n".join(out) + "\n"


def write_qasm(path, circuit: Circuit, measured=None, header_lines=()) -> int:
    """Write ``circuit`` to ``path`` as QASM 2.0; return the number of bytes written."""
    text = circuit_to_qasm(circuit, measured=measured, header_lines=header_lines)
    with open(path, "w") as fh:
        fh.write(text)
    return len(text)
