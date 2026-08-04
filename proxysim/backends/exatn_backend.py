import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# Setup logging
logger = logging.getLogger("proxysim.backends.exatn")

try:
    import xacc
    XACC_AVAILABLE = True
except ImportError:
    XACC_AVAILABLE = False

# A minimal, unambiguously valid OpenQASM 2.0 program used at init time to prove that a
# candidate XACC compiler really speaks OpenQASM. Merely resolving the service name is not
# enough: 'xasm' is always registered but parses XACC's own DSL, so it resolves fine and
# then rejects every line of every .qasm file in the bank.
_PROBE_QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""

# Ordered by preference. 'staq' is XACC's bundled OpenQASM 2.0 front end; the others are
# aliases some builds register. 'xasm' is deliberately absent -- it cannot parse OpenQASM.
_QASM_COMPILERS = ("staq", "openqasm", "qasm")


@dataclass
class SimulationResult:
    """Result of a shot-based simulation run via XACC/ExaTN."""
    counts: Dict[str, int]
    shots: int
    num_qubits: int
    backend: str
    qasm_path: str
    metadata: Dict[str, Any]

class ExaTNBackend:
    """
    XACC/TNQVM/ExaTN backend for executing QASM circuits.
    
    This backend leverages the XACC framework and the TNQVM accelerator 
    with the ExaTN visitor to simulate quantum circuits.
    """
    def __init__(
        self,
        shots: int,
        visitor: str = "exatn",
        seed: Optional[int] = None,
        reverse_bits: bool = True,
    ) -> None:
        self.shots = shots
        self.visitor = visitor
        self.seed = seed
        # XACC reports measurement bitstrings with qubit 0 as the *leftmost* character,
        # while the C3PQ .f64 convention indexes states with qubit 0 as the LSB. Reversing
        # is therefore the correct default; expose it so a differing build can be corrected
        # without editing the backend.
        self.reverse_bits = reverse_bits
        self.qpu = None
        self.compiler = None

        self._initialize_xacc()

    def _initialize_xacc(self) -> None:
        """Initializes XACC and verifies the availability of required services."""
        if not XACC_AVAILABLE:
            raise RuntimeError(
                "XACC is not installed or not importable in the current Python environment. "
                "Please ensure the XACC environment is properly activated."
            )

        try:
            # Initialize XACC - using Initialize() based on working script
            xacc.Initialize()
            
            # Attempt to instantiate the accelerator
            try:
                self.qpu = xacc.getAccelerator(
                    "tnqvm",
                    {
                        "tnqvm-visitor": self.visitor,
                        "shots": self.shots,
                    },
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to instantiate TNQVM accelerator with visitor '{self.visitor}': {e}. "
                    "Check if the visitor is supported by the installed TNQVM/ExaTN version."
                )

            # Determine which OpenQASM compiler to use. Each candidate must survive an
            # actual compile of _PROBE_QASM -- a registered service name proves nothing.
            self.compiler_name = None
            probe_errors: Dict[str, str] = {}
            for name in _QASM_COMPILERS:
                try:
                    candidate = xacc.getCompiler(name)
                    ir = candidate.compile(_PROBE_QASM)
                    self._extract_composite(ir)
                except Exception as e:
                    probe_errors[name] = str(e).strip().splitlines()[0] if str(e) else repr(e)
                    continue
                self.compiler = candidate
                self.compiler_name = name
                break

            if self.compiler_name is None:
                detail = "; ".join(f"{k}: {v}" for k, v in probe_errors.items())
                raise RuntimeError(
                    "No XACC compiler in this installation can parse OpenQASM 2.0. "
                    f"Tried {list(_QASM_COMPILERS)} -- {detail}. "
                    "The 'staq' compiler plugin ships with XACC; if it is missing, rebuild "
                    "XACC with the staq TPL enabled."
                )

            logger.info(f"XACC backend initialized: Accelerator=tnqvm, Visitor={self.visitor}, Compiler={self.compiler_name}")

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise e
            raise RuntimeError(f"XACC initialization failed: {e}")

    def run_qasm(self, qasm_path: Path) -> SimulationResult:
        """
        Executes a QASM file and returns shot counts.
        """
        try:
            with open(qasm_path, "r") as f:
                qasm_text = f.read()

            # Compile QASM to XACC circuit
            try:
                # The compiler may expect the QASM as a string or file
                # We use the compiler we detected during init
                circ = self.compiler.compile(qasm_text)
            except Exception as e:
                raise RuntimeError(
                    f"XACC compiler '{self.compiler_name}' failed to parse QASM file {qasm_path}: {e}. "
                    "Ensure the file is valid OpenQASM 2.0. If unsupported gates are present, "
                    "they may need to be decomposed manually."
                )

            # Extract executable composite circuit from the compiled IR
            exec_circ = self._extract_composite(circ)

            # Determine number of qubits
            num_qubits = self._infer_num_qubits(qasm_text, exec_circ)

            # Measurement handling
            # We need to ensure the circuit has terminal measurements on all qubits.
            exec_circ = self._ensure_terminal_measurements(exec_circ, num_qubits)

            # Allocate XACC qubits
            try:
                # Using qalloc based on working script
                q = xacc.qalloc(num_qubits)
            except Exception as e:
                raise RuntimeError(f"Failed to allocate XACC qubits for {num_qubits} qubits: {e}")

            # Execute
            try:
                # execute() takes (qubit_alloc, program)
                self.qpu.execute(q, exec_circ)
            except Exception as e:
                raise RuntimeError(f"Simulation failure for {qasm_path}: {e}")

            # Retrieve shot counts
            try:
                # Based on working script: q.getMeasurementCounts()
                raw_counts = q.getMeasurementCounts()
            except Exception as e:
                raise RuntimeError(f"Failed to extract counts from XACC qubits: {e}")

            # Normalize keys to zero-padded binary strings of length num_qubits
            counts = self.normalize_counts(raw_counts, num_qubits, reverse_bits=self.reverse_bits)

            # Validate total shots
            total_shots = sum(counts.values())
            if total_shots != self.shots:
                logger.warning(
                    f"Shot count mismatch for {qasm_path}: expected {self.shots}, got {total_shots}"
                )

            return SimulationResult(
                counts=counts,
                shots=total_shots,
                num_qubits=num_qubits,
                backend=f"xacc-tnqvm-{self.visitor}",
                qasm_path=str(qasm_path),
                metadata={
                    "tnqvm_visitor": self.visitor,
                    "xacc_compiler": self.compiler_name,
                    "reverse_bits": self.reverse_bits,
                }
            )

        except Exception as e:
            logger.error(f"Error running {qasm_path}: {e}")
            raise

    @staticmethod
    def _extract_composite(ir) -> Any:
        """Pull the single executable CompositeInstruction out of a compiled XACC IR.

        ``IR::getComposite`` takes a name, so calling it bare raises TypeError under
        pybind11 -- ``getComposites()`` is the accessor that returns the kernel list.
        """
        if hasattr(ir, "getComposites"):
            composites = ir.getComposites()
            if not composites:
                raise RuntimeError("Compiled IR contains no composite instructions (empty kernel).")
            return composites[0]
        # Already a CompositeInstruction (some compilers hand one back directly).
        if hasattr(ir, "getInstructions"):
            return ir
        raise RuntimeError(f"Unrecognized XACC compile() return type: {type(ir)!r}")

    def _infer_num_qubits(self, qasm_text: str, circ) -> int:
        """Determine the circuit width, preferring the QASM ``qreg`` declaration.

        The declaration is authoritative: a compiled composite only knows about qubits
        that some instruction actually touches, so an idle trailing qubit would silently
        shrink the width and corrupt the 2^n .f64 layout downstream.
        """
        widths = [int(m) for m in re.findall(r"qreg\s+\w+\[(\d+)\]", qasm_text)]
        if widths:
            if len(widths) > 1:
                raise RuntimeError(
                    f"Multiple qreg declarations found ({widths}); only single-register "
                    "QASM is supported by the C3PQ .f64 layout."
                )
            return widths[0]

        for accessor in ("nLogicalBits", "nPhysicalBits", "getNQuBits"):
            fn = getattr(circ, accessor, None)
            if fn is None:
                continue
            try:
                n = int(fn())
            except Exception:
                continue
            if n > 0:
                return n

        raise RuntimeError("Could not determine number of qubits from QASM or XACC circuit.")

    def _ensure_terminal_measurements(self, circ, num_qubits) -> Any:
        """Guarantee the circuit ends with a terminal measurement of every qubit.

        Unmeasured circuits get measurements appended. Partial or mid-circuit measurement
        is rejected rather than silently producing a distribution over the wrong register.
        """
        instructions = list(circ.getInstructions())
        measured: List[int] = []
        last_non_measure = -1
        for idx, inst in enumerate(instructions):
            if inst.name() == "Measure":
                measured.extend(int(b) for b in inst.bits())
            else:
                last_non_measure = idx

        if not measured:
            for q in range(num_qubits):
                circ.addInstruction(xacc.gate.create("Measure", [q]))
            return circ

        first_measure = next(
            i for i, inst in enumerate(instructions) if inst.name() == "Measure"
        )
        if first_measure < last_non_measure:
            raise RuntimeError(
                "Mid-circuit measurement is not supported: gates follow the first Measure."
            )
        if sorted(measured) != list(range(num_qubits)):
            raise RuntimeError(
                f"Partial measurement is not supported: measured qubits {sorted(measured)} "
                f"but the register has {num_qubits} qubits."
            )
        return circ

    def normalize_counts(
        self,
        raw_counts: Mapping[str, int],
        num_qubits: int,
        reverse_bits: bool,
    ) -> Dict[str, int]:
        """
        Converts raw count keys to MSB-first binary strings, i.e. qubit 0 is the rightmost
        character, so ``int(key, 2)`` is the C3PQ state index.

        XACC hands back strings whose *leftmost* character is qubit 0, so ``reverse_bits``
        should be True for those. Integer keys are already state indices and are never
        reversed.
        """
        normalized: Dict[str, int] = {}
        for key, count in raw_counts.items():
            if isinstance(key, int):
                bit_str = format(key, f"0{num_qubits}b")
            else:
                bit_str = str(key).strip()
                if len(bit_str) != num_qubits:
                    raise RuntimeError(
                        f"XACC returned a {len(bit_str)}-bit key {bit_str!r} for a "
                        f"{num_qubits}-qubit circuit."
                    )
                if reverse_bits:
                    bit_str = bit_str[::-1]

            normalized[bit_str] = normalized.get(bit_str, 0) + count

        return normalized
