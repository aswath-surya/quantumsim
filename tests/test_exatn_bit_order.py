import pytest
import numpy as np
from pathlib import Path
from proxysim.backends.exatn_backend import ExaTNBackend

# Use a marker to skip if XACC is not available
pytest.mark.exatn = pytest.mark.skipif(
    # This is a dummy check; the actual Backend init will fail if XACC is missing
    False, 
    reason="XACC/ExaTN not installed"
)

def create_qasm(body: str, n_qubits: int) -> Path:
    path = Path(f"test_circuit_{np.random.randint(1000)}.qasm")
    header = f"OPENQASM 2.0;\nqreg q[{n_qubits}];\ncreg c[{n_qubits}];\n"
    footer = "measure q -> c;"
    path.write_text(header + body + "\n" + footer)
    return path

@pytest.mark.exatn
def test_bit_order_all_zero():
    """Test A: all-zero state should produce '0' * n."""
    n = 3
    backend = ExaTNBackend(shots=100)
    qasm = create_qasm("", n)
    try:
        res = backend.run_qasm(qasm)
        # Expected: '000' with probability 1
        assert "0" * n in res.counts
        assert sum(res.counts.values()) == 100
        assert len(res.counts) == 1
    finally:
        qasm.unlink()

@pytest.mark.exatn
def test_bit_order_q0_flip():
    """Test B: flip q[0], expect LSB to be 1."""
    n = 3
    backend = ExaTNBackend(shots=100)
    qasm = create_qasm("x q[0];", n)
    try:
        res = backend.run_qasm(qasm)
        # C3PQ convention: q0 is LSB. Result should be '001'.
        # If result is '100', then q0 is MSB.
        expected = "0" * (n-1) + "1"
        assert expected in res.counts
        assert len(res.counts) == 1
    finally:
        qasm.unlink()

@pytest.mark.exatn
def test_bit_order_qn_flip():
    """Test C: flip q[n-1], expect MSB to be 1."""
    n = 3
    backend = ExaTNBackend(shots=100)
    qasm = create_qasm(f"x q[{n-1}];", n)
    try:
        res = backend.run_qasm(qasm)
        # C3PQ convention: q(n-1) is MSB. Result should be '100'.
        expected = "1" + "0" * (n-1)
        assert expected in res.counts
        assert len(res.counts) == 1
    finally:
        qasm.unlink()

if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__])
