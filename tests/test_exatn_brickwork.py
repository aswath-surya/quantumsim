import pytest
import numpy as np
from pathlib import Path
from proxysim.backends.exatn_backend import ExaTNBackend

# Use a marker to skip if XACC is not available
# In a real environment, this would be tied to a check for the xacc module
pytest.mark.exatn = pytest.mark.skipif(
    False, 
    reason="XACC/ExaTN not installed"
)

def create_qasm(body: str, n_qubits: int) -> Path:
    path = Path(f"test_brickwork_{np.random.randint(1000)}.qasm")
    header = f"OPENQASM 2.0;\nqreg q[{n_qubits}];\ncreg c[{n_qubits}];\n"
    footer = "measure q -> c;"
    path.write_text(header + body + "\n" + footer)
    return path

@pytest.mark.exatn
def test_brickwork_smoke():
    """Run a small brickwork circuit and check for valid counts."""
    n = 2
    backend = ExaTNBackend(shots=1000)
    # Simple 2-qubit brickwork: H on all, CZ, measure
    qasm_body = "H q[0];\nH q[1];\nCZ q[0],q[1];"
    qasm = create_qasm(qasm_body, n)
    try:
        res = backend.run_qasm(qasm)
        
        assert res.num_qubits == n
        assert sum(res.counts.values()) == 1000
        assert len(res.counts) > 0
        # For H-CZ-H (Bell state), we expect 00 and 11
        # But here we just have H-CZ, so we expect some distribution
        for k in res.counts.keys():
            assert len(k) == n
            assert set(k).issubset({"0", "1"})
            
    finally:
        qasm.unlink()

@pytest.mark.exatn
def test_brickwork_tvd():
    """Compare a small brickwork circuit against a known reference (simplified)."""
    n = 2
    backend = ExaTNBackend(shots=10000)
    qasm_body = "H q[0];\nH q[1];\nCZ q[0],q[1];"
    qasm = create_qasm(qasm_body, n)
    
    # Reference distribution for H(0)H(1)CZ(0,1)
    # State is 1/2(|00> + |11>) ? No, H0H1|00> = 1/2(|00>+|01>+|10>+|11>)
    # CZ maps |11> to -|11>.
    # Probabilities: |00>: 1/4, |01>: 1/4, |10>: 1/4, |11>: 1/4
    ref_probs = {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
    
    try:
        res = backend.run_qasm(qasm)
        
        # Compute TVD
        all_keys = set(ref_probs.keys())
        tvd = 0.0
        for k in all_keys:
            p_ref = ref_probs[k]
            p_exatn = res.counts.get(k, 0) / 10000
            tvd += abs(p_ref - p_exatn)
        tvd *= 0.5
        
        assert tvd < 0.05
    finally:
        qasm.unlink()

if __name__ == "__main__":
    pytest.main([__file__])
