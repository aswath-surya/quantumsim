from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import qiskit
from mqt.bench import BenchmarkLevel, get_benchmark
from qiskit import qasm3


BENCHMARKS = [
    "ae",
    "bv",
    "dj",
    "ghz",
    "graphstate",
    "grover",
    "qft",
    "qftentangled",
    "qnn",
    "randomcircuit",
    "vqe_real_amp",
    "vqe_su2",
    "vqe_two_local",
    "wstate",
]


def export_benchmarks(
    requested_qubits: int,
    output_dir: str | Path,
    require_exact_size: bool = True,
) -> None:
    """Generate MQT Bench circuits and export them as OpenQASM 3."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []

    for benchmark_name in BENCHMARKS:
        try:
            circuit = get_benchmark(
                benchmark=benchmark_name,
                level=BenchmarkLevel.ALG,
                circuit_size=requested_qubits,
            )

            actual_qubits = circuit.num_qubits

            if require_exact_size and actual_qubits != requested_qubits:
                message = (
                    f"requested {requested_qubits} qubits, "
                    f"generated {actual_qubits}"
                )
                print(f"[skipped] {benchmark_name}: {message}")

                manifest_rows.append(
                    {
                        "benchmark": benchmark_name,
                        "requested_qubits": requested_qubits,
                        "actual_qubits": actual_qubits,
                        "depth": circuit.depth(),
                        "gates": circuit.size(),
                        "status": "size_mismatch",
                        "path": "",
                        "error": message,
                    }
                )
                continue

            filename = output_path / (
                f"{benchmark_name}_{actual_qubits}q.qasm"
            )

            # More robust across Qiskit versions than passing Path to dump().
            qasm_text = qasm3.dumps(circuit)
            filename.write_text(qasm_text, encoding="utf-8")

            print(
                f"[saved]   {benchmark_name}: "
                f"{actual_qubits} qubits, depth={circuit.depth()}, "
                f"gates={circuit.size()}"
            )

            manifest_rows.append(
                {
                    "benchmark": benchmark_name,
                    "requested_qubits": requested_qubits,
                    "actual_qubits": actual_qubits,
                    "depth": circuit.depth(),
                    "gates": circuit.size(),
                    "status": "saved",
                    "path": str(filename),
                    "error": "",
                }
            )

        except Exception as error:
            print(
                f"[skipped] {benchmark_name}: "
                f"{type(error).__name__}: {error}"
            )

            manifest_rows.append(
                {
                    "benchmark": benchmark_name,
                    "requested_qubits": requested_qubits,
                    "actual_qubits": "",
                    "depth": "",
                    "gates": "",
                    "status": "generation_failed",
                    "path": "",
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    manifest_path = output_path / "manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "benchmark",
                "requested_qubits",
                "actual_qubits",
                "depth",
                "gates",
                "status",
                "path",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nManifest saved to: {manifest_path}")
    print(f"Python:    {sys.version.split()[0]}")
    print(f"MQT Bench: {_package_version('mqt.bench')}")
    print(f"Qiskit:    {qiskit.__version__}")
    print(f"NumPy:     {np.__version__}")


def _package_version(package_name: str) -> str:
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return "unknown"


if __name__ == "__main__":
    N=8
    export_benchmarks(
        requested_qubits=N,
        output_dir="mqtbench_{}q".format(N),
        require_exact_size=True,
    )