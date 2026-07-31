"""Correctness checks for the RC/CB QASM bank produced by ``build_qasm_bank.py``.

The bank is consumed by an external runner, so the failure mode that matters is a file
that *looks* fine and quietly encodes the wrong circuit. Each check below pins down one
property that would otherwise fail silently:

  1. round-trip      -- every emitted file re-parses, gate for gate.
  2. RC is faithful  -- a twirled circuit implements the same unitary as the original.
                        This is the one non-negotiable property of randomized compiling.
  3. RC twirls       -- randomizations actually differ, and cp(theta) draws stay in the
                        commuting Z-subgroup (an X/Y draw there would silently change
                        the circuit -- caught by check 2, pinned down here).
  4. RC does its job -- under COHERENT noise, the twirled circuit reproduces
                        NoiseModel.twirled() and the untwirled one does not. This is the
                        reason RC exists, and the condition under which a QCAP bound
                        built from cycle_benchmark applies to the circuit at all.
  5. CB analyzable   -- noiseless CB survival is exactly +1, i.e. the recorded support
                        and sign really do describe the emitted circuit.
  6. CB recovers e_F -- shot data from the emitted CB files, pushed through analyze_cb,
                        matches an in-process cycle_benchmark on the same noise model.

Run:  python examples/verify_qasm_bank.py
"""

from __future__ import annotations

import random
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from qiskit.quantum_info import Statevector

import _bootstrap  # noqa: F401
from proxysim import cb_emit, mqtbank
from proxysim.backends import StatevectorBackend
from proxysim.benchmarking import cycle_benchmark
from proxysim.circuit import circuit_from_qasm
from proxysim.noise import NoiseModel, sample_trajectory
from proxysim.qasm import circuit_to_qasm
from proxysim.rc import pauli_twirl, twirl_options, _DIAG_2Q, _FULL_2Q

_SV = StatevectorBackend()
ALGOS = mqtbank.ALGORITHMS
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _vec(circ):
    return np.asarray(Statevector(_SV._build(circ)).data)


def _same_state(u, v, tol=1e-9):
    """Equal up to a global phase -- the twirl's P' G P = +-G leaves a sign."""
    return abs(abs(np.vdot(u, v)) - 1.0) < tol


# ---------------------------------------------------------------------------
def test_roundtrip_and_rc_unitary():
    print("\n1+2. QASM round-trip and RC unitary equivalence (n<=8, d in {1,2})")
    worst = 0.0
    for algo in ALGOS:
        for n in (4, 6, 8):
            circ, measured = mqtbank.fetch(algo, n)
            for d in (1, 2):
                rep = mqtbank.repeat(circ, d)
                ref = _vec(rep)
                for r in range(5):
                    tw = pauli_twirl(rep, random.Random(1000 * d + r))
                    back, m2 = circuit_from_qasm(
                        circuit_to_qasm(tw, measured=measured))
                    if [repr(g) for g in back.gates] != [repr(g) for g in tw.gates]:
                        check(f"round-trip {algo} n={n} d={d} r={r}", False)
                        return
                    if m2 != list(measured):
                        check(f"measured qubits {algo} n={n}", False, f"{m2} != {measured}")
                        return
                    ov = abs(np.vdot(ref, _vec(back)))
                    worst = max(worst, abs(ov - 1.0))
    check("round-trip is gate-exact for all 6 algorithms x {4,6,8} x {1,2} x 5 rand", True)
    check("twirled unitary == original (up to global phase)", worst < 1e-9,
          f"worst |1-|<psi|psi'>|| = {worst:.2e}")


def test_twirl_is_random_and_restricted():
    print("\n3. Twirl randomizes, and cp(theta) stays in the commuting Z-subgroup")
    import math
    from proxysim.circuit import Gate

    ok = (twirl_options(Gate("cp", (0, 1), (math.pi / 4,)))[0] == _DIAG_2Q
          and twirl_options(Gate("cp", (0, 1), (math.pi,)))[0] == _FULL_2Q
          and all(twirl_options(Gate(g, (0, 1)))[0] == _FULL_2Q
                  for g in ("cx", "cz", "cy", "swap")))
    check("cp(theta) -> {I,Z}^2; cp(k*pi)/cx/cz/cy/swap -> full Pauli group", ok)

    circ, _ = mqtbank.fetch("qft", 8)
    variants = {tuple(repr(g) for g in pauli_twirl(circ, random.Random(r)).gates)
                for r in range(20)}
    check("20 randomizations are 20 distinct circuits", len(variants) == 20)

    tw = pauli_twirl(circ, random.Random(0))
    check("twirl overhead is modest after Pauli merging", len(tw.gates) < 3 * len(circ.gates),
          f"{len(circ.gates)} -> {len(tw.gates)} gates ({len(tw.gates)/len(circ.gates):.2f}x)")


def test_rc_against_coherent_noise():
    """The point of RC: it turns coherent error into the Pauli channel CB measures.

    Only ``theta_zz`` is switched on. That is deliberate, and the reason is worth
    recording: this twirl is *uncompiled*, so it roughly doubles the single-qubit gate
    count, and ``theta_1q`` attaches a coherent over-rotation to every single-qubit
    gate. A model with ``theta_1q`` would therefore charge the RC circuit for twice as
    much 1q error as the reference and the comparison would measure that, not the
    twirl. The twirl does NOT change the two-qubit gate count, so the 2q coherent
    channel isolates the effect cleanly.
    """
    print("\n4. Under COHERENT noise, RC reproduces NoiseModel.twirled()")
    coherent = NoiseModel(enabled=True, p1=0.0, p2=0.0, p_readout=0.0, p_idle=0.0,
                          theta_zz=0.35)
    twirled_ref = coherent.twirled()
    # ghz is built from cx only, so every gate gets the FULL Pauli twirl (no cp(theta)
    # Z-subgroup restriction) -- the case the closed-form twirled() model describes.
    circ, _ = mqtbank.fetch("ghz", 6)
    ideal = _vec(circ)

    def mean_infidelity(build, noise, n_traj, seed):
        rng = random.Random(seed)
        fid = 0.0
        for _ in range(n_traj):
            psi = _vec(sample_trajectory(build(rng), noise, rng))
            fid += abs(np.vdot(ideal, psi)) ** 2
        return 1.0 - fid / n_traj

    n_traj, seed = 1500, 5
    untwirled = mean_infidelity(lambda rng: circ, coherent, n_traj, seed)
    rc_coherent = mean_infidelity(lambda rng: pauli_twirl(circ, rng), coherent, n_traj, seed)
    reference = mean_infidelity(lambda rng: circ, twirled_ref, n_traj, seed)

    tol = 4.0 / np.sqrt(n_traj)          # Monte-Carlo error on the infidelity estimate
    print(f"      untwirled + coherent noise : {untwirled:.4f}   (error grows in amplitude)")
    print(f"      RC        + coherent noise : {rc_coherent:.4f}")
    print(f"      untwirled + twirled(model) : {reference:.4f}   <- the target")
    check("RC under coherent noise matches the twirled Pauli channel",
          abs(rc_coherent - reference) < tol,
          f"|diff| = {abs(rc_coherent - reference):.4f} < {tol:.4f}")
    check("untwirled circuit does NOT match it (so the test has teeth)",
          abs(untwirled - reference) > tol,
          f"|diff| = {abs(untwirled - reference):.4f} > {tol:.4f}")


def test_cb_metadata():
    print("\n5. Emitted CB circuits: noiseless survival is exactly +1")
    bad = 0
    for n in (4, 6, 8):
        for twoq in ("cz", "cx", "swap"):
            for d in (1, 2, 4, 8, 16, 32, 64, 128):
                c, meta = cb_emit.cb_circuit(n, [(0, 1)], d, random.Random(d * 31 + n),
                                             twoq=twoq)
                back, _ = circuit_from_qasm(circuit_to_qasm(c, measured=[]))
                s = cb_emit._to_stim(back)
                s.append("M", list(range(n)))
                bits = s.compile_sampler(seed=7).sample(128)
                if abs(cb_emit.survival(bits, meta) - 1.0) > 1e-12:
                    bad += 1
    check("survival == +1 for 3 widths x 3 entanglers x 8 depths (after QASM round-trip)",
          bad == 0, f"{bad} bad")


def test_cb_recovers_ef():
    print("\n6. e_F from emitted CB files == e_F from in-process cycle_benchmark")
    n, depths, n_rand, shots = 4, [1, 2, 4, 8, 16], 40, 4000
    noise = NoiseModel(enabled=True, p1=2e-3, p2=1e-2, p_readout=1e-2, p_idle=2e-3)
    from proxysim.noise import to_stim_noisy

    surv = {}
    rng = random.Random(0)
    for d in depths:
        vals = []
        for _ in range(n_rand):
            circ, meta = cb_emit.cb_circuit(n, [(0, 1)], d, rng, twoq="cz")
            back, _ = circuit_from_qasm(circuit_to_qasm(circ, measured=[]))
            noisy = to_stim_noisy(back, noise)
            noisy.append("X_ERROR", list(range(n)), noise.p_readout)
            noisy.append("M", list(range(n)))
            bits = noisy.compile_sampler(seed=rng.randint(0, 2**31 - 1)).sample(shots)
            vals.append(cb_emit.survival(bits, meta))
        surv[d] = vals

    emitted = cb_emit.analyze_cb(surv, n)
    inproc = cycle_benchmark([(0, 1)], n, depths, noise, n_decays=n_rand,
                             shots=shots, seed=1, twoq="CZ")
    diff = abs(emitted["e_F"] - inproc["e_F"])
    tol = 3.0 * (emitted["e_F_std"] + inproc["e_F_std"]) + 1e-3
    print(f"      emitted-bank e_F = {emitted['e_F']:.5f} +- {emitted['e_F_std']:.5f}")
    print(f"      in-process   e_F = {inproc['e_F']:.5f} +- {inproc['e_F_std']:.5f}")
    check("the two agree within combined error bars", diff < tol,
          f"|diff| = {diff:.5f} < {tol:.5f}")


def main():
    print("Verifying the RC/CB QASM bank generator")
    test_roundtrip_and_rc_unitary()
    test_twirl_is_random_and_restricted()
    test_rc_against_coherent_noise()
    test_cb_metadata()
    test_cb_recovers_ef()
    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
