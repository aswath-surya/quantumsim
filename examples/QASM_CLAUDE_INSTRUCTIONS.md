claude --permission-mode acceptEdits "$(cat <<'PROMPT'
Work inside the current repository.

NON-NEGOTIABLE FILE-SAFETY RULES
- You may read and inspect any existing repository file.
- Do not modify, overwrite, delete, rename, move, reformat, or stage any file that existed when this session began.
- Do not edit the original 2qubit_CB notebook or any QFT/QPE source file.
- Create exactly one repository deliverable:
    2qubit_CB_qasm_cycles.ipynb
- Copy all code needed for the adaptation into that new notebook rather than changing existing modules.
- Temporary files may be created only under /tmp/claude-qasm-cb and must be removed before finishing.
- Do not run git commit, git add, git checkout, git reset, git restore, git clean, or commands that alter repository history or existing files.
- At the beginning, record:
    1. `git status --porcelain`
    2. all existing repository paths
    3. the hash or checksum of every pre-existing file that could be touched
- At the end, verify:
    1. no pre-existing file changed
    2. no pre-existing file was deleted, renamed, or moved
    3. the only new repository file is:
         2qubit_CB_qasm_cycles.ipynb
- If completing the task would require changing an existing file, do not make that change. Keep the implementation self-contained in the new notebook and document the limitation.

SOURCE NOTEBOOK INSPECTION
1. Locate the executable `.ipynb` source corresponding to `2qubit_CB.ipynb`.
2. Do not treat a PDF export as the editable source when an `.ipynb` exists.
3. Inspect every Markdown and code cell in the source notebook.
4. Before implementing anything, build an internal source-notebook inventory containing:
   - section headings
   - analysis cells
   - numerical calculations
   - printed summaries
   - tables
   - validation checks
   - plotting functions
   - every cell that displays a figure
   - every figure title
   - every figure's x-axis and y-axis quantities
   - every parameter sweep used by a figure
5. Use this inventory as a completion checklist.

QFT/QPE REPOSITORY INSPECTION
Locate the repository's QFT and QPE implementations and identify exactly how they:
- load or construct algorithm circuits
- remove or defer measurements
- normalize and transpile two-qubit gates
- define one complete algorithm cycle
- repeat complete algorithm cycles
- dress or randomized-compile two-qubit gates
- propagate corrections or Pauli frames
- handle controlled-phase gates
- preserve qubit ordering and endianness

Treat the QFT/QPE implementation as the source of truth for cycle construction. Add a concise Markdown section to the new notebook listing the relevant repository paths, functions, and conventions.

CREATE THE COPY
Create:

    2qubit_CB_qasm_cycles.ipynb

as a copy and adaptation of the original executable notebook.

Do not construct a minimal replacement notebook. Preserve the original notebook's pedagogical organization, explanations, analyses, diagnostics, validation cells, summaries, and figures.

MAIN OBJECTIVE
Modify the copied notebook so that it:
1. reads an algorithm circuit from a QASM file
2. identifies and dresses every supported two-qubit gate
3. treats one complete execution of the normalized QASM algorithm body as one algorithm cycle
4. constructs circuits containing m repeated algorithm cycles
5. uses fresh randomized dressing for every two-qubit-gate occurrence in every repeated cycle
6. propagates correction frames across gates and cycle boundaries
7. repeats all analyses from the original notebook using algorithm-cycle count m as the depth variable

CONFIGURATION
Include a configuration cell containing at least:

    QASM_PATH = "path/to/circuit.qasm"
    QASM_VERSION = "auto"
    CYCLE_DEPTHS = [1, 2, 4, 8, 16, 32]
    N_RC = ...
    SHOTS_PER_RC = ...
    TOTAL_SHOTS = N_RC * SHOTS_PER_RC
    CB_DEPTHS = [1, 2, 4, 8, 16, 32]
    CB_N_RC = ...
    CB_SHOTS = ...
    CB_N_PAULIS = ...
    CB_EXHAUSTIVE_MAX_QUBITS = ...
    SEED = 42
    FAST_MODE = False

`FAST_MODE=False` must run the complete analysis and generate every adapted figure. A fast smoke-test mode may exist, but it must not be the default saved configuration.

Infer N_QUBITS from the loaded QASM circuit. Do not hard-code it to two.

QASM LOADING AND NORMALIZATION
Implement a function such as:

    def load_algorithm_cycle(qasm_path, qasm_version="auto"):
        ...

It must:
- load QASM 2
- load QASM 3 when supported by the installed Qiskit version
- validate that the circuit contains quantum operations
- separate terminal measurements from the unitary algorithm body
- reject or explicitly handle:
    - mid-circuit measurements
    - resets
    - classical conditions
    - classical control flow
    - delay or calibration instructions that cannot be safely normalized
- preserve qubit ordering
- preserve classical-bit interpretation where relevant
- use the same native basis and transpilation conventions as the repository's QFT/QPE code
- return a measurement-free QuantumCircuit representing one complete algorithm cycle
- print the original and normalized gate counts
- print the normalized two-qubit gate types and their parameters

Only add measurement instructions after all repeated cycles and any final measurement-basis rotations.

TWO-QUBIT-GATE DRESSING
Generalize the original CZ-specific randomized-compiling implementation.

Implement an interface such as:

    def dress_algorithm_cycle(
        cycle,
        rng,
        pauli_frame=None,
        return_metadata=False,
    ):
        ...

Requirements:
1. Traverse instructions in circuit order.
2. Maintain an explicit pending frame for every qubit.
3. For every supported two-qubit gate on qubits `(q0, q1)`:
   - sample an independent product Pauli
   - absorb or insert the pre-dressing operation
   - compute the exact post-correction by conjugating through the ideal gate
   - update the pending frame
   - record gate name, parameters, qubits, location, sampled Pauli, and correction
4. Derive Clifford Pauli-conjugation rules programmatically.
5. Do not use a manually typed CZ-only table.
6. Merge pending Pauli corrections into adjacent one-qubit operations whenever possible.
7. Do not add extra noisy one-qubit pulses merely because randomized compiling is enabled.
8. Carry unresolved corrections across algorithm-cycle boundaries.
9. Apply a final correction only once, or absorb it into final measurement rotations.
10. Preserve the pulse-count/noise-budget logic of the original notebook.

NON-CLIFFORD SAFETY
Do not assume that arbitrary two-qubit gates map Paulis to Paulis.

QFT and QPE circuits may contain controlled-phase rotations that are not Clifford gates.

Therefore:
- first normalize using the QFT/QPE repository transpilation path
- dress only gates for which the repository's randomized-compiling method is mathematically valid
- use the repository's exact method for supported non-Clifford controlled-phase gates
- otherwise raise an informative exception including:
    - gate name
    - gate parameters
    - qubits
    - instruction index
    - cycle index
- do not silently approximate a non-Clifford conjugated correction by a Pauli
- do not silently replace a controlled-phase gate by CZ
- do not silently discard unsupported operations

REPEATED ALGORITHM CYCLES
Implement something similar to:

    def build_repeated_algorithm(
        base_cycle,
        repetitions,
        rng=None,
        measure=True,
        return_metadata=False,
    ):
        ...

For repetitions = m:
- append exactly m complete copies of the normalized algorithm cycle
- independently dress every two-qubit-gate occurrence
- propagate pending frames across cycle boundaries
- apply the final frame only once
- append measurements only at the end
- return metadata that identifies the cycle number and within-cycle gate location of every dressed operation

The undressed circuit must agree with:

    base_cycle.power(m)

up to global phase and explicitly documented transpilation equivalences.

All plots and printed summaries must state clearly that depth m is the number of complete algorithm cycles, not the number of elementary gates.

CYCLE BENCHMARKING
Refactor the original cycle-benchmarking analysis so that it benchmarks one complete normalized QASM algorithm cycle rather than a bare CZ or a random brickwork round.

Preserve the original logic where mathematically valid:
- prepare a Pauli eigenstate
- apply m independently randomized dressed algorithm cycles
- track the ideal evolved observable with its sign
- measure the appropriate final observable
- average over randomizations and sampled Paulis
- fit:

      survival(m) = A * f**m

- convert using:

      e_F = (1 - 4**(-n)) * (1 - f)

- compute:
    - f
    - uncertainty in f
    - e_F
    - uncertainty in e_F
    - entanglement fidelity
    - average gate infidelity
    - average gate fidelity

Do not call `Clifford(base_cycle)` unless the complete normalized cycle is Clifford.

If the complete cycle is Clifford:
- track the signed Pauli image under the full repeated cycle exactly.

If the complete cycle is non-Clifford:
- inspect and follow the benchmarking estimator used by the repository's QFT/QPE implementation.
- If the original Pauli-observable CB estimator is mathematically inapplicable, state that explicitly.
- Do not force an invalid Clifford or Pauli calculation.
- Preserve the corresponding analysis section and figure position, and display a clear diagnostic explaining which quantity is unavailable and why.

For larger n:
- do not enumerate all 4**n - 1 nonidentity Paulis
- use deterministic Monte Carlo Pauli sampling controlled by CB_N_PAULIS
- retain exhaustive enumeration as a validation option for small n

MANDATORY ANALYSIS AND FIGURE PARITY
The new notebook must repeat every analysis and reproduce an adapted counterpart of every figure in the original notebook.

Do not omit a figure because the circuit source changed from brickwork to QASM.

For each original analysis or figure:
- preserve its scientific purpose
- preserve its metric
- preserve its sweep variable where meaningful
- preserve comparable labels, legends, uncertainty bars, reference curves, and annotations
- replace brickwork depth with complete algorithm-cycle count where appropriate
- replace CZ-cycle quantities with complete algorithm-cycle quantities
- adapt hard-coded two-qubit dimensions to N_QUBITS
- adapt two-circuit-family comparisons to the loaded QASM algorithm where necessary

At minimum, preserve and generalize all original analyses involving:
- noise-model construction and parameter reporting
- coherent and stochastic two-qubit errors
- readout confusion matrix
- readout fidelity and readout standard deviation
- randomized-compiling checks
- merged-gate validation
- twirl-conjugation validation
- operation-count and noise-budget checks
- ideal output distributions
- noisy output distributions
- RC-averaged output distributions
- finite-shot TVD floor
- total variation distance
- Hellinger or classical fidelity
- Shannon entropy in bits
- cycle-benchmarking survival decay
- exponential CB fits
- per-Pauli or sampled-Pauli CB diagnostics
- exact or analytic process-polarization comparisons when available
- exact-versus-fitted e_F checks when available
- Pauli-spectrum or orbit-averaged diagnostic plots
- QCAP calculations
- QCAP bound versus algorithm-cycle depth
- measured TVD versus QCAP bound
- measured fidelity versus depth
- scatter and error-bar plots across circuit/randomization instances
- entropy-versus-bound-looseness analysis
- randomization statistics
- mean, standard deviation, and SEM calculations
- analytic white-noise or depolarizing predictions present in the source notebook
- Rényi-related analyses or bounds present in the source notebook
- optional package/proxysim cross-checks present in the source notebook
- final numerical summaries

FIGURE-PARITY RULES
- Count the figures produced by the original notebook.
- Count the figures produced by the new notebook.
- The new notebook must contain one adapted figure slot for every original figure.
- Do not merge several original figures into one figure unless the original notebook already did so.
- Do not replace plots with only printed numbers.
- Do not comment out expensive plotting cells.
- Do not leave plotting cells unexecuted.
- Preserve uncertainty bars wherever the original uses them.
- Preserve logarithmic or linear axis choices.
- Preserve reference lines and theoretical curves.
- Update titles and axes so they say "complete algorithm cycles" where appropriate.
- Where an exact analytic curve is unavailable for an arbitrary QASM algorithm, retain the figure and:
    1. plot all empirically available quantities
    2. label the unavailable analytic comparator explicitly
    3. include a nearby Markdown explanation
- Where an original figure compares random and structured circuit families, adapt it into a comparison meaningful for the loaded QASM algorithm, such as:
    - independently randomized dressings
    - different QASM parameter instances when available
    - dressed versus undressed/noisy variants
  Preserve the underlying metric and explain the adaptation.
- Do not silently drop a figure due to non-Clifford behavior.

Add a final "Analysis and figure parity report" cell containing a table with columns:

    Original section
    Original analysis or figure
    Adapted section
    Adapted analysis or figure
    Status
    Notes

Every row must have status:
- reproduced
- generalized
- unavailable for a stated mathematical reason

No row may be absent or silently skipped.

PRESERVE THE ORIGINAL ANALYSIS FLOW
Keep the original notebook's progression as closely as possible:
1. background and configuration
2. noise model
3. readout characterization
4. randomized compiling
5. metrics
6. cycle benchmarking
7. analytic or exact checks
8. QCAP calculation
9. circuit or algorithm construction
10. depth sweeps
11. output-distribution comparisons
12. entropy and looseness diagnostics
13. optional package cross-checks
14. final summary and parity report

Do not replace this with a short API demonstration.

REQUIRED VALIDATION CELLS
Add executable assertions for:

1. QASM round-trip
   - normalized measurement-free cycle agrees with the original unitary QASM body.

2. Dressed ideal equivalence
   - several random dressed cycles agree with the undressed ideal cycle up to global phase.

3. Repetition equivalence
   - for small m, the undressed repeated builder agrees with base_cycle.power(m).

4. Frame continuity
   - pending corrections propagate correctly across adjacent gates and cycle boundaries.

5. Gate-count/noise-budget equivalence
   Report and compare:
   - algorithm-cycle count
   - two-qubit gates per cycle
   - dressed two-qubit occurrences
   - physical one-qubit operation count
   - physical two-qubit operation count
   - measurement count

6. Reproducibility
   - identical seeds produce identical dressed circuits, metadata, and numerical results.

7. Non-Clifford safety
   - unsupported non-Clifford two-qubit gates raise an informative error.

8. Output normalization
   - all ideal and noisy distributions sum to one within tolerance.

9. Metric ranges
   - TVD is in [0, 1]
   - fidelities are in [0, 1]
   - entropy is in [0, N_QUBITS]

10. Figure completeness
    - the parity table includes every source figure.
    - every expected adapted plotting cell executes successfully.

11. Repository preservation
    - no pre-existing repository path has changed.

PERFORMANCE
- Keep the complete analysis as the default.
- Use batching where possible.
- Avoid rebuilding identical transpiled circuits unnecessarily.
- Cache gate conjugation and merged-unitary calculations.
- Use deterministic seeds.
- Do not reduce sample counts solely to make the notebook finish quickly unless FAST_MODE=True.
- Clearly print estimated circuit and shot counts before running expensive sweeps.

FINAL NOTEBOOK OUTPUT
At the end, print:

    QASM file:
    QASM version:
    Number of qubits:
    Normalized native basis:
    Complete algorithm cycle definition:
    Two-qubit gate types:
    Two-qubit gates per algorithm cycle:
    Cycle depths:
    RC randomizations:
    Shots per randomization:
    CB Pauli sampling mode:
    Number of sampled Paulis:
    Fitted f:
    Estimated e_F:
    Estimated F_avg:
    Number of original analyses:
    Number of adapted analyses:
    Number of original figures:
    Number of adapted figure slots:
    Figure-parity status:
    Repository-preservation status:

EXECUTION REQUIREMENT
- Run the entire new notebook from a clean kernel with FAST_MODE=False.
- Save the executed cell outputs and all figures in the notebook.
- Confirm there are no cell errors.
- Confirm every adapted figure was generated.
- Confirm the parity report is complete.
- Confirm no existing repository code changed.

FINAL RESPONSE
Return:
1. the path to `2qubit_CB_qasm_cycles.ipynb`
2. a concise implementation summary
3. the number of original and adapted analyses
4. the number of original and adapted figures
5. any mathematical limitations
6. final `git status --porcelain`

Do not create a separate summary file.
PROMPT
)"