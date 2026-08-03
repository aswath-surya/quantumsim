Create a new script that adapts the analysis in:

examples/run_bounding_renyi_focused.py

to a family of generated N-qubit quantum phase estimation circuits.

The new script should be named:

examples/run_qpe_renyi_scaling.py

Do not modify run_bounding_renyi_focused.py.

PRIMARY GOAL

Generate QPE circuits programmatically for multiple system sizes N, repeat each QPE circuit to several circuit depths, and reproduce the fixed-depth versus-N analysis of the Rényi-2 quantity.

The main quantity of interest is the unclipped Rényi factor

S2(N, d)
    = 0.5 * sqrt(
        2^m * sum_x p_{N,d}(x)^2 - 1
      ),

where:

- N is the total number of qubits in the generated QPE circuit;
- d is the repetition depth;
- m is the number of measured qubits;
- p_{N,d}(x) is the ideal measured-register output distribution of the N-qubit QPE circuit repeated d times.

The script should generate plots of S2 versus N at each fixed repetition depth, analogous to the existing renyi_factors_vs_N_depth{d}.png figures in run_bounding_renyi_focused.py.

GENERAL DESIGN

The script should combine:

1. programmatic generation of QPE circuits for arbitrary N;
2. the repeated-circuit depth semantics used in the QASM/QPE analysis;
3. the ideal-distribution, collision-entropy, exact-white-noise, and Rényi-factor calculations from run_bounding_renyi_focused.py;
4. optional RC/CB/QCAP analysis at each N and depth;
5. fixed-depth fits versus N;
6. finite-N Haar / Porter-Thomas comparisons.

The script should not require the user to upload a QASM file.

1. QPE CIRCUIT GENERATION

Implement a helper:

build_qpe_circuit(
    n_counting_qubits: int,
    n_system_qubits: int = 1,
    phase: float = ...,
    measured_register: str = "counting",
    inverse_qft: bool = True,
)

The default circuit should use:

- n_counting_qubits counting qubits;
- one system/eigenstate qubit;
- total qubits:

      N = n_counting_qubits + n_system_qubits;

- Hadamards on all counting qubits;
- controlled powers of a unitary U acting on the system register;
- an inverse QFT on the counting register;
- measurement of the counting register.

Use a simple exactly known eigenunitary such as:

U = phase gate P(2 * pi * phase)

acting on a system qubit initialized in |1>.

For counting qubit k, apply the controlled power:

controlled-U^(2^k)

using either:

- a controlled phase gate with angle 2 * pi * phase * 2^k;
- or an equivalent native decomposition already supported by proxysim.

The phase should be configurable.

Choose a default phase that is not exactly representable by every counting-register size, so the ideal QPE distribution has nontrivial finite-width structure. For example:

QPE_PHASE = 0.3141592653589793

Do not choose a phase such as 1/4 that becomes deterministic for many values of N unless a separate deterministic test mode is provided.

2. DEFINITIONS OF N

Be explicit throughout the code about three sizes:

n_counting
n_system
n_total

Use:

n_total = n_counting + n_system

The x-axis in the main scaling plots should be configurable.

Default:

N_AXIS = "counting"

so that the plotted N is the number of measured counting qubits.

Support:

N_AXIS = "total"

as an alternative.

All collision and Rényi formulas must use the measured dimension:

D = 2^m

where:

m = len(measured_qubits)

Do not automatically use 2^n_total unless all qubits are measured.

3. SIZE SWEEP

Add configuration such as:

N_COUNTING_SCALING = [2, 3, 4, 5, 6, 7, 8, 9, 10]
N_SYSTEM_QUBITS = 1

SCALING_DEPTHS = [1, 2, 4, 8, 16, 32]

For every counting-register size and every repetition depth:

1. generate the base QPE circuit;
2. construct:

       target = repeat(base_qpe_circuit, depth)

3. compute the exact ideal measured-register distribution;
4. compute all Rényi and collision-entropy quantities;
5. optionally run RC/CB/QCAP and noisy-TVD analysis.

Do not reuse the ideal distribution from depth 1 at other depths.

The repeated target U_QPE^d may have a different ideal output distribution at every depth.

4. IDEAL DISTRIBUTION

Use the cheapest exact backend compatible with each generated circuit:

- stabilizer backend for Clifford circuits;
- statevector backend otherwise.

Implement or reuse helpers for:

- exact ideal distribution;
- dense probability-vector conversion;
- measured-register marginalization;
- normalization checks.

The measured distribution p must contain all 2^m outcomes, including outcomes absent from a sparse distribution dictionary.

5. RÉNYI AND COLLISION-ENTROPY QUANTITIES

For every (depth, N) compute:

uniform_tvd
collision_probability
renyi2_divergence
renyi_sqrt_factor_unclipped
renyi_sqrt_factor_clipped
collision_entropy
collision_entropy_density
collision_entropy_deficit

Use:

u = 1 / D

uniform_tvd
    = 0.5 * sum_x abs(p(x) - u)

collision_probability
    = sum_x p(x)^2

renyi2_divergence
    = log(D * collision_probability)

renyi_sqrt_factor_unclipped
    = 0.5 * sqrt(
        max(D * collision_probability - 1, 0)
      )

renyi_sqrt_factor_clipped
    = min(
        1,
        renyi_sqrt_factor_unclipped
      )

collision_entropy
    = -log2(collision_probability)

collision_entropy_deficit
    = m - collision_entropy
    = log2(D * collision_probability)

collision_entropy_density
    = collision_entropy / m

Use the term “Rényi factor” in plot labels and output filenames.

6. PRIMARY OUTPUT: S2 VERSUS N AT FIXED DEPTH

For every value in SCALING_DEPTHS, generate one figure:

results/qpe_renyi_factor_vs_N_depth{d}.png

Each figure should plot:

- unclipped Rényi factor S2 versus N;
- exact factor D_TV(p, u) versus N;
- finite-N Haar prediction for S2;
- the exponential fit used in run_bounding_renyi_focused.py;
- the collision-entropy-based fit;
- the Haar-crossover fit.

Use separate curves and clearly label each model.

The measured data should be exact deterministic ideal values for each generated QPE circuit. Do not show “instance standard deviation” unless an explicit ensemble of different phases or circuit instances is added.

7. FIT 1: EXPONENTIAL RÉNYI-FACTOR FIT

At every fixed depth, fit:

S2(N) = A * 2^(alpha * N)

Use the same fit logic as run_bounding_renyi_focused.py.

Report:

A
A_err
alpha
alpha_err
R2
R2_log

Interpret this only as a phenomenological finite-N fit.

Do not claim that it is asymptotically correct if the data flatten.

8. FIT 2: COLLISION-ENTROPY-DEFICIT FIT

This should be the preferred information-theoretic fit.

Use the exact identity:

Delta2
    = log2(1 + 4 * S2^2)
    = m - H2

Fit:

Delta2(N) = a * N + c

Then reconstruct:

S2_fit(N)
    = 0.5 * sqrt(
        2^(a * N + c) - 1
      )

Report:

a
a_err
c
c_err
R2
h2_density = 1 - a

If the x-axis is total-qubit count but only the counting register is measured, do not interpret 1-a as the measured collision-entropy density without accounting for the relation between total N and measured m.

Prefer using measured-register size for entropy-density interpretations.

9. FIT 3: HAAR / PORTER-THOMAS PREDICTION

For measured dimension D = 2^m, use:

haar_collision_probability
    = 2 / (D + 1)

haar_renyi_factor
    = 0.5 * sqrt(
        (D - 1) / (D + 1)
      )

This approaches 1/2 at large D.

Also fit a finite-N Haar-crossover model:

S2(N)
    = S2_Haar(N) + A * exp(-b * N)

Allow A to be positive or negative.

Require b >= 0.

Report:

A
A_err
b
b_err
R2
fixed_haar_R2

The fixed Haar curve has no fitted parameters.

10. MODEL COMPARISON

At every depth compare:

- exponential model;
- entropy-deficit model;
- Haar-crossover model;
- fixed finite-N Haar prediction.

Compute:

R2
weighted_R2 if uncertainties are available
RMSE
AIC
AICc where defined

Because these are deterministic ideal values, do not invent SEM weights.

Use unweighted fitting by default unless an explicit ensemble over phases is enabled.

Add an optional leave-largest-N-out test:

- fit using all but the largest one or two N values;
- predict the held-out values;
- report test RMSE for each model.

This is important because in-sample R2 is not sufficient for judging large-N predictability.

11. OPTIONAL PHASE ENSEMBLE

Add an optional mode:

PHASE_ENSEMBLE = False

If enabled, use a configurable set or sample of phases:

N_PHASE_INSTANCES = 40
PHASE_SEED = ...
PHASE_RANGE = (0, 1)

For every (N, depth), generate multiple QPE circuits with independently sampled phases.

Then compute:

- mean Rényi factor;
- instance standard deviation;
- SEM;
- percentile bands.

Only in this mode should the figures use error bars across circuit instances.

Keep the default mode deterministic with one fixed phase.

12. OPTIONAL RC / CB / QCAP ANALYSIS

Add a configuration switch:

RUN_NOISY_ANALYSIS = True

If enabled, preserve the RC/CB/QCAP analysis conventions from the QPE bounding code.

For each N:

1. identify distinct two-qubit cycles in the generated QPE circuit;
2. benchmark each distinct cycle;
3. measure readout fidelity;
4. count cycle occurrences at each repeated depth;
5. compute raw QCAP epsilon;
6. simulate explicit randomized compiling;
7. compute measured TVD from the exact ideal measured distribution.

For every (N, depth), compute:

qcap_bound
qcap_bound_std
measured_tvd_points
trajectory_floor
exact_white_noise_prediction
renyi_bound

where:

exact_white_noise_prediction
    = qcap_bound * uniform_tvd

renyi_bound
    = qcap_bound * renyi_sqrt_factor_clipped

Preserve:

exact_white_noise_prediction
    <= renyi_bound
    <= qcap_bound

within numerical tolerance.

13. WHITE-NOISE MODEL DIAGNOSTIC

If RUN_NOISY_ANALYSIS is enabled, test the full output-distribution model.

For every explicit-RC noisy distribution q, compute:

q_wn_qcap
    = (1 - epsilon_qcap) * p
      + epsilon_qcap * u

white_noise_model_residual_qcap
    = TVD(q, q_wn_qcap)

Also fit:

epsilon_white_noise_fit
    = argmin over epsilon in [0, 1]
      TVD(
          q,
          (1 - epsilon) * p + epsilon * u
      )

Store:

epsilon_white_noise_fit
white_noise_model_residual_fit
white_noise_model_residual_qcap

This allows the script to compare Rényi scaling with the actual quality of the white-noise output model.

14. REQUIRED FIGURES

A. One Rényi-factor-versus-N figure per depth:

qpe_renyi_factor_vs_N_depth{d}.png

Include:

- S2 data;
- D_TV(p,u);
- exponential fit;
- entropy-deficit reconstruction;
- finite-N Haar prediction;
- Haar-crossover fit.

B. One collision-entropy figure per depth:

qpe_collision_entropy_vs_N_depth{d}.png

Include:

- H2;
- H2 / m;
- Delta2 = m - H2;
- linear fit of Delta2;
- Haar prediction H2 approximately log2((D+1)/2).

C. One model-comparison summary:

qpe_renyi_model_comparison_vs_depth.png

Plot versus depth:

- exponential alpha;
- entropy-deficit slope a;
- Haar-crossover rate b;
- R2 or AICc difference between models;
- held-out prediction RMSE.

D. One all-depth summary:

qpe_renyi_factor_vs_N_all_depths.png

Use one curve per depth.

E. If noisy analysis is enabled:

qpe_tvd_exact_renyi_vs_N_depth{d}.png

At each fixed depth show versus N:

- measured TVD;
- raw QCAP;
- exact white-noise prediction;
- Rényi bound.

F. If noisy analysis is enabled:

qpe_white_noise_residual_vs_N_depth{d}.png

Show:

- mean QCAP-model residual;
- mean best-fit residual;
- fitted epsilon;
- QCAP epsilon.

15. PORTER-THOMAS DIAGNOSTICS

For every (N, depth), compute:

z = D * p(x)

and:

mean_scaled_probability
variance_scaled_probability
porter_thomas_ks_statistic
porter_thomas_ks_pvalue
collision_ratio_to_haar
renyi_factor_minus_haar

For selected depths, produce:

qpe_porter_thomas_summary_depth{d}.png

The figure should show versus N:

- collision ratio to Haar;
- Rényi factor minus Haar;
- KS statistic against an exponential distribution;
- variance of z, whose Porter-Thomas prediction is approximately 1.

Do not describe the QPE circuit as Haar-random merely because one statistic is near the Haar value.

16. DATA SHAPES

Use consistent array shapes.

In deterministic fixed-phase mode:

ideal arrays:
    shape = (n_depths, n_sizes)

If noisy analysis is enabled:

measured_tvd_points:
    shape = (n_depths, n_sizes, n_replicates)

epsilon_white_noise_fit:
    shape = (n_depths, n_sizes, n_replicates)

white_noise_model_residual_qcap:
    shape = (n_depths, n_sizes, n_replicates)

In phase-ensemble mode:

ideal factor arrays:
    shape = (n_depths, n_sizes, n_phase_instances)

Do not collapse axes ambiguously.

17. SAVED OUTPUT

Save:

results/qpe_renyi_scaling_data.npz

Include at least:

n_counting_values
n_total_values
n_system_qubits
depths
phase
measured_qubit_counts
measured_dimensions

ideal_uniform_tvd
ideal_collision_probability
renyi2_divergence
renyi_sqrt_factor_unclipped
renyi_sqrt_factor_clipped
collision_entropy
collision_entropy_density
collision_entropy_deficit

haar_collision_probability
haar_renyi_factor
collision_ratio_to_haar
renyi_factor_minus_haar

porter_thomas_ks_statistic
porter_thomas_ks_pvalue
mean_scaled_probability
variance_scaled_probability

All fit parameters and fit-quality metrics should also be saved with clear names containing the corresponding depth.

If noisy analysis is enabled, also save:

qcap_bound
qcap_bound_std
measured_tvd_points
trajectory_floor
exact_white_noise_prediction
renyi_bound
epsilon_white_noise_fit
white_noise_model_residual_qcap
white_noise_model_residual_fit

18. VALIDATION TESTS

Add deterministic self-tests.

Uniform distribution:

uniform_tvd == 0
renyi_sqrt_factor_unclipped == 0
collision_entropy == m

Deterministic distribution:

uniform_tvd == 1 - 1/D
renyi_sqrt_factor_unclipped == 0.5 * sqrt(D - 1)
collision_entropy == 0

Haar finite-D consistency:

haar_renyi_factor
    == 0.5 * sqrt((D - 1)/(D + 1))

For every generated ideal distribution:

sum(p) == 1

For every bound:

0 <= exact_white_noise_prediction <= renyi_bound <= qcap_bound <= 1

within tolerance.

19. RESOURCE HANDLING

Before simulating an N value, estimate dense-state memory usage.

Skip sizes exceeding:

- the exact backend qubit limit;
- SCALING_MEM_BUDGET_GB.

Print a warning for skipped sizes.

Do not allocate all ideal probability vectors for all N and depths simultaneously unless required.

Store scalar diagnostics and release large statevectors after each point.

20. TERMINOLOGY

Use:

- “Rényi factor” for S2;
- “counting-register size” for measured QPE width;
- “total circuit qubits” for counting plus system qubits;
- “repetition depth” for repeated applications of the complete QPE circuit;
- “finite-N Haar prediction” for the Porter-Thomas collision benchmark;
- “exact under the global white-noise model” for epsilon * D_TV(p,u).

Do not call the Haar-crossover model a theorem.

Do not call the QPE circuit Haar-random based only on S2 approaching 1/2.

21. IMPLEMENTATION REQUIREMENTS

Before writing code:

1. Read run_bounding_renyi_focused.py completely.
2. Read the QPE/QASM bounding script completely for its RC/CB implementation.
3. Identify which helpers can be reused directly.
4. Preserve the mathematical definitions from the Rényi script.
5. Replace QASM loading with programmatic QPE generation.
6. Preserve measured-register marginalization and repeated-circuit semantics.
7. Keep deterministic ideal-factor values separate from noisy replicate scatter.
8. Do not silently alter the meaning of N or depth.

After implementation:

1. Run:

python -m py_compile examples/run_qpe_renyi_scaling.py

2. Run a reduced smoke test with:

N_COUNTING_SCALING = [2, 3]
SCALING_DEPTHS = [1, 2]
RUN_NOISY_ANALYSIS = False

3. Run a second reduced noisy smoke test if noisy analysis is implemented:

N_COUNTING_SCALING = [2]
SCALING_DEPTHS = [1]
N_INSTANCES = 2
N_TRAJ = 5
CB_DECAYS = 2
CB_SHOTS = 100
FLOOR_PAIRS = 1

4. Restore intended defaults.
5. Report:
   - whether py_compile passed;
   - whether both smoke tests passed;
   - all generated files;
   - all skipped N values;
   - any unsupported gates in the QPE RC/CB decomposition;
   - any differences from the source Rényi analysis.

The central deliverable is a clean fixed-depth N sweep of:

S2(N, d)
    = 0.5 * sqrt(
        2^m * sum_x p_{N,d}(x)^2 - 1
      )

for programmatically generated QPE circuits, together with exponential, collision-entropy, and Haar-crossover comparisons.