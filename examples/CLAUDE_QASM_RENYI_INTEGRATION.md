Create a new script:

examples/run_qasm_bounding_renyi.py

Use these two files as the source implementations:

examples/run_qpe_bounding.py
examples/run_bounding_renyi_focused.py

The new script should combine:

- the QASM loading, cycle extraction, explicit randomized compiling, cycle benchmarking/QCAP calculation, repeated-circuit depth semantics, parallel trajectory simulation, and trajectory-noise-floor handling from run_qpe_bounding.py;
- the exact white-noise factor, Rényi-2 factor, collision-entropy analysis, Haar comparison, validation checks, data saving, and plotting logic from run_bounding_renyi_focused.py.

Do not modify either source file.

PRIMARY GOAL

Given one QASM circuit, evaluate at each repetition depth

U, U^2, U^4, ...

and produce:

1. measured explicit-RC TVD from the ideal output;
2. raw QCAP bound from CB and readout fidelity;
3. exact global-white-noise prediction;
4. Rényi-2 upper bound;
5. ideal output-distribution diagnostics;
6. tests of whether the ideal output probabilities become Porter-Thomas/Haar-like with depth.

Preserve the statistical interpretation of the QASM script: this is one fixed QASM circuit, not an ensemble of randomly generated ideal circuits. Repeated TVD values at a fixed depth come from independent RC and trajectory realizations, not from distinct ideal circuit instances.

1. CONFIGURATION

Use a configurable QASM path:

QASM = os.path.join(os.path.dirname(__file__), "qpe11.qasm")

Keep the runtime controls from run_qpe_bounding.py, including:

DEPTHS
N_INSTANCES
N_TRAJ
FLOOR_PAIRS
CB_DEPTHS
CB_SHOTS
CB_DECAYS
THETA_ZZ
SEED

Use output names derived from the QASM stem, for example:

results/qpe11_bounding_renyi.png
results/qpe11_bounding_renyi_data.npz
results/qpe11_renyi_factors_vs_depth.png
results/qpe11_output_statistics_vs_depth.png
results/qpe11_white_noise_diagnostics.png
results/qpe11_porter_thomas_diagnostics_depth{d}.png

Do not hard-code "qpe" into general helper functions.

2. QASM LOADING AND CIRCUIT SEMANTICS

Reuse the QASM parsing logic from run_qpe_bounding.py:

base_circuit, measured_qubits = circuit_from_qasm(...)

At every depth use:

target = repeat(base_circuit, depth)

The ideal distribution at each depth must be computed from that exact repeated target:

p_depth = ideal_distribution(target)

Do not reuse the depth-1 ideal distribution at all depths.

Marginalize the full ideal distribution onto measured_qubits exactly as in the QASM implementation.

Use the number of measured qubits, not necessarily target.n_qubits, in all output-distribution formulas.

3. QASM CYCLE BENCHMARKING

Preserve the CB/QCAP procedure from run_qpe_bounding.py.

Identify every distinct two-qubit cycle in the parsed QASM circuit, including the neighboring one-qubit gates that the existing QPE script merges into each cycle definition.

Benchmark each distinct cycle separately.

For every repeated target, count how often each distinct cycle occurs and call:

qcap_bound(cycle_counts, efs, readout_fid, readout_std)

Do not approximate the QASM circuit as generic brickwork layers.

Retain existing warnings and handling for QASM gate types whose explicit Pauli twirl is restricted. Distinguish fully twirled Clifford entanglers such as CX from restricted subgroup twirls such as CP(theta).

4. EXPLICIT RANDOMIZED COMPILING

Preserve the explicit-RC path from run_qpe_bounding.py.

For every depth and TVD replicate:

- generate a fresh randomized compilation;
- use an independent simulation seed;
- simulate under the original physical noise model;
- compare against the ideal distribution of the corresponding repeated target;
- retain every raw TVD point.

Use pmap as in run_qpe_bounding.py so trajectory jobs run in parallel.

Do not replace explicit RC with simulation under NOISE.twirled() unless that is included as a separately labeled comparison series.

If THETA_ZZ == 0, avoid redundantly running identical explicit-RC and already-Pauli-twirled series.

5. TRAJECTORY-ESTIMATOR FLOOR

Retain the trajectory-noise-floor analysis from run_qpe_bounding.py.

At every depth, estimate the Monte Carlo TVD floor from independent noisy ensemble pairs using FLOOR_PAIRS.

Keep this quantity separate from:

- CB/readout parameter uncertainty;
- RC/trajectory replicate scatter;
- variation of ideal-distribution quantities.

The ideal factors at one depth are deterministic because there is only one ideal repeated QASM circuit at that depth.

6. IDEAL-DISTRIBUTION QUANTITIES

Port the dense-probability and factor calculations from run_bounding_renyi_focused.py.

For each depth compute from the ideal measured-register distribution p:

uniform_tvd
collision_probability
renyi2_divergence
renyi_sqrt_factor_unclipped
renyi_sqrt_factor_clipped
collision_entropy_deficit
collision_entropy
collision_entropy_density

Let m be the number of measured qubits and D = 2^m.

Compute:

uniform_tvd
    = 0.5 * sum_x abs(p(x) - 1 / D)

collision_probability
    = sum_x p(x)^2

renyi2_divergence
    = log(D * collision_probability)

renyi_sqrt_factor_unclipped
    = 0.5 * sqrt(D * collision_probability - 1)

renyi_sqrt_factor_clipped
    = min(1, renyi_sqrt_factor_unclipped)

collision_entropy
    = -log2(collision_probability)

collision_entropy_deficit
    = m - collision_entropy
    = log2(D * collision_probability)

collision_entropy_density
    = collision_entropy / m

Clamp the square-root radicand at zero to avoid floating-point negatives for uniform distributions.

Use the terminology "Rényi factor" for the square-root quantity.

7. BOUNDS AND MODEL PREDICTIONS

For each depth let epsilon_qcap be the raw QCAP error parameter.

Compute:

exact_white_noise_prediction
    = epsilon_qcap * uniform_tvd

renyi_bound
    = epsilon_qcap * renyi_sqrt_factor_clipped

The exact quantity is exact only under the global white-noise mixture:

q = (1 - epsilon) p + epsilon u

where u is uniform over the measured outcomes.

Preserve assertions:

0 <= epsilon_qcap <= 1
0 <= exact_white_noise_prediction <= 1
0 <= renyi_bound <= 1
exact_white_noise_prediction <= renyi_bound + tolerance
renyi_bound <= epsilon_qcap + tolerance

Do not label exact_white_noise_prediction as a rigorous bound without explicitly qualifying it as model-dependent.

8. DIRECT WHITE-NOISE-MODEL TEST

For every noisy output distribution q from an explicit-RC replicate, test the full distributional model, not only its scalar TVD.

Construct the QCAP-predicted white-noise mixture:

q_wn_qcap
    = (1 - epsilon_qcap) * p + epsilon_qcap * u

Compute:

white_noise_model_residual_qcap
    = TVD(q, q_wn_qcap)

Also find the best-fit white-noise parameter:

epsilon_white_noise_fit
    = argmin over epsilon in [0, 1]
      TVD(q, (1 - epsilon) * p + epsilon * u)

Use scipy.optimize.minimize_scalar with bounds=(0, 1) and method="bounded", or a stable equivalent.

Store:

epsilon_white_noise_fit
white_noise_model_residual_fit
white_noise_model_residual_qcap

At each depth report and plot:

mean_white_noise_residual_qcap
std_white_noise_residual_qcap
mean_white_noise_residual_fit
std_white_noise_residual_fit
mean_fitted_epsilon
std_fitted_epsilon
qcap_minus_fitted_epsilon

This is the central test of whether RC plus circuit propagation produces an approximately global white-noise output mixture.

9. HAAR / PORTER-THOMAS DIAGNOSTICS

At every depth compute finite-D Haar predictions using D = 2^m:

haar_collision_probability
    = 2 / (D + 1)

haar_renyi_factor
    = 0.5 * sqrt((D - 1) / (D + 1))

Store:

haar_collision_probability
haar_renyi_factor
collision_ratio_to_haar
renyi_factor_minus_haar

where:

collision_ratio_to_haar
    = collision_probability / haar_collision_probability

renyi_factor_minus_haar
    = renyi_sqrt_factor_unclipped - haar_renyi_factor

Also compute scaled probabilities:

z = D * p

For Porter-Thomas output probabilities, z should approximately follow an exponential distribution with mean 1.

For every depth compute:

mean_scaled_probability
variance_scaled_probability
porter_thomas_ks_statistic
porter_thomas_ks_pvalue

Use scipy.stats.kstest against the unit-rate exponential distribution.

Label this as a Porter-Thomas/Haar-like output-probability diagnostic. Do not claim that the circuit unitary itself is Haar-random.

10. OPTIONAL ERROR-PROPAGATION / SCRAMBLING DIAGNOSTICS

Add per-depth diagnostics that distinguish these two claims:

A. Ideal output probabilities are Haar-like:
   - Rényi factor near the finite-D Haar value;
   - collision ratio near 1;
   - Porter-Thomas KS statistic small.

B. Noisy output is well modeled by global white noise:
   - residual_qcap small;
   - residual_fit small;
   - fitted epsilon close to QCAP epsilon.

Do not equate S2 approximately equal to 1/2 with a small white-noise residual. These test different properties.

11. REQUIRED FIGURES

A. Main bounding plot

Plot versus repeated-circuit depth:

- every explicit-RC measured TVD point;
- raw QCAP bound;
- QCAP CB/readout uncertainty;
- exact global-white-noise prediction;
- Rényi-2 bound;
- trajectory TVD floor.

Do not connect the raw TVD replicate points.

B. Rényi-factor and entropy diagnostics

Plot versus depth:

- unclipped Rényi factor S2;
- finite-D Haar Rényi factor;
- D_TV(p, u);
- collision entropy H2;
- collision entropy density H2 / m;
- collision entropy deficit m - H2.

There is one deterministic ideal value at each depth, so do not show circuit-instance error bars on these quantities.

C. White-noise-model diagnostic

Use separate panels for:

Panel 1:
- QCAP epsilon;
- mean fitted epsilon with replicate standard deviation.

Panel 2:
- mean residual using QCAP epsilon;
- mean best-fit residual;
- trajectory TVD floor if useful for context.

Do not place epsilon values and residual TVDs on one unlabeled axis.

D. Porter-Thomas histograms

For configurable depths such as:

PORTER_THOMAS_DEPTHS = [1, 4, 16]

write one histogram per selected depth of:

z = D * p(x)

Overlay the unit-rate exponential density:

f(z) = exp(-z)

Include the KS statistic, collision ratio to Haar, and Rényi factor in the title or annotation.

E. Haar-comparison summary

Plot versus depth:

collision_ratio_to_haar
renyi_factor_minus_haar
porter_thomas_ks_statistic

Use separate axes or panels if their scales differ substantially.

12. STATISTICAL TREATMENT

For one fixed QASM circuit:

- ideal factors are deterministic at each depth;
- QCAP uncertainty comes from cycle-benchmark and readout fits;
- measured TVD scatter comes from explicit-RC and trajectory sampling;
- fitted epsilon and white-noise residual scatter come from explicit-RC and trajectory sampling;
- the trajectory floor is a Monte Carlo estimator floor.

Do not reuse the random-circuit instance SD/SEM interpretation from run_bounding_renyi_focused.py.

Do not fit versus qubit count N unless the script is later extended to accept a family of QASM files at different widths.

13. SAVED NPZ SCHEMA

Save at least:

depths
cycle_depths
measured_qubits
n_measured_qubits

qcap_bound
qcap_bound_std

tvd_points
trajectory_floor

ideal_uniform_tvd
ideal_collision_probability
renyi2_divergence
renyi_sqrt_factor_unclipped
renyi_sqrt_factor_clipped
collision_entropy
collision_entropy_density
collision_entropy_deficit

exact_white_noise_prediction
renyi_bound

haar_collision_probability
haar_renyi_factor
collision_ratio_to_haar
renyi_factor_minus_haar
porter_thomas_ks_statistic
porter_thomas_ks_pvalue
mean_scaled_probability
variance_scaled_probability

epsilon_white_noise_fit
white_noise_model_residual_fit
white_noise_model_residual_qcap

Also save all major simulation settings, seeds, QASM path or stem, number of RC replicates, number of trajectories, CB settings, and noise-model parameters.

14. VALIDATION

Retain or add normalization checks for every probability vector.

Check that ideal and noisy distributions have the same measured-register dimension.

Add deterministic self-tests:

Uniform distribution:
uniform_tvd == 0
renyi_sqrt_factor_unclipped == 0
collision_entropy == m

Deterministic distribution:
uniform_tvd == 1 - 1 / D
renyi_sqrt_factor_unclipped == 0.5 * sqrt(D - 1)
collision_entropy == 0

Use numerical tolerances appropriate for floating-point statevector output.

15. NAMING AND TERMINOLOGY

Use:

- "Rényi factor" for S2;
- "raw QCAP bound" for epsilon_qcap;
- "exact under the global white-noise model" for epsilon * D_TV(p, u);
- "Porter-Thomas/Haar-like output probabilities" rather than "the circuit is Haar";
- "RC/trajectory replicate scatter" rather than "circuit-instance spread";
- "white-noise-model residual" for TVD between the actual noisy distribution and the corresponding white-noise mixture.

16. IMPLEMENTATION REQUIREMENTS

Before editing:

1. Read both source scripts completely.
2. Reuse helpers rather than duplicating incompatible implementations.
3. Preserve the QASM-specific cycle extraction, cycle counting, explicit RC, pmap simulation, and trajectory-floor code from run_qpe_bounding.py.
4. Preserve the mathematical definitions, self-tests, and ordering assertions from run_bounding_renyi_focused.py.
5. Do not silently change the meaning of depth, replicates, measured qubits, or error bars.
6. Add comments explaining which source script each major block came from.

After implementation run:

python -m py_compile examples/run_qasm_bounding_renyi.py

Then run a reduced smoke test with temporary settings such as:

DEPTHS = [1, 2]
N_INSTANCES = 2
N_TRAJ = 5
FLOOR_PAIRS = 1
CB_DECAYS = 2
CB_SHOTS = 100
PORTER_THOMAS_DEPTHS = [1, 2]

Restore the intended default settings after the smoke test.

Report:

- the new file created;
- whether py_compile passed;
- whether the reduced smoke test passed;
- all plots and data files produced;
- any incompatibilities between the two source scripts;
- any cases where the QASM gate set limits the validity of the explicit Pauli twirl.