Create a new script by copying:

```text
examples/run_bounding.py
```

to a new file, preferably:

```text
examples/run_bounding_renyi.py
```

Do not modify the original script.

The new script should preserve the existing circuit construction, cycle benchmarking, randomized-compiling simulation, random/structured ansätze, plotting style, and `.npz` output. Add the following analyses.

## 1. Compute three bounds at every depth

The existing QCAP error parameter is

[
\epsilon_{\mathrm{QCAP}}
========================

1-F_{\mathrm{RO}}
\prod_c (1-e_{F,c})^{m_c},
]

where (m_c) is the number of times cycle (c) appears. Continue obtaining this from `qcap_bound`.

Call this quantity the **raw QCAP bound**.

For each ideal circuit distribution (p), let (u) be the uniform distribution on all (2^N) bit strings:

[
u(x)=2^{-N}.
]

Under the global white-noise model

[
q=(1-\epsilon_{\mathrm{QCAP}})p+\epsilon_{\mathrm{QCAP}}u,
]

the TVD from the ideal distribution is exactly

[
D_{\mathrm{TV}}(p,q)
====================

\epsilon_{\mathrm{QCAP}}D_{\mathrm{TV}}(p,u).
]

Therefore compute the circuit-specific **exact white-noise bound**

[
B_{\mathrm{exact}}
==================

\epsilon_{\mathrm{QCAP}}
\frac{1}{2}\sum_x |p(x)-2^{-N}|.
]

This is “exact” only for the assumed global white-noise mixture. Make that qualification explicit in comments, console output, legends, and labels.

Also compute an order-2 Rényi upper bound. Define

[
D_2(p|u)
========

# \log\left(\sum_x\frac{p(x)^2}{u(x)}\right)

\log\left(2^N\sum_x p(x)^2\right).
]

Using the chi-squared/Rényi inequality,

[
D_{\mathrm{TV}}(p,u)
\le
\frac{1}{2}\sqrt{e^{D_2(p|u)}-1},
]

define

[
R_2(p)
======

\frac{1}{2}
\sqrt{e^{D_2(p|u)}-1}
=====================

\frac{1}{2}
\sqrt{2^N\sum_x p(x)^2-1}.
]

Since TVD cannot exceed one, use

[
R_{2,\mathrm{clipped}}(p)=\min(1,R_2(p)).
]

The **Rényi bound** is then

[
B_{\mathrm{Renyi}}
==================

\epsilon_{\mathrm{QCAP}}R_{2,\mathrm{clipped}}(p).
]

Compute and retain both the unclipped Rényi factor and the clipped one.

The three plotted quantities should therefore be:

```text
raw QCAP:
    epsilon_qcap

exact white-noise:
    epsilon_qcap * TVD(p_ideal, uniform)

Renyi-2:
    epsilon_qcap * min(
        1.0,
        0.5 * sqrt(2**N * sum_x p_ideal[x]**2 - 1)
    )
```

Handle small negative floating-point values inside the square root using:

```python
max(argument, 0.0)
```

## 2. Compute the bounds per circuit instance

The exact and Rényi factors depend on the ideal output distribution, so do not compute only one value per depth.

For every circuit instance:

1. Construct the circuit exactly as the current code does.
2. Compute its ideal distribution.
3. Compute its measured RC TVD.
4. Compute:

   * `uniform_tvd`
   * `renyi2_divergence`
   * `renyi_sqrt_factor_unclipped`
   * `renyi_sqrt_factor_clipped`
   * `exact_white_noise_bound`
   * `renyi_bound`

Refactor `noisy_tvds` if useful so that the ideal distribution is not calculated twice. A reasonable interface would return a dictionary containing the measured TVD and all ideal-distribution-derived quantities.

Use a common helper that converts the backend distribution into a dense probability vector of length (2^N). It must correctly include outcomes absent from a sparse dictionary as zero-probability outcomes.

For each depth, report the mean and standard deviation across instances for:

* measured RC TVD;
* exact white-noise bound;
* Rényi bound;
* (D_{\mathrm{TV}}(p,u));
* unclipped Rényi square-root factor;
* fraction of instances for which the Rényi factor was clipped at one.

The raw QCAP bound remains common to all instances at a fixed depth.

## 3. Update the main bounding figure

Keep the existing two-panel random-versus-structured figure.

For each panel, show:

* measured RC TVD instance scatter;
* raw QCAP bound as a line;
* mean exact white-noise bound as a line;
* mean Rényi-2 bound as a line;
* uncertainty bands showing the spread across circuit instances for the exact and Rényi bounds.

Use either standard deviation or the 16th–84th percentile interval for the circuit-instance bands. Label the choice clearly.

Do not use `qcap_bound`’s parameter-estimation uncertainty as though it were circuit-instance spread. If practical, show the raw QCAP uncertainty separately from the instance-to-instance variation.

Use a linear depth axisc                                                                                                                                                                            

Clip all plotted TVD bounds into ([0,1]).

Use precise legend labels such as:

```text
Measured TVD, randomly compiled
Raw QCAP bound
Exact under white-noise model
Rényi-2 upper bound
```

## 4. Add an N-scaling analysis

The existing script fixes `N = 2`. Add a separate configurable list:

```python
N_SCALING = [2, 3, 4, 5, 6, 7, 8]
```

and configurable values such as:

```python
SCALING_DEPTH = 8
SCALING_INSTANCES = 40
```

For every (N) in `N_SCALING`, generate random and structured circuit instances at `SCALING_DEPTH` and compute from each ideal distribution:

[
D_{\mathrm{TV}}(p,u),
]

[
S_2(p)
======

\frac{1}{2}
\sqrt{2^N\sum_xp(x)^2-1},
]

and

[
S_{2,\mathrm{clipped}}(p)=\min(1,S_2(p)).
]

The requested “sqrt factor versus (N)” is the **unclipped**

[
S_2(p)
======

\frac{1}{2}\sqrt{e^{D_2(p|u)}-1}.
]

Plot it before clipping so its scaling is visible.

Do not rerun cycle benchmarking for this scaling study unless it is actually needed. This analysis concerns ideal-distribution geometry, not the noise strength.

Create a new figure with one panel per ansatz, showing:

* individual `S_2` values versus (N);
* mean or median `S_2` versus (N);
* uncertainty bars or a percentile band;
* the exact factor (D_{\mathrm{TV}}(p,u)) for comparison.

Save it as something like:

```text
results/renyi_sqrt_factor_vs_N_<gate_set>.png
```

## 5. Fit the square-root factor versus N

Fit the ensemble mean of the **unclipped** square-root factor.

Perform at least these fits:

### Exponential fit

[
S_2(N)=A e^{bN}.
]

Fit this using `scipy.optimize.curve_fit`, or equivalently use a linear fit to (\log S_2) after excluding nonpositive points.

Report (A), (b), parameter uncertainties, and (R^2).

### Dimension-motivated fit

Because the Hilbert-space dimension is (d=2^N), also fit

[
S_2(N)=A,2^{\alpha N}.
]

Report (A), (\alpha), uncertainties, and (R^2).

Note that these two fits are reparameterizations of one another, with

[
b=\alpha\log 2.
]

This second representation is still useful because (\alpha=1/2) corresponds to square-root Hilbert-space growth.

Also overlay the reference scaling

[
C,2^{N/2},
]

where (C) is fitted with the exponent fixed to (1/2). Report its (R^2) so we can assess whether the observed factor follows square-root dimension scaling.

Perform fits separately for the random and structured ansätze.

Use only finite, positive mean values in logarithmic fits. Do not silently replace zeros.

Include fit equations and (R^2) values in the legend or in a compact text box.

A log-scale y-axis is preferred for the fit figure if the values span more than roughly one decade.

## 6. Save all raw scaling data

Extend the `.npz` output or write a second `.npz` file containing, for each ansatz and each (N):

* all exact uniform-TVD factors;
* all unclipped Rényi square-root factors;
* all clipped Rényi factors;
* means;
* standard deviations;
* medians;
* 16th and 84th percentiles;
* fitted parameters;
* fitted-parameter covariance or standard errors;
* (R^2) values;
* `N_SCALING`;
* `SCALING_DEPTH`;
* `SCALING_INSTANCES`;
* seeds and gate-set configuration.

Also save the per-depth, per-instance exact and Rényi bounds from the main experiment, not merely their averages.

## 7. Avoid exponential simulation cost during the N-scaling study

The N-scaling study requires only ideal distributions. Use:

* the stabilizer backend for Clifford circuits;
* the statevector backend for non-Clifford circuits.

Do not run noisy trajectories or shot sampling for this analysis.

Before using large values of (N), estimate memory requirements for a dense statevector and skip unsupported values with a clear warning rather than crashing.

## 8. Validation checks

Add assertions or printed diagnostics verifying:

[
D_{\mathrm{TV}}(p,u)
\le
\min\left(
1,
\frac12\sqrt{2^N\sum_xp(x)^2-1}
\right)
]

up to numerical tolerance for every circuit.

Also verify:

```python
exact_white_noise_bound <= renyi_bound + tolerance
renyi_bound <= raw_qcap_bound + tolerance
0 <= all_bounds <= 1
```

For a uniform ideal distribution, verify that:

```text
uniform_tvd = 0
renyi factor = 0
exact bound = 0
Renyi bound = 0
```

For a deterministic ideal distribution, verify analytically that:

[
D_{\mathrm{TV}}(p,u)=1-2^{-N},
]

and

[
S_2(p)=\frac12\sqrt{2^N-1}.
]

The latter is allowed to exceed one before clipping.

Add a small deterministic helper test for these two distributions.

## 9. Console output

At each depth print a table similar to:

```text
 depth    QCAP    measured    exact-WN    Renyi-2    <TVD(p,u)>    <sqrt factor>    clipped
```

For the N-scaling fits print something like:

```text
random:
  A exp(bN):          A=..., b=..., R2=...
  A 2^(alpha N):      A=..., alpha=..., R2=...
  C 2^(N/2):          C=..., R2=...

structured:
  ...
```

## 10. Preserve reproducibility and compatibility

Preserve the existing seed conventions where possible, but ensure that circuits generated for different (N), ansätze, and instances receive distinct deterministic seeds.

Keep all new dependencies minimal. `numpy`, `matplotlib`, and `scipy` are acceptable.

Do not change the behavior of the existing `proxysim` library unless necessary. Prefer implementing the new analysis locally in the copied example script.

Run the new script and fix all errors. Then report:

1. the new file created;
2. the commands used to run it;
3. output files generated;
4. validation-test results;
5. fitted scaling parameters;
6. any values of (N) skipped because of computational limitations.

Before finalizing, inspect the figures to make sure labels are readable, uncertainty bands are meaningful, and no fit line is drawn across missing or invalid data.
