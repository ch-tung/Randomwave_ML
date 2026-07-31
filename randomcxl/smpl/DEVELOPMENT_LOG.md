# Random-Wave Line Scattering Development Log

This folder is a side project for analytic/numerical scattering from the
line network

```text
L = {r : psi_1(r) = 0, psi_2(r) = 0}
```

where `psi_1` and `psi_2` are independent isotropic Gaussian random waves.

## Files

- `rw_line_scattering.py`: reusable numerical routines and CLI.
- `rw_line_scattering_demo.ipynb`: monochromatic covariance notebook.
- `rw_line_general.ipynb`: general radial-spectrum notebook using
  numerical averages over sampled `|k|`.

## Core Correlation

For the monochromatic case,

```text
g(r) = sin(k0 r)/(k0 r)
a = k0^2/3
rho0 = a/pi
C_L(r) = M_J(r)/(4*pi^2*(1-g(r)^2))
```

Small-r series are used for `g`, `g'`, and `g''` to avoid cancellation near
`r=0`. The conditional gradient covariance is sampled with common random
numbers across all `r` values so `M_J(r)` is smooth.

## Finite-Q Transform Convention

The isotropic scattering transform is

```text
I(Q) = 4*pi * integral r^2 C(r) sinc(Q*r/pi) dr
```

where NumPy's normalized `np.sinc(x)=sin(pi*x)/(pi*x)` is used.

The background large-r plateau is

```text
C_L(r -> infinity) = rho0^2
```

This corresponds to the `Q=0` forward-scattering delta peak. For numerical
finite-Q scattering, the transform input should subtract this plateau before
windowing:

```python
CL_coherent = CL - rho0**2
CL_for_transform = CL_coherent * w_tail
```

The transform of `CL*w_tail` is retained only as a diagnostic for finite-window
leakage.

`compute_coherent_transform_diagnostics(...)` is the shared no-split helper
for both the monochromatic and general radial-spectrum notebooks. The caller
provides the appropriate `rho0`, and the helper returns consistently named
arrays:

- `CL_coherent = C_L-rho0**2`
- `CL_transform_raw = CL_coherent`
- `CL_transform = CL_coherent*w_tail`
- `I_coherent_raw`
- `I_coherent_windowed`
- `I_full_windowed`
- `I_plateau_windowed`

The monochromatic notebook still uses `compute_transform_diagnostics(...)`
when the optional small-r singular split is desired, but the no-split and
general-spectrum usage now follow the same naming convention.

## Tail Window

`tail_window(r, r_taper_start, r_max)` is a half-cosine taper:

- `w=1` for `r <= r_taper_start`
- smoothly decreases to zero at `r_max`
- default notebook choice: `r_taper_start = 0.75*r_max`

This is intended to reduce low-Q ringing from abrupt truncation of the
large-r tail.

## Singular Split Diagnostic

The same-line singular behavior is

```text
C_sing(r) = rho0/(2*pi*r^2) * exp(-(r/rc)^2)
```

with default notebook setting

```text
rc = 0.5/k0
```

When enabled, the code computes

```python
C_reg = C_L - rho0**2 - C_sing
I_total = H[C_sing] + H[C_reg*w_tail]
```

The current implementation integrates `C_sing` numerically on the same `r_grid`.
This split is a diagnostic/stabilization option; the no-split coherent
transform remains available through `use_singular_split = False`.

## Diagnostic Transforms

`compute_transform_diagnostics(...)` returns:

- `I_full_windowed = H[C_L*w_tail]`
- `I_plateau_windowed = H[rho0^2*w_tail]`
- `I_coherent_windowed = H[(C_L-rho0^2)*w_tail]`
- `I_sing = H[C_sing]`
- `I_reg_windowed = H[(C_L-rho0^2-C_sing)*w_tail]`
- `I_total = I_sing + I_reg_windowed`

The notebook also runs convergence checks over taper starts and finite `r_max`
cutoffs to test whether low-Q features, especially around `Q/k0 ~ 1`, are
window-sensitive.

## Q/k0 = 1 Tail Diagnostics

The monochromatic line notebook now includes a dedicated diagnostic
block for the feature near `Q/k0 = 1`. It keeps the plateau subtraction
explicit,

```python
C_coherent = C_L - rho0**2
```

then checks:

- linear and tail-weighted views of `C_coherent(r)`;
- Hann-window tail projections onto `k0` and `2*k0`;
- Hankel transforms of `C_coherent`, the large-r tail alone, the window, and the
  diagnostic plateau leakage;
- comparison across `cosine`, `hann_tail`, `tukey`, `exponential`, and `none`
  transform windows;
- finite-`r_max` truncation sensitivity;
- transforms of simplified covariance-derived ingredients;
- the tail of `M_J(r)-4*a**2`.

The notebook saves these diagnostics as `diagnostics_*.png` in its output
folder and prints numerical indicators for
the `Q/k0 ~= 1` sample. These are intended as diagnostics only; they do not
automatically decide whether the feature is physical or numerical.

Current interpretation: the diagnostic Hankel transform

```python
H[f](Q) = 4*pi * integral r**2 * f(r) * sinc(Q*r/pi) dr
```

shows a pronounced feature near `Q/k0 ~= 1` already for `f(r)=g(r)`, where
`g(r)=sin(k0*r)/(k0*r)` is the monochromatic random-wave covariance. This
indicates that the `Q/k0 ~= 1` feature can arise from the monochromatic shell
structure itself, not only from the finite tail window. The full `C_L`
transform still needs the window/rmax checks because tail handling can change
amplitudes and ringing.

## General Radial Spectrum

The general line notebook uses sampled isotropic k radii, for example
`gaussian_radial`, and evaluates

```text
g(r) = mean_k sinc(k r)
```

with numerical derivatives `g'` and `g''`. The gradient variance becomes

```text
a = <k^2>/3
rho0 = a/pi
```

This notebook is the proper path for non-monochromatic radial spectra; using
only an effective `k0=sqrt(<k^2>)` is only a rough comparison.

The radial-spectrum covariance routines are chunked over `r`:

```python
g_num, gp_num, gpp_num = radial_covariance_numeric(r_grid, k_radii)
```

This avoids allocating the full `len(r_grid) x len(k_radii)` matrix. For the
default general notebook values `Nr=10000` and `num_modes_k=2**14`, a full
outer-product array would be about `1.31 GB` per float64 work array, with
several such arrays alive during `sin`, `cos`, and masking. With the default
chunk size `256`, each main work array is about `34 MB`. The individual
wrappers `g_radial_numeric`, `gp_radial_numeric`, and `gpp_radial_numeric`
also chunk, while `radial_covariance_numeric` computes all three together to
reuse the same `k*r`, `sin(k*r)`, and `cos(k*r)` work.

The general notebook supports three radial wavevector sampling modes:

```python
k_sampling = "qmc"         # Sobol directions and radii
k_sampling = "random"      # pseudorandom directions and radii
k_sampling = "quadrature"  # deterministic radial nodes with weights
```

`use_qmc_k=True` in `make_field_k_sets(...)` uses Sobol points for isotropic
directions and for the radial variable. For `gaussian_radial`, the radial
Sobol coordinate is mapped through the positive-truncated Gaussian inverse CDF.

For cases where the radial probability weights are known, the code can avoid a
large random list of `k` samples:

```python
k_radii, k_weights = make_radial_k_quadrature(...)
g, gp, gpp = radial_covariance_numeric(r_grid, k_radii, k_weights=k_weights)
M_J, C_L = compute_CL_general(r_grid, k_radii, N_samp, k_weights=k_weights)
```

The weights are normalized internally and are used consistently in `<k^2>`,
`rho0`, `g`, `g'`, `g''`, conditional covariance, and `C_L`.

## Current Numerical Notes

- Avoid `r=0`; the notebooks start at a small positive `r_min`.
- `Q*I(Q)` should approach `pi*rho0 = k0^2/3` in a local-line regime for the
  monochromatic case.
- Low-Q behavior is sensitive to `r_max`, tail noise, and the taper.
- Increasing `N_samp` improves the Monte Carlo/QMC estimate of `M_J(r)` and
  reduces noisy tail structure in `C_L(r)`, which makes the `Q/k0 ~= 1`
  diagnostic cleaner.
- The inverse transform check is band-limited by the available `Q_grid`; it is
  best interpreted as a consistency diagnostic rather than an exact recovery of
  `C_L(r)`.

## 4Field Cross-Line Scattering

`rw_line_scattering.py` now also implements cross correlations between

```text
L_12 = {psi_1=psi_2=0}
L_34 = {psi_3=psi_4=0}
```

with field-label correlations `rho13` and `rho24`. The fields are not generated
on a grid; the calculation conditions Gaussian gradients on zero field values.
The cross zero vector is

```text
[psi_1(0), psi_2(0), psi_3(r), psi_4(r)]
```

and the conditional gradient vector is 12D:

```text
[grad psi_1(0), grad psi_2(0), grad psi_3(r), grad psi_4(r)].
```

The shared line-scattering module provides:

- `compute_self_correlation(...)`
- `compute_cross_correlation(...)`
- `compute_wave_stats_for_r_grid(...)`
- `compute_self_correlation_from_wave_stats(...)`
- `compute_cross_correlation_from_wave_stats(...)`
- `run_four_field_scan(...)`
- `save_four_field_scan_outputs(...)`

`rw_4field_line_demo.ipynb` is the test notebook for this branch. It follows
the same parameter-cell style as `rw_line_general.ipynb`: wave statistics are
kept in one place, and the line-pair wave correlations are controlled by a
tunable `rho_pairs = [(rho13, rho24), ...]` list.
It now supports non-monochromatic radial spectra through precomputed wave
statistics `g(r)`, `g'(r)`, and `g''(r)` from the sampled or weighted `k_radii`.
The default transform settings were aligned with the robust general-line
notebook values: `r in [1e-3/k_eff, 5e2/k_eff]`, `Nr=10000`,
`Q/k_eff in [0.1, 20]`, `NQ=256`, `tail_start=0.8*r_max`, and
`2**16` Sobol samples for the conditional Jacobian averages.

QMC sample-count handling is now consistent between the two-wave and four-field
methods. Public notebook/API settings use `N_samp` directly. Sobol-specific
power-of-two conversion is handled internally by `make_qmc_normals(...)` and
`standard_normal_samples(...)`; older `qmc_power` callers are retained only as a
compatibility path in `run_four_field_scan(...)`.

Both self and cross use the same 12D Sobol normal samples across all `r` values.
The cross finite-Q transform subtracts the same large-distance baseline
`rho0**2` before applying the tail window. Limiting checks:

- `rho13=rho24=0`: `C_cross(r)` is close to `rho0**2` within QMC error.
- `rho13=rho24=1`: `C_cross(r)` converges to `C_self(r)` as the QMC sample
  count increases.

For two equal-weight line networks `A=L_12` and `B=L_34`, the physical summed
scattering is assembled after the separate transforms as
`I_total = I_AA + I_BB + 2*I_AB`. Since the two auto terms are statistically
identical in the current notebook, `I_total = 2*I_AA + 2*I_AB`. The notebook
plots `I_AA`, `I_AB`, and `I_total` for each selected `(rho13, rho24)` pair,
and overlays the self-only total reference `2*I_AA` as a black dashed curve on
the total panels.

## Conditional 6D+2D Jacobian Estimator

Added a selectable estimator for the two-field self-correlation Jacobian
average `M_J(r)`.

The original estimator remains available as:

```python
jacobian_method = "direct_12d"
```

It samples two independent 6D conditional gradient vectors directly and
estimates:

```python
M_J(r) = E[|u_0 x v_0| |u_r x v_r|]
```

The new estimator is:

```python
jacobian_method = "conditional_6d_2d"
```

It samples only the outer 6D vector `U=[u_0,u_r]` and evaluates the Gaussian
average over `V=[v_0,v_r]` through the square-root Laplace representation. For
fixed `U`, the code forms:

```python
K(u) = |u|^2 I - u u.T
A0 = blockdiag(K(u_0), 0)
Ar = blockdiag(0, K(u_r))
```

and computes the deterministic auxiliary integral over positive `t,s` using:

```python
D(t,s) = det(I + 2 Sigma_half (t*A0 + s*Ar) Sigma_half)^(-1/2)
F(t,s) = 1 - D(t,0) - D(0,s) + D(t,s)
```

The positive quadrant is mapped from the unit square by default with:

```python
t = tau*x/(1-x)
s = tau*y/(1-y)
```

where `tau` is chosen from the covariance scale unless supplied internally.
The determinant path uses `slogdet`, symmetrizes the 6x6 matrix, and raises a
diagnostic error if the matrix is materially indefinite.

The first implementation evaluated the inner integral with scalar Python loops:
one `slogdet` triplet for every `(U,t,s)` combination. This was correct but
slow. The conditional path now batches the determinant work:

```python
evaluate_inner_st_integrals_batched(...)
```

This broadcasts a chunk of outer `U` samples against a chunk of `t,s` nodes and
uses NumPy's stacked `slogdet` on arrays of 6x6 matrices. The scalar
`evaluate_inner_st_integral(...)` remains as a readable reference/debug path.
A small sanity benchmark with `64` outer samples and about `64` inner nodes gave
identical values to roundoff and about a `3.8x` speedup for the inner integral.
The conditional estimator can still be slower than `direct_12d` for large scans,
because it pays the full `N_samp_U*N_samp_st` determinant cost.

The default conditional path was further optimized with the matrix determinant
lemma. Since each cross-product quadratic form is rank 2,

```python
A0 = L0 @ L0.T
Ar = Lr @ Lr.T
```

the determinant

```python
det(I + 2*Sigma_half @ (t*A0+s*Ar) @ Sigma_half)
```

is evaluated in the low-rank column space:

```python
det(I + 2*X.T @ Sigma @ X)
X = [sqrt(t)*L0, sqrt(s)*Lr]
```

This reduces the mixed determinant from 6x6 to 4x4, and the one-sided
determinants to 2x2. The full 6x6 batched path remains available as
`evaluate_inner_st_integrals_batched(...)`; the default estimator uses
`evaluate_inner_st_integrals_lowrank(...)`. A local check with `64` outer
samples and about `64` inner nodes matched the scalar 6x6 reference to
`~1.5e-15`, with the low-rank path about `5.5x` faster than batched 6x6 and
about `80x` faster than the original scalar loop for that inner-integral test.

For long `r_grid` scans, `compute_CL(...)` and `compute_CL_general(...)` now
reuse the same outer standard-normal samples and the same positive-quadrant
`t,s` nodes across all `r` values. The shared map scale defaults to
`st_tau=1/a**2`, with `a=k0**2/3` for the monochromatic model and
`a=<k**2>/3` for a radial spectrum. A custom `st_tau` can be supplied if a
specific quadrature map scale is desired. The functions also accept `n_jobs`;
values above one use a thread pool over `r` values. This is intentionally
thread-based rather than process-based to avoid repeatedly copying large QMC
arrays on Windows. A smoke check gave identical serial/threaded results for the
conditional estimator.

Public controls added to `compute_CL(...)` and `compute_CL_general(...)`:

```python
jacobian_method = "direct_12d" | "conditional_6d_2d"
N_samp_U
N_samp_st
U_sampling = "qmc" | "random"
st_sampling = "quadrature" | "qmc"
st_transform = "rational" | "logistic"
st_tau
n_jobs
```

The old positional `n_samp` remains a compatibility alias for `N_samp_U`; if
both are supplied inconsistently, a deprecation warning is issued and
`N_samp_U` wins.

The existing notebooks `rw_line_scattering_demo.ipynb` and
`rw_line_general.ipynb` now expose these controls while defaulting to
`direct_12d`, preserving previous behavior.

Added `rw_line_mj_qmc_efficiency_benchmark.ipynb` to compare the old direct
12D QMC estimator against the conditional 6D+2D estimator. The benchmark
reports:

- sample budgets (`N_samp_U`, `N_samp_st`);
- wall time;
- `M_J(r)` at representative separations;
- absolute and relative error against a high-sample direct reference;
- error versus runtime plots;
- a log-log fit to the direct-sampling walltime/error scaling line, with each
  conditional estimator point reported as a ratio relative to that fitted
  direct baseline at the same wall time.

The benchmark intentionally does not assume the conditional method is faster;
it measures the full cost, including the two-dimensional auxiliary integral.
It now uses the same full `r` and `Q` ranges as `rw_line_scattering_demo.ipynb`
by default (`r in [2e-3/k0, 5e2/k0]`, `Nr=5000`, `Q/k0 in [0.1, 20]`,
`NQ=1001`) and times the full path:

```python
M_J(r) -> C_L(r) -> coherent/windowed I(Q)
```

The main efficiency score is the relative L2 error of the coherent real-space
correlation `C_L-rho0^2` against the high-sample direct reference. The final
`I(Q)` error is still printed as a secondary end-to-end transform diagnostic.

## 2026-06-23 Heterogeneous Mask Demo Notebook Updates

- Updated `rw_hetero_demo.ipynb` so the notebook is the main control surface
  for the heterogeneous-mask examples.
- Removed the redundant `Selected Full Curve` block.
- Set the main demo parameter block to `r_sigma_k = 0.2`.
- Kept the lower `Quick View` decomposition plot, now selecting
  `k_H/<k> = 0.1` and `b = -1`, with a printed status line showing the active
  `k_H/<k>`, `b`, and `r_sigma_k`.
- Added a status-printing spectrum-width scan that fixes `k_H/<k> = 0.1` and
  `b = -1`, then recomputes the full uniform-line workflow for
  `r_sigma_k = 0.5, 0.2, 0.1`.
- Added a status-printing clipping-level scan that fixes `k_H/<k> = 0.1` and
  `r_sigma_k = 0.2`, then scans `b = -2, -1, 0` while plotting the three
  smooth-model components separately.
- Preserved the existing mean-`k` Q-axis convention and the enlarged real-space
  grid (`r_max_factor = 600`, `Nr = 5000`) used for the `Q/<k> = 0.05..20`
  evaluation window.
- The smoothed-line contribution continues to use the refined-grid
  `smoothing_operator_A(...)` path with `q_max = max(Q_grid)`, avoiding the
  earlier numerical splice artifact near `Q/<k> ~ 10`.
- Reorganized `rw_hetero_demo.py` so it contains reusable notebook/tool
  functions only. Demo parameter choices such as `K_H_OVER_K_VALUES`,
  `B_VALUES`, spectrum settings, grid sizes, and output path are now declared
  explicitly in `rw_hetero_demo.ipynb` and passed into the helper functions.
- Removed the script-style `main()`/CLI wrapper and hidden module defaults from
  `rw_hetero_demo.py`; the heterogeneous physics kernels remain in
  `rw_line_scattering.py`, while the demo module handles notebook-facing
  orchestration, plotting, and table/report export.

## 2026-06-25 Heterogeneous Demo Call Organization

- Added `evaluate_heterogeneous_case(...)` in `rw_hetero_demo.py` so notebook
  quick views and grid evaluations share the same single-case call path.
- Updated `evaluate_heterogeneous_grid(...)` to accept either scalar or
  one-dimensional sequence inputs for `k_h_over_k_values` and `b_values`.
- Updated `rw_hetero_demo.ipynb` so the quick view evaluates one explicit case
  (`k_H/<k> = 0.10`, `b = -1`) instead of selecting from the saved grid.
- Set the notebook grid parameters explicitly to
  `K_H_OVER_K_VALUES = (0.10, 0.05)` and `B_VALUES = (-1.0, 0.0)`.
- Replaced direct heterogeneous calls in the scan section with
  `demo.evaluate_heterogeneous_case(...)` for a more coherent workflow.
- Repaired the joint `r_sigma_k`/`b` scan section in `rw_hetero_demo.ipynb`:
  scan parameters are now declared in the main parameter block, the scan data
  cell actually constructs `sigma_b_scan_results`, and both plot cells guard
  against being run before the scan data cell.
- Added a local static notebook-order check and fake-data execution check for
  the scan plot cells after the repair.

## 2026-06-25 Low-Q Asymptotic Line-Scattering Stabilization

- Added `LowQAsymptoticFit` diagnostics and `stabilize_low_q_quadratic(...)`
  to `rw_line_scattering.py`.
- Extended `compute_coherent_transform_diagnostics(...)` with
  `use_asymptotic`, `lowq_fit_bounds`, and `lowq_replace_max` options.
  With `use_asymptotic=True`, the returned `I_coherent_windowed` uses the
  fitted low-`Q` form `I(Q)=I0+I2 Q^2` in the selected replacement range while
  retaining the original finite-window transform outside that range.
- The automatic fit skips modes below the finite-`r_max` resolution estimate
  and chooses a low-`Q` quadratic window by minimizing the relative residual;
  explicit fit and replacement bounds can still be supplied by the caller.
- Threaded `use_asymptotic` through `rw_hetero_demo.compute_uniform_line_scattering`
  and attached the low-`Q` fit diagnostics to the returned line result.
- Added `rw_line_asymp.ipynb`, reusing the `rw_line_general.ipynb` parameter
  convention while computing its own `C_L(r)` and transforms inside the
  notebook, to compare the original finite-window transform with the low-`Q`
  stitched result.
- Added direct real-space moment reporting through
  `line_low_q_moments_from_CL(...)`, with a notebook note that these moments
  are sensitive tail-convergence diagnostics because the integrands are
  weighted by `r^2` and `r^4`.
- Updated `rw_line_asymp.ipynb` to compute a separate
  `N_samp_ref = 2**16` reference curve. The low-`Q` quadratic fit remains based
  only on the lower-count `N_samp` calculation, while the reference is plotted
  as a black comparison line and used for relative-error plots with and without
  the low-`Q` fit.

## 2026-06-25 Heterogeneous Demo Low-Q Stitch Integration

- Extended `rw_hetero_demo.compute_uniform_line_scattering(...)` with
  `lowq_fit_bounds_over_k` and `lowq_replace_max_over_k`, so notebook fit and
  replacement ranges can be specified in the same `Q/<k>` convention used by
  the plots.
- The returned uniform-line result now keeps both the stitched `I_L` and the
  raw finite-window `I_L_original`, plus low-`Q` fit masks and diagnostics.
- Updated `rw_hetero_demo.ipynb` with explicit low-`Q stitch controls:
  `use_lowq_stitch`, `lowq_fit_min_over_k`, `lowq_fit_max_over_k`, and
  `lowq_replace_max_over_k`.
- The main heterogeneous calculation and the `r_sigma_k`/`b` scans now pass
  the same stitch settings into the uniform-line calculation before applying
  the heterogeneous mask layer.

## 2026-06-25 Heterogeneous Demo Parameter And Import Cleanup

- Standardized the random-line scattering module alias to
  `import rw_line_scattering as rls` in `rw_hetero_demo.py` and
  `rw_hetero_demo.ipynb`.
- Reorganized the `rw_hetero_demo.ipynb` parameter block into labeled sections:
  line-wave spectrum, saved heterogeneous grid, Quick View case, joint scan
  settings, output location, nonuniform real-space grid, Q grid,
  conditional-Jacobian sampling, and low-`Q` stitch controls.
- Clarified which parameters drive the saved demo grid versus the Quick View
  and scanned calculations, using `QUICK_*` and `SCAN_*` names for those
  notebook-only workflows.
- Exposed the mixed nonuniform real-space grid settings explicitly in the
  notebook parameter block: `r_grid_mode`, `r_min_factor`, `r_split_factor`,
  `r_max_factor`, `Nr`, and `Nr_small`.

## 2026-06-26 Low-Q Extrapolation In Heterogeneous Convolution

- Extended `smoothing_operator_A(...)` with optional `lowq_I0` and `lowq_I2`
  inputs. When available, the convolution now fills the missing interval from
  `Q=0` to the first tabulated uniform-line point with the fitted quadratic
  form `I(Q)=I0+I2 Q^2`.
- Updated `heterogeneous_line_scattering(...)` to automatically read
  `lowQ_I0` and `lowQ_I2` metadata from a uniform-line result and pass those
  coefficients into the smoothed-line convolution term.
- Added a deterministic regression check showing that low-`Q` extrapolation
  removes the abrupt small-`Q` truncation artifact in
  `sigma_H^2 A_kappa[I_L]`, while preserving the existing kernel,
  tail-correction, high-`Q`, and uniform-line tests.

## 2026-07-01 55B/55C Density, Absolute Scale, and SLD Analysis

Extended `curvefit/cf_compare_55B_55C.ipynb` with a physical interpretation of
the fitted 55B and 55C line models. The notebook now compares the fitted line
density and absolute scattering scale with the nominal PSSNa concentrations
of 1.5 wt% for 55B and 4.5 wt% for 55C.

### Line-Length Density and Polymer Content

The fitted random-wave parameter

```text
rho0 = <k^2>/(3*pi)
```

is the unmasked geometrical line length per unit volume. The heterogeneous
mask retains the fraction `p_H`, so the retained density is `p_H*rho0`.
For `Q` in inverse angstroms,

```text
1 A^-2 = 100 nm/nm^3
```

and the fitted results are:

- 55B: `0.0125091 nm/nm^3` unmasked and `0.00984155 nm/nm^3` retained.
- 55C: `0.0336456 nm/nm^3` unmasked and `0.0320908 nm/nm^3` retained.

The mass conversion assumes a PSSNa repeat molar mass of `206.19 g/mol` and
an all-trans contour increment of `0.25 nm/repeat`. It gives apparent polymer
contents of:

- 55B: `1.234 wt%` from the unmasked density and `0.974 wt%` from the
  retained density.
- 55C: `3.252 wt%` from the unmasked density and `3.106 wt%` from the
  retained density.

Inverting the independently known 1.5/4.5 wt% compositions gives the amount
of chemical contour represented by one unit of fitted smooth-line length.
For the unmasked density, the true-contour/fitted-line factors are `1.219`
for 55B and `1.402` for 55C, corresponding to effective repeat advances of
`0.205 nm` and `0.178 nm`. The analogous retained-line factors are `1.549`
and `1.470`, with advances of `0.161 nm` and `0.170 nm`.

These factors measure unresolved local tortuosity or coarse-graining; they are
not Kuhn lengths. The all-trans sp3 geometry gives about `0.25 nm` per vinyl
repeat, about `4 repeats/nm`, or about `8 backbone carbons/nm`. The inferred
values of roughly `9.75-12.39 backbone carbons/nm` indicate additional contour
per fitted line length, but a Kuhn length requires a tangent-correlation
measurement or a long-chain relation such as `b_K ~= 6 R_g^2/L`.

### Coherent Contrast and Effective Cross-Section

Bound coherent scattering lengths were assembled for hydrogenated PSSNa
`C8H7NaO3S` and glycerol-d8 `C3D8O3`. The working material values are:

```text
PSSNa dry SLD       = 2.248e-6 A^-2
glycerol-d8 SLD     = 7.47928e-6 A^-2
Delta rho           = -5.23128e-6 A^-2
PSSNa dry density   = 1.51272 g/cm^3
glycerol-d8 density = 1.371 g/cm^3
```

For a thin line, the excess coherent scattering length per unit contour is

```text
beta_L = integral_cross_section Delta_rho(r_perp) dA
       ~= Delta_rho * A_eff
```

and, with the measured intensity in `cm^-1` and `Q` in `A^-1`, the fitted
dimensionless multiplier obeys

```text
scale = 1e8 * (Delta_rho * A_eff)^2.
```

Thus `A_eff` is derived after the fit from the absolute intensity multiplier;
it is not an independently fitted geometric radius. The fitted values are:

- 55B: `A_eff = 1.15912 nm^2`, equivalent circular radius `0.6074 nm`.
- 55C: `A_eff = 1.07189 nm^2`, equivalent circular radius `0.5841 nm`.

The random-wave line density and correlations determine the shape and
geometrical amplitude of `I_L(Q)`, while `scale` supplies the squared coherent
cross-section per unit length. They are separate model parameters, although
absolute data constrain their product; for example, the local-line high-`Q`
amplitude is proportional to `scale*p_H*rho0`.

The uniform solvent SLD contributes only to the forward-scattering component.
The model removes the large-distance `rho0^2` plateau before the finite-`Q`
transform, so the fitted signal represents excess SLD fluctuations relative
to the background. Hydration, dissociated-counterion partitioning, a diffuse
radial SLD profile, and residual incoherent background are not separately
resolved and are therefore folded into `A_eff` and the fitted background.

### Nominal-Composition SLD Distribution

Assuming additive component volumes, the polymer volume fraction is calculated
from the nominal mass fraction `w` as

```text
phi_p = (w/rho_p) / ((w/rho_p) + ((1-w)/rho_s)).
```

The simplest molecular-resolution, two-component SLD distribution is then

```text
P(rho) = (1-phi_p) delta(rho-rho_s) + phi_p delta(rho-rho_p),
mean(rho) = (1-phi_p) rho_s + phi_p rho_p,
variance(rho) = phi_p (1-phi_p) (rho_p-rho_s)^2.
```

This nominal composition determines only the one-point distribution. Its
spatial covariance, and hence the `Q` dependence, is supplied by the fitted
random-wave line and mask model.

The resulting estimates are:

- 55B: `phi_p = 0.0136139`, mean SLD `7.40806e-6 A^-2`, and RMS local
  fluctuation `6.06209e-7 A^-2`.
- 55C: `phi_p = 0.0409569`, mean SLD `7.26502e-6 A^-2`, and RMS local
  fluctuation `1.03679e-6 A^-2`.

Using the unmasked density, which represents the total polymer before the
heterogeneous visibility mask:

- 55B nominal composition implies `A = 1.08832 nm^2` and
  `scale = 32.4136`. The fitted area is 6.5% higher and the fitted scale is
  13.4% higher, so the density and absolute intensity are reasonably
  consistent.
- 55C nominal composition implies `A = 1.21731 nm^2` and
  `scale = 40.5523`. The fitted area is 11.9% lower and the fitted scale is
  22.5% lower. The constraints agree in order of magnitude, but exact
  consistency is not reached.

Because the intensity depends on `(Delta_rho*A)^2`, a modest error in area or
effective contrast produces about twice that relative error in scale. The
remaining 55C discrepancy may reflect hydration or counterion effects,
concentration and absolute-calibration uncertainty, a diffuse cross-sectional
SLD distribution, or covariance between fitted line density and intensity
scale. The two-delta distribution is therefore a dry-polymer/solvent limiting
model rather than a resolved radial SLD profile.

## Pecora Digitized-Curve Fitting and Comparison

Added a Pecora-specific curvefit workflow under `curvefit/` for the five
digitized Borsali/Nguyen/Pecora 1998 salt conditions. `cf_pecora.ipynb` now
fits one selected series at a time and writes the corrected observation,
anchor diagnostics, fitted parameters, fitted curve, and figures to
`curvefit/output/pecora`.

The low-`Q` DAB anchor was changed from a fully automatic global grid search to
a notebook-local controlled anchor:

```python
LOWQ_DAB_KAPPA_POSITION
LOWQ_DAB_KAPPA_FACTOR_BOUNDS
LOWQ_DAB_N_KAPPA
LOWQ_INITIAL_KAPPA_SCALE
LOWQ_ANCHOR_WEIGHT
```

`LOWQ_DAB_KAPPA_POSITION` directly moves the low-`Q` asymptotic crossover. With
`LOWQ_DAB_KAPPA_FACTOR_BOUNDS = (1.0, 1.0)`, the DAB anchor uses exactly that
chosen `kappa`; wider bounds allow a local search around the selected
position. The heterogeneous-line initial guess now uses this controlled
low-`Q` `kappa` directly, and the nonlinear fit uses `LOWQ_ANCHOR_WEIGHT` for
the soft low-`Q` constraint.

Added `cf_compare_pecora.ipynb`, following the YYW comparison notebook pattern
but without 3D rendering. It loads all five Pecora fitted outputs, plots the
background-corrected observations and fitted curves together, summarizes the
floating and derived fit parameters, and overlays all fitted curves on the
original Pecora 1998 figure. The overlay uses the same linear plot-frame
calibration as `curvefit/data/pecora/digitize_pecora.py`, and adds the
subtracted constant background back to each fitted curve so it sits in the
printed figure's original `(q, I(q))` coordinate system.

## 2026-07-11 General-Line Low-Q And Tangent-Correlation Methods

- Added `rw_line_tcorr.ipynb` as the reference workflow for computing and
  plotting the signed ordered-field and line-density-weighted nematic tangent
  correlations, with its numerical arrays saved under `rw_line_tcorr_output`.
- Updated `rw_line_general.ipynb` with explicit controls for fitting and
  stitching the quadratic low-`Q` asymptotic form into the finite-window
  coherent transform. The notebook now retains the original and stitched
  transforms, the asymptotic curve, fit/replacement masks, fitted coefficients,
  and relative fit error in its saved output.
- Added exact ordered-field signed tangent-moment helpers to
  `rw_line_scattering.py`. They evaluate the conditional Wick contraction
  `M_T = 2 b^2 + 4 b c_z` and the raw signed correlation `K_T^raw = M_T/M_J`
  from the radial covariance and its derivatives.
- Added a direct conditional-sampling estimator for the line-density-weighted
  nematic moment. It reuses common Sobol or pseudorandom normal samples over
  separation and returns `M_J`, `M_2`, and `K_2 = M_2/M_J`.

## 2026-07-13 Sampled Nematic Infinite-Separation Baseline

- Extended `compute_nematic_tangent_correlation(...)` to return
  `K_2_sampled` and `K_2_inf_sampled` in addition to the backward-compatible
  `K_2` key.
- `K_2_inf_sampled` is evaluated from the same standardized 12D Sobol or
  pseudorandom gradient samples used at every separation. It exposes the
  finite-sample residual around the exact isotropic limit `K_2(infinity)=0`
  without imposing a numerical lower bound or altering the sampled curve.

## 2026-07-13 Pecora Real-Space And Orientation Comparisons

- Extended `curvefit/cf_compare_pecora.ipynb` with matched heterogeneous and
  unmasked 3D slices for all five salt conditions. The 1 M fit defines a
  common physical box, while a fixed visual tube radius avoids assigning an
  unsupported physical cross-section without Pecora composition/SLD inputs.
- Added two-panel comparison of the exact signed ordered-field correlation
  `K_T^raw` and the line-density-weighted nematic correlation `K_2`, using
  each condition's fitted maximum-entropy radial spectrum and a common
  dimensionless separation axis `r*k_eff`.
- Saved the renderings, orientation plot, and per-condition orientation data
  under `curvefit/output/pecora`.

## 2026-07-13 Shared Fit-Orientation Comparison Workflow

- Added `compute_fit_orientation_correlations(...)` to `curvefit/cf_tools.py`
  to centralize construction of the fitted maximum-entropy spectrum and the
  signed and nematic real-space orientation correlations derived from it.
- Refactored `curvefit/cf_compare_pecora.ipynb` to use the shared helper while
  preserving its edited two-panel plotting choices.
- Added matching calculation and plotting cells to
  `curvefit/cf_compare_yyw.ipynb` for the 55A, 55B, and 55C fits. The cells
  save per-sample arrays and a combined two-panel figure under
  `curvefit/output/yyw`.

## 2026-07-14 YYW PB-Series Fitting Notebook

- Added `curvefit/cf_yyw_pb.ipynb` for the five PB profiles (`12I` through
  `12V`) under `curvefit/data/yyw/PB_data/PB`.
- The notebook uses one `SERIES_KEY` selector, preserves the supplied
  intensity errors and `dQ`, constructs per-curve DAB and high-`Q` anchors,
  and derives dataset-specific heterogeneous-line initial guesses and bounds.
- Nonlinear optimization is guarded by `RUN_FIT = False`, allowing the anchor
  and initial-model diagnostics to be inspected before each fit is launched.

## 2026-07-15 Fixed Nonlinear Parameters In Curve Fits

- Added a `fixed_parameters` option to
  `curvefit/cf_tools.py::fit_heterogeneous_line_least_squares(...)`. Fixed
  parameters are validated against the active bounds, excluded from the
  `least_squares` vector, and still reported in the final fitted parameter
  dictionary.
- Exposed this in `curvefit/cf_yyw_pb.ipynb` through a `FIXED_PARAMETERS`
  dictionary near the initial guesses and bounds, with commented examples for
  freezing `skewness`, `b`, or `k_H_over_k`.

## 2026-07-30 Ca-Series Reference-Feature Fitting

- Added `curvefit/cf_ca.ipynb` for the GRASP `Q, I, error, dQ` profiles under
  `curvefit/data/Ca`. The notebook discovers the series from filenames, parses
  the Ca concentration in millimolar, and assigns the `D2O` file to `0 mM`.
- The salt-free fit is the reference for the line-feature wavelength. After
  the D2O parameters are saved, salt-containing fits load and fix `mean_k` to
  that reference by default, with a direct override available.
- No low-Q DAB or Porod anchor is used. The default first trial fixes a very
  small `k_H_over_k` and a low threshold `b`, producing occupation close to
  one. A notebook switch selects a true line-only calculation instead.
- Extended the shared curve-fit evaluation and least-squares helpers with a
  `model_mode="line_only"` option that bypasses the heterogeneous mask and
  removes `k_H_over_k` and `b` from the nonlinear parameter vector.
- Centralized each fit parameter's initial value, bounds, and fixed/free state
  into one notebook table; fitting remains disabled until the D2O initial
  model has been tuned manually.
- Added `desmear_resolution_maximum_entropy(...)` to `curvefit/cf_tools.py`.
  It constructs a row-normalized Gaussian resolution operator from the
  pointwise `dQ` values and recovers a positive unsmeared curve by weighted
  maximum-entropy inversion. The implementation applies the convolution and
  entropy principles of Huang et al., J. Appl. Cryst. 58 (2025) 1355-1359,
  globally so the broad CA profiles do not have to be separated into isolated
  peaks.
- After inspection showed that direct desmearing was not sufficiently well
  behaved for the CA profiles, restored the measured intensity as the fitting
  observation. Added a reusable Gaussian resolution-matrix constructor and
  optional `resolution_sigma` inputs to the shared initial-model and
  least-squares helpers. Trial curves are now resolution smeared before their
  scale and regression loss are evaluated.
- Updated `curvefit/cf_ca.ipynb` to pass its fourth-column `dQ` values into the
  forward model. The centralized starting values now use `r_sigma_k = 0.2` to
  retain a visible line-spectrum peak and fixed `b = -2` for mask stability.
- Diagnosed the flattened high-Q initial curve by comparing identical
  unsmeared and smeared trials. The line-only model retained slopes of
  approximately `-1.00` unsmeared and `-0.98` smeared. Correcting the intended
  threshold to `b = -2` gives `p_H = 0.97725`, although the heterogeneous mask
  contribution still flattens the trial when `k_H_over_k = 5e-5`.
- Made `line_only` the recommended CA default while preserving the
  heterogeneous option. The initial-model plot now overlays unsmeared and
  resolution-smeared predictions and reports both fitted high-Q log-log
  slopes, making resolution effects directly auditable.
- Added `skew_normal_radial` as an alternative radial wave-number
  distribution. It uses independently controlled mean, standard deviation,
  and standardized skewness, applies a positivity truncation at `k = 0`, and
  supports both sampled and deterministic-quadrature spectrum construction.
- Added a CA notebook `K_DISTRIBUTION` selector between
  `skew_normal_radial` and `max_entropy_radial`, with skewness free by default
  over `(-0.9, 0.9)`. Distribution-specific filenames keep the two fit trials
  separate.
- The first D2O skew-normal trial converged to `mean_k = 0.0515459 A^-1`,
  `r_sigma_k = 0.174617`, and `skewness = 0.286953`, reducing the cost from
  about `60.15` for the saved maximum-entropy solution to `56.50`.
- Added `radial_k_distribution_density(...)` to `curvefit/cf_tools.py` and a
  third panel to both CA initial-model and fitted-result figures. The panel
  displays the selected fitted radial probability density over exactly
  `0 <= k <= 2*mean_k`, with the mean marked by a vertical dotted line.
- Added the normalized positive `bimodal_gaussian_radial` spectrum. Its free
  parameters are overall center of mass, center separation divided by that
  mean, major-component width divided by the mean, minor/major width ratio,
  and minor/major weight ratio. Both components are truncated at `k = 0`, and
  a shared location shift preserves the requested overall center of mass after
  truncation.
- The first D2O bimodal fit converged with cost `58.9232`, between the saved
  maximum-entropy (`60.15`) and skew-normal (`56.50`) trials. It found centers
  `0.0473924` and `0.0581520 A^-1`, widths `0.0075351` and
  `0.0057448 A^-1`, and normalized weights `0.63753` and `0.36247`. The two
  fitted components overlap enough that their summed density is not visibly
  double-peaked.
- Added optional positive pointwise `regression_weights` to the shared initial
  evaluator and nonlinear fitter. The CA notebook uses one editable multiplier
  to emphasize `0.10 <= Q <= 0.20 A^-1` while retaining the full fitting range,
  resolution smearing, and high-Q coefficient anchor.
- Guided the bimodal D2O trial toward a left major and right minor component
  using narrower physical bounds and an 8x feature-region regression weight.
  The unweighted RMS relative mismatch over the emphasized interval decreased
  from `0.05362` to `0.03301`; the fitted centers are `0.0485392` and
  `0.0670343 A^-1`, widths are `0.0055016` and `0.0047888 A^-1`, and normalized
  weights are `0.74602` and `0.25398`. Guided outputs use a separate filename
  tag so the original unconstrained bimodal trial is preserved.
- Added `spline_maxent_radial`, a compact-support positive radial spectrum with
  an exact prescribed mean. It combines a Gaussian width reference, a fixed
  `k**low_k_power` envelope, three natural-cubic-spline ordinates, and an
  internally solved exponential tilt. Sampling, deterministic quadrature, and
  plotting-density evaluation all use the same normalized construction.
- Added optional Gaussian parameter penalties to the shared nonlinear fitter
  and used them in the new `curvefit/cf_ca_spline.ipynb`. Its credibility-first
  D2O baseline uses the unweighted full-Q likelihood, `low_k_power=2`, support
  `[0.15, 2.5]*mean_k`, and weak zero-centered penalties on three spline
  coefficients. It converged at cost `57.1473`, with `mean_k=0.051425 A^-1`,
  relative standard deviation `0.17366`, and skewness `0.0553`. Only `0.112%`
  of the fitted probability lies below `0.5*mean_k`.
- A separate shoulder-seeded full-Q start returned essentially the same
  scattering residual and a smooth distribution. This indicates that a
  visibly separated second feature is not independently selected by the
  current full-Q objective; the stronger bimodality seen previously depends on
  explicit weighting of `0.10 <= Q <= 0.20 A^-1`.

## 2026-07-31 Affine-Anisotropy Effects and Gaussian Fits

- Added `rw_line_aniso.ipynb` as a self-contained study of an affinely
  transformed isotropic random-line spectrum. The notebook accepts a radial
  `k` distribution directly (default `gaussian_radial`) rather than loading a
  previously saved spectrum.
- For a real-space deformation `r' = F r`, implemented the scalar-spectrum
  mapping

  ```text
  I_F(Q n) = A_F I_0(Q sqrt(n^T F F^T n))
  ```

  and its powder average over direction `n`. The general tensor calculation
  uses angular quadrature; an incompressible uniaxial stretch
  `diag(lambda^(-1/2), lambda^(-1/2), lambda)` reduces exactly to a stable
  one-dimensional Gauss-Legendre integral.
- Derived and numerically tested a centered log-stretch expansion. With
  `delta(n) = log(sqrt(n^T F F^T n))`, expansion about
  `delta_0 = <delta>` gives

  ```text
  I_powder(Q) = A_F sum_m M_m D^m I_0(Q exp(delta_0)) / m!
  D = d/d(log Q),  M_m = <(delta-delta_0)^m>
  ```

  where `M_1 = 0`, so the leading anisotropic shape correction is second
  order. The notebook compares successive truncation orders with numerical
  powder averaging and includes a second panel showing relative loss. It uses
  the mixed real-space grid `r_min=1e-3/k_eff`, `r_max=250/k_eff`,
  `r_split=5/k_eff`, `Nr=3000`, and `Nr_small=600`.
- Added reusable numerical helpers to `curvefit/cf_tools.py`:
  `uniaxial_affine_powder_average(...)` and
  `uniaxial_affine_highq_factor(...)`. Extended the shared initial-model and
  least-squares routines with optional `affine_stretch=True`, a fitted
  `stretch_ratio`, stretched-Q support, numerical powder averaging, and the
  corresponding high-Q coefficient correction. The default isotropic paths
  remain unchanged.
- Generalized `compute_fit_orientation_correlations(...)` to accept the fitted
  radial `k_distribution` and its parameters. This allows the comparison
  workflow to use the Gaussian spectrum directly instead of implicitly
  assuming the older maximum-entropy/skewed spectrum.
- Added `curvefit/cf_ca_aniso.ipynb` for the salt-free Ca/D2O dataset. It uses
  one positive radial Gaussian, pointwise resolution smearing, and a slight
  incompressible uniaxial stretch. The centralized initial values are
  `r_sigma_k=0.17` and `stretch_ratio=1.2`; `RUN_FIT=False` is retained so the
  initial model can be inspected before fitting. No guide-Q fitting region or
  skew/multigaussian distribution is used.
- Added `curvefit/cf_yyw_aniso.ipynb` to batch-fit 55A, 55B, and 55C. Earlier
  fitted physical parameters seed each curve, while the old skewed spectrum is
  replaced by `gaussian_radial` and `stretch_ratio` is added as a nonlinear
  parameter. The production fit uses numerical powder averaging rather than
  the derivative series.
- Refit 55A with the requested `r_sigma_k <= 0.25` constraint. The saved result
  is `mean_k=0.01891675 A^-1`, `r_sigma_k=0.2078183`,
  `stretch_ratio=1.193519`, `k_H/k=0.0591402`, and `b=0.108607`. The fitted
  stretch ratios for 55B and 55C are `1.276470` and `1.266947`, respectively.
- Updated `curvefit/cf_compare_yyw.ipynb` to demonstrate the Gaussian-affine
  fits from `output/yyw/aniso`, report stretch instead of skewness, use the
  Gaussian spectrum in orientation calculations and renderings, and include
  the affine factor in high-Q physical-consistency checks. Expensive
  orientation and rendering cells remain unexecuted; the lightweight combined
  fit comparison was regenerated and validated.
- The affine calculation here is deliberately a scalar-spectrum
  postprocessing model. A fully material deformation of line elements would
  introduce the tangent-dependent weight `|F t|` and requires a
  tangent-resolved correlation or new Kac-Rice calculation; it cannot in
  general be reconstructed from the isotropic powder curve alone.
