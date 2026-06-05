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
