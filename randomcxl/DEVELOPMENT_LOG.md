# RandomCXL Development Log

This log records implementation decisions and performance notes for the
random-wave crosslink visualization workflow.

## Core Model

- Three real random-wave scalar fields are generated: `phi1`, `phi2`, `phi3`.
- Only the retained pairwise zero-line families are rendered:
  - `Gamma_12`: `phi1 = 0`, `phi2 = 0`
  - `Gamma_13`: `phi1 = 0`, `phi3 = 0`
- `Gamma_23` is intentionally not constructed or rendered.
- Triple-zero points, `phi1 = phi2 = phi3 = 0`, are interpreted as crosslink
  nodes between the retained line families.

## K-Sampling

- `K0`, `r_SIGMA_K`, `r_K_MIN`, and `r_K_MAX` can be scalars or 3-tuples,
  allowing separate wave-number settings for the three fields.
- Relative controls are used:
  - `sigma_k = r_SIGMA_K*K0`
  - `k_min = r_K_MIN*K0`
  - `k_max = r_K_MAX*K0`
- Supported distributions:
  - `single_shell`
  - `gaussian_radial`
  - `uniform_band`
  - `user_list`
- `SHARED_K_VECTORS` controls whether the three fields share sampled k-vectors
  while retaining independent amplitudes/phases.

## Phi2/Phi3 Coupling

The optional coupling construction is:

```python
phi2 = (Sa + c*Sb) / sqrt(1 + c**2)
phi3 = (Sa - c*Sb) / sqrt(1 + c**2)
```

For iid Gaussian-like base waves:

```python
corr(phi2, phi3) = (1 - c**2) / (1 + c**2)
```

Interpretation:

- `c = 0`: `phi2` and `phi3` overlap exactly.
- `c = 1`: zero correlation, approximately independent in the Gaussian limit.

## Normalization

Field normalization is now estimated from the random-wave coefficients rather
than fitted to the sampled grid:

```python
std(phi) ~= sqrt(sum(A_n**2) / 2)
```

This avoids silently changing field amplitude when `NUM_BLOCK` changes the
sampled domain size.

## Vortex Tracing

- Vortex lines are traced from phase winding of:

```python
psi12 = phi1 + 1j*phi2
psi13 = phi1 + 1j*phi3
```

- This replaced direct isosurfaces of `phi_a**2 + phi_b**2 = epsilon**2`,
  which produced inflated or flattened tube-domain shells rather than clean
  line geometry.
- Optional spline smoothing follows the same spirit as the reference
  `Vortex.py` workflow.

## Face Prefilter

`VORTEX_FACE_PREFILTER` accelerates vortex detection by computing phase winding
only on plaquettes where both scalar fields can plausibly cross zero.

The relaxed mask uses:

```python
min(phi_face) <= VORTEX_FACE_ZERO_TOL
max(phi_face) >= -VORTEX_FACE_ZERO_TOL
```

Notes:

- `VORTEX_FACE_ZERO_TOL = 0.0` gives strict zero bracketing.
- A small positive tolerance keeps near-tangent cases and improves robustness.
- In tests with `GRID_SIZE=121`, the prefilter roughly halved raw tracing time
  while preserving line counts.

## Crosslinks

- Raw close contacts are found between the two retained vortex-line families.
- Candidate contacts are clustered into raw crosslink centers.
- Dense contacts can also be traced into ordered crosslink paths.
- Displayed crosslink sites can be sampled at a controlled spacing.
- Consecutive sites may be rendered as tubes with ball caps, or as individual
  balls.

Known caveat:

- For very low `PHI23_COUPLING_C`, `Gamma_12` and `Gamma_13` can overlap along
  extended paths. Crosslinks should then be interpreted as dense contact sites
  along line-like overlap regions, not isolated generic triple-zero nodes.

## Block-Wise Field Assembly

Large domains can be assembled from smaller blocks:

```python
NUM_BLOCK = 2
BLOCK_OVERLAP = 1
```

Definitions:

- `GRID_SIZE`: point count along one block side.
- `NUM_BLOCK`: number of blocks per direction.
- stitched side length:

```python
(GRID_SIZE - 1)*NUM_BLOCK + 1
```

Important behavior:

- The same random-wave coefficients are reused across blocks.
- Coordinates are scaled by the block size, so `kL` remains tied to one block
  of side length 1.
- Adjacent blocks share boundary planes.
- Vortex and crosslink detection are performed only after recombination.

Memory caveat:

- Current block-wise assembly reduces temporary field-generation memory but
  still stores the recombined fields before tracing. Final memory therefore
  remains `O(N^3)` for stitched side `N`.

## Performance Notes

Expected scaling:

- Wave generation: approximately `O(3*M*N^3)`.
- Raw vortex face scan: approximately `O(N^3)`.
- Output line geometry: closer to vortex arclength, roughly proportional to
  `k^2*volume`.
- Crosslink geometry can become large when the two retained line families
  overlap strongly.

Issue found:

- `NUM_BLOCK=2` initially took far more than the expected 8x scaling.
- Profiling showed the bottleneck was not raw vortex detection.
- Slow calls were:
  - repeated PyVista `merge(...)` while building many crosslink balls;
  - brute-force all-pairs endpoint checks while bridging crosslink paths.

Fixes:

- Crosslink balls/caps now use PyVista glyphing instead of repeated sphere
  merging.
- Endpoint bridging uses a `cKDTree` instead of nested all-pairs distance scans.

Observed timing with notebook-style settings:

```text
before optimization:
NUM_BLOCK=1:  1.68 s
NUM_BLOCK=2: 43.5 s

after optimization:
NUM_BLOCK=1:  1.38 s
NUM_BLOCK=2: 11.53 s
```

The optimized `NUM_BLOCK=2` case is close to the expected 8x volume scaling.

## Rendering Notes

- Slack-inspired colors are used:
  - red: `Gamma_12`
  - green: `Gamma_13`
  - yellow: crosslinks
- Camera is controlled by azimuth and polar angle.
- Lighting includes a headlight and a top light.
- Bounding box size is fixed by `BOX_SIZE_L`; if `None`, it uses the stitched
  grid side length minus one.

## Future Improvements

- True streaming tracing could avoid storing the full recombined grid.
- Periodic boundary stitching across opposite box faces is not yet implemented.
- Face prefiltering still computes the full phase array; deeper optimization
  could compute phase only on candidate faces.
- Crosslink interpretation near tangent contacts and extended overlaps could be
  refined with more geometry-aware criteria.

## 2026-05-29 Scattering Workflow Notes

Added a separate scattering module/notebook workflow:

- `rw_line_scattering.py`
- `interactive_rw_line_scattering.ipynb`

The workflow reuses the traced `Gamma_12` and `Gamma_13` line network from
`rw_line_network.py`, then evaluates orientationally averaged

```python
A(q) = sum_j w_j exp(i q dot r_j)
I(q) = |A(q)|^2
```

Important options added today:

- `SCATTERING_AMPLITUDE_METHOD = "points"`:
  uses discrete bead scatterers along traced lines.
- `SCATTERING_AMPLITUDE_METHOD = "line_segments"`:
  uses the analytic straight-segment integral
  `L * exp(i(qr1 + qr2)/2) * sinc((qr2 - qr1)/2)`.
- `LINE_SAMPLE_SPACING`:
  controls bead spacing along traced ordered polylines for point scattering
  and real-space bead visualization.
- `DYNAMIC_LINE_SAMPLE_SPACING`:
  allows coarser bead spacing at lower `Q`.
- `DYNAMIC_LINE_SAMPLE_EXPONENT`:
  controls the rule `Q^n * spacing = constant`.
- `DYNAMIC_LINE_SAMPLE_POWER2_SUBSETS`:
  samples the finest high-`Q` bead chain once and reuses power-of-two subsets
  for lower `Q`, avoiding full line resampling at every `Q`.

Visualization/debugging additions:

- The real-space scattering preview can hide line tubes and render the sampled
  bead scatterers as balls.
- Timing output was added to `compute_seed_averaged_scattering(...)` to report
  field building, vortex tracing, smoothing, sampling, scattering, and box
  reference costs.

Normalization and scale notes:

- Raw `Q` has units inverse grid-coordinate length because the phase is
  `exp(i q dot r)`.
- Notebook plotting can scale `Q` by the random-wave angular scale
  `2*pi*<k>/block_side`.
- `INTENSITY_NORMALIZATION = "i0"` divides by total scattering measure squared.
- `INTENSITY_NORMALIZATION = "length_density"` divides by
  `total_measure * (total_measure / box_volume)`.
- `total_measure` is the weighted bead count for point scattering, or weighted
  line length plus optional point weights for segment-integral scattering.

Finite-size/low-Q mean-density tools added:

- `SUBTRACT_EXPLICIT_BOX_MEAN = True` subtracts the analytic central-box mean
  amplitude:

```python
A = A_network - rho_mean * A_inner_box
```

- `ANALYTIC_MEAN_BUFFER_BLOCKS = B` adds an analytic mean-SLD surrounding
  shell of side `(2B + 1)L`:

```python
A_buffer = rho_mean * (A_outer_box - A_inner_box)
```

- `ANALYTIC_MEAN_BUFFER_MODE = "incoherent"` adds `|A_buffer|^2` as a low-Q
  diagnostic.
- `ANALYTIC_MEAN_BUFFER_MODE = "coherent"` adds amplitudes before squaring.
- `ANALYTIC_MEAN_BUFFER_NORMALIZE_TOTAL = True` normalizes using the explicit
  box plus analytic buffer mean measure. When `False`, normalization remains
  based on the explicit central box only.

Interpretation caveat:

- The analytic buffer corrects only the smooth mean-density/window component.
  It does not replace the missing random line correlations outside the explicit
  simulated box.

## 2026-06-01 Scattering Window Correction

Added smooth observation-window support directly on the line-network scattering
measure without voxelizing the density field.

Window controls:

- `SCATTERING_WINDOW = "none" | "tukey_box" | "hann_box" | "gaussian"`
- `SCATTERING_WINDOW_TAPER`
- `SUBTRACT_WINDOWED_MEAN`
- `WINDOW_MEAN_METHOD = "numeric_1d"`
- `WINDOW_NORMALIZATION = "windowed_measure"`

Implementation details:

- Point scattering now uses windowed bead weights:

```python
A(q) = sum_j b_j W(r_j) exp(i q.r_j)
```

- Segment scattering uses midpoint windowing:

```python
A_s(q) = b_s L_s W(r_mid) exp(i q.r_mid) sinc(q.dr/2)
```

- When `SUBTRACT_WINDOWED_MEAN` is enabled, the mean is subtracted at the
  complex-amplitude level:

```python
A_corr(q) = A(q) - rho_windowed * W_hat(q)
```

- `W_hat(q)` is computed as a separable product of cached numeric 1D
  quadratures:

```python
W_hat(qx, qy, qz) = Wx(qx) Wy(qy) Wz(qz)
```

- If a smooth window is active and `SUBTRACT_WINDOWED_MEAN=True`, the older
  rectangular-box mean subtraction via `SUBTRACT_EXPLICIT_BOX_MEAN` is ignored
  because it corresponds to a different observation window.

Normalization correction:

- For `WINDOW_NORMALIZATION = "windowed_measure"`, the finite-sample
  window-consistent normalization is now used.
- For point scattering:

```python
norm = sum_j b_j^2 W_j^2
```

- For midpoint segment quadrature, the analogous squared windowed segment
  weights are used.
- The previous no-window path remains unchanged when
  `SCATTERING_WINDOW="none"` and `SUBTRACT_WINDOWED_MEAN=False`.

Notebook note:

- `interactive_rw_line_scattering.ipynb` now uses the consistent windowed
  setting:

```python
s.SCATTERING_WINDOW = "hann_box"
s.SUBTRACT_WINDOWED_MEAN = True
s.WINDOW_NORMALIZATION = "windowed_measure"
```

- The scattering plot also includes FFT-grid reference lines and highlights the
  lowest FFT wavevector `2*pi/L_box`.

## 2026-06-05 Single-Chain Scattering Notebook

Duplicated the interactive line-network scattering notebook as
`interactive_rw_single_chain_scattering.ipynb`.

Purpose:

- Test one vortex-line family only, `S12 = {phi1 + i phi2 = 0}`, as a
  single-chain / one-complex-field scattering case.
- Start with a monochromatic random wave:

```python
r.K_DISTRIBUTION = "single_shell"
r.K0 = (3, 3, 3)
```

Implementation notes:

- The original structure builder still creates both `S12` and `S13`; the
  notebook defines a local `single_chain_structure(...)` helper that filters the
  generated structure down to `S12` points and segments.
- Seed averaging is performed by a notebook-local
  `compute_seed_averaged_single_chain_scattering(...)` helper.
- The comparison curve loads `smpl/rw_line_scattering.py` under a separate
  module name and computes the monochromatic Gaussian conditional-sampling
  line-correlation transform.
- Because the finite sampled-chain scattering and continuum Gaussian-sampling
  transform use different absolute normalizations, the Gaussian-sampling curve
  is scaled to the sampled-chain result at `MODEL_ANCHOR_Q_OVER_K`.
- Added execution-state prints with elapsed times in the single-chain notebook:
  `[build]`, `[scatter]`, `[model]`, `[plot]`, and `[save]` messages report
  preview structure construction, per-seed structure/scattering time,
  Gaussian-sampling reference stages, plotting, and output saving.
- The per-seed scattering step now runs in Q chunks controlled by
  `SCATTERING_Q_CHUNK_SIZE`, printing chunk index, elapsed time, and ETA so long
  orientational averages are visibly progressing.
- The single-chain comparison now labels and stores the comparison axis as
  `q_over_k_angular = q*GRID_SIZE/(2*pi*<k_cycles>)`. This matches the
  `rw_line_network.py` phase convention `cos(2*pi*k*x/GRID_SIZE + theta)` and
  the `smpl` Gaussian-sampling convention, where the monochromatic covariance is
  written with angular wave number `k` as `sin(k*r)/(k*r)`. The box length is
  still `GRID_SIZE-1` grid units; that is a finite-window length, not the wave
  coordinate scale.
- The single-chain notebook now distinguishes the wave-coordinate scale from
  the finite box length. The random-wave module uses
  `cos(2*pi*k*x/GRID_SIZE + theta)`, so the angular wave number in grid
  coordinates is `k_grid = 2*pi*<k_cycles>/GRID_SIZE`. The line-density estimate
  in grid units is therefore `rho0 = k_grid^2/(3*pi)`, and the box-length
  estimate is `L_est = rho0*V_box`.
- The single-chain notebook uses arclength point weights,
  `POINT_WEIGHT_MODE = "arclength"`, so point amplitudes approximate a line
  integral instead of a bead-count model. The module default remains `"unit"`
  for backward compatibility.
- The notebook no longer hard-codes the high-Q normalization as `pi/(Q*L)`.
  Each seed computes the same normalization quantities used by
  `structure_scattering_intensity_1d(...)`:

```python
M = sum_i w_i W_i
N = intensity normalization divisor
I_line(Q) ~ (pi*M/Q) / N
```

  With `INTENSITY_NORMALIZATION = "i0"` and
  `WINDOW_NORMALIZATION = "total_measure"`, this reduces to `pi/(Q*M)`. If the
  setting is changed back to `"windowed_measure"`, then `N = sum_i (w_i W_i)^2`
  and the plotted reference automatically follows that normalization instead.
- Superseded note: the Gaussian conditional-sampling comparison was briefly
  converted through an effective finite-chain volume `V_eff = M/rho_actual`.
  That was replaced by the window-volume spectral-density normalization below,
  because for smooth windows the fluctuating convolved line measure `M=sum wW`
  should be diagnostic output, not the intensity scale.
- The single-chain notebook default scattering window is now `hann_box`.
- The single-chain notebook now uses a window-volume intensity
  normalization for the reduced plots. The scattering calculation is run with
  `INTENSITY_NORMALIZATION = "none"` and the raw windowed intensity is converted
  after the fact:

```python
I(Q) = I_raw(Q) / int W(r)^2 dV
```

  This avoids normalizing by the fluctuating convolved line measure
  `sum_i w_i W_i`, which can drift substantially for a Hann window in one
  finite random realization. The theoretical line density is denoted:

```python
d = K^2/(3*pi)
N = d^2
K = 2*pi*<k_cycles>/GRID_SIZE
```

  The plots now use `Q/k` on the x axis and show both `I(Q)/d^2` and a
  high-Q guide-normalized ratio whose local-line limit should approach one
  without carrying an explicit box-size factor.
- The right-hand reduced panel is now labeled as the explicit ratio to the
  left-panel blue guide:

```python
left_y = I(Q)/d^2
left_guide = pi/(Q*d)
right_y = left_y / left_guide
```

  This is algebraically equivalent to `I(Q)*Q/(pi*d)`, but the ratio label is
  less ambiguous when the left panel already displays `I(Q)/d^2`.
- For Hann-windowed finite boxes, the absolute high-Q scale is now tied to the
  density actually sampled by the windowed line geometry:

```python
d_theory = K^2/(3*pi)
d = sum(ds * W^2) / int W^2 dV
N = d^2
```

  The theoretical `d_theory` is still printed as a reference. Using the
  window-squared measured density avoids attributing finite-realization density
  fluctuations to the Fourier normalization. The Gaussian reference is rescaled
  to the same chosen `d` before forming the reduced plots.
- The single-chain notebook x axis is labeled `Q/k` to match the lower-case
  convention used by the Gaussian sampling model. The default real-space
  scattering method was switched to `SCATTERING_AMPLITUDE_METHOD =
  "line_segments"` for the high-Q asymptote check. Point-sample scattering is
  still available for debugging, but at high enough Q it develops a discrete
  bead self-term floor controlled by `LINE_SAMPLE_SPACING`, so it should not be
  used as the primary test of the continuous-line `pi d / Q` behavior.
- Single-chain helper logic was moved from
  `interactive_rw_single_chain_scattering.ipynb` into `rw_line_scattering.py` to
  avoid duplicate notebook-local definitions. The module now provides:

```python
as_single_s12_structure(...)
single_chain_window_diagnostics(...)
compute_structure_scattering_chunked(...)
compute_seed_averaged_single_chain_scattering(...)
```

- Two notebook knobs are now module settings:

```python
POINT_WEIGHT_MODE = "unit" | "arclength"
SCATTERING_Q_CHUNK_SIZE = 10
```

  `POINT_WEIGHT_MODE="unit"` preserves the historical equal-bead point model.
  `"arclength"` assigns each sampled line point weight
  `LINE_SAMPLE_SPACING * family_weight`, so point amplitudes approximate a
  continuous line integral. `SCATTERING_Q_CHUNK_SIZE` controls progress chunks
  for long notebook runs and does not change the scattering formula.

## 2026-06-05 CXL-Chain Scattering Notebook

Added `interactive_rw_cxl_chain_scattering.ipynb` as the three-wave extension of
the single-chain scattering notebook.

Purpose:

- Keep both retained line families, `Gamma_12` and `Gamma_13`, as the simplest
  crosslinked-chain scattering test.
- Start from independent waves with `PHI23_CORRELATION_RHO = 0`.
- Allow the `phi_2`/`phi_3` correlation to be tuned through the same coupling
  construction used by `rw_line_network.py`.

The notebook follows the same style as
`interactive_rw_single_chain_scattering.ipynb`: matching parameter blocks,
`hann_box` windowing, `line_segments` as the default continuous-line scattering
method, `Q/k` on the x axis, and the two-panel reduced plot:

```python
left_y = I(Q)/d^2
left_guide = pi/(Q*d)
right_y = left_y / left_guide
```

For the CXL-chain case,

```python
d = d12 + d13
d12 = sum_{Gamma_12}(ds * W^2) / int W^2 dV
d13 = sum_{Gamma_13}(ds * W^2) / int W^2 dV
```

The Gaussian comparison follows the four-field setup from
`smpl/rw_4field_line_demo.ipynb`. With
`A = Gamma_12 = (phi_1, phi_2)` and
`B = Gamma_13 = (phi_1, phi_3)`, the four-field correlation parameters are:

```python
rho13 = 1.0
rho24 = PHI23_CORRELATION_RHO
```

Thus independent three-wave fields correspond to `(rho13, rho24) = (1, 0)`.
The total model comparison is assembled as
`I_total = I_AA + I_BB + 2*I_AB`. The self terms are scaled by the measured
window-squared densities `d12` and `d13`, while the cross term is scaled by
`sqrt(d12*d13)`, so the reduced total uses the same `d = d12 + d13` as the
trajectory calculation.

The notebook uses the same exposed Gaussian conditional-sampling parameter
style as the single-chain notebook:

```python
MODEL_R_MIN = 1e-3
MODEL_R_MAX = 5e2
MODEL_NR = 5000
MODEL_N_SAMP = 2**15
MODEL_TAIL_START_FRACTION = 0.8
```

The four-field comparison now passes `MODEL_N_SAMP` directly to
`make_qmc_normals(...)`, matching the two-wave conditional-sampling convention.
The Sobol power-of-two check is handled inside the shared helper.
Both `interactive_rw_single_chain_scattering.ipynb` and
`interactive_rw_cxl_chain_scattering.ipynb` now expose the same Gaussian
comparison parameter block:

```python
MODEL_R_MIN
MODEL_R_MAX
MODEL_NR
MODEL_N_SAMP
MODEL_TAIL_START_FRACTION
```

Neither notebook exposes a separate QMC-power variable.

Added module helpers in `rw_line_scattering.py`:

```python
as_single_s13_structure(...)
compute_seed_averaged_cxl_chain_scattering(...)
```
