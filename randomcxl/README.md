# Random Crosslinked Random-Wave Line Networks

This folder contains a PyVista/NumPy workflow for generating three real
random-wave scalar fields in a periodic cubic box and visualizing two retained
pairwise zero-line families:

- `Gamma_12`: points where `phi_1 = 0` and `phi_2 = 0`
- `Gamma_13`: points where `phi_1 = 0` and `phi_3 = 0`

The third pair, `Gamma_23`, is not rendered. Crosslink nodes are estimated where
the retained `(1,2)` and `(1,3)` vortex-line traces meet, corresponding to
triple-zero points `phi_1 = phi_2 = phi_3 = 0`.

![Crosslink mechanism schematic](schematic_cxl.png)

In the retained network, the red `Gamma_12` and green `Gamma_13` line families
share the `phi_1 = 0` surface. Where `phi_2` and `phi_3` vanish at the same
location on that surface, a triple-zero point acts as a crosslink between the
two line families. Since `Gamma_23` is omitted, each generic retained crosslink
has the local topology of two crossing line families rather than a third
rendered branch.

## Files

- `rw_line_network.py`: reusable script with field generation,
  k-vector sampling, vortex-line tracing, smoothing, crosslink detection, and
  PyVista rendering.
- `interactive_rw_line_network.ipynb`: notebook interface for tuning
  sampling, tracing, crosslink, and render settings.
- `figure_coupling/`: coupling sweep figures and the settings used to generate
  them.
- `schematic.ipynb`: notebook used to generate the schematic figures.
- `schematic_cxl.png`: schematic explaining the retained crosslink mechanism.
- `schematic_nocxl.png`: companion schematic without the crosslink highlight.
- `schematic_crosslink.png`: combined schematic crosslink visualization.

## Main Controls

The notebook organizes settings into three groups.

### Sampling settings

The fields are sampled on a cubic grid with normalized block coordinates
`r_tilde = (x/N, y/N, z/N)`, so one block side is effectively `L = 1`.

Important controls:

- `GRID_SIZE`: number of grid points along each direction in one sampled block.
- `NUM_BLOCK`: number of blocks per direction. `NUM_BLOCK = 1` uses the
  original single-box method; `NUM_BLOCK = 2` stitches a `2 by 2 by 2`
  expansion.
- `BLOCK_OVERLAP`: number of extra grid points evaluated around each block
  before cropping back to the block core.
- `RANDOM_SEED`: reproducible random seed.
- `NUM_MODES`: number of random-wave modes for `(phi_1, phi_2, phi_3)`.
- `K_DISTRIBUTION`: one of `single_shell`, `gaussian_radial`, `uniform_band`,
  or `user_list`.
- `K0`: per-field central wave number, in units of `k*L`.
- `r_SIGMA_K`, `r_K_MIN`, `r_K_MAX`: relative width and band limits, multiplied
  by each field's `K0`.
- `SHARED_K_VECTORS`: whether fields share sampled k-vectors.

For block-wise assembly, the recombined grid has side length
`(GRID_SIZE - 1)*NUM_BLOCK + 1`. The same random-wave coefficients are reused in
every block. Field normalization is estimated from the mode amplitudes
(`sum(A_n^2)/2`) rather than fitted from the sampled grid, so changing
`NUM_BLOCK` does not silently rescale the fields. Vortex and crosslink detection
are then performed on the recombined grid.

### Coupling phi_2 and phi_3

The script can tune the expected overlap of the two retained line families by
constructing `phi_2` and `phi_3` from two independent base random waves:

```python
phi_2 = (Sa + c*Sb) / sqrt(1 + c**2)
phi_3 = (Sa - c*Sb) / sqrt(1 + c**2)
```

Enable with:

```python
COUPLE_PHI2_PHI3 = True
PHI23_COUPLING_C = c
```

For iid Gaussian-like base waves, the expected pointwise correlation is:

```python
corr(phi_2, phi_3) = (1 - c**2) / (1 + c**2)
```

Thus `c = 0` makes `phi_2` and `phi_3` identical, while `c = 1` gives zero
correlation and behaves as independent in the Gaussian random-wave limit.

### Line tracing settings

When `USE_VORTEX_TRACING = True`, the code traces the zero lines using phase
winding of the complex fields:

```python
psi_12 = phi_1 + 1j*phi_2
psi_13 = phi_1 + 1j*phi_3
```

The raw phase-winding line segments are traced first. Crosslink candidates are
identified on the raw traces, then the displayed lines are smoothed and the
crosslink centers can be adjusted to the closest pair of smoothed segments.

Useful controls:

- `SMOOTH_VORTEX_LINES`: enable spline smoothing of traced lines.
- `VORTEX_FACE_PREFILTER`: compute phase winding only on faces where both
  scalar fields can plausibly cross zero.
- `VORTEX_FACE_ZERO_TOL`: relaxed zero-bracketing tolerance for the prefilter;
  larger values are safer for near-tangent cases but admit more faces.
- `VORTEX_SMOOTHING_SCALE`: interpolation density along smoothed lines.
- `VORTEX_TUBE_RADIUS`: rendered tube radius, in grid-coordinate units.
- `CROSSLINK_SEARCH_RADIUS`: distance used to find raw contacts between the
  retained line families.
- `CROSSLINK_MERGE_RADIUS`: distance below which candidate crosslinks are merged.
- `CROSSLINK_ADJUST_TO_SMOOTHED_LINES`: move detected nodes onto the smoothed
  line geometry.
- `CROSSLINK_BALL_RADIUS`: rendered crosslink sphere radius, in grid-coordinate
  units.

### Render settings

The notebook exposes controls for PyVista window size, camera zoom, bounding-box
visibility, surface colors/opacities, tube colors/opacities, and optional mesh
edge display on surfaces and tubes.

## Running

Use the `pyvista` conda environment:

```powershell
conda activate pyvista
cd C:\Users\ccu\Documents\codex_projects\project_randomcxl
python rw_line_network.py
```

For interactive exploration, open `interactive_rw_line_network.ipynb`
with the same environment selected as the notebook kernel.

## Coupling Sweep

The `figure_coupling/` folder contains a sweep over `PHI23_COUPLING_C`. Each
image uses the same notebook settings except for the coupling value. The
associated `settings.txt` records the expected correlation and detected
crosslink counts for each figure.
