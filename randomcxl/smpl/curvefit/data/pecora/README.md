# Pecora 1998 figure digitization

These files are a pixel digitization of the supplied figure from:

R. Borsali, H. Nguyen, and R. Pecora, “Small-Angle Neutron Scattering and
Dynamic Light Scattering from a Polyelectrolyte Solution: DNA,”
*Macromolecules* **31** (1998), 1548–1555.
DOI: [10.1021/ma970919b](https://doi.org/10.1021/ma970919b).

## Data files

- `pecora_1998_digitized.csv`: all series in long form, including source-pixel
  coordinates and a template-match score.
- `pecora_1998_<series>.csv`: two-column `q_A^-1,I_q` files for curve fitting.
- `pecora_1998_digitized_plot.png`: the reconstructed data-only plot.
- `pecora_1998_digitization_overlay.png`: marker centers over the source image.
- `pecora_1998_figure.png`: full-resolution supplied image.
- `digitize_pecora.py`: reproducible extraction script.

The five series correspond to salt-free, 10 mM, 50 mM, 0.5 M, and 1 M added
salt.

## Calibration and limitations

The plot frame was calibrated as `q = 0 ... 0.22 Å^-1` and
`I(q) = 0 ... 0.50`. Printed marker glyphs were located by normalized template
matching and constrained to hand-checked curve centerlines where multiple
series overlap.

This is digitized, not original instrument data. For isolated symbols, the
pixel resolution corresponds roughly to ±0.0004 Å^-1 in `q` and ±0.003 in
`I(q)`. Uncertainty can be larger in the crowded high-q region. The
`match_score` column is a detection diagnostic, not a physical uncertainty.

Regenerate the outputs from this directory with:

```powershell
python -B digitize_pecora.py
```
