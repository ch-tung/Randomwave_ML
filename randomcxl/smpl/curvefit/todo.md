# Next TODO — production analysis and finite-cutoff interaction inference

## Reference-condition option — 2026-08-24

- [x] Added `REFERENCE_CONDITION = "zero"` or `"highest"` to both active
  notebooks. Zero salt remains the primary analysis; highest salt is a
  separate comparison check.
- [x] Highest-salt calculations use their own output and trace-cache folder,
  so they cannot overwrite zero-salt results.
- [x] All coefficient changes, normalized variables, distribution fits, and
  optional distant-interaction corrections use the selected condition's
  (k_{\rm ref}), sample, and trace consistently.
- [x] Report labels show the actual reference concentration and use
  (g/g_{\rm ref}), (D/D_{\rm ref}), and (r/D_{\rm ref}) when the
  reference is not zero salt.
- [ ] Run the highest-salt production comparison and assess whether reversing
  the comparison improves sample overlap and the measured-versus-fit curves.

## Conservative interaction reporting — 2026-08-24

- [x] Overlay the measured distribution and the fits before and after the
  distant-interaction correction on the same panels.
- [x] Require both checks to support a shape-dependent correction: the distant
  interaction must vary on the reference contours, and the actual
  reference-to-target interaction change must remain predictable from local
  shape. Keep a forced correction only as an explicit comparison run.
- [x] Make measured structural changes the primary result:
  \(k_{\rm eff}\), spectral width, and the distributions/moments of
  \(\kappa,\kappa',\kappa\tau\), with uncertainties.
- [x] Add salt-dependent correlations between the magnitudes of
  \(\kappa,\kappa',\kappa\tau\), evaluated on the common stable range. Signed
  correlations are not the primary measure because symmetry can make them
  vanish even when the magnitudes are coupled.
- [x] Include \(P(\kappa')\) in the held-out measured-versus-fit checks before
  interpreting the two coefficient summaries.
- [ ] Report the fitted local free-energy change as the next level of result.
  Compare the two-term expression from the note, separate fourth-order terms,
  and the broader fourth-degree fit on held-out samples.
- [ ] Report \(\Delta c_2,\Delta c_4\) as interaction-moment summaries only
  where the two-term expression reproduces the measured distributions. Where
  it fails, show the unrepresented part explicitly rather than mapping the two
  coefficients to a unique potential.
- [ ] Keep (g,D) curves as conditional illustrations unless independent
  information fixes the reference interaction and the local expression passes
  the distribution checks.

## Audit status — 2026-08-20

This list was checked against the current cf_ca_interaction.py,
the compute/results notebooks, and test_cf_ca_interaction.py, taking
../writings/note_interaction.tex as the scientific authority. No production
calculation is required to resolve the implementation items below.

Current compatible pieces:

- the local state is \(X=\{\kappa,\kappa',\kappa\tau\}\);
- the implemented linked density is
  \(9\kappa^4/64-(\kappa')^2/24-\kappa^2\tau^2/24\);
- the SVD diagnostic centers the response and basis, and grouped cross
  validation keeps contour groups together;
- the conditional model uses
  \(-\log\langle e^{-\beta\Delta H_{\rm nl}}\mid X\rangle_0\), and the
  direct maximum-entropy base weights now contain the required factor
  \(\exp[-\beta\Delta F_{\rm nl}(X)]\);
- reusable definitions remain in the Python module rather than the notebook.
- all calculations now live in `cf_ca_interaction_compute.ipynb`: local
  sampling and fitting first, followed by separately controlled expansion-range,
  uncertainty, trace, absolute nonlocal-dependence, contour-integral, and
  convergence checks;
- final coefficient and conditional interaction-variation plots can be
  regenerated from saved CSV inputs in `cf_ca_interaction_results.ipynb`
  without sampling local geometry or loading traces.
- the superseded interaction notebooks are retained under
  `archive/legacy_interaction_notebooks/`, leaving exactly one active compute
  notebook and one active results notebook.

Blocking differences that remain in the current code:

- [x] The finite-cutoff functions \(f_2,f_4\), implicit positive-root inversion,
  signed interaction-change inversion, numerical covariance propagation,
  notebook diagnostics, and iterative nonlocal kernels now use the
  finite-cutoff mapping.
- [x] The nonlocal diagnostic now has separate relative-cell and absolute
  \(\epsilon_{\rm nl}(s)\) APIs. The notebook now uses the direct
  interaction-change cell Hamiltonian without an absolute reference kernel.
- [x] A single inherited zero-salt normalization is no longer used in notebook
  outputs. \(g_\Delta,D_\Delta\) are inferred directly from
  \(\Delta c_2,\Delta c_4\), and \(D_\Delta\) is re-inferred after every
  nonlocal free-energy update until the coefficients and kernel converge.
- [x] The conditional absolute-Yukawa visualization has been restored as a
  common-reference family rather than a single assigned normalization. It
  plots the normalized potential envelope together with a simple earlier-style
  \(g/g_0\) trend and
  \(D/D_0\), evaluates the finite-cutoff factors using each target
  \(\ell_i=1/k_{{\rm eff},i}\), displays \(D_i k_{{\rm eff},i}\), and exports
  the full reference grid and mean/standard-deviation summary. No fixed
  \(g_0/k_{\rm ref}\) or \(D_0k_{\rm ref}\) enters the reported curve.
- [x] The notebook now plots each target \(\kappa\) and signed \(\tau\)
  distribution against the zero-salt distribution reweighted by the fitted
  \(\Delta c_2,\Delta c_4\), on the retained fit support with weighted KS
  diagnostics.
- [x] The final reconstruction report uses the fourth-degree local fit by
  default and shows only measured and fitted \(P(\kappa)\) and
  \(P(\kappa\tau)\). The reported \(\Delta c_2,\Delta c_4\) are the weighted
  projection of this fitted log-density change onto the note's \(K_2,K_4\)
  form. The projection error is saved and warned on when large.
- [x] Report figures use at most two columns and contain no panel titles,
  figure titles, or explanatory annotations beyond necessary axes, ticks, and
  legends. The \(Dk_{\rm eff}\) panel is no longer reported.
- [ ] Replace the notebook's broad exploratory \(g_0/k_{\rm ref},D_0k_{\rm ref}\)
  bounds with physically justified prior bounds before interpreting the
  conditional envelopes quantitatively.
- [x] The currently saved nonzero fitted conditions have
  \(\Delta c_2\Delta c_4<0\). Since \(f_2,f_4>0\), none admits the requested
  single effective Yukawa change. The optional correction therefore uses an
  explicitly conditional two-range signed Yukawa sum, with both ranges saved
  and exposed for sensitivity checks. This is a numerical continuation, not a
  unique inferred potential or a real single \(D_\Delta\).
- [ ] The current
  \(\langle e^{-\beta\Delta H_{\rm nl}}\mid X\rangle_0\) estimator is a ridge
  regression of positive weights followed by empirical clipping. It needs
  grouped held-out validation, a positivity-preserving fit, and comparison with
  the first-cumulant approximation before it can be treated as a controlled
  estimate of \(\Delta F_{\rm nl}\).
- [x] The active compute notebook can now apply a self-consistent
  \(\Delta F_{\rm nl}(X)\) correction without an absolute zero-salt potential.
  It saves corrected base weights, corrected coefficients, held-out
  reconstruction tables, conditional-model diagnostics, and fixed-point
  status. The absolute zero-salt dependence result is the default shared gate;
  a forced shape-dependent correction is available only as a labelled
  sensitivity override. The switch remains off until the production reference
  trace and kernel-range sensitivity calculation are run.
- [x] The linked fourth-order invariant now has a residual-force check and
  warning based on \(\ell^6\beta V'(\ell)/1152\). The code reports the residual
  fraction but continues with the linked model so the warning can be assessed.
- [x] Reduced deterministic tests now cover the finite factors, synthetic
  inversion, condition-specific cutoffs, unphysical roots, separate nonlocal
  response scaling, default diagnostic mode, the linked-model warning, and
  normalization of the conditional Yukawa reference family.

Implementation order: complete Sections 1, 2, 3, and 4 below with reduced
deterministic tests; then perform the production work in Section 5 onward.

## Anisotropic geometry — 2026-08-21

- [x] Added an off-by-default geometry switch independent of the scattering
  fit source. With the switch off, the original local sampling path and output
  directory are unchanged.
- [x] Apply the fixed fitted Hencky tensor through (F=\exp H), recompute the
  tangent and curve jet with physical arclength, and include the exact line
  factor in the coarea weights before fitting the fourth-degree density and
  projecting to \(\Delta c_2,\Delta c_4\).
- [x] Accept either the fitted uniaxial `stretch_ratio` convention or explicit
  symmetric traceless 3-by-3 tensors supplied per salt condition.
- [x] Transform cached isotropic contours, reparameterize them by physical
  arclength, and write separate `_aniso` trace caches for contour and nonlocal
  checks.
- [x] Keep the existing result figures and add a separate two-panel plot of
  the principal stretch and Hencky-strain inputs.
- [ ] Run the production local sample with anisotropic geometry and assess the
  held-out \(P(\kappa)\), \(P(\kappa\tau)\), and linked-projection warnings.
- [ ] If the optional nonlocal correction is used for final inference, run it
  on the transformed physical trace for each target tensor and complete the
  existing convergence checks.

## 1. Replace the short-range Yukawa conversion by the finite-cutoff mapping

Implement the manuscript relations

\[
a
=
c_2-\frac{\ell_{p,0}}{2}
=
\frac{gD^2}{8}
f_2\!\left(\frac{\ell}{D}\right),
\]

\[
c_4
=
gD^4
f_4\!\left(\frac{\ell}{D}\right),
\]

with

\[
f_2(u)
=
1-\frac{e^{-u}}{3}
\left(u^2+3u+3\right),
\]

\[
f_4(u)
=
1-\frac{e^{-u}}{30}
\left(
u^4+5u^3+15u^2+30u+30
\right).
\]

For every condition use its fitted

\[
\ell=\frac{1}{k_{\rm eff}}.
\]

Solve

\[
D^2
=
\frac{c_4}{8a}
\frac{f_2(\ell/D)}{f_4(\ell/D)}
\]

numerically for \(D\), then obtain

\[
g
=
\frac{8a}
{D^2 f_2(\ell/D)}.
\]

Keep the \(f_2=f_4=1\) result only as a short-range limiting check.

- [x] Added an off-by-default results-only switch that exposes this legacy
  short-range mapping as a clearly labelled conditional result. It writes
  separate figures and tables and does not replace the finite-cutoff default.

Add tests that:
- recover the short-range formula for \(D/\ell\rightarrow0\);
- recover synthetic finite-\(\ell\) inputs;
- reject unphysical roots or parameter combinations.

Implementation details:

- evaluate \(f_2(u)\) and \(f_4(u)\) stably for both small and large \(u\)
  (use series or expm1 handling where direct subtraction loses precision);
- solve in a positive dimensionless variable such as \(D/\ell\), bracket the
  physical root, and report failure, non-uniqueness, or a large residual rather
  than silently accepting a numerical root;
- pass both the reference cutoff \(\ell_0=1/k_{{\rm eff},0}\) and each target
  cutoff \(\ell=1/k_{\rm eff}\) through the relative mapping;
- distinguish the local expansion range used for density-ratio
  inference from the condition-specific physical cutoff used to invert
  \((c_2,c_4)\mapsto(g,D)\);
- update covariance propagation for the implicit root, preferably by implicit
  differentiation checked against finite differences or by bootstrap;
- update the nonlocal iteration and notebook plots to use the same finite-cutoff
  mapping, so the diagnostic figures and correction kernel cannot disagree.

## 2. Recast the nonlocal-dependence test to match the manuscript exactly

The manuscript test is an absolute nonlocal-energy dependence test,

\[
\epsilon_{\rm nl}(s)
=
\int_\ell^\infty d\Delta s\,
\mathcal V[d(s,\Delta s),\Delta s],
\]

performed on the zero-salt reference geometry.

Separate this diagnostic clearly from the relative correction workflow:

1. evaluate \(\epsilon_{\rm nl}\) for representative \(D\sim1/k_{\rm eff}\);
2. measure its \(X\)-dependence with the existing SVD / grouped-CV machinery;
3. only if appreciable dependence is found, activate the existing target-minus-reference \(\Delta\mathcal H_{\rm nl}\) correction.

Do not use the relative correction itself as the locality test.

Implement separate, explicitly named data paths:

- absolute_nonlocal_energy_density for the zero-salt
  \(\epsilon_{\rm nl}(s)\) locality diagnostic; do not multiply it by the local
  expansion range;
- relative_nonlocal_cell_hamiltonian for
  \(\beta\Delta H_{\rm nl}\), including the cell integration needed by the
  local density-ratio model.

Compute the locality decision once from the absolute diagnostic over a
documented grid around \(Dk_{{\rm eff},0}\sim1\), with amplitude choices also
reported because the absolute energy scale changes with \(g\). Do not remake
this decision separately from each target-minus-reference table.

## 3. Validate the conditional nonlocal free-energy estimate

When the absolute locality test requires a correction, estimate

\[
\beta\Delta F_{\rm nl}(X)
=
-\log\left\langle e^{-\beta\Delta H_{\rm nl}}\mid X\right\rangle_0
\]

on the zero-salt reference configurations. Preserve
-beta_delta_free_energy as the logistic offset; the current sign is correct.

- fit and assess the conditional expectation using contour-grouped held-out
  folds or cross-fitting;
- use a positivity-preserving parameterization rather than relying on clipping
  negative polynomial predictions after the fit;
- report conditional-weight overlap, effective sample size, clipping (if any),
  and held-out error;
- compare the exact log-exponential estimate with the leading first-cumulant
  form
  \(\mathrm{const}+\langle\Delta H_{\rm nl}\mid X\rangle_0\);
- retain the first-cumulant shortcut only when their \(X\)-dependent parts agree
  within the statistical uncertainty relevant to \(\Delta c_2,\Delta c_4\);
- add deterministic synthetic tests for a constant correction, a known
  \(X\)-dependent conditional Boltzmann factor, sign convention, and grouped
  out-of-sample behavior.

## 4. Check the finite-cutoff validity of the linked fourth-order invariant

For every reference/target candidate interaction, evaluate the residual-force
boundary contribution identified in the note,

\[
\frac{\ell^6\beta V'(\ell)}{1152},
\]

and compare its magnitude with the retained fourth-order coefficient and its
uncertainty. Record a quantitative tolerance.

- If negligible, retain the linked \(K_4\) model and report the diagnostic.
- If not negligible, do not absorb the discrepancy into a single fitted
  \(c_4\). Use the unreduced independent fourth-order coefficients or explicit
  boundary term, then repeat the held-out linked-model comparison.

## 5. Run the production zero-salt contour ensemble

The current implementation has only passed reduced smoke tests.

Run the full zero-salt reference trace with production settings and save:
- traced contours;
- \(X(s)=\{\kappa,\kappa',\kappa\tau\}\);
- nonlocal chord tables \(d(s,\Delta s)\);
- cell-level invariants;
- contour-level \(K_2,K_4\);
- replicate / field-realization identifiers.

Check convergence with:
- field resolution;
- number of random-wave modes;
- tracing resolution;
- number of traced contours;
- contour length.

This becomes the fixed reference sample used by all target conditions.

## 6. Perform the production nonlocal-dependence test

Using the production zero-salt contours:

- evaluate the dependence test for representative
  \(Dk_{\rm eff}\) values around unity;
- report both:
  - in-sample SVD retained variance;
  - grouped-contour cross-validated \(R^2\);
- establish a numerical decision criterion for whether the nonlocal correction is required.

The result should be reported as a diagnostic, not assumed in advance.

## 7. Run finite-tail convergence

The current implementation supports tail convergence but it has not been run at production scale.

Increase the upper contour separation in

\[
\int_\ell^{\Delta s_{\max}}
d\Delta s
\]

until:
- \(\epsilon_{\rm nl}\) dependence metrics converge;
- the conditional \(\Delta F_{\rm nl}(X)\) correction converges when used;
- the inferred \(\Delta c_2,\Delta c_4\) are stable.

Record the minimum converged \(\Delta s_{\max}\).

## 8. Run the full interaction inference for every salt condition

For each target salt condition:

1. load its fitted random-wave spectrum;
2. obtain its target local geometric statistics;
3. start from the local-only \(\Delta c_2,\Delta c_4\) fit;
4. if required by the dependence test, construct the parametrized target interaction;
5. evaluate
   \[
   \Delta\mathcal H_{\rm nl}
   =
   \mathcal H_{\rm nl}
   -
   \mathcal H_{{\rm nl},0};
   \]
6. estimate
   \[
   \beta\Delta F_{\rm nl}(X)
   =
   -\log
   \left\langle
   e^{-\beta\Delta\mathcal H_{\rm nl}}
   \mid X
   \right\rangle_0,
   \]
   and compare it with the first-cumulant approximation specified in Section 3;
7. refit \(\Delta c_2,\Delta c_4\);
8. iterate until the coefficients and nonlocal correction converge.

Save both the local-only and corrected results.

## 9. Decide whether the local-cell approximation is sufficient

The code currently retains the pointwise correlation-cell treatment as the default and has an optional contour-integrated calculation.

Run both at production scale and compare:

\[
(\Delta c_2,\Delta c_4)_{\rm cell}
\]

against

\[
(\Delta c_2,\Delta c_4)_{\rm contour}.
\]

Quantify the difference relative to bootstrap uncertainty.

If the difference is small, retain the cell approximation as the primary result.

If not, use the contour-integrated result or introduce the discrepancy explicitly as a correlation correction.

## 10. Test whether the linked \(K_2,K_4\) model is actually sufficient

The previous held-out analysis showed degradation of the linked two-coefficient reconstruction at higher salt.

For every target condition compare:

- linked \(K_2,K_4\) model;
- independent fourth-order invariants;
- flexible even-polynomial density-ratio model.

Evaluate on held-out data:
- likelihood / density-ratio loss;
- reconstruction of \(\kappa\);
- reconstruction of \(\kappa'\);
- reconstruction of \(\kappa\tau\);
- relevant chord statistics.

The manuscript's two-coefficient interpretation should only be used where the linked model is statistically adequate.

## 11. Check reweighting support

For every salt condition report:
- effective sample size;
- maximum / distribution of weights;
- overlap of target and zero-salt \(X\)-support;
- sensitivity to clipping or regularization;
- stability across independent zero-salt realizations.

Flag conditions for which the target distribution requires extrapolation beyond the reference ensemble.

## 12. Resolve the zero-salt interaction scale

The primary identifiable quantities remain

\[
\Delta c_2,\qquad \Delta c_4.
\]

Absolute or relative \(g,D\) still require the reference quantities

\[
a_0
=
c_{2,0}-\frac{\ell_{p,0}}{2},
\qquad
c_{4,0},
\]

or equivalent information about \(g_0,D_0\).

Do not retain a single inherited normalization such as
\(g_0/k_{\rm ref}=5\),
\(D_0k_{\rm ref}=1\)
as the final result.

Implementation status: the primary notebook path no longer uses these absolute
reference quantities. It constructs an effective \(\Delta V\) directly from
\(\Delta c_2,\Delta c_4\). Absolute reference inputs remain relevant only if a
separate absolute \(g,D\) interpretation is later requested.

Instead:
- define a physically justified range of reference inputs;
- propagate each reference choice through the finite-cutoff \(f_2,f_4\) inversion;
- determine which trends in \(D/D_0\) and \(g/g_0\) are invariant across that range.

If no independent constraint on \(\ell_{p,0}\) or the zero-salt interaction is available, report \(\Delta c_2,\Delta c_4\) as the primary inference and \(g,D\) only as conditional mappings.

## 13. Production uncertainty propagation

Run the existing replicate/bootstrap machinery at full scale.

Propagate uncertainty from:
- fitted random-wave spectral parameters;
- finite random-wave sampling;
- contour tracing;
- density-ratio fitting;
- \(\Delta F_{\rm nl}\) estimation;
- finite-cutoff \(D\) root solving;
- reference interaction parameters.

Report covariance between \(\Delta c_2\) and \(\Delta c_4\), not only independent error bars.

## 14. Final scientific outputs

Produce one table per salt condition containing:

- \(k_{\rm eff}\);
- spectral width;
- nonlocal-dependence SVD fraction;
- grouped-CV dependence statistic;
- whether \(\Delta F_{\rm nl}\) was retained;
- \(\Delta c_2\);
- \(\Delta c_4\);
- cell vs contour difference;
- effective sample size;
- linked-model held-out quality;
- \(D/D_0\) and \(g/g_0\), only when sufficiently constrained.

Produce final diagnostic figures for:
- target vs reweighted local geometric distributions;
- \(\Delta c_2\) and \(\Delta c_4\) vs salt;
- nonlocal-dependence measure vs salt / tested interaction range;
- cell vs contour inference;
- finite-tail convergence;
- linked vs flexible model performance;
- finite-cutoff interaction mapping;
- reference-normalization sensitivity of \(D/D_0\) and \(g/g_0\).

## Direct linked inference update — 2026-08-21

Completed in the fast local-cell workflow:

- report \(\Delta c_2,\Delta c_4\) from direct maximum-entropy matching of
  \(K_2,K_4\), rather than projecting a flexible density-ratio fit;
- validate the actual linked reweighting on held-out \(P(\kappa)\) and
  \(P(\kappa\tau)\);
- save fixed-\(k_{\rm ref}\) spectral and geometric moment trends;
- decompose the fourth-order moment into its curvature, curvature-gradient,
  and torsional contributions on a fixed zero-salt support window;
- fit the three fourth-order invariants independently and report the fraction
  outside the note's linked direction;
- retain the flexible polynomial only as a model-quality diagnostic;
- use the same direct maximum-entropy estimator in the optional expansion-range,
  replicate-bootstrap, and contour-integrated coefficient workflows.

Pending production/scientific decisions:

- run the full local sample for both geometry modes and decide whether the
  linked held-out reconstruction is adequate;
- if the linked fit fails while the independent fit succeeds, decide whether
  the note should retain independent fourth-order residuals or treat them only
  as an inadequacy diagnostic;
- propagate scattering-fit parameter covariance; current fit tables do not
  provide the joint covariance needed for this step;
- run block/contour-integrated inference and block-length convergence;
- run the nonlocal-dependence test and add \(\Delta F_{\rm nl}\) only when its
  dependence on local geometry is supported;
- consider a jointly smooth salt trend only after inspecting the raw estimates;
  do not impose monotonicity merely to create a systematic curve.

## Adjacent-state inference update — 2026-08-24

Implemented:

- use consecutive salt pairs as the default local inference, with each pair's
  own fitted spectrum, anisotropy tensor, (k_{m eff}), and local cutoff;
- retain the former single-reference calculation as `COMPARISON_MODE =
  "single_reference"` without changing the fit result classes or established
  coefficient columns;
- save pair identifiers, source and target salt, source-scale coefficients,
  and common-reporting-scale coefficients as metadata;
- infer one finite-cutoff interaction-change kernel for each step and sum the
  kernels exactly to obtain the cumulative real-space interaction change;
- apply the optional distant-interaction correction on the traced source
  geometry of every adjacent transition, then save corrected stepwise and
  cumulative profiles;
- implement optional direct lowest-to-highest closure comparisons with
  one-standard-deviation bands for both the local and corrected workflows.

Pending production checks:

- regenerate or load every source-state trace needed by the adjacent chain;
- run the full anisotropic local inference and inspect overlap, effective
  sample size, and held-out κ, κ′, and κτ reconstruction for every step;
- run the corrected adjacent workflow and verify fixed-point convergence for
  every transition;
- set `RUN_LONG_BASELINE_CLOSURE = True` and compare the direct endpoint
  profile with the adjacent sum; do not interpret the closure as successful
  unless their uncertainty bands are compatible over the reported separation
  range;
- assess correlations between adjacent fitted steps before replacing the
  current independent-step uncertainty accumulation in a final report.
