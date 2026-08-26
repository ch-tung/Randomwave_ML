"""Reusable local-interaction inference tools for the Ca random-wave series.

The scientific definitions follow ``../writings/note_interaction.tex``. This
module has no import-time analysis or filesystem/plotting side effects.

The default ``local_cell_features`` path interprets one sampled local state over
the local expansion range ``ell=1/k_ref``. The optional
``contour_features`` path evaluates the contour-integrated K2 and K4 from the
note. Pair-potential strengths and distances must use the same physical length
unit as the trace; integrating the pair kernel over both contour coordinates
then gives a dimensionless ``beta*H``.
"""

from __future__ import annotations

import csv
import importlib
import importlib.util
import re
import sys
import warnings
from dataclasses import dataclass, field, replace
from itertools import combinations_with_replacement, product
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline
from scipy.linalg import expm
from scipy.optimize import brentq, minimize
from scipy.special import gammainc, gammaln, logsumexp
from scipy.stats import gamma as gamma_distribution
from scipy.stats import norm, qmc

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class FitSpectrum:
    concentration_mM: float
    k_eff: float
    mean_k: float
    r_sigma_k: float
    source: Path | None = None
    stretch_ignored: float = np.nan
    label: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.concentration_mM) or self.concentration_mM < 0:
            raise ValueError("concentration_mM must be finite and nonnegative")
        if not np.isfinite(self.k_eff) or self.k_eff <= 0:
            raise ValueError("k_eff must be finite and positive")
        if not np.isfinite(self.mean_k) or self.mean_k <= 0:
            raise ValueError("mean_k must be finite and positive")
        if not np.isfinite(self.r_sigma_k) or self.r_sigma_k <= 0:
            raise ValueError("r_sigma_k must be finite and positive")


@dataclass(frozen=True)
class InteractionComparisonPair:
    """One directed structural comparison without changing fit result types."""

    source_concentration_mM: float
    target_concentration_mM: float
    pair_id: str
    step_index: int
    comparison_mode: str
    consistency_check: bool = False

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "source_concentration_mM": self.source_concentration_mM,
            "target_concentration_mM": self.target_concentration_mM,
            "pair_id": self.pair_id,
            "step_index": self.step_index,
            "comparison_mode": self.comparison_mode,
            "consistency_check": self.consistency_check,
        }


@dataclass(frozen=True)
class LocalGeometrySample:
    """Coarea-weighted samples of X=(kappa,kappa_prime,kappa*tau)."""

    kappa: FloatArray
    kappa_prime: FloatArray
    kappa_tau: FloatArray
    weights: FloatArray
    k_eff: float
    stable_fraction: float = 1.0
    mean_jacobian_dimensionless: float = np.nan

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(value, dtype=float) for value in (
            self.kappa, self.kappa_prime, self.kappa_tau, self.weights
        ))
        size = len(arrays[0])
        if size == 0 or any(array.ndim != 1 or len(array) != size for array in arrays):
            raise ValueError("sample arrays must be nonempty 1D arrays of equal length")
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("sample arrays must be finite")
        if np.any(arrays[0] < 0) or np.any(arrays[3] < 0) or np.sum(arrays[3]) <= 0:
            raise ValueError("curvature and weights must be nonnegative with positive mass")
        if not np.isfinite(self.k_eff) or self.k_eff <= 0:
            raise ValueError("k_eff must be finite and positive")
        object.__setattr__(self, "kappa", arrays[0])
        object.__setattr__(self, "kappa_prime", arrays[1])
        object.__setattr__(self, "kappa_tau", arrays[2])
        object.__setattr__(self, "weights", arrays[3] / np.sum(arrays[3]))

    @property
    def state(self) -> FloatArray:
        return np.column_stack((self.kappa, self.kappa_prime, self.kappa_tau))

    @property
    def fourth_order_density(self) -> FloatArray:
        return local_fourth_order_density(self.kappa, self.kappa_prime, self.kappa_tau)


@dataclass(frozen=True)
class TraceConfig:
    grid_size: int = 64
    num_blocks: int = 1
    num_modes: int = 256
    trace_k0: float = 5.0
    random_seed: int = 894894
    seed_spacing: float = 0.35
    q_spacing: float = 0.08

    def __post_init__(self) -> None:
        if self.grid_size < 8 or self.num_blocks < 1 or self.num_modes < 8:
            raise ValueError("trace counts are too small")
        if self.trace_k0 <= 0 or self.seed_spacing <= 0 or self.q_spacing <= 0:
            raise ValueError("trace scales must be positive")


@dataclass(frozen=True)
class ContourTrace:
    points: tuple[FloatArray, ...]
    tangents: tuple[FloatArray, ...]
    r2: tuple[FloatArray, ...]
    r3: tuple[FloatArray, ...]
    k_eff: float
    q_spacing: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        count = len(self.points)
        if count == 0 or not (len(self.tangents) == len(self.r2) == len(self.r3) == count):
            raise ValueError("trace component lists must be nonempty and aligned")
        for components in zip(self.points, self.tangents, self.r2, self.r3):
            arrays = [np.asarray(value, dtype=float) for value in components]
            if len({len(value) for value in arrays}) != 1 or len(arrays[0]) == 0:
                raise ValueError("each contour must have aligned nonempty arrays")
            if any(value.shape != (len(arrays[0]), 3) for value in arrays):
                raise ValueError("trace arrays must have shape (n,3)")
        if self.k_eff <= 0 or self.q_spacing <= 0:
            raise ValueError("trace k_eff and q_spacing must be positive")

    @property
    def physical_spacing(self) -> float:
        return self.q_spacing / self.k_eff


@dataclass(frozen=True)
class YukawaPotential:
    """Yukawa pair kernel beta*V(r)=g*exp(-r/D)/r.

    In the note this kernel is integrated over two contour coordinates, so
    ``strength`` has inverse-length units and ``screening_length`` has length
    units. The resulting double-contour integral is dimensionless.
    """

    strength: float
    screening_length: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.strength) or self.strength < 0:
            raise ValueError("strength must be finite and nonnegative")
        if not np.isfinite(self.screening_length) or self.screening_length <= 0:
            raise ValueError("screening_length must be finite and positive")

    def evaluate(self, distance: ArrayLike, floor: float = 0.0) -> FloatArray:
        values = np.asarray(distance, dtype=float)
        if floor > 0:
            values = np.maximum(values, floor)
        if np.any(values <= 0):
            raise ValueError("pair distances must be positive")
        return self.strength * np.exp(-values / self.screening_length) / values

    def derivative(self, distance: ArrayLike) -> FloatArray:
        """Return the radial derivative of beta*V."""
        values = np.asarray(distance, dtype=float)
        if np.any(values <= 0):
            raise ValueError("pair distances must be positive")
        exponential = np.exp(-values / self.screening_length)
        return -self.strength * exponential * (
            1.0 / (self.screening_length * values) + 1.0 / values**2
        )


@dataclass(frozen=True)
class YukawaChangeKernel:
    """Signed effective kernel beta*Delta V=g_delta*exp(-r/D_delta)/r."""

    strength: float
    screening_length: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.strength):
            raise ValueError("change-kernel strength must be finite")
        if not np.isfinite(self.screening_length) or self.screening_length<=0:
            raise ValueError("change-kernel screening_length must be finite and positive")

    def evaluate(self, distance: ArrayLike, floor: float=0.0) -> FloatArray:
        values=np.asarray(distance,dtype=float)
        if floor>0:
            values=np.maximum(values,floor)
        if np.any(values<=0):
            raise ValueError("pair distances must be positive")
        return self.strength*np.exp(-values/self.screening_length)/values

    def derivative(self, distance: ArrayLike) -> FloatArray:
        values=np.asarray(distance,dtype=float)
        if np.any(values<=0):
            raise ValueError("pair distances must be positive")
        exponential=np.exp(-values/self.screening_length)
        return -self.strength*exponential*(
            1/(self.screening_length*values)+1/values**2)


@dataclass(frozen=True)
class YukawaMixtureChangeKernel:
    """Signed sum of Yukawa changes used when one range cannot match two moments."""

    strengths: FloatArray
    screening_lengths: FloatArray

    def __post_init__(self) -> None:
        strengths=np.asarray(self.strengths,dtype=float)
        lengths=np.asarray(self.screening_lengths,dtype=float)
        if strengths.ndim!=1 or lengths.shape!=strengths.shape or len(strengths)<1:
            raise ValueError("mixture strengths and screening lengths must be aligned vectors")
        if np.any(~np.isfinite(strengths)) or np.any(~np.isfinite(lengths)) or np.any(lengths<=0):
            raise ValueError("mixture parameters must be finite with positive ranges")
        object.__setattr__(self,"strengths",strengths)
        object.__setattr__(self,"screening_lengths",lengths)

    def evaluate(self, distance: ArrayLike, floor: float=0.0) -> FloatArray:
        original=np.asarray(distance,dtype=float); values=np.atleast_1d(original)
        if floor>0:
            values=np.maximum(values,floor)
        if np.any(values<=0):
            raise ValueError("pair distances must be positive")
        result=np.sum(
            self.strengths[:,None]*np.exp(-values[None,:]/self.screening_lengths[:,None])
            /values[None,:],axis=0)
        return np.asarray(result[0]) if original.ndim==0 else result

    def derivative(self, distance: ArrayLike) -> FloatArray:
        original=np.asarray(distance,dtype=float); values=np.atleast_1d(original)
        if np.any(values<=0):
            raise ValueError("pair distances must be positive")
        exponential=np.exp(-values[None,:]/self.screening_lengths[:,None])
        result=np.sum(-self.strengths[:,None]*exponential*(
            1/(self.screening_lengths[:,None]*values[None,:])+1/values[None,:]**2),axis=0)
        return np.asarray(result[0]) if original.ndim==0 else result


@dataclass(frozen=True)
class NonlocalConfig:
    local_cutoff: float
    max_contour_separation: float
    sample_spacing: float
    chord_floor: float
    max_points: int = 60000
    random_seed: int = 19071

    def __post_init__(self) -> None:
        if self.local_cutoff <= 0 or self.max_contour_separation <= self.local_cutoff:
            raise ValueError("nonlocal cutoffs must be positive and ordered")
        if self.sample_spacing <= 0 or self.chord_floor <= 0 or self.max_points < 1:
            raise ValueError("nonlocal numerical controls must be positive")

    @classmethod
    def from_dimensionless(
        cls, k_eff: float, q_spacing: float = 0.08, q_min: float = 1.0,
        q_max: float = 8.0, chord_q_floor: float = 0.08,
        max_points: int = 60000, random_seed: int = 19071,
    ) -> "NonlocalConfig":
        if k_eff <= 0:
            raise ValueError("k_eff must be positive")
        return cls(q_min/k_eff, q_max/k_eff, q_spacing/k_eff,
                   chord_q_floor/k_eff, max_points, random_seed)


@dataclass(frozen=True)
class NonlocalTable:
    """Local states paired with a documented nonlocal scalar response."""

    state: FloatArray
    delta_energy: FloatArray
    groups: IntArray

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=float)
        energy = np.asarray(self.delta_energy, dtype=float)
        groups = np.asarray(self.groups, dtype=int)
        if state.ndim != 2 or state.shape[1] != 3:
            raise ValueError("state must have shape (n,3)")
        if energy.shape != (len(state),) or groups.shape != (len(state),) or len(state) == 0:
            raise ValueError("nonlocal arrays must be nonempty and aligned")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(energy)):
            raise ValueError("nonlocal arrays must be finite")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "delta_energy", energy)
        object.__setattr__(self, "groups", groups)


@dataclass(frozen=True)
class DependenceResult:
    cv_r2: FloatArray
    mean_r2: float
    standard_error: float
    dependent: bool
    threshold: float
    svd_retained_fraction: float = np.nan
    svd_rank: int = 0
    basis_degree: int = 2


@dataclass(frozen=True)
class LinkedInvariantCheck:
    cutoff: float
    c4: float
    residual_boundary_c4: float
    relative_magnitude: float
    tolerance: float
    adequate: bool


@dataclass(frozen=True)
class ConditionalNonlocalModel:
    center: FloatArray
    scale: FloatArray
    coefficients: FloatArray
    lower_weight: float
    upper_weight: float
    dependence: DependenceResult
    basis_degree: int = 2

    def log_weight(self, state: ArrayLike) -> FloatArray:
        coordinates, _, _ = invariant_coordinates(state, self.center, self.scale)
        estimate = polynomial_basis(coordinates, self.basis_degree) @ self.coefficients
        return np.log(np.clip(estimate, self.lower_weight, self.upper_weight))

    def beta_delta_free_energy(self, state: ArrayLike) -> FloatArray:
        return -self.log_weight(state)


@dataclass(frozen=True)
class DensityRatioResult:
    delta_c2_kref: float
    delta_c4_kref3: float
    covariance: FloatArray
    intercept: float
    loss: float
    success: bool
    target_mass_retained: float
    reference_effective_fraction: float
    feature_bounds: FloatArray
    reference_keep: NDArray[np.bool_]
    target_keep: NDArray[np.bool_]

    @property
    def standard_errors(self) -> FloatArray:
        return np.sqrt(np.maximum(np.diag(self.covariance), 0.0))


@dataclass(frozen=True)
class RelativeNonlocalCorrectionResult:
    """Self-consistent nonlocal correction under a stated reference normalization."""

    reference_potential: YukawaPotential
    target_potential: YukawaPotential
    g_over_g0: float
    d_over_d0: float
    table: NonlocalTable
    dependence: DependenceResult
    model: ConditionalNonlocalModel
    corrected_fit: DensityRatioResult
    coefficient_history: FloatArray
    converged: bool
    dependence_mode: str
    dependence_table: NonlocalTable
    reference_linked_check: LinkedInvariantCheck
    target_linked_check: LinkedInvariantCheck


@dataclass(frozen=True)
class DirectNonlocalChangeResult:
    """Self-consistent correction from a normalization-free Delta V kernel."""

    change_kernel: YukawaChangeKernel
    g_delta_over_kref: float
    d_delta_kref: float
    table: NonlocalTable
    dependence: DependenceResult
    model: ConditionalNonlocalModel
    corrected_fit: DensityRatioResult
    coefficient_history: FloatArray
    kernel_history: FloatArray
    converged: bool
    linked_check: LinkedInvariantCheck


@dataclass(frozen=True)
class NonlocalMaximumEntropyResult:
    """Linked maximum-entropy fit with configuration-dependent nonlocal weights."""

    change_kernel: YukawaChangeKernel|YukawaMixtureChangeKernel
    kernel_mode: str
    kernel_ranges_kref: FloatArray
    kernel_strengths_over_kref: FloatArray
    kernel_condition_number: float
    table: NonlocalTable
    dependence: DependenceResult
    shape_dependence_required: bool
    model: ConditionalNonlocalModel
    corrected_fit: "MaximumEntropyResult"
    beta_delta_free_energy: FloatArray
    corrected_reference_weights: FloatArray
    coefficient_history: FloatArray
    converged: bool


@dataclass(frozen=True)
class NonlocalNormalizationSensitivity:
    """Conditional nonlocal fits over nuisance reference normalizations."""

    results: Mapping[tuple[float,float],RelativeNonlocalCorrectionResult]
    failures: Mapping[tuple[float,float],str]
    local_coefficients: FloatArray
    corrected_coefficients: FloatArray
    coefficient_minimum: FloatArray
    coefficient_maximum: FloatArray
    coefficient_span: FloatArray
    robustness_tolerance: FloatArray
    stable: bool
    dependence_consistent: bool
    all_converged: bool
    linked_model_adequate: bool


@dataclass(frozen=True)
class ConditionalYukawaFamily:
    """Finite-cutoff relative Yukawa mappings over reference choices.

    Rows index candidate zero-salt ``(g0/k_ref,D0*k_ref)`` pairs and columns
    index the supplied coefficient changes. Invalid target inversions are NaN
    and marked false in ``valid``.
    """

    reference_g0_over_kref: FloatArray
    reference_d0_kref: FloatArray
    g_over_g0: FloatArray
    d_over_d0: FloatArray
    valid: NDArray[np.bool_]


@dataclass(frozen=True)
class FlexibleDensityRatioResult:
    parameters: FloatArray
    coordinate_center: FloatArray
    coordinate_scale: FloatArray
    term_center: FloatArray
    term_scale: FloatArray
    exponents: tuple[tuple[int, ...], ...]
    loss: float
    success: bool

    def log_ratio(self, state: ArrayLike, k_ref: float) -> FloatArray:
        coordinates = even_local_coordinates(state, k_ref)
        standardized = np.clip((coordinates-self.coordinate_center)/self.coordinate_scale, -5, 5)
        terms = (polynomial_matrix(standardized, self.exponents)-self.term_center)/self.term_scale
        return np.column_stack((np.ones(len(terms)), terms)) @ self.parameters


@dataclass(frozen=True)
class LinkedCoefficientProjection:
    """Two physical coefficients summarizing a flexible log-density fit."""

    delta_c2_kref: float
    delta_c4_kref3: float
    covariance: FloatArray
    intercept: float
    unexplained_fraction: float
    effective_count: float

    @property
    def standard_errors(self) -> FloatArray:
        return np.sqrt(np.maximum(np.diag(self.covariance),0.0))


@dataclass(frozen=True)
class CoefficientBootstrapResult:
    estimates: FloatArray
    mean: FloatArray
    covariance: FloatArray

    @property
    def standard_errors(self) -> FloatArray:
        return np.sqrt(np.maximum(np.diag(self.covariance), 0.0))


@dataclass(frozen=True)
class IndependentInvariantResult:
    coefficients: FloatArray
    covariance: FloatArray
    intercept: float
    loss: float
    success: bool
    feature_bounds: FloatArray

    def log_ratio(self, sample: LocalGeometrySample, k_ref: float) -> FloatArray:
        return -(independent_local_features(sample, k_ref) @ self.coefficients)


@dataclass(frozen=True)
class MaximumEntropyResult:
    """Direct exponential-family fit to target invariant moments.

    ``coefficients`` multiply the supplied features in
    ``P/P0 = exp(-features @ coefficients)/Z``.  The covariance is a naive
    weighted-sample estimate; replicate input fits remain the appropriate
    uncertainty calculation for final reporting.
    """

    coefficients: FloatArray
    covariance: FloatArray
    objective: float
    success: bool
    message: str
    target_mass_retained: float
    reference_effective_fraction: float
    feature_bounds: FloatArray
    reference_keep: NDArray[np.bool_]
    target_keep: NDArray[np.bool_]
    target_moments: FloatArray
    fitted_moments: FloatArray

    @property
    def standard_errors(self) -> FloatArray:
        return np.sqrt(np.maximum(np.diag(self.covariance), 0.0))

    @property
    def moment_residual(self) -> FloatArray:
        return self.fitted_moments-self.target_moments

    @property
    def delta_c2_kref(self) -> float:
        if len(self.coefficients)!=2:
            raise AttributeError("delta_c2_kref is defined only for the linked fit")
        return float(self.coefficients[0])

    @property
    def delta_c4_kref3(self) -> float:
        if len(self.coefficients)!=2:
            raise AttributeError("delta_c4_kref3 is defined only for the linked fit")
        return float(self.coefficients[1])


@dataclass(frozen=True)
class ContourHeldoutResult:
    """A feasible deterministic train/test split for contour validation."""

    fit: MaximumEntropyResult
    reference_train: NDArray[np.bool_]
    target_train: NDArray[np.bool_]
    target_test_keep: NDArray[np.bool_]
    split_attempt: int


NOTE_FUNCTION_MAP = {
    "section_1_local_nonlocal_cutoff": "NonlocalConfig",
    "section_2_local_geometry": "conditional_line_geometry, anisotropic_curve_jet, transform_contour_trace, local_fourth_order_density",
    "section_3_nonlocal_dependence": "absolute_nonlocal_energy_density, relative_nonlocal_cell_hamiltonian, svd_retained_variance, assess_nonlocal_dependence",
    "section_4_conditional_nonlocal_free_energy": "fit_conditional_nonlocal_weight, fit_direct_nonlocal_change_correction",
    "section_5_zero_salt_reweighting": "fit_linked_maximum_entropy, fit_independent_maximum_entropy, fit_local_cell_reweighting, fit_contour_reweighting",
    "section_6_numerical_procedure": "build_isotropic_trace, run_trace_convergence_sweep",
    "section_7_yukawa_parameters": "yukawa_cutoff_factors, solve_yukawa_change, yukawa_change_uncertainty",
}


def load_fit_parameters(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[row["name"]] = float(row["value"])
    return values


def discover_fit_spectra(input_dir: Path, pattern: str,
                         concentration_pattern: str = r"ca_(\d+(?:\.\d+)?)mM_") -> list[FitSpectrum]:
    rows: list[FitSpectrum] = []
    for path in Path(input_dir).glob(pattern):
        values = load_fit_parameters(path)
        match = re.search(concentration_pattern, path.name)
        if match is None:
            continue
        concentration = float(values.get("salt_concentration_mM", match.group(1)))
        rows.append(FitSpectrum(concentration, float(values["k_eff"]),
            float(values["mean_k"]), float(values["r_sigma_k"]), path,
            float(values.get("stretch_ratio", np.nan)), f"{concentration:g} mM"))
    if not rows:
        raise FileNotFoundError(f"No fit files matching {pattern!r} under {input_dir}")
    return sorted(rows, key=lambda item: item.concentration_mM)


def select_zero_salt_reference(spectra: Sequence[FitSpectrum]) -> FitSpectrum:
    if not spectra:
        raise ValueError("at least one spectrum is required")
    return min(spectra, key=lambda item: abs(item.concentration_mM))


def select_reference_spectrum(spectra: Sequence[FitSpectrum],
                              choice: str="zero") -> FitSpectrum:
    """Select the zero- or highest-salt condition as the comparison reference."""
    if not spectra:
        raise ValueError("at least one spectrum is required")
    if choice=="zero":
        reference=select_zero_salt_reference(spectra)
        if not np.isclose(reference.concentration_mM,0.0,atol=1e-12):
            raise ValueError("the fitted series has no zero-salt condition")
        return reference
    if choice=="highest":
        return max(spectra,key=lambda item:item.concentration_mM)
    raise ValueError("reference choice must be 'zero' or 'highest'")


def interaction_comparison_pairs(
        spectra: Sequence[FitSpectrum], mode: str="adjacent",
        reference_choice: str="zero") -> tuple[InteractionComparisonPair, ...]:
    """Return directed pairs for adjacent, single-reference, or endpoint fits.

    Adjacent pairs always follow increasing salt. ``reference_choice`` selects
    the source only for the legacy single-reference mode; it also remains the
    state to which cumulative adjacent changes are reported by the notebook.
    """
    ordered=sorted(spectra,key=lambda item:item.concentration_mM)
    if len(ordered)<2:
        raise ValueError("at least two salt conditions are required")
    if mode not in {"adjacent","single_reference","long_baseline"}:
        raise ValueError(
            "comparison mode must be 'adjacent', 'single_reference', or "
            "'long_baseline'")
    if mode=="adjacent":
        raw=tuple(zip(ordered[:-1],ordered[1:]))
    elif mode=="long_baseline":
        raw=((ordered[0],ordered[-1]),)
    else:
        reference=select_reference_spectrum(ordered,reference_choice)
        raw=tuple((reference,target) for target in ordered if target is not reference)
    return tuple(InteractionComparisonPair(
        source.concentration_mM,target.concentration_mM,
        f"{source.concentration_mM:g}mM_to_{target.concentration_mM:g}mM",
        index,mode,mode=="long_baseline",
    ) for index,(source,target) in enumerate(raw))


def rescale_dimensionless_coefficient_change(
        coefficients: ArrayLike, source_k_eff: float,
        report_k_eff: float) -> FloatArray:
    """Express one physical ``(Delta c2, Delta c4)`` at another k scale."""
    values=np.asarray(coefficients,dtype=float)
    if values.shape!=(2,) or np.any(~np.isfinite(values)):
        raise ValueError("coefficients must contain two finite values")
    if source_k_eff<=0 or report_k_eff<=0:
        raise ValueError("wave-number scales must be positive")
    ratio=report_k_eff/source_k_eff
    return values*np.array([ratio,ratio**3])


def rescale_coefficient_covariance(
        covariance: ArrayLike, source_k_eff: float,
        report_k_eff: float) -> FloatArray:
    """Transform covariance consistently with coefficient rescaling."""
    matrix=np.asarray(covariance,dtype=float)
    if matrix.shape!=(2,2) or np.any(~np.isfinite(matrix)):
        raise ValueError("covariance must be a finite 2 by 2 matrix")
    ratio=report_k_eff/source_k_eff
    scale=np.diag([ratio,ratio**3])
    return scale@matrix@scale


def cumulative_adjacent_coefficient_changes(
        pairs: Sequence[InteractionComparisonPair], changes: ArrayLike,
        reference_concentration_mM: float) -> dict[float, FloatArray]:
    """Accumulate directed adjacent changes relative to any state in the chain."""
    values=np.asarray(changes,dtype=float)
    if values.shape!=(len(pairs),2) or np.any(~np.isfinite(values)):
        raise ValueError("changes must have shape (number of pairs, 2)")
    if any(pair.comparison_mode!="adjacent" for pair in pairs):
        raise ValueError("cumulative changes require adjacent comparison pairs")
    nodes=sorted({pair.source_concentration_mM for pair in pairs}
                 |{pair.target_concentration_mM for pair in pairs})
    matches=[index for index,value in enumerate(nodes)
             if np.isclose(value,reference_concentration_mM)]
    if len(matches)!=1:
        raise ValueError("reference concentration is not a unique chain node")
    edge_by_source={pair.source_concentration_mM: values[index]
                    for index,pair in enumerate(pairs)}
    cumulative={nodes[0]:np.zeros(2,dtype=float)}
    for source,target in zip(nodes[:-1],nodes[1:]):
        if source not in edge_by_source:
            raise ValueError("pairs do not form one complete increasing chain")
        cumulative[target]=cumulative[source]+edge_by_source[source]
    offset=cumulative[nodes[matches[0]]]
    return {salt:value-offset for salt,value in cumulative.items()}


def gamma_radial_moment(order: int, r_sigma_k: float, k_eff: float = 1.0) -> float:
    if order < 0 or r_sigma_k <= 0 or k_eff <= 0:
        raise ValueError("moment order and spectrum scales are invalid")
    mean_k = k_eff / np.sqrt(1+r_sigma_k**2)
    shape = 1/r_sigma_k**2
    scale = mean_k*r_sigma_k**2
    return float(np.exp(order*np.log(scale)+gammaln(shape+order)-gammaln(shape)))


def gamma_radial_density(k: ArrayLike, k_eff: float,
                         r_sigma_k: float) -> FloatArray:
    """Positive radial spectral density parameterized by its RMS scale."""
    values=np.asarray(k,dtype=float)
    if k_eff<=0 or r_sigma_k<=0:
        raise ValueError("k_eff and relative spectral width must be positive")
    mean_k=k_eff/np.sqrt(1+r_sigma_k**2)
    shape=1/r_sigma_k**2
    scale=mean_k*r_sigma_k**2
    return np.asarray(gamma_distribution.pdf(values,a=shape,scale=scale),dtype=float)


def multi_indices_through(max_order: int) -> list[tuple[int, int, int]]:
    return [(nx, ny, total-nx-ny) for total in range(max_order+1)
            for nx in range(total+1) for ny in range(total-nx+1)]


def odd_double_factorial(value: int) -> float:
    result = 1.0
    for item in range(value, 0, -2):
        result *= item
    return result


def spherical_monomial_average(exponents: tuple[int, int, int]) -> float:
    if any(power % 2 for power in exponents):
        return 0.0
    return float(np.prod([odd_double_factorial(power-1) for power in exponents]) /
                 odd_double_factorial(sum(exponents)+1))


DERIVATIVE_INDICES = tuple(multi_indices_through(3))
NONZERO_INDICES = DERIVATIVE_INDICES[1:]
DERIVATIVE_LOOKUP = {multi: index for index, multi in enumerate(NONZERO_INDICES)}


def conditional_derivative_covariance(r_sigma_k: float) -> FloatArray:
    moments = {order: gamma_radial_moment(order, r_sigma_k) for order in (0,2,4,6)}
    def entry(alpha, beta):
        exponents = tuple(a+b for a,b in zip(alpha,beta))
        total = sum(exponents)
        return 0.0 if total % 2 else ((-1.0)**(sum(beta)+total//2) *
            moments[total]*spherical_monomial_average(exponents))
    covariance = np.array([[entry(a,b) for b in DERIVATIVE_INDICES]
                           for a in DERIVATIVE_INDICES])
    conditioned = covariance[1:,1:] - np.outer(covariance[1:,0], covariance[1:,0])
    return np.asarray(0.5*(conditioned+conditioned.T))


def expanded_tensor(samples: FloatArray, order: int) -> FloatArray:
    tensor = np.empty((len(samples),)+(3,)*order)
    for axes in product(range(3), repeat=order):
        counts = tuple(axes.count(axis) for axis in range(3))
        tensor[(slice(None),)+axes] = samples[:,DERIVATIVE_LOOKUP[counts]]
    return tensor


def contract2(tensor: FloatArray, first: FloatArray, second: FloatArray) -> FloatArray:
    return np.einsum("nij,ni,nj->n", tensor, first, second, optimize=True)


def contract3(tensor: FloatArray, first: FloatArray, second: FloatArray,
              third: FloatArray) -> FloatArray:
    return np.einsum("nijk,ni,nj,nk->n", tensor, first, second, third, optimize=True)


def sobol_standard_normals(sample_power: int, seed: int = 73129) -> FloatArray:
    if sample_power < 4:
        raise ValueError("sample_power must be at least 4")
    sampler = qmc.Sobol(d=2*len(NONZERO_INDICES), scramble=True, seed=seed)
    unit = np.clip(sampler.random_base2(sample_power), np.finfo(float).eps,
                   1-np.finfo(float).eps)
    return np.asarray(norm.ppf(unit))


def validate_hencky_tensor(H: ArrayLike) -> FloatArray:
    """Return a finite symmetric traceless Hencky tensor."""
    tensor=np.asarray(H,dtype=float)
    if tensor.shape!=(3,3) or not np.all(np.isfinite(tensor)):
        raise ValueError("H must be a finite 3 by 3 tensor")
    scale=max(float(np.linalg.norm(tensor)),1.0)
    if not np.allclose(tensor,tensor.T,rtol=0.0,atol=1e-12*scale):
        raise ValueError("H must be symmetric")
    if not np.isclose(np.trace(tensor),0.0,rtol=0.0,atol=1e-12*scale):
        raise ValueError("H must be traceless")
    return np.asarray(0.5*(tensor+tensor.T))


def uniaxial_hencky_tensor(stretch_ratio: float, axis: int=2) -> FloatArray:
    """Construct the volume-preserving tensor used by the scattering fit."""
    stretch=float(stretch_ratio)
    if not np.isfinite(stretch) or stretch<=0 or axis not in (0,1,2):
        raise ValueError("stretch_ratio must be positive and axis must be 0, 1, or 2")
    strain=np.log(stretch)
    diagonal=np.full(3,-0.5*strain); diagonal[axis]=strain
    return np.diag(diagonal)


def fitted_hencky_tensors(spectra: Sequence[FitSpectrum], axis: int=2
        ) -> dict[float,FloatArray]:
    """Return fixed fitted H tensors without changing ``FitSpectrum``."""
    output={}
    for row in spectra:
        if not np.isfinite(row.stretch_ignored) or row.stretch_ignored<=0:
            raise ValueError(f"no fitted stretch ratio for {row.concentration_mM:g} mM")
        output[row.concentration_mM]=uniaxial_hencky_tensor(row.stretch_ignored,axis)
    return output


def hencky_stretch_tensor(H: ArrayLike) -> FloatArray:
    """Return the right stretch F=expm(H)."""
    return np.asarray(expm(validate_hencky_tensor(H)),dtype=float)


def arclength_curve_jet(first: ArrayLike, second: ArrayLike,
                        third: ArrayLike) -> tuple[FloatArray,...]:
    """Convert derivatives in an arbitrary parameter to physical arclength."""
    v,a,b=(np.asarray(value,dtype=float) for value in (first,second,third))
    if v.ndim!=2 or v.shape[1]!=3 or a.shape!=v.shape or b.shape!=v.shape:
        raise ValueError("curve derivatives must be aligned arrays with shape (n,3)")
    speed=np.linalg.norm(v,axis=1)
    if np.any(~np.isfinite(speed)) or np.any(speed<=1e-14):
        raise ValueError("curve derivative has zero or nonfinite speed")
    va=np.einsum("ni,ni->n",v,a)
    aa_vb=np.einsum("ni,ni->n",a,a)+np.einsum("ni,ni->n",v,b)
    tangent=v/speed[:,None]
    r2=a/speed[:,None]**2-v*va[:,None]/speed[:,None]**4
    r3=(b/speed[:,None]**3-3*a*va[:,None]/speed[:,None]**5
        -v*aa_vb[:,None]/speed[:,None]**5
        +4*v*va[:,None]**2/speed[:,None]**7)
    return tangent,r2,r3,speed


def anisotropic_curve_jet(tangent: ArrayLike, r2: ArrayLike, r3: ArrayLike,
                          H: ArrayLike) -> tuple[FloatArray,...]:
    """Affine-map an arclength curve jet and reparameterize it exactly."""
    F=hencky_stretch_tensor(H)
    first=np.asarray(tangent,dtype=float)@F.T
    second=np.asarray(r2,dtype=float)@F.T
    third=np.asarray(r3,dtype=float)@F.T
    return arclength_curve_jet(first,second,third)


def conditional_line_geometry(k_eff: float, r_sigma_k: float,
                              standard_normals: ArrayLike,
                              aniso: bool=False,
                              H: ArrayLike|None=None) -> LocalGeometrySample:
    normals = np.asarray(standard_normals, dtype=float)
    expected = 2*len(NONZERO_INDICES)
    if normals.ndim != 2 or normals.shape[1] != expected:
        raise ValueError(f"standard_normals must have shape (n,{expected})")
    covariance = conditional_derivative_covariance(r_sigma_k)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = 1e-11*max(float(np.max(eigenvalues)),1.0)
    if float(np.min(eigenvalues)) < -tolerance:
        raise RuntimeError("conditional covariance is not positive semidefinite")
    half = (eigenvectors*np.sqrt(np.clip(eigenvalues,0,None))) @ eigenvectors.T
    count = len(NONZERO_INDICES)
    field1, field2 = normals[:,:count]@half.T, normals[:,count:]@half.T
    g1,h1,d31 = (expanded_tensor(field1,order) for order in (1,2,3))
    g2,h2,d32 = (expanded_tensor(field2,order) for order in (1,2,3))
    omega = np.cross(g1,g2)
    jacobian = np.linalg.norm(omega,axis=1)
    stable = jacobian > 1e-12
    if not np.any(stable):
        raise RuntimeError("no stable conditional samples")
    g1,h1,d31,g2,h2,d32,omega,jac = (value[stable] for value in
        (g1,h1,d31,g2,h2,d32,omega,jacobian))
    tangent = omega/jac[:,None]
    matrix = np.stack((g1,g2,tangent),axis=1)
    rhs2 = np.column_stack((-contract2(h1,tangent,tangent),
                            -contract2(h2,tangent,tangent), np.zeros(len(tangent))))
    r2 = np.linalg.solve(matrix,rhs2[...,None])[...,0]
    rhs3 = np.column_stack((
        -contract3(d31,tangent,tangent,tangent)-3*contract2(h1,tangent,r2),
        -contract3(d32,tangent,tangent,tangent)-3*contract2(h2,tangent,r2),
        -np.einsum("ni,ni->n",r2,r2)))
    r3 = np.linalg.solve(matrix,rhs3[...,None])[...,0]
    if aniso:
        if H is None:
            raise ValueError("H is required when aniso=True")
        tangent,r2,r3,line_factor=anisotropic_curve_jet(tangent,r2,r3,H)
        jac=jac*line_factor
    elif H is not None:
        raise ValueError("H must be omitted when aniso=False")
    kappa0 = np.linalg.norm(r2,axis=1)
    normal_frame = np.zeros_like(r2)
    regular = kappa0 > 1e-10
    normal_frame[regular] = r2[regular]/kappa0[regular,None]
    binormal = np.cross(tangent,normal_frame)
    return LocalGeometrySample(k_eff*kappa0,
        k_eff**2*np.einsum("ni,ni->n",normal_frame,r3),
        k_eff**2*np.einsum("ni,ni->n",binormal,r3), jac, k_eff,
        float(np.mean(stable)), float(np.mean(jac)))


def sample_local_geometry_series(spectra: Sequence[FitSpectrum], sample_power: int = 15,
                                 seed: int = 73129, aniso: bool=False,
                                 H_by_concentration: Mapping[float,ArrayLike]|None=None
                                 ) -> dict[float, LocalGeometrySample]:
    normals = sobol_standard_normals(sample_power,seed)
    if not aniso:
        if H_by_concentration is not None:
            raise ValueError("H_by_concentration must be omitted when aniso=False")
        return {row.concentration_mM: conditional_line_geometry(
            row.k_eff,row.r_sigma_k,normals) for row in spectra}
    if H_by_concentration is None:
        raise ValueError("H_by_concentration is required when aniso=True")
    missing=[row.concentration_mM for row in spectra
             if row.concentration_mM not in H_by_concentration]
    if missing:
        raise ValueError(f"missing H tensors for concentrations {missing}")
    return {row.concentration_mM: conditional_line_geometry(
        row.k_eff,row.r_sigma_k,normals,True,H_by_concentration[row.concentration_mM])
        for row in spectra}


def local_fourth_order_density(kappa: ArrayLike, kappa_prime: ArrayLike,
                               kappa_tau: ArrayLike) -> FloatArray:
    k,p,t = (np.asarray(value,dtype=float) for value in (kappa,kappa_prime,kappa_tau))
    return (9/64)*k**4-p**2/24-t**2/24


def local_cell_features(sample: LocalGeometrySample, k_ref: float,
                        cell_length: float | None = None) -> FloatArray:
    """Features multiplied by (Delta c2*k_ref, Delta c4*k_ref**3)."""
    if k_ref <= 0:
        raise ValueError("k_ref must be positive")
    length = 1/k_ref if cell_length is None else float(cell_length)
    if length <= 0:
        raise ValueError("cell_length must be positive")
    factor = length*k_ref
    return np.column_stack((factor*sample.kappa**2/k_ref**2,
                            factor*sample.fourth_order_density/k_ref**4))


def normalized_local_state(sample: LocalGeometrySample, k_ref: float) -> FloatArray:
    """Return (kappa/k_ref,kappa_prime/k_ref**2,kappa*tau/k_ref**2)."""
    if k_ref<=0: raise ValueError("k_ref must be positive")
    return sample.state/np.array([k_ref,k_ref**2,k_ref**2])


def torsion_from_local_geometry(sample: LocalGeometrySample,
        curvature_floor: float|None=None) -> tuple[FloatArray,NDArray[np.bool_]]:
    """Recover signed torsion from the sampled ``kappa*tau`` invariant.

    Torsion is undefined at zero curvature. The returned mask removes that
    numerically singular set while preserving alignment with the sample arrays.
    """
    floor=(np.sqrt(np.finfo(float).eps)*sample.k_eff
           if curvature_floor is None else float(curvature_floor))
    if not np.isfinite(floor) or floor<0:
        raise ValueError("curvature_floor must be finite and nonnegative")
    keep=sample.kappa>floor
    if not np.any(keep):
        raise ValueError("no samples remain above the torsion curvature floor")
    return sample.kappa_tau[keep]/sample.kappa[keep],keep


def contour_observables(kappa: ArrayLike, kappa_prime: ArrayLike,
                        kappa_tau: ArrayLike, arclength: ArrayLike) -> tuple[float,float]:
    k,p,t,s = (np.asarray(value,dtype=float) for value in
               (kappa,kappa_prime,kappa_tau,arclength))
    if any(value.shape != s.shape for value in (k,p,t)):
        raise ValueError("geometry and arclength arrays must align")
    if s.ndim != 1 or len(s)<2 or np.any(np.diff(s)<=0):
        raise ValueError("arclength must be strictly increasing")
    return float(np.trapezoid(k**2,s)), float(np.trapezoid(local_fourth_order_density(k,p,t),s))


def contour_features(observables: ArrayLike, k_ref: float) -> FloatArray:
    values = np.asarray(observables,dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or k_ref <= 0:
        raise ValueError("observables must have shape (n,2) and k_ref>0")
    return np.column_stack((values[:,0]/k_ref, values[:,1]/k_ref**3))


def packed_cells(cells: Sequence[ArrayLike]) -> tuple[FloatArray, IntArray]:
    arrays = [np.asarray(cell,dtype=float) for cell in cells]
    lengths = np.asarray([len(cell) for cell in arrays],dtype=int)
    offsets = np.concatenate(([0],np.cumsum(lengths))).astype(int)
    return (np.vstack(arrays) if arrays else np.empty((0,3))), offsets


def unpacked_cells(points: ArrayLike, offsets: ArrayLike) -> tuple[FloatArray,...]:
    values = np.asarray(points,dtype=float)
    bounds = np.asarray(offsets,dtype=int)
    return tuple(values[bounds[i]:bounds[i+1]] for i in range(len(bounds)-1))


def save_trace_cache(path: Path, trace: ContourTrace, config: TraceConfig,
                     spectrum: FitSpectrum|None=None) -> None:
    points, offsets = packed_cells(trace.points)
    tangents,_ = packed_cells(trace.tangents)
    r2,_ = packed_cells(trace.r2)
    r3,_ = packed_cells(trace.r3)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    anisotropic=bool(trace.metadata.get("anisotropic_geometry",False))
    hencky=np.asarray(trace.metadata.get("hencky_tensor",np.full((3,3),np.nan)),dtype=float)
    np.savez_compressed(path,points=points,tangents=tangents,r2=r2,r3=r3,offsets=offsets,
        trace_k_eff=trace.k_eff,q_spacing=trace.q_spacing,grid_size=config.grid_size,
        num_blocks=config.num_blocks,num_modes=config.num_modes,trace_k0=config.trace_k0,
        random_seed=config.random_seed,seed_spacing=config.seed_spacing,
        anisotropic_geometry=anisotropic,hencky_tensor=hencky,
        spectrum_concentration_mM=np.nan if spectrum is None else spectrum.concentration_mM,
        spectrum_k_eff=np.nan if spectrum is None else spectrum.k_eff,
        spectrum_r_sigma_k=np.nan if spectrum is None else spectrum.r_sigma_k)


def load_trace_cache(path: Path, config: TraceConfig,
                     spectrum: FitSpectrum|None=None) -> ContourTrace | None:
    if not Path(path).exists():
        return None
    with np.load(path,allow_pickle=False) as saved:
        matches = (int(saved["grid_size"])==config.grid_size and
            int(saved["num_blocks"])==config.num_blocks and
            int(saved["num_modes"])==config.num_modes and
            np.isclose(float(saved["trace_k0"]),config.trace_k0) and
            int(saved["random_seed"])==config.random_seed and
            np.isclose(float(saved["seed_spacing"]),config.seed_spacing) and
            np.isclose(float(saved["q_spacing"]),config.q_spacing))
        if spectrum is not None:
            spectrum_fields={"spectrum_concentration_mM","spectrum_k_eff","spectrum_r_sigma_k"}
            matches = matches and spectrum_fields.issubset(saved.files)
            if matches:
                matches = (
                    np.isclose(float(saved["spectrum_concentration_mM"]),spectrum.concentration_mM)
                    and np.isclose(float(saved["spectrum_k_eff"]),spectrum.k_eff)
                    and np.isclose(float(saved["spectrum_r_sigma_k"]),spectrum.r_sigma_k)
                )
        if not matches:
            return None
        offsets = np.asarray(saved["offsets"],dtype=int)
        metadata={"grid_size":config.grid_size,"num_modes":config.num_modes,
                  "random_seed":config.random_seed}
        if "anisotropic_geometry" in saved.files and bool(saved["anisotropic_geometry"]):
            metadata["anisotropic_geometry"]=True
            metadata["hencky_tensor"]=np.asarray(saved["hencky_tensor"],dtype=float).tolist()
            metadata["stretch_tensor"]=hencky_stretch_tensor(saved["hencky_tensor"]).tolist()
        return ContourTrace(unpacked_cells(saved["points"],offsets),
            unpacked_cells(saved["tangents"],offsets),unpacked_cells(saved["r2"],offsets),
            unpacked_cells(saved["r3"],offsets),float(saved["trace_k_eff"]),
            float(saved["q_spacing"]),metadata)


def resample_contour_by_arclength(points: ArrayLike, spacing: float) -> FloatArray:
    coordinates = np.asarray(points,dtype=float)
    if coordinates.ndim!=2 or coordinates.shape[1]!=3 or len(coordinates)<4 or spacing<=0:
        raise ValueError("points must have shape (n>=4,3) and spacing>0")
    lengths = np.linalg.norm(np.diff(coordinates,axis=0),axis=1)
    coordinates = coordinates[np.concatenate(([True],lengths>1e-12))]
    if len(coordinates)<4:
        raise ValueError("too few distinct points")
    s = np.concatenate(([0.0],np.cumsum(np.linalg.norm(np.diff(coordinates,axis=0),axis=1))))
    sample_s = np.arange(0,s[-1],spacing)
    if len(sample_s)<4:
        raise ValueError("contour is too short")
    return np.column_stack([CubicSpline(s,coordinates[:,axis],bc_type="natural")(sample_s)
                            for axis in range(3)])


def transform_contour_trace(trace: ContourTrace, H: ArrayLike,
                            spacing: float|None=None) -> ContourTrace:
    """Affine-map a trace and rebuild its jet on physical-arclength samples."""
    tensor=validate_hencky_tensor(H); F=hencky_stretch_tensor(tensor)
    physical_spacing=trace.physical_spacing if spacing is None else float(spacing)
    if not np.isfinite(physical_spacing) or physical_spacing<=0:
        raise ValueError("spacing must be finite and positive")
    output_points=[]; output_tangent=[]; output_r2=[]; output_r3=[]
    for points in trace.points:
        transformed=np.asarray(points,dtype=float)@F.T
        segments=np.linalg.norm(np.diff(transformed,axis=0),axis=1)
        transformed=transformed[np.concatenate(([True],segments>1e-12))]
        if len(transformed)<4:
            continue
        source_s=np.concatenate(([0.0],np.cumsum(
            np.linalg.norm(np.diff(transformed,axis=0),axis=1))))
        sample_s=np.arange(0.0,source_s[-1],physical_spacing)
        if len(sample_s)<4:
            continue
        splines=[CubicSpline(source_s,transformed[:,axis],bc_type="natural")
                 for axis in range(3)]
        sampled=np.column_stack([spline(sample_s) for spline in splines])
        first=np.column_stack([spline(sample_s,1) for spline in splines])
        second=np.column_stack([spline(sample_s,2) for spline in splines])
        third=np.column_stack([spline(sample_s,3) for spline in splines])
        tangent,r2,r3,_=arclength_curve_jet(first,second,third)
        output_points.append(sampled); output_tangent.append(tangent)
        output_r2.append(r2); output_r3.append(r3)
    if not output_points:
        raise RuntimeError("no transformed contour is long enough to retain")
    metadata=dict(trace.metadata)
    metadata.update({"anisotropic_geometry":True,
                     "hencky_tensor":tensor.tolist(),
                     "stretch_tensor":F.tolist()})
    return ContourTrace(tuple(output_points),tuple(output_tangent),tuple(output_r2),
        tuple(output_r3),trace.k_eff,physical_spacing*trace.k_eff,metadata)


def trace_local_state(trace: ContourTrace, normalize: bool = False) -> tuple[FloatArray,...]:
    states=[]
    for tangent,r2,r3 in zip(trace.tangents,trace.r2,trace.r3):
        kappa=np.linalg.norm(r2,axis=1)
        normal_frame=np.zeros_like(r2)
        regular=kappa>1e-12*trace.k_eff
        normal_frame[regular]=r2[regular]/kappa[regular,None]
        binormal=np.cross(tangent,normal_frame)
        state=np.column_stack((kappa,np.einsum("ni,ni->n",normal_frame,r3),
                               np.einsum("ni,ni->n",binormal,r3)))
        if normalize:
            state=state/np.array([trace.k_eff,trace.k_eff**2,trace.k_eff**2])
        states.append(state)
    return tuple(states)


def trace_local_state_blocks(trace: ContourTrace, block_length: float
        ) -> tuple[FloatArray,...]:
    """Split traced local states into nonoverlapping equal-arclength blocks."""
    if not np.isfinite(block_length) or block_length<=0:
        raise ValueError("block_length must be finite and positive")
    intervals=int(round(block_length/trace.physical_spacing))
    if intervals<2:
        raise ValueError("block_length must span at least two trace intervals")
    points_per_block=intervals+1
    blocks=[]
    for state in trace_local_state(trace):
        for start in range(0,len(state)-points_per_block+1,intervals):
            blocks.append(state[start:start+points_per_block])
    if not blocks:
        raise RuntimeError("no trace is long enough for one expansion block")
    return tuple(blocks)


def _import_project_module(project_root: Path, name: str):
    root=str(Path(project_root).resolve())
    if root not in sys.path:
        sys.path.insert(0,root)
    return importlib.import_module(name)


def _load_module_from_path(name: str, path: Path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def build_isotropic_trace(project_root: Path, spectrum: FitSpectrum,
                          config: TraceConfig) -> ContourTrace:
    """Run the expensive production trace; never called at import or in fast tests."""
    rwn=_import_project_module(project_root,"rw_line_network")
    scattering=_load_module_from_path("rw_line_scattering_ca_interaction",
                                      Path(project_root)/"rw_line_scattering.py")
    rwn.GRID_SIZE=config.grid_size; rwn.NUM_BLOCK=config.num_blocks; rwn.BLOCK_OVERLAP=0
    rwn.RANDOM_SEED=config.random_seed; rwn.NUM_MODES=(config.num_modes,)*3
    rwn.K_DISTRIBUTION="gamma_radial"; rwn.K0=(config.trace_k0,)*3
    rwn.r_SIGMA_K=(spectrum.r_sigma_k,)*3; rwn.SHARED_K_VECTORS=False
    rwn.COUPLE_PHI2_PHI3=False; rwn.USE_VORTEX_TRACING=True
    rwn.VORTEX_FACE_PREFILTER=True; rwn.SMOOTH_VORTEX_LINES=False
    rng=np.random.default_rng(config.random_seed)
    ksets=rwn.make_field_k_sets(rwn.NUM_MODES,rwn.K_DISTRIBUTION,rng,shared_k_vectors=False)
    mean_k2=0.5*(np.mean(np.einsum("ni,ni->n",ksets.phi1,ksets.phi1))+
                 np.mean(np.einsum("ni,ni->n",ksets.phi2,ksets.phi2)))
    trace_k_eff=(2*np.pi/config.grid_size)*np.sqrt(mean_k2)
    coefficients1=rwn.make_wave_coefficients(ksets.phi1,rng)
    coefficients2=rwn.make_wave_coefficients(ksets.phi2,rng)

    def wave_jet(points,coefficients,order=0):
        q=(2*np.pi/config.grid_size)*coefficients.k_vectors
        phase=points@q.T+coefficients.phases
        cw=np.cos(phase)*coefficients.amplitudes; sw=np.sin(phase)*coefficients.amplitudes
        value=np.sum(cw,axis=1)
        if order==0: return (value,)
        gradient=(-sw)@q
        if order==1: return value,gradient
        hessian=np.einsum("nm,mi,mj->nij",-cw,q,q,optimize=True)
        if order==2: return value,gradient,hessian
        third=np.einsum("nm,mi,mj,mk->nijk",sw,q,q,q,optimize=True)
        return value,gradient,hessian,third

    def plane_coefficients(g1,g2,rhs1,rhs2):
        a11=np.einsum("ij,ij->i",g1,g1); a12=np.einsum("ij,ij->i",g1,g2)
        a22=np.einsum("ij,ij->i",g2,g2); determinant=a11*a22-a12*a12
        safe=np.abs(determinant)>1e-14*np.maximum(a11*a22,1.0)
        c1=np.zeros_like(rhs1); c2=np.zeros_like(rhs2)
        c1[safe]=(rhs1[safe]*a22[safe]-rhs2[safe]*a12[safe])/determinant[safe]
        c2[safe]=(rhs2[safe]*a11[safe]-rhs1[safe]*a12[safe])/determinant[safe]
        return c1,c2,safe

    def project(points,iterations=3):
        output=points.copy()
        for _ in range(iterations):
            f1,g1=wave_jet(output,coefficients1,1); f2,g2=wave_jet(output,coefficients2,1)
            c1,c2,safe=plane_coefficients(g1,g2,-f1,-f2)
            output[safe]+=c1[safe,None]*g1[safe]+c2[safe,None]*g2[safe]
        return output

    def geometry(points,batch_size=4096):
        tangents=np.empty_like(points); r2out=np.empty_like(points); r3out=np.empty_like(points)
        for start in range(0,len(points),batch_size):
            stop=min(start+batch_size,len(points)); local=points[start:stop]
            _,g1,h1,d31=wave_jet(local,coefficients1,3); _,g2,h2,d32=wave_jet(local,coefficients2,3)
            tangent=np.cross(g1,g2); tangent/=np.linalg.norm(tangent,axis=1)[:,None]
            matrix=np.stack((g1,g2,tangent),axis=1)
            rhs2=np.column_stack((-contract2(h1,tangent,tangent),-contract2(h2,tangent,tangent),np.zeros(len(local))))
            local_r2=np.linalg.solve(matrix,rhs2[...,None])[...,0]
            rhs3=np.column_stack((-contract3(d31,tangent,tangent,tangent)-3*contract2(h1,tangent,local_r2),
                -contract3(d32,tangent,tangent,tangent)-3*contract2(h2,tangent,local_r2),
                -np.einsum("ni,ni->n",local_r2,local_r2)))
            local_r3=np.linalg.solve(matrix,rhs3[...,None])[...,0]
            tangents[start:stop]=tangent; r2out[start:stop]=local_r2; r3out[start:stop]=local_r3
        return tangents,r2out,r3out

    phi1,phi2,_=scattering._build_fields()
    connected=rwn.smooth_vortex_polydata(rwn.trace_vortex_segments(phi1,phi2))
    seed_cells=[]
    for cell in scattering._line_cells(connected):
        sampled=scattering._sample_line_cell_by_arclength(cell,config.seed_spacing)
        if len(sampled)>3: seed_cells.append(sampled[:-1])
    seed_points,seed_offsets=packed_cells(seed_cells)
    projected=unpacked_cells(project(seed_points),seed_offsets)
    fine=[]
    for points in projected:
        try: sampled=resample_contour_by_arclength(points,config.q_spacing/trace_k_eff)
        except ValueError: continue
        if len(sampled)>=8: fine.append(sampled)
    fine_points,fine_offsets=packed_cells(fine); fine_points=project(fine_points,2)
    tangent,r2,r3=geometry(fine_points)
    # Convert the grid trace to the physical units of the fitted spectrum.
    # q=k_eff*s is invariant under this rescaling.
    length_scale=trace_k_eff/spectrum.k_eff
    physical_points=fine_points*length_scale
    physical_r2=r2/length_scale
    physical_r3=r3/length_scale**2
    return ContourTrace(unpacked_cells(physical_points,fine_offsets),unpacked_cells(tangent,fine_offsets),
        unpacked_cells(physical_r2,fine_offsets),unpacked_cells(physical_r3,fine_offsets),spectrum.k_eff,config.q_spacing,
        {"grid_size":config.grid_size,"num_modes":config.num_modes,"random_seed":config.random_seed,
         "concentration_mM":spectrum.concentration_mM,"r_sigma_k":spectrum.r_sigma_k,
         "raw_trace_k_eff":float(trace_k_eff),"physical_length_scale":float(length_scale)})


def pair_energy_change(chord_distance: ArrayLike, contour_separation: ArrayLike,
                       target: YukawaPotential|YukawaChangeKernel,
                       reference: YukawaPotential|YukawaChangeKernel,
                       chord_floor: float) -> FloatArray:
    """beta*[Delta V(d)-Delta V(Delta s)] from the note."""
    chord=np.maximum(np.asarray(chord_distance,dtype=float),chord_floor)
    separation=np.asarray(contour_separation,dtype=float)
    return ((target.evaluate(chord)-target.evaluate(separation))-
            (reference.evaluate(chord)-reference.evaluate(separation)))


def nonlocal_energy_table(trace: ContourTrace,
                          target: YukawaPotential|YukawaChangeKernel,
                          reference: YukawaPotential|YukawaChangeKernel,
                          config: NonlocalConfig) -> NonlocalTable:
    """Evaluate the relative beta*Delta H_nl per local correlation cell.

    New code should call relative_nonlocal_cell_hamiltonian for clarity.
    This name remains as a compatibility alias for existing notebook work.
    """
    minimum=int(np.ceil(config.local_cutoff/config.sample_spacing))
    maximum=int(np.floor(config.max_contour_separation/config.sample_spacing))
    rng=np.random.default_rng(config.random_seed); state_cells=trace_local_state(trace,normalize=True)
    candidates=sum(max(0,len(cell)-2*minimum) for cell in trace.points)
    probability=min(1.0,config.max_points/max(candidates,1))
    states=[]; energies=[]; groups=[]
    for group,(points,state) in enumerate(zip(trace.points,state_cells)):
        if len(points)<=2*minimum: continue
        centers=np.arange(minimum,len(points)-minimum)
        centers=centers[rng.random(len(centers))<probability]
        local_energy=np.zeros(len(centers))
        for row,center in enumerate(centers):
            local_max=min(maximum,center,len(points)-1-center)
            if local_max<minimum: continue
            lags=np.arange(minimum,local_max+1)
            neighbors=np.concatenate((center-lags,center+lags))
            chord=np.linalg.norm(points[neighbors]-points[center],axis=1)
            separation=np.tile(lags*config.sample_spacing,2)
            contribution=pair_energy_change(chord,separation,target,reference,config.chord_floor)
            energy_density=0.5*config.sample_spacing*np.sum(contribution)
            local_energy[row]=config.local_cutoff*energy_density
        states.append(state[centers]); energies.append(local_energy)
        groups.append(np.full(len(centers),group,dtype=int))
    if not states: raise RuntimeError("no points satisfy the nonlocal lag window")
    return NonlocalTable(np.vstack(states),np.concatenate(energies),np.concatenate(groups))


def relative_nonlocal_cell_hamiltonian(trace: ContourTrace,
        target: YukawaPotential, reference: YukawaPotential,
        config: NonlocalConfig) -> NonlocalTable:
    """Explicit API for the relative beta*Delta H_nl cell response."""
    return nonlocal_energy_table(trace,target,reference,config)


def absolute_nonlocal_energy_density(trace: ContourTrace,
        potential: YukawaPotential, config: NonlocalConfig) -> NonlocalTable:
    """Evaluate the absolute epsilon_nl(s) used by the manuscript diagnostic."""
    zero=YukawaPotential(0.0,potential.screening_length)
    cell_table=nonlocal_energy_table(trace,potential,zero,config)
    return NonlocalTable(
        cell_table.state,
        cell_table.delta_energy/config.local_cutoff,
        cell_table.groups,
    )


def nonlocal_change_cell_hamiltonian(trace: ContourTrace,
        change: YukawaChangeKernel|YukawaMixtureChangeKernel,
        config: NonlocalConfig) -> NonlocalTable:
    """Evaluate beta*Delta H_nl directly from the effective change kernel."""
    if isinstance(change,YukawaMixtureChangeKernel):
        zero=YukawaMixtureChangeKernel(np.zeros_like(change.strengths),
                                       change.screening_lengths)
    else:
        zero=YukawaChangeKernel(0.0,change.screening_length)
    return nonlocal_energy_table(trace,change,zero,config)


def invariant_coordinates(state: ArrayLike, center: ArrayLike|None=None,
                          scale: ArrayLike|None=None) -> tuple[FloatArray,FloatArray,FloatArray]:
    values=np.asarray(state,dtype=float)
    if values.ndim!=2 or values.shape[1]!=3: raise ValueError("state must have shape (n,3)")
    transformed=np.log1p(values**2)
    local_center=np.median(transformed,axis=0) if center is None else np.asarray(center,dtype=float)
    if scale is None:
        q25,q75=np.quantile(transformed,[0.25,0.75],axis=0); local_scale=np.maximum(q75-q25,1e-6)
    else: local_scale=np.asarray(scale,dtype=float)
    return np.clip((transformed-local_center)/local_scale,-5,5),local_center,local_scale


def polynomial_basis(coordinates: ArrayLike, maximum_degree: int=2) -> FloatArray:
    """Polynomial basis in the three normalized even local invariants."""
    values=np.asarray(coordinates,dtype=float)
    if values.ndim!=2 or values.shape[1]!=3:
        raise ValueError("coordinates must have shape (n,3)")
    if maximum_degree<1:
        raise ValueError("maximum_degree must be positive")
    exponents=polynomial_exponents(3,maximum_degree)
    return np.column_stack((np.ones(len(values)),polynomial_matrix(values,exponents)))


def svd_retained_variance(basis: ArrayLike, response: ArrayLike,
                          relative_tolerance: float|None=None) -> tuple[float,int]:
    """Return the note's SVD projection fraction and numerical basis rank.

    The intercept and response mean are removed before projection. This is the
    ideal in-sample fraction; ``assess_nonlocal_dependence`` also reports the
    grouped-contour cross-validated fraction required by the note.
    """
    matrix=np.asarray(basis,dtype=float); values=np.asarray(response,dtype=float)
    if matrix.ndim!=2 or values.shape!=(len(matrix),):
        raise ValueError("basis and response must be aligned")
    centered_values=values-np.mean(values)
    centered_matrix=matrix[:,1:]-np.mean(matrix[:,1:],axis=0)
    if centered_matrix.shape[1]==0 or np.dot(centered_values,centered_values)<=0:
        return 0.0,0
    u,singular_values,_=np.linalg.svd(centered_matrix,full_matrices=False)
    if len(singular_values)==0 or singular_values[0]<=0:
        return 0.0,0
    tolerance=(max(centered_matrix.shape)*np.finfo(float).eps if relative_tolerance is None
               else relative_tolerance)*singular_values[0]
    rank=int(np.count_nonzero(singular_values>tolerance))
    if rank==0:
        return 0.0,0
    projection=u[:,:rank].T@centered_values
    fraction=float(np.dot(projection,projection)/np.dot(centered_values,centered_values))
    return float(np.clip(fraction,0.0,1.0)),rank


def ridge_coefficients(basis: ArrayLike, response: ArrayLike, ridge: float=2e-3) -> FloatArray:
    matrix=np.asarray(basis,dtype=float); values=np.asarray(response,dtype=float)
    penalty=np.eye(matrix.shape[1])*ridge; penalty[0,0]=0
    return np.linalg.solve(matrix.T@matrix+penalty,matrix.T@values)


def grouped_cv_r2(basis: ArrayLike, response: ArrayLike, groups: ArrayLike,
                  folds: int=6, ridge: float=2e-3, seed: int=4821) -> FloatArray:
    matrix=np.asarray(basis,dtype=float); values=np.asarray(response,dtype=float)
    group_values=np.asarray(groups,dtype=int); unique=np.unique(group_values)
    if len(unique)<2: raise ValueError("at least two independent groups are required")
    fold_count=min(folds,len(unique)); assignment=np.arange(len(unique))%fold_count
    rng=np.random.default_rng(seed); rng.shuffle(assignment); mapping=dict(zip(unique,assignment))
    scores=[]
    for fold in range(fold_count):
        test=np.array([mapping[group]==fold for group in group_values]); train=~test
        if np.count_nonzero(test)<2 or np.count_nonzero(train)<matrix.shape[1]+1: continue
        coefficients=ridge_coefficients(matrix[train],values[train],ridge)
        prediction=matrix[test]@coefficients; baseline=np.mean(values[train])
        denominator=np.sum((values[test]-baseline)**2)
        response_scale=max(float(np.sum(values[test]**2)),1.0)
        if denominator<=np.finfo(float).eps*response_scale:
            scores.append(0.0)
        else:
            scores.append(1-np.sum((values[test]-prediction)**2)/denominator)
    if not scores: raise RuntimeError("no valid grouped CV folds")
    return np.asarray(scores)


def assess_nonlocal_dependence(table: NonlocalTable, threshold: float=0.01,
                               folds: int=6, ridge: float=2e-3,
                               basis_degree: int=2) -> DependenceResult:
    coordinates,_,_=invariant_coordinates(table.state)
    basis=polynomial_basis(coordinates,basis_degree)
    svd_fraction,svd_rank=svd_retained_variance(basis,table.delta_energy)
    scores=grouped_cv_r2(basis,table.delta_energy,table.groups,folds,ridge)
    mean=float(np.mean(scores)); se=float(np.std(scores,ddof=1)/np.sqrt(len(scores))) if len(scores)>1 else np.nan
    evidence=threshold if not np.isfinite(se) else max(threshold,2*se)
    return DependenceResult(scores,mean,se,bool(mean>evidence),threshold,
                            svd_fraction,svd_rank,basis_degree)


def fit_conditional_nonlocal_weight(table: NonlocalTable,
                                    dependence: DependenceResult|None=None,
                                    ridge: float=2e-3,
                                    correction_degree: int=3) -> ConditionalNonlocalModel:
    result=assess_nonlocal_dependence(table,ridge=ridge) if dependence is None else dependence
    coordinates,center,scale=invariant_coordinates(table.state)
    degree=correction_degree if result.dependent else result.basis_degree
    boltzmann=np.exp(np.clip(-table.delta_energy,-60,60)); basis=polynomial_basis(coordinates,degree)
    if result.dependent: coefficients=ridge_coefficients(basis,boltzmann,ridge)
    else:
        coefficients=np.zeros(basis.shape[1]); coefficients[0]=np.mean(boltzmann)
    lower=max(float(np.quantile(boltzmann,0.002)),1e-12)
    upper=max(float(np.quantile(boltzmann,0.998)),lower)
    return ConditionalNonlocalModel(center,scale,coefficients,lower,upper,result,degree)


def weighted_quantile(values: ArrayLike, probabilities: ArrayLike,
                      weights: ArrayLike) -> FloatArray:
    data=np.asarray(values,dtype=float); requested=np.asarray(probabilities,dtype=float)
    local_weights=np.asarray(weights,dtype=float)
    valid=np.isfinite(data)&np.isfinite(local_weights)&(local_weights>=0)
    data,local_weights=data[valid],local_weights[valid]
    if len(data)==0 or np.sum(local_weights)<=0: raise ValueError("no positive finite mass")
    order=np.argsort(data); data,local_weights=data[order],local_weights[order]
    cumulative=np.cumsum(local_weights); cumulative/=cumulative[-1]
    return np.interp(requested,cumulative,data)


def effective_sample_fraction(weights: ArrayLike) -> float:
    values=np.asarray(weights,dtype=float); values/=np.sum(values)
    return float(1/(len(values)*np.sum(values**2)))


def _fit_two_feature_density_ratio(reference_features: FloatArray,
        target_features: FloatArray, reference_weights: FloatArray,
        target_weights: FloatArray, reference_offset: FloatArray,
        target_offset: FloatArray, support_quantiles: tuple[float,float],
        reference_mask: NDArray[np.bool_]|None,
        target_mask: NDArray[np.bool_]|None) -> DensityRatioResult:
    bounds=np.vstack([weighted_quantile(reference_features[:,i],support_quantiles,reference_weights)
                      for i in range(2)])
    keep_ref=np.all((reference_features>=bounds[:,0])&(reference_features<=bounds[:,1]),axis=1)
    keep_target=np.all((target_features>=bounds[:,0])&(target_features<=bounds[:,1]),axis=1)
    if reference_mask is not None: keep_ref&=np.asarray(reference_mask,dtype=bool)
    if target_mask is not None: keep_target&=np.asarray(target_mask,dtype=bool)
    if np.count_nonzero(keep_ref)<20 or np.count_nonzero(keep_target)<20:
        raise RuntimeError("insufficient overlap with reference support")
    xref,xtarget=reference_features[keep_ref],target_features[keep_target]
    wref=reference_weights[keep_ref]; wtarget=target_weights[keep_target]
    wref=0.5*wref/np.sum(wref); wtarget=0.5*wtarget/np.sum(wtarget)
    center=np.sum((2*wref[:,None])*xref,axis=0)
    scale=np.sqrt(np.sum((2*wref[:,None])*(xref-center)**2,axis=0)); scale=np.maximum(scale,1e-10)
    design=np.vstack(((xref-center)/scale,(xtarget-center)/scale))
    design=np.column_stack((np.ones(len(design)),design))
    labels=np.concatenate((np.zeros(len(xref)),np.ones(len(xtarget))))
    weights=np.concatenate((wref,wtarget))
    offset=np.concatenate((reference_offset[keep_ref],target_offset[keep_target]))
    def objective(parameters):
        eta=design@parameters+offset
        loss=np.sum(weights*(np.logaddexp(0,eta)-labels*eta))
        probability=1/(1+np.exp(-np.clip(eta,-40,40)))
        return float(loss),design.T@(weights*(probability-labels))
    fit=minimize(objective,np.zeros(3),jac=True,method="BFGS",options={"gtol":1e-10,"maxiter":400})
    parameters=np.asarray(fit.x); eta=design@parameters+offset
    probability=1/(1+np.exp(-np.clip(eta,-40,40)))
    hessian=design.T@((weights*probability*(1-probability))[:,None]*design)
    parameter_covariance=np.linalg.pinv(hessian*(1/np.sum(weights**2)))
    coefficient=-parameters[1:]/scale
    transform=np.diag(1/scale); covariance=transform@parameter_covariance[1:,1:]@transform
    log_weight=-(xref@coefficient)+reference_offset[keep_ref]; log_weight-=np.max(log_weight)
    tilted=wref*np.exp(np.maximum(log_weight,-60)); tilted/=np.sum(tilted)
    return DensityRatioResult(float(coefficient[0]),float(coefficient[1]),covariance,
        float(parameters[0]),float(objective(parameters)[0]),bool(fit.success),
        float(np.sum(target_weights[keep_target])),effective_sample_fraction(tilted),
        bounds,keep_ref,keep_target)


def fit_local_cell_reweighting(reference: LocalGeometrySample,
        target: LocalGeometrySample, k_ref: float, cell_length: float|None=None,
        reference_beta_delta_f_nl: ArrayLike|None=None,
        target_beta_delta_f_nl: ArrayLike|None=None,
        support_quantiles: tuple[float,float]=(0.002,0.998),
        reference_mask: NDArray[np.bool_]|None=None,
        target_mask: NDArray[np.bool_]|None=None) -> DensityRatioResult:
    """Fit effective local-cell coefficients; omitted Delta F_nl means constant."""
    xref=local_cell_features(reference,k_ref,cell_length)
    xtarget=local_cell_features(target,k_ref,cell_length)
    free_ref=np.zeros(len(reference.weights)) if reference_beta_delta_f_nl is None else np.asarray(reference_beta_delta_f_nl,dtype=float)
    free_target=np.zeros(len(target.weights)) if target_beta_delta_f_nl is None else np.asarray(target_beta_delta_f_nl,dtype=float)
    return _fit_two_feature_density_ratio(xref,xtarget,reference.weights,target.weights,
        -free_ref,-free_target,support_quantiles,reference_mask,target_mask)


def fit_contour_reweighting(reference_observables: ArrayLike,
        target_observables: ArrayLike, reference_weights: ArrayLike,
        target_weights: ArrayLike, k_ref: float,
        reference_beta_delta_f_nl: ArrayLike|None=None,
        target_beta_delta_f_nl: ArrayLike|None=None,
        support_quantiles: tuple[float,float]=(0.002,0.998)) -> DensityRatioResult:
    xref=contour_features(reference_observables,k_ref); xtarget=contour_features(target_observables,k_ref)
    wref=np.asarray(reference_weights,dtype=float); wref/=np.sum(wref)
    wtarget=np.asarray(target_weights,dtype=float); wtarget/=np.sum(wtarget)
    free_ref=np.zeros(len(wref)) if reference_beta_delta_f_nl is None else np.asarray(reference_beta_delta_f_nl,dtype=float)
    free_target=np.zeros(len(wtarget)) if target_beta_delta_f_nl is None else np.asarray(target_beta_delta_f_nl,dtype=float)
    return _fit_two_feature_density_ratio(xref,xtarget,wref,wtarget,-free_ref,-free_target,
                                          support_quantiles,None,None)


def independent_local_features(sample: LocalGeometrySample, k_ref: float) -> FloatArray:
    return np.column_stack((sample.kappa**2/k_ref**2,sample.kappa**4/k_ref**4,
                            sample.kappa_prime**2/k_ref**4,sample.kappa_tau**2/k_ref**4))


def independent_local_cell_features(sample: LocalGeometrySample, k_ref: float,
        cell_length: float|None=None) -> FloatArray:
    """Cell-integrated independent quadratic and fourth-order invariants."""
    if k_ref<=0:
        raise ValueError("k_ref must be positive")
    length=1/k_ref if cell_length is None else float(cell_length)
    if length<=0:
        raise ValueError("cell_length must be positive")
    return (length*k_ref)*independent_local_features(sample,k_ref)


def local_invariant_observables(sample: LocalGeometrySample,
                                k_ref: float) -> FloatArray:
    """Return normalized kappa2, kappa4, kappa-prime2, kappa-tau2 and J4."""
    independent=independent_local_features(sample,k_ref)
    linked_fourth=(9/64)*independent[:,1]-independent[:,2]/24-independent[:,3]/24
    return np.column_stack((independent,linked_fourth))


def weighted_mean_and_std(values: ArrayLike,
                          weights: ArrayLike) -> tuple[FloatArray,FloatArray]:
    data=np.asarray(values,dtype=float); local_weights=np.asarray(weights,dtype=float)
    if data.ndim==1: data=data[:,None]
    if len(data)!=len(local_weights) or np.sum(local_weights)<=0:
        raise ValueError("values and positive weights are required")
    local_weights=local_weights/np.sum(local_weights)
    mean=np.sum(local_weights[:,None]*data,axis=0)
    variance=np.sum(local_weights[:,None]*(data-mean)**2,axis=0)
    return mean,np.sqrt(np.maximum(variance,0.0))


def weighted_correlation_matrix(values: ArrayLike,
                                weights: ArrayLike) -> FloatArray:
    """Return the weighted Pearson correlation matrix for aligned columns."""
    data=np.asarray(values,dtype=float); local_weights=np.asarray(weights,dtype=float)
    if data.ndim!=2 or local_weights.shape!=(len(data),):
        raise ValueError("values must be a matrix with one weight per row")
    valid=np.all(np.isfinite(data),axis=1)&np.isfinite(local_weights)&(local_weights>=0)
    data=data[valid]; local_weights=local_weights[valid]
    if len(data)<2 or np.sum(local_weights)<=0:
        raise ValueError("at least two finite rows with positive total weight are required")
    local_weights=local_weights/np.sum(local_weights)
    mean=np.sum(local_weights[:,None]*data,axis=0)
    centered=data-mean
    covariance=centered.T@(local_weights[:,None]*centered)
    scale=np.sqrt(np.maximum(np.diag(covariance),0.0))
    denominator=np.outer(scale,scale)
    correlation=np.divide(covariance,denominator,
        out=np.full_like(covariance,np.nan),where=denominator>0)
    positive=scale>0
    correlation[np.diag_indices_from(correlation)]=np.where(positive,1.0,np.nan)
    return correlation


def _maximum_entropy_fit(reference_features: FloatArray,
        target_features: FloatArray, reference_weights: FloatArray,
        target_weights: FloatArray,
        support_quantiles: tuple[float,float]=(0.002,0.998),
        reference_mask: NDArray[np.bool_]|None=None,
        target_mask: NDArray[np.bool_]|None=None,
        initial: ArrayLike|None=None,
        reference_beta_delta_f_nl: ArrayLike|None=None) -> MaximumEntropyResult:
    """Fit P/P0 proportional to exp(-theta.features) by moment matching."""
    xref=np.asarray(reference_features,dtype=float)
    xtarget=np.asarray(target_features,dtype=float)
    wref_all=np.asarray(reference_weights,dtype=float)
    wtarget_all=np.asarray(target_weights,dtype=float)
    if xref.ndim!=2 or xtarget.ndim!=2 or xref.shape[1]!=xtarget.shape[1]:
        raise ValueError("reference and target features must have matching columns")
    if reference_beta_delta_f_nl is None:
        beta_delta_f=np.zeros(len(xref),dtype=float)
    else:
        beta_delta_f=np.asarray(reference_beta_delta_f_nl,dtype=float)
        if beta_delta_f.shape!=(len(xref),) or np.any(~np.isfinite(beta_delta_f)):
            raise ValueError("reference_beta_delta_f_nl must be one finite value per reference sample")
    bounds=np.vstack([weighted_quantile(xref[:,i],support_quantiles,wref_all)
                      for i in range(xref.shape[1])])
    support_ref=np.all((xref>=bounds[:,0])&(xref<=bounds[:,1]),axis=1)
    support_target=np.all((xtarget>=bounds[:,0])&(xtarget<=bounds[:,1]),axis=1)
    keep_ref=support_ref.copy(); keep_target=support_target.copy()
    if reference_mask is not None: keep_ref&=np.asarray(reference_mask,dtype=bool)
    if target_mask is not None: keep_target&=np.asarray(target_mask,dtype=bool)
    if np.count_nonzero(keep_ref)<20 or np.count_nonzero(keep_target)<20:
        raise RuntimeError("insufficient overlap with reference support")
    xref_kept=xref[keep_ref]; xtarget_kept=xtarget[keep_target]
    wref=wref_all[keep_ref]; wref/=np.sum(wref)
    wtarget=wtarget_all[keep_target]; wtarget/=np.sum(wtarget)
    feature_center=np.sum(wref[:,None]*xref_kept,axis=0)
    centered_for_scale=xref_kept-feature_center
    feature_scale=np.sqrt(np.sum(
        wref[:,None]*centered_for_scale**2,axis=0))
    feature_scale=np.maximum(feature_scale,
        np.sqrt(np.finfo(float).eps)*np.maximum(np.abs(feature_center),1.0))
    zref=(xref_kept-feature_center)/feature_scale
    ztarget=(xtarget_kept-feature_center)/feature_scale
    target_mean_z=np.sum(wtarget[:,None]*ztarget,axis=0)
    # P_target/P_reference is proportional to
    # exp[-theta.features-beta*Delta F_nl(X)].
    log_reference=(np.log(np.maximum(wref,np.finfo(float).tiny))
                   -beta_delta_f[keep_ref])
    log_reference-=logsumexp(log_reference)

    def distribution(parameters_z: FloatArray) -> tuple[FloatArray,FloatArray]:
        log_weight=log_reference-zref@parameters_z
        normalized=np.exp(log_weight-logsumexp(log_weight))
        mean=np.sum(normalized[:,None]*zref,axis=0)
        return normalized,mean

    def objective(parameters_z: FloatArray) -> tuple[float,FloatArray]:
        log_weight=log_reference-zref@parameters_z
        value=float(logsumexp(log_weight)+target_mean_z@parameters_z)
        _,mean=distribution(parameters_z)
        return value,target_mean_z-mean

    start=np.zeros(xref.shape[1]) if initial is None else np.asarray(initial,dtype=float)
    if start.shape!=(xref.shape[1],):
        raise ValueError("initial coefficients have the wrong shape")
    start_z=start*feature_scale
    fit=minimize(objective,start_z,jac=True,method="BFGS",
                 options={"gtol":1e-10,"maxiter":600})
    coefficients_z=np.asarray(fit.x,dtype=float)
    if np.any(~np.isfinite(coefficients_z)):
        raise RuntimeError(
            "maximum-entropy fit diverged; target moments are outside the "
            "resolved reference-feature overlap")
    coefficients=coefficients_z/feature_scale
    tilted,fitted_mean_z=distribution(coefficients_z)
    centered_ref=zref-fitted_mean_z
    hessian=centered_ref.T@(tilted[:,None]*centered_ref)
    target_centered=ztarget-target_mean_z
    target_covariance=target_centered.T@(wtarget[:,None]*target_centered)
    if (np.any(~np.isfinite(hessian)) or
            np.any(~np.isfinite(target_covariance))):
        raise RuntimeError(
            "maximum-entropy covariance is nonfinite; reduce feature outliers "
            "or compare better-overlapped ensembles")
    n_reference=1/np.sum(tilted**2); n_target=1/np.sum(wtarget**2)
    inverse=np.linalg.pinv(hessian)
    covariance_z=inverse@(hessian/n_reference+target_covariance/n_target)@inverse
    inverse_scale=np.diag(1/feature_scale)
    covariance=inverse_scale@covariance_z@inverse_scale
    effective_fraction=effective_sample_fraction(tilted)
    moment_scale=np.maximum(np.sqrt(np.maximum(np.diag(hessian),0.0)),1e-10)
    moment_converged=bool(np.max(
        np.abs(fitted_mean_z-target_mean_z)/moment_scale)<1e-6)
    success=bool(fit.success or moment_converged)
    message=str(fit.message) if fit.success else f"{fit.message}; moment_converged={moment_converged}"
    target_mass=float(np.sum(wtarget_all[support_target])/np.sum(wtarget_all))
    target_mean=feature_center+feature_scale*target_mean_z
    fitted_mean=feature_center+feature_scale*fitted_mean_z
    return MaximumEntropyResult(coefficients,covariance,float(objective(coefficients_z)[0]),
        success,message,target_mass,
        effective_fraction,bounds,keep_ref,keep_target,target_mean,fitted_mean)


def fit_linked_maximum_entropy(reference: LocalGeometrySample,
        target: LocalGeometrySample, k_ref: float,
        cell_length: float|None=None,
        support_quantiles: tuple[float,float]=(0.002,0.998),
        reference_mask: NDArray[np.bool_]|None=None,
        target_mask: NDArray[np.bool_]|None=None,
        initial: ArrayLike|None=None,
        reference_beta_delta_f_nl: ArrayLike|None=None) -> MaximumEntropyResult:
    """Directly fit Delta c2 and linked Delta c4 by invariant moment matching."""
    return _maximum_entropy_fit(
        local_cell_features(reference,k_ref,cell_length),
        local_cell_features(target,k_ref,cell_length),
        reference.weights,target.weights,support_quantiles,
        reference_mask,target_mask,initial,reference_beta_delta_f_nl)


def fit_independent_maximum_entropy(reference: LocalGeometrySample,
        target: LocalGeometrySample, k_ref: float,
        cell_length: float|None=None,
        support_quantiles: tuple[float,float]=(0.002,0.998),
        reference_mask: NDArray[np.bool_]|None=None,
        target_mask: NDArray[np.bool_]|None=None,
        initial: ArrayLike|None=None) -> MaximumEntropyResult:
    """Fit kappa2 and the three independent fourth-order invariants."""
    return _maximum_entropy_fit(
        independent_local_cell_features(reference,k_ref,cell_length),
        independent_local_cell_features(target,k_ref,cell_length),
        reference.weights,target.weights,support_quantiles,
        reference_mask,target_mask,initial)


def fit_contour_maximum_entropy(reference_observables: ArrayLike,
        target_observables: ArrayLike, reference_weights: ArrayLike,
        target_weights: ArrayLike, k_ref: float,
        support_quantiles: tuple[float,float]=(0.002,0.998),
        initial: ArrayLike|None=None) -> MaximumEntropyResult:
    """Direct linked fit using the note's contour-integrated K2 and K4."""
    return _maximum_entropy_fit(
        contour_features(reference_observables,k_ref),
        contour_features(target_observables,k_ref),
        np.asarray(reference_weights,dtype=float),
        np.asarray(target_weights,dtype=float),support_quantiles,
        None,None,initial)


def fit_contour_heldout_maximum_entropy(reference_observables: ArrayLike,
        target_observables: ArrayLike, k_ref: float,
        support_quantiles: tuple[float,float]=(0.002,0.998),
        train_fraction: float=0.67, seed: int=48031,
        maximum_attempts: int=24, minimum_test_count: int=4
        ) -> ContourHeldoutResult:
    """Find a reproducible feasible split for held-out contour matching.

    A finite contour sample can occasionally place the training target mean
    outside the convex support of the training reference blocks even when the
    complete adjacent ensembles overlap.  Independent deterministic splits are
    tried until both held-out ensembles retain usable common-support blocks.
    The first feasible split is returned; failed attempts are not treated as
    evidence for or against the physical model.
    """
    reference=np.asarray(reference_observables,dtype=float)
    target=np.asarray(target_observables,dtype=float)
    if (reference.ndim!=2 or target.ndim!=2 or reference.shape[1:]!=(2,) or
            target.shape[1:]!=(2,) or np.any(~np.isfinite(reference)) or
            np.any(~np.isfinite(target))):
        raise ValueError("contour observables must be finite arrays with shape (n,2)")
    if not 0<train_fraction<1:
        raise ValueError("train_fraction must lie strictly between zero and one")
    if maximum_attempts<1 or minimum_test_count<1:
        raise ValueError("attempt and held-out counts must be positive")
    reference_features=contour_features(reference,k_ref)
    target_features=contour_features(target,k_ref)
    last_error="no split retained enough training and test blocks"
    for attempt in range(maximum_attempts):
        rng=np.random.default_rng(int(seed)+1009*attempt)
        reference_train=rng.random(len(reference))<train_fraction
        target_train=rng.random(len(target))<train_fraction
        counts=(np.count_nonzero(reference_train),np.count_nonzero(~reference_train),
                np.count_nonzero(target_train),np.count_nonzero(~target_train))
        if min(counts)<minimum_test_count:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",RuntimeWarning)
                result=fit_contour_maximum_entropy(
                    reference[reference_train],target[target_train],
                    np.ones(counts[0]),np.ones(counts[2]),k_ref,
                    support_quantiles=support_quantiles)
            reference_test_keep=(~reference_train)&np.all(
                (reference_features>=result.feature_bounds[:,0])&
                (reference_features<=result.feature_bounds[:,1]),axis=1)
            target_test_keep=(~target_train)&np.all(
                (target_features>=result.feature_bounds[:,0])&
                (target_features<=result.feature_bounds[:,1]),axis=1)
            if (np.count_nonzero(reference_test_keep)<minimum_test_count or
                    np.count_nonzero(target_test_keep)<minimum_test_count):
                last_error="too few held-out blocks remained in common support"
                continue
            return ContourHeldoutResult(result,reference_train,target_train,
                                        target_test_keep,attempt)
        except (RuntimeError,ValueError) as error:
            last_error=str(error)
    raise RuntimeError(
        f"no feasible held-out contour split after {maximum_attempts} attempts: "
        f"{last_error}")


def contour_maximum_entropy_weights(observables: ArrayLike,
        result: MaximumEntropyResult, k_ref: float,
        base_weights: ArrayLike|None=None,
        mask: NDArray[np.bool_]|None=None) -> FloatArray:
    """Return normalized whole-contour weights from a contour fit."""
    features=contour_features(observables,k_ref)
    if features.shape[1]!=len(result.coefficients):
        raise ValueError("result and contour feature basis are incompatible")
    weights=(np.ones(len(features),dtype=float) if base_weights is None
             else np.asarray(base_weights,dtype=float))
    if (weights.shape!=(len(features),) or np.any(~np.isfinite(weights)) or
            np.any(weights<0) or np.sum(weights)<=0):
        raise ValueError("base_weights must be finite, nonnegative, and nonzero")
    keep=np.all((features>=result.feature_bounds[:,0])&
                (features<=result.feature_bounds[:,1]),axis=1)
    if mask is not None:
        selected=np.asarray(mask,dtype=bool)
        if selected.shape!=(len(features),):
            raise ValueError("mask must have one value per contour")
        keep&=selected
    if not np.any(keep):
        raise ValueError("no contours remain inside fit support")
    log_weight=-(features[keep]@result.coefficients)
    log_weight-=np.max(log_weight)
    output=np.zeros(len(features),dtype=float)
    output[keep]=weights[keep]*np.exp(np.maximum(log_weight,-60.0))
    output/=np.sum(output)
    return output


def local_state_point_weights(states: Sequence[ArrayLike],
        contour_weights: ArrayLike, reporting_k_ref: float
        ) -> tuple[FloatArray,FloatArray]:
    """Expand contour/block weights to normalized local-state point weights."""
    if reporting_k_ref<=0:
        raise ValueError("reporting_k_ref must be positive")
    local_states=tuple(np.asarray(state,dtype=float) for state in states)
    if any(state.ndim!=2 or state.shape[1]!=3 or len(state)==0
           for state in local_states):
        raise ValueError("states must be nonempty arrays with shape (n,3)")
    weights=np.asarray(contour_weights,dtype=float)
    if (weights.shape!=(len(local_states),) or np.any(~np.isfinite(weights)) or
            np.any(weights<0) or np.sum(weights)<=0):
        raise ValueError("contour_weights must align with traces and have positive mass")
    point_states=[]; point_weights=[]
    scale=np.array([reporting_k_ref,reporting_k_ref**2,reporting_k_ref**2])
    for state,weight in zip(local_states,weights):
        if weight<=0:
            continue
        point_states.append(state/scale)
        point_weights.append(np.full(len(state),weight/len(state),dtype=float))
    if not point_states:
        raise ValueError("no positively weighted contour points remain")
    combined_weights=np.concatenate(point_weights)
    combined_weights/=np.sum(combined_weights)
    return np.vstack(point_states),combined_weights


def trace_point_state_weights(trace: ContourTrace,
        contour_weights: ArrayLike, reporting_k_ref: float
        ) -> tuple[FloatArray,FloatArray]:
    """Compatibility wrapper expanding weights for complete stored traces."""
    return local_state_point_weights(
        trace_local_state(trace),contour_weights,reporting_k_ref)


def maximum_entropy_reweighted_weights(sample: LocalGeometrySample,
        result: MaximumEntropyResult, k_ref: float,
        cell_length: float|None=None, independent: bool=False,
        mask: NDArray[np.bool_]|None=None,
        beta_delta_f_nl: ArrayLike|None=None) -> FloatArray:
    """Return normalized full-length weights, zero outside fit support/mask."""
    features=(independent_local_cell_features(sample,k_ref,cell_length)
              if independent else local_cell_features(sample,k_ref,cell_length))
    if features.shape[1]!=len(result.coefficients):
        raise ValueError("result and feature basis are incompatible")
    if beta_delta_f_nl is None:
        beta_delta_f=np.zeros(len(sample.weights),dtype=float)
    else:
        beta_delta_f=np.asarray(beta_delta_f_nl,dtype=float)
        if beta_delta_f.shape!=(len(sample.weights),) or np.any(~np.isfinite(beta_delta_f)):
            raise ValueError("beta_delta_f_nl must be one finite value per sample")
    keep=np.all((features>=result.feature_bounds[:,0])&
                (features<=result.feature_bounds[:,1]),axis=1)
    if mask is not None: keep&=np.asarray(mask,dtype=bool)
    output=np.zeros(len(sample.weights),dtype=float)
    if not np.any(keep):
        raise ValueError("no samples remain inside fit support")
    log_weight=(np.log(np.maximum(sample.weights[keep],np.finfo(float).tiny))
                -beta_delta_f[keep]-features[keep]@result.coefficients)
    output[keep]=np.exp(log_weight-logsumexp(log_weight))
    return output


def decompose_independent_fourth_coefficients(coefficients: ArrayLike
        ) -> tuple[float,FloatArray,float]:
    """Split independent fourth-order coefficients into linked and residual parts."""
    values=np.asarray(coefficients,dtype=float)
    if values.shape!=(4,):
        raise ValueError("four independent coefficients are required")
    linked=np.array([9/64,-1/24,-1/24],dtype=float)
    delta_c4=float(np.dot(values[1:],linked)/np.dot(linked,linked))
    residual=values[1:]-delta_c4*linked
    denominator=max(float(np.linalg.norm(values[1:])),np.finfo(float).eps)
    return delta_c4,residual,float(np.linalg.norm(residual)/denominator)


def fit_independent_invariant_density_ratio(reference: LocalGeometrySample,
        target: LocalGeometrySample, k_ref: float,
        support_quantiles: tuple[float,float]=(0.002,0.998),
        reference_mask: NDArray[np.bool_]|None=None,
        target_mask: NDArray[np.bool_]|None=None) -> IndependentInvariantResult:
    """Fit kappa^2,kappa^4,(kappa')^2,(kappa*tau)^2 without Yukawa linking."""
    xref=independent_local_features(reference,k_ref); xtarget=independent_local_features(target,k_ref)
    bounds=np.vstack([weighted_quantile(xref[:,i],support_quantiles,reference.weights)
                      for i in range(xref.shape[1])])
    keep_ref=np.all((xref>=bounds[:,0])&(xref<=bounds[:,1]),axis=1)
    keep_target=np.all((xtarget>=bounds[:,0])&(xtarget<=bounds[:,1]),axis=1)
    if reference_mask is not None: keep_ref&=np.asarray(reference_mask,dtype=bool)
    if target_mask is not None: keep_target&=np.asarray(target_mask,dtype=bool)
    if np.count_nonzero(keep_ref)<20 or np.count_nonzero(keep_target)<20:
        raise RuntimeError("insufficient independent-invariant support overlap")
    xref,xtarget=xref[keep_ref],xtarget[keep_target]
    wref=reference.weights[keep_ref]; wref=0.5*wref/np.sum(wref)
    wtarget=target.weights[keep_target]; wtarget=0.5*wtarget/np.sum(wtarget)
    center=np.sum((2*wref[:,None])*xref,axis=0)
    scale=np.sqrt(np.sum((2*wref[:,None])*(xref-center)**2,axis=0)); scale=np.maximum(scale,1e-10)
    design=np.vstack(((xref-center)/scale,(xtarget-center)/scale)); design=np.column_stack((np.ones(len(design)),design))
    labels=np.concatenate((np.zeros(len(xref)),np.ones(len(xtarget))))
    weights=np.concatenate((wref,wtarget))
    def objective(parameters):
        eta=design@parameters; loss=np.sum(weights*(np.logaddexp(0,eta)-labels*eta))
        probability=1/(1+np.exp(-np.clip(eta,-40,40)))
        return float(loss),design.T@(weights*(probability-labels))
    fit=minimize(objective,np.zeros(design.shape[1]),jac=True,method="BFGS",
                 options={"gtol":1e-10,"maxiter":500})
    parameters=np.asarray(fit.x); eta=design@parameters
    probability=1/(1+np.exp(-np.clip(eta,-40,40)))
    hessian=design.T@((weights*probability*(1-probability))[:,None]*design)
    parameter_covariance=np.linalg.pinv(hessian*(1/np.sum(weights**2)))
    coefficients=-parameters[1:]/scale; transform=np.diag(1/scale)
    covariance=transform@parameter_covariance[1:,1:]@transform
    return IndependentInvariantResult(coefficients,covariance,float(parameters[0]),
                                      float(objective(parameters)[0]),bool(fit.success),bounds)


def even_local_coordinates(state: ArrayLike, k_ref: float) -> FloatArray:
    values=np.asarray(state,dtype=float)/np.array([k_ref,k_ref**2,k_ref**2])
    return np.log1p(values**2)


def polynomial_exponents(variable_count: int, maximum_degree: int) -> tuple[tuple[int,...],...]:
    exponents=[]
    for degree in range(1,maximum_degree+1):
        for indices in combinations_with_replacement(range(variable_count),degree):
            exponent=[0]*variable_count
            for index in indices: exponent[index]+=1
            exponents.append(tuple(exponent))
    return tuple(exponents)


def polynomial_matrix(coordinates: ArrayLike,
                      exponents: Sequence[tuple[int,...]]) -> FloatArray:
    values=np.asarray(coordinates,dtype=float)
    return np.column_stack([np.prod(values**np.asarray(exponent),axis=1) for exponent in exponents])


def fit_flexible_density_ratio(reference: LocalGeometrySample,
        target: LocalGeometrySample, k_ref: float, degree: int=4,
        ridge: float=1e-4,
        exponents: Sequence[tuple[int,...]]|None=None,
        reference_mask: NDArray[np.bool_]|None=None,
        target_mask: NDArray[np.bool_]|None=None) -> FlexibleDensityRatioResult:
    active=polynomial_exponents(3,degree) if exponents is None else tuple(exponents)
    keep_ref=np.ones(len(reference.weights),dtype=bool) if reference_mask is None else np.asarray(reference_mask,dtype=bool)
    keep_target=np.ones(len(target.weights),dtype=bool) if target_mask is None else np.asarray(target_mask,dtype=bool)
    cref=even_local_coordinates(reference.state,k_ref)[keep_ref]
    ctarget=even_local_coordinates(target.state,k_ref)[keep_target]
    wref=reference.weights[keep_ref]; wref/=np.sum(wref)
    wtarget=target.weights[keep_target]; wtarget/=np.sum(wtarget)
    center=np.sum(wref[:,None]*cref,axis=0)
    scale=np.sqrt(np.sum(wref[:,None]*(cref-center)**2,axis=0)); scale=np.maximum(scale,1e-8)
    sref=np.clip((cref-center)/scale,-5,5); starget=np.clip((ctarget-center)/scale,-5,5)
    raw_ref=polynomial_matrix(sref,active); raw_target=polynomial_matrix(starget,active)
    term_center=np.sum(wref[:,None]*raw_ref,axis=0)
    term_scale=np.sqrt(np.sum(wref[:,None]*(raw_ref-term_center)**2,axis=0)); term_scale=np.maximum(term_scale,1e-8)
    ref_terms=(raw_ref-term_center)/term_scale; target_terms=(raw_target-term_center)/term_scale
    design=np.vstack((ref_terms,target_terms)); design=np.column_stack((np.ones(len(design)),design))
    labels=np.concatenate((np.zeros(len(ref_terms)),np.ones(len(target_terms))))
    weights=np.concatenate((0.5*wref,0.5*wtarget))
    def objective(parameters):
        eta=design@parameters; penalty=0.5*ridge*np.sum(parameters[1:]**2)
        loss=np.sum(weights*(np.logaddexp(0,eta)-labels*eta))+penalty
        probability=1/(1+np.exp(-np.clip(eta,-40,40)))
        gradient=design.T@(weights*(probability-labels)); gradient[1:]+=ridge*parameters[1:]
        return float(loss),gradient
    fit=minimize(objective,np.zeros(design.shape[1]),jac=True,method="L-BFGS-B",
                 options={"ftol":1e-12,"maxiter":400})
    return FlexibleDensityRatioResult(np.asarray(fit.x),center,scale,term_center,term_scale,
                                      active,float(objective(np.asarray(fit.x))[0]),bool(fit.success))


def project_flexible_fit_to_linked_coefficients(sample: LocalGeometrySample,
        flexible: FlexibleDensityRatioResult, k_ref: float,
        cell_length: float|None=None,
        mask: NDArray[np.bool_]|None=None) -> LinkedCoefficientProjection:
    """Project a flexible log density ratio onto the note's K2,K4 form.

    The fitted relation is ``log(P/P0) = constant - dc2*K2 - dc4*K4``.
    ``unexplained_fraction`` measures the weighted squared part of the flexible
    log ratio not represented by those two physical invariants.
    """
    features=local_cell_features(sample,k_ref,cell_length)
    keep=np.ones(len(sample.weights),dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    if keep.shape!=(len(sample.weights),) or np.count_nonzero(keep)<4:
        raise ValueError("projection mask must retain at least four samples")
    response=flexible.log_ratio(sample.state,k_ref)[keep]
    weights=np.asarray(sample.weights[keep],dtype=float); weights/=np.sum(weights)
    design=np.column_stack((np.ones(len(response)),features[keep]))
    gram=design.T@(weights[:,None]*design)
    coefficients=np.linalg.pinv(gram)@(design.T@(weights*response))
    residual=response-design@coefficients
    centered=response-np.sum(weights*response)
    total=float(np.sum(weights*centered**2))
    residual_variance=float(np.sum(weights*residual**2))
    unexplained=0.0 if total<=np.finfo(float).eps else residual_variance/total
    effective_count=1/np.sum(weights**2)
    degrees=max(effective_count-design.shape[1],1.0)
    # ``gram`` and ``residual_variance`` use weights normalized to one.  For
    # equal weights they are X.T@X/n and RSS/n, so the usual OLS covariance is
    # inv(gram)*residual_variance/(n-p), not inv(gram)*residual_variance*n/(n-p).
    covariance_full=np.linalg.pinv(gram)*(residual_variance/degrees)
    transform=np.diag([-1.0,-1.0])
    covariance=transform@covariance_full[1:,1:]@transform
    return LinkedCoefficientProjection(
        float(-coefficients[1]),float(-coefficients[2]),covariance,
        float(coefficients[0]),float(np.clip(unexplained,0.0,1.0)),
        float(effective_count))


def reweighted_reference_weights(sample: LocalGeometrySample, result: DensityRatioResult,
        k_ref: float, cell_length: float|None=None,
        reference_beta_delta_f_nl: ArrayLike|None=None) -> FloatArray:
    features=local_cell_features(sample,k_ref,cell_length)
    free=np.zeros(len(sample.weights)) if reference_beta_delta_f_nl is None else np.asarray(reference_beta_delta_f_nl,dtype=float)
    log_weight=-result.delta_c2_kref*features[:,0]-result.delta_c4_kref3*features[:,1]-free
    log_weight-=np.max(log_weight); weights=sample.weights*np.exp(np.maximum(log_weight,-60))
    return weights/np.sum(weights)


def local_cell_support_mask(sample: LocalGeometrySample,
        result: DensityRatioResult, k_ref: float,
        cell_length: float|None=None) -> NDArray[np.bool_]:
    """Return the common local-feature support defined by a fitted model.

    This deliberately does not reuse ``result.reference_keep`` or
    ``result.target_keep``: those masks can include a training split.  The
    returned mask can therefore be combined with an independent held-out mask
    when validating reconstructed distributions.
    """
    features=local_cell_features(sample,k_ref,cell_length)
    bounds=np.asarray(result.feature_bounds,dtype=float)
    if bounds.shape!=(features.shape[1],2):
        raise ValueError("feature bounds are incompatible with local-cell features")
    return np.all((features>=bounds[:,0])&(features<=bounds[:,1]),axis=1)


def weighted_ks_distance(first_values: ArrayLike, first_weights: ArrayLike,
                         second_values: ArrayLike, second_weights: ArrayLike) -> float:
    first=np.asarray(first_values,dtype=float); second=np.asarray(second_values,dtype=float)
    wfirst=np.asarray(first_weights,dtype=float); wsecond=np.asarray(second_weights,dtype=float)
    o1=np.argsort(first); o2=np.argsort(second); first,wfirst=first[o1],wfirst[o1]; second,wsecond=second[o2],wsecond[o2]
    c1=np.cumsum(wfirst)/np.sum(wfirst); c2=np.cumsum(wsecond)/np.sum(wsecond)
    grid=np.unique(np.concatenate((first,second)))
    return float(np.max(np.abs(np.interp(grid,first,c1,left=0,right=1)-np.interp(grid,second,c2,left=0,right=1))))


def local_reconstruction_distances(reference: LocalGeometrySample,
        target: LocalGeometrySample, predicted_reference_weights: ArrayLike) -> dict[str,float]:
    predicted=np.asarray(predicted_reference_weights,dtype=float)
    return {name: weighted_ks_distance(getattr(target,name),target.weights,
                                      getattr(reference,name),predicted)
            for name in ("kappa","kappa_prime","kappa_tau")}


def local_expansion_range_sensitivity(reference: LocalGeometrySample,
        target: LocalGeometrySample, k_ref: float,
        expansion_ranges: Iterable[float]
        ) -> list[tuple[float,MaximumEntropyResult]]:
    """Repeat linked inference over candidate local expansion cutoffs."""
    return [(float(length),fit_linked_maximum_entropy(
                reference,target,k_ref,float(length)))
            for length in expansion_ranges]


def local_cell_length_sensitivity(reference: LocalGeometrySample,
        target: LocalGeometrySample, k_ref: float,
        cell_lengths: Iterable[float]) -> list[tuple[float,MaximumEntropyResult]]:
    """Compatibility alias for :func:`local_expansion_range_sensitivity`."""
    return local_expansion_range_sensitivity(
        reference,target,k_ref,cell_lengths)


def combine_local_geometry_samples(samples: Sequence[LocalGeometrySample]) -> LocalGeometrySample:
    """Pool independent realizations with equal total weight per realization."""
    if not samples:
        raise ValueError("at least one local sample is required")
    k_eff=samples[0].k_eff
    if any(not np.isclose(sample.k_eff,k_eff) for sample in samples):
        raise ValueError("all pooled samples must share k_eff")
    count=len(samples)
    return LocalGeometrySample(
        np.concatenate([sample.kappa for sample in samples]),
        np.concatenate([sample.kappa_prime for sample in samples]),
        np.concatenate([sample.kappa_tau for sample in samples]),
        np.concatenate([sample.weights/count for sample in samples]),
        k_eff,
        float(np.mean([sample.stable_fraction for sample in samples])),
        float(np.mean([sample.mean_jacobian_dimensionless for sample in samples])),
    )


def bootstrap_local_cell_replicates(reference_replicates: Sequence[LocalGeometrySample],
        target_replicates: Sequence[LocalGeometrySample], k_ref: float,
        cell_length: float|None=None, bootstrap_samples: int=200,
        seed: int=3107) -> CoefficientBootstrapResult:
    """Resample independent Sobol/field replicates, not points within a replicate."""
    if len(reference_replicates)!=len(target_replicates) or len(reference_replicates)<2:
        raise ValueError("paired reference and target replicates require at least two realizations")
    if bootstrap_samples<2: raise ValueError("bootstrap_samples must be at least two")
    rng=np.random.default_rng(seed); replicate_count=len(reference_replicates); estimates=[]
    for _ in range(bootstrap_samples):
        indices=rng.integers(0,replicate_count,size=replicate_count)
        reference=combine_local_geometry_samples([reference_replicates[i] for i in indices])
        target=combine_local_geometry_samples([target_replicates[i] for i in indices])
        result=fit_linked_maximum_entropy(reference,target,k_ref,cell_length)
        estimates.append((result.delta_c2_kref,result.delta_c4_kref3))
    values=np.asarray(estimates,dtype=float)
    return CoefficientBootstrapResult(values,np.mean(values,axis=0),np.cov(values,rowvar=False,ddof=1))


def nonlocal_tail_convergence(trace: ContourTrace, target: YukawaPotential,
        reference: YukawaPotential, base_config: NonlocalConfig,
        maximum_separations: Iterable[float]) -> list[dict[str,float]]:
    rows=[]; previous=None
    for maximum in maximum_separations:
        table=nonlocal_energy_table(trace,target,reference,
                                    replace(base_config,max_contour_separation=float(maximum)))
        mean=float(np.mean(table.delta_energy))
        increment=np.nan if previous is None else abs(mean-previous)/max(abs(mean),1e-12)
        rows.append({"max_contour_separation":float(maximum),"mean_delta_energy":mean,
            "std_delta_energy":float(np.std(table.delta_energy)),
            "relative_mean_increment":float(increment),"sample_count":float(len(table.delta_energy))})
        previous=mean
    return rows


def run_trace_convergence_sweep(project_root: Path, spectrum: FitSpectrum,
        configurations: Sequence[TraceConfig],
        metric: Callable[[ContourTrace],Mapping[str,float]]) -> list[dict[str,float]]:
    rows=[]
    for config in configurations:
        trace=build_isotropic_trace(project_root,spectrum,config)
        row={"grid_size":float(config.grid_size),"num_modes":float(config.num_modes),
             "q_spacing":config.q_spacing,"contour_count":float(len(trace.points)),
             "point_count":float(sum(len(cell) for cell in trace.points))}
        row.update({name:float(value) for name,value in metric(trace).items()}); rows.append(row)
    return rows


def yukawa_cutoff_factors(u: ArrayLike) -> tuple[FloatArray,FloatArray]:
    """Return the finite-upper-limit factors f2(u) and f4(u)."""
    values=np.asarray(u,dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values<=0):
        raise ValueError("u=ell/D must be finite and positive")
    exponential=np.exp(-values)
    # These forms avoid subtracting two numbers close to one when u is small.
    f2=gammainc(2,values)-exponential*values**2/3
    f4=gammainc(4,values)-exponential*values**4/30
    return np.maximum(f2,0.0),np.maximum(f4,0.0)


def yukawa_dimensionless_coefficients(g_over_kref: ArrayLike,
        d_kref: ArrayLike, ell_kref: ArrayLike) -> tuple[FloatArray,FloatArray]:
    """Return a*k_ref and c4*k_ref**3 in one common reference length unit."""
    strength,distance,cutoff=np.broadcast_arrays(
        np.asarray(g_over_kref,dtype=float),
        np.asarray(d_kref,dtype=float),
        np.asarray(ell_kref,dtype=float),
    )
    if (np.any(~np.isfinite(strength)) or np.any(~np.isfinite(distance)) or
            np.any(~np.isfinite(cutoff)) or np.any(strength<=0) or
            np.any(distance<=0) or np.any(cutoff<=0)):
        raise ValueError("Yukawa parameters and cutoff must be finite and positive")
    f2,f4=yukawa_cutoff_factors(cutoff/distance)
    return strength*distance**2*f2/8,strength*distance**4*f4


def yukawa_change_coefficients(g_delta_over_kref: ArrayLike,
        d_delta_kref: ArrayLike,
        ell_kref: ArrayLike) -> tuple[FloatArray,FloatArray]:
    """Return signed Delta c2*k_ref and Delta c4*k_ref**3."""
    strength,distance,cutoff=np.broadcast_arrays(
        np.asarray(g_delta_over_kref,dtype=float),
        np.asarray(d_delta_kref,dtype=float),
        np.asarray(ell_kref,dtype=float),
    )
    if (np.any(~np.isfinite(strength)) or np.any(~np.isfinite(distance)) or
            np.any(~np.isfinite(cutoff)) or np.any(distance<=0) or
            np.any(cutoff<=0)):
        raise ValueError("change-kernel parameters must be finite with positive range and cutoff")
    f2,f4=yukawa_cutoff_factors(cutoff/distance)
    return strength*distance**2*f2/8,strength*distance**4*f4


def _solve_one_yukawa(a_kref: float, c4_kref3: float,
                       ell_kref: float) -> tuple[float,float]:
    if not np.isfinite(a_kref) or not np.isfinite(c4_kref3) or not np.isfinite(ell_kref):
        raise ValueError("Yukawa coefficients and cutoff must be finite")
    if a_kref<=0 or c4_kref3<=0 or ell_kref<=0:
        raise ValueError("Yukawa coefficients and cutoff must be positive")
    ratio=c4_kref3/(8*a_kref*ell_kref**2)

    def residual(log_x: float) -> float:
        x=np.exp(log_x)
        f2,f4=yukawa_cutoff_factors(1/x)
        if float(f2)<=0:
            return -ratio
        return x**2*float(f4/f2)-ratio

    lower,upper=-30.0,30.0
    if residual(lower)>=0 or residual(upper)<=0:
        raise ValueError(
            "finite-cutoff coefficients have no resolved positive Yukawa root; "
            "typically c4/(8*a*ell**2) must lie below 1/20"
        )
    log_x=brentq(residual,lower,upper,xtol=1e-12,rtol=1e-12,maxiter=200)
    d_kref=ell_kref*np.exp(log_x)
    f2,_=yukawa_cutoff_factors(ell_kref/d_kref)
    g_over_kref=8*a_kref/(d_kref**2*float(f2))
    reconstructed=yukawa_dimensionless_coefficients(g_over_kref,d_kref,ell_kref)
    relative_residual=max(
        abs(float(reconstructed[0])-a_kref)/a_kref,
        abs(float(reconstructed[1])-c4_kref3)/c4_kref3,
    )
    if relative_residual>1e-8:
        raise RuntimeError("finite-cutoff Yukawa root failed its reconstruction check")
    return g_over_kref,d_kref


def solve_yukawa_parameters(a_kref: ArrayLike, c4_kref3: ArrayLike,
        ell_kref: ArrayLike) -> tuple[FloatArray,FloatArray]:
    """Invert finite-cutoff local coefficients to g/k_ref and D*k_ref."""
    a,c4,cutoff=np.broadcast_arrays(
        np.asarray(a_kref,dtype=float),
        np.asarray(c4_kref3,dtype=float),
        np.asarray(ell_kref,dtype=float),
    )
    strength=np.empty(a.shape,dtype=float); distance=np.empty(a.shape,dtype=float)
    for index in np.ndindex(a.shape):
        strength[index],distance[index]=_solve_one_yukawa(
            float(a[index]),float(c4[index]),float(cutoff[index]))
    return strength,distance


def solve_yukawa_change(delta_c2_kref: ArrayLike,
        delta_c4_kref3: ArrayLike,
        ell_kref: ArrayLike) -> tuple[FloatArray,FloatArray]:
    """Invert two signed interaction-change moments to g_delta and D_delta.

    A one-Yukawa continuation exists only when the two nonzero moments have the
    same sign. The zero-change case does not identify a range.
    """
    delta2,delta4,cutoff=np.broadcast_arrays(
        np.asarray(delta_c2_kref,dtype=float),
        np.asarray(delta_c4_kref3,dtype=float),
        np.asarray(ell_kref,dtype=float),
    )
    if np.any(~np.isfinite(delta2)) or np.any(~np.isfinite(delta4)):
        raise ValueError("interaction-change moments must be finite")
    if np.any(delta2==0) or np.any(delta4==0):
        raise ValueError("both interaction-change moments must be nonzero to identify D_delta")
    if np.any(np.sign(delta2)!=np.sign(delta4)):
        raise ValueError(
            "one effective Yukawa change requires Delta c2 and Delta c4 "
            "to have the same sign")
    magnitude_g,distance=solve_yukawa_parameters(
        np.abs(delta2),np.abs(delta4),cutoff)
    return np.sign(delta2)*magnitude_g,distance


def moment_matched_yukawa_change(delta_c2_kref: float,
        delta_c4_kref3: float, ell_kref: float=1.0,
        mode: str="auto", two_ranges_kref: tuple[float,float]=(0.5,2.0),
        condition_warning: float=1e6
        ) -> tuple[YukawaChangeKernel|YukawaMixtureChangeKernel,str,FloatArray,FloatArray,float]:
    """Continue two finite-cutoff moments into an explicit signed kernel.

    ``one_yukawa`` is the note's minimal continuation and is available only
    for equal-sign moments.  ``two_range`` uses two stated fixed ranges and
    solves their signed strengths linearly; it is conditional on those ranges,
    but can represent opposite-sign moments without inventing an absolute
    reference potential.  Returned kernel parameters are dimensionless, so the
    caller must convert them with its reference ``k_ref``.
    """
    if mode not in {"auto","one_yukawa","two_range"}:
        raise ValueError("mode must be 'auto', 'one_yukawa', or 'two_range'")
    moments=np.array([delta_c2_kref,delta_c4_kref3],dtype=float)
    if np.any(~np.isfinite(moments)) or not np.isfinite(ell_kref) or ell_kref<=0:
        raise ValueError("moments and cutoff must be finite with positive cutoff")
    equal_sign=bool(np.all(moments!=0) and np.sign(moments[0])==np.sign(moments[1]))
    selected="one_yukawa" if mode=="auto" and equal_sign else mode
    if selected=="auto":
        selected="two_range"
    if selected=="one_yukawa":
        try:
            strength,distance=solve_yukawa_change(moments[0],moments[1],ell_kref)
        except ValueError:
            if mode!="auto":
                raise
            # Equal signs are necessary but not sufficient at finite cutoff.
            # When the moment ratio has no positive one-range root, retain both
            # measured moments with the explicit two-range continuation.
            selected="two_range"
        else:
            ranges=np.array([float(distance)]); strengths=np.array([float(strength)])
            return YukawaChangeKernel(strengths[0],ranges[0]),selected,ranges,strengths,1.0
    ranges=np.asarray(two_ranges_kref,dtype=float)
    if ranges.shape!=(2,) or np.any(~np.isfinite(ranges)) or np.any(ranges<=0) or np.isclose(ranges[0],ranges[1]):
        raise ValueError("two_ranges_kref must contain two distinct positive ranges")
    columns=[]
    for distance in ranges:
        c2,c4=yukawa_change_coefficients(1.0,distance,ell_kref)
        columns.append([float(c2),float(c4)])
    matrix=np.asarray(columns,dtype=float).T
    condition=float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition>condition_warning:
        warnings.warn(
            f"two-range Yukawa moment map is ill-conditioned (condition={condition:.3g}); "
            "vary NONLOCAL_TWO_RANGES_KREF",RuntimeWarning,stacklevel=2)
    strengths=np.linalg.solve(matrix,moments)
    kernel=YukawaMixtureChangeKernel(strengths,ranges)
    return kernel,"two_range",ranges,strengths,condition


def physical_yukawa_change_from_moments(
        delta_c2_ksource: float, delta_c4_ksource3: float,
        source_k_eff: float, ell_ksource: float=1.0,
        mode: str="auto", two_ranges_ksource: tuple[float,float]=(0.5,2.0)
        ) -> tuple[YukawaChangeKernel|YukawaMixtureChangeKernel,str,FloatArray,FloatArray,float]:
    """Infer one step's physical change kernel using the source-state scale."""
    if source_k_eff<=0:
        raise ValueError("source_k_eff must be positive")
    kernel,selected,ranges,strengths,condition=moment_matched_yukawa_change(
        delta_c2_ksource,delta_c4_ksource3,ell_ksource,mode,
        two_ranges_ksource)
    physical_ranges=np.asarray(ranges,dtype=float)/source_k_eff
    physical_strengths=np.asarray(strengths,dtype=float)*source_k_eff
    if len(physical_ranges)==1:
        physical: YukawaChangeKernel|YukawaMixtureChangeKernel=YukawaChangeKernel(
            float(physical_strengths[0]),float(physical_ranges[0]))
    else:
        physical=YukawaMixtureChangeKernel(physical_strengths,physical_ranges)
    return physical,selected,physical_ranges,physical_strengths,condition


def combine_yukawa_change_kernels(
        kernels: Sequence[YukawaChangeKernel|YukawaMixtureChangeKernel],
        signs: ArrayLike|None=None) -> YukawaMixtureChangeKernel:
    """Return the exact signed sum of one- or multi-range change kernels."""
    if not kernels:
        raise ValueError("at least one change kernel is required")
    factors=np.ones(len(kernels),dtype=float) if signs is None else np.asarray(signs,dtype=float)
    if factors.shape!=(len(kernels),) or np.any(~np.isfinite(factors)):
        raise ValueError("signs must be finite and aligned with kernels")
    strengths=[]; ranges=[]
    for factor,kernel in zip(factors,kernels):
        if isinstance(kernel,YukawaMixtureChangeKernel):
            strengths.extend(factor*kernel.strengths)
            ranges.extend(kernel.screening_lengths)
        else:
            strengths.append(factor*kernel.strength)
            ranges.append(kernel.screening_length)
    return YukawaMixtureChangeKernel(
        np.asarray(strengths,dtype=float),np.asarray(ranges,dtype=float))


def cumulative_adjacent_yukawa_changes(
        pairs: Sequence[InteractionComparisonPair],
        kernels: Sequence[YukawaChangeKernel|YukawaMixtureChangeKernel],
        reference_concentration_mM: float
        ) -> dict[float,YukawaMixtureChangeKernel|None]:
    """Sum adjacent ``Delta V`` kernels relative to a chosen chain node."""
    if len(pairs)!=len(kernels) or not pairs:
        raise ValueError("pairs and kernels must be nonempty and aligned")
    if any(pair.comparison_mode!="adjacent" for pair in pairs):
        raise ValueError("cumulative kernels require adjacent comparison pairs")
    nodes=sorted({pair.source_concentration_mM for pair in pairs}
                 |{pair.target_concentration_mM for pair in pairs})
    matches=[index for index,value in enumerate(nodes)
             if np.isclose(value,reference_concentration_mM)]
    if len(matches)!=1:
        raise ValueError("reference concentration is not a unique chain node")
    reference_index=matches[0]
    kernel_by_source={pair.source_concentration_mM:kernels[index]
                      for index,pair in enumerate(pairs)}
    result: dict[float,YukawaMixtureChangeKernel|None]={nodes[reference_index]:None}
    for node_index,node in enumerate(nodes):
        if node_index==reference_index:
            continue
        if node_index>reference_index:
            selected=[kernel_by_source[nodes[index]]
                      for index in range(reference_index,node_index)]
            factors=np.ones(len(selected))
        else:
            selected=[kernel_by_source[nodes[index]]
                      for index in range(node_index,reference_index)]
            factors=-np.ones(len(selected))
        result[node]=combine_yukawa_change_kernels(selected,factors)
    return result


def sampled_yukawa_change_profile(
        coefficient_means: ArrayLike, coefficient_covariances: ArrayLike,
        source_k_eff: ArrayLike, distances: ArrayLike,
        ell_ksource: ArrayLike=1.0, samples: int=400, seed: int=71403,
        mode: str="auto", two_ranges_ksource: tuple[float,float]=(0.5,2.0),
        signs: ArrayLike|None=None) -> tuple[FloatArray,FloatArray,int]:
    """Monte Carlo mean and one-standard-deviation band for summed changes.

    Each row is one independently inferred transition. Invalid nonlinear
    inversions are skipped and the returned count records the retained draws.
    """
    means=np.atleast_2d(np.asarray(coefficient_means,dtype=float))
    covariances=np.asarray(coefficient_covariances,dtype=float)
    scales=np.atleast_1d(np.asarray(source_k_eff,dtype=float))
    cutoffs=np.broadcast_to(np.asarray(ell_ksource,dtype=float),(len(means),))
    distance_values=np.asarray(distances,dtype=float)
    factors=np.ones(len(means),dtype=float) if signs is None else np.asarray(signs,dtype=float)
    if (means.shape[1:]!=(2,) or covariances.shape!=(len(means),2,2) or
            scales.shape!=(len(means),) or factors.shape!=(len(means),) or
            np.any(scales<=0) or np.any(cutoffs<=0) or samples<2 or
            distance_values.ndim!=1 or len(distance_values)==0 or
            np.any(distance_values<=0)):
        raise ValueError("profile uncertainty inputs have incompatible shapes or scales")
    rng=np.random.default_rng(seed); profiles=[]
    for _ in range(samples):
        kernels=[]
        try:
            for index in range(len(means)):
                draw=rng.multivariate_normal(means[index],covariances[index])
                kernel,*_=physical_yukawa_change_from_moments(
                    draw[0],draw[1],scales[index],cutoffs[index],mode,
                    two_ranges_ksource)
                kernels.append(kernel)
            profiles.append(combine_yukawa_change_kernels(
                kernels,factors).evaluate(distance_values))
        except (ValueError,np.linalg.LinAlgError):
            continue
    if len(profiles)<2:
        raise RuntimeError("fewer than two valid sampled interaction profiles")
    values=np.asarray(profiles)
    return np.mean(values,axis=0),np.std(values,axis=0,ddof=1),len(values)


def yukawa_change_uncertainty(delta_c2_kref: float,
        delta_c4_kref3: float, coefficient_covariance: ArrayLike,
        ell_kref: float=1.0) -> tuple[float,float]:
    """Delta-method errors for the direct signed change-kernel inversion."""
    covariance=np.asarray(coefficient_covariance,dtype=float)
    if covariance.shape!=(2,2):
        raise ValueError("coefficient_covariance must have shape (2,2)")
    center=np.array([delta_c2_kref,delta_c4_kref3],dtype=float)

    def mapping(values: FloatArray) -> FloatArray:
        strength,distance=solve_yukawa_change(values[0],values[1],ell_kref)
        return np.array([float(strength),float(distance)])

    base=mapping(center); jacobian=np.empty((2,2),dtype=float)
    for column in range(2):
        step=1e-5*max(abs(center[column]),1.0)
        plus=center.copy(); minus=center.copy()
        plus[column]+=step; minus[column]-=step
        try:
            jacobian[:,column]=(mapping(plus)-mapping(minus))/(2*step)
        except ValueError:
            jacobian[:,column]=(mapping(plus)-base)/step
    output_covariance=jacobian@covariance@jacobian.T
    return tuple(np.sqrt(np.maximum(np.diag(output_covariance),0.0)))


def linked_fourth_order_check(
        potential: YukawaPotential|YukawaChangeKernel, cutoff: float,
        relative_tolerance: float=0.01, emit_warning: bool=True) -> LinkedInvariantCheck:
    """Check the residual-force correction to the linked fourth-order form."""
    if cutoff<=0 or relative_tolerance<=0:
        raise ValueError("cutoff and relative_tolerance must be positive")
    _,f4=yukawa_cutoff_factors(cutoff/potential.screening_length)
    c4=float(
        potential.strength*potential.screening_length**4*float(f4))
    residual=float(cutoff**6*potential.derivative(cutoff)/1152)
    relative=abs(residual)/max(abs(c4),np.finfo(float).tiny)
    adequate=bool(relative<=relative_tolerance)
    result=LinkedInvariantCheck(
        cutoff,c4,residual,relative,relative_tolerance,adequate)
    if emit_warning and not adequate:
        warnings.warn(
            "linked K4 residual-force correction is "
            f"{relative:.3g} of c4, above tolerance {relative_tolerance:.3g}; "
            "consider independent fourth-order invariants",
            RuntimeWarning,stacklevel=2,
        )
    return result


def relative_yukawa_parameters(delta_c2_kref: ArrayLike,
        delta_c4_kref3: ArrayLike, a0_kref: float,
        c4_0_kref3: float, ell_kref: ArrayLike=1.0,
        reference_ell_kref: float=1.0) -> tuple[FloatArray,FloatArray]:
    """Return finite-cutoff g/g0 and D/D0 from local coefficient changes."""
    delta2=np.asarray(delta_c2_kref,dtype=float); delta4=np.asarray(delta_c4_kref3,dtype=float)
    if a0_kref<=0 or c4_0_kref3<=0: raise ValueError("reference coefficients must be positive")
    a=a0_kref+delta2; c4=c4_0_kref3+delta4
    if np.any(a<=0) or np.any(c4<=0): raise ValueError("changes produce nonpositive Yukawa products")
    g0,d0=solve_yukawa_parameters(a0_kref,c4_0_kref3,reference_ell_kref)
    g,d=solve_yukawa_parameters(a,c4,ell_kref)
    return np.asarray(g/float(g0)),np.asarray(d/float(d0))


def short_range_relative_yukawa_parameters(delta_c2_kref: ArrayLike,
        delta_c4_kref3: ArrayLike, a0_kref: float,
        c4_0_kref3: float) -> tuple[FloatArray,FloatArray]:
    """Conditional relative mapping in the short-range ``f2=f4=1`` limit."""
    delta2,delta4=np.broadcast_arrays(
        np.asarray(delta_c2_kref,dtype=float),
        np.asarray(delta_c4_kref3,dtype=float),
    )
    if a0_kref<=0 or c4_0_kref3<=0:
        raise ValueError("short-range reference coefficients must be positive")
    a=a0_kref+delta2; c4=c4_0_kref3+delta4
    if np.any(a<=0) or np.any(c4<=0):
        raise ValueError("coefficient changes make the conditional short-range mapping nonpositive")
    d_ratio=np.sqrt(c4*a0_kref/(c4_0_kref3*a))
    g_ratio=a**2*c4_0_kref3/(a0_kref**2*c4)
    return np.asarray(g_ratio),np.asarray(d_ratio)


def short_range_relative_yukawa_uncertainty(delta_c2_kref: float,
        delta_c4_kref3: float, coefficient_covariance: ArrayLike,
        a0_kref: float, c4_0_kref3: float) -> tuple[float,float]:
    """Delta-method errors for the conditional short-range relative mapping."""
    covariance=np.asarray(coefficient_covariance,dtype=float)
    if covariance.shape!=(2,2):
        raise ValueError("coefficient_covariance must have shape (2,2)")
    g_ratio,d_ratio=short_range_relative_yukawa_parameters(
        delta_c2_kref,delta_c4_kref3,a0_kref,c4_0_kref3)
    a=a0_kref+delta_c2_kref; c4=c4_0_kref3+delta_c4_kref3
    jacobian=np.array([
        [float(g_ratio)*2/a,-float(g_ratio)/c4],
        [-float(d_ratio)/(2*a),float(d_ratio)/(2*c4)],
    ])
    output=jacobian@covariance@jacobian.T
    errors=np.sqrt(np.maximum(np.diag(output),0.0))
    return float(errors[0]),float(errors[1])


def conditional_relative_yukawa_family(
        delta_c2_kref: ArrayLike, delta_c4_kref3: ArrayLike,
        reference_g0_over_kref: ArrayLike, reference_d0_kref: ArrayLike,
        ell_kref: ArrayLike=1.0,
        reference_ell_kref: float=1.0) -> ConditionalYukawaFamily:
    """Map local coefficient changes over a grid of zero-salt references.

    ``ell_kref`` may contain one target cutoff per coefficient pair, while
    ``reference_ell_kref`` defines the zero-salt cutoff. This is a conditional
    sensitivity mapping, not an inference of the reference scales.
    """
    delta2=np.asarray(delta_c2_kref,dtype=float)
    delta4=np.asarray(delta_c4_kref3,dtype=float)
    g0_values=np.asarray(reference_g0_over_kref,dtype=float)
    d0_values=np.asarray(reference_d0_kref,dtype=float)
    target_cutoffs=np.broadcast_to(np.asarray(ell_kref,dtype=float),delta2.shape)
    if delta2.ndim!=1 or delta4.shape!=delta2.shape or len(delta2)==0:
        raise ValueError("coefficient changes must be aligned nonempty 1D arrays")
    if g0_values.ndim!=1 or d0_values.ndim!=1 or len(g0_values)==0 or len(d0_values)==0:
        raise ValueError("reference grids must be nonempty 1D arrays")
    if (np.any(~np.isfinite(delta2)) or np.any(~np.isfinite(delta4)) or
            np.any(~np.isfinite(g0_values)) or np.any(~np.isfinite(d0_values)) or
            np.any(g0_values<=0) or np.any(d0_values<=0) or
            np.any(~np.isfinite(target_cutoffs)) or np.any(target_cutoffs<=0) or
            not np.isfinite(reference_ell_kref) or reference_ell_kref<=0):
        raise ValueError("coefficients must be finite and reference scales positive")

    reference_pairs=np.asarray(list(product(g0_values,d0_values)),dtype=float)
    g_ratio=np.full((len(reference_pairs),len(delta2)),np.nan,dtype=float)
    d_ratio=np.full_like(g_ratio,np.nan)
    valid=np.zeros(g_ratio.shape,dtype=bool)
    for row,(g0,d0) in enumerate(reference_pairs):
        a0,c40=yukawa_dimensionless_coefficients(g0,d0,reference_ell_kref)
        for column,(change2,change4,target_cutoff) in enumerate(
                zip(delta2,delta4,target_cutoffs)):
            try:
                relative_g,relative_d=relative_yukawa_parameters(
                    change2,change4,float(a0),float(c40),
                    ell_kref=target_cutoff,
                    reference_ell_kref=reference_ell_kref)
            except ValueError:
                continue
            g_ratio[row,column]=float(relative_g)
            d_ratio[row,column]=float(relative_d)
            valid[row,column]=True
    return ConditionalYukawaFamily(
        reference_pairs[:,0],reference_pairs[:,1],g_ratio,d_ratio,valid)


def conditional_short_range_yukawa_family(
        delta_c2_kref: ArrayLike, delta_c4_kref3: ArrayLike,
        reference_g0_over_kref: ArrayLike,
        reference_d0_kref: ArrayLike) -> ConditionalYukawaFamily:
    """Short-range relative mappings over a grid of reference potentials."""
    delta2=np.asarray(delta_c2_kref,dtype=float)
    delta4=np.asarray(delta_c4_kref3,dtype=float)
    g0_values=np.asarray(reference_g0_over_kref,dtype=float)
    d0_values=np.asarray(reference_d0_kref,dtype=float)
    if (delta2.ndim!=1 or delta4.shape!=delta2.shape or len(delta2)==0 or
            g0_values.ndim!=1 or d0_values.ndim!=1 or len(g0_values)==0 or
            len(d0_values)==0 or np.any(g0_values<=0) or np.any(d0_values<=0)):
        raise ValueError("coefficient changes and reference grids are invalid")
    reference_pairs=np.asarray(list(product(g0_values,d0_values)),dtype=float)
    g_ratio=np.full((len(reference_pairs),len(delta2)),np.nan,dtype=float)
    d_ratio=np.full_like(g_ratio,np.nan); valid=np.zeros(g_ratio.shape,dtype=bool)
    for row,(g0,d0) in enumerate(reference_pairs):
        a0=g0*d0**2/8.0; c40=g0*d0**4
        for column,(change2,change4) in enumerate(zip(delta2,delta4)):
            try:
                relative_g,relative_d=short_range_relative_yukawa_parameters(
                    change2,change4,a0,c40)
            except ValueError:
                continue
            g_ratio[row,column]=float(relative_g)
            d_ratio[row,column]=float(relative_d)
            valid[row,column]=True
    return ConditionalYukawaFamily(
        reference_pairs[:,0],reference_pairs[:,1],g_ratio,d_ratio,valid)


def normalized_yukawa_profile(reduced_distance: ArrayLike,
        g_over_g0: ArrayLike, d_over_d0: ArrayLike) -> FloatArray:
    """Return ``D0*beta*V/g0`` on the common coordinate ``r/D0``."""
    distance=np.asarray(reduced_distance,dtype=float)
    strength=np.asarray(g_over_g0,dtype=float)
    screening=np.asarray(d_over_d0,dtype=float)
    if distance.ndim!=1 or len(distance)==0 or np.any(~np.isfinite(distance)) or np.any(distance<=0):
        raise ValueError("reduced_distance must be a nonempty positive 1D array")
    if (np.any(~np.isfinite(strength)) or np.any(~np.isfinite(screening)) or
            np.any(strength<=0) or np.any(screening<=0)):
        raise ValueError("relative Yukawa parameters must be finite and positive")
    return (strength[...,None]
            *np.exp(-distance/screening[...,None])/distance)


def relative_yukawa_uncertainty(delta_c2_kref: float, delta_c4_kref3: float,
        coefficient_covariance: ArrayLike, a0_kref: float,
        c4_0_kref3: float, ell_kref: float=1.0,
        reference_ell_kref: float=1.0) -> tuple[float,float]:
    """Numerical delta-method errors for the implicit finite-cutoff inversion."""
    covariance=np.asarray(coefficient_covariance,dtype=float)
    if covariance.shape!=(2,2): raise ValueError("coefficient_covariance must have shape (2,2)")
    center=np.array([delta_c2_kref,delta_c4_kref3],dtype=float)

    def mapping(values: FloatArray) -> FloatArray:
        g,d=relative_yukawa_parameters(
            values[0],values[1],a0_kref,c4_0_kref3,
            ell_kref,reference_ell_kref)
        return np.array([float(g),float(d)])

    jacobian=np.empty((2,2),dtype=float)
    base=mapping(center)
    for column in range(2):
        step=1e-5*max(abs(center[column]),
                      abs(a0_kref if column==0 else c4_0_kref3),1.0)
        plus=center.copy(); minus=center.copy()
        plus[column]+=step; minus[column]-=step
        try:
            jacobian[:,column]=(mapping(plus)-mapping(minus))/(2*step)
        except ValueError:
            jacobian[:,column]=(mapping(plus)-base)/step
    output_covariance=jacobian@covariance@jacobian.T
    return tuple(np.sqrt(np.maximum(np.diag(output_covariance),0.0)))


def relative_yukawa_potentials(k_ref: float, g0_over_kref: float,
        d0_kref: float, delta_c2_kref: float,
        delta_c4_kref3: float, target_ell_kref: float=1.0,
        reference_ell_kref: float=1.0
        ) -> tuple[YukawaPotential,YukawaPotential,float,float,float,float]:
    """Construct reference and target kernels from relative coefficient changes.

    The coefficient products include the finite-cutoff factors evaluated at
    the reference and target values of ell*k_ref.
    """
    if k_ref<=0 or g0_over_kref<=0 or d0_kref<=0:
        raise ValueError("reference normalization and k_ref must be positive")
    a0_kref,c4_0_kref3=yukawa_dimensionless_coefficients(
        g0_over_kref,d0_kref,reference_ell_kref)
    g_relative,d_relative=relative_yukawa_parameters(
        np.array([delta_c2_kref]),np.array([delta_c4_kref3]),
        float(a0_kref),float(c4_0_kref3),
        target_ell_kref,reference_ell_kref)
    reference=YukawaPotential(g0_over_kref*k_ref,d0_kref/k_ref)
    target=YukawaPotential(float(g_relative[0])*reference.strength,
                            float(d_relative[0])*reference.screening_length)
    return reference,target,float(g_relative[0]),float(d_relative[0]),a0_kref,c4_0_kref3


def fit_direct_nonlocal_change_correction(trace: ContourTrace,
        reference_sample: LocalGeometrySample, target_sample: LocalGeometrySample,
        k_ref: float, initial_delta_c2_kref: float,
        initial_delta_c4_kref3: float, config: NonlocalConfig,
        cell_length: float|None=None, maximum_iterations: int=4,
        coefficient_tolerance: float=5e-3, dependence_threshold: float=0.01,
        correction_degree: int=3, ridge: float=2e-3,
        linked_residual_tolerance: float=0.01,
        kernel_relative_tolerance: float=0.01) -> DirectNonlocalChangeResult:
    """Iterate Delta F_nl using a Yukawa representation of Delta V itself."""
    local_length=1/k_ref if cell_length is None else float(cell_length)
    if (maximum_iterations<1 or coefficient_tolerance<=0 or
            kernel_relative_tolerance<=0):
        raise ValueError("iteration controls must be positive")
    if not np.isclose(config.local_cutoff,local_length,rtol=1e-10,atol=1e-12):
        raise ValueError("nonlocal cutoff and local expansion range must agree")
    if not np.isclose(trace.k_eff,k_ref,rtol=1e-8,atol=1e-12):
        raise ValueError("trace and reference k_eff must agree")
    ell_kref=config.local_cutoff*k_ref
    current=np.array([initial_delta_c2_kref,initial_delta_c4_kref3],dtype=float)
    history=[current.copy()]; converged=False
    initial_strength,initial_distance=solve_yukawa_change(
        current[0],current[1],ell_kref)
    kernel_history=[np.array([float(initial_strength),float(initial_distance)])]
    for _ in range(maximum_iterations):
        strength_bar,distance_bar=solve_yukawa_change(
            current[0],current[1],ell_kref)
        change_kernel=YukawaChangeKernel(
            float(strength_bar)*k_ref,float(distance_bar)/k_ref)
        table=nonlocal_change_cell_hamiltonian(trace,change_kernel,config)
        dependence=assess_nonlocal_dependence(
            table,threshold=dependence_threshold,ridge=ridge)
        model=fit_conditional_nonlocal_weight(
            table,dependence,ridge=ridge,correction_degree=correction_degree)
        reference_free_energy=model.beta_delta_free_energy(
            normalized_local_state(reference_sample,k_ref))
        target_free_energy=model.beta_delta_free_energy(
            normalized_local_state(target_sample,k_ref))
        corrected_fit=fit_local_cell_reweighting(
            reference_sample,target_sample,k_ref,local_length,
            reference_beta_delta_f_nl=-reference_free_energy,
            target_beta_delta_f_nl=-target_free_energy)
        updated=np.array([
            corrected_fit.delta_c2_kref,corrected_fit.delta_c4_kref3])
        # Re-infer the change kernel so D_delta is part of the fixed point.
        updated_strength,updated_distance=solve_yukawa_change(
            updated[0],updated[1],ell_kref)
        updated_kernel=np.array(
            [float(updated_strength),float(updated_distance)])
        history.append(updated.copy())
        kernel_history.append(updated_kernel)
        coefficient_difference=np.max(np.abs(updated-current))
        kernel_scale=np.maximum(np.abs(kernel_history[-2]),1e-12)
        kernel_difference=np.max(
            np.abs(updated_kernel-kernel_history[-2])/kernel_scale)
        current=updated
        if (coefficient_difference<=coefficient_tolerance and
                kernel_difference<=kernel_relative_tolerance):
            converged=True
            break
    strength_bar,distance_bar=kernel_history[-1]
    change_kernel=YukawaChangeKernel(
        float(strength_bar)*k_ref,float(distance_bar)/k_ref)
    linked_check=linked_fourth_order_check(
        change_kernel,config.local_cutoff,
        linked_residual_tolerance,emit_warning=True)
    return DirectNonlocalChangeResult(
        change_kernel,float(strength_bar),float(distance_bar),table,
        dependence,model,corrected_fit,np.asarray(history),
        np.asarray(kernel_history),converged,linked_check)


def fit_nonlocal_corrected_maximum_entropy(trace: ContourTrace,
        reference_sample: LocalGeometrySample, target_sample: LocalGeometrySample,
        k_ref: float, initial_delta_c2_kref: float,
        initial_delta_c4_kref3: float, config: NonlocalConfig,
        cell_length: float|None=None, maximum_iterations: int=5,
        coefficient_tolerance: float=5e-3, damping: float=0.5,
        dependence_threshold: float=0.01, correction_degree: int=3,
        ridge: float=2e-3, kernel_mode: str="auto",
        two_ranges_kref: tuple[float,float]=(0.5,2.0),
        shape_dependence_required: bool|None=None,
        force_shape_dependence: bool=False,
        support_quantiles: tuple[float,float]=(0.002,0.998),
        reference_mask: NDArray[np.bool_]|None=None,
        target_mask: NDArray[np.bool_]|None=None) -> NonlocalMaximumEntropyResult:
    """Self-consistently estimate and apply ``exp(-beta*Delta F_nl(X))``.

    The nonlocal free energy is estimated numerically on traced reference
    contours, conditioned on the local invariant state, and evaluated on every
    reference sample.  The corrected base weights are then used by the linked
    maximum-entropy fit.  No absolute ``g0`` or ``D0`` enters this calculation.
    """
    local_length=1/k_ref if cell_length is None else float(cell_length)
    if maximum_iterations<1 or coefficient_tolerance<=0 or not 0<damping<=1:
        raise ValueError("iteration controls must be positive and 0 < damping <= 1")
    if not np.isclose(config.local_cutoff,local_length,rtol=1e-10,atol=1e-12):
        raise ValueError("nonlocal cutoff and local expansion range must agree")
    if not np.isclose(trace.k_eff,k_ref,rtol=1e-8,atol=1e-12):
        raise ValueError("trace and reference k_eff must agree")
    ell_kref=config.local_cutoff*k_ref
    current=np.array([initial_delta_c2_kref,initial_delta_c4_kref3],dtype=float)
    if np.any(~np.isfinite(current)) or np.allclose(current,0):
        raise ValueError("a nonzero finite initial local fit is required")
    state=normalized_local_state(reference_sample,k_ref)
    history=[current.copy()]; converged=False

    def correction(values: FloatArray) -> tuple[
            YukawaChangeKernel|YukawaMixtureChangeKernel,str,FloatArray,FloatArray,
            float,NonlocalTable,DependenceResult,ConditionalNonlocalModel,FloatArray]:
        dimensionless,selected,ranges,strengths,condition=moment_matched_yukawa_change(
            values[0],values[1],ell_kref,kernel_mode,two_ranges_kref)
        if isinstance(dimensionless,YukawaMixtureChangeKernel):
            physical=YukawaMixtureChangeKernel(
                dimensionless.strengths*k_ref,
                dimensionless.screening_lengths/k_ref)
        else:
            physical=YukawaChangeKernel(
                dimensionless.strength*k_ref,
                dimensionless.screening_length/k_ref)
        table=nonlocal_change_cell_hamiltonian(trace,physical,config)
        dependence=assess_nonlocal_dependence(
            table,threshold=dependence_threshold,ridge=ridge)
        if force_shape_dependence:
            required=True
        elif shape_dependence_required is None:
            required=dependence.dependent
        else:
            # The reference check is a screening step.  A varying correction
            # is retained only if the actual reference-to-target change is
            # also predictable from local shape.
            required=bool(shape_dependence_required) and dependence.dependent
        model_decision=DependenceResult(
            dependence.cv_r2,dependence.mean_r2,dependence.standard_error,
            required,dependence.threshold,dependence.svd_retained_fraction,
            dependence.svd_rank,dependence.basis_degree)
        model=fit_conditional_nonlocal_weight(
            table,model_decision,ridge=ridge,correction_degree=correction_degree)
        beta_delta_f=model.beta_delta_free_energy(state)
        return physical,selected,ranges,strengths,condition,table,dependence,model,beta_delta_f

    for _ in range(maximum_iterations):
        _,_,_,_,_,_,_,_,beta_delta_f=correction(current)
        fit=fit_linked_maximum_entropy(
            reference_sample,target_sample,k_ref,local_length,support_quantiles,
            reference_mask,target_mask,current,
            reference_beta_delta_f_nl=beta_delta_f)
        raw=np.asarray(fit.coefficients,dtype=float)
        residual=float(np.max(np.abs(raw-current)))
        current=current+damping*(raw-current)
        history.append(current.copy())
        if residual<=coefficient_tolerance:
            current=raw
            history[-1]=current.copy()
            converged=True
            break

    physical,selected,ranges,strengths,condition,table,dependence,model,beta_delta_f=correction(current)
    corrected_fit=fit_linked_maximum_entropy(
        reference_sample,target_sample,k_ref,local_length,support_quantiles,
        reference_mask,target_mask,current,
        reference_beta_delta_f_nl=beta_delta_f)
    log_base=(np.log(np.maximum(reference_sample.weights,np.finfo(float).tiny))
              -beta_delta_f)
    corrected_base=np.exp(log_base-logsumexp(log_base))
    if force_shape_dependence:
        required=True
    elif shape_dependence_required is None:
        required=dependence.dependent
    else:
        required=bool(shape_dependence_required) and dependence.dependent
    return NonlocalMaximumEntropyResult(
        physical,selected,np.asarray(ranges),np.asarray(strengths),condition,table,
        dependence,required,model,corrected_fit,beta_delta_f,corrected_base,
        np.asarray(history),converged)


def fit_relative_nonlocal_correction(trace: ContourTrace,
        reference_sample: LocalGeometrySample, target_sample: LocalGeometrySample,
        k_ref: float, initial_delta_c2_kref: float,
        initial_delta_c4_kref3: float, g0_over_kref: float,
        d0_kref: float, config: NonlocalConfig, cell_length: float|None=None,
        maximum_iterations: int=4, coefficient_tolerance: float=5e-3,
        dependence_threshold: float=0.01, correction_degree: int=3,
        ridge: float=2e-3, dependence_mode: str="relative_sensitivity",
        linked_residual_tolerance: float=0.01) -> RelativeNonlocalCorrectionResult:
    """Iterate relative potentials and the conditional nonlocal correction.

    The initial local coefficients define candidate ``g/g0`` and ``D/D0``.
    The default relative_sensitivity mode is conditional on the supplied
    reference normalization. The absolute mode uses the manuscript's reference
    epsilon_nl response for the dependence decision.
    """
    local_length=1/k_ref if cell_length is None else float(cell_length)
    if maximum_iterations<1 or coefficient_tolerance<=0:
        raise ValueError("iteration controls must be positive")
    if dependence_mode=="relative":
        warnings.warn(
            "dependence_mode='relative' is deprecated; use "
            "'relative_sensitivity' to expose the conditional normalization",
            DeprecationWarning,stacklevel=2,
        )
        dependence_mode="relative_sensitivity"
    if dependence_mode not in {"relative_sensitivity","absolute"}:
        raise ValueError(
            "dependence_mode must be 'relative_sensitivity' or 'absolute'")
    if not np.isclose(config.local_cutoff,local_length,rtol=1e-10,atol=1e-12):
        raise ValueError("nonlocal cutoff and local expansion range must agree")
    if not np.isclose(trace.k_eff,k_ref,rtol=1e-8,atol=1e-12):
        raise ValueError("trace and reference k_eff must agree")
    current=np.array([initial_delta_c2_kref,initial_delta_c4_kref3],dtype=float)
    history=[current.copy()]; converged=False
    target_ell_kref=k_ref/target_sample.k_eff
    absolute_table=None
    absolute_dependence=None
    if dependence_mode=="absolute":
        fixed_reference=YukawaPotential(g0_over_kref*k_ref,d0_kref/k_ref)
        absolute_table=absolute_nonlocal_energy_density(trace,fixed_reference,config)
        absolute_dependence=assess_nonlocal_dependence(
            absolute_table,threshold=dependence_threshold,ridge=ridge)
    for _ in range(maximum_iterations):
        reference_potential,target_potential,g_relative,d_relative,_,_=relative_yukawa_potentials(
            k_ref,g0_over_kref,d0_kref,float(current[0]),float(current[1]),
            target_ell_kref=target_ell_kref)
        table=relative_nonlocal_cell_hamiltonian(
            trace,target_potential,reference_potential,config)
        if dependence_mode=="relative_sensitivity":
            dependence_table=table
            dependence=assess_nonlocal_dependence(
                dependence_table,threshold=dependence_threshold,ridge=ridge)
        else:
            dependence_table=absolute_table
            dependence=absolute_dependence
        model=fit_conditional_nonlocal_weight(
            table,dependence,ridge=ridge,correction_degree=correction_degree)
        reference_free_energy=model.beta_delta_free_energy(
            normalized_local_state(reference_sample,k_ref))
        target_free_energy=model.beta_delta_free_energy(
            normalized_local_state(target_sample,k_ref))
        corrected_fit=fit_local_cell_reweighting(
            reference_sample,target_sample,k_ref,local_length,
            reference_beta_delta_f_nl=-reference_free_energy,
            target_beta_delta_f_nl=-target_free_energy)
        updated=np.array([corrected_fit.delta_c2_kref,
                          corrected_fit.delta_c4_kref3])
        history.append(updated.copy())
        if np.max(np.abs(updated-current))<=coefficient_tolerance:
            converged=True
            break
        current=updated
    reference_linked_check=linked_fourth_order_check(
        reference_potential,1/k_ref,
        linked_residual_tolerance,emit_warning=True)
    target_linked_check=linked_fourth_order_check(
        target_potential,1/target_sample.k_eff,
        linked_residual_tolerance,emit_warning=True)
    return RelativeNonlocalCorrectionResult(
        reference_potential,target_potential,g_relative,d_relative,table,
        dependence,model,corrected_fit,np.asarray(history),converged,
        dependence_mode,dependence_table,reference_linked_check,target_linked_check)


def fit_nonlocal_normalization_sensitivity(trace: ContourTrace,
        reference_sample: LocalGeometrySample, target_sample: LocalGeometrySample,
        k_ref: float, initial_delta_c2_kref: float,
        initial_delta_c4_kref3: float, g0_over_kref_values: Iterable[float],
        d0_kref_values: Iterable[float], config: NonlocalConfig,
        cell_length: float|None=None, maximum_iterations: int=4,
        coefficient_tolerance: float=5e-3, dependence_threshold: float=0.01,
        correction_degree: int=3, ridge: float=2e-3,
        dependence_mode: str="relative_sensitivity",
        linked_residual_tolerance: float=0.01,
        robustness_tolerance: ArrayLike=(0.02,0.02)
        ) -> NonlocalNormalizationSensitivity:
    """Evaluate conditional nonlocal fits across reference nuisance values."""
    strengths=np.asarray(tuple(g0_over_kref_values),dtype=float)
    distances=np.asarray(tuple(d0_kref_values),dtype=float)
    tolerance=np.asarray(robustness_tolerance,dtype=float)
    if (strengths.ndim!=1 or distances.ndim!=1 or len(strengths)==0 or
            len(distances)==0 or np.any(~np.isfinite(strengths)) or
            np.any(~np.isfinite(distances)) or np.any(strengths<=0) or
            np.any(distances<=0)):
        raise ValueError("reference-normalization grids must be nonempty and positive")
    if tolerance.shape!=(2,) or np.any(~np.isfinite(tolerance)) or np.any(tolerance<=0):
        raise ValueError("robustness_tolerance must contain two positive values")
    results={}; failures={}
    for strength,distance in product(strengths,distances):
        key=(float(strength),float(distance))
        try:
            results[key]=fit_relative_nonlocal_correction(
                trace,reference_sample,target_sample,k_ref,
                initial_delta_c2_kref,initial_delta_c4_kref3,
                key[0],key[1],config,cell_length,
                maximum_iterations,coefficient_tolerance,
                dependence_threshold,correction_degree,ridge,
                dependence_mode,linked_residual_tolerance)
        except (ValueError,RuntimeError) as error:
            failures[key]=str(error)
    if not results:
        raise RuntimeError(
            "no reference normalization produced a valid conditional nonlocal fit")
    coefficients=np.asarray([
        [result.corrected_fit.delta_c2_kref,
         result.corrected_fit.delta_c4_kref3]
        for result in results.values()
    ])
    minimum=np.min(coefficients,axis=0); maximum=np.max(coefficients,axis=0)
    span=maximum-minimum
    dependence_values={result.dependence.dependent for result in results.values()}
    dependence_consistent=len(dependence_values)==1
    all_converged=all(result.converged for result in results.values())
    linked_adequate=all(
        result.reference_linked_check.adequate and result.target_linked_check.adequate
        for result in results.values()
    )
    stable=bool(
        np.all(span<=tolerance) and not failures and dependence_consistent and
        all_converged and linked_adequate
    )
    return NonlocalNormalizationSensitivity(
        results,failures,
        np.array([initial_delta_c2_kref,initial_delta_c4_kref3],dtype=float),
        coefficients,minimum,maximum,span,tolerance,
        stable,dependence_consistent,all_converged,linked_adequate,
    )


def relative_yukawa_sensitivity_grid(delta_c2_kref: float, delta_c4_kref3: float,
        a0_kref_values: Iterable[float], c4_0_kref3_values: Iterable[float],
        ell_kref: float=1.0, reference_ell_kref: float=1.0
        ) -> list[dict[str,float]]:
    """Map reference-input sensitivity when only plausible ranges are available."""
    rows=[]
    for a0 in a0_kref_values:
        for c40 in c4_0_kref3_values:
            try: g,d=relative_yukawa_parameters(
                np.array([delta_c2_kref]),np.array([delta_c4_kref3]),
                float(a0),float(c40),ell_kref,reference_ell_kref)
            except ValueError: continue
            rows.append({"a0_kref":float(a0),"c4_0_kref3":float(c40),
                         "g_over_g0":float(g[0]),"D_over_D0":float(d[0])})
    return rows


__all__=[name for name in globals() if not name.startswith("_") and name not in {
    "csv","importlib","re","sys","warnings","np","qmc","norm","gammainc","gammaln",
    "brentq","minimize","CubicSpline",
    "ArrayLike","NDArray","Callable","Iterable","Mapping","Sequence","Path","product",
    "combinations_with_replacement","dataclass","field","replace"}]
