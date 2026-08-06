"""Curvefit helpers for stitched scattering data and heterogeneous-line models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping
import sys
from math import erfc

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.stats import gamma as gamma_dist, norm


GAMMA_SVD1_WIDTH_BASIS_PATH = Path(__file__).with_name("svd_width_basis.npz")
GAUSSIAN_SVD1_WIDTH_BASIS_PATH = (
    Path(__file__).resolve().parents[1]
    / "rw_line_scan_output"
    / "gaussian_width_svd_model.npz"
)
SVD1_WIDTH_BASIS_PATH = GAMMA_SVD1_WIDTH_BASIS_PATH


@dataclass(frozen=True)
class SvdWidthEstimate:
    """Width estimate obtained from the first line-scan SVD basis vector."""

    r_sigma_k: float
    svd1_coefficient: float
    k_eff: float
    q_min: float
    q_max: float
    n_points: int
    reconstruction_rmse: float
    k_distribution: str


@dataclass(frozen=True)
class LinearizedBootstrapResult:
    """Parameter samples from a fixed-Jacobian residual bootstrap."""

    parameter_names: tuple[str, ...]
    samples: np.ndarray
    deltas: np.ndarray
    jacobian_rank: int
    jacobian_condition: float


def linearized_residual_bootstrap(
    parameter_names: tuple[str, ...],
    center: np.ndarray,
    data_jacobian: np.ndarray,
    standardized_residuals: np.ndarray,
    *,
    fixed_jacobian_rows: np.ndarray | None = None,
    n_bootstrap: int = 4000,
    random_seed: int = 12345,
    rcond: float = 1.0e-10,
) -> LinearizedBootstrapResult:
    """Bootstrap local parameter changes without repeating nonlinear fits.

    ``data_jacobian`` is the derivative of the standardized fitted curve with
    respect to ``center``. Experimental standardized residuals are centered
    and resampled, while optional fixed rows (for priors or anchors) receive a
    zero bootstrap perturbation. The returned approximation solves the local
    least-squares problem with one fixed pseudoinverse.
    """

    names = tuple(str(name) for name in parameter_names)
    center = np.asarray(center, dtype=float)
    jacobian = np.asarray(data_jacobian, dtype=float)
    residuals = np.asarray(standardized_residuals, dtype=float)
    if center.shape != (len(names),):
        raise ValueError("center must contain one value for every parameter name.")
    if jacobian.ndim != 2 or jacobian.shape != (residuals.size, center.size):
        raise ValueError(
            "data_jacobian must have shape (n_residuals, n_parameters)."
        )
    if residuals.ndim != 1 or residuals.size < center.size + 1:
        raise ValueError("Need more one-dimensional residuals than parameters.")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(jacobian)):
        raise ValueError("Bootstrap center and Jacobian must be finite.")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("Bootstrap residuals must be finite.")
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least two.")
    fixed_rows = np.empty((0, center.size), dtype=float)
    if fixed_jacobian_rows is not None:
        fixed_rows = np.asarray(fixed_jacobian_rows, dtype=float)
        if fixed_rows.ndim == 1:
            fixed_rows = fixed_rows[None, :]
        if fixed_rows.ndim != 2 or fixed_rows.shape[1] != center.size:
            raise ValueError(
                "fixed_jacobian_rows must have one column per parameter."
            )
        if not np.all(np.isfinite(fixed_rows)):
            raise ValueError("fixed_jacobian_rows must be finite.")
    full_jacobian = np.vstack((jacobian, fixed_rows))
    singular_values = np.linalg.svd(full_jacobian, compute_uv=False)
    tolerance = float(rcond) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else np.inf
    )
    pseudoinverse = np.linalg.pinv(full_jacobian, rcond=float(rcond))
    centered_residuals = residuals - np.mean(residuals)
    generator = np.random.default_rng(int(random_seed))
    sampled_data = generator.choice(
        centered_residuals,
        size=(n_bootstrap, centered_residuals.size),
        replace=True,
    )
    if fixed_rows.shape[0]:
        sampled_rhs = np.hstack(
            (sampled_data, np.zeros((n_bootstrap, fixed_rows.shape[0]), dtype=float))
        )
    else:
        sampled_rhs = sampled_data
    deltas = sampled_rhs @ pseudoinverse.T
    return LinearizedBootstrapResult(
        parameter_names=names,
        samples=center[None, :] + deltas,
        deltas=deltas,
        jacobian_rank=rank,
        jacobian_condition=condition,
    )


@dataclass(frozen=True)
class LineFitConstraints:
    """Reusable constraints for random-line least-squares fits.

    ``line_density_anchor`` is ``(reference_density, relative_sigma)`` and
    penalizes the logarithmic deviation of the model's geometrical high-Q line
    density from the reference.  ``line_density_anchor_mode`` may be
    ``"two_sided"`` or ``"below_reference_only"``.  A relative sigma of
    ``0.10`` corresponds approximately to a ten-percent one-sigma tolerance.
    """

    fixed_parameters: Mapping[str, float] | None = None
    parameter_penalties: Mapping[str, tuple[float, float]] | None = None
    soft_lower_bounds: Mapping[str, tuple[float, float]] | None = None
    inverse_square_lower_barriers: Mapping[
        str, tuple[float, float, float, float]
    ] | None = None
    line_density_anchor: tuple[float, float] | None = None
    line_density_anchor_mode: str = "two_sided"

    def with_fixed_parameters(
        self,
        fixed_parameters: Mapping[str, float] | None,
    ) -> "LineFitConstraints":
        """Return a copy with the supplied fit-specific fixed parameters."""

        return replace(
            self,
            fixed_parameters=dict(fixed_parameters or {}),
        )


def line_density_from_fit_parameters(parameters: Mapping[str, float]) -> float:
    """Recover geometrical line density from saved fit parameters.

    New results store ``line_density_geometric`` explicitly.  Older results
    are reconstructed from their fitted high-Q coefficient and scale, which
    gives the same ``rho0 * p_H * affine_highq_factor`` quantity.
    """

    if "line_density_geometric" in parameters:
        density = float(parameters["line_density_geometric"])
    elif "highq_coefficient" in parameters and "scale" in parameters:
        scale = float(parameters["scale"])
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("Saved fit scale must be finite and positive.")
        density = float(parameters["highq_coefficient"]) / (np.pi * scale)
    elif "rho0" in parameters:
        density = float(parameters["rho0"])
    else:
        raise KeyError(
            "Need line_density_geometric, highq_coefficient with scale, or rho0."
        )
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError("Recovered line density must be finite and positive.")
    return density


def estimate_k_distribution_width_svd1(
    q: np.ndarray,
    intensity: np.ndarray,
    k_eff: float,
    *,
    q_range: tuple[float, float] | None = None,
    k_distribution: str = "gamma_radial",
    basis_path: str | Path | None = None,
) -> SvdWidthEstimate:
    """Estimate radial ``r_sigma_k`` using its distribution-specific SVD basis.

    ``q`` and ``k_eff`` must use the same inverse-length units.  The selected
    data are normalized by a fitted Q^-1 tail, interpolated onto the stored
    dimensionless ``Q/k_eff`` grid, and projected onto SVD component 1.  The
    component coefficient is then inverted against the line-scan calibration.
    """

    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if q.ndim != 1 or intensity.ndim != 1 or q.size != intensity.size:
        raise ValueError("q and intensity must be equal-length one-dimensional arrays.")
    k_eff = float(k_eff)
    if not np.isfinite(k_eff) or k_eff <= 0.0:
        raise ValueError("k_eff must be finite and positive.")
    good = np.isfinite(q) & np.isfinite(intensity) & (q > 0.0) & (intensity > 0.0)
    if q_range is not None:
        q_lo, q_hi = map(float, q_range)
        if not 0.0 < q_lo < q_hi:
            raise ValueError("q_range must be an increasing positive pair.")
        good &= (q >= q_lo) & (q <= q_hi)
    q_data = q[good]
    i_data = intensity[good]
    order = np.argsort(q_data)
    q_data, i_data = q_data[order], i_data[order]
    if q_data.size < 12:
        raise ValueError("Need at least 12 positive points in the selected Q range.")
    q_data, unique = np.unique(q_data, return_index=True)
    i_data = i_data[unique]

    k_distribution = str(k_distribution)
    if k_distribution not in {"gamma_radial", "gaussian_radial"}:
        raise ValueError("k_distribution must be gamma_radial or gaussian_radial.")
    if basis_path is None:
        basis_path = (
            GAMMA_SVD1_WIDTH_BASIS_PATH
            if k_distribution == "gamma_radial"
            else GAUSSIAN_SVD1_WIDTH_BASIS_PATH
        )
    basis_path = Path(basis_path)
    if not basis_path.exists():
        raise FileNotFoundError(f"Stored SVD1 width basis not found: {basis_path}")
    with np.load(basis_path) as basis:
        u_grid = np.asarray(basis["u_grid"], dtype=float)
        mean_shape = np.asarray(basis["mean_shape"], dtype=float)
        component = np.asarray(basis["components"], dtype=float)[0]
        sigma_values = np.asarray(basis["sigma_values"], dtype=float)
        polynomial = np.asarray(basis["polynomial_coefficients"], dtype=float)[0]
        sigma_center = float(basis["sigma_center"])
        sigma_scale = float(basis["sigma_scale"])
        tail_u_min = float(basis["inverse_tail_u_min"])
        if (
            "k_distribution" in basis
            and str(basis["k_distribution"].item()) != k_distribution
        ):
            raise ValueError(
                f"The stored SVD width basis is not calibrated for {k_distribution}."
            )
        if "k_scale_convention" in basis and str(basis["k_scale_convention"].item()) != "k_eff":
            raise ValueError("The stored SVD width basis does not use the k_eff convention.")

    u_data = q_data / k_eff
    overlap = (u_grid >= u_data[0]) & (u_grid <= u_data[-1])
    tail = u_data >= tail_u_min
    if np.count_nonzero(overlap) < 12:
        raise ValueError("The selected Q range has fewer than 12 points on the stored SVD grid.")
    if np.count_nonzero(tail) < 4:
        raise ValueError(
            f"Need at least four points with Q/k_eff >= {tail_u_min:g} for tail normalization."
        )
    tail_line = np.polyfit(np.log(u_data[tail]), np.log(i_data[tail]), 1)
    log_shape_data = np.log(i_data) - np.polyval(tail_line, np.log(u_data))
    log_shape = np.interp(np.log(u_grid[overlap]), np.log(u_data), log_shape_data)
    centered = log_shape - mean_shape[overlap]
    weight = component[overlap]
    coefficient = float(np.dot(centered, weight) / np.dot(weight, weight))

    sigma_dense = np.linspace(float(sigma_values.min()), float(sigma_values.max()), 10001)
    coordinate = (sigma_dense - sigma_center) / sigma_scale
    calibrated = np.polyval(polynomial, coordinate)
    sigma_index = int(np.argmin(np.abs(calibrated - coefficient)))
    r_sigma_k = float(sigma_dense[sigma_index])
    reconstructed = mean_shape[overlap] + coefficient * weight
    rmse = float(np.sqrt(np.mean((log_shape - reconstructed) ** 2)))
    return SvdWidthEstimate(
        r_sigma_k=r_sigma_k,
        svd1_coefficient=coefficient,
        k_eff=k_eff,
        q_min=float(q_data[0]),
        q_max=float(q_data[-1]),
        n_points=int(q_data.size),
        reconstruction_rmse=rmse,
        k_distribution=k_distribution,
    )


@dataclass(frozen=True)
class IqxyData:
    label: str
    path: Path
    qx: np.ndarray
    qy: np.ndarray
    q: np.ndarray
    intensity: np.ndarray
    error: np.ndarray


@dataclass(frozen=True)
class RadialProfile:
    label: str
    q: np.ndarray
    intensity: np.ndarray
    error: np.ndarray
    count: np.ndarray
    scale: float = 1.0


@dataclass(frozen=True)
class ResolutionDesmearingResult:
    """Maximum-entropy inversion of a Q-dependent resolution convolution."""

    q: np.ndarray
    smeared_intensity: np.ndarray
    desmeared_intensity: np.ndarray
    error: np.ndarray
    resolution_sigma: np.ndarray
    resmeared_intensity: np.ndarray
    default_model: np.ndarray
    resolution_matrix: np.ndarray
    entropy_weight: float
    reduced_chi_squared: float
    success: bool
    message: str
    nfev: int


@dataclass(frozen=True)
class AnchorFit:
    name: str
    parameters: dict[str, float]
    q_fit_min: float
    q_fit_max: float
    q: np.ndarray
    y: np.ndarray
    y_fit: np.ndarray
    relative_rmse: float


def gaussian_resolution_matrix(
    q: np.ndarray,
    resolution_sigma: np.ndarray,
    *,
    source_q: np.ndarray | None = None,
) -> np.ndarray:
    """Return a row-normalized Gaussian Q-resolution convolution matrix.

    Row ``i`` maps an unsmeared profile tabulated on ``source_q`` to the
    measured intensity at ``q[i]`` using the pointwise standard deviation
    ``resolution_sigma[i]`` and trapezoid-like integration weights. When
    ``source_q`` is omitted, the input and output grids are both ``q``.
    """

    q = np.asarray(q, dtype=float)
    resolution_sigma = np.asarray(resolution_sigma, dtype=float)
    source_q = q if source_q is None else np.asarray(source_q, dtype=float)
    if q.ndim != 1 or resolution_sigma.ndim != 1 or q.size != resolution_sigma.size:
        raise ValueError("q and resolution_sigma must be equal-length one-dimensional arrays.")
    if source_q.ndim != 1 or source_q.size < 2:
        raise ValueError("source_q must contain at least two Q points.")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(source_q)) or not np.all(np.isfinite(resolution_sigma)):
        raise ValueError("q, source_q, and resolution_sigma must be finite.")
    if np.any(np.diff(q) <= 0.0) or np.any(np.diff(source_q) <= 0.0):
        raise ValueError("q and source_q must be strictly increasing.")
    if np.any(q <= 0.0) or np.any(source_q <= 0.0) or np.any(resolution_sigma <= 0.0):
        raise ValueError("q, source_q, and resolution_sigma must be positive.")

    dq_weights = np.empty_like(source_q)
    dq_weights[1:-1] = 0.5 * (source_q[2:] - source_q[:-2])
    dq_weights[0] = 0.5 * (source_q[1] - source_q[0])
    dq_weights[-1] = 0.5 * (source_q[-1] - source_q[-2])
    delta = q[:, None] - source_q[None, :]
    matrix = np.exp(
        -0.5 * (delta / resolution_sigma[:, None]) ** 2
    ) * dq_weights[None, :]
    row_sum = np.sum(matrix, axis=1)
    if np.any(row_sum <= np.finfo(float).tiny):
        raise ValueError("Resolution kernel has an empty row; check resolution_sigma.")
    return matrix / row_sum[:, None]


def desmear_resolution_maximum_entropy(
    q: np.ndarray,
    intensity: np.ndarray,
    error: np.ndarray,
    resolution_sigma: np.ndarray,
    *,
    entropy_weight: float = 10.0,
    prior_smoothing_points: int = 11,
    maxiter: int = 2000,
    intensity_floor: float | None = None,
) -> ResolutionDesmearingResult:
    """Desmear a 1D profile using its pointwise Gaussian Q resolution.

    The forward model discretizes

    ``I_exp(Q_i) = integral I(q) R_i(Q_i-q) dq``

    with a normalized Gaussian ``R_i`` whose standard deviation is supplied
    by ``resolution_sigma[i]``. The positive unsmeared profile is recovered by
    minimizing weighted data misfit plus relative maximum-entropy regularization
    about a gently smoothed positive default model. This implements the
    convolution and maximum-entropy principles of Huang et al., J. Appl.
    Cryst. 58 (2025) 1355-1359, while applying them globally so broad SAS
    profiles do not need to be split into isolated peaks.

    The returned errors remain the experimental errors. Desmearing introduces
    inter-point covariance, so a diagonal error transformation would imply
    unsupported precision; downstream fits should regard these errors as
    conservative weights and inspect the resmeared residual diagnostic.
    """

    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    error = np.asarray(error, dtype=float)
    resolution_sigma = np.asarray(resolution_sigma, dtype=float)
    if not (q.ndim == intensity.ndim == error.ndim == resolution_sigma.ndim == 1):
        raise ValueError("q, intensity, error, and resolution_sigma must be one-dimensional.")
    if not (q.size == intensity.size == error.size == resolution_sigma.size):
        raise ValueError("q, intensity, error, and resolution_sigma must have equal lengths.")
    if q.size < 8:
        raise ValueError("Need at least eight points for resolution desmearing.")
    if not (
        np.all(np.isfinite(q))
        and np.all(np.isfinite(intensity))
        and np.all(np.isfinite(error))
        and np.all(np.isfinite(resolution_sigma))
    ):
        raise ValueError("Desmearing inputs must be finite.")
    if np.any(np.diff(q) <= 0.0):
        raise ValueError("q must be strictly increasing.")
    if np.any(q <= 0.0) or np.any(error <= 0.0) or np.any(resolution_sigma <= 0.0):
        raise ValueError("q, error, and resolution_sigma must be positive.")
    if not np.isfinite(entropy_weight) or entropy_weight < 0.0:
        raise ValueError("entropy_weight must be finite and nonnegative.")

    resolution_matrix = gaussian_resolution_matrix(q, resolution_sigma)
    dq_weights = np.empty_like(q)
    dq_weights[1:-1] = 0.5 * (q[2:] - q[:-2])
    dq_weights[0] = 0.5 * (q[1] - q[0])
    dq_weights[-1] = 0.5 * (q[-1] - q[-2])

    positive = intensity[intensity > 0.0]
    if intensity_floor is None:
        if positive.size:
            intensity_floor = max(
                np.finfo(float).tiny,
                0.05 * float(np.percentile(positive, 10.0)),
            )
        else:
            intensity_floor = max(np.finfo(float).tiny, 0.1 * float(np.median(error)))
    intensity_floor = float(intensity_floor)
    if not np.isfinite(intensity_floor) or intensity_floor <= 0.0:
        raise ValueError("intensity_floor must be finite and positive.")

    clipped = np.maximum(intensity, intensity_floor)
    window = int(prior_smoothing_points)
    if window < 1:
        raise ValueError("prior_smoothing_points must be positive.")
    if window % 2 == 0:
        window += 1
    window = min(window, q.size if q.size % 2 == 1 else q.size - 1)
    pad = window // 2
    if window > 1:
        log_clipped = np.log(clipped)
        padded = np.pad(log_clipped, pad, mode="edge")
        default_model = np.exp(np.convolve(padded, np.ones(window) / window, mode="valid"))
    else:
        default_model = clipped.copy()
    default_model = np.maximum(default_model, intensity_floor)

    entropy_weights = dq_weights / np.sum(dq_weights)
    inverse_variance = 1.0 / (error * error)
    regularization_scale = float(q.size) * float(entropy_weight)

    def objective_and_gradient(log_ratio: np.ndarray) -> tuple[float, np.ndarray]:
        safe_log_ratio = np.clip(log_ratio, -30.0, 30.0)
        ratio = np.exp(safe_log_ratio)
        candidate = default_model * ratio
        residual = resolution_matrix @ candidate - intensity
        chi_term = 0.5 * float(np.dot(residual * inverse_variance, residual))
        entropy_term = float(
            np.dot(entropy_weights, ratio * safe_log_ratio - ratio + 1.0)
        )
        gradient_data = candidate * (
            resolution_matrix.T @ (inverse_variance * residual)
        )
        gradient_entropy = (
            regularization_scale
            * entropy_weights
            * ratio
            * safe_log_ratio
        )
        return chi_term + regularization_scale * entropy_term, gradient_data + gradient_entropy

    optimization = minimize(
        objective_and_gradient,
        np.zeros(q.size, dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-30.0, 30.0)] * q.size,
        options={"maxiter": int(maxiter), "ftol": 1.0e-11, "gtol": 1.0e-7},
    )
    desmeared = default_model * np.exp(np.clip(optimization.x, -30.0, 30.0))
    resmeared = resolution_matrix @ desmeared
    reduced_chi_squared = float(
        np.mean(((resmeared - intensity) / error) ** 2)
    )
    return ResolutionDesmearingResult(
        q=q.copy(),
        smeared_intensity=intensity.copy(),
        desmeared_intensity=desmeared,
        error=error.copy(),
        resolution_sigma=resolution_sigma.copy(),
        resmeared_intensity=resmeared,
        default_model=default_model,
        resolution_matrix=resolution_matrix,
        entropy_weight=float(entropy_weight),
        reduced_chi_squared=reduced_chi_squared,
        success=bool(optimization.success),
        message=str(optimization.message),
        nfev=int(optimization.nfev),
    )


@dataclass(frozen=True)
class HeterogeneousFitResult:
    parameters: dict[str, float]
    free_parameters: np.ndarray
    residual: np.ndarray
    q: np.ndarray
    intensity: np.ndarray
    error: np.ndarray
    model: np.ndarray
    unscaled_model: np.ndarray
    success: bool
    message: str
    cost: float
    nfev: int


def load_iqxy(path: str | Path, *, label: str | None = None) -> IqxyData:
    """Load four-column ``Qx Qy I err`` ASCII detector data."""

    path = Path(path)
    data = np.loadtxt(path, skiprows=2)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"{path} does not look like a four-column Iqxy file.")
    qx = np.asarray(data[:, 0], dtype=float)
    qy = np.asarray(data[:, 1], dtype=float)
    intensity = np.asarray(data[:, 2], dtype=float)
    error = np.asarray(data[:, 3], dtype=float)
    return IqxyData(
        label=label or path.stem,
        path=path,
        qx=qx,
        qy=qy,
        q=np.hypot(qx, qy),
        intensity=intensity,
        error=error,
    )


def radial_average_iqxy(
    data: IqxyData,
    *,
    n_bins: int = 80,
    q_min: float | None = None,
    q_max: float | None = None,
    min_count: int = 6,
) -> RadialProfile:
    """Radially average a 2D Iqxy map using inverse-variance weights."""

    q = np.asarray(data.q, dtype=float)
    intensity = np.asarray(data.intensity, dtype=float)
    error = np.asarray(data.error, dtype=float)
    good = np.isfinite(q) & np.isfinite(intensity) & np.isfinite(error) & (q > 0.0) & (error > 0.0)
    if q_min is not None:
        good &= q >= float(q_min)
    if q_max is not None:
        good &= q <= float(q_max)
    if np.count_nonzero(good) < 2:
        raise ValueError(f"No usable points for {data.label}.")

    q_good = q[good]
    i_good = intensity[good]
    e_good = error[good]
    edges = np.geomspace(float(q_good.min()), float(q_good.max()), int(n_bins) + 1)
    bin_index = np.digitize(q_good, edges) - 1

    q_out: list[float] = []
    i_out: list[float] = []
    e_out: list[float] = []
    c_out: list[int] = []
    for idx in range(int(n_bins)):
        mask = bin_index == idx
        count = int(np.count_nonzero(mask))
        if count < int(min_count):
            continue
        q_bin = q_good[mask]
        i_bin = i_good[mask]
        e_bin = e_good[mask]
        weights = 1.0 / np.maximum(e_bin * e_bin, np.finfo(float).tiny)
        w_sum = float(np.sum(weights))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            continue
        q_out.append(float(np.exp(np.mean(np.log(q_bin)))))
        i_out.append(float(np.sum(weights * i_bin) / w_sum))
        e_out.append(float(np.sqrt(1.0 / w_sum)))
        c_out.append(count)

    return RadialProfile(
        label=data.label,
        q=np.asarray(q_out, dtype=float),
        intensity=np.asarray(i_out, dtype=float),
        error=np.asarray(e_out, dtype=float),
        count=np.asarray(c_out, dtype=int),
    )


def scale_profile_to_reference(
    profile: RadialProfile,
    reference: RadialProfile,
    *,
    q_overlap: tuple[float, float] | None = None,
) -> tuple[RadialProfile, dict[str, float]]:
    """Scale a profile onto a reference curve using their positive overlap."""

    q_lo = max(float(profile.q.min()), float(reference.q.min()))
    q_hi = min(float(profile.q.max()), float(reference.q.max()))
    if q_overlap is not None:
        q_lo = max(q_lo, float(q_overlap[0]))
        q_hi = min(q_hi, float(q_overlap[1]))
    mask = (profile.q >= q_lo) & (profile.q <= q_hi) & (profile.intensity > 0.0)
    ref_positive = reference.intensity > 0.0
    if np.count_nonzero(mask) < 3 or np.count_nonzero(ref_positive) < 3 or q_lo >= q_hi:
        scale = 1.0
        used = 0
    else:
        ref_log_i = np.interp(np.log(profile.q[mask]), np.log(reference.q[ref_positive]), np.log(reference.intensity[ref_positive]))
        ratio = np.exp(ref_log_i) / profile.intensity[mask]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0.0)]
        scale = float(np.median(ratio)) if ratio.size else 1.0
        used = int(ratio.size)
    scaled = RadialProfile(
        label=profile.label,
        q=profile.q.copy(),
        intensity=profile.intensity * scale,
        error=profile.error * abs(scale),
        count=profile.count.copy(),
        scale=scale,
    )
    return scaled, {"scale": scale, "q_overlap_min": q_lo, "q_overlap_max": q_hi, "n_overlap": used}


def stitch_profiles(
    profiles: Mapping[str, RadialProfile],
    *,
    n_bins: int = 180,
    min_count: int = 1,
) -> RadialProfile:
    """Combine scaled radial profiles into one inverse-variance weighted curve."""

    q_all = np.concatenate([profile.q for profile in profiles.values()])
    i_all = np.concatenate([profile.intensity for profile in profiles.values()])
    e_all = np.concatenate([profile.error for profile in profiles.values()])
    good = np.isfinite(q_all) & np.isfinite(i_all) & np.isfinite(e_all) & (q_all > 0.0) & (e_all > 0.0)
    q_all = q_all[good]
    i_all = i_all[good]
    e_all = e_all[good]
    edges = np.geomspace(float(q_all.min()), float(q_all.max()), int(n_bins) + 1)
    bin_index = np.digitize(q_all, edges) - 1

    q_out: list[float] = []
    i_out: list[float] = []
    e_out: list[float] = []
    c_out: list[int] = []
    for idx in range(int(n_bins)):
        mask = bin_index == idx
        count = int(np.count_nonzero(mask))
        if count < int(min_count):
            continue
        weights = 1.0 / np.maximum(e_all[mask] * e_all[mask], np.finfo(float).tiny)
        w_sum = float(np.sum(weights))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            continue
        q_out.append(float(np.exp(np.mean(np.log(q_all[mask])))))
        i_out.append(float(np.sum(weights * i_all[mask]) / w_sum))
        e_out.append(float(np.sqrt(1.0 / w_sum)))
        c_out.append(count)

    return RadialProfile(
        label="stitched",
        q=np.asarray(q_out, dtype=float),
        intensity=np.asarray(i_out, dtype=float),
        error=np.asarray(e_out, dtype=float),
        count=np.asarray(c_out, dtype=int),
    )


def subtract_constant_background(
    profile: RadialProfile,
    background: float,
) -> RadialProfile:
    """Return a profile with a fitted constant baseline removed."""

    background = float(background)
    if not np.isfinite(background):
        raise ValueError("background must be finite.")
    return RadialProfile(
        label=f"{profile.label}_bg_subtracted",
        q=profile.q.copy(),
        intensity=profile.intensity - background,
        error=profile.error.copy(),
        count=profile.count.copy(),
        scale=profile.scale,
    )


def profile_table(profile: RadialProfile) -> np.ndarray:
    """Return a simple machine-readable ``Q, I, err, count`` table."""

    return np.column_stack([profile.q, profile.intensity, profile.error, profile.count.astype(float)])


def _fit_mask(profile: RadialProfile, q_bounds: tuple[float, float]) -> np.ndarray:
    q_min, q_max = map(float, q_bounds)
    return (
        np.isfinite(profile.q)
        & np.isfinite(profile.intensity)
        & np.isfinite(profile.error)
        & (profile.q >= q_min)
        & (profile.q <= q_max)
        & (profile.q > 0.0)
        & (profile.intensity > 0.0)
        & (profile.error > 0.0)
    )


def _weighted_linear_fit(design: np.ndarray, y: np.ndarray, err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = 1.0 / np.maximum(err, np.finfo(float).tiny)
    aw = design * weights[:, None]
    yw = y * weights
    coeff, *_ = np.linalg.lstsq(aw, yw, rcond=None)
    return coeff, design @ coeff


def _background_upper_bound(y: np.ndarray, background_max: float | None) -> float:
    if background_max is not None:
        return float(background_max)
    return float(np.min(y[y > 0.0]))


def _fit_positive_basis_with_bounded_background(
    basis: np.ndarray,
    y: np.ndarray,
    err: np.ndarray,
    *,
    background_min: float = 0.0,
    background_max: float | None = None,
) -> tuple[float, float, np.ndarray]:
    """Fit ``amplitude*basis + background`` with a small nonnegative background."""

    basis = np.asarray(basis, dtype=float)
    y = np.asarray(y, dtype=float)
    err = np.asarray(err, dtype=float)
    b_min = float(background_min)
    b_max = _background_upper_bound(y, background_max)
    if b_max < b_min:
        b_max = b_min

    design = np.column_stack([basis, np.ones_like(basis)])
    unconstrained, _ = _weighted_linear_fit(design, y, err)
    candidates = [b_min, b_max]
    if unconstrained[0] >= 0.0 and b_min <= unconstrained[1] <= b_max:
        candidates.append(float(unconstrained[1]))

    weights = 1.0 / np.maximum(err * err, np.finfo(float).tiny)
    denom = float(np.sum(weights * basis * basis))
    best: tuple[float, float, float, np.ndarray] | None = None
    for background in candidates:
        if denom <= 0.0:
            amplitude = 0.0
        else:
            amplitude = float(np.sum(weights * basis * (y - background)) / denom)
        amplitude = max(0.0, amplitude)
        y_fit = amplitude * basis + background
        score = float(np.sum(weights * (y_fit - y) ** 2))
        if best is None or score < best[0]:
            best = (score, amplitude, float(background), y_fit)
    assert best is not None
    return best[1], best[2], best[3]


def _relative_rmse(y: np.ndarray, y_fit: np.ndarray) -> float:
    denom = np.maximum(np.abs(y), np.finfo(float).tiny)
    return float(np.sqrt(np.mean(((y_fit - y) / denom) ** 2)))


def fit_lowq_dab_anchor(
    profile: RadialProfile,
    q_bounds: tuple[float, float],
    *,
    n_kappa: int = 160,
    background_max: float | None = None,
) -> AnchorFit:
    """Fit ``I(Q) ~= C/(Q^2+kappa^2)^2 + B`` over a low-Q window.

    By default the background is constrained to ``0 <= B <= min(I_obs)`` in
    the fitting window, matching the expectation that an incoherent background
    should be small compared with the observed intensity floor.
    """

    mask = _fit_mask(profile, q_bounds)
    q = profile.q[mask]
    y = profile.intensity[mask]
    err = profile.error[mask]
    if q.size < 4:
        raise ValueError("Need at least four positive points for the DAB anchor fit.")
    q_min, q_max = float(np.min(q)), float(np.max(q))
    kappa_grid = np.geomspace(max(q_min / 20.0, np.finfo(float).tiny), q_max * 20.0, int(n_kappa))
    best: tuple[float, float, float, np.ndarray] | None = None
    for kappa in kappa_grid:
        basis = 1.0 / (q * q + kappa * kappa) ** 2
        amplitude, background, y_fit = _fit_positive_basis_with_bounded_background(
            basis,
            y,
            err,
            background_max=background_max,
        )
        if amplitude <= 0.0:
            continue
        score = _relative_rmse(y, y_fit)
        if best is None or score < best[0]:
            best = (score, float(kappa), float(amplitude), float(background), y_fit)
    if best is None:
        raise ValueError("DAB anchor fit did not find a positive-amplitude solution.")
    score, kappa, amplitude, background, y_fit = best
    return AnchorFit(
        name="lowQ_DAB",
        parameters={"amplitude": amplitude, "kappa": kappa, "background": background},
        q_fit_min=float(q_bounds[0]),
        q_fit_max=float(q_bounds[1]),
        q=q,
        y=y,
        y_fit=y_fit,
        relative_rmse=_relative_rmse(y, y_fit),
    )


def fit_lowq_porod_anchor(
    profile: RadialProfile,
    q_bounds: tuple[float, float],
    *,
    background_max: float | None = None,
) -> AnchorFit:
    """Fit ``I(Q) ~= C/Q^4 + B`` over a low-Q window."""

    mask = _fit_mask(profile, q_bounds)
    q = profile.q[mask]
    y = profile.intensity[mask]
    err = profile.error[mask]
    if q.size < 3:
        raise ValueError("Need at least three positive points for the Porod anchor fit.")
    coeff0, background, y_fit = _fit_positive_basis_with_bounded_background(
        q ** -4,
        y,
        err,
        background_max=background_max,
    )
    return AnchorFit(
        name="lowQ_Qminus4",
        parameters={"coefficient": float(coeff0), "background": float(background)},
        q_fit_min=float(q_bounds[0]),
        q_fit_max=float(q_bounds[1]),
        q=q,
        y=y,
        y_fit=y_fit,
        relative_rmse=_relative_rmse(y, y_fit),
    )


def fit_highq_line_anchor(
    profile: RadialProfile,
    q_bounds: tuple[float, float],
    *,
    background_max: float | None = None,
    allow_signed_background: bool = False,
) -> AnchorFit:
    """Fit ``I(Q) ~= C/Q + B`` and report the apparent ``rho0=C/pi``."""

    mask = _fit_mask(profile, q_bounds)
    q = profile.q[mask]
    y = profile.intensity[mask]
    err = profile.error[mask]
    if q.size < 3:
        raise ValueError("Need at least three positive points for the line anchor fit.")
    if allow_signed_background:
        coefficients, y_fit = _weighted_linear_fit(
            np.column_stack((1.0 / q, np.ones_like(q))),
            y,
            err,
        )
        coefficient, background = map(float, coefficients)
        if coefficient <= 0.0:
            raise ValueError("Signed-background high-Q fit did not find a positive Q^-1 coefficient.")
    else:
        coefficient, background, y_fit = _fit_positive_basis_with_bounded_background(
            1.0 / q,
            y,
            err,
            background_max=background_max,
        )
    return AnchorFit(
        name="highQ_line",
        parameters={
            "coefficient": coefficient,
            "background": float(background),
            "rho0_apparent": coefficient / np.pi,
        },
        q_fit_min=float(q_bounds[0]),
        q_fit_max=float(q_bounds[1]),
        q=q,
        y=y,
        y_fit=y_fit,
        relative_rmse=_relative_rmse(y, y_fit),
    )


def evaluate_anchor_fit(fit: AnchorFit, q: np.ndarray) -> np.ndarray:
    """Evaluate an anchor fit on an arbitrary Q grid."""

    q = np.asarray(q, dtype=float)
    if fit.name == "lowQ_DAB":
        return fit.parameters["amplitude"] / (q * q + fit.parameters["kappa"] ** 2) ** 2 + fit.parameters["background"]
    if fit.name == "lowQ_Qminus4":
        return fit.parameters["coefficient"] * q ** -4 + fit.parameters["background"]
    if fit.name == "highQ_line":
        return fit.parameters["coefficient"] / q + fit.parameters["background"]
    raise ValueError(f"Unknown anchor fit type: {fit.name}")


def estimate_peak_position(
    profile: RadialProfile,
    q_bounds: tuple[float, float],
    *,
    local_points: int = 7,
) -> dict[str, float]:
    """Estimate a local smooth peak using a weighted quadratic in log-Q/log-I."""

    mask = _fit_mask(profile, q_bounds)
    q = profile.q[mask]
    y = profile.intensity[mask]
    err = profile.error[mask]
    if q.size < 3:
        raise ValueError("Need at least three positive points for the polynomial peak estimate.")

    local_points = min(int(local_points), q.size)
    if local_points < 3:
        raise ValueError("local_points must be at least three.")
    peak_index = int(np.argmax(y))
    start = max(0, min(peak_index - local_points // 2, q.size - local_points))
    local = slice(start, start + local_points)
    q = q[local]
    y = y[local]
    err = err[local]

    log_q = np.log(q)
    log_y = np.log(y)
    center = float(log_q[np.argmax(y)])
    relative_error = np.maximum(err / y, np.finfo(float).eps)
    coefficients = np.polyfit(
        log_q - center,
        log_y,
        deg=2,
        w=1.0 / relative_error,
    )
    curvature, slope, intercept = map(float, coefficients)
    if curvature >= 0.0:
        raise ValueError("Quadratic peak fit is not concave within the selected Q window.")

    offset_peak = -slope / (2.0 * curvature)
    log_q_peak = center + offset_peak
    if not float(np.min(log_q)) <= log_q_peak <= float(np.max(log_q)):
        raise ValueError("Quadratic peak lies outside the selected Q window.")
    log_i_peak = np.polyval(coefficients, offset_peak)
    return {
        "q_peak": float(np.exp(log_q_peak)),
        "i_peak": float(np.exp(log_i_peak)),
    }


def _normal_sf(x: float) -> float:
    return 0.5 * float(erfc(float(x) / np.sqrt(2.0)))


def _normal_pdf(x: float) -> float:
    return float(np.exp(-0.5 * float(x) ** 2) / np.sqrt(2.0 * np.pi))


def _kappa_over_kh_from_b(b: float) -> float:
    p_h = _normal_sf(b)
    sigma2 = p_h * (1.0 - p_h)
    if sigma2 <= np.finfo(float).tiny:
        return float("inf")
    return _normal_pdf(b) / (np.sqrt(6.0 * np.pi) * sigma2)


def initial_heterogeneous_guess_from_anchors(
    *,
    q_peak: float,
    lowq_kappa: float,
    highq_coefficient: float,
    b0: float = -1.0,
    r_sigma_k0: float = 0.2,
    distribution_parameter_set: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, float]:
    """Build a physically coupled first guess from the anchor fits."""

    mean_k = float(q_peak)
    parameter_set = dict(distribution_parameter_set or {})
    r_sigma_k = float(parameter_set.get("r_sigma_k", {}).get("initial", r_sigma_k0))
    b = float(b0)
    factor = _kappa_over_kh_from_b(b)
    k_h_over_k = float(lowq_kappa) / max(mean_k * factor, np.finfo(float).tiny)
    k_h_over_k = float(np.clip(k_h_over_k, 0.005, 0.5))
    p_h = _normal_sf(b)
    mu2 = mean_k * mean_k * (1.0 + r_sigma_k * r_sigma_k)
    rho0 = mu2 / (3.0 * np.pi)
    scale = float(highq_coefficient) / max(np.pi * p_h * rho0, np.finfo(float).tiny)
    initial = {
        "scale": max(scale, np.finfo(float).tiny),
        "mean_k": k_scale_metadata["mean_k"],
        "r_sigma_k": r_sigma_k,
        "k_H_over_k": k_h_over_k,
        "b": b,
    }
    for name, specification in parameter_set.items():
        initial[name] = float(specification["initial"])
    return initial


def _distribution_parameter_values(
    parameters: Mapping[str, float],
    distribution_parameter_set: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, float]:
    if distribution_parameter_set is None:
        return {"r_sigma_k": float(parameters["r_sigma_k"])}
    return {
        str(name): float(parameters[name])
        for name in distribution_parameter_set
    }


def _import_line_scattering():
    try:
        import rw_line_scattering as rls  # type: ignore

        return rls
    except ModuleNotFoundError:
        smpl_dir = Path(__file__).resolve().parents[1]
        if str(smpl_dir) not in sys.path:
            sys.path.insert(0, str(smpl_dir))
        import rw_line_scattering as rls  # type: ignore

        return rls


def radial_k_distribution_density(
    *,
    mean_k: float,
    r_sigma_k: float,
    skewness: float = 0.0,
    k_distribution: str = "max_entropy_radial",
    distribution_parameters: Mapping[str, float] | None = None,
    num_points: int = 600,
    num_nodes: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a radial wave-number probability density for plotting.

    The returned display grid always spans ``0 <= k <= 2*mean_k``. Density is
    reconstructed from deterministic distribution quantiles, so this helper
    follows the same positive radial distribution used by the scattering
    calculation for both ``max_entropy_radial`` and ``skew_normal_radial``.
    """

    mean_k = float(mean_k)
    r_sigma_k = float(r_sigma_k)
    num_points = int(num_points)
    num_nodes = int(num_nodes)
    if not np.isfinite(mean_k) or mean_k <= 0.0:
        raise ValueError("mean_k must be finite and positive.")
    if not np.isfinite(r_sigma_k) or r_sigma_k <= 0.0:
        raise ValueError("r_sigma_k must be finite and positive.")
    if num_points < 2 or num_nodes < 64:
        raise ValueError("num_points must be at least 2 and num_nodes at least 64.")

    rls = _import_line_scattering()
    parameters = {
        "r_sigma_k": r_sigma_k,
        "skewness": float(skewness),
    }
    parameters.update(
        {str(name): float(value) for name, value in dict(distribution_parameters or {}).items()}
    )
    k_grid = np.linspace(0.0, 2.0 * mean_k, num_points)
    if str(k_distribution) == "bimodal_gaussian_radial":
        required = ("center_distance_ratio", "width_ratio", "weight_ratio")
        missing = tuple(name for name in required if name not in parameters)
        if missing:
            raise ValueError(f"Missing bimodal distribution parameter(s): {missing}.")
        density = rls.bimodal_gaussian_radial_density(
            k_grid,
            mean_k=mean_k,
            r_sigma_k=r_sigma_k,
            center_distance_ratio=parameters["center_distance_ratio"],
            width_ratio=parameters["width_ratio"],
            weight_ratio=parameters["weight_ratio"],
        )
        return k_grid, density
    if str(k_distribution) == "spline_maxent_radial":
        density = rls.spline_maxent_radial_density(
            k_grid,
            mean_k=mean_k,
            r_sigma_k=r_sigma_k,
            distribution_params=parameters,
            num_nodes=num_nodes,
        )
        return k_grid, density
    nodes, weights = rls.make_radial_k_quadrature(
        num_nodes,
        str(k_distribution),
        k0=mean_k,
        sigma_k=r_sigma_k * mean_k,
        distribution_params=parameters,
    )
    nodes = np.asarray(nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(nodes)
    nodes = nodes[order]
    weights = weights[order]
    node_width = np.gradient(nodes)
    density_nodes = weights / np.maximum(node_width, np.finfo(float).tiny)
    density = np.interp(k_grid, nodes, density_nodes, left=0.0, right=0.0)
    return k_grid, density


def gamma_radial_parameters_from_k_eff(
    k_eff: float,
    r_sigma_k: float,
) -> dict[str, float]:
    """Convert ``(k_eff, sigma_k / mean_k)`` to gamma parameters."""

    k_eff = float(k_eff)
    r_sigma_k = float(r_sigma_k)
    if not np.isfinite(k_eff) or k_eff <= 0.0:
        raise ValueError("k_eff must be finite and positive.")
    if not np.isfinite(r_sigma_k) or r_sigma_k <= 0.0:
        raise ValueError("r_sigma_k must be finite and positive.")
    mean_k = k_eff / np.sqrt(1.0 + r_sigma_k * r_sigma_k)
    sigma_k = r_sigma_k * mean_k
    return {
        "k_eff": k_eff,
        "mean_k": float(mean_k),
        "sigma_k": float(sigma_k),
        "gamma_shape": float(1.0 / (r_sigma_k * r_sigma_k)),
        "gamma_scale": float(mean_k * r_sigma_k * r_sigma_k),
    }


def gamma_radial_density_from_k_eff(
    k_eff: float,
    r_sigma_k: float,
    *,
    num_points: int = 600,
    upper_quantile: float = 0.9995,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the positive gamma density parameterized by ``k_eff``."""

    parameters = gamma_radial_parameters_from_k_eff(k_eff, r_sigma_k)
    upper = float(gamma_dist.ppf(
        float(upper_quantile),
        a=parameters["gamma_shape"],
        scale=parameters["gamma_scale"],
    ))
    k_grid = np.linspace(0.0, upper, int(num_points))
    density = gamma_dist.pdf(
        k_grid,
        a=parameters["gamma_shape"],
        scale=parameters["gamma_scale"],
    )
    return k_grid, np.asarray(density, dtype=float)


def gaussian_radial_parameters_from_k_eff(
    k_eff: float,
    r_sigma_k: float,
) -> dict[str, float]:
    """Convert ``k_eff`` to parameters of a Gaussian truncated at ``k=0``.

    ``r_sigma_k`` retains the Gaussian-model definition ``sigma/location``.
    Returned ``mean_k`` and ``sigma_k`` are the moments after truncation.
    """

    k_eff = float(k_eff)
    r_sigma_k = float(r_sigma_k)
    if not np.isfinite(k_eff) or k_eff <= 0.0:
        raise ValueError("k_eff must be finite and positive.")
    if not np.isfinite(r_sigma_k) or r_sigma_k <= 0.0:
        raise ValueError("r_sigma_k must be finite and positive.")
    inverse_width = 1.0 / r_sigma_k
    mills_ratio = float(norm.pdf(inverse_width) / norm.cdf(inverse_width))
    unit_second_moment = (
        1.0 + r_sigma_k * r_sigma_k + r_sigma_k * mills_ratio
    )
    location = k_eff / np.sqrt(unit_second_moment)
    nominal_sigma = r_sigma_k * location
    mean_k = location * (1.0 + r_sigma_k * mills_ratio)
    variance = nominal_sigma**2 * (
        1.0 - inverse_width * mills_ratio - mills_ratio * mills_ratio
    )
    return {
        "k_eff": k_eff,
        "k0_nominal": float(location),
        "gaussian_location": float(location),
        "gaussian_nominal_sigma": float(nominal_sigma),
        "mean_k": float(mean_k),
        "sigma_k": float(np.sqrt(max(variance, 0.0))),
    }


def gaussian_radial_density_from_k_eff(
    k_eff: float,
    r_sigma_k: float,
    *,
    num_points: int = 600,
    upper_quantile: float = 0.9995,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the positive truncated-Gaussian density through a tail quantile."""

    num_points = int(num_points)
    upper_quantile = float(upper_quantile)
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")
    if not 0.0 < upper_quantile < 1.0:
        raise ValueError("upper_quantile must satisfy 0 < value < 1.")
    parameters = gaussian_radial_parameters_from_k_eff(k_eff, r_sigma_k)
    location = float(parameters["gaussian_location"])
    sigma = float(parameters["gaussian_nominal_sigma"])
    lower_cdf = float(norm.cdf(-location / sigma))
    target_cdf = lower_cdf + upper_quantile * (1.0 - lower_cdf)
    upper = location + sigma * float(norm.ppf(target_cdf))
    k_grid = np.linspace(0.0, upper, num_points)
    density = norm.pdf((k_grid - location) / sigma) / (
        sigma * (1.0 - lower_cdf)
    )
    return k_grid, np.asarray(density, dtype=float)


def _resolve_line_k_scale(
    parameters: Mapping[str, float],
    model_settings: Mapping[str, float | int | str | bool | None],
) -> tuple[str, float, float, dict[str, float]]:
    """Resolve the fitted k scale and the distribution mean used by the line model."""

    convention = str(model_settings.get("k_scale_convention", "mean")).lower()
    distribution = str(model_settings.get("k_distribution", "gaussian_radial"))
    r_sigma_k = float(parameters["r_sigma_k"])
    if convention == "mean":
        mean_k = float(parameters["mean_k"])
        metadata = {
            "mean_k": mean_k,
            "sigma_k": r_sigma_k * mean_k,
            "k_eff": mean_k * np.sqrt(1.0 + r_sigma_k * r_sigma_k),
        }
        return "mean_k", mean_k, mean_k, metadata
    if convention != "k_eff":
        raise ValueError("k_scale_convention must be 'mean' or 'k_eff'.")
    k_eff = float(parameters["k_eff"])
    if distribution == "gamma_radial":
        metadata = gamma_radial_parameters_from_k_eff(k_eff, r_sigma_k)
        k0_nominal = metadata["mean_k"]
    elif distribution == "gaussian_radial":
        metadata = gaussian_radial_parameters_from_k_eff(k_eff, r_sigma_k)
        k0_nominal = metadata["k0_nominal"]
    else:
        raise ValueError(
            "k_scale_convention='k_eff' is defined for gamma_radial and gaussian_radial."
        )
    return "k_eff", k_eff, float(k0_nominal), metadata


def compute_fit_orientation_correlations(
    fit_parameters: Mapping[str, float],
    *,
    x_grid: np.ndarray,
    num_k_modes: int = 2**13,
    n_samp: int = 2**14,
    random_seed: int = 12345,
    progress: bool = False,
    k_distribution: str = "max_entropy_radial",
    k_distribution_params: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray | float]:
    """Compute signed and nematic orientation correlations for one saved fit.

    ``k_distribution`` selects the radial spectrum used by the saved fit;
    legacy callers retain ``max_entropy_radial`` as the default. ``x_grid`` is the common
    dimensionless separation ``r*k_eff``.  The nematic calculation supplies
    the conditional Jacobian moment used to normalize the exact signed
    correlation, avoiding a duplicate direct-12D calculation.
    """

    x_grid = np.asarray(x_grid, dtype=float)
    if x_grid.ndim != 1 or x_grid.size == 0 or np.any(x_grid <= 0.0):
        raise ValueError("x_grid must be a nonempty one-dimensional array of positive values.")
    num_k_modes = int(num_k_modes)
    n_samp = int(n_samp)
    if num_k_modes <= 0 or n_samp <= 0:
        raise ValueError("num_k_modes and n_samp must be positive.")

    rls = _import_line_scattering()
    mean_k = float(fit_parameters["mean_k"])
    r_sigma_k = float(fit_parameters["r_sigma_k"])
    distribution_params = dict(k_distribution_params or {})
    if str(k_distribution) in {"max_entropy_radial", "skew_normal_radial"}:
        distribution_params.setdefault("skewness", float(fit_parameters.get("skewness", 0.0)))
    k_rng = np.random.default_rng(int(random_seed))
    k_sets = rls.make_field_k_sets(
        num_k_modes,
        str(k_distribution),
        k_rng,
        k0=mean_k,
        r_sigma_k=r_sigma_k,
        distribution_params=distribution_params,
        shared_k_vectors=True,
        use_qmc_k=True,
        qmc_seed=int(random_seed),
    )
    k_radii = rls.k_radii_from_vectors(k_sets.psi1)
    k_eff = float(
        np.sqrt(3.0 * rls.gradient_variance_from_k_radii(k_radii))
    )
    r_grid = x_grid / k_eff

    nematic = rls.compute_nematic_tangent_correlation(
        r_grid,
        k_radii,
        n_samp,
        use_qmc=True,
        random_seed=int(random_seed),
        progress=bool(progress),
    )
    signed = rls.compute_signed_tangent_correlation(
        r_grid,
        k_radii,
        nematic["M_J"],
    )
    return {
        "r_grid": r_grid,
        "r_k_eff": x_grid.copy(),
        "k_eff": k_eff,
        "M_J": nematic["M_J"],
        "M_T": signed["M_T"],
        "M_2": nematic["M_2"],
        "K_T_raw": signed["K_T_raw"],
        "K_2": nematic["K_2"],
        "K_2_sampled": nematic["K_2_sampled"],
        "K_2_inf_sampled": float(nematic["K_2_inf_sampled"]),
    }


def uniaxial_affine_powder_average(
    q: np.ndarray,
    model_q: np.ndarray,
    model_intensity: np.ndarray,
    stretch_ratio: float,
    *,
    n_mu: int = 64,
) -> np.ndarray:
    """Numerically powder-average an incompressibly stretched isotropic curve.

    The real-space principal stretches are ``(lambda**-1/2,
    lambda**-1/2, lambda)``.  For each observation magnitude ``q``, the
    isotropic intensity is sampled at ``q*sqrt(lambda_perp**2 +
    (lambda_parallel**2-lambda_perp**2)*mu**2)`` and integrated over
    ``mu=cos(theta)`` by Gauss--Legendre quadrature.
    """

    q = np.asarray(q, dtype=float)
    model_q = np.asarray(model_q, dtype=float)
    model_intensity = np.asarray(model_intensity, dtype=float)
    stretch_ratio = float(stretch_ratio)
    n_mu = int(n_mu)
    if stretch_ratio <= 0.0 or not np.isfinite(stretch_ratio):
        raise ValueError("stretch_ratio must be finite and positive.")
    if n_mu < 8:
        raise ValueError("n_mu must be at least 8 for powder averaging.")
    if model_q.ndim != 1 or model_intensity.shape != model_q.shape:
        raise ValueError("model_q and model_intensity must be equal-length one-dimensional arrays.")
    if np.any(model_q <= 0.0) or np.any(np.diff(model_q) <= 0.0):
        raise ValueError("model_q must be strictly positive and increasing.")
    mu, weights = np.polynomial.legendre.leggauss(n_mu)
    lambda_parallel = stretch_ratio
    lambda_perp = stretch_ratio ** -0.5
    directional_scale = np.sqrt(
        lambda_perp**2
        + (lambda_parallel**2 - lambda_perp**2) * mu**2
    )
    q_directional = q[:, None] * directional_scale[None, :]
    if np.min(q_directional) < model_q[0] or np.max(q_directional) > model_q[-1]:
        raise ValueError("The isotropic model Q range does not cover all stretched powder arguments.")
    directional_intensity = np.interp(
        np.log(q_directional),
        np.log(model_q),
        model_intensity,
    )
    return 0.5 * np.sum(directional_intensity * weights[None, :], axis=1)


def uniaxial_affine_highq_factor(stretch_ratio: float, *, n_mu: int = 128) -> float:
    """Return ``<1/s(mu)>`` for an incompressible uniaxial stretch."""

    stretch_ratio = float(stretch_ratio)
    if stretch_ratio <= 0.0 or not np.isfinite(stretch_ratio):
        raise ValueError("stretch_ratio must be finite and positive.")
    mu, weights = np.polynomial.legendre.leggauss(int(n_mu))
    lambda_perp = stretch_ratio ** -0.5
    scale = np.sqrt(
        lambda_perp**2
        + (stretch_ratio**2 - lambda_perp**2) * mu**2
    )
    return float(0.5 * np.sum(weights / scale))


def _affine_powder_curve(
    q: np.ndarray,
    line: object,
    model_q: np.ndarray,
    model_intensity: np.ndarray,
    stretch_ratio: float,
    *,
    model_mode: str,
    model_settings: Mapping[str, float | int | str | bool | None],
    rls: object,
    k_h: float = 0.0,
    b: float = -np.inf,
) -> tuple[np.ndarray, float, object | None]:
    """Evaluate the selected incompressible affine powder model.

    For an independent heterogeneous mask, the physical-line spectrum is
    transformed first and the existing mask operator is then applied to the
    transformed spectrum and its geometrical line density.
    """

    affine_model = str(model_settings.get("affine_model", "scalar")).lower()
    if affine_model == "scalar":
        curve = uniaxial_affine_powder_average(
            q,
            model_q,
            model_intensity,
            stretch_ratio,
            n_mu=int(model_settings.get("powder_n_mu", 64)),
        )
        return (
            curve,
            uniaxial_affine_highq_factor(stretch_ratio),
            None,
        )
    if affine_model != "geometric_second_order":
        raise ValueError(
            "affine_model must be 'scalar' or 'geometric_second_order'."
        )
    geometric = rls.compute_incompressible_uniaxial_scattering(
        stretch_ratio=stretch_ratio,
        line_result=line,
        N_samp_tangent=int(
            model_settings.get(
                "N_samp_tangent",
                model_settings.get("N_samp_U", 2**12),
            )
        ),
        tangent_use_qmc=bool(model_settings.get("tangent_use_qmc", True)),
        tangent_random_seed=int(
            model_settings.get("tangent_random_seed", 12345)
        ),
        n_mu=int(model_settings.get("powder_n_mu", 64)),
        derivative_smoothing_fraction=float(
            model_settings.get("derivative_smoothing_fraction", 1.0e-4)
        ),
        tail_start_fraction=float(
            model_settings.get("tail_start_fraction", 0.8)
        ),
        tangent_tail_start_factor=(
            None
            if model_settings.get("tangent_tail_start_factor") is None
            else float(model_settings["tangent_tail_start_factor"])
        ),
        tangent_tail_end_factor=(
            None
            if model_settings.get("tangent_tail_end_factor") is None
            else float(model_settings["tangent_tail_end_factor"])
        ),
        progress=bool(model_settings.get("progress", False)),
    )
    if np.min(q) < geometric.Q_grid[0] or np.max(q) > geometric.Q_grid[-1]:
        raise ValueError(
            "The geometric powder Q range does not cover all observations."
        )
    tail = geometric.Q_grid / geometric.k_eff >= 5.0
    if np.count_nonzero(tail) < 6:
        tail = np.zeros_like(geometric.Q_grid, dtype=bool)
        tail[-min(12, tail.size):] = True
    positive = tail & np.isfinite(geometric.I_geom_second_order)
    positive &= geometric.I_geom_second_order > 0.0
    if np.count_nonzero(positive) < 3:
        raise ValueError("Could not determine the geometric high-Q coefficient.")
    highq_coefficient = float(
        np.median(
            geometric.Q_grid[positive]
            * geometric.I_geom_second_order[positive]
        )
    )
    highq_factor = highq_coefficient / (
        np.pi * float(geometric.rho0)
    )
    geometric_intensity = geometric.I_geom_second_order
    if model_mode == "heterogeneous":
        stretched_line = rls.make_line_scattering_spectrum(
            geometric.Q_grid,
            geometric_intensity,
            rho0=float(geometric.rho0) * highq_factor,
        )
        heterogeneous = rls.heterogeneous_line_scattering(
            stretched_line,
            k_H=float(k_h),
            b=float(b),
            return_components=True,
        )
        geometric_intensity = heterogeneous.I_h
    elif model_mode != "line_only":
        raise ValueError("model_mode must be 'heterogeneous' or 'line_only'.")
    curve = np.interp(
        np.log(q),
        np.log(geometric.Q_grid),
        geometric_intensity,
    )
    return curve, highq_factor, geometric


def _can_share_affine_tangent_samples(
    affine_stretch: bool,
    model_settings: Mapping[str, float | int | str | bool | None],
) -> bool:
    """Whether one direct-12D pass can supply density and tangent moments."""

    n_density = int(model_settings.get("N_samp_U", 2**12))
    return bool(
        affine_stretch
        and str(model_settings.get("affine_model", "scalar")).lower()
        == "geometric_second_order"
        and str(model_settings.get("jacobian_method", "direct_12d")).lower()
        == "direct_12d"
        and int(model_settings.get("N_samp_tangent", n_density)) == n_density
        and bool(model_settings.get("tangent_use_qmc", True))
        == bool(model_settings.get("use_qmc", True))
        and int(model_settings.get("tangent_random_seed", 12345))
        == int(model_settings.get("random_seed", 12345))
    )


def evaluate_heterogeneous_line_guess(
    observation: RadialProfile,
    *,
    q_bounds: tuple[float, float],
    parameters: Mapping[str, float],
    model_mode: str = "heterogeneous",
    resolution_sigma: np.ndarray | None = None,
    regression_weights: np.ndarray | None = None,
    model_settings: Mapping[str, float | int | str | bool | None] | None = None,
    log_error_floor: float = 0.04,
    regression_loss: str | None = None,
    distribution_parameter_set: Mapping[str, Mapping[str, object]] | None = None,
    affine_stretch: bool = False,
) -> HeterogeneousFitResult:
    """Evaluate one line-model trial and solve only the scale factor.

    ``model_mode="heterogeneous"`` applies the independent binary mask.
    ``model_mode="line_only"`` fits the uniform random-line spectrum directly.
    When ``resolution_sigma`` is supplied, the trial curve is Gaussian-smeared
    on the full available observation Q grid before it is restricted to
    ``q_bounds`` for scale and residual evaluation. ``regression_weights`` may
    be used to emphasize selected observation points without discarding the rest.
    """

    rls = _import_line_scattering()
    model_mode = str(model_mode).lower()
    if model_mode not in {"heterogeneous", "line_only"}:
        raise ValueError("model_mode must be 'heterogeneous' or 'line_only'.")
    model_settings = dict(model_settings or {})
    q_min, q_max = map(float, q_bounds)
    mask = (
        np.isfinite(observation.q)
        & np.isfinite(observation.intensity)
        & np.isfinite(observation.error)
        & (observation.q >= q_min)
        & (observation.q <= q_max)
        & (observation.q > 0.0)
        & (observation.intensity > 0.0)
        & (observation.error > 0.0)
    )
    q_obs = observation.q[mask]
    i_obs = observation.intensity[mask]
    e_obs = observation.error[mask]
    if q_obs.size < 8:
        raise ValueError("Need at least eight positive observation points for model evaluation.")
    if regression_weights is None:
        point_weights = np.ones_like(q_obs)
    else:
        regression_weights = np.asarray(regression_weights, dtype=float)
        if regression_weights.shape != observation.q.shape:
            raise ValueError("regression_weights must have the same shape as observation.q.")
        point_weights = regression_weights[mask]
        if not np.all(np.isfinite(point_weights) & (point_weights > 0.0)):
            raise ValueError("regression_weights must be finite and strictly positive.")
    resolution_matrix = None
    q_model_eval = q_obs
    if resolution_sigma is not None:
        resolution_sigma = np.asarray(resolution_sigma, dtype=float)
        if resolution_sigma.shape != observation.q.shape:
            raise ValueError("resolution_sigma must have the same shape as observation.q.")
        support_mask = np.isfinite(observation.q) & (observation.q > 0.0)
        q_model_eval = observation.q[support_mask]
        resolution_matrix = gaussian_resolution_matrix(
            q_obs,
            resolution_sigma[mask],
            source_q=q_model_eval,
        )

    stretch_ratio = float(parameters.get("stretch_ratio", 1.0)) if affine_stretch else 1.0
    k_distribution_params = _distribution_parameter_values(
        parameters,
        distribution_parameter_set,
    )
    r_sigma_k = float(k_distribution_params["r_sigma_k"])
    _, k_scale, mean_k, k_scale_metadata = _resolve_line_k_scale(
        {**parameters, "r_sigma_k": r_sigma_k}, model_settings
    )
    k_h_over_k = float(parameters.get("k_H_over_k", 0.0))
    b = float(parameters.get("b", -np.inf))
    lambda_perp = stretch_ratio ** -0.5
    min_stretch = min(lambda_perp, stretch_ratio)
    max_stretch = max(lambda_perp, stretch_ratio)
    q_min_factor = max(0.5 * float(q_model_eval[0]) * min_stretch / k_scale, 1.0e-5)
    q_max_factor = max(1.3 * float(q_model_eval[-1]) * max_stretch / k_scale, q_min_factor * 2.0)
    line_kwargs = {
        "k0_nominal": mean_k,
        "r_sigma_k": r_sigma_k,
        "k_distribution_params": k_distribution_params,
        "Q_min_factor": q_min_factor,
        "Q_max_factor": q_max_factor,
        "NQ": int(model_settings.get("NQ", 160)),
        "num_modes_k": int(model_settings.get("num_modes_k", 2**9)),
        "Nr": int(model_settings.get("Nr", 2500)),
        "Nr_small": model_settings.get("Nr_small", 700),
        "N_samp_U": int(model_settings.get("N_samp_U", 2**12)),
        "N_samp_st": int(model_settings.get("N_samp_st", 2**7)),
        "random_seed": int(model_settings.get("random_seed", 12345)),
        "k_sampling": str(model_settings.get("k_sampling", "qmc")),
        "k_distribution": str(model_settings.get("k_distribution", "gaussian_radial")),
        "r_grid_mode": str(model_settings.get("r_grid_mode", "mixed")),
        "r_min_factor": float(model_settings.get("r_min_factor", 1.0e-3)),
        "r_split_factor": float(model_settings.get("r_split_factor", 5.0)),
        "r_max_factor": float(model_settings.get("r_max_factor", 250.0)),
        "tail_start_fraction": float(model_settings.get("tail_start_fraction", 0.8)),
        "use_qmc": bool(model_settings.get("use_qmc", True)),
        "jacobian_method": str(model_settings.get("jacobian_method", "direct_12d")),
        "use_asymptotic": bool(model_settings.get("use_asymptotic", True)),
        "lowq_fit_bounds_over_k_eff": model_settings.get("lowq_fit_bounds_over_k_eff", (0.35, 0.8)),
        "lowq_replace_max_over_k_eff": model_settings.get("lowq_replace_max_over_k_eff", 0.5),
        "compute_tangent_correlations": _can_share_affine_tangent_samples(
            affine_stretch, model_settings
        ),
        "progress": bool(model_settings.get("progress", False)),
    }
    line = rls.compute_uniform_line_scattering(**line_kwargs)
    if model_mode == "heterogeneous":
        k_h = k_h_over_k * float(getattr(line, "uniform_meta", {}).get("k_mean", mean_k))
        hetero = rls.heterogeneous_line_scattering(line, k_H=k_h, b=b, return_components=True)
        model_q = hetero.Q_grid
        model_i = hetero.I_h
        p_h = float(hetero.p_H)
        sigma_h_squared = float(hetero.sigma_H_squared)
        alpha_h = float(hetero.alpha_H)
        kappa_h = float(hetero.kappa_H)
        rho0 = float(hetero.rho0)
    else:
        k_h = 0.0
        model_q = line.Q_grid
        model_i = line.I_L
        p_h = 1.0
        sigma_h_squared = 0.0
        alpha_h = 0.0
        kappa_h = 0.0
        rho0 = float(line.rho0)
    affine_highq_factor = 1.0
    affine_geometric_result = None
    if affine_stretch:
        raw, affine_highq_factor, affine_geometric_result = _affine_powder_curve(
            q_model_eval,
            line,
            model_q,
            model_i,
            stretch_ratio,
            model_mode=model_mode,
            model_settings=model_settings,
            rls=rls,
            k_h=k_h,
            b=b,
        )
    else:
        raw = np.interp(np.log(q_model_eval), np.log(model_q), model_i)
    if resolution_matrix is not None:
        raw = resolution_matrix @ raw
    raw = np.maximum(raw, np.finfo(float).tiny)
    loss_mode = str(regression_loss or model_settings.get("regression_loss", "log")).lower()
    if loss_mode == "relative":
        err_abs = np.maximum(e_obs, float(log_error_floor) * np.maximum(np.abs(i_obs), np.finfo(float).tiny))
        weights = point_weights / (err_abs * err_abs)
        scale = float(np.sum(weights * raw * i_obs) / max(np.sum(weights * raw * raw), np.finfo(float).tiny))
        scale = max(scale, np.finfo(float).tiny)
        model = scale * raw
        residual = np.sqrt(point_weights) * (model - i_obs) / err_abs
    elif loss_mode == "log":
        err_log = np.maximum(e_obs / np.maximum(i_obs, np.finfo(float).tiny), float(log_error_floor))
        weights = point_weights / (err_log * err_log)
        log_scale = float(np.sum(weights * (np.log(i_obs) - np.log(raw))) / np.sum(weights))
        scale = float(np.exp(log_scale))
        model = scale * raw
        residual = np.sqrt(point_weights) * (np.log(model) - np.log(i_obs)) / err_log
    else:
        raise ValueError("regression_loss must be 'relative' or 'log'.")
    params = {
        "scale": scale,
        "mean_k": mean_k,
        "k_eff": k_scale_metadata["k_eff"],
        "sigma_k": k_scale_metadata["sigma_k"],
        "r_sigma_k": r_sigma_k,
        "k_H_over_k": k_h_over_k,
        "b": b,
        "k_H": k_h,
        "p_H": p_h,
        "sigma_H_squared": sigma_h_squared,
        "alpha_H": alpha_h,
        "kappa_H": kappa_h,
        "rho0": rho0,
        "stretch_ratio": stretch_ratio,
        "highq_coefficient": scale * np.pi * p_h * rho0 * affine_highq_factor,
    }
    params.update(
        {
            name: k_scale_metadata[name]
            for name in (
                "gamma_shape",
                "gamma_scale",
                "gaussian_location",
                "gaussian_nominal_sigma",
            )
            if name in k_scale_metadata
        }
    )
    params.update(k_distribution_params)
    result = HeterogeneousFitResult(
        parameters=params,
        free_parameters=np.asarray(
            (
                [k_scale, *k_distribution_params.values(), *([stretch_ratio] if affine_stretch else []), k_h_over_k, b]
                if model_mode == "heterogeneous"
                else [k_scale, *k_distribution_params.values(), *([stretch_ratio] if affine_stretch else [])]
            ),
            dtype=float,
        ),
        residual=residual,
        q=q_obs,
        intensity=i_obs,
        error=e_obs,
        model=model,
        unscaled_model=raw,
        success=True,
        message="Initial guess evaluated without nonlinear optimization.",
        cost=0.5 * float(np.sum(residual * residual)),
        nfev=1,
    )
    if affine_geometric_result is not None:
        object.__setattr__(result, "affine_geometric_result", affine_geometric_result)
    object.__setattr__(
        result,
        "affine_model",
        str(model_settings.get("affine_model", "scalar")),
    )
    return result


def _smooth_inverse_square_lower_residual(
    value: float,
    *,
    lower_limit: float,
    cutoff: float,
    strength: float,
    epsilon_fraction: float,
) -> float:
    """Compact C1 inverse-square barrier used for a soft lower limit."""

    width = float(cutoff) - float(lower_limit)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("inverse-square barrier requires lower_limit < cutoff")
    epsilon = float(epsilon_fraction)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon_fraction must be finite and positive")
    amplitude = float(strength)
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("strength must be finite and positive")

    t = (float(value) - float(lower_limit)) / width
    if t >= 1.0:
        return 0.0

    endpoint_inverse = 1.0 / (1.0 + epsilon)

    def profile(x: float) -> float:
        return (
            1.0 / (x + epsilon) ** 2
            - endpoint_inverse**2
            + 2.0 * (x - 1.0) * endpoint_inverse**3
        )

    if t <= 0.0:
        value_at_lower = profile(0.0)
        slope_at_lower = -2.0 / epsilon**3 + 2.0 * endpoint_inverse**3
        barrier = value_at_lower + slope_at_lower * t
    else:
        barrier = profile(t)
    return amplitude * max(0.0, float(barrier))


def fit_heterogeneous_line_least_squares(
    observation: RadialProfile,
    *,
    q_bounds: tuple[float, float],
    initial: Mapping[str, float],
    model_mode: str = "heterogeneous",
    resolution_sigma: np.ndarray | None = None,
    regression_weights: np.ndarray | None = None,
    lowq_kappa_anchor: float | None = None,
    highq_coefficient_anchor: float | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    model_settings: Mapping[str, float | int | str | bool | None] | None = None,
    max_nfev: int = 12,
    anchor_weight: float = 2.0,
    log_error_floor: float = 0.03,
    regression_loss: str | None = None,
    distribution_parameter_set: Mapping[str, Mapping[str, object]] | None = None,
    constraints: LineFitConstraints | None = None,
    fixed_parameters: Mapping[str, float] | None = None,
    parameter_penalties: Mapping[str, tuple[float, float]] | None = None,
    soft_lower_bounds: Mapping[str, tuple[float, float]] | None = None,
    inverse_square_lower_barriers: Mapping[
        str, tuple[float, float, float, float]
    ] | None = None,
    verbose: int = 0,
    affine_stretch: bool = False,
) -> HeterogeneousFitResult:
    """Constrained first-pass fit of a random-line model.

    Heterogeneous mode searches the selected k scale, ``k_H_over_k``, ``b``, and the
    distribution parameters. ``line_only`` mode omits the two mask parameters.
    Entries supplied in ``fixed_parameters`` are excluded from either search.
    ``parameter_penalties`` maps a parameter to ``(center, sigma)`` and adds a
    Gaussian regularization residual, useful for parsimonious spline shapes.
    ``soft_lower_bounds`` maps a parameter to ``(threshold, sigma)`` and adds
    ``max(0, (threshold-value)/sigma)^2`` to the residual. It is therefore
    inactive above the threshold and quartic in the least-squares objective
    below it.
    ``inverse_square_lower_barriers`` maps a parameter to
    ``(lower, cutoff, strength, epsilon_fraction)``. Between ``lower`` and
    ``cutoff`` it follows a regularized inverse-square profile whose value and
    first derivative vanish at ``cutoff``. Below ``lower`` its tangent is
    continued linearly, keeping the residual and derivative continuous while
    strongly driving the parameter upward.
    ``constraints`` groups all of these optional regularizers and additionally
    supports a derived geometrical line-density anchor.  Do not mix the grouped
    object with the individual legacy constraint arguments.
    The overall scale is solved at each trial by weighted amplitude matching.
    When ``resolution_sigma`` is supplied, every trial curve is evaluated and
    convolved on the full available observation Q grid. Only the output points
    inside ``q_bounds`` enter the regression loss. Positive ``regression_weights``
    can emphasize a feature region while retaining the full fitted Q range.
    """

    rls = _import_line_scattering()
    model_mode = str(model_mode).lower()
    if model_mode not in {"heterogeneous", "line_only"}:
        raise ValueError("model_mode must be 'heterogeneous' or 'line_only'.")
    bounds = dict(bounds or {})
    model_settings = dict(model_settings or {})
    if constraints is not None:
        if not isinstance(constraints, LineFitConstraints):
            raise TypeError("constraints must be a LineFitConstraints instance.")
        legacy_constraints = (
            fixed_parameters,
            parameter_penalties,
            soft_lower_bounds,
            inverse_square_lower_barriers,
        )
        if any(value is not None for value in legacy_constraints):
            raise ValueError(
                "Pass either constraints=LineFitConstraints(...) or the individual "
                "constraint arguments, not both."
            )
        fixed_parameters = constraints.fixed_parameters
        parameter_penalties = constraints.parameter_penalties
        soft_lower_bounds = constraints.soft_lower_bounds
        inverse_square_lower_barriers = constraints.inverse_square_lower_barriers
        line_density_anchor = constraints.line_density_anchor
        line_density_anchor_mode = constraints.line_density_anchor_mode
    else:
        line_density_anchor = None
        line_density_anchor_mode = "two_sided"
    q_min, q_max = map(float, q_bounds)
    mask = (
        np.isfinite(observation.q)
        & np.isfinite(observation.intensity)
        & np.isfinite(observation.error)
        & (observation.q >= q_min)
        & (observation.q <= q_max)
        & (observation.q > 0.0)
        & (observation.intensity > 0.0)
        & (observation.error > 0.0)
    )
    q_obs = observation.q[mask]
    i_obs = observation.intensity[mask]
    e_obs = observation.error[mask]
    if q_obs.size < 8:
        raise ValueError("Need at least eight positive observation points for constrained fitting.")
    if regression_weights is None:
        point_weights = np.ones_like(q_obs)
    else:
        regression_weights = np.asarray(regression_weights, dtype=float)
        if regression_weights.shape != observation.q.shape:
            raise ValueError("regression_weights must have the same shape as observation.q.")
        point_weights = regression_weights[mask]
        if not np.all(np.isfinite(point_weights) & (point_weights > 0.0)):
            raise ValueError("regression_weights must be finite and strictly positive.")
    resolution_matrix = None
    q_model_eval = q_obs
    if resolution_sigma is not None:
        resolution_sigma = np.asarray(resolution_sigma, dtype=float)
        if resolution_sigma.shape != observation.q.shape:
            raise ValueError("resolution_sigma must have the same shape as observation.q.")
        support_mask = np.isfinite(observation.q) & (observation.q > 0.0)
        q_model_eval = observation.q[support_mask]
        resolution_matrix = gaussian_resolution_matrix(
            q_obs,
            resolution_sigma[mask],
            source_q=q_model_eval,
        )

    parameter_set = dict(distribution_parameter_set or {})
    distribution_names = tuple(parameter_set) if parameter_set else ("r_sigma_k",)
    if "r_sigma_k" not in distribution_names:
        raise ValueError("distribution_parameter_set must include 'r_sigma_k'.")
    stretch_names = ("stretch_ratio",) if affine_stretch else ()
    k_scale_convention = str(model_settings.get("k_scale_convention", "mean")).lower()
    if k_scale_convention not in {"mean", "k_eff"}:
        raise ValueError("k_scale_convention must be 'mean' or 'k_eff'.")
    if (
        k_scale_convention == "k_eff"
        and str(model_settings.get("k_distribution", "gaussian_radial"))
        not in {"gamma_radial", "gaussian_radial"}
    ):
        raise ValueError(
            "k_scale_convention='k_eff' is defined for gamma_radial and gaussian_radial."
        )
    k_parameter_name = "k_eff" if k_scale_convention == "k_eff" else "mean_k"
    names = (
        (k_parameter_name, *distribution_names, *stretch_names, "k_H_over_k", "b")
        if model_mode == "heterogeneous"
        else (k_parameter_name, *distribution_names, *stretch_names)
    )
    fixed = {name: float(value) for name, value in dict(fixed_parameters or {}).items()}
    unknown_fixed = tuple(name for name in fixed if name not in names)
    if unknown_fixed:
        allowed = ", ".join(names)
        raise ValueError(f"Unknown fixed parameter(s) {unknown_fixed}; allowed names are: {allowed}.")
    if not all(np.isfinite(value) for value in fixed.values()):
        raise ValueError("All fixed parameter values must be finite.")
    penalties = {
        str(name): tuple(map(float, specification))
        for name, specification in dict(parameter_penalties or {}).items()
    }
    unknown_penalties = tuple(name for name in penalties if name not in names)
    if unknown_penalties:
        raise ValueError(f"Unknown penalized parameter(s): {unknown_penalties}.")
    for name, (center, sigma) in penalties.items():
        if not np.isfinite(center) or not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"Penalty for {name!r} needs a finite center and positive sigma.")
    lower_penalties = {
        str(name): tuple(map(float, specification))
        for name, specification in dict(soft_lower_bounds or {}).items()
    }
    unknown_lower_penalties = tuple(name for name in lower_penalties if name not in names)
    if unknown_lower_penalties:
        raise ValueError(f"Unknown soft-lower-bound parameter(s): {unknown_lower_penalties}.")
    for name, (threshold, sigma) in lower_penalties.items():
        if not np.isfinite(threshold) or not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(
                f"Soft lower bound for {name!r} needs a finite threshold and positive sigma."
            )
    inverse_square_barriers = {
        str(name): tuple(map(float, specification))
        for name, specification in dict(inverse_square_lower_barriers or {}).items()
    }
    unknown_inverse_square = tuple(name for name in inverse_square_barriers if name not in names)
    if unknown_inverse_square:
        raise ValueError(
            f"Unknown inverse-square lower-barrier parameter(s): {unknown_inverse_square}."
        )
    for name, specification in inverse_square_barriers.items():
        if len(specification) != 4:
            raise ValueError(
                f"Inverse-square lower barrier for {name!r} requires "
                "(lower, cutoff, strength, epsilon_fraction)."
            )
        lower_limit, cutoff, strength, epsilon_fraction = specification
        if not all(np.isfinite(specification)):
            raise ValueError(f"Inverse-square lower barrier for {name!r} must be finite.")
        if not lower_limit < cutoff:
            raise ValueError(f"Inverse-square lower barrier for {name!r} needs lower < cutoff.")
        if strength <= 0.0 or epsilon_fraction <= 0.0:
            raise ValueError(
                f"Inverse-square lower barrier for {name!r} needs positive strength "
                "and epsilon_fraction."
            )
    penalties = {name: value for name, value in penalties.items() if name not in fixed}
    lower_penalties = {
        name: value for name, value in lower_penalties.items() if name not in fixed
    }
    inverse_square_barriers = {
        name: value for name, value in inverse_square_barriers.items() if name not in fixed
    }
    density_anchor: tuple[float, float] | None = None
    density_anchor_mode = str(line_density_anchor_mode).lower()
    valid_density_anchor_modes = {"two_sided", "below_reference_only"}
    if density_anchor_mode not in valid_density_anchor_modes:
        raise ValueError(
            "line_density_anchor_mode must be 'two_sided' or "
            "'below_reference_only'."
        )
    if line_density_anchor is not None:
        if len(line_density_anchor) != 2:
            raise ValueError(
                "line_density_anchor requires (reference_density, relative_sigma)."
            )
        reference_density, relative_sigma = map(float, line_density_anchor)
        if (
            not np.isfinite(reference_density)
            or reference_density <= 0.0
            or not np.isfinite(relative_sigma)
            or relative_sigma <= 0.0
        ):
            raise ValueError(
                "line_density_anchor needs a positive finite reference density "
                "and relative sigma."
            )
        density_anchor = (reference_density, relative_sigma)
    free_names = tuple(name for name in names if name not in fixed)
    default_bounds = {
        "mean_k": (0.03, 0.3),
        "k_eff": (0.03, 0.3),
        "r_sigma_k": (0.03, 0.8),
        "stretch_ratio": (0.7, 1.8),
        "k_H_over_k": (0.005, 0.5),
        "b": (-3.0, 1.5),
    }
    for name, specification in parameter_set.items():
        parameter_bounds = specification.get("bounds")
        if parameter_bounds is None or len(parameter_bounds) != 2:
            raise ValueError(f"Distribution parameter {name!r} needs a two-value 'bounds'.")
        default_bounds[name] = tuple(map(float, parameter_bounds))
    all_bounds = {name: tuple(map(float, bounds.get(name, default_bounds[name]))) for name in names}
    for name, (lo, hi) in all_bounds.items():
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"Parameter {name!r} needs finite bounds with lower < upper.")
    for name, value in fixed.items():
        lo, hi = all_bounds[name]
        if value < lo or value > hi:
            raise ValueError(f"Fixed value for {name!r}={value} is outside bounds ({lo}, {hi}).")
    lower = np.array([all_bounds[name][0] for name in free_names], dtype=float)
    upper = np.array([all_bounds[name][1] for name in free_names], dtype=float)
    x0 = np.array([float(initial[name]) for name in free_names], dtype=float)
    x0 = np.minimum(np.maximum(x0, lower + 1.0e-12), upper - 1.0e-12)
    cache: dict[
        tuple[float, ...],
        tuple[np.ndarray, object, object, float, object | None],
    ] = {}
    line_cache: dict[tuple[float, ...], object] = {}
    if affine_stretch:
        if "stretch_ratio" in fixed:
            stretch_coverage = (fixed["stretch_ratio"], fixed["stretch_ratio"])
        else:
            stretch_coverage = all_bounds["stretch_ratio"]
        stretch_candidates = np.asarray(
            [
                stretch_coverage[0],
                stretch_coverage[1],
                stretch_coverage[0] ** -0.5,
                stretch_coverage[1] ** -0.5,
            ],
            dtype=float,
        )
        coverage_min_stretch = float(np.min(stretch_candidates))
        coverage_max_stretch = float(np.max(stretch_candidates))
    else:
        coverage_min_stretch = 1.0
        coverage_max_stretch = 1.0

    def values_from_free(params: np.ndarray) -> dict[str, float]:
        values = dict(fixed)
        values.update(zip(free_names, map(float, params)))
        return values

    def evaluate_raw(
        params: np.ndarray,
    ) -> tuple[np.ndarray, object, object, float, object | None]:
        values = values_from_free(params)
        r_sigma_k = values["r_sigma_k"]
        _, k_scale, mean_k, _ = _resolve_line_k_scale(values, model_settings)
        stretch_ratio = values.get("stretch_ratio", 1.0)
        k_h_over_k = values.get("k_H_over_k", 0.0)
        b = values.get("b", -np.inf)
        k_distribution_params = {
            name: values[name]
            for name in distribution_names
        }
        key = tuple(round(values[name], 10) for name in names)
        if key in cache:
            return cache[key]
        q_min_factor = max(
            0.5 * float(q_model_eval[0]) * coverage_min_stretch / k_scale,
            1.0e-5,
        )
        q_max_factor = max(
            1.3 * float(q_model_eval[-1]) * coverage_max_stretch / k_scale,
            q_min_factor * 2.0,
        )
        line_kwargs = {
            "k0_nominal": mean_k,
            "r_sigma_k": r_sigma_k,
            "k_distribution_params": k_distribution_params,
            "Q_min_factor": q_min_factor,
            "Q_max_factor": q_max_factor,
            "NQ": int(model_settings.get("NQ", 160)),
            "num_modes_k": int(model_settings.get("num_modes_k", 2**9)),
            "Nr": int(model_settings.get("Nr", 2500)),
            "Nr_small": model_settings.get("Nr_small", 700),
            "N_samp_U": int(model_settings.get("N_samp_U", 2**12)),
            "N_samp_st": int(model_settings.get("N_samp_st", 2**7)),
            "random_seed": int(model_settings.get("random_seed", 12345)),
            "k_sampling": str(model_settings.get("k_sampling", "qmc")),
            "k_distribution": str(model_settings.get("k_distribution", "gaussian_radial")),
            "r_grid_mode": str(model_settings.get("r_grid_mode", "mixed")),
            "r_min_factor": float(model_settings.get("r_min_factor", 1.0e-3)),
            "r_split_factor": float(model_settings.get("r_split_factor", 5.0)),
            "r_max_factor": float(model_settings.get("r_max_factor", 250.0)),
            "tail_start_fraction": float(model_settings.get("tail_start_fraction", 0.8)),
            "use_qmc": bool(model_settings.get("use_qmc", True)),
            "jacobian_method": str(model_settings.get("jacobian_method", "direct_12d")),
            "use_asymptotic": bool(model_settings.get("use_asymptotic", True)),
            "lowq_fit_bounds_over_k_eff": model_settings.get("lowq_fit_bounds_over_k_eff", (0.35, 0.8)),
            "lowq_replace_max_over_k_eff": model_settings.get("lowq_replace_max_over_k_eff", 0.5),
            "compute_tangent_correlations": _can_share_affine_tangent_samples(
                affine_stretch, model_settings
            ),
            "progress": bool(model_settings.get("progress", False)),
        }
        line_key = tuple(
            round(values[name], 10)
            for name in (k_parameter_name, *distribution_names)
        )
        if line_key not in line_cache:
            line_cache[line_key] = rls.compute_uniform_line_scattering(**line_kwargs)
        line = line_cache[line_key]
        if model_mode == "heterogeneous":
            k_h = k_h_over_k * float(getattr(line, "uniform_meta", {}).get("k_mean", mean_k))
            model_result = rls.heterogeneous_line_scattering(
                line,
                k_H=k_h,
                b=b,
                return_components=True,
            )
            model_q = model_result.Q_grid
            model_i = model_result.I_h
        else:
            model_result = line
            model_q = line.Q_grid
            model_i = line.I_L
        affine_highq_factor = 1.0
        affine_geometric_result = None
        if affine_stretch:
            (
                model_grid,
                affine_highq_factor,
                affine_geometric_result,
            ) = _affine_powder_curve(
                q_model_eval,
                line,
                model_q,
                model_i,
                stretch_ratio,
                model_mode=model_mode,
                model_settings=model_settings,
                rls=rls,
                k_h=k_h,
                b=b,
            )
        else:
            model_grid = np.interp(np.log(q_model_eval), np.log(model_q), model_i)
        if resolution_matrix is not None:
            model_grid = resolution_matrix @ model_grid
        cache[key] = (
            model_grid,
            line,
            model_result,
            affine_highq_factor,
            affine_geometric_result,
        )
        return cache[key]

    loss_mode = str(regression_loss or model_settings.get("regression_loss", "log")).lower()
    if loss_mode not in {"relative", "log"}:
        raise ValueError("regression_loss must be 'relative' or 'log'.")

    def scaled_model_and_residual(params: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        raw, line, model_result, affine_highq_factor, _ = evaluate_raw(params)
        raw = np.maximum(raw, np.finfo(float).tiny)
        if loss_mode == "relative":
            err_abs = np.maximum(e_obs, float(log_error_floor) * np.maximum(np.abs(i_obs), np.finfo(float).tiny))
            weights = point_weights / (err_abs * err_abs)
            scale = float(np.sum(weights * raw * i_obs) / max(np.sum(weights * raw * raw), np.finfo(float).tiny))
            scale = max(scale, np.finfo(float).tiny)
            residual = np.sqrt(point_weights) * (scale * raw - i_obs) / err_abs
        else:
            err_log = np.maximum(e_obs / np.maximum(i_obs, np.finfo(float).tiny), float(log_error_floor))
            weights = point_weights / (err_log * err_log)
            log_scale = float(np.sum(weights * (np.log(i_obs) - np.log(raw))) / np.sum(weights))
            scale = float(np.exp(log_scale))
            residual = np.sqrt(point_weights) * (np.log(scale * raw) - np.log(i_obs)) / err_log
        anchor_residuals: list[float] = []
        if highq_coefficient_anchor is not None:
            p_h = float(model_result.p_H) if model_mode == "heterogeneous" else 1.0
            rho0 = float(model_result.rho0)
            coeff = scale * np.pi * p_h * rho0 * affine_highq_factor
            anchor_residuals.append(float(anchor_weight) * np.log(coeff / float(highq_coefficient_anchor)))
        if (
            model_mode == "heterogeneous"
            and lowq_kappa_anchor is not None
            and float(model_result.kappa_H) > 0.0
        ):
            anchor_residuals.append(
                float(anchor_weight)
                * np.log(float(model_result.kappa_H) / float(lowq_kappa_anchor))
            )
        if anchor_residuals:
            residual = np.concatenate([residual, np.asarray(anchor_residuals, dtype=float)])
        if density_anchor is not None:
            reference_density, relative_sigma = density_anchor
            p_h = float(model_result.p_H) if model_mode == "heterogeneous" else 1.0
            line_density = (
                p_h * float(model_result.rho0) * float(affine_highq_factor)
            )
            density_log_ratio = np.log(line_density / reference_density)
            if density_anchor_mode == "below_reference_only":
                density_log_ratio = min(0.0, density_log_ratio)
            density_residual = density_log_ratio / relative_sigma
            residual = np.concatenate(
                [residual, np.asarray([density_residual], dtype=float)]
            )
        if penalties:
            values = values_from_free(params)
            penalty_residuals = np.asarray(
                [(values[name] - center) / sigma for name, (center, sigma) in penalties.items()],
                dtype=float,
            )
            residual = np.concatenate([residual, penalty_residuals])
        if lower_penalties:
            values = values_from_free(params)
            lower_residuals = np.asarray(
                [
                    max(0.0, (threshold - values[name]) / sigma) ** 2
                    for name, (threshold, sigma) in lower_penalties.items()
                ],
                dtype=float,
            )
            residual = np.concatenate([residual, lower_residuals])
        if inverse_square_barriers:
            values = values_from_free(params)
            barrier_residuals = np.asarray(
                [
                    _smooth_inverse_square_lower_residual(
                        values[name],
                        lower_limit=lower_limit,
                        cutoff=cutoff,
                        strength=strength,
                        epsilon_fraction=epsilon_fraction,
                    )
                    for name, (
                        lower_limit,
                        cutoff,
                        strength,
                        epsilon_fraction,
                    ) in inverse_square_barriers.items()
                ],
                dtype=float,
            )
            residual = np.concatenate([residual, barrier_residuals])
        return scale, scale * raw, residual

    def residual_fn(params: np.ndarray) -> np.ndarray:
        _, _, residual = scaled_model_and_residual(params)
        if not np.all(np.isfinite(residual)):
            n_anchors = int(highq_coefficient_anchor is not None)
            n_anchors += int(model_mode == "heterogeneous" and lowq_kappa_anchor is not None)
            return np.full(
                q_obs.size
                + n_anchors
                + len(penalties)
                + len(lower_penalties)
                + len(inverse_square_barriers)
                + int(density_anchor is not None),
                1.0e6,
                dtype=float,
            )
        return residual

    if free_names:
        opt = least_squares(
            residual_fn,
            x0,
            bounds=(lower, upper),
            max_nfev=int(max_nfev),
            verbose=int(verbose),
        )
        free_values = np.asarray(opt.x, dtype=float)
        success = bool(opt.success)
        message = str(opt.message)
        nfev = int(opt.nfev)
        cost = float(opt.cost)
    else:
        free_values = np.asarray([], dtype=float)
        _, _, direct_residual = scaled_model_and_residual(free_values)
        success = True
        message = "All nonlinear parameters were fixed; evaluated model without optimization."
        nfev = 1
        cost = 0.5 * float(np.sum(direct_residual * direct_residual))
    scale, model, residual = scaled_model_and_residual(free_values)
    (
        raw,
        line,
        model_result,
        affine_highq_factor,
        affine_geometric_result,
    ) = evaluate_raw(free_values)
    fitted = values_from_free(free_values)
    r_sigma_k = fitted["r_sigma_k"]
    _, k_scale, mean_k, k_scale_metadata = _resolve_line_k_scale(fitted, model_settings)
    stretch_ratio = fitted.get("stretch_ratio", 1.0)
    k_h_over_k = fitted.get("k_H_over_k", 0.0)
    b = fitted.get("b", -np.inf)
    if model_mode == "heterogeneous":
        k_h = float(model_result.kappa_H) / max(
            _kappa_over_kh_from_b(b),
            np.finfo(float).tiny,
        )
        p_h = float(model_result.p_H)
        sigma_h_squared = float(model_result.sigma_H_squared)
        alpha_h = float(model_result.alpha_H)
        kappa_h = float(model_result.kappa_H)
    else:
        k_h = 0.0
        p_h = 1.0
        sigma_h_squared = 0.0
        alpha_h = 0.0
        kappa_h = 0.0
    rho0 = float(model_result.rho0)
    line_density_geometric = p_h * rho0 * float(affine_highq_factor)
    chi_square_error = np.maximum(
        e_obs,
        float(log_error_floor) * np.maximum(np.abs(i_obs), np.finfo(float).tiny),
    )
    data_chi_square = float(np.sum(((model - i_obs) / chi_square_error) ** 2))
    n_fitted_parameters = len(free_names) + 1  # nonlinear parameters plus fitted scale
    data_degrees_of_freedom = int(q_obs.size - n_fitted_parameters)
    data_reduced_chi_square = (
        data_chi_square / data_degrees_of_freedom
        if data_degrees_of_freedom > 0
        else np.nan
    )
    params = {
        "scale": scale,
        "mean_k": k_scale_metadata["mean_k"],
        "k_eff": k_scale_metadata["k_eff"],
        "sigma_k": k_scale_metadata["sigma_k"],
        "r_sigma_k": r_sigma_k,
        "k_H_over_k": k_h_over_k,
        "b": b,
        "k_H": k_h,
        "p_H": p_h,
        "sigma_H_squared": sigma_h_squared,
        "alpha_H": alpha_h,
        "kappa_H": kappa_h,
        "rho0": rho0,
        "affine_highq_factor": float(affine_highq_factor),
        "line_density_geometric": line_density_geometric,
        "stretch_ratio": stretch_ratio,
        "highq_coefficient": scale * np.pi * line_density_geometric,
        "data_chi_square": data_chi_square,
        "data_reduced_chi_square": data_reduced_chi_square,
        "data_degrees_of_freedom": float(data_degrees_of_freedom),
        "n_fitted_parameters": float(n_fitted_parameters),
    }
    params.update(
        {
            name: k_scale_metadata[name]
            for name in (
                "gamma_shape",
                "gamma_scale",
                "gaussian_location",
                "gaussian_nominal_sigma",
            )
            if name in k_scale_metadata
        }
    )
    params.update({name: fitted[name] for name in distribution_names})
    result = HeterogeneousFitResult(
        parameters=params,
        free_parameters=free_values,
        residual=residual,
        q=q_obs,
        intensity=i_obs,
        error=e_obs,
        model=model,
        unscaled_model=raw,
        success=success,
        message=message,
        cost=cost,
        nfev=nfev,
    )
    if affine_geometric_result is not None:
        object.__setattr__(result, "affine_geometric_result", affine_geometric_result)
    object.__setattr__(
        result,
        "affine_model",
        str(model_settings.get("affine_model", "scalar")),
    )
    return result


# -----------------------------------------------------------------------------
# 3D heterogeneous-line preview helpers
# -----------------------------------------------------------------------------

@dataclass
class HeterogeneousPreview:
    plotter: object
    screenshot_path: Path | None
    line_cells: int
    retained_cells: int
    retained_points: int
    mask_threshold: float
    p_H: float
    k_line: float
    k_distribution: str
    r_sigma_k: float
    k_distribution_params: dict[str, float]
    k_H: float
    lateral_size: float
    thickness: float


def make_coordinates(nx: int, ny: int, nz: int, lx: float, ly: float, lz: float):
    x = np.linspace(-0.5 * lx, 0.5 * lx, nx)
    y = np.linspace(-0.5 * ly, 0.5 * ly, ny)
    z = np.linspace(-0.5 * lz, 0.5 * lz, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return x, y, z, X, Y, Z


def sample_isotropic_k_vectors(num_modes: int, k0: float, rng: np.random.Generator) -> np.ndarray:
    directions = rng.normal(size=(num_modes, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return float(k0) * directions


def sample_preview_k_vectors(
    num_modes: int,
    k0: float,
    rng: np.random.Generator,
    *,
    k_distribution: str = "single_shell",
    r_sigma_k: float = 0.0,
    k_distribution_params: Mapping[str, float] | None = None,
    random_seed: int = 12345,
) -> np.ndarray:
    if k_distribution == "single_shell":
        return sample_isotropic_k_vectors(num_modes, k0, rng)
    rls = _import_line_scattering()
    return rls.sample_k_vectors(
        int(num_modes),
        k_distribution,
        rng,
        k0=float(k0),
        sigma_k=float(r_sigma_k) * float(k0),
        distribution_params=k_distribution_params,
        use_qmc=False,
        qmc_seed=int(random_seed),
    )


def random_wave_field(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    k_vectors: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(k_vectors))
    field = np.zeros_like(X, dtype=float)
    for (kx, ky, kz), phase in zip(k_vectors, phases):
        field += np.cos(kx * X + ky * Y + kz * Z + phase)
    field *= np.sqrt(2.0 / len(k_vectors))
    field -= np.mean(field)
    std = np.std(field)
    if std > 0.0:
        field /= std
    return field


def dab_filtered_gaussian_field(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    xi: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a smooth Gaussian mask field with a DAB-like spectral envelope."""

    nx, ny, nz = shape
    dx, dy, dz = spacing
    white = rng.normal(size=shape)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = KX**2 + KY**2 + KZ**2
    spectral_density = 1.0 / (1.0 + k2 * float(xi) ** 2) ** 2
    spectral_density[0, 0, 0] = 0.0
    filtered = np.fft.ifftn(np.fft.fftn(white) * np.sqrt(spectral_density)).real
    filtered -= np.mean(filtered)
    std = np.std(filtered)
    if std > 0.0:
        filtered /= std
    return filtered


def binary_mask_from_fraction(mask_field: np.ndarray, volume_fraction: float) -> tuple[np.ndarray, float]:
    p_H = float(np.clip(volume_fraction, 0.0, 1.0))
    threshold = np.quantile(mask_field, 1.0 - p_H)
    return mask_field >= threshold, float(threshold)


def vtk_image_from_array(values: np.ndarray, lx: float, ly: float, lz: float):
    import pyvista as pv

    nx, ny, nz = values.shape
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.origin = (-0.5 * lx, -0.5 * ly, -0.5 * lz)
    grid.spacing = (lx / (nx - 1), ly / (ny - 1), lz / (nz - 1))
    grid.point_data["values"] = np.asarray(values, dtype=float).ravel(order="F")
    return grid


def _empty_polydata():
    import pyvista as pv

    return pv.PolyData()


def filter_segments_by_mask(index_poly, mask: np.ndarray):
    if index_poly.n_points == 0 or index_poly.lines.size == 0:
        return _empty_polydata(), _empty_polydata()
    points = np.asarray(index_poly.points, dtype=float)
    lines = np.asarray(index_poly.lines, dtype=np.int64)
    kept_points: list[np.ndarray] = []
    kept_lines: list[int] = []
    rejected_points: list[np.ndarray] = []
    rejected_lines: list[int] = []
    nx, ny, nz = mask.shape

    def append_segment(target_points, target_lines, p0, p1):
        base = len(target_points)
        target_points.extend([p0, p1])
        target_lines.extend([2, base, base + 1])

    cursor = 0
    while cursor < len(lines):
        n = int(lines[cursor])
        ids = lines[cursor + 1 : cursor + 1 + n]
        cursor += n + 1
        if n < 2:
            continue
        for i0, i1 in zip(ids[:-1], ids[1:]):
            p0 = points[int(i0)]
            p1 = points[int(i1)]
            mid = 0.5 * (p0 + p1)
            ix = int(np.clip(np.rint(mid[0]), 0, nx - 1))
            iy = int(np.clip(np.rint(mid[1]), 0, ny - 1))
            iz = int(np.clip(np.rint(mid[2]), 0, nz - 1))
            if mask[ix, iy, iz]:
                append_segment(kept_points, kept_lines, p0, p1)
            else:
                append_segment(rejected_points, rejected_lines, p0, p1)

    def build(points_list, lines_list):
        import pyvista as pv

        if not points_list:
            return pv.PolyData()
        poly = pv.PolyData(np.asarray(points_list, dtype=float))
        poly.lines = np.asarray(lines_list, dtype=np.int64)
        return poly

    return build(kept_points, kept_lines), build(rejected_points, rejected_lines)


def line_cell_count(poly) -> int:
    lines = np.asarray(getattr(poly, "lines", np.array([], dtype=np.int64)), dtype=np.int64)
    count = 0
    cursor = 0
    while cursor < len(lines):
        n = int(lines[cursor])
        count += 1
        cursor += n + 1
    return count


def has_line_cells(poly) -> bool:
    return line_cell_count(poly) > 0


def prune_short_line_cells(poly, min_length: float):
    if min_length <= 0.0 or not has_line_cells(poly):
        return poly
    import pyvista as pv

    raw_points = np.asarray(poly.points, dtype=float)
    raw_lines = np.asarray(poly.lines, dtype=np.int64)
    points: list[np.ndarray] = []
    lines: list[int] = []
    cursor = 0
    while cursor < len(raw_lines):
        n = int(raw_lines[cursor])
        ids = raw_lines[cursor + 1 : cursor + 1 + n]
        cursor += n + 1
        if n < 2:
            continue
        path = raw_points[ids]
        length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        if length < min_length:
            continue
        start = len(points)
        points.extend(path)
        lines.extend([n, *range(start, start + n)])
    if not points:
        return pv.PolyData()
    out = pv.PolyData(np.asarray(points, dtype=float))
    out.lines = np.asarray(lines, dtype=np.int64)
    return out


def index_poly_to_physical(index_poly, lx: float, ly: float, lz: float, shape: tuple[int, int, int]):
    if index_poly.n_points == 0:
        return _empty_polydata()
    nx, ny, nz = shape
    poly = index_poly.copy(deep=True)
    pts = np.asarray(poly.points, dtype=float).copy()
    pts[:, 0] = pts[:, 0] / (nx - 1) * lx - 0.5 * lx
    pts[:, 1] = pts[:, 1] / (ny - 1) * ly - 0.5 * ly
    pts[:, 2] = pts[:, 2] / (nz - 1) * lz - 0.5 * lz
    poly.points = pts
    return poly


def apply_anti_aliasing(plotter, mode: str = "ssaa", enabled: bool = True) -> bool:
    if not enabled:
        return False
    try:
        plotter.enable_anti_aliasing(mode)
        return True
    except Exception:
        try:
            plotter.enable_anti_aliasing()
            return True
        except Exception:
            return False


def apply_ambient_occlusion(plotter, radius: float, bias: float = 0.01, enabled: bool = True) -> bool:
    if not enabled:
        return False
    renderer = plotter.renderer
    try:
        renderer.UseSSAOOn()
        renderer.SetSSAORadius(float(radius))
        renderer.SetSSAOBias(float(bias))
        return True
    except Exception:
        try:
            renderer.enable_ssao(radius=float(radius), bias=float(bias))
            return True
        except Exception:
            return False


def set_flat_camera(
    plotter,
    lx: float,
    ly: float,
    lz: float,
    view: str = "normal",
    window_size: tuple[int, int] = (800, 800),
    fill_fraction: float = 0.95,
) -> float:
    aspect = float(window_size[0]) / float(window_size[1])
    fill_fraction = float(np.clip(fill_fraction, 0.05, 1.0))
    if view == "normal":
        plotter.camera_position = [
            (0.0, 0.0, 3.0 * max(lx, ly)),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
        scale = max(0.5 * ly / fill_fraction, 0.5 * lx / (aspect * fill_fraction))
    else:
        plotter.camera_position = [
            (0.0, -2.4 * ly, 0.35 * lz),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        scale = max(0.5 * lz / fill_fraction, 0.5 * lx / (aspect * fill_fraction))
    plotter.enable_parallel_projection()
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = scale
    plotter.camera.SetParallelProjection(True)
    plotter.camera.SetParallelScale(scale)
    plotter.reset_camera_clipping_range()
    return scale


def render_fit_heterogeneous_preview(
    fit_parameters: dict[str, float],
    *,
    output_path: str | Path | None = None,
    visual_k_line: float = 10.0,
    line_k_distribution: str = "single_shell",
    line_r_sigma_k: float | None = None,
    line_k_distribution_params: Mapping[str, float] | None = None,
    random_seed: int = 12345,
    num_line_modes: int = 128,
    nx: int = 160,
    ny: int = 160,
    nz: int = 10,
    lateral_size_over_mask_length: float = 20.0,
    thickness_over_mask_length: float = 1.0,
    line_tube_radius_fraction: float = 0.002,
    min_retained_line_length_over_k: float = 10.0,
    show_mask_boundary: bool = True,
    show_box_boundary: bool = True,
    show_rejected_lines: bool = False,
    window_size: tuple[int, int] = (800, 800),
    window_fill_fraction: float = 0.95,
    initial_view: str = "normal",
    enable_anti_aliasing: bool = True,
    anti_aliasing_mode: str = "ssaa",
    enable_ambient_occlusion: bool = True,
    screenshot_scale: int = 3,
):
    """Render a compact 3D preview using fitted heterogeneity ratios.

    The visual uses a normalized line wavenumber so the fitted length-scale
    ratios are visible without making the experimental box enormous.
    """

    import pyvista as pv

    root_dir = Path(__file__).resolve().parents[2]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    import rw_line_network as rln

    k_h_over_k = float(fit_parameters["k_H_over_k"])
    p_H = float(fit_parameters.get("p_H", 1.0))
    k_line = float(visual_k_line)
    r_sigma_k = (
        float(fit_parameters.get("r_sigma_k", 0.15))
        if line_r_sigma_k is None
        else float(line_r_sigma_k)
    )
    k_distribution_params = dict(line_k_distribution_params or {})
    k_distribution_params.setdefault("r_sigma_k", r_sigma_k)
    k_H = max(k_h_over_k * k_line, np.finfo(float).eps)
    xi_dab = 1.0 / k_H
    lateral_size = float(lateral_size_over_mask_length) / k_H
    thickness = float(thickness_over_mask_length) / k_H

    rng = np.random.default_rng(random_seed)
    x, y, z, X, Y, Z = make_coordinates(nx, ny, nz, lateral_size, lateral_size, thickness)
    spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    k_vectors = sample_preview_k_vectors(
        num_line_modes,
        k_line,
        rng,
        k_distribution=str(line_k_distribution),
        r_sigma_k=r_sigma_k,
        k_distribution_params=k_distribution_params,
        random_seed=random_seed,
    )
    phi_real = random_wave_field(X, Y, Z, k_vectors, rng)
    phi_imag = random_wave_field(X, Y, Z, k_vectors, rng)
    mask_field = dab_filtered_gaussian_field((nx, ny, nz), spacing, xi_dab, rng)
    mask, mask_threshold = binary_mask_from_fraction(mask_field, p_H)

    old_smooth = getattr(rln, "SMOOTH_VORTEX_LINES", True)
    old_scale = getattr(rln, "VORTEX_SMOOTHING_SCALE", 2)
    old_spline = getattr(rln, "VORTEX_SPLINE_SMOOTHING", None)
    try:
        rln.SMOOTH_VORTEX_LINES = True
        rln.VORTEX_SMOOTHING_SCALE = 2
        rln.VORTEX_SPLINE_SMOOTHING = None
        index_segments = rln.trace_vortex_segments(phi_real, phi_imag, min_segment_length=0.0)
        kept_index, rejected_index = filter_segments_by_mask(index_segments, mask)
        retained_raw = index_poly_to_physical(kept_index, lateral_size, lateral_size, thickness, (nx, ny, nz))
        rejected_raw = index_poly_to_physical(rejected_index, lateral_size, lateral_size, thickness, (nx, ny, nz))
        retained_poly = rln.smooth_vortex_polydata(retained_raw) if has_line_cells(retained_raw) else retained_raw
        rejected_poly = rln.smooth_vortex_polydata(rejected_raw) if has_line_cells(rejected_raw) else rejected_raw
        retained_poly = prune_short_line_cells(
            retained_poly,
            float(min_retained_line_length_over_k) / k_line,
        )
    finally:
        rln.SMOOTH_VORTEX_LINES = old_smooth
        rln.VORTEX_SMOOTHING_SCALE = old_scale
        rln.VORTEX_SPLINE_SMOOTHING = old_spline

    tube_radius = float(line_tube_radius_fraction) * lateral_size
    tube = retained_poly.tube(radius=tube_radius, n_sides=24, capping=True) if retained_poly.n_points else _empty_polydata()
    rejected_tube = (
        rejected_poly.tube(radius=0.7 * tube_radius, n_sides=16, capping=True)
        if rejected_poly.n_points
        else _empty_polydata()
    )
    mask_grid = vtk_image_from_array(mask.astype(float), lateral_size, lateral_size, thickness)
    mask_boundary = mask_grid.contour([0.5], scalars="values")

    plotter = pv.Plotter(window_size=window_size, off_screen=output_path is not None)
    plotter.set_background("white")
    apply_anti_aliasing(plotter, anti_aliasing_mode, enable_anti_aliasing)

    if show_mask_boundary and mask_boundary.n_points:
        plotter.add_mesh(
            mask_boundary,
            color="#666666",
            opacity=1.0,
            smooth_shading=True,
            label="mask boundary",
        )
    if show_rejected_lines and rejected_tube.n_points:
        plotter.add_mesh(
            rejected_tube,
            color="#c7c7c7",
            opacity=0.16,
            smooth_shading=True,
            show_edges=False,
            specular=0.18,
            label="outside mask",
        )
    if tube.n_points:
        plotter.add_mesh(
            tube,
            color=rln.SLACK_RED,
            opacity=1.0,
            smooth_shading=True,
            show_edges=False,
            edge_color=getattr(rln, "TUBE_EDGE_COLOR", "#222222"),
            specular=0.25,
            label="retained line",
            name="retained_line",
        )
    if show_box_boundary:
        box = pv.Box(
            bounds=(
                -0.5 * lateral_size,
                0.5 * lateral_size,
                -0.5 * lateral_size,
                0.5 * lateral_size,
                -0.5 * thickness,
                0.5 * thickness,
            )
        )
        plotter.add_mesh(box, style="wireframe", color="#555555", line_width=1.0, opacity=0.55)

    plotter.add_light(
        pv.Light(
            position=(0.0, -2.0 * lateral_size, 1.2 * lateral_size),
            focal_point=(0.0, 0.0, 0.0),
            intensity=0.9,
        )
    )
    plotter.add_light(
        pv.Light(
            position=(1.2 * lateral_size, 1.0 * lateral_size, 0.8 * lateral_size),
            focal_point=(0.0, 0.0, 0.0),
            intensity=0.45,
        )
    )
    apply_ambient_occlusion(
        plotter,
        radius=0.18 * lateral_size,
        bias=0.01,
        enabled=enable_ambient_occlusion,
    )
    set_flat_camera(
        plotter,
        lateral_size,
        lateral_size,
        thickness,
        view=initial_view,
        window_size=window_size,
        fill_fraction=window_fill_fraction,
    )

    screenshot_path = Path(output_path) if output_path is not None else None
    if screenshot_path is not None:
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(screenshot_path), scale=int(screenshot_scale))

    return HeterogeneousPreview(
        plotter=plotter,
        screenshot_path=screenshot_path,
        line_cells=line_cell_count(index_segments),
        retained_cells=line_cell_count(retained_poly),
        retained_points=int(retained_poly.n_points),
        mask_threshold=mask_threshold,
        p_H=p_H,
        k_line=k_line,
        k_distribution=str(line_k_distribution),
        r_sigma_k=r_sigma_k,
        k_distribution_params=k_distribution_params,
        k_H=k_H,
        lateral_size=lateral_size,
        thickness=thickness,
    )
