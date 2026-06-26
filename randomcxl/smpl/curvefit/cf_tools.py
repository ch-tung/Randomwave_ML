"""Curvefit helpers for stitched scattering data and heterogeneous-line models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import sys
from math import erfc

import numpy as np
from scipy.optimize import least_squares


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
class AnchorFit:
    name: str
    parameters: dict[str, float]
    q_fit_min: float
    q_fit_max: float
    q: np.ndarray
    y: np.ndarray
    y_fit: np.ndarray
    relative_rmse: float


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
) -> AnchorFit:
    """Fit ``I(Q) ~= C/Q + B`` and report the apparent ``rho0=C/pi``."""

    mask = _fit_mask(profile, q_bounds)
    q = profile.q[mask]
    y = profile.intensity[mask]
    err = profile.error[mask]
    if q.size < 3:
        raise ValueError("Need at least three positive points for the line anchor fit.")
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


def estimate_peak_position(profile: RadialProfile, q_bounds: tuple[float, float]) -> dict[str, float]:
    """Return the highest positive-intensity point in the selected Q window."""

    mask = _fit_mask(profile, q_bounds)
    if np.count_nonzero(mask) == 0:
        raise ValueError("No positive points available for peak-position estimate.")
    idx_local = int(np.argmax(profile.intensity[mask]))
    q = profile.q[mask]
    y = profile.intensity[mask]
    return {"q_peak": float(q[idx_local]), "i_peak": float(y[idx_local])}


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
) -> dict[str, float]:
    """Build a physically coupled first guess from the anchor fits."""

    mean_k = float(q_peak)
    r_sigma_k = float(r_sigma_k0)
    b = float(b0)
    factor = _kappa_over_kh_from_b(b)
    k_h_over_k = float(lowq_kappa) / max(mean_k * factor, np.finfo(float).tiny)
    k_h_over_k = float(np.clip(k_h_over_k, 0.005, 0.5))
    p_h = _normal_sf(b)
    mu2 = mean_k * mean_k * (1.0 + r_sigma_k * r_sigma_k)
    rho0 = mu2 / (3.0 * np.pi)
    scale = float(highq_coefficient) / max(np.pi * p_h * rho0, np.finfo(float).tiny)
    return {
        "scale": max(scale, np.finfo(float).tiny),
        "mean_k": mean_k,
        "r_sigma_k": r_sigma_k,
        "k_H_over_k": k_h_over_k,
        "b": b,
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


def evaluate_heterogeneous_line_guess(
    observation: RadialProfile,
    *,
    q_bounds: tuple[float, float],
    parameters: Mapping[str, float],
    model_settings: Mapping[str, float | int | str | bool | None] | None = None,
    log_error_floor: float = 0.04,
    regression_loss: str | None = None,
) -> HeterogeneousFitResult:
    """Evaluate one heterogeneous-line trial and solve only the scale factor."""

    rls = _import_line_scattering()
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

    mean_k = float(parameters["mean_k"])
    r_sigma_k = float(parameters["r_sigma_k"])
    k_h_over_k = float(parameters["k_H_over_k"])
    b = float(parameters["b"])
    q_min_factor = max(0.5 * q_min / mean_k, 1.0e-5)
    q_max_factor = max(1.3 * q_max / mean_k, q_min_factor * 2.0)
    line_kwargs = {
        "k0_nominal": mean_k,
        "r_sigma_k": r_sigma_k,
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
        "progress": bool(model_settings.get("progress", False)),
    }
    line = rls.compute_uniform_line_scattering(**line_kwargs)
    k_h = k_h_over_k * float(getattr(line, "uniform_meta", {}).get("k_mean", mean_k))
    hetero = rls.heterogeneous_line_scattering(line, k_H=k_h, b=b, return_components=True)
    raw = np.interp(np.log(q_obs), np.log(hetero.Q_grid), hetero.I_h)
    raw = np.maximum(raw, np.finfo(float).tiny)
    loss_mode = str(regression_loss or model_settings.get("regression_loss", "log")).lower()
    if loss_mode == "relative":
        err_abs = np.maximum(e_obs, float(log_error_floor) * np.maximum(np.abs(i_obs), np.finfo(float).tiny))
        weights = 1.0 / (err_abs * err_abs)
        scale = float(np.sum(weights * raw * i_obs) / max(np.sum(weights * raw * raw), np.finfo(float).tiny))
        scale = max(scale, np.finfo(float).tiny)
        model = scale * raw
        residual = (model - i_obs) / err_abs
    elif loss_mode == "log":
        err_log = np.maximum(e_obs / np.maximum(i_obs, np.finfo(float).tiny), float(log_error_floor))
        weights = 1.0 / (err_log * err_log)
        log_scale = float(np.sum(weights * (np.log(i_obs) - np.log(raw))) / np.sum(weights))
        scale = float(np.exp(log_scale))
        model = scale * raw
        residual = (np.log(model) - np.log(i_obs)) / err_log
    else:
        raise ValueError("regression_loss must be 'relative' or 'log'.")
    params = {
        "scale": scale,
        "mean_k": mean_k,
        "r_sigma_k": r_sigma_k,
        "k_H_over_k": k_h_over_k,
        "b": b,
        "k_H": k_h,
        "p_H": float(hetero.p_H),
        "sigma_H_squared": float(hetero.sigma_H_squared),
        "alpha_H": float(hetero.alpha_H),
        "kappa_H": float(hetero.kappa_H),
        "rho0": float(hetero.rho0),
        "highq_coefficient": scale * np.pi * float(hetero.p_H) * float(hetero.rho0),
    }
    return HeterogeneousFitResult(
        parameters=params,
        free_parameters=np.asarray([mean_k, r_sigma_k, k_h_over_k, b], dtype=float),
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


def fit_heterogeneous_line_least_squares(
    observation: RadialProfile,
    *,
    q_bounds: tuple[float, float],
    initial: Mapping[str, float],
    lowq_kappa_anchor: float | None = None,
    highq_coefficient_anchor: float | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    model_settings: Mapping[str, float | int | str | bool | None] | None = None,
    max_nfev: int = 12,
    anchor_weight: float = 2.0,
    log_error_floor: float = 0.03,
    regression_loss: str | None = None,
    verbose: int = 0,
) -> HeterogeneousFitResult:
    """Constrained first-pass fit of the smooth heterogeneous-line model.

    The nonlinear search uses four active parameters: ``mean_k``,
    ``r_sigma_k``, ``k_H_over_k``, and ``b``. The overall scale is solved at
    each trial by weighted log-amplitude matching, then high- and low-Q anchor
    residuals are added as soft constraints. Set ``regression_loss="relative"``
    to fit exact relative residuals instead of log residuals.
    """

    rls = _import_line_scattering()
    bounds = dict(bounds or {})
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
        raise ValueError("Need at least eight positive observation points for constrained fitting.")

    names = ("mean_k", "r_sigma_k", "k_H_over_k", "b")
    default_bounds = {
        "mean_k": (0.03, 0.3),
        "r_sigma_k": (0.03, 0.8),
        "k_H_over_k": (0.005, 0.5),
        "b": (-3.0, 1.5),
    }
    lower = np.array([bounds.get(name, default_bounds[name])[0] for name in names], dtype=float)
    upper = np.array([bounds.get(name, default_bounds[name])[1] for name in names], dtype=float)
    x0 = np.array([float(initial[name]) for name in names], dtype=float)
    x0 = np.minimum(np.maximum(x0, lower + 1.0e-12), upper - 1.0e-12)
    cache: dict[tuple[float, float, float, float], tuple[np.ndarray, object, object]] = {}

    def evaluate_raw(params: np.ndarray) -> tuple[np.ndarray, object, object]:
        mean_k, r_sigma_k, k_h_over_k, b = map(float, params)
        key = tuple(np.round(params, 10))
        if key in cache:
            return cache[key]
        q_min_factor = max(0.5 * q_min / mean_k, 1.0e-5)
        q_max_factor = max(1.3 * q_max / mean_k, q_min_factor * 2.0)
        line_kwargs = {
            "k0_nominal": mean_k,
            "r_sigma_k": r_sigma_k,
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
            "progress": bool(model_settings.get("progress", False)),
        }
        line = rls.compute_uniform_line_scattering(**line_kwargs)
        k_h = k_h_over_k * float(getattr(line, "uniform_meta", {}).get("k_mean", mean_k))
        hetero = rls.heterogeneous_line_scattering(line, k_H=k_h, b=b, return_components=True)
        model_grid = np.interp(np.log(q_obs), np.log(hetero.Q_grid), hetero.I_h)
        cache[key] = (model_grid, line, hetero)
        return cache[key]

    loss_mode = str(regression_loss or model_settings.get("regression_loss", "log")).lower()
    if loss_mode not in {"relative", "log"}:
        raise ValueError("regression_loss must be 'relative' or 'log'.")

    def scaled_model_and_residual(params: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        raw, line, hetero = evaluate_raw(params)
        raw = np.maximum(raw, np.finfo(float).tiny)
        if loss_mode == "relative":
            err_abs = np.maximum(e_obs, float(log_error_floor) * np.maximum(np.abs(i_obs), np.finfo(float).tiny))
            weights = 1.0 / (err_abs * err_abs)
            scale = float(np.sum(weights * raw * i_obs) / max(np.sum(weights * raw * raw), np.finfo(float).tiny))
            scale = max(scale, np.finfo(float).tiny)
            residual = (scale * raw - i_obs) / err_abs
        else:
            err_log = np.maximum(e_obs / np.maximum(i_obs, np.finfo(float).tiny), float(log_error_floor))
            weights = 1.0 / (err_log * err_log)
            log_scale = float(np.sum(weights * (np.log(i_obs) - np.log(raw))) / np.sum(weights))
            scale = float(np.exp(log_scale))
            residual = (np.log(scale * raw) - np.log(i_obs)) / err_log
        anchor_residuals: list[float] = []
        if highq_coefficient_anchor is not None:
            p_h = float(hetero.p_H)
            rho0 = float(hetero.rho0)
            coeff = scale * np.pi * p_h * rho0
            anchor_residuals.append(float(anchor_weight) * np.log(coeff / float(highq_coefficient_anchor)))
        if lowq_kappa_anchor is not None and float(hetero.kappa_H) > 0.0:
            anchor_residuals.append(float(anchor_weight) * np.log(float(hetero.kappa_H) / float(lowq_kappa_anchor)))
        if anchor_residuals:
            residual = np.concatenate([residual, np.asarray(anchor_residuals, dtype=float)])
        return scale, scale * raw, residual

    def residual_fn(params: np.ndarray) -> np.ndarray:
        _, _, residual = scaled_model_and_residual(params)
        if not np.all(np.isfinite(residual)):
            return np.full(q_obs.size + 2, 1.0e6, dtype=float)
        return residual

    opt = least_squares(
        residual_fn,
        x0,
        bounds=(lower, upper),
        max_nfev=int(max_nfev),
        verbose=int(verbose),
    )
    scale, model, residual = scaled_model_and_residual(opt.x)
    raw, line, hetero = evaluate_raw(opt.x)
    mean_k, r_sigma_k, k_h_over_k, b = map(float, opt.x)
    params = {
        "scale": scale,
        "mean_k": mean_k,
        "r_sigma_k": r_sigma_k,
        "k_H_over_k": k_h_over_k,
        "b": b,
        "k_H": float(hetero.kappa_H) / max(_kappa_over_kh_from_b(b), np.finfo(float).tiny),
        "p_H": float(hetero.p_H),
        "sigma_H_squared": float(hetero.sigma_H_squared),
        "alpha_H": float(hetero.alpha_H),
        "kappa_H": float(hetero.kappa_H),
        "rho0": float(hetero.rho0),
        "highq_coefficient": scale * np.pi * float(hetero.p_H) * float(hetero.rho0),
    }
    return HeterogeneousFitResult(
        parameters=params,
        free_parameters=opt.x,
        residual=residual,
        q=q_obs,
        intensity=i_obs,
        error=e_obs,
        model=model,
        unscaled_model=raw,
        success=bool(opt.success),
        message=str(opt.message),
        cost=float(opt.cost),
        nfev=int(opt.nfev),
    )


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
    k_H = max(k_h_over_k * k_line, np.finfo(float).eps)
    xi_dab = 1.0 / k_H
    lateral_size = float(lateral_size_over_mask_length) / k_H
    thickness = float(thickness_over_mask_length) / k_H

    rng = np.random.default_rng(random_seed)
    x, y, z, X, Y, Z = make_coordinates(nx, ny, nz, lateral_size, lateral_size, thickness)
    spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    k_vectors = sample_isotropic_k_vectors(num_line_modes, k_line, rng)
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
        k_H=k_H,
        lateral_size=lateral_size,
        thickness=thickness,
    )
