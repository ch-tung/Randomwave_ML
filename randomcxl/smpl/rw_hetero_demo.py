"""Utility functions for heterogeneous-mask random-line scattering demos."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

import rw_line_scattering as rls


DEFAULT_K_DISTRIBUTION = rls.DEFAULT_HETERO_K_DISTRIBUTION
DEFAULT_NUM_MODES_K = rls.DEFAULT_HETERO_NUM_MODES_K
DEFAULT_K_SAMPLING = rls.DEFAULT_HETERO_K_SAMPLING
DEFAULT_R_GRID_MODE = rls.DEFAULT_HETERO_R_GRID_MODE
DEFAULT_R_MIN_FACTOR = rls.DEFAULT_HETERO_R_MIN_FACTOR
DEFAULT_R_SPLIT_FACTOR = rls.DEFAULT_HETERO_R_SPLIT_FACTOR
DEFAULT_R_MAX_FACTOR = rls.DEFAULT_HETERO_R_MAX_FACTOR
DEFAULT_NR = rls.DEFAULT_HETERO_NR
DEFAULT_NR_SMALL = rls.DEFAULT_HETERO_NR_SMALL
DEFAULT_TAIL_START_FRACTION = rls.DEFAULT_HETERO_TAIL_START_FRACTION
DEFAULT_N_SAMP = rls.DEFAULT_HETERO_N_SAMP
DEFAULT_USE_QMC = rls.DEFAULT_HETERO_USE_QMC
DEFAULT_JACOBIAN_METHOD = rls.DEFAULT_HETERO_JACOBIAN_METHOD
DEFAULT_USE_ASYMPTOTIC = rls.DEFAULT_HETERO_USE_ASYMPTOTIC
DEFAULT_LOWQ_FIT_BOUNDS_OVER_K_EFF = rls.DEFAULT_HETERO_LOWQ_FIT_BOUNDS_OVER_K_EFF
DEFAULT_LOWQ_REPLACE_MAX_OVER_K_EFF = rls.DEFAULT_HETERO_LOWQ_REPLACE_MAX_OVER_K_EFF
DEFAULT_N_SAMP_ST = rls.DEFAULT_HETERO_N_SAMP_ST
DEFAULT_HETERO_LINE_SCATTERING_SETTINGS = rls.DEFAULT_HETERO_LINE_SCATTERING_SETTINGS


def _as_float_tuple(values: float | tuple[float, ...] | list[float] | np.ndarray) -> tuple[float, ...]:
    """Normalize a scalar or one-dimensional sequence to a float tuple."""

    arr = np.atleast_1d(np.asarray(values, dtype=float))
    if arr.ndim != 1:
        raise ValueError("Expected a scalar or one-dimensional sequence.")
    return tuple(float(v) for v in arr)


def _mean_k_from_line_result(line_result: rls.LineScatteringSpectrum) -> float:
    meta = getattr(line_result, "uniform_meta", {})
    if "k_mean" in meta:
        return float(meta["k_mean"])
    if line_result.mu2 is not None:
        return float(np.sqrt(line_result.mu2))
    return float(np.sqrt(3.0 * np.pi * line_result.rho0))


def make_demo_k_spectrum(
    *,
    k0_nominal: float,
    k_distribution: str = DEFAULT_K_DISTRIBUTION,
    r_sigma_k: float,
    k_distribution_params: Mapping[str, float] | None = None,
    random_seed: int,
    num_modes_k: int = DEFAULT_NUM_MODES_K,
    k_sampling: str = DEFAULT_K_SAMPLING,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float | str]]:
    """Construct the radial line-wave spectrum used by the hetero demo."""

    return rls.make_radial_line_spectrum(
        k0_nominal=k0_nominal,
        k_distribution=k_distribution,  # type: ignore[arg-type]
        num_modes_k=num_modes_k,
        r_sigma_k=r_sigma_k,
        k_distribution_params=k_distribution_params,
        random_seed=random_seed,
        k_sampling=k_sampling,  # type: ignore[arg-type]
    )


def compute_uniform_line_scattering(
    *,
    k0_nominal: float,
    k_distribution: str = DEFAULT_K_DISTRIBUTION,
    r_sigma_k: float,
    k_distribution_params: Mapping[str, float] | None = None,
    random_seed: int,
    num_modes_k: int = DEFAULT_NUM_MODES_K,
    k_sampling: str = DEFAULT_K_SAMPLING,
    Nr: int = DEFAULT_NR,
    NQ: int = 256,
    r_min_factor: float = DEFAULT_R_MIN_FACTOR,
    r_max_factor: float = DEFAULT_R_MAX_FACTOR,
    Q_min_factor: float = 0.05,
    Q_max_factor: float = 20.0,
    N_samp_U: int = DEFAULT_N_SAMP,
    N_samp_st: int = DEFAULT_N_SAMP_ST,
    r_grid_mode: str = DEFAULT_R_GRID_MODE,
    r_split_factor: float | None = DEFAULT_R_SPLIT_FACTOR,
    Nr_small: int | None = DEFAULT_NR_SMALL,
    jacobian_method: str = DEFAULT_JACOBIAN_METHOD,
    use_asymptotic: bool = DEFAULT_USE_ASYMPTOTIC,
    lowq_fit_bounds: tuple[float | None, float | None] | None = None,
    lowq_replace_max: float | None = None,
    lowq_fit_bounds_over_k: tuple[float | None, float | None] | None = None,
    lowq_replace_max_over_k: float | None = None,
    lowq_fit_bounds_over_k_eff: tuple[float | None, float | None] | None = DEFAULT_LOWQ_FIT_BOUNDS_OVER_K_EFF,
    lowq_replace_max_over_k_eff: float | None = DEFAULT_LOWQ_REPLACE_MAX_OVER_K_EFF,
    tail_start_fraction: float = DEFAULT_TAIL_START_FRACTION,
    use_qmc: bool = DEFAULT_USE_QMC,
    progress: bool = True,
) -> rls.LineScatteringSpectrum:
    """Compute the uniform-line scattering curve inside the demo workflow."""

    return rls.compute_uniform_line_scattering(
        k0_nominal=k0_nominal,
        k_distribution=k_distribution,
        r_sigma_k=r_sigma_k,
        k_distribution_params=k_distribution_params,
        random_seed=random_seed,
        num_modes_k=num_modes_k,
        k_sampling=k_sampling,
        Nr=Nr,
        NQ=NQ,
        r_min_factor=r_min_factor,
        r_max_factor=r_max_factor,
        Q_min_factor=Q_min_factor,
        Q_max_factor=Q_max_factor,
        N_samp_U=N_samp_U,
        N_samp_st=N_samp_st,
        r_grid_mode=r_grid_mode,
        r_split_factor=r_split_factor,
        Nr_small=Nr_small,
        jacobian_method=jacobian_method,  # type: ignore[arg-type]
        use_asymptotic=use_asymptotic,
        lowq_fit_bounds=lowq_fit_bounds,
        lowq_replace_max=lowq_replace_max,
        lowq_fit_bounds_over_k=lowq_fit_bounds_over_k,
        lowq_replace_max_over_k=lowq_replace_max_over_k,
        lowq_fit_bounds_over_k_eff=lowq_fit_bounds_over_k_eff,
        lowq_replace_max_over_k_eff=lowq_replace_max_over_k_eff,
        tail_start_fraction=tail_start_fraction,
        use_qmc=use_qmc,
        progress=progress,
    )


def evaluate_heterogeneous_case(
    line_result: rls.LineScatteringSpectrum,
    *,
    k_h_over_k: float,
    b: float,
    q_max: float | None = None,
) -> dict[str, float | rls.HeterogeneousLineScatteringResult | np.ndarray]:
    """Evaluate one heterogeneous-mask case using the line-result mean-k convention."""

    k = _mean_k_from_line_result(line_result)
    if q_max is None:
        q_max = float(np.max(line_result.Q_grid))
    k_h = float(k_h_over_k) * k
    smooth = rls.heterogeneous_line_scattering(
        line_result,
        k_h,
        float(b),
        approximation="smooth",
        q_max=q_max,
        return_components=True,
    )
    high_q = rls.heterogeneous_line_scattering(
        line_result,
        k_h,
        float(b),
        approximation="high_q",
        q_max=q_max,
    )
    return {
        "k": k,
        "k_H": k_h,
        "k_H_over_k": float(k_h_over_k),
        "b": float(b),
        "smooth": smooth,
        "high_q": high_q,
    }


def evaluate_heterogeneous_grid(
    line_result: rls.LineScatteringSpectrum,
    *,
    k_h_over_k_values: float | tuple[float, ...] | list[float] | np.ndarray,
    b_values: float | tuple[float, ...] | list[float] | np.ndarray,
    q_max: float | None = None,
) -> dict[str, np.ndarray | float]:
    """Evaluate all heterogeneous cases using one reused uniform-line result."""

    kh_values = _as_float_tuple(k_h_over_k_values)
    b_input_values = _as_float_tuple(b_values)
    meta = getattr(line_result, "uniform_meta", {})
    k = _mean_k_from_line_result(line_result)
    if q_max is None:
        # Use the full available uniform I_L range for the numerical part and
        # the known pi*rho0/Q tail only above the plotted/calculated window.
        q_max = float(np.max(line_result.Q_grid))

    q = np.asarray(line_result.Q_grid, dtype=float)
    shape = (len(kh_values), len(b_input_values), q.size)
    smooth = np.empty(shape, dtype=float)
    high_q = np.empty(shape, dtype=float)
    weighted_line = np.empty(shape, dtype=float)
    smoothed_line = np.empty(shape, dtype=float)
    mask_only = np.empty(shape, dtype=float)
    p_h = np.empty((len(kh_values), len(b_input_values)), dtype=float)
    sigma2 = np.empty_like(p_h)
    alpha = np.empty_like(p_h)
    kappa = np.empty_like(p_h)
    k_h = np.array(kh_values, dtype=float) * k
    b_arr = np.array(b_input_values, dtype=float)
    q_cut = np.empty_like(p_h)

    for i, kh_ratio in enumerate(kh_values):
        for j, b_value in enumerate(b_arr):
            case = evaluate_heterogeneous_case(
                line_result,
                k_h_over_k=float(kh_ratio),
                b=float(b_value),
                q_max=q_max,
            )
            result = case["smooth"]
            high_q_curve = case["high_q"]
            smooth[i, j] = result.I_h
            high_q[i, j] = high_q_curve
            weighted_line[i, j] = result.uniform_component
            smoothed_line[i, j] = result.smoothed_line_component
            mask_only[i, j] = result.mask_component
            p_h[i, j] = result.p_H
            sigma2[i, j] = result.sigma_H_squared
            alpha[i, j] = result.alpha_H
            kappa[i, j] = result.kappa_H
            q_cut[i, j] = result.q_max

    return {
        "Q": q,
        "I_L": np.asarray(line_result.I_L, dtype=float),
        "I_L_original": np.asarray(
            getattr(line_result, "I_L_original", np.asarray(line_result.I_L, dtype=float)),
            dtype=float,
        ),
        "I_L_lowQ_asymptotic": np.asarray(
            getattr(line_result, "I_L_lowQ_asymptotic", np.asarray(line_result.I_L, dtype=float)),
            dtype=float,
        ),
        "lowQ_fit_mask": np.asarray(
            getattr(line_result, "lowQ_fit_mask", np.zeros_like(q, dtype=bool)),
            dtype=bool,
        ),
        "lowQ_replaced_mask": np.asarray(
            getattr(line_result, "lowQ_replaced_mask", np.zeros_like(q, dtype=bool)),
            dtype=bool,
        ),
        "lowQ_I0": float(getattr(line_result, "lowQ_I0", np.nan)),
        "lowQ_I2": float(getattr(line_result, "lowQ_I2", np.nan)),
        "lowQ_fit_min": float(getattr(line_result, "lowQ_fit_min", np.nan)),
        "lowQ_fit_max": float(getattr(line_result, "lowQ_fit_max", np.nan)),
        "lowQ_replace_max": float(getattr(line_result, "lowQ_replace_max", np.nan)),
        "lowQ_relative_rmse": float(getattr(line_result, "lowQ_relative_rmse", np.nan)),
        "use_asymptotic": bool(getattr(line_result, "use_asymptotic", False)),
        "tail_start_fraction": float(getattr(line_result, "tail_start_fraction", np.nan)),
        "use_qmc": bool(getattr(line_result, "use_qmc", DEFAULT_USE_QMC)),
        "r_grid": np.asarray(getattr(line_result, "r_grid", np.array([], dtype=float)), dtype=float),
        "M_J": np.asarray(getattr(line_result, "M_J", np.array([], dtype=float)), dtype=float),
        "C_L": np.asarray(getattr(line_result, "C_L", np.array([], dtype=float)), dtype=float),
        "I_h_smooth": smooth,
        "I_h_highQ": high_q,
        "weighted_line_component": weighted_line,
        "smoothed_line_component": smoothed_line,
        "mask_only_component": mask_only,
        "k_H_over_k": np.array(kh_values, dtype=float),
        "k_H": k_h,
        "b": b_arr,
        "p_H": p_h,
        "sigma_H_squared": sigma2,
        "alpha_H": alpha,
        "kappa_H": kappa,
        "rho0": float(line_result.rho0),
        "mu2": float(line_result.mu2) if line_result.mu2 is not None else np.nan,
        "k": k,
        "k_mean": k,
        "k_eff": float(meta.get("k_eff", np.sqrt(line_result.mu2) if line_result.mu2 is not None else np.nan)),
        "Q_max": q_cut,
        "k0_nominal": float(meta.get("k0_nominal", np.nan)),
        "r_sigma_k": float(meta.get("r_sigma_k", np.nan)),
        "num_modes_k": int(meta.get("num_modes_k", -1)),
        "random_seed": int(meta.get("random_seed", -1)),
        "k_distribution": str(meta.get("k_distribution", "")),
        "k_sampling": str(meta.get("k_sampling", "")),
    }


def save_results(data: dict[str, np.ndarray | float], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "rw_hetero_demo_data.npz"
    np.savez_compressed(path, **data)
    save_parameter_tables(data, output_dir / "tables")
    save_baseline_report(data, output_dir / "baseline_report.txt")
    return path


def save_baseline_report(data: dict[str, np.ndarray | float], path: Path) -> None:
    report = baseline_report(data)
    k_mean = float(data["k_mean"])
    lines = [
        "baseline k_H/<k>=0.05, b=-1",
        f"p_H = {report['p_H']:.12g}",
        f"sigma_H2 = {report['sigma_H2']:.12g}",
        f"alpha_H = {report['alpha_H']:.12g}",
        f"kappa_H = {report['kappa_H']:.12g}",
        f"rho0 = {report['rho0']:.12g}",
        f"min relative highQ/smooth difference = {report['min_rel_diff']:.12g}",
        f"Q/<k> at min difference = {report['q_at_min_rel_diff'] / k_mean:.12g}",
    ]
    for pct in (10, 5, 1):
        start = report[f"q_range_for_{pct}pct_start"]
        end = report[f"q_range_for_{pct}pct_end"]
        if np.isfinite(start) and np.isfinite(end):
            lines.append(f"{pct}% range in Q/<k> = {start / k_mean:.12g} to {end / k_mean:.12g}")
        else:
            lines.append(f"{pct}% range in Q/<k> = not reached")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_value_for_filename(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return text


def save_parameter_tables(data: dict[str, np.ndarray | float], output_dir: Path) -> None:
    """Save one machine-readable CSV table for each heterogeneity parameter set."""

    output_dir.mkdir(parents=True, exist_ok=True)
    q = np.asarray(data["Q"], dtype=float)
    i_l = np.asarray(data["I_L"], dtype=float)
    kh_over_k = np.asarray(data["k_H_over_k"], dtype=float)
    k_h = np.asarray(data["k_H"], dtype=float)
    b_values = np.asarray(data["b"], dtype=float)
    rho0 = float(data["rho0"])
    smooth = np.asarray(data["I_h_smooth"], dtype=float)
    high_q = np.asarray(data["I_h_highQ"], dtype=float)
    weighted = np.asarray(data["weighted_line_component"], dtype=float)
    smoothed = np.asarray(data["smoothed_line_component"], dtype=float)
    mask_only = np.asarray(data["mask_only_component"], dtype=float)
    p_h = np.asarray(data["p_H"], dtype=float)
    sigma2 = np.asarray(data["sigma_H_squared"], dtype=float)
    alpha = np.asarray(data["alpha_H"], dtype=float)
    kappa = np.asarray(data["kappa_H"], dtype=float)
    q_max = np.asarray(data["Q_max"], dtype=float)

    header = (
        "Q,I_L,weighted_line,smoothed_line,mask_only,I_h_smooth,I_h_highQ,"
        "I_h_over_I_L,Q_I_h_over_pi_rho0,k_H,b,p_H,sigma_H2,alpha_H,kappa_H,rho0,Q_max"
    )
    for i, kh_ratio in enumerate(kh_over_k):
        for j, b_value in enumerate(b_values):
            table = np.column_stack(
                (
                    q,
                    i_l,
                    weighted[i, j],
                    smoothed[i, j],
                    mask_only[i, j],
                    smooth[i, j],
                    high_q[i, j],
                    smooth[i, j] / i_l,
                    q * smooth[i, j] / (np.pi * rho0),
                    np.full_like(q, k_h[i]),
                    np.full_like(q, b_value),
                    np.full_like(q, p_h[i, j]),
                    np.full_like(q, sigma2[i, j]),
                    np.full_like(q, alpha[i, j]),
                    np.full_like(q, kappa[i, j]),
                    np.full_like(q, rho0),
                    np.full_like(q, q_max[i, j]),
                )
            )
            filename = (
                f"hetero_kH_over_k_{_safe_value_for_filename(kh_ratio)}"
                f"_b_{_safe_value_for_filename(b_value)}.csv"
            )
            np.savetxt(output_dir / filename, table, delimiter=",", header=header, comments="")


def high_q_thresholds(q: np.ndarray, smooth: np.ndarray, high_q: np.ndarray) -> dict[str, float]:
    """Return longest computed Q interval where high-Q and smooth agree."""

    q = np.asarray(q, dtype=float)
    smooth = np.asarray(smooth, dtype=float)
    high_q = np.asarray(high_q, dtype=float)
    finite = (q > 0.0) & (smooth > 0.0) & np.isfinite(high_q)
    qv = q[finite]
    rel = np.abs(high_q[finite] / smooth[finite] - 1.0)
    thresholds: dict[str, float] = {}
    thresholds["min_rel_diff"] = float(np.min(rel)) if rel.size else float("nan")
    thresholds["q_at_min_rel_diff"] = float(qv[np.argmin(rel)]) if rel.size else float("nan")
    for tol in (0.10, 0.05, 0.01):
        good = rel < tol
        key = f"q_range_for_{int(100 * tol)}pct"
        if not np.any(good):
            thresholds[f"{key}_start"] = float("nan")
            thresholds[f"{key}_end"] = float("nan")
            continue
        starts = np.flatnonzero(good & np.r_[True, ~good[:-1]])
        ends = np.flatnonzero(good & np.r_[~good[1:], True])
        lengths = qv[ends] - qv[starts]
        best = int(np.argmax(lengths))
        thresholds[f"{key}_start"] = float(qv[starts[best]])
        thresholds[f"{key}_end"] = float(qv[ends[best]])
    return thresholds


def baseline_report(data: dict[str, np.ndarray | float]) -> dict[str, float]:
    """Report the requested scalar values for k_H/<k>=0.05 and b=-1."""

    kh_over_k = np.asarray(data["k_H_over_k"], dtype=float)
    b_values = np.asarray(data["b"], dtype=float)
    i = int(np.argmin(np.abs(kh_over_k - 0.05)))
    j = int(np.argmin(np.abs(b_values - (-1.0))))
    thresholds = high_q_thresholds(
        np.asarray(data["Q"], dtype=float),
        np.asarray(data["I_h_smooth"], dtype=float)[i, j],
        np.asarray(data["I_h_highQ"], dtype=float)[i, j],
    )
    report = {
        "k_H_over_k": float(kh_over_k[i]),
        "b": float(b_values[j]),
        "p_H": float(np.asarray(data["p_H"])[i, j]),
        "sigma_H2": float(np.asarray(data["sigma_H_squared"])[i, j]),
        "alpha_H": float(np.asarray(data["alpha_H"])[i, j]),
        "kappa_H": float(np.asarray(data["kappa_H"])[i, j]),
        "rho0": float(data["rho0"]),
    }
    report.update(thresholds)
    return report


def _case_label(kh_over_k: float, b_value: float, p_h: float) -> str:
    return rf"$k_H/\langle k\rangle={kh_over_k:g}$, $b={b_value:g}$, $p_H={p_h:.3g}$"


def make_plots(data: dict[str, np.ndarray | float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    q = np.asarray(data["Q"], dtype=float)
    k = float(data["k"])
    q_over_k = q / k
    i_l = np.asarray(data["I_L"], dtype=float)
    rho0 = float(data["rho0"])
    kh_over_k = np.asarray(data["k_H_over_k"], dtype=float)
    b_values = np.asarray(data["b"], dtype=float)
    p_h = np.asarray(data["p_H"], dtype=float)
    smooth = np.asarray(data["I_h_smooth"], dtype=float)
    high_q = np.asarray(data["I_h_highQ"], dtype=float)
    weighted = np.asarray(data["weighted_line_component"], dtype=float)
    smoothed = np.asarray(data["smoothed_line_component"], dtype=float)
    mask_only = np.asarray(data["mask_only_component"], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.loglog(q_over_k, i_l, color="black", lw=1.4, label=r"$I_L(Q)$")
    for i, kh in enumerate(kh_over_k):
        for j, b_value in enumerate(b_values):
            ax.loglog(q_over_k, smooth[i, j], lw=1.0, label=_case_label(kh, b_value, p_h[i, j]))
    ax.set_xlabel(r"$Q/\langle k\rangle$")
    ax.set_ylabel(r"$I(Q)$")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "hetero_IQ_all_cases.png", dpi=180)
    plt.close(fig)

    rep_i = min(1, len(kh_over_k) - 1)
    rep_j = min(1, len(b_values) - 1)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.loglog(q_over_k, i_l, color="black", lw=1.2, label=r"$I_L(Q)$")
    ax.loglog(q_over_k, smooth[rep_i, rep_j], color="tab:red", lw=1.4, label=r"$I_h^{smooth}(Q)$")
    high_ratio_finite = np.isfinite(high_q[rep_i, rep_j]) & (high_q[rep_i, rep_j] > 0.0)
    ax.loglog(
        q_over_k[high_ratio_finite],
        high_q[rep_i, rep_j, high_ratio_finite],
        color="tab:red",
        lw=1.0,
        ls="--",
        label=r"$I_h^{highQ}(Q)$",
    )
    ax.set_xlabel(r"$Q/\langle k\rangle$")
    ax.set_ylabel(r"$I(Q)$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "hetero_selected_full_IQ.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.loglog(q_over_k, weighted[rep_i, rep_j], label=r"$p_H^2 I_L(Q)$")
    ax.loglog(q_over_k, smoothed[rep_i, rep_j], label=r"$\sigma_H^2\mathcal{A}_{\kappa_H}[I_L](Q)$")
    ax.loglog(q_over_k, mask_only[rep_i, rep_j], label=r"$\rho_0^2\sigma_H^2K_H(Q)$")
    ax.loglog(q_over_k, smooth[rep_i, rep_j], color="black", lw=1.2, label=r"$I_h^{smooth}(Q)$")
    ax.set_xlabel(r"$Q/\langle k\rangle$")
    ax.set_ylabel(r"$I(Q)$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "hetero_smooth_components.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for i, kh in enumerate(kh_over_k):
        for j, b_value in enumerate(b_values):
            ax.semilogx(q_over_k, q * smooth[i, j] / (np.pi * rho0), lw=1.0, label=_case_label(kh, b_value, p_h[i, j]))
    for p_value in np.unique(np.round(p_h, 12)):
        ax.axhline(p_value, color="black", lw=0.6, ls=":", alpha=0.45)
    ax.set_xlabel(r"$Q/\langle k\rangle$")
    ax.set_ylabel(r"$Q I_h(Q)/(\pi\rho_0)$")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "hetero_QI_over_pi_rho0.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for i, kh in enumerate(kh_over_k):
        for j, b_value in enumerate(b_values):
            ax.semilogx(q_over_k, smooth[i, j] / i_l, lw=1.0, label=_case_label(kh, b_value, p_h[i, j]))
    ax.set_xlabel(r"$Q/\langle k\rangle$")
    ax.set_ylabel(r"$I_h(Q)/I_L(Q)$")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "hetero_Ih_over_IL.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for i, kh in enumerate(kh_over_k):
        for j, b_value in enumerate(b_values):
            ratio = high_q[i, j] / smooth[i, j]
            finite = np.isfinite(ratio) & (smooth[i, j] > 0.0)
            ax.semilogx(q_over_k[finite], ratio[finite], lw=1.0, label=_case_label(kh, b_value, p_h[i, j]))
    ax.axhline(1.0, color="black", lw=0.8, ls=":")
    ax.set_xlabel(r"$Q/\langle k\rangle$")
    ax.set_ylabel(r"$I_h^{highQ}(Q)/I_h^{smooth}(Q)$")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "hetero_highQ_ratio.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.8), sharex=True)
    axes[0].semilogx(q_over_k, smooth[rep_i, rep_j] / i_l, color="tab:red", lw=1.2)
    axes[0].set_ylabel(r"$I_h(Q)/I_L(Q)$")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].semilogx(q_over_k, q * smooth[rep_i, rep_j] / (np.pi * rho0), color="tab:red", lw=1.2)
    axes[1].axhline(p_h[rep_i, rep_j], color="black", lw=0.8, ls=":", label=r"$p_H$")
    axes[1].set_xlabel(r"$Q/\langle k\rangle$")
    axes[1].set_ylabel(r"$Q I_h(Q)/(\pi\rho_0)$")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "hetero_baseline_normalized.png", dpi=180)
    plt.close(fig)

    fixed_b_index = min(1, len(b_values) - 1)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.loglog(q_over_k, i_l, color="black", lw=1.2, label=r"$I_L(Q)$")
    for i, kh in enumerate(kh_over_k):
        ax.loglog(q_over_k, smooth[i, fixed_b_index], lw=1.2, label=rf"$k_H/\langle k\rangle={kh:g}$")
    ax.set_xlabel(r"$Q/\langle k\rangle$")
    ax.set_ylabel(r"$I_h^{smooth}(Q)$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "hetero_vary_kH_fixed_b.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.8), sharex=True)
    for i, kh in enumerate(kh_over_k):
        axes[0].loglog(q_over_k, smoothed[i, fixed_b_index], lw=1.2, label=rf"$k_H/\langle k\rangle={kh:g}$")
        axes[1].loglog(q_over_k, mask_only[i, fixed_b_index], lw=1.2, label=rf"$k_H/\langle k\rangle={kh:g}$")
    axes[0].set_ylabel(r"$\sigma_H^2\mathcal{A}_{\kappa_H}[I_L](Q)$")
    axes[1].set_xlabel(r"$Q/\langle k\rangle$")
    axes[1].set_ylabel(r"$\rho_0^2\sigma_H^2K_H(Q)$")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "hetero_kH_sweep_components.png", dpi=180)
    plt.close(fig)

    fixed_kh_index = min(1, len(kh_over_k) - 1)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.loglog(q_over_k, i_l, color="black", lw=1.2, label=r"$I_L(Q)$")
    for j, b_value in enumerate(b_values):
        ax.loglog(q_over_k, smooth[fixed_kh_index, j], lw=1.2, label=rf"$b={b_value:g}$, $p_H={p_h[fixed_kh_index, j]:.3g}$")
    ax.set_xlabel(r"$Q/\langle k\rangle$")
    ax.set_ylabel(r"$I_h^{smooth}(Q)$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "hetero_vary_b_fixed_kH.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.2), sharex=True)
    for j, b_value in enumerate(b_values):
        label = rf"$b={b_value:g}$, $p_H={p_h[fixed_kh_index, j]:.3g}$"
        axes[0].loglog(q_over_k, weighted[fixed_kh_index, j], lw=1.2, label=label)
        axes[1].loglog(q_over_k, smoothed[fixed_kh_index, j], lw=1.2, label=label)
        axes[2].loglog(q_over_k, mask_only[fixed_kh_index, j], lw=1.2, label=label)
    axes[0].set_ylabel(r"$p_H^2I_L(Q)$")
    axes[1].set_ylabel(r"$\sigma_H^2\mathcal{A}_{\kappa_H}[I_L](Q)$")
    axes[2].set_xlabel(r"$Q/\langle k\rangle$")
    axes[2].set_ylabel(r"$\rho_0^2\sigma_H^2K_H(Q)$")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "hetero_b_sweep_components.png", dpi=180)
    plt.close(fig)



