"""Deterministic checks for heterogeneous random-line scattering."""

from __future__ import annotations

import argparse

import numpy as np
from scipy.integrate import simpson

import rw_line_scattering as rw


def _relative_error(actual: np.ndarray, expected: np.ndarray, floor: float = 1.0e-300) -> np.ndarray:
    return np.abs(np.asarray(actual) - np.asarray(expected)) / np.maximum(np.abs(expected), floor)


def _kernel_norm(kappa: float, x_max: float = 2.0e5, n: int = 200_001) -> float:
    x = np.geomspace(1.0e-8, float(x_max), int(n))
    q = float(kappa) * x
    kernel = rw.smooth_mask_kernel(q, kappa)
    return float(simpson(q * q * kernel, x=q) / (2.0 * np.pi**2))


def _make_powerlaw_line_result(
    rho0: float = 0.37,
    q_min: float = 1.0e-5,
    q_max: float = 1.0e5,
    n: int = 5000,
) -> rw.LineScatteringSpectrum:
    q = np.geomspace(float(q_min), float(q_max), int(n))
    i_l = np.pi * float(rho0) / q
    return rw.make_line_scattering_spectrum(q, i_l, float(rho0))


def _make_representative_line_result(rho0: float = 0.37) -> rw.LineScatteringSpectrum:
    q = np.geomspace(1.0e-4, 1.0e4, 3500)
    line_asymptote = np.pi * float(rho0) / q
    shoulder = 1.0 + 0.35 * np.exp(-0.5 * (np.log(q / 3.0) / 0.55) ** 2)
    low_q_rolloff = 1.0 / (1.0 + (0.03 / q) ** 2)
    i_l = line_asymptote * shoulder * low_q_rolloff
    return rw.make_line_scattering_spectrum(q, i_l, float(rho0))


def _direct_smoothing_operator(
    q_eval: np.ndarray,
    q_grid: np.ndarray,
    i_l: np.ndarray,
    kappa: float,
) -> np.ndarray:
    out = np.empty_like(q_eval, dtype=float)
    for idx, q_value in enumerate(q_eval):
        kernel = 1.0 / ((q_value - q_grid) ** 2 + kappa * kappa)
        kernel -= 1.0 / ((q_value + q_grid) ** 2 + kappa * kappa)
        out[idx] = kappa / (np.pi * q_value) * float(simpson(q_grid * i_l * kernel, x=q_grid))
    return out


def check_kernel_normalization() -> dict[str, float]:
    errors = {}
    for kappa in (0.02, 0.2, 2.0, 20.0):
        norm = _kernel_norm(kappa)
        err = abs(norm - 1.0)
        errors[f"kappa={kappa:g}"] = err
        assert err < 4.0e-5, f"K_H normalization failed for kappa={kappa:g}: norm={norm:.12g}"
    return errors


def check_mask_variance() -> dict[str, float]:
    stats = {}
    for b in (-2.0, 0.0, 1.25):
        params = rw.mask_occupancy_parameters(k_H=0.7, b=b)
        sigma2 = float(params["sigma_H_squared"])
        kappa = float(params["kappa_H"])
        variance = sigma2 * _kernel_norm(kappa)
        err = abs(variance - sigma2)
        stats[f"b={b:g}"] = err
        assert err < 4.0e-5 * max(sigma2, 1.0e-12), (
            f"Mask variance integral failed for b={b:g}: integral={variance:.12g}, "
            f"sigma2={sigma2:.12g}"
        )
    return stats


def check_no_mask_and_empty_limits() -> dict[str, float]:
    line = _make_powerlaw_line_result()
    no_mask = rw.heterogeneous_line_scattering(line, k_H=0.3, b=-50.0, return_components=True)
    empty = rw.heterogeneous_line_scattering(line, k_H=0.3, b=50.0, return_components=True)
    no_mask_err = float(np.max(_relative_error(no_mask.I_h, line.I_L)))
    empty_abs = float(np.max(np.abs(empty.I_h)))
    assert no_mask.p_H == 1.0 and no_mask.sigma_H_squared == 0.0, (
        f"No-mask occupancy failed: p_H={no_mask.p_H}, sigma2={no_mask.sigma_H_squared}"
    )
    assert np.isfinite(no_mask.kappa_H), f"No-mask kappa_H is not finite: {no_mask.kappa_H}"
    assert no_mask_err < 1.0e-14, f"No-mask smooth I_h did not return I_L: relerr={no_mask_err:.4g}"
    assert empty.p_H == 0.0, f"Empty-mask occupancy failed: p_H={empty.p_H}"
    assert np.isfinite(empty.kappa_H), f"Empty-mask kappa_H is not finite: {empty.kappa_H}"
    assert empty_abs == 0.0, f"Empty-mask smooth I_h did not vanish: max_abs={empty_abs:.4g}"
    return {"no_mask_relerr": no_mask_err, "empty_max_abs": empty_abs}


def check_slow_mask_limit() -> dict[str, float]:
    rho0 = 0.37
    q_grid = np.linspace(1.0e-4, 40.0, 40_000)
    i_l = np.pi * rho0 / np.maximum(q_grid, 1.0e-12)
    i_l *= 1.0 + 0.25 * np.exp(-0.5 * (np.log(q_grid / 3.0) / 0.65) ** 2)
    q_eval = np.linspace(0.25, 20.0, 80)
    i_eval = np.interp(q_eval, q_grid, i_l)
    errors = {}
    for k_H in (0.1, 0.03, 0.01):
        params = rw.mask_occupancy_parameters(k_H=k_H, b=0.0)
        A, _ = rw.smoothing_operator_A(q_eval, q_grid, i_l, float(params["kappa_H"]), rho0)
        err = float(np.median(_relative_error(A, i_eval)))
        errors[f"k_H={k_H:g}"] = err
    assert errors["k_H=0.01"] < errors["k_H=0.03"] < errors["k_H=0.1"], (
        f"Slow-mask identity errors are not decreasing: {errors}"
    )
    assert errors["k_H=0.01"] < 0.025, f"Slow-mask identity limit too inaccurate: {errors}"

    line = rw.make_line_scattering_spectrum(q_grid, i_l, rho0)
    result = rw.heterogeneous_line_scattering(line, k_H=0.01, b=0.0, return_components=True)
    result_eval = np.interp(q_eval, result.Q_grid, result.I_h)
    pH_line_err = float(np.median(_relative_error(result_eval, 0.5 * i_eval)))
    assert pH_line_err < 0.035, f"Slow-mask I_h did not approach p_H I_L: relerr={pH_line_err:.4g}"
    errors["pH_line_relerr"] = pH_line_err
    return errors


def check_pure_line_asymptote_identity() -> dict[str, float]:
    line = _make_powerlaw_line_result(q_min=1.0e-7, q_max=1.0e7, n=9000)
    stats = {}
    for kappa in (0.03, 0.3, 3.0):
        q_eval = kappa * np.geomspace(1.0e-2, 1.0e2, 80)
        A, _ = rw.smoothing_operator_A(q_eval, line.Q_grid, line.I_L, kappa, line.rho0)
        exact = 2.0 * line.rho0 / q_eval * np.arctan(q_eval / kappa)
        max_err = float(np.max(_relative_error(A, exact)))
        stats[f"kappa={kappa:g}"] = max_err
        assert max_err < 1.0e-3, (
            f"Pure-line smoothing identity failed for kappa={kappa:g}: max_relerr={max_err:.4g}"
        )
    return stats


def check_tail_correction_convergence() -> dict[str, float]:
    line = _make_representative_line_result()
    params = rw.mask_occupancy_parameters(k_H=0.3, b=0.0)
    kappa = float(params["kappa_H"])
    q_eval = np.geomspace(0.2, 5.0, 60)
    wide_grid = np.geomspace(line.Q_grid[0], 1.0e6, 9000)
    wide_i = np.interp(wide_grid, line.Q_grid, line.I_L)
    wide_i[wide_grid > line.Q_grid[-1]] = np.pi * line.rho0 / wide_grid[wide_grid > line.Q_grid[-1]]
    direct = _direct_smoothing_operator(q_eval, wide_grid, wide_i, kappa)

    ranges = (20.0, 80.0, 300.0)
    corrected_errors = []
    direct_truncated_errors = []
    for q_max in ranges:
        corrected, _ = rw.smoothing_operator_A(q_eval, line.Q_grid, line.I_L, kappa, line.rho0, q_max=q_max)
        finite = line.Q_grid <= q_max
        truncated = _direct_smoothing_operator(q_eval, line.Q_grid[finite], line.I_L[finite], kappa)
        corrected_errors.append(float(np.max(_relative_error(corrected, direct))))
        direct_truncated_errors.append(float(np.max(_relative_error(truncated, direct))))

    assert corrected_errors[1] <= 1.05 * corrected_errors[0], (
        f"Tail-corrected errors did not improve before reaching the quadrature floor: {corrected_errors}"
    )
    assert corrected_errors[1] < 1.0e-4, f"Tail correction at q_max=80 is too inaccurate: {corrected_errors[1]:.4g}"
    assert corrected_errors[1] < 1.0e-4, (
        "Tail-corrected convolution did not reach the wide-range reference: "
        f"corrected={corrected_errors[1]:.4g}, truncated={direct_truncated_errors[1]:.4g}"
    )
    return {
        "q_max_20_corrected": corrected_errors[0],
        "q_max_80_corrected": corrected_errors[1],
        "q_max_300_corrected": corrected_errors[2],
        "q_max_80_truncated": direct_truncated_errors[1],
    }


def check_lowq_extrapolated_convolution() -> dict[str, float]:
    rho0 = 0.37
    i0 = 1.2
    i2 = 0.35
    kappa = 0.05
    q_eval = np.geomspace(0.01, 0.18, 36)
    q_dense = np.linspace(0.0, 50.0, 25_001)
    i_dense = i0 + i2 * q_dense * q_dense
    q_sparse = np.linspace(0.2, 50.0, 4000)
    i_sparse = i0 + i2 * q_sparse * q_sparse

    reference, _ = rw.smoothing_operator_A(q_eval, q_dense, i_dense, kappa, rho0, q_max=50.0)
    extrapolated, _ = rw.smoothing_operator_A(
        q_eval,
        q_sparse,
        i_sparse,
        kappa,
        rho0,
        q_max=50.0,
        lowq_I0=i0,
        lowq_I2=i2,
    )
    truncated, _ = rw.smoothing_operator_A(q_eval, q_sparse, i_sparse, kappa, rho0, q_max=50.0)

    extrapolated_err = float(np.max(_relative_error(extrapolated, reference)))
    truncated_err = float(np.max(_relative_error(truncated, reference)))
    assert extrapolated_err < 3.0e-4, (
        "Low-Q extrapolated convolution did not match the dense reference: "
        f"relerr={extrapolated_err:.4g}"
    )
    assert extrapolated_err < 0.05 * truncated_err, (
        "Low-Q extrapolation did not substantially reduce truncation error: "
        f"extrapolated={extrapolated_err:.4g}, truncated={truncated_err:.4g}"
    )
    return {"extrapolated": extrapolated_err, "truncated": truncated_err}


def check_high_q_coefficient() -> dict[str, float]:
    line = _make_powerlaw_line_result(q_min=1.0e-4, q_max=1.0e7, n=7000)
    result = rw.heterogeneous_line_scattering(line, k_H=0.2, b=-0.4, q_max=100.0, return_components=True)
    q = result.Q_grid
    mask = q > 1.0e4
    target = np.pi * result.p_H * result.rho0
    relerr = float(np.max(np.abs(q[mask] * result.I_h[mask] / target - 1.0)))
    assert relerr < 0.025, f"High-Q coefficient failed: max_relerr={relerr:.4g}"
    return {"target": target, "max_relerr": relerr}


def check_approximation_levels() -> dict[str, float]:
    line = _make_powerlaw_line_result(q_min=1.0e-3, q_max=1.0e6, n=5000)
    result = rw.heterogeneous_line_scattering(line, k_H=0.25, b=-0.2, q_max=100.0, return_components=True)
    q = result.Q_grid
    mask = (q > 0.1) & np.isfinite(result.I_highQ)
    rel = _relative_error(result.I_highQ[mask], result.I_h[mask])
    q_masked = q[mask]
    thresholds = {}
    for tol in (0.10, 0.05, 0.01):
        good_suffix = np.array([np.all(rel[idx:] < tol) for idx in range(rel.size)])
        if np.any(good_suffix):
            thresholds[f"q_min_for_{int(100*tol)}pct"] = float(q_masked[np.argmax(good_suffix)])
        else:
            thresholds[f"q_min_for_{int(100*tol)}pct"] = float("nan")
    assert np.isfinite(thresholds["q_min_for_10pct"]), (
        f"High-Q approximation never reached 10% agreement; thresholds={thresholds}"
    )
    assert thresholds["q_min_for_10pct"] <= thresholds["q_min_for_5pct"] or np.isnan(thresholds["q_min_for_5pct"])
    return thresholds


def check_uniform_line_unchanged() -> dict[str, float]:
    k0 = 1.7
    r = np.geomspace(1.0e-3 / k0, 20.0 / k0, 120)
    q = np.geomspace(0.2 * k0, 50.0 * k0, 160)
    g_before = rw.g_mono(r, k0)
    gp_before = rw.gp_mono(r, k0)
    gpp_before = rw.gpp_mono(r, k0)
    rho0 = k0 * k0 / (3.0 * np.pi)
    i_l = np.pi * rho0 / q
    line = rw.make_line_scattering_spectrum(q.copy(), i_l.copy(), rho0)
    _ = rw.heterogeneous_line_scattering(line, k_H=0.2 * k0, b=-0.3, return_components=True)
    assert np.array_equal(q, line.Q_grid), "heterogeneous layer modified Q_grid in-place."
    assert np.array_equal(i_l, line.I_L), "heterogeneous layer modified I_L in-place."
    assert np.allclose(g_before, rw.g_mono(r, k0)), "g_mono changed after heterogeneous call."
    assert np.allclose(gp_before, rw.gp_mono(r, k0)), "gp_mono changed after heterogeneous call."
    assert np.allclose(gpp_before, rw.gpp_mono(r, k0)), "gpp_mono changed after heterogeneous call."
    expected_rho0 = rw.rho0_from_k_radii(np.array([k0]))
    expected_coeff = np.pi * expected_rho0
    assert abs(expected_rho0 - rho0) < 1.0e-15, f"rho0 changed: {expected_rho0} vs {rho0}"
    assert abs(expected_coeff - k0 * k0 / 3.0) < 1.0e-15, (
        f"Uniform high-Q coefficient changed: {expected_coeff}"
    )
    return {"rho0": rho0, "pi_rho0": expected_coeff}


def check_max_entropy_radial_moments() -> dict[str, float]:
    target_mean = 1.0
    target_std = 0.28
    target_skewness = 0.0
    radii, weights = rw.make_radial_k_quadrature(
        512,
        "max_entropy_radial",
        k0=target_mean,
        sigma_k=target_std,
        distribution_params={
            "r_sigma_k": target_std / target_mean,
            "skewness": target_skewness,
            "support_sigma": 8.0,
        },
    )
    mean = float(np.dot(weights, radii))
    centered = radii - mean
    std = float(np.sqrt(np.dot(weights, centered * centered)))
    skewness = float(np.dot(weights, centered**3) / std**3)
    assert abs(mean - target_mean) < 1.0e-10
    assert abs(std - target_std) < 1.0e-10
    assert abs(skewness - target_skewness) < 1.0e-8
    assert np.all(radii > 0.0)
    return {"mean": mean, "std": std, "skewness": skewness}


def check_nematic_sampled_infinite_baseline() -> dict[str, float]:
    n_samp = 2**10
    random_seed = 12345
    k_radii = np.array([0.9, 1.0, 1.1])
    result = rw.compute_nematic_tangent_correlation(
        np.array([1.0, 50.0]),
        k_radii,
        n_samp,
        use_qmc=True,
        random_seed=random_seed,
        progress=False,
    )
    z_u, z_v = rw.standard_normal_samples(
        n_samp,
        use_qmc=True,
        random_seed=random_seed,
    )
    omega_0 = np.cross(z_u[:, :3], z_v[:, :3])
    omega_r = np.cross(z_u[:, 3:], z_v[:, 3:])
    weight = np.linalg.norm(omega_0, axis=1) * np.linalg.norm(omega_r, axis=1)
    mu = np.divide(
        np.einsum("ij,ij->i", omega_0, omega_r),
        weight,
        out=np.zeros_like(weight),
        where=weight > 0.0,
    )
    p2 = 0.5 * (3.0 * np.clip(mu, -1.0, 1.0) ** 2 - 1.0)
    expected = float(np.sum(weight * p2) / np.sum(weight))
    assert result["K_2_sampled"] is result["K_2"]
    assert abs(float(result["K_2_inf_sampled"]) - expected) < 1.0e-15
    return {
        "K_2_inf_sampled": float(result["K_2_inf_sampled"]),
        "K_2_large_r": float(result["K_2"][-1]),
    }


def run_all_checks() -> dict[str, dict[str, float]]:
    results = {
        "kernel_normalization": check_kernel_normalization(),
        "mask_variance": check_mask_variance(),
        "no_mask_empty_limits": check_no_mask_and_empty_limits(),
        "slow_mask_limit": check_slow_mask_limit(),
        "pure_line_asymptote_identity": check_pure_line_asymptote_identity(),
        "tail_correction_convergence": check_tail_correction_convergence(),
        "lowq_extrapolated_convolution": check_lowq_extrapolated_convolution(),
        "high_q_coefficient": check_high_q_coefficient(),
        "approximation_levels": check_approximation_levels(),
        "uniform_line_unchanged": check_uniform_line_unchanged(),
        "max_entropy_radial_moments": check_max_entropy_radial_moments(),
        "nematic_sampled_infinite_baseline": check_nematic_sampled_infinite_baseline(),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    results = run_all_checks()
    for name, stats in results.items():
        fields = " ".join(f"{key}={value:.6g}" for key, value in stats.items())
        print(f"{name}: passed {fields}")


if __name__ == "__main__":
    main()
