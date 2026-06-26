"""
Scattering spectrum for the line network of two random waves.

The line set is L = {r: psi_1(r)=0, psi_2(r)=0}, where psi_1 and psi_2 are
independent real isotropic monochromatic Gaussian random waves with covariance
g(r)=sin(k0*r)/(k0*r). This script estimates the full line-density correlation
and computes finite-Q isotropic scattering spectra. By default the transform
uses the coherent correlation C_L(r)-rho0^2, because the background plateau
represents only the Q=0 forward-scattering delta peak in the infinite-domain
limit but contaminates finite-window numerical transforms.
"""

from __future__ import annotations

import argparse
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson
from scipy.stats import norm, qmc


DEFAULT_K0 = 1.0
DEFAULT_R_MIN_FACTOR = 1.0e-3
DEFAULT_R_MAX_FACTOR = 300.0
DEFAULT_NR = 3000
DEFAULT_Q_MIN_FACTOR = 1.0e-2
DEFAULT_Q_MAX_FACTOR = 1.0e2
DEFAULT_NQ = 300
DEFAULT_N_SAMP = 2**14
DEFAULT_RANDOM_SEED = 12345
DEFAULT_OUTPUT_DIR = "smpl/rw_line_scattering_output"
DEFAULT_HETERO_K_DISTRIBUTION = "gaussian_radial"
DEFAULT_HETERO_NUM_MODES_K = 2**10
DEFAULT_HETERO_K_SAMPLING = "qmc"
DEFAULT_HETERO_R_GRID_MODE = "mixed"
DEFAULT_HETERO_R_MIN_FACTOR = 1.0e-3
DEFAULT_HETERO_R_SPLIT_FACTOR = 5.0
DEFAULT_HETERO_R_MAX_FACTOR = 250.0
DEFAULT_HETERO_NR = 5000
DEFAULT_HETERO_NR_SMALL = 1200
DEFAULT_HETERO_TAIL_START_FRACTION = 0.8
DEFAULT_HETERO_N_SAMP = 2**14
DEFAULT_HETERO_N_SAMP_ST = 2**8
DEFAULT_HETERO_USE_QMC = True
DEFAULT_HETERO_JACOBIAN_METHOD = "direct_12d"
DEFAULT_HETERO_USE_ASYMPTOTIC = True
DEFAULT_HETERO_LOWQ_FIT_BOUNDS_OVER_K_EFF = (0.35, 0.8)
DEFAULT_HETERO_LOWQ_REPLACE_MAX_OVER_K_EFF = 0.5
DEFAULT_HETERO_LINE_SCATTERING_SETTINGS = {
    "k_distribution": DEFAULT_HETERO_K_DISTRIBUTION,
    "num_modes_k": DEFAULT_HETERO_NUM_MODES_K,
    "k_sampling": DEFAULT_HETERO_K_SAMPLING,
    "r_grid_mode": DEFAULT_HETERO_R_GRID_MODE,
    "r_min_factor": DEFAULT_HETERO_R_MIN_FACTOR,
    "r_split_factor": DEFAULT_HETERO_R_SPLIT_FACTOR,
    "r_max_factor": DEFAULT_HETERO_R_MAX_FACTOR,
    "Nr": DEFAULT_HETERO_NR,
    "Nr_small": DEFAULT_HETERO_NR_SMALL,
    "tail_start_fraction": DEFAULT_HETERO_TAIL_START_FRACTION,
    "N_samp": DEFAULT_HETERO_N_SAMP,
    "N_samp_U": DEFAULT_HETERO_N_SAMP,
    "N_samp_st": DEFAULT_HETERO_N_SAMP_ST,
    "use_qmc": DEFAULT_HETERO_USE_QMC,
    "jacobian_method": DEFAULT_HETERO_JACOBIAN_METHOD,
    "use_asymptotic": DEFAULT_HETERO_USE_ASYMPTOTIC,
    "lowq_fit_bounds_over_k_eff": DEFAULT_HETERO_LOWQ_FIT_BOUNDS_OVER_K_EFF,
    "lowq_replace_max_over_k_eff": DEFAULT_HETERO_LOWQ_REPLACE_MAX_OVER_K_EFF,
}
SMALL_X = 1.0e-4
DEFAULT_RADIAL_CHUNK_SIZE = 256
DEFAULT_CONDITIONAL_U_BATCH_SIZE = 64
DEFAULT_CONDITIONAL_ST_BATCH_SIZE = 256
KDistribution = Literal["single_shell", "gaussian_radial", "uniform_band"]
JacobianMethod = Literal["direct_12d", "conditional_6d_2d"]
SamplingMethod = Literal["qmc", "random"]
STSamplingMethod = Literal["quadrature", "qmc"]
STTransform = Literal["rational", "logistic"]


def qmc_power_from_n_samp(n_samp: int) -> int:
    """Return Sobol ``random_base2`` power for a power-of-two sample count."""

    n_samp = int(n_samp)
    if n_samp <= 0:
        raise ValueError("n_samp must be positive.")
    qmc_power = int(np.log2(n_samp))
    if 2**qmc_power != n_samp:
        raise ValueError("Sobol QMC requires n_samp to be a power of two.")
    return qmc_power


@dataclass(frozen=True)
class FieldKSets:
    """Container for independent sampled k-vector sets for psi_1 and psi_2."""

    psi1: np.ndarray
    psi2: np.ndarray


def _mode_counts(num_modes: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(num_modes, int):
        return (int(num_modes), int(num_modes))
    if len(num_modes) != 2:
        raise ValueError("num_modes must be an integer or a length-2 sequence.")
    return tuple(int(n) for n in num_modes)


def _field_parameter_values(value: float | Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        scalar = float(value)
        return (scalar, scalar)
    if len(value) != 2:
        raise ValueError(f"{name} must be a float or a length-2 sequence.")
    return tuple(float(item) for item in value)


def _isotropic_unit_vectors(count: int, rng: np.random.Generator) -> np.ndarray:
    directions = rng.normal(size=(count, 3))
    norms = np.linalg.norm(directions, axis=1)
    while np.any(norms == 0.0):
        bad = norms == 0.0
        directions[bad] = rng.normal(size=(np.count_nonzero(bad), 3))
        norms = np.linalg.norm(directions, axis=1)
    return directions / norms[:, None]


def _sobol_points(count: int, dim: int, seed: int) -> np.ndarray:
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    m = int(np.log2(count))
    if 2**m == count:
        return sampler.random_base2(m=m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return sampler.random(int(count))


def _qmc_isotropic_unit_vectors(count: int, seed: int) -> np.ndarray:
    u = _sobol_points(count, 2, seed)
    z = 1.0 - 2.0 * u[:, 0]
    phi = 2.0 * np.pi * u[:, 1]
    xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((xy * np.cos(phi), xy * np.sin(phi), z))


def _positive_gaussian_radii_from_unit(
    u: np.ndarray,
    k0: float,
    sigma_k: float,
) -> np.ndarray:
    lower_cdf = norm.cdf((0.0 - float(k0)) / float(sigma_k))
    u = np.clip(lower_cdf + np.asarray(u, dtype=float) * (1.0 - lower_cdf), 1.0e-12, 1.0 - 1.0e-12)
    return float(k0) + float(sigma_k) * norm.ppf(u)


def sample_k_vectors(
    count: int,
    distribution: KDistribution,
    rng: np.random.Generator,
    *,
    k0: float = DEFAULT_K0,
    sigma_k: float | None = None,
    k_min: float | None = None,
    k_max: float | None = None,
    use_qmc: bool = False,
    qmc_seed: int = DEFAULT_RANDOM_SEED,
) -> np.ndarray:
    """Sample isotropic 3D k-vectors for approximate radial-spectrum moments."""

    if count <= 0:
        raise ValueError("count must be positive.")
    count = int(count)
    directions = _qmc_isotropic_unit_vectors(count, qmc_seed) if use_qmc else _isotropic_unit_vectors(count, rng)
    if distribution == "single_shell":
        radii = np.full(count, float(k0))
    elif distribution == "gaussian_radial":
        if sigma_k is None:
            sigma_k = 0.15 * float(k0)
        if sigma_k <= 0.0:
            raise ValueError("sigma_k must be positive for gaussian_radial.")
        if use_qmc:
            u = _sobol_points(count, 3, qmc_seed)[:, 2]
            radii = _positive_gaussian_radii_from_unit(u, float(k0), float(sigma_k))
        else:
            radii = rng.normal(float(k0), float(sigma_k), size=count)
            while np.any(radii <= 0.0):
                bad = radii <= 0.0
                radii[bad] = rng.normal(float(k0), float(sigma_k), size=np.count_nonzero(bad))
    elif distribution == "uniform_band":
        if k_min is None:
            k_min = 0.7 * float(k0)
        if k_max is None:
            k_max = 1.3 * float(k0)
        if k_max <= k_min:
            raise ValueError("k_max must exceed k_min for uniform_band.")
        if use_qmc:
            u = _sobol_points(count, 3, qmc_seed)[:, 2]
            radii = float(k_min) + u * (float(k_max) - float(k_min))
        else:
            radii = rng.uniform(float(k_min), float(k_max), size=count)
    else:
        raise ValueError(f"Unknown k-distribution: {distribution!r}")
    return radii[:, None] * directions


def make_radial_k_quadrature(
    num_nodes: int,
    distribution: KDistribution,
    *,
    k0: float = DEFAULT_K0,
    sigma_k: float | None = None,
    k_min: float | None = None,
    k_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic radial k nodes and probability weights."""

    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    num_nodes = int(num_nodes)
    if distribution == "single_shell":
        return np.array([float(k0)]), np.array([1.0])
    u = (np.arange(num_nodes, dtype=float) + 0.5) / num_nodes
    weights = np.full(num_nodes, 1.0 / num_nodes)
    if distribution == "gaussian_radial":
        if sigma_k is None:
            sigma_k = 0.15 * float(k0)
        if sigma_k <= 0.0:
            raise ValueError("sigma_k must be positive for gaussian_radial.")
        return _positive_gaussian_radii_from_unit(u, float(k0), float(sigma_k)), weights
    if distribution == "uniform_band":
        if k_min is None:
            k_min = 0.7 * float(k0)
        if k_max is None:
            k_max = 1.3 * float(k0)
        if k_max <= k_min:
            raise ValueError("k_max must exceed k_min for uniform_band.")
        return float(k_min) + u * (float(k_max) - float(k_min)), weights
    raise ValueError(f"Unknown k-distribution: {distribution!r}")


def make_field_k_sets(
    num_modes: int | Sequence[int],
    distribution: KDistribution,
    rng: np.random.Generator,
    *,
    k0: float | Sequence[float] = DEFAULT_K0,
    r_sigma_k: float | Sequence[float] = 0.15,
    r_k_min: float | Sequence[float] = 0.7,
    r_k_max: float | Sequence[float] = 1.3,
    shared_k_vectors: bool = False,
    use_qmc_k: bool = False,
    qmc_seed: int = DEFAULT_RANDOM_SEED,
) -> FieldKSets:
    """Create k-vector sets for the two random waves used by this script."""

    counts = _mode_counts(num_modes)
    k0_values = _field_parameter_values(k0, "k0")
    r_sigma_values = _field_parameter_values(r_sigma_k, "r_sigma_k")
    r_k_min_values = _field_parameter_values(r_k_min, "r_k_min")
    r_k_max_values = _field_parameter_values(r_k_max, "r_k_max")
    sigma_values = tuple(r_sigma * kval for r_sigma, kval in zip(r_sigma_values, k0_values))
    k_min_values = tuple(rmin * kval for rmin, kval in zip(r_k_min_values, k0_values))
    k_max_values = tuple(rmax * kval for rmax, kval in zip(r_k_max_values, k0_values))

    if shared_k_vectors:
        if (
            len(set(k0_values)) > 1
            or len(set(sigma_values)) > 1
            or len(set(k_min_values)) > 1
            or len(set(k_max_values)) > 1
        ):
            raise ValueError("shared_k_vectors=True requires scalar k-distribution parameters.")
        base = sample_k_vectors(
            max(counts),
            distribution,
            rng,
            k0=k0_values[0],
            sigma_k=sigma_values[0],
            k_min=k_min_values[0],
            k_max=k_max_values[0],
            use_qmc=use_qmc_k,
            qmc_seed=qmc_seed,
        )
        return FieldKSets(base[: counts[0]].copy(), base[: counts[1]].copy())

    return FieldKSets(
        sample_k_vectors(
            counts[0],
            distribution,
            rng,
            k0=k0_values[0],
            sigma_k=sigma_values[0],
            k_min=k_min_values[0],
            k_max=k_max_values[0],
            use_qmc=use_qmc_k,
            qmc_seed=qmc_seed,
        ),
        sample_k_vectors(
            counts[1],
            distribution,
            rng,
            k0=k0_values[1],
            sigma_k=sigma_values[1],
            k_min=k_min_values[1],
            k_max=k_max_values[1],
            use_qmc=use_qmc_k,
            qmc_seed=qmc_seed + 1,
        ),
    )


def effective_k0_from_k_vectors(k_vectors: np.ndarray) -> float:
    """
    Return sqrt(<|k|^2>), the monochromatic k0 equivalent for gradient variance.

    The analytic C_L implementation below still assumes a monochromatic
    covariance. For a broadened radial spectrum this gives a simple effective
    scale for diagnostics and first-pass comparisons.
    """

    radii = np.linalg.norm(np.asarray(k_vectors, dtype=float), axis=1)
    if radii.size == 0:
        raise ValueError("k_vectors must not be empty.")
    return float(np.sqrt(np.mean(radii * radii)))


def k_radii_from_vectors(k_vectors: np.ndarray) -> np.ndarray:
    """Return positive |k| values from a k-vector array."""

    radii = np.linalg.norm(np.asarray(k_vectors, dtype=float), axis=1)
    if radii.size == 0:
        raise ValueError("k_vectors must not be empty.")
    if np.any(radii <= 0.0):
        raise ValueError("all k-vector radii must be positive.")
    return radii


def _as_positive_k_radii(k_radii: np.ndarray) -> np.ndarray:
    radii = np.asarray(k_radii, dtype=float)
    if radii.size == 0:
        raise ValueError("k_radii must not be empty.")
    if np.any(radii <= 0.0):
        raise ValueError("all k_radii values must be positive.")
    return radii


def _as_k_weights(k_radii: np.ndarray, k_weights: np.ndarray | None) -> np.ndarray | None:
    if k_weights is None:
        return None
    weights = np.asarray(k_weights, dtype=float)
    if weights.shape != np.asarray(k_radii).shape:
        raise ValueError("k_weights must have the same shape as k_radii.")
    if np.any(weights < 0.0):
        raise ValueError("k_weights must be nonnegative.")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("k_weights must have positive total weight.")
    return weights / total


def _radial_chunk_size(chunk_size: int | None) -> int:
    if chunk_size is None:
        return DEFAULT_RADIAL_CHUNK_SIZE
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return chunk_size


def radial_covariance_numeric(
    r: np.ndarray | float,
    k_radii: np.ndarray,
    *,
    k_weights: np.ndarray | None = None,
    chunk_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numerically average g(r), g'(r), and g''(r) over a radial k spectrum.

    The computation is chunked over r to avoid allocating a full
    len(r) x len(k_radii) work array for large notebook grids.
    """

    r_arr = np.asarray(r, dtype=float)
    radii = _as_positive_k_radii(k_radii)
    weights = _as_k_weights(radii, k_weights)
    r_flat = r_arr.ravel()
    n_r = r_flat.size
    chunk = _radial_chunk_size(chunk_size)

    g_out = np.empty(n_r, dtype=float)
    gp_out = np.empty(n_r, dtype=float)
    gpp_out = np.empty(n_r, dtype=float)
    k = radii[None, :]
    k2 = k * k

    for start in range(0, n_r, chunk):
        stop = min(start + chunk, n_r)
        x = np.multiply.outer(r_flat[start:stop], radii)
        sin_x = np.sin(x)
        cos_x = np.cos(x)
        small = np.abs(x) < SMALL_X
        large = ~small

        g_values = np.empty_like(x, dtype=float)
        gp_values = np.empty_like(x, dtype=float)
        gpp_values = np.empty_like(x, dtype=float)

        if np.any(small):
            xs = x[small]
            ks = np.broadcast_to(k, x.shape)[small]
            k2s = np.broadcast_to(k2, x.shape)[small]
            g_values[small] = 1.0 - xs**2 / 6.0 + xs**4 / 120.0 - xs**6 / 5040.0
            gp_values[small] = ks * (-xs / 3.0 + xs**3 / 30.0 - xs**5 / 840.0)
            gpp_values[small] = k2s * (-1.0 / 3.0 + xs**2 / 10.0 - xs**4 / 168.0)

        if np.any(large):
            xl = x[large]
            sin_l = sin_x[large]
            cos_l = cos_x[large]
            kl = np.broadcast_to(k, x.shape)[large]
            k2l = np.broadcast_to(k2, x.shape)[large]
            g_values[large] = sin_l / xl
            gp_values[large] = kl * (xl * cos_l - sin_l) / xl**2
            gpp_values[large] = k2l * (-sin_l / xl - 2.0 * cos_l / xl**2 + 2.0 * sin_l / xl**3)

        if weights is None:
            g_out[start:stop] = np.mean(g_values, axis=1)
            gp_out[start:stop] = np.mean(gp_values, axis=1)
            gpp_out[start:stop] = np.mean(gpp_values, axis=1)
        else:
            g_out[start:stop] = g_values @ weights
            gp_out[start:stop] = gp_values @ weights
            gpp_out[start:stop] = gpp_values @ weights

    return g_out.reshape(r_arr.shape), gp_out.reshape(r_arr.shape), gpp_out.reshape(r_arr.shape)


def g_radial_numeric(
    r: np.ndarray | float,
    k_radii: np.ndarray,
    *,
    k_weights: np.ndarray | None = None,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Numerically average sinc(k*r) over an isotropic radial k spectrum."""

    r_arr = np.asarray(r, dtype=float)
    radii = _as_positive_k_radii(k_radii)
    weights = _as_k_weights(radii, k_weights)
    r_flat = r_arr.ravel()
    out = np.empty(r_flat.size, dtype=float)
    chunk = _radial_chunk_size(chunk_size)

    for start in range(0, r_flat.size, chunk):
        stop = min(start + chunk, r_flat.size)
        x = np.multiply.outer(r_flat[start:stop], radii)
        values = np.empty_like(x, dtype=float)
        small = np.abs(x) < SMALL_X
        if np.any(small):
            xs = x[small]
            values[small] = 1.0 - xs**2 / 6.0 + xs**4 / 120.0 - xs**6 / 5040.0
        if np.any(~small):
            xl = x[~small]
            values[~small] = np.sin(xl) / xl
        out[start:stop] = np.mean(values, axis=1) if weights is None else values @ weights
    return out.reshape(r_arr.shape)


def gp_radial_numeric(
    r: np.ndarray | float,
    k_radii: np.ndarray,
    *,
    k_weights: np.ndarray | None = None,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Numerically average d/dr sinc(k*r) over an isotropic radial k spectrum."""

    r_arr = np.asarray(r, dtype=float)
    radii = _as_positive_k_radii(k_radii)
    weights = _as_k_weights(radii, k_weights)
    r_flat = r_arr.ravel()
    out = np.empty(r_flat.size, dtype=float)
    chunk = _radial_chunk_size(chunk_size)
    k = radii[None, :]

    for start in range(0, r_flat.size, chunk):
        stop = min(start + chunk, r_flat.size)
        x = np.multiply.outer(r_flat[start:stop], radii)
        values = np.empty_like(x, dtype=float)
        small = np.abs(x) < SMALL_X
        if np.any(small):
            xs = x[small]
            ks = np.broadcast_to(k, x.shape)[small]
            values[small] = ks * (-xs / 3.0 + xs**3 / 30.0 - xs**5 / 840.0)
        if np.any(~small):
            xl = x[~small]
            kl = np.broadcast_to(k, x.shape)[~small]
            values[~small] = kl * (xl * np.cos(xl) - np.sin(xl)) / xl**2
        out[start:stop] = np.mean(values, axis=1) if weights is None else values @ weights
    return out.reshape(r_arr.shape)


def gpp_radial_numeric(
    r: np.ndarray | float,
    k_radii: np.ndarray,
    *,
    k_weights: np.ndarray | None = None,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Numerically average d^2/dr^2 sinc(k*r) over an isotropic radial k spectrum."""

    r_arr = np.asarray(r, dtype=float)
    radii = _as_positive_k_radii(k_radii)
    weights = _as_k_weights(radii, k_weights)
    r_flat = r_arr.ravel()
    out = np.empty(r_flat.size, dtype=float)
    chunk = _radial_chunk_size(chunk_size)
    k2 = (radii[None, :]) ** 2

    for start in range(0, r_flat.size, chunk):
        stop = min(start + chunk, r_flat.size)
        x = np.multiply.outer(r_flat[start:stop], radii)
        values = np.empty_like(x, dtype=float)
        small = np.abs(x) < SMALL_X
        if np.any(small):
            xs = x[small]
            k2s = np.broadcast_to(k2, x.shape)[small]
            values[small] = k2s * (-1.0 / 3.0 + xs**2 / 10.0 - xs**4 / 168.0)
        if np.any(~small):
            xl = x[~small]
            k2l = np.broadcast_to(k2, x.shape)[~small]
            values[~small] = k2l * (
                -np.sin(xl) / xl - 2.0 * np.cos(xl) / xl**2 + 2.0 * np.sin(xl) / xl**3
            )
        out[start:stop] = np.mean(values, axis=1) if weights is None else values @ weights
    return out.reshape(r_arr.shape)


def gradient_variance_from_k_radii(k_radii: np.ndarray, k_weights: np.ndarray | None = None) -> float:
    """Return one-component gradient variance a=<k^2>/3."""

    radii = _as_positive_k_radii(k_radii)
    weights = _as_k_weights(radii, k_weights)
    moment = np.mean(radii * radii) if weights is None else float(np.dot(weights, radii * radii))
    return float(moment / 3.0)


def rho0_from_k_radii(k_radii: np.ndarray, k_weights: np.ndarray | None = None) -> float:
    """Return mean line density rho0=a/pi for an isotropic radial k spectrum."""

    return gradient_variance_from_k_radii(k_radii, k_weights=k_weights) / np.pi


def coherent_CL_general(
    c_l: np.ndarray,
    k_radii: np.ndarray,
    k_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Subtract the background rho0^2 plateau for a radial k spectrum."""

    rho0 = rho0_from_k_radii(k_radii, k_weights=k_weights)
    return np.asarray(c_l, dtype=float) - rho0**2


def g_mono(r: np.ndarray | float, k0: float = DEFAULT_K0) -> np.ndarray:
    """Monochromatic covariance g(r)=sin(k0*r)/(k0*r)."""

    r_arr = np.asarray(r, dtype=float)
    x = k0 * r_arr
    out = np.empty_like(x, dtype=float)
    small = np.abs(x) < SMALL_X
    xs = x[small]
    out[small] = 1.0 - xs**2 / 6.0 + xs**4 / 120.0 - xs**6 / 5040.0
    xl = x[~small]
    out[~small] = np.sin(xl) / xl
    return out


def gp_mono(r: np.ndarray | float, k0: float = DEFAULT_K0) -> np.ndarray:
    """First derivative of g(r) with respect to r."""

    r_arr = np.asarray(r, dtype=float)
    x = k0 * r_arr
    out = np.empty_like(x, dtype=float)
    small = np.abs(x) < SMALL_X
    xs = x[small]
    out[small] = k0 * (-xs / 3.0 + xs**3 / 30.0 - xs**5 / 840.0)
    xl = x[~small]
    out[~small] = k0 * (xl * np.cos(xl) - np.sin(xl)) / xl**2
    return out


def gpp_mono(r: np.ndarray | float, k0: float = DEFAULT_K0) -> np.ndarray:
    """Second derivative of g(r) with respect to r."""

    r_arr = np.asarray(r, dtype=float)
    x = k0 * r_arr
    out = np.empty_like(x, dtype=float)
    small = np.abs(x) < SMALL_X
    xs = x[small]
    out[small] = k0**2 * (-1.0 / 3.0 + xs**2 / 10.0 - xs**4 / 168.0)
    xl = x[~small]
    out[~small] = k0**2 * (
        -np.sin(xl) / xl - 2.0 * np.cos(xl) / xl**2 + 2.0 * np.sin(xl) / xl**3
    )
    return out


def _as_float(value: np.ndarray | float) -> float:
    return float(np.asarray(value, dtype=float))


def conditional_covariance(r: float, k0: float = DEFAULT_K0) -> np.ndarray:
    """Return the 6x6 conditional gradient covariance for one field."""

    if r <= 0.0:
        raise ValueError("r must be positive; the conditional covariance is singular at r=0.")

    a = k0**2 / 3.0
    g = _as_float(g_mono(r, k0))
    gp = _as_float(gp_mono(r, k0))
    gpp = _as_float(gpp_mono(r, k0))
    denom = max(1.0 - g * g, np.finfo(float).tiny)
    b = -gp / r
    a_z = a - gp * gp / denom
    c_z = -gpp - g * gp * gp / denom

    return np.array(
        [
            [a, 0.0, 0.0, b, 0.0, 0.0],
            [0.0, a, 0.0, 0.0, b, 0.0],
            [0.0, 0.0, a_z, 0.0, 0.0, c_z],
            [b, 0.0, 0.0, a, 0.0, 0.0],
            [0.0, b, 0.0, 0.0, a, 0.0],
            [0.0, 0.0, c_z, 0.0, 0.0, a_z],
        ],
        dtype=float,
    )


def conditional_covariance_from_radial_spectrum(
    r: float,
    k_radii: np.ndarray,
    k_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return the 6x6 conditional gradient covariance for a radial k spectrum."""

    if r <= 0.0:
        raise ValueError("r must be positive; the conditional covariance is singular at r=0.")

    a = gradient_variance_from_k_radii(k_radii, k_weights=k_weights)
    g_arr, gp_arr, gpp_arr = radial_covariance_numeric(r, k_radii, k_weights=k_weights)
    g = _as_float(g_arr)
    gp = _as_float(gp_arr)
    gpp = _as_float(gpp_arr)
    denom = max(1.0 - g * g, np.finfo(float).tiny)
    b = -gp / r
    a_z = a - gp * gp / denom
    c_z = -gpp - g * gp * gp / denom

    return np.array(
        [
            [a, 0.0, 0.0, b, 0.0, 0.0],
            [0.0, a, 0.0, 0.0, b, 0.0],
            [0.0, 0.0, a_z, 0.0, 0.0, c_z],
            [b, 0.0, 0.0, a, 0.0, 0.0],
            [0.0, b, 0.0, 0.0, a, 0.0],
            [0.0, 0.0, c_z, 0.0, 0.0, a_z],
        ],
        dtype=float,
    )


def covariance_factor(sigma: np.ndarray, jitter: float) -> np.ndarray:
    """Return a square-root factor L such that L @ L.T is approximately sigma."""

    try:
        return np.linalg.cholesky(sigma + jitter * np.eye(sigma.shape[0]))
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(sigma)
        eigvals = np.clip(eigvals, 0.0, None)
        return eigvecs @ np.diag(np.sqrt(eigvals))


def standard_normal_samples(
    n_samp: int,
    *,
    use_qmc: bool = True,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate common standard-normal samples for the two independent fields."""

    if n_samp <= 0:
        raise ValueError("n_samp must be positive.")
    if use_qmc:
        m = qmc_power_from_n_samp(n_samp)
        sampler = qmc.Sobol(d=12, scramble=True, seed=random_seed)
        uniform = sampler.random_base2(m=m)
        uniform = np.clip(uniform, 1.0e-12, 1.0 - 1.0e-12)
        z = norm.ppf(uniform)
    else:
        rng = np.random.default_rng(random_seed)
        z = rng.standard_normal((n_samp, 12))
    return z[:, :6], z[:, 6:]


def standard_normal_matrix(
    n_samp: int,
    dim: int,
    *,
    sampling: SamplingMethod = "qmc",
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> np.ndarray:
    """Generate standard-normal samples with either Sobol QMC or pseudorandom MC."""

    n_samp = int(n_samp)
    dim = int(dim)
    if n_samp <= 0:
        raise ValueError("n_samp must be positive.")
    if dim <= 0:
        raise ValueError("dim must be positive.")
    sampling = str(sampling).lower()
    if sampling == "qmc":
        sampler = qmc.Sobol(d=dim, scramble=True, seed=random_seed)
        uniform = sampler.random_base2(m=qmc_power_from_n_samp(n_samp))
        uniform = np.clip(uniform, 1.0e-12, 1.0 - 1.0e-12)
        return norm.ppf(uniform)
    if sampling == "random":
        rng = np.random.default_rng(random_seed)
        return rng.standard_normal((n_samp, dim))
    raise ValueError("sampling must be 'qmc' or 'random'.")


def symmetric_covariance_sqrt(
    sigma: np.ndarray,
    *,
    tol: float = 1.0e-10,
) -> np.ndarray:
    """Return a symmetric square root of a positive semidefinite covariance."""

    sigma = 0.5 * (np.asarray(sigma, dtype=float) + np.asarray(sigma, dtype=float).T)
    eigvals, eigvecs = np.linalg.eigh(sigma)
    scale = max(float(np.max(np.abs(eigvals))), 1.0)
    min_allowed = -float(tol) * scale
    if float(np.min(eigvals)) < min_allowed:
        raise np.linalg.LinAlgError(
            f"covariance is materially indefinite: min eigenvalue={float(np.min(eigvals)):.6g}"
        )
    eigvals = np.clip(eigvals, 0.0, None)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def make_r_grid(
    r_min: float,
    r_max: float,
    nr: int,
    *,
    mode: str = "mixed",
    r_split: float | None = None,
    n_small: int | None = None,
) -> np.ndarray:
    """Build a positive r grid, optionally dense near r=0 with a long tail."""

    nr = int(nr)
    if nr < 2:
        raise ValueError("nr must be at least 2.")
    if r_min <= 0.0 or r_max <= r_min:
        raise ValueError("Require 0 < r_min < r_max.")
    mode = str(mode).lower()
    if mode == "linear":
        return np.linspace(r_min, r_max, nr)
    if mode != "mixed":
        raise ValueError("r grid mode must be 'mixed' or 'linear'.")

    if r_split is None:
        r_split = min(1.0, 0.1 * r_max)
    r_split = float(np.clip(r_split, r_min, r_max))
    if r_split >= r_max:
        return np.linspace(r_min, r_max, nr)

    if n_small is None:
        n_small = min(max(nr // 3, 400), nr - 2)
    n_small = int(np.clip(n_small, 2, nr - 2))
    n_tail = nr - n_small + 1

    small = np.linspace(r_min, r_split, n_small, endpoint=False)
    tail = np.linspace(r_split, r_max, n_tail)
    return np.concatenate([small, tail])


def sample_MJ_for_r(
    r: float,
    k0: float,
    z_u: np.ndarray,
    z_v: np.ndarray,
    *,
    jitter_scale: float = 1.0e-12,
) -> float:
    """Estimate M_J(r)=E[|u0 x v0| |ur x vr|] with common random samples."""

    sigma = conditional_covariance(r, k0)
    a = k0**2 / 3.0
    factor = covariance_factor(sigma, jitter_scale * a)
    u_samples = z_u @ factor.T
    v_samples = z_v @ factor.T

    u0 = u_samples[:, :3]
    ur = u_samples[:, 3:]
    v0 = v_samples[:, :3]
    vr = v_samples[:, 3:]
    x = np.linalg.norm(np.cross(u0, v0), axis=1) * np.linalg.norm(np.cross(ur, vr), axis=1)
    return float(np.mean(x))


def sample_MJ_for_r_general(
    r: float,
    k_radii: np.ndarray,
    z_u: np.ndarray,
    z_v: np.ndarray,
    *,
    k_weights: np.ndarray | None = None,
    jitter_scale: float = 1.0e-12,
) -> float:
    """Estimate M_J(r) for a general isotropic radial k spectrum."""

    sigma = conditional_covariance_from_radial_spectrum(r, k_radii, k_weights=k_weights)
    a = gradient_variance_from_k_radii(k_radii, k_weights=k_weights)
    factor = covariance_factor(sigma, jitter_scale * a)
    u_samples = z_u @ factor.T
    v_samples = z_v @ factor.T

    u0 = u_samples[:, :3]
    ur = u_samples[:, 3:]
    v0 = v_samples[:, :3]
    vr = v_samples[:, 3:]
    x = np.linalg.norm(np.cross(u0, v0), axis=1) * np.linalg.norm(np.cross(ur, vr), axis=1)
    return float(np.mean(x))


def cross_product_projector(u: np.ndarray) -> np.ndarray:
    """Return ``K(u)=|u|^2 I - u u^T`` so ``|u x v|^2 = v^T K(u) v``."""

    u = np.asarray(u, dtype=float)
    if u.shape != (3,):
        raise ValueError("u must have shape (3,).")
    return float(np.dot(u, u)) * np.eye(3) - np.outer(u, u)


def cross_product_projector_batch(u: np.ndarray) -> np.ndarray:
    """Vectorized ``K(u)=|u|^2 I-u u^T`` for an array with shape ``(..., 3)``."""

    u = np.asarray(u, dtype=float)
    if u.shape[-1] != 3:
        raise ValueError("u must have trailing shape 3.")
    norm2 = np.sum(u * u, axis=-1)
    return norm2[..., None, None] * np.eye(3) - u[..., :, None] * u[..., None, :]


def perpendicular_frame_batch(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``norm(u)`` and two orthonormal vectors perpendicular to each row."""

    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[1] != 3:
        raise ValueError("u must have shape (n, 3).")
    norms = np.linalg.norm(u, axis=1)
    unit = np.zeros_like(u)
    nonzero = norms > np.finfo(float).tiny
    unit[nonzero] = u[nonzero] / norms[nonzero, None]

    axis_index = np.argmin(np.abs(unit), axis=1)
    refs = np.eye(3)[axis_index]
    e1 = np.cross(unit, refs)
    e1_norm = np.linalg.norm(e1, axis=1)
    good = e1_norm > np.finfo(float).tiny
    e1[good] /= e1_norm[good, None]
    e1[~good] = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(unit, e1)
    e2_norm = np.linalg.norm(e2, axis=1)
    good_e2 = e2_norm > np.finfo(float).tiny
    e2[good_e2] /= e2_norm[good_e2, None]
    e2[~good_e2] = np.array([0.0, 1.0, 0.0])
    frames = np.stack((e1, e2), axis=-1)
    return norms, frames


def make_cross_product_quadratic_matrices(U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the 6x6 matrices A0(U), Ar(U) for the two cross-product norms."""

    U = np.asarray(U, dtype=float)
    if U.shape != (6,):
        raise ValueError("U must have shape (6,).")
    A0 = np.zeros((6, 6), dtype=float)
    Ar = np.zeros((6, 6), dtype=float)
    A0[:3, :3] = cross_product_projector(U[:3])
    Ar[3:, 3:] = cross_product_projector(U[3:])
    return A0, Ar


def make_cross_product_quadratic_matrices_batch(U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized 6x6 matrices ``A0(U), Ar(U)`` for ``U`` with shape ``(n, 6)``."""

    U = np.asarray(U, dtype=float)
    if U.ndim != 2 or U.shape[1] != 6:
        raise ValueError("U must have shape (n, 6).")
    A0 = np.zeros((len(U), 6, 6), dtype=float)
    Ar = np.zeros((len(U), 6, 6), dtype=float)
    A0[:, :3, :3] = cross_product_projector_batch(U[:, :3])
    Ar[:, 3:, 3:] = cross_product_projector_batch(U[:, 3:])
    return A0, Ar


def make_cross_product_low_rank_factors_batch(U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return 6x2 factors ``L0, Lr`` so ``A0=L0 L0.T`` and ``Ar=Lr Lr.T``.

    Each 3D cross-product projector is rank 2:
    ``|u x v|^2 = v.T (|u|^2 I-u u.T) v``.
    """

    U = np.asarray(U, dtype=float)
    if U.ndim != 2 or U.shape[1] != 6:
        raise ValueError("U must have shape (n, 6).")
    norm0, frame0 = perpendicular_frame_batch(U[:, :3])
    normr, framer = perpendicular_frame_batch(U[:, 3:])
    L0 = np.zeros((len(U), 6, 2), dtype=float)
    Lr = np.zeros((len(U), 6, 2), dtype=float)
    L0[:, :3, :] = norm0[:, None, None] * frame0
    Lr[:, 3:, :] = normr[:, None, None] * framer
    return L0, Lr


def gaussian_quadratic_determinant(
    sigma_half: np.ndarray,
    A0: np.ndarray,
    Ar: np.ndarray,
    t: float,
    s: float,
    *,
    diagnostic: str = "",
    eig_tol: float = 1.0e-9,
) -> float:
    """
    Return ``det(I + 2 Sigma_half (t A0 + s Ar) Sigma_half)^(-1/2)``.

    The determinant matrix is symmetrized before `slogdet`; materially
    nonpositive determinants raise a diagnostic error.
    """

    B = float(t) * A0 + float(s) * Ar
    mat = np.eye(6) + 2.0 * (sigma_half @ B @ sigma_half)
    mat = 0.5 * (mat + mat.T)
    sign, logdet = np.linalg.slogdet(mat)
    if sign <= 0:
        eigvals = np.linalg.eigvalsh(mat)
        if float(np.min(eigvals)) < -eig_tol:
            raise np.linalg.LinAlgError(
                f"nonpositive determinant in gaussian_quadratic_determinant "
                f"({diagnostic}); min eigenvalue={float(np.min(eigvals)):.6g}, "
                f"t={float(t):.6g}, s={float(s):.6g}"
            )
        eigvals = np.clip(eigvals, 0.0, None)
        logdet = float(np.sum(np.log(np.maximum(eigvals, np.finfo(float).tiny))))
    return float(np.exp(-0.5 * logdet))


def gaussian_quadratic_determinant_batch(
    sigma_half: np.ndarray,
    A0: np.ndarray,
    Ar: np.ndarray,
    t: np.ndarray,
    s: np.ndarray,
    *,
    diagnostic: str = "",
    eig_tol: float = 1.0e-9,
) -> np.ndarray:
    """
    Batched determinant helper for many ``U`` samples and ``t,s`` nodes.

    ``A0`` and ``Ar`` have shape ``(n_U, 6, 6)``. ``t`` and ``s`` are 1D node
    arrays. The return value has shape ``(n_U, n_nodes)``.
    """

    sigma_half = np.asarray(sigma_half, dtype=float)
    A0 = np.asarray(A0, dtype=float)
    Ar = np.asarray(Ar, dtype=float)
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    B = t[None, :, None, None] * A0[:, None, :, :] + s[None, :, None, None] * Ar[:, None, :, :]
    mats = np.eye(6) + 2.0 * np.einsum("ij,...jk,kl->...il", sigma_half, B, sigma_half, optimize=True)
    mats = 0.5 * (mats + np.swapaxes(mats, -1, -2))
    sign, logdet = np.linalg.slogdet(mats)
    bad = sign <= 0
    if np.any(bad):
        eigvals = np.linalg.eigvalsh(mats[bad])
        min_eig = float(np.min(eigvals))
        if min_eig < -eig_tol:
            raise np.linalg.LinAlgError(
                f"nonpositive determinant in gaussian_quadratic_determinant_batch "
                f"({diagnostic}); min eigenvalue={min_eig:.6g}"
            )
        clipped = np.clip(eigvals, 0.0, None)
        logdet = np.array(logdet, copy=True)
        logdet[bad] = np.sum(np.log(np.maximum(clipped, np.finfo(float).tiny)), axis=-1)
    return np.exp(-0.5 * logdet)


def _positive_det_factor_from_logdet_mats(
    mats: np.ndarray,
    *,
    diagnostic: str = "",
    eig_tol: float = 1.0e-9,
) -> np.ndarray:
    """Return ``det(mats)^(-1/2)`` for stacked small SPD matrices."""

    mats = 0.5 * (mats + np.swapaxes(mats, -1, -2))
    sign, logdet = np.linalg.slogdet(mats)
    bad = sign <= 0
    if np.any(bad):
        eigvals = np.linalg.eigvalsh(mats[bad])
        min_eig = float(np.min(eigvals))
        if min_eig < -eig_tol:
            raise np.linalg.LinAlgError(
                f"nonpositive low-rank determinant ({diagnostic}); min eigenvalue={min_eig:.6g}"
            )
        clipped = np.clip(eigvals, 0.0, None)
        logdet = np.array(logdet, copy=True)
        logdet[bad] = np.sum(np.log(np.maximum(clipped, np.finfo(float).tiny)), axis=-1)
    return np.exp(-0.5 * logdet)


def gaussian_quadratic_determinants_lowrank_batch(
    sigma: np.ndarray,
    L0: np.ndarray,
    Lr: np.ndarray,
    t: np.ndarray,
    s: np.ndarray,
    *,
    diagnostic: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Low-rank determinant lemma version of the three auxiliary determinants.

    With ``A0=L0 L0.T`` and ``Ar=Lr Lr.T``,
    ``det(I + 2 Sigma_half (t*A0+s*Ar) Sigma_half)`` equals
    ``det(I + 2 X.T Sigma X)``, where ``X=[sqrt(t)L0, sqrt(s)Lr]``.
    The mixed term is therefore only a 4x4 determinant.
    """

    sigma = np.asarray(sigma, dtype=float)
    L0 = np.asarray(L0, dtype=float)
    Lr = np.asarray(Lr, dtype=float)
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    G00 = np.einsum("nki,kl,nlj->nij", L0, sigma, L0, optimize=True)
    Grr = np.einsum("nki,kl,nlj->nij", Lr, sigma, Lr, optimize=True)
    G0r = np.einsum("nki,kl,nlj->nij", L0, sigma, Lr, optimize=True)

    eye2 = np.eye(2)
    mats_t0 = eye2 + 2.0 * t[None, :, None, None] * G00[:, None, :, :]
    mats_0s = eye2 + 2.0 * s[None, :, None, None] * Grr[:, None, :, :]

    n_u = len(L0)
    n_st = len(t)
    mats_ts = np.zeros((n_u, n_st, 4, 4), dtype=float)
    mats_ts[..., :2, :2] = eye2 + 2.0 * t[None, :, None, None] * G00[:, None, :, :]
    mats_ts[..., 2:, 2:] = eye2 + 2.0 * s[None, :, None, None] * Grr[:, None, :, :]
    cross_scale = 2.0 * np.sqrt(t * s)[None, :, None, None]
    mats_ts[..., :2, 2:] = cross_scale * G0r[:, None, :, :]
    mats_ts[..., 2:, :2] = cross_scale * np.swapaxes(G0r, -1, -2)[:, None, :, :]

    return (
        _positive_det_factor_from_logdet_mats(mats_t0, diagnostic=f"{diagnostic}, t0"),
        _positive_det_factor_from_logdet_mats(mats_0s, diagnostic=f"{diagnostic}, 0s"),
        _positive_det_factor_from_logdet_mats(mats_ts, diagnostic=f"{diagnostic}, ts"),
    )


def auxiliary_integrand_st(
    t: float,
    s: float,
    U: np.ndarray,
    sigma_half: np.ndarray,
    *,
    diagnostic: str = "",
) -> float:
    """Return the finite positive-quadrant integrand for ``G(U; r)``."""

    if t <= 0.0 or s <= 0.0:
        return 0.0
    A0, Ar = make_cross_product_quadratic_matrices(U)
    d_t0 = gaussian_quadratic_determinant(sigma_half, A0, Ar, t, 0.0, diagnostic=diagnostic)
    d_0s = gaussian_quadratic_determinant(sigma_half, A0, Ar, 0.0, s, diagnostic=diagnostic)
    d_ts = gaussian_quadratic_determinant(sigma_half, A0, Ar, t, s, diagnostic=diagnostic)
    F = 1.0 - d_t0 - d_0s + d_ts
    return float(F / (4.0 * np.pi * (t ** 1.5) * (s ** 1.5)))


def make_st_nodes(
    n_samp_st: int,
    *,
    sampling: STSamplingMethod = "quadrature",
    transform: STTransform = "rational",
    tau_t: float = 1.0,
    tau_s: float = 1.0,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build positive-quadrant nodes and weights for the auxiliary integral.

    The default rational map is ``t=tau*x/(1-x)``, ``s=tau*y/(1-y)`` from
    the unit square. With deterministic quadrature, ``n_samp_st`` is treated as
    an approximate total node budget and rounded up to a square tensor rule.
    """

    n_samp_st = int(n_samp_st)
    if n_samp_st <= 0:
        raise ValueError("n_samp_st must be positive.")
    sampling = str(sampling).lower()
    transform = str(transform).lower()
    tau_t = float(tau_t)
    tau_s = float(tau_s)
    if tau_t <= 0.0 or tau_s <= 0.0:
        raise ValueError("tau_t and tau_s must be positive.")

    if sampling == "quadrature":
        n_side = int(np.ceil(np.sqrt(n_samp_st)))
        x_1d, w_1d = np.polynomial.legendre.leggauss(n_side)
        x_1d = 0.5 * (x_1d + 1.0)
        w_1d = 0.5 * w_1d
        x, y = np.meshgrid(x_1d, x_1d, indexing="ij")
        wx, wy = np.meshgrid(w_1d, w_1d, indexing="ij")
        unit = np.column_stack((x.ravel(), y.ravel()))
        unit_w = (wx * wy).ravel()
    elif sampling == "qmc":
        sampler = qmc.Sobol(d=2, scramble=True, seed=random_seed)
        unit = sampler.random_base2(m=qmc_power_from_n_samp(n_samp_st))
        unit = np.clip(unit, 1.0e-12, 1.0 - 1.0e-12)
        unit_w = np.full(len(unit), 1.0 / len(unit), dtype=float)
    else:
        raise ValueError("st_sampling must be 'quadrature' or 'qmc'.")

    x = unit[:, 0]
    y = unit[:, 1]
    if transform == "rational":
        t = tau_t * x / (1.0 - x)
        s = tau_s * y / (1.0 - y)
        jac = tau_t / (1.0 - x) ** 2 * tau_s / (1.0 - y) ** 2
    elif transform == "logistic":
        # x in (0,1) -> z in (-inf,inf) -> t=tau*exp(z)
        zt = np.log(x) - np.log1p(-x)
        zs = np.log(y) - np.log1p(-y)
        t = tau_t * np.exp(zt)
        s = tau_s * np.exp(zs)
        jac = (t / (x * (1.0 - x))) * (s / (y * (1.0 - y)))
    else:
        raise ValueError("st_transform must be 'rational' or 'logistic'.")

    nodes = np.column_stack((t, s))
    weights = unit_w * jac
    return nodes, weights


def evaluate_inner_st_integral(
    U: np.ndarray,
    sigma_half: np.ndarray,
    st_nodes: np.ndarray,
    st_weights: np.ndarray,
    *,
    diagnostic: str = "",
) -> float:
    """Evaluate ``G(U; r)`` with precomputed positive-quadrant nodes."""

    U = np.asarray(U, dtype=float)
    A0, Ar = make_cross_product_quadratic_matrices(U)
    total = 0.0
    for node_index, ((t, s), weight) in enumerate(zip(st_nodes, st_weights)):
        d_t0 = gaussian_quadratic_determinant(
            sigma_half, A0, Ar, float(t), 0.0, diagnostic=f"{diagnostic}, st={node_index}"
        )
        d_0s = gaussian_quadratic_determinant(
            sigma_half, A0, Ar, 0.0, float(s), diagnostic=f"{diagnostic}, st={node_index}"
        )
        d_ts = gaussian_quadratic_determinant(
            sigma_half, A0, Ar, float(t), float(s), diagnostic=f"{diagnostic}, st={node_index}"
        )
        F = 1.0 - d_t0 - d_0s + d_ts
        total += float(weight) * F / (4.0 * np.pi * (float(t) ** 1.5) * (float(s) ** 1.5))
    return float(total)


def evaluate_inner_st_integrals_batched(
    U_samples: np.ndarray,
    sigma_half: np.ndarray,
    st_nodes: np.ndarray,
    st_weights: np.ndarray,
    *,
    u_batch_size: int = DEFAULT_CONDITIONAL_U_BATCH_SIZE,
    st_batch_size: int = DEFAULT_CONDITIONAL_ST_BATCH_SIZE,
    diagnostic: str = "",
) -> np.ndarray:
    """
    Evaluate ``G(U; r)`` for many outer samples with batched 6x6 determinants.

    This computes the same integral as ``evaluate_inner_st_integral`` but avoids
    the scalar Python loop over every ``(U, t, s)`` determinant. Memory use is
    controlled by the two batch sizes.
    """

    U_samples = np.asarray(U_samples, dtype=float)
    if U_samples.ndim != 2 or U_samples.shape[1] != 6:
        raise ValueError("U_samples must have shape (n, 6).")
    st_nodes = np.asarray(st_nodes, dtype=float)
    st_weights = np.asarray(st_weights, dtype=float)
    if st_nodes.ndim != 2 or st_nodes.shape[1] != 2:
        raise ValueError("st_nodes must have shape (n, 2).")
    if st_weights.shape != (len(st_nodes),):
        raise ValueError("st_weights must have shape (len(st_nodes),).")

    u_batch_size = int(u_batch_size)
    st_batch_size = int(st_batch_size)
    if u_batch_size <= 0 or st_batch_size <= 0:
        raise ValueError("u_batch_size and st_batch_size must be positive.")

    t_all = st_nodes[:, 0]
    s_all = st_nodes[:, 1]
    kernel_weights = st_weights / (4.0 * np.pi * np.power(t_all, 1.5) * np.power(s_all, 1.5))
    values = np.empty(len(U_samples), dtype=float)

    for u_start in range(0, len(U_samples), u_batch_size):
        u_stop = min(u_start + u_batch_size, len(U_samples))
        A0, Ar = make_cross_product_quadratic_matrices_batch(U_samples[u_start:u_stop])
        block_values = np.zeros(u_stop - u_start, dtype=float)
        for st_start in range(0, len(st_nodes), st_batch_size):
            st_stop = min(st_start + st_batch_size, len(st_nodes))
            t = t_all[st_start:st_stop]
            s = s_all[st_start:st_stop]
            d_t0 = gaussian_quadratic_determinant_batch(
                sigma_half,
                A0,
                Ar,
                t,
                np.zeros_like(s),
                diagnostic=f"{diagnostic}, U={u_start}:{u_stop}, t0={st_start}:{st_stop}",
            )
            d_0s = gaussian_quadratic_determinant_batch(
                sigma_half,
                A0,
                Ar,
                np.zeros_like(t),
                s,
                diagnostic=f"{diagnostic}, U={u_start}:{u_stop}, 0s={st_start}:{st_stop}",
            )
            d_ts = gaussian_quadratic_determinant_batch(
                sigma_half,
                A0,
                Ar,
                t,
                s,
                diagnostic=f"{diagnostic}, U={u_start}:{u_stop}, ts={st_start}:{st_stop}",
            )
            F = 1.0 - d_t0 - d_0s + d_ts
            block_values += F @ kernel_weights[st_start:st_stop]
        values[u_start:u_stop] = block_values
    return values


def evaluate_inner_st_integrals_lowrank(
    U_samples: np.ndarray,
    sigma: np.ndarray,
    st_nodes: np.ndarray,
    st_weights: np.ndarray,
    *,
    u_batch_size: int = DEFAULT_CONDITIONAL_U_BATCH_SIZE,
    st_batch_size: int = DEFAULT_CONDITIONAL_ST_BATCH_SIZE,
    diagnostic: str = "",
) -> np.ndarray:
    """
    Evaluate ``G(U; r)`` with the rank-2 projector determinant reduction.

    This is algebraically equivalent to ``evaluate_inner_st_integrals_batched``
    but uses 2x2 and 4x4 determinants through the matrix determinant lemma
    instead of stacked 6x6 determinants.
    """

    U_samples = np.asarray(U_samples, dtype=float)
    if U_samples.ndim != 2 or U_samples.shape[1] != 6:
        raise ValueError("U_samples must have shape (n, 6).")
    sigma = 0.5 * (np.asarray(sigma, dtype=float) + np.asarray(sigma, dtype=float).T)
    if sigma.shape != (6, 6):
        raise ValueError("sigma must have shape (6, 6).")
    st_nodes = np.asarray(st_nodes, dtype=float)
    st_weights = np.asarray(st_weights, dtype=float)
    if st_nodes.ndim != 2 or st_nodes.shape[1] != 2:
        raise ValueError("st_nodes must have shape (n, 2).")
    if st_weights.shape != (len(st_nodes),):
        raise ValueError("st_weights must have shape (len(st_nodes),).")

    u_batch_size = int(u_batch_size)
    st_batch_size = int(st_batch_size)
    if u_batch_size <= 0 or st_batch_size <= 0:
        raise ValueError("u_batch_size and st_batch_size must be positive.")

    t_all = st_nodes[:, 0]
    s_all = st_nodes[:, 1]
    kernel_weights = st_weights / (4.0 * np.pi * np.power(t_all, 1.5) * np.power(s_all, 1.5))
    values = np.empty(len(U_samples), dtype=float)

    for u_start in range(0, len(U_samples), u_batch_size):
        u_stop = min(u_start + u_batch_size, len(U_samples))
        L0, Lr = make_cross_product_low_rank_factors_batch(U_samples[u_start:u_stop])
        block_values = np.zeros(u_stop - u_start, dtype=float)
        for st_start in range(0, len(st_nodes), st_batch_size):
            st_stop = min(st_start + st_batch_size, len(st_nodes))
            d_t0, d_0s, d_ts = gaussian_quadratic_determinants_lowrank_batch(
                sigma,
                L0,
                Lr,
                t_all[st_start:st_stop],
                s_all[st_start:st_stop],
                diagnostic=f"{diagnostic}, U={u_start}:{u_stop}, st={st_start}:{st_stop}",
            )
            F = 1.0 - d_t0 - d_0s + d_ts
            block_values += F @ kernel_weights[st_start:st_stop]
        values[u_start:u_stop] = block_values
    return values


@dataclass(frozen=True)
class ConditionalMJDiagnostics:
    """Diagnostics for the 6D+2D conditional M_J estimator."""

    mean: float
    stderr: float
    sample_std: float
    n_samp_U: int
    n_samp_st: int
    st_nodes_used: int
    tau: float
    U_sampling: str
    st_sampling: str
    st_transform: str


@dataclass(frozen=True)
class LowQAsymptoticFit:
    """Diagnostics for the quadratic low-Q line-intensity continuation."""

    Q_grid: np.ndarray
    I_original: np.ndarray
    I_stabilized: np.ndarray
    I_asymptotic: np.ndarray
    replaced_mask: np.ndarray
    fit_mask: np.ndarray
    I0: float
    I2: float
    q_fit_min: float
    q_fit_max: float
    q_replace_max: float
    relative_rmse: float
    q_resolution: float


def estimate_MJ_conditional_with_nodes(
    sigma: np.ndarray,
    z_u: np.ndarray,
    st_nodes: np.ndarray,
    st_weights: np.ndarray,
) -> tuple[float, float, float]:
    """Estimate conditional ``M_J`` using reused outer samples and st nodes."""

    sigma = 0.5 * (np.asarray(sigma, dtype=float) + np.asarray(sigma, dtype=float).T)
    sigma_half = symmetric_covariance_sqrt(sigma)
    z_u = np.asarray(z_u, dtype=float)
    if z_u.ndim != 2 or z_u.shape[1] != 6:
        raise ValueError("z_u must have shape (N_samp_U, 6).")
    U_samples = z_u @ sigma_half.T
    values = evaluate_inner_st_integrals_lowrank(
        U_samples,
        sigma,
        st_nodes,
        st_weights,
        diagnostic="estimate_MJ_conditional_with_nodes",
    )
    mean = float(np.mean(values))
    sample_std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    stderr = sample_std / np.sqrt(max(len(values), 1))
    return mean, sample_std, stderr


def estimate_MJ_conditional(
    sigma: np.ndarray,
    *,
    N_samp_U: int,
    N_samp_st: int,
    U_sampling: SamplingMethod = "qmc",
    st_sampling: STSamplingMethod = "quadrature",
    st_transform: STTransform = "rational",
    random_seed: int = DEFAULT_RANDOM_SEED,
    st_random_seed: int | None = None,
    z_u: np.ndarray | None = None,
    tau: float | None = None,
    return_diagnostics: bool = False,
) -> float | tuple[float, ConditionalMJDiagnostics]:
    """
    Estimate M_J with a 6D outer Gaussian average and 2D deterministic/QMC integral.

    ``tau`` is the rational/logistic map scale for both positive variables.
    If omitted, it is set to ``1/a^2`` using the average one-component
    gradient variance ``a=trace(sigma)/6``.
    """

    sigma = 0.5 * (np.asarray(sigma, dtype=float) + np.asarray(sigma, dtype=float).T)
    a_eff = max(float(np.trace(sigma) / 6.0), np.finfo(float).tiny)
    if tau is None:
        tau = 1.0 / max(a_eff * a_eff, np.finfo(float).tiny)
    tau = float(tau)

    if z_u is None:
        z_u = standard_normal_matrix(
            int(N_samp_U),
            6,
            sampling=U_sampling,
            random_seed=random_seed,
        )
    else:
        z_u = np.asarray(z_u, dtype=float)
        if z_u.ndim != 2 or z_u.shape[1] != 6:
            raise ValueError("z_u must have shape (N_samp_U, 6).")
        N_samp_U = len(z_u)
    if st_random_seed is None:
        st_random_seed = random_seed + 1009
    st_nodes, st_weights = make_st_nodes(
        int(N_samp_st),
        sampling=st_sampling,
        transform=st_transform,
        tau_t=tau,
        tau_s=tau,
        random_seed=st_random_seed,
    )
    mean, sample_std, stderr = estimate_MJ_conditional_with_nodes(sigma, z_u, st_nodes, st_weights)
    diagnostics = ConditionalMJDiagnostics(
        mean=mean,
        stderr=stderr,
        sample_std=sample_std,
        n_samp_U=int(N_samp_U),
        n_samp_st=int(N_samp_st),
        st_nodes_used=len(st_nodes),
        tau=tau,
        U_sampling=str(U_sampling),
        st_sampling=str(st_sampling),
        st_transform=str(st_transform),
    )
    return (mean, diagnostics) if return_diagnostics else mean


def compute_CL(
    r_grid: np.ndarray,
    k0: float,
    n_samp: int | None = None,
    *,
    use_qmc: bool = True,
    random_seed: int = DEFAULT_RANDOM_SEED,
    progress: bool = True,
    jacobian_method: JacobianMethod = "direct_12d",
    N_samp_U: int | None = None,
    N_samp_st: int = 2**8,
    U_sampling: SamplingMethod | None = None,
    st_sampling: STSamplingMethod = "quadrature",
    st_transform: STTransform = "rational",
    st_tau: float | None = None,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute M_J(r) and C_L(r) over the supplied positive r grid."""

    jacobian_method = str(jacobian_method).lower()
    if N_samp_U is None:
        if n_samp is None:
            N_samp_U = DEFAULT_N_SAMP
        else:
            N_samp_U = int(n_samp)
    elif n_samp is not None and int(n_samp) != int(N_samp_U):
        warnings.warn(
            "Both n_samp and N_samp_U were supplied; using N_samp_U. "
            "n_samp is retained only as a compatibility alias.",
            DeprecationWarning,
            stacklevel=2,
        )
    N_samp_U = int(N_samp_U)
    if U_sampling is None:
        U_sampling = "qmc" if use_qmc else "random"
    n_jobs = max(1, int(n_jobs))

    if jacobian_method == "direct_12d":
        z_u, z_v = standard_normal_samples(N_samp_U, use_qmc=use_qmc, random_seed=random_seed)
        st_nodes = st_weights = None
    elif jacobian_method == "conditional_6d_2d":
        z_u = standard_normal_matrix(N_samp_U, 6, sampling=U_sampling, random_seed=random_seed)
        z_v = None
        if st_tau is None:
            a = float(k0) ** 2 / 3.0
            st_tau = 1.0 / max(a * a, np.finfo(float).tiny)
        st_nodes, st_weights = make_st_nodes(
            int(N_samp_st),
            sampling=st_sampling,
            transform=st_transform,
            tau_t=float(st_tau),
            tau_s=float(st_tau),
            random_seed=random_seed + 1009,
        )
    else:
        raise ValueError("jacobian_method must be 'direct_12d' or 'conditional_6d_2d'.")

    m_j = np.empty_like(r_grid, dtype=float)
    t0 = time.perf_counter()
    report_every = max(1, len(r_grid) // 20)

    def compute_one(r_value: float) -> float:
        if jacobian_method == "direct_12d":
            return sample_MJ_for_r(float(r_value), k0, z_u, z_v)
        sigma = conditional_covariance(float(r_value), k0)
        mean, _, _ = estimate_MJ_conditional_with_nodes(sigma, z_u, st_nodes, st_weights)
        return mean

    if n_jobs == 1:
        iterator = ((idx, compute_one(float(r_value))) for idx, r_value in enumerate(r_grid))
    else:
        executor = ThreadPoolExecutor(max_workers=n_jobs)
        iterator = enumerate(executor.map(compute_one, [float(r) for r in r_grid]))
    try:
        for idx, value in iterator:
            m_j[idx] = value
            if progress and ((idx + 1) % report_every == 0 or idx + 1 == len(r_grid)):
                elapsed = time.perf_counter() - t0
                print(f"M_J {jacobian_method}: {idx + 1}/{len(r_grid)} r values ({elapsed:.1f}s)")
    finally:
        if n_jobs != 1:
            executor.shutdown(wait=True)

    g = g_mono(r_grid, k0)
    denom = 4.0 * np.pi**2 * np.maximum(1.0 - g * g, np.finfo(float).tiny)
    c_l = m_j / denom
    return m_j, c_l


def compute_CL_general(
    r_grid: np.ndarray,
    k_radii: np.ndarray,
    n_samp: int | None = None,
    *,
    k_weights: np.ndarray | None = None,
    use_qmc: bool = True,
    random_seed: int = DEFAULT_RANDOM_SEED,
    progress: bool = True,
    jacobian_method: JacobianMethod = "direct_12d",
    N_samp_U: int | None = None,
    N_samp_st: int = 2**8,
    U_sampling: SamplingMethod | None = None,
    st_sampling: STSamplingMethod = "quadrature",
    st_transform: STTransform = "rational",
    st_tau: float | None = None,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute M_J(r) and C_L(r) using a numerical radial k-spectrum covariance."""

    jacobian_method = str(jacobian_method).lower()
    if N_samp_U is None:
        if n_samp is None:
            N_samp_U = DEFAULT_N_SAMP
        else:
            N_samp_U = int(n_samp)
    elif n_samp is not None and int(n_samp) != int(N_samp_U):
        warnings.warn(
            "Both n_samp and N_samp_U were supplied; using N_samp_U. "
            "n_samp is retained only as a compatibility alias.",
            DeprecationWarning,
            stacklevel=2,
        )
    N_samp_U = int(N_samp_U)
    if U_sampling is None:
        U_sampling = "qmc" if use_qmc else "random"
    n_jobs = max(1, int(n_jobs))

    if jacobian_method == "direct_12d":
        z_u, z_v = standard_normal_samples(N_samp_U, use_qmc=use_qmc, random_seed=random_seed)
        st_nodes = st_weights = None
    elif jacobian_method == "conditional_6d_2d":
        z_u = standard_normal_matrix(N_samp_U, 6, sampling=U_sampling, random_seed=random_seed)
        z_v = None
        if st_tau is None:
            a = gradient_variance_from_k_radii(k_radii, k_weights=k_weights)
            st_tau = 1.0 / max(a * a, np.finfo(float).tiny)
        st_nodes, st_weights = make_st_nodes(
            int(N_samp_st),
            sampling=st_sampling,
            transform=st_transform,
            tau_t=float(st_tau),
            tau_s=float(st_tau),
            random_seed=random_seed + 1009,
        )
    else:
        raise ValueError("jacobian_method must be 'direct_12d' or 'conditional_6d_2d'.")

    m_j = np.empty_like(r_grid, dtype=float)
    t0 = time.perf_counter()
    report_every = max(1, len(r_grid) // 20)

    def compute_one(r_value: float) -> float:
        if jacobian_method == "direct_12d":
            return sample_MJ_for_r_general(float(r_value), k_radii, z_u, z_v, k_weights=k_weights)
        sigma = conditional_covariance_from_radial_spectrum(float(r_value), k_radii, k_weights=k_weights)
        mean, _, _ = estimate_MJ_conditional_with_nodes(sigma, z_u, st_nodes, st_weights)
        return mean

    if n_jobs == 1:
        iterator = ((idx, compute_one(float(r_value))) for idx, r_value in enumerate(r_grid))
    else:
        executor = ThreadPoolExecutor(max_workers=n_jobs)
        iterator = enumerate(executor.map(compute_one, [float(r) for r in r_grid]))
    try:
        for idx, value in iterator:
            m_j[idx] = value
            if progress and ((idx + 1) % report_every == 0 or idx + 1 == len(r_grid)):
                elapsed = time.perf_counter() - t0
                print(f"M_J {jacobian_method}: {idx + 1}/{len(r_grid)} r values ({elapsed:.1f}s)")
    finally:
        if n_jobs != 1:
            executor.shutdown(wait=True)

    g, _, _ = radial_covariance_numeric(r_grid, k_radii, k_weights=k_weights)
    denom = 4.0 * np.pi**2 * np.maximum(1.0 - g * g, np.finfo(float).tiny)
    c_l = m_j / denom
    return m_j, c_l


def compute_IQ(r_grid: np.ndarray, c_l: np.ndarray, q_grid: np.ndarray) -> np.ndarray:
    """Compute I(Q)=4*pi*int r^2 C_L(r) sin(Qr)/(Qr) dr."""

    return hankel_transform(r_grid, c_l, q_grid)


def compute_coherent_transform_diagnostics(
    r_grid: np.ndarray,
    c_l: np.ndarray,
    q_grid: np.ndarray,
    rho0: float,
    *,
    r_taper_start: float | None = None,
    use_asymptotic: bool = False,
    lowq_fit_bounds: tuple[float | None, float | None] | None = None,
    lowq_replace_max: float | None = None,
) -> dict[str, np.ndarray | float]:
    """Transform full C_L with consistent background subtraction and windowing.

    This helper is spectrum-agnostic: callers provide rho0 for the underlying
    monochromatic or radial-spectrum model. The background rho0^2 plateau is
    subtracted before applying the finite-r window to the coherent transform.
    """

    r_grid = np.asarray(r_grid, dtype=float)
    c_l = np.asarray(c_l, dtype=float)
    q_grid = np.asarray(q_grid, dtype=float)
    if r_grid.shape != c_l.shape:
        raise ValueError("r_grid and c_l must have the same shape.")
    if np.any(r_grid <= 0.0):
        raise ValueError("r_grid must be strictly positive.")

    r_max = float(r_grid[-1])
    if r_taper_start is None:
        r_taper_start = 0.75 * r_max
    rho0 = float(rho0)
    cl_coherent = c_l - rho0**2
    w_tail = tail_window(r_grid, float(r_taper_start), r_max)
    cl_transform_raw = cl_coherent
    cl_transform = cl_coherent * w_tail
    plateau = np.full_like(c_l, rho0**2)
    i_raw = hankel_transform(r_grid, cl_transform_raw, q_grid)
    i_windowed = hankel_transform(r_grid, cl_transform, q_grid)
    lowq_fit = stabilize_low_q_quadratic(
        q_grid,
        i_windowed,
        r_grid=r_grid,
        r_taper_start=float(r_taper_start),
        fit_bounds=lowq_fit_bounds,
        q_replace_max=lowq_replace_max,
    )
    i_output = lowq_fit.I_stabilized if use_asymptotic else i_windowed

    return {
        "CL": c_l,
        "CL_coherent": cl_coherent,
        "CL_transform_raw": cl_transform_raw,
        "CL_transform": cl_transform,
        "w_tail": w_tail,
        "I_coherent_raw": i_raw,
        "I_coherent_windowed": i_output,
        "I_coherent_windowed_original": i_windowed,
        "I_lowQ_asymptotic": lowq_fit.I_asymptotic,
        "I_lowQ_stabilized": lowq_fit.I_stabilized,
        "lowQ_replaced_mask": lowq_fit.replaced_mask,
        "lowQ_fit_mask": lowq_fit.fit_mask,
        "lowQ_I0": lowq_fit.I0,
        "lowQ_I2": lowq_fit.I2,
        "lowQ_fit_min": lowq_fit.q_fit_min,
        "lowQ_fit_max": lowq_fit.q_fit_max,
        "lowQ_replace_max": lowq_fit.q_replace_max,
        "lowQ_relative_rmse": lowq_fit.relative_rmse,
        "lowQ_q_resolution": lowq_fit.q_resolution,
        "use_asymptotic": bool(use_asymptotic),
        "I_full_windowed": hankel_transform(r_grid, c_l * w_tail, q_grid),
        "I_plateau_windowed": hankel_transform(r_grid, plateau * w_tail, q_grid),
        "rho0": rho0,
        "rho0_squared": float(rho0**2),
        "r_max": r_max,
        "r_taper_start": float(r_taper_start),
    }


def _fit_quadratic_low_q(q: np.ndarray, intensity: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    x = q[mask] ** 2
    y = intensity[mask]
    if x.size < 2:
        return (float("nan"), float("nan"), float("inf"))
    coeff = np.polyfit(x, y, deg=1)
    i2 = float(coeff[0])
    i0 = float(coeff[1])
    y_fit = i0 + i2 * x
    denom = max(float(np.sqrt(np.mean(y * y))), np.finfo(float).tiny)
    rel_rmse = float(np.sqrt(np.mean((y - y_fit) ** 2)) / denom)
    return i0, i2, rel_rmse


def stabilize_low_q_quadratic(
    q_grid: np.ndarray,
    intensity: np.ndarray,
    *,
    r_grid: np.ndarray,
    r_taper_start: float | None = None,
    fit_bounds: tuple[float | None, float | None] | None = None,
    q_replace_max: float | None = None,
    min_fit_points: int = 6,
) -> LowQAsymptoticFit:
    """Fit ``I(Q)=I0+I2 Q^2`` and replace the low-Q finite-window ringing.

    The automatic fit avoids the least reliable finite-box modes below roughly
    ``2*pi/r_max``. Candidate upper bounds are tested over the lowest available
    Q range, and the window with the smallest relative quadratic residual is
    used. Supplying ``fit_bounds`` overrides the automatic bounds.
    """

    q = np.asarray(q_grid, dtype=float)
    i_original = np.asarray(intensity, dtype=float)
    r = np.asarray(r_grid, dtype=float)
    if q.shape != i_original.shape:
        raise ValueError("q_grid and intensity must have the same shape.")
    if q.ndim != 1 or q.size < max(3, int(min_fit_points)):
        raise ValueError("q_grid must be one-dimensional with enough points.")
    if np.any(q <= 0.0):
        raise ValueError("q_grid values must be positive for low-Q fitting.")
    if r.ndim != 1 or r.size < 2 or np.any(r <= 0.0):
        raise ValueError("r_grid must be a positive one-dimensional grid.")

    finite = np.isfinite(i_original)
    r_max = float(np.max(r))
    q_resolution = 2.0 * np.pi / r_max
    if r_taper_start is not None:
        taper_width = max(r_max - float(r_taper_start), np.finfo(float).tiny)
        q_resolution = min(q_resolution, np.pi / taper_width)

    if fit_bounds is not None:
        q_fit_min = float(q[0] if fit_bounds[0] is None else fit_bounds[0])
        q_fit_max = float(q[-1] if fit_bounds[1] is None else fit_bounds[1])
        fit_mask = finite & (q >= q_fit_min) & (q <= q_fit_max)
        if np.count_nonzero(fit_mask) < min_fit_points:
            raise ValueError("low-Q fit_bounds select too few finite Q points.")
        i0, i2, rel_rmse = _fit_quadratic_low_q(q, i_original, fit_mask)
    else:
        q_fit_min_floor = max(float(q[0]), q_resolution)
        start_idx = int(np.searchsorted(q, q_fit_min_floor, side="left"))
        start_idx = min(start_idx, max(0, q.size - int(min_fit_points)))
        candidates: list[tuple[float, float, float, np.ndarray]] = []
        max_stop = min(q.size, start_idx + max(int(min_fit_points) + 12, q.size // 3))
        for stop in range(start_idx + int(min_fit_points), max_stop + 1):
            mask = finite.copy()
            mask[:start_idx] = False
            mask[stop:] = False
            if np.count_nonzero(mask) < min_fit_points:
                continue
            i0_try, i2_try, err_try = _fit_quadratic_low_q(q, i_original, mask)
            if np.isfinite(i0_try) and np.isfinite(i2_try) and np.isfinite(err_try):
                candidates.append((err_try, i0_try, i2_try, mask))
        if not candidates:
            fit_mask = finite.copy()
            fit_mask[:start_idx] = False
            fit_mask[start_idx + int(min_fit_points):] = False
            i0, i2, rel_rmse = _fit_quadratic_low_q(q, i_original, fit_mask)
        else:
            rel_rmse, i0, i2, fit_mask = min(candidates, key=lambda item: item[0])
        q_fit_values = q[fit_mask]
        q_fit_min = float(q_fit_values[0])
        q_fit_max = float(q_fit_values[-1])

    i_asymptotic = i0 + i2 * q * q
    if q_replace_max is None:
        q_replace_max = q_fit_max
    q_replace_max = float(q_replace_max)
    replaced = q <= q_replace_max
    i_stabilized = i_original.copy()
    i_stabilized[replaced] = i_asymptotic[replaced]

    return LowQAsymptoticFit(
        Q_grid=q.copy(),
        I_original=i_original.copy(),
        I_stabilized=i_stabilized,
        I_asymptotic=i_asymptotic,
        replaced_mask=replaced,
        fit_mask=fit_mask.copy(),
        I0=float(i0),
        I2=float(i2),
        q_fit_min=float(q_fit_min),
        q_fit_max=float(q_fit_max),
        q_replace_max=q_replace_max,
        relative_rmse=float(rel_rmse),
        q_resolution=float(q_resolution),
    )


def line_low_q_moments_from_CL(
    r_grid: np.ndarray,
    c_l: np.ndarray,
    rho0: float,
    *,
    window: np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    """Compute direct real-space low-Q moments from ``C_L(r)-rho0^2``.

    The returned coefficients correspond to

    ``I0 = 4*pi int r^2 [C_L(r)-rho0^2] dr``

    and

    ``I2 = -(2*pi/3) int r^4 [C_L(r)-rho0^2] dr``.

    A finite-r window may be supplied to match the transform window used in
    numerical Hankel transforms.
    """

    r = np.asarray(r_grid, dtype=float)
    c = np.asarray(c_l, dtype=float)
    if r.shape != c.shape:
        raise ValueError("r_grid and c_l must have the same shape.")
    if r.ndim != 1 or r.size < 2 or np.any(r <= 0.0):
        raise ValueError("r_grid must be a positive one-dimensional grid.")
    coherent = c - float(rho0) ** 2
    if window is None:
        w = np.ones_like(r, dtype=float)
    else:
        w = np.asarray(window, dtype=float)
        if w.shape != r.shape:
            raise ValueError("window must have the same shape as r_grid.")
    moment2_integrand = r * r * coherent * w
    moment4_integrand = r**4 * coherent * w
    moment2 = float(simpson(moment2_integrand, x=r))
    moment4 = float(simpson(moment4_integrand, x=r))
    return {
        "I0": 4.0 * np.pi * moment2,
        "I2": -(2.0 * np.pi / 3.0) * moment4,
        "moment2": moment2,
        "moment4": moment4,
        "coherent": coherent,
        "window": w,
    }


def hankel_transform(r_grid: np.ndarray, c_r: np.ndarray, q_grid: np.ndarray) -> np.ndarray:
    """Compute H[c](Q)=4*pi*int r^2 c(r) sin(Qr)/(Qr) dr."""

    r_grid = np.asarray(r_grid, dtype=float)
    c_r = np.asarray(c_r, dtype=float)
    q_grid = np.asarray(q_grid, dtype=float)
    if r_grid.shape != c_r.shape:
        raise ValueError("r_grid and c_r must have the same shape.")
    weighted = r_grid**2 * c_r
    out = np.empty_like(q_grid, dtype=float)
    for idx, q_value in enumerate(q_grid):
        kernel = np.sinc(q_value * r_grid / np.pi)
        out[idx] = 4.0 * np.pi * float(simpson(weighted * kernel, x=r_grid))
    return out


def coherent_CL(c_l: np.ndarray, k0: float = DEFAULT_K0) -> np.ndarray:
    """Subtract the background rho0^2 plateau from C_L(r)."""

    rho0 = k0**2 / (3.0 * np.pi)
    return np.asarray(c_l, dtype=float) - rho0**2


def singular_CL(r_grid: np.ndarray, rho0: float, rc: float) -> np.ndarray:
    """Return rho0/(2*pi*r^2)*exp(-(r/rc)^2)."""

    r_grid = np.asarray(r_grid, dtype=float)
    if np.any(r_grid <= 0.0):
        raise ValueError("r_grid must be strictly positive for singular_CL.")
    if rc <= 0.0:
        raise ValueError("rc must be positive.")
    return rho0 / (2.0 * np.pi * r_grid**2) * np.exp(-(r_grid / float(rc)) ** 2)


def tail_window(r_grid: np.ndarray, r_start: float, r_max: float | None = None) -> np.ndarray:
    """
    Smoothly taper the high-r tail while leaving small and intermediate r intact.

    The window is 1 for r <= r_start and follows a half-cosine taper to 0 at
    r_max. This reduces finite-r_max ringing in the low-Q transform while
    leaving the small-r samples unchanged.
    """

    r_grid = np.asarray(r_grid, dtype=float)
    if r_max is None:
        r_max = float(np.max(r_grid))
    r_start = float(r_start)
    r_max = float(r_max)
    if r_start < 0.0 or r_max <= r_start:
        raise ValueError("Require 0 <= r_start < r_max.")

    window = np.ones_like(r_grid, dtype=float)
    tail = r_grid > r_start
    if np.any(tail):
        t = np.clip((r_grid[tail] - r_start) / (r_max - r_start), 0.0, 1.0)
        window[tail] = 0.5 * (1.0 + np.cos(np.pi * t))
    window[r_grid >= r_max] = 0.0
    return window


def transform_window(
    r_grid: np.ndarray,
    *,
    window_type: str = "cosine",
    r_taper_start: float | None = None,
    r_max: float | None = None,
    r_decay: float | None = None,
    power: float = 6.0,
) -> np.ndarray:
    """Return a smooth finite-transform window on the supplied r grid."""

    r_grid = np.asarray(r_grid, dtype=float)
    if r_max is None:
        r_max = float(r_grid[-1])
    if r_taper_start is None:
        r_taper_start = 0.75 * float(r_max)
    window_type = str(window_type).lower()
    if window_type in {"none", "box", "rect"}:
        return np.ones_like(r_grid, dtype=float)
    if window_type in {"cosine", "hann_tail", "tukey"}:
        return tail_window(r_grid, float(r_taper_start), float(r_max))
    if window_type == "exponential":
        if r_decay is None:
            r_decay = 0.8 * float(r_max)
        if r_decay <= 0.0:
            raise ValueError("r_decay must be positive.")
        return np.exp(-((r_grid / float(r_decay)) ** float(power)))
    raise ValueError("window_type must be cosine, hann_tail, tukey, exponential, or none.")


def restricted_hann_window(r_grid: np.ndarray, r1: float, r2: float) -> np.ndarray:
    """Hann window restricted to [r1, r2], zero elsewhere."""

    r_grid = np.asarray(r_grid, dtype=float)
    if r2 <= r1:
        raise ValueError("r2 must exceed r1.")
    w = np.zeros_like(r_grid, dtype=float)
    mask = (r_grid >= r1) & (r_grid <= r2)
    x = (r_grid[mask] - float(r1)) / (float(r2) - float(r1))
    w[mask] = 0.5 * (1.0 - np.cos(2.0 * np.pi * x))
    return w


def tail_frequency_projection(
    r_grid: np.ndarray,
    values: np.ndarray,
    k0: float,
    *,
    r_tail_min: float | None = None,
    r_tail_max: float | None = None,
) -> dict[str, float | np.ndarray]:
    """Project a real-space tail onto k0 and 2*k0 sine/cosine components."""

    r_grid = np.asarray(r_grid, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.shape != r_grid.shape:
        raise ValueError("values must have the same shape as r_grid.")
    if r_tail_min is None:
        r_tail_min = 20.0 / float(k0)
    if r_tail_max is None:
        r_tail_max = float(r_grid[-1])
    w = restricted_hann_window(r_grid, float(r_tail_min), float(r_tail_max))
    f_tail = values * w
    norm = float(simpson(np.abs(f_tail), x=r_grid))
    result: dict[str, float | np.ndarray] = {
        "window": w,
        "f_tail": f_tail,
        "norm": norm,
        "r_tail_min": float(r_tail_min),
        "r_tail_max": float(r_tail_max),
    }
    for label, kval in (("1", float(k0)), ("2", 2.0 * float(k0))):
        a_cos = float(simpson(f_tail * np.cos(kval * r_grid), x=r_grid))
        a_sin = float(simpson(f_tail * np.sin(kval * r_grid), x=r_grid))
        amp = float(np.hypot(a_cos, a_sin))
        result[f"A{label}_cos"] = a_cos
        result[f"A{label}_sin"] = a_sin
        result[f"Amp{label}"] = amp
        result[f"RelAmp{label}"] = amp / norm if norm > 0.0 else np.nan
    return result


def tail_only_window(r_grid: np.ndarray, r_tail_min: float, r_tail_max: float | None = None) -> np.ndarray:
    """Smooth window selecting only the large-r tail interval."""

    if r_tail_max is None:
        r_tail_max = float(np.asarray(r_grid)[-1])
    return restricted_hann_window(r_grid, r_tail_min, r_tail_max)


def compute_window_comparison(
    r_grid: np.ndarray,
    c_coherent: np.ndarray,
    q_grid: np.ndarray,
    *,
    window_types: Sequence[str] = ("cosine", "exponential", "none"),
    taper_fractions: Sequence[float] = (0.50, 0.65, 0.80, 0.90),
    r_max: float | None = None,
) -> list[dict[str, object]]:
    """Compute H[C_coherent*w] for several window types and taper starts."""

    r_grid = np.asarray(r_grid, dtype=float)
    c_coherent = np.asarray(c_coherent, dtype=float)
    if r_max is None:
        r_max = float(r_grid[-1])
    out: list[dict[str, object]] = []
    for window_type in window_types:
        fractions = (np.nan,) if window_type == "none" else taper_fractions
        for frac in fractions:
            r_taper_start = 0.75 * float(r_max) if np.isnan(frac) else float(frac) * float(r_max)
            w = transform_window(
                r_grid,
                window_type=window_type,
                r_taper_start=r_taper_start,
                r_max=float(r_max),
            )
            out.append(
                {
                    "window_type": window_type,
                    "taper_fraction": float(frac) if not np.isnan(frac) else np.nan,
                    "r_taper_start": r_taper_start,
                    "window": w,
                    "I": hankel_transform(r_grid, c_coherent * w, q_grid),
                }
            )
    return out


def compute_rmax_comparison(
    r_grid: np.ndarray,
    c_coherent: np.ndarray,
    q_grid: np.ndarray,
    k0: float,
    *,
    r_max_values: Sequence[float],
    taper_fraction: float = 0.75,
    window_type: str = "cosine",
) -> list[dict[str, object]]:
    """Compute H[C_coherent*w] after truncating to several r_max values."""

    out: list[dict[str, object]] = []
    for r_max_value in r_max_values:
        mask = r_grid <= float(r_max_value)
        if np.count_nonzero(mask) < 8:
            continue
        r_sub = np.asarray(r_grid)[mask]
        c_sub = np.asarray(c_coherent)[mask]
        rmax_actual = float(r_sub[-1])
        w = transform_window(
            r_sub,
            window_type=window_type,
            r_taper_start=float(taper_fraction) * rmax_actual,
            r_max=rmax_actual,
        )
        out.append(
            {
                "r_max": rmax_actual,
                "r_max_over_k0": rmax_actual * float(k0),
                "taper_fraction": float(taper_fraction),
                "window": w,
                "I": hankel_transform(r_sub, c_sub * w, q_grid),
            }
        )
    return out


def simplified_function_transforms(
    r_grid: np.ndarray,
    q_grid: np.ndarray,
    k0: float,
    window: np.ndarray,
    *,
    subtract_mean: bool = True,
) -> dict[str, np.ndarray]:
    """Transform covariance-derived diagnostic functions."""

    r_grid = np.asarray(r_grid, dtype=float)
    window = np.asarray(window, dtype=float)
    g = g_mono(r_grid, k0)
    gp = gp_mono(r_grid, k0)
    gpp = gpp_mono(r_grid, k0)
    funcs = {
        "g": g,
        "g2": g * g,
        "gp2": gp * gp,
        "gpp2": gpp * gpp,
        "bg": (-gp / r_grid) ** 2,
        "cg": (-gpp) ** 2,
    }
    out: dict[str, np.ndarray] = {}
    for name, values in funcs.items():
        transform_values = np.asarray(values, dtype=float).copy()
        if subtract_mean:
            transform_values = transform_values - float(np.mean(transform_values[-max(8, len(transform_values) // 20) :]))
        out[name] = hankel_transform(r_grid, transform_values * window, q_grid)
    return out


def prepare_CL_for_transform(
    r_grid: np.ndarray,
    c_l: np.ndarray,
    k0: float = DEFAULT_K0,
    *,
    subtract_background: bool = True,
    window: np.ndarray | None = None,
) -> np.ndarray:
    """
    Prepare the whole C_L(r) for finite-window numerical transformation.

    This function does not split short- and long-distance pieces. It starts
    from the supplied full correlation, optionally subtracts the background
    background rho0^2, and optionally applies a real-space window.
    """

    r_grid = np.asarray(r_grid, dtype=float)
    c_transform = np.asarray(c_l, dtype=float)
    if c_transform.shape != r_grid.shape:
        raise ValueError("c_l must have the same shape as r_grid.")
    if subtract_background:
        c_transform = coherent_CL(c_transform, k0)
    else:
        c_transform = c_transform.copy()
    if window is not None:
        window = np.asarray(window, dtype=float)
        if window.shape != r_grid.shape:
            raise ValueError("window must have the same shape as r_grid.")
        c_transform = c_transform * window
    return c_transform


def compute_transform_diagnostics(
    r_grid: np.ndarray,
    c_l: np.ndarray,
    q_grid: np.ndarray,
    k0: float = DEFAULT_K0,
    *,
    r_taper_start: float | None = None,
    use_singular_split: bool = True,
    rc: float | None = None,
) -> dict[str, np.ndarray | float | bool]:
    """
    Compute windowed finite-Q scattering plus diagnostic Hankel transforms.

    The required order is:
      1. start from the full C_L(r),
      2. subtract rho0^2,
      3. optionally split off the small-r same-line singular contribution,
      4. apply the high-r tail window to the finite-Q correlation/remainder,
      5. transform.

    The full-windowed and plateau-windowed transforms are returned only as
    diagnostics for finite-window leakage.
    """

    r_grid = np.asarray(r_grid, dtype=float)
    c_l = np.asarray(c_l, dtype=float)
    q_grid = np.asarray(q_grid, dtype=float)
    if r_grid.shape != c_l.shape:
        raise ValueError("r_grid and c_l must have the same shape.")
    if np.any(r_grid <= 0.0):
        raise ValueError("r_grid must be strictly positive.")

    r_max = float(r_grid[-1])
    if r_taper_start is None:
        r_taper_start = 0.75 * r_max
    if rc is None:
        rc = 0.5 / float(k0)

    rho0 = k0**2 / (3.0 * np.pi)
    cl_coherent = c_l - rho0**2
    w_tail = tail_window(r_grid, float(r_taper_start), r_max)

    i_full_windowed = hankel_transform(r_grid, c_l * w_tail, q_grid)
    i_plateau_windowed = hankel_transform(r_grid, np.full_like(c_l, rho0**2) * w_tail, q_grid)
    i_coherent_windowed = hankel_transform(r_grid, cl_coherent * w_tail, q_grid)

    c_sing = np.zeros_like(c_l)
    c_reg = cl_coherent.copy()
    i_sing = np.zeros_like(q_grid)
    i_reg_windowed = i_coherent_windowed.copy()
    i_total = i_coherent_windowed.copy()

    if use_singular_split:
        c_sing = singular_CL(r_grid, rho0, float(rc))
        c_reg = cl_coherent - c_sing
        i_sing = hankel_transform(r_grid, c_sing, q_grid)
        i_reg_windowed = hankel_transform(r_grid, c_reg * w_tail, q_grid)
        i_total = i_sing + i_reg_windowed

    return {
        "CL": c_l,
        "CL_coherent": cl_coherent,
        "CL_sing": c_sing,
        "CL_reg": c_reg,
        "w_tail": w_tail,
        "I_full_windowed": i_full_windowed,
        "I_plateau_windowed": i_plateau_windowed,
        "I_coherent_windowed": i_coherent_windowed,
        "I_sing": i_sing,
        "I_reg_windowed": i_reg_windowed,
        "I_total": i_total,
        "rho0": float(rho0),
        "rho0_squared": float(rho0**2),
        "k0": float(k0),
        "r_max": r_max,
        "r_taper_start": float(r_taper_start),
        "rc": float(rc),
        "use_singular_split": bool(use_singular_split),
    }


def save_plots(
    output_dir: Path,
    r_grid: np.ndarray,
    q_grid: np.ndarray,
    k0: float,
    m_j: np.ndarray,
    c_l: np.ndarray,
    i_q: np.ndarray,
) -> None:
    """Save the requested diagnostic PNG figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    a = k0**2 / 3.0
    rho0 = a / np.pi

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(r_grid, g_mono(r_grid, k0), label="g(r)")
    ax.plot(r_grid, gp_mono(r_grid, k0), label="g'(r)")
    ax.plot(r_grid, gpp_mono(r_grid, k0), label="g''(r)")
    ax.set_xlabel("r")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "covariance_derivatives.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(r_grid, m_j, label="M_J(r)")
    ax.axhline(2.0 * a**2, color="tab:orange", linestyle="--", label="2 a^2")
    ax.axhline(4.0 * a**2, color="tab:green", linestyle="--", label="4 a^2")
    ax.set_xlabel("r")
    ax.set_ylabel("M_J")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "MJ_vs_r.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.loglog(r_grid, c_l, label="C_L(r)")
    ax.loglog(r_grid, rho0 / (2.0 * np.pi * r_grid**2), "--", label="rho0/(2 pi r^2)")
    ax.axhline(rho0**2, color="tab:green", linestyle=":", label="rho0^2")
    ax.set_xlabel("r")
    ax.set_ylabel("C_L")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "CL_loglog.png", dpi=180)
    plt.close(fig)

    c_coherent = coherent_CL(c_l, k0)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.semilogx(r_grid, c_coherent, label="C_L(r) - rho0^2")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("r")
    ax.set_ylabel("C_L coherent")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "CL_coherent.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.loglog(q_grid, i_q, label="I(Q)")
    ax.set_xlabel("Q")
    ax.set_ylabel("I(Q)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "IQ_loglog.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.semilogx(q_grid, q_grid * i_q, label="Q I(Q)")
    ax.axhline(k0**2 / 3.0, color="tab:orange", linestyle="--", label="k0^2/3")
    ax.set_xlabel("Q")
    ax.set_ylabel("Q I(Q)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "QIQ_plateau.png", dpi=180)
    plt.close(fig)


@dataclass(frozen=True)
class LineCorrelationResult:
    C: np.ndarray
    M: np.ndarray
    pzeros: np.ndarray


@dataclass(frozen=True)
class FourFieldScanResult:
    r_grid: np.ndarray
    Q_grid: np.ndarray
    k0: float
    rho0: float
    C_self: np.ndarray
    C_self_coherent: np.ndarray
    I_self: np.ndarray
    M_self: np.ndarray
    pzeros_self: np.ndarray
    rho_pairs: list[tuple[float, float]]
    C_cross: np.ndarray
    C_cross_coherent: np.ndarray
    I_cross: np.ndarray
    M_cross: np.ndarray
    pzeros_cross: np.ndarray


@dataclass(frozen=True)
class LineScatteringSpectrum:
    """Minimal package for a uniform-line intensity curve used by mask models."""

    Q_grid: np.ndarray
    I_L: np.ndarray
    rho0: float
    mu2: float | None = None
    k_radii: np.ndarray | None = None
    k_weights: np.ndarray | None = None


@dataclass(frozen=True)
class HeterogeneousLineScatteringResult:
    """Components of the independent heterogeneous-mask line intensity."""

    Q_grid: np.ndarray
    I_h: np.ndarray
    I_highQ: np.ndarray
    p_H: float
    sigma_H_squared: float
    alpha_H: float
    kappa_H: float
    uniform_component: np.ndarray
    smoothed_line_component: np.ndarray
    mask_component: np.ndarray
    approximation: str
    q_max: float
    rho0: float


def make_line_scattering_spectrum(
    Q_grid: np.ndarray,
    I_L: np.ndarray,
    rho0: float | None = None,
    *,
    mu2: float | None = None,
    k_radii: np.ndarray | None = None,
    k_weights: np.ndarray | None = None,
) -> LineScatteringSpectrum:
    """Create the smallest result object needed by ``heterogeneous_line_scattering``."""

    if rho0 is None:
        if mu2 is not None:
            rho0 = float(mu2) / (3.0 * np.pi)
        elif k_radii is not None:
            rho0 = rho0_from_k_radii(k_radii, k_weights=k_weights)
        else:
            raise ValueError("rho0, mu2, or k_radii is required.")
    if mu2 is None and k_radii is not None:
        mu2 = 3.0 * np.pi * float(rho0)
    return LineScatteringSpectrum(
        Q_grid=np.asarray(Q_grid, dtype=float),
        I_L=np.asarray(I_L, dtype=float),
        rho0=float(rho0),
        mu2=None if mu2 is None else float(mu2),
        k_radii=None if k_radii is None else np.asarray(k_radii, dtype=float),
        k_weights=None if k_weights is None else np.asarray(k_weights, dtype=float),
    )


def make_radial_line_spectrum(
    *,
    k0_nominal: float,
    r_sigma_k: float,
    random_seed: int = DEFAULT_RANDOM_SEED,
    k_distribution: KDistribution = DEFAULT_HETERO_K_DISTRIBUTION,  # type: ignore[assignment]
    num_modes_k: int = DEFAULT_HETERO_NUM_MODES_K,
    k_sampling: Literal["qmc", "random", "quadrature"] = DEFAULT_HETERO_K_SAMPLING,  # type: ignore[assignment]
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float | str]]:
    """Construct a radial line-wave spectrum and its basic moment metadata."""

    if k_sampling == "quadrature":
        k_radii, k_weights = make_radial_k_quadrature(
            int(num_modes_k),
            k_distribution,
            k0=float(k0_nominal),
            sigma_k=float(r_sigma_k) * float(k0_nominal),
        )
    elif k_sampling in {"qmc", "random"}:
        k_rng = np.random.default_rng(int(random_seed))
        k_sets = make_field_k_sets(
            int(num_modes_k),
            k_distribution,
            k_rng,
            k0=float(k0_nominal),
            r_sigma_k=float(r_sigma_k),
            shared_k_vectors=True,
            use_qmc_k=(k_sampling == "qmc"),
            qmc_seed=int(random_seed),
        )
        k_radii = k_radii_from_vectors(k_sets.psi1)
        k_weights = None
    else:
        raise ValueError("k_sampling must be 'qmc', 'random', or 'quadrature'.")

    a = gradient_variance_from_k_radii(k_radii, k_weights=k_weights)
    rho0 = a / np.pi
    mu2 = 3.0 * a
    k_eff = float(np.sqrt(mu2))
    k_mean = float(np.mean(k_radii) if k_weights is None else np.dot(k_weights, k_radii))
    k_std = float(
        np.std(k_radii)
        if k_weights is None
        else np.sqrt(np.dot(k_weights, (k_radii - k_mean) ** 2))
    )
    meta = {
        "k0_nominal": float(k0_nominal),
        "k_distribution": str(k_distribution),
        "num_modes_k": int(num_modes_k),
        "r_sigma_k": float(r_sigma_k),
        "random_seed": int(random_seed),
        "k_sampling": str(k_sampling),
        "k_eff": k_eff,
        "k_mean": k_mean,
        "k_std": k_std,
        "mu2": float(mu2),
        "rho0": float(rho0),
        "a": float(a),
    }
    return k_radii, k_weights, meta


def compute_uniform_line_scattering(
    *,
    k0_nominal: float = DEFAULT_K0,
    r_sigma_k: float,
    random_seed: int = DEFAULT_RANDOM_SEED,
    k_distribution: KDistribution = DEFAULT_HETERO_K_DISTRIBUTION,  # type: ignore[assignment]
    num_modes_k: int = DEFAULT_HETERO_NUM_MODES_K,
    k_sampling: Literal["qmc", "random", "quadrature"] = DEFAULT_HETERO_K_SAMPLING,  # type: ignore[assignment]
    Nr: int = DEFAULT_HETERO_NR,
    NQ: int = 256,
    r_min_factor: float = DEFAULT_HETERO_R_MIN_FACTOR,
    r_max_factor: float = DEFAULT_HETERO_R_MAX_FACTOR,
    Q_min_factor: float = 0.05,
    Q_max_factor: float = 20.0,
    N_samp_U: int = DEFAULT_HETERO_N_SAMP,
    N_samp_st: int = DEFAULT_HETERO_N_SAMP_ST,
    r_grid_mode: str = DEFAULT_HETERO_R_GRID_MODE,
    r_split_factor: float | None = DEFAULT_HETERO_R_SPLIT_FACTOR,
    Nr_small: int | None = DEFAULT_HETERO_NR_SMALL,
    jacobian_method: JacobianMethod = DEFAULT_HETERO_JACOBIAN_METHOD,  # type: ignore[assignment]
    use_asymptotic: bool = DEFAULT_HETERO_USE_ASYMPTOTIC,
    lowq_fit_bounds: tuple[float | None, float | None] | None = None,
    lowq_replace_max: float | None = None,
    lowq_fit_bounds_over_k: tuple[float | None, float | None] | None = None,
    lowq_replace_max_over_k: float | None = None,
    lowq_fit_bounds_over_k_eff: tuple[float | None, float | None] | None = DEFAULT_HETERO_LOWQ_FIT_BOUNDS_OVER_K_EFF,
    lowq_replace_max_over_k_eff: float | None = DEFAULT_HETERO_LOWQ_REPLACE_MAX_OVER_K_EFF,
    tail_start_fraction: float = DEFAULT_HETERO_TAIL_START_FRACTION,
    use_qmc: bool = DEFAULT_HETERO_USE_QMC,
    progress: bool = True,
) -> LineScatteringSpectrum:
    """Compute a uniform random-line scattering spectrum with reusable metadata.

    This is the core callable used before applying ``heterogeneous_line_scattering``.
    The defaults are tuned for the heterogeneous-mask workflows over the current
    Q range of interest.
    """

    k_radii, k_weights, meta = make_radial_line_spectrum(
        k0_nominal=k0_nominal,
        k_distribution=k_distribution,
        num_modes_k=num_modes_k,
        r_sigma_k=r_sigma_k,
        random_seed=random_seed,
        k_sampling=k_sampling,
    )
    k_eff = float(meta["k_eff"])
    k_mean = float(meta["k_mean"])
    rho0 = float(meta["rho0"])
    mu2 = float(meta["mu2"])
    if lowq_fit_bounds_over_k is not None:
        lowq_fit_bounds = tuple(
            None if value is None else float(value) * k_mean for value in lowq_fit_bounds_over_k
        )  # type: ignore[assignment]
    elif lowq_fit_bounds_over_k_eff is not None:
        lowq_fit_bounds = tuple(
            None if value is None else float(value) * k_eff for value in lowq_fit_bounds_over_k_eff
        )  # type: ignore[assignment]
    if lowq_replace_max_over_k is not None:
        lowq_replace_max = float(lowq_replace_max_over_k) * k_mean
    elif lowq_replace_max_over_k_eff is not None:
        lowq_replace_max = float(lowq_replace_max_over_k_eff) * k_eff
    if (
        lowq_fit_bounds is None
        and lowq_fit_bounds_over_k is None
        and lowq_fit_bounds_over_k_eff is None
    ):
        lowq_fit_bounds = (
            DEFAULT_HETERO_LOWQ_FIT_BOUNDS_OVER_K_EFF[0] * k_eff,
            DEFAULT_HETERO_LOWQ_FIT_BOUNDS_OVER_K_EFF[1] * k_eff,
        )
    if (
        lowq_replace_max is None
        and lowq_replace_max_over_k is None
        and lowq_replace_max_over_k_eff is None
    ):
        lowq_replace_max = DEFAULT_HETERO_LOWQ_REPLACE_MAX_OVER_K_EFF * k_eff

    r_grid = make_r_grid(
        float(r_min_factor) / k_eff,
        float(r_max_factor) / k_eff,
        int(Nr),
        mode=r_grid_mode,
        r_split=None if r_split_factor is None else float(r_split_factor) / k_eff,
        n_small=Nr_small,
    )
    q_grid = np.geomspace(float(Q_min_factor) * k_mean, float(Q_max_factor) * k_mean, int(NQ))

    m_j, c_l = compute_CL_general(
        r_grid,
        k_radii,
        None,
        k_weights=k_weights,
        use_qmc=bool(use_qmc),
        random_seed=int(random_seed),
        progress=progress,
        jacobian_method=jacobian_method,
        N_samp_U=int(N_samp_U),
        N_samp_st=int(N_samp_st),
        st_sampling="quadrature",
        n_jobs=1,
    )
    diag = compute_coherent_transform_diagnostics(
        r_grid,
        c_l,
        q_grid,
        rho0,
        r_taper_start=float(tail_start_fraction) * float(r_grid[-1]),
        use_asymptotic=use_asymptotic,
        lowq_fit_bounds=lowq_fit_bounds if use_asymptotic else None,
        lowq_replace_max=lowq_replace_max if use_asymptotic else None,
    )
    result = make_line_scattering_spectrum(
        q_grid,
        np.asarray(diag["I_coherent_windowed"], dtype=float),
        rho0,
        mu2=mu2,
        k_radii=k_radii,
        k_weights=k_weights,
    )
    object.__setattr__(result, "r_grid", r_grid)
    object.__setattr__(result, "M_J", m_j)
    object.__setattr__(result, "C_L", c_l)
    object.__setattr__(result, "I_L_original", diag["I_coherent_windowed_original"])
    object.__setattr__(result, "I_L_lowQ_asymptotic", diag["I_lowQ_asymptotic"])
    object.__setattr__(result, "lowQ_fit_mask", diag["lowQ_fit_mask"])
    object.__setattr__(result, "lowQ_replaced_mask", diag["lowQ_replaced_mask"])
    object.__setattr__(result, "lowQ_I0", diag["lowQ_I0"])
    object.__setattr__(result, "lowQ_I2", diag["lowQ_I2"])
    object.__setattr__(result, "lowQ_fit_min", diag["lowQ_fit_min"])
    object.__setattr__(result, "lowQ_fit_max", diag["lowQ_fit_max"])
    object.__setattr__(result, "lowQ_replace_max", diag["lowQ_replace_max"])
    object.__setattr__(result, "lowQ_relative_rmse", diag["lowQ_relative_rmse"])
    object.__setattr__(result, "use_asymptotic", bool(use_asymptotic))
    object.__setattr__(result, "tail_start_fraction", float(tail_start_fraction))
    object.__setattr__(result, "use_qmc", bool(use_qmc))
    object.__setattr__(result, "uniform_meta", meta)
    return result


def _get_line_result_value(line_result: object, names: Sequence[str]) -> object | None:
    for name in names:
        if isinstance(line_result, dict) and name in line_result:
            return line_result[name]
        try:
            if name in line_result:  # type: ignore[operator]
                return line_result[name]  # type: ignore[index]
        except (TypeError, KeyError):
            pass
        if hasattr(line_result, name):
            return getattr(line_result, name)
    return None


def _optional_float(value: object | None) -> float | None:
    """Return a finite scalar float, or ``None`` when metadata is absent."""

    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size != 1:
        return None
    scalar = float(arr.reshape(-1)[0])
    return scalar if np.isfinite(scalar) else None


def _coerce_line_scattering_spectrum(line_result: object) -> LineScatteringSpectrum:
    """Accept current dict/npz/dataclass outputs and return a uniform interface."""

    if isinstance(line_result, LineScatteringSpectrum):
        return line_result
    if isinstance(line_result, tuple) and len(line_result) >= 3:
        return make_line_scattering_spectrum(line_result[0], line_result[1], line_result[2])

    q_grid = _get_line_result_value(line_result, ("Q_grid", "q_grid", "Q", "q"))
    i_l = _get_line_result_value(line_result, ("I_L", "I_Q", "I_total", "I_self", "intensity"))
    if q_grid is None or i_l is None:
        raise ValueError("line_result must provide Q_grid/Q and I_L/I_Q/I_total.")

    rho0 = _get_line_result_value(line_result, ("rho0", "line_density"))
    mu2 = _get_line_result_value(line_result, ("mu2", "mu_2"))
    k_radii = _get_line_result_value(line_result, ("k_radii", "k_nodes"))
    k_weights = _get_line_result_value(line_result, ("k_weights", "weights"))
    k0 = _get_line_result_value(line_result, ("k0", "k_eff"))
    if rho0 is None:
        if mu2 is not None:
            rho0 = float(mu2) / (3.0 * np.pi)
        elif k_radii is not None:
            rho0 = rho0_from_k_radii(k_radii, k_weights=k_weights)
        elif k0 is not None:
            rho0 = float(k0) ** 2 / (3.0 * np.pi)
        else:
            raise ValueError("line_result must provide rho0, mu2, k_radii, or k0.")
    return make_line_scattering_spectrum(
        q_grid,
        i_l,
        float(rho0),
        mu2=None if mu2 is None else float(mu2),
        k_radii=None if k_radii is None else np.asarray(k_radii, dtype=float),
        k_weights=None if k_weights is None else np.asarray(k_weights, dtype=float),
    )


def mask_occupancy_parameters(k_H: float, b: float) -> dict[str, float | str]:
    """Return p_H, sigma_H^2, alpha_H, and kappa_H for the clipped mask."""

    if k_H < 0.0:
        raise ValueError("k_H must be nonnegative.")
    p_H = float(norm.sf(float(b)))
    sigma2 = float(p_H * (1.0 - p_H))
    alpha = float(norm.pdf(float(b)) * float(k_H) / np.sqrt(6.0 * np.pi))
    tiny = np.finfo(float).tiny
    if p_H <= tiny or sigma2 <= tiny and p_H < 0.5:
        return {"p_H": p_H, "sigma_H_squared": sigma2, "alpha_H": alpha, "kappa_H": 0.0, "state": "empty"}
    if (1.0 - p_H) <= tiny or sigma2 <= tiny and p_H >= 0.5:
        return {"p_H": p_H, "sigma_H_squared": sigma2, "alpha_H": alpha, "kappa_H": 0.0, "state": "full"}
    kappa = alpha / sigma2 if alpha > 0.0 else 0.0
    return {"p_H": p_H, "sigma_H_squared": sigma2, "alpha_H": alpha, "kappa_H": float(kappa), "state": "partial"}


def smooth_mask_kernel(Q: np.ndarray | float, kappa_H: float) -> np.ndarray:
    """K_H(Q)=8*pi*kappa_H/(Q^2+kappa_H^2)^2."""

    Q = np.asarray(Q, dtype=float)
    kappa_H = float(kappa_H)
    if kappa_H <= 0.0:
        return np.zeros_like(Q, dtype=float)
    return 8.0 * np.pi * kappa_H / (Q * Q + kappa_H * kappa_H) ** 2


def _validate_q_grid(q_grid: np.ndarray, i_l: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_grid = np.asarray(q_grid, dtype=float)
    i_l = np.asarray(i_l, dtype=float)
    if q_grid.shape != i_l.shape:
        raise ValueError("Q_grid and I_L must have the same shape.")
    if q_grid.ndim != 1:
        raise ValueError("Q_grid and I_L must be one-dimensional.")
    if len(q_grid) < 2:
        raise ValueError("At least two Q samples are required.")
    order = np.argsort(q_grid)
    q_sorted = q_grid[order]
    i_sorted = i_l[order]
    if np.any(np.diff(q_sorted) <= 0.0):
        raise ValueError("Q_grid values must be unique.")
    if np.any(q_sorted < 0.0):
        raise ValueError("Q_grid values must be nonnegative.")
    return q_sorted, i_sorted


def smoothing_operator_A(
    Q: np.ndarray,
    q_grid: np.ndarray,
    I_L: np.ndarray,
    kappa_H: float,
    rho0: float,
    *,
    q_max: float | None = None,
    refine_grid: bool = True,
    points_per_kappa: float = 5.0,
    max_refined_points: int = 50_000,
    lowq_I0: float | None = None,
    lowq_I2: float | None = None,
) -> tuple[np.ndarray, float]:
    """Evaluate A_kappa[I_L](Q) using finite data plus low/high-Q extrapolation."""

    Q = np.asarray(Q, dtype=float)
    q_grid, I_L = _validate_q_grid(q_grid, I_L)
    kappa_H = float(kappa_H)
    rho0 = float(rho0)
    if kappa_H <= 0.0:
        return I_L.copy() if Q.shape == I_L.shape and np.allclose(Q, q_grid) else np.interp(Q, q_grid, I_L), float(q_grid[-1])
    if q_max is None:
        q_max = float(q_grid[-1])
    q_max = float(q_max)
    finite = q_grid <= q_max
    if np.count_nonzero(finite) < 2:
        raise ValueError("q_max must include at least two available I_L grid points.")
    q_finite = q_grid[finite]
    i_finite = I_L[finite]
    q_cut = float(q_finite[-1])

    if lowq_I0 is not None and lowq_I2 is not None and q_finite[0] > 0.0:
        lowq_I0 = float(lowq_I0)
        lowq_I2 = float(lowq_I2)
        if np.isfinite(lowq_I0) and np.isfinite(lowq_I2):
            q0 = float(q_finite[0])
            target_step = kappa_H / max(float(points_per_kappa), 1.0) if kappa_H > 0.0 else q0
            n_low = max(8, int(np.ceil(q0 / max(target_step, np.finfo(float).tiny))) + 1)
            q_low = np.linspace(0.0, q0, n_low, endpoint=False)
            i_low = lowq_I0 + lowq_I2 * q_low * q_low
            q_finite = np.concatenate([q_low, q_finite])
            i_finite = np.concatenate([i_low, i_finite])

    if refine_grid and kappa_H > 0.0 and q_finite.size >= 2:
        target_step = kappa_H / max(float(points_per_kappa), 1.0)
        current_step = float(np.max(np.diff(q_finite)))
        needed = int(np.ceil((q_cut - float(q_finite[0])) / max(target_step, np.finfo(float).tiny))) + 1
        if current_step > target_step and needed <= int(max_refined_points):
            q_refined = np.linspace(float(q_finite[0]), q_cut, needed)
            i_refined = np.interp(q_refined, q_finite, i_finite)
            q_finite = q_refined
            i_finite = i_refined

    out = np.empty_like(Q, dtype=float)
    for idx, q_value in enumerate(Q):
        q_value = float(q_value)
        if np.isclose(q_value, 0.0, rtol=0.0, atol=1.0e-14):
            # A_kappa[I_L](0) = 4*kappa/pi * int q^2 I_L(q)/(q^2+kappa^2)^2 dq.
            integrand = q_finite * q_finite * i_finite / (q_finite * q_finite + kappa_H * kappa_H) ** 2
            finite_part = 4.0 * kappa_H / np.pi * float(simpson(integrand, x=q_finite))
            tail_part = 2.0 * kappa_H * rho0 / (q_cut * q_cut + kappa_H * kappa_H)
            out[idx] = finite_part + tail_part
            continue

        # A_kappa^<[I_L](Q), evaluated only over the supplied finite q grid.
        kernel = 1.0 / ((q_value - q_finite) ** 2 + kappa_H * kappa_H)
        kernel -= 1.0 / ((q_value + q_finite) ** 2 + kappa_H * kappa_H)
        finite_part = kappa_H / (np.pi * q_value) * float(simpson(q_finite * i_finite * kernel, x=q_finite))
        # A_kappa^>(Q;Q_max), using I_L(q) ~= pi*rho0/q above q_cut.
        tail_part = rho0 / q_value * (
            np.arctan((q_cut + q_value) / kappa_H)
            - np.arctan((q_cut - q_value) / kappa_H)
        )
        out[idx] = finite_part + tail_part
    return out, q_cut


def heterogeneous_highQ_intensity(
    Q: np.ndarray,
    rho0: float,
    p_H: float,
    alpha_H: float,
    *,
    high_q_min: float | None = None,
) -> np.ndarray:
    """High-Q heterogeneous asymptote, exposed separately from the smooth model."""

    Q = np.asarray(Q, dtype=float)
    if high_q_min is not None and np.any(Q < float(high_q_min)):
        warnings.warn(
            "heterogeneous high-Q approximation evaluated below high_q_min.",
            RuntimeWarning,
            stacklevel=2,
        )
    out = np.zeros_like(Q, dtype=float)
    positive = Q > 0.0
    q_pos = Q[positive]
    if q_pos.size:
        out[positive] = (
            np.pi * float(p_H) * float(rho0) / q_pos
            - 2.0 * float(alpha_H) * float(rho0) / (q_pos * q_pos)
            + 8.0 * np.pi * float(alpha_H) * float(rho0) ** 2 / (q_pos**4)
        )
    singular = ~positive
    if np.any(singular) and (p_H != 0.0 or alpha_H != 0.0):
        out[singular] = np.inf
    return out


def heterogeneous_line_scattering(
    line_result: object,
    k_H: float,
    b: float,
    *,
    approximation: Literal["smooth", "highQ", "high_q"] = "smooth",
    q_max: float | None = None,
    high_q_min: float | None = None,
    return_components: bool = False,
) -> np.ndarray | HeterogeneousLineScatteringResult:
    """Apply an independent smooth clipped mask to an existing uniform-line result."""

    spectrum = _coerce_line_scattering_spectrum(line_result)
    Q, I_L = _validate_q_grid(spectrum.Q_grid, spectrum.I_L)
    lowq_I0 = _optional_float(_get_line_result_value(line_result, ("lowQ_I0", "lowq_I0", "I0")))
    lowq_I2 = _optional_float(_get_line_result_value(line_result, ("lowQ_I2", "lowq_I2", "I2")))
    params = mask_occupancy_parameters(float(k_H), float(b))
    p_H = float(params["p_H"])
    sigma2 = float(params["sigma_H_squared"])
    alpha = float(params["alpha_H"])
    kappa = float(params["kappa_H"])
    state = str(params["state"])

    if state == "full":
        uniform_component = I_L.copy()
        smoothed_component = np.zeros_like(I_L)
        mask_component = np.zeros_like(I_L)
        smooth_total = I_L.copy()
        q_cut = float(Q[-1])
    elif state == "empty":
        uniform_component = np.zeros_like(I_L)
        smoothed_component = np.zeros_like(I_L)
        mask_component = np.zeros_like(I_L)
        smooth_total = np.zeros_like(I_L)
        q_cut = float(Q[-1])
    else:
        # I_h^smooth = p_H^2 I_L + sigma_H^2 A_kappa[I_L] + rho0^2 sigma_H^2 K_H.
        A_kappa, q_cut = smoothing_operator_A(
            Q,
            Q,
            I_L,
            kappa,
            spectrum.rho0,
            q_max=q_max,
            lowq_I0=lowq_I0,
            lowq_I2=lowq_I2,
        )
        uniform_component = p_H * p_H * I_L
        smoothed_component = sigma2 * A_kappa
        mask_component = spectrum.rho0**2 * sigma2 * smooth_mask_kernel(Q, kappa)
        smooth_total = uniform_component + smoothed_component + mask_component

    approximation = str(approximation)
    if approximation == "high_q":
        approximation = "highQ"
    if approximation == "smooth":
        total = smooth_total
    elif approximation == "highQ":
        total = heterogeneous_highQ_intensity(
            Q,
            spectrum.rho0,
            p_H,
            alpha,
            high_q_min=high_q_min,
        )
    else:
        raise ValueError("approximation must be 'smooth' or 'highQ'.")

    if not return_components:
        return total
    if approximation == "highQ":
        highQ = total
    else:
        highQ = heterogeneous_highQ_intensity(
            Q,
            spectrum.rho0,
            p_H,
            alpha,
            high_q_min=high_q_min,
        )
    return HeterogeneousLineScatteringResult(
        Q_grid=Q,
        I_h=total,
        I_highQ=highQ,
        p_H=p_H,
        sigma_H_squared=sigma2,
        alpha_H=alpha,
        kappa_H=kappa,
        uniform_component=uniform_component,
        smoothed_line_component=smoothed_component,
        mask_component=mask_component,
        approximation=approximation,
        q_max=q_cut,
        rho0=float(spectrum.rho0),
    )


def make_qmc_normals(dim: int, n_samp: int, seed: int = DEFAULT_RANDOM_SEED) -> np.ndarray:
    """Sobol normal samples with shape ``(n_samp, dim)``.

    ``n_samp`` must be a power of two because Sobol sampling uses
    ``random_base2``. This matches the two-wave conditional-sampling API, where
    users specify the actual sample count rather than the Sobol exponent.
    """

    sampler = qmc.Sobol(d=int(dim), scramble=True, seed=seed)
    u = sampler.random_base2(m=qmc_power_from_n_samp(n_samp))
    u = np.clip(u, 1.0e-12, 1.0 - 1.0e-12)
    return norm.ppf(u)


def covariance_sqrt(sigma: np.ndarray, jitter: float = 1.0e-10) -> np.ndarray:
    """Return L with L @ L.T approximately sigma."""

    sigma = np.asarray(sigma, dtype=float)
    scale = max(float(np.max(np.diag(sigma))), 1.0)
    try:
        return np.linalg.cholesky(sigma + jitter * scale * np.eye(sigma.shape[0]))
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(sigma)
        if np.min(eigvals) < -100.0 * jitter * scale:
            print(f"warning: covariance has negative eigenvalue {np.min(eigvals):.3e}")
        eigvals = np.clip(eigvals, 0.0, None)
        return eigvecs @ np.diag(np.sqrt(eigvals))


def bc_from_g(r: float, k0: float = DEFAULT_K0) -> tuple[float, float, float, float]:
    """Return a, g, b=-g'/r, and c=-g'' for r along the z-axis."""

    if r <= 0.0:
        raise ValueError("r must be positive.")
    a = float(k0) ** 2 / 3.0
    g = _as_float(g_mono(r, k0))
    gp = _as_float(gp_mono(r, k0))
    gpp = _as_float(gpp_mono(r, k0))
    return bc_from_stats(r, a, g, gp, gpp)


def bc_from_stats(r: float, a: float, g: float, gp: float, gpp: float) -> tuple[float, float, float, float]:
    """Return a, g, b=-g'/r, and c=-g'' from radial covariance statistics."""

    if r <= 0.0:
        raise ValueError("r must be positive.")
    k_eff = np.sqrt(max(3.0 * float(a), 0.0))
    b = a if abs(k_eff * r) < SMALL_X else -gp / float(r)
    c = -gpp
    return a, g, b, c


def self_conditional_covariance_from_stats(r: float, a: float, g: float, gp: float, gpp: float) -> np.ndarray:
    """6x6 gradient covariance for one field conditioned on psi(0)=psi(r)=0."""

    a, g, b, c = bc_from_stats(r, a, g, gp, gpp)
    denom = max(1.0 - g * g, np.finfo(float).tiny)
    az = a - gp * gp / denom
    cz = c - g * gp * gp / denom
    return np.array(
        [
            [a, 0.0, 0.0, b, 0.0, 0.0],
            [0.0, a, 0.0, 0.0, b, 0.0],
            [0.0, 0.0, az, 0.0, 0.0, cz],
            [b, 0.0, 0.0, a, 0.0, 0.0],
            [0.0, b, 0.0, 0.0, a, 0.0],
            [0.0, 0.0, cz, 0.0, 0.0, az],
        ],
        dtype=float,
    )


def self_conditional_covariance(r: float, k0: float = DEFAULT_K0) -> np.ndarray:
    """6x6 gradient covariance for a monochromatic wave."""

    a = float(k0) ** 2 / 3.0
    g = _as_float(g_mono(r, k0))
    gp = _as_float(gp_mono(r, k0))
    gpp = _as_float(gpp_mono(r, k0))
    return self_conditional_covariance_from_stats(r, a, g, gp, gpp)


def cross_zero_covariance_from_g(g: float, rho13: float, rho24: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, rho13 * g, 0.0],
            [0.0, 1.0, 0.0, rho24 * g],
            [rho13 * g, 0.0, 1.0, 0.0],
            [0.0, rho24 * g, 0.0, 1.0],
        ],
        dtype=float,
    )


def cross_zero_covariance(r: float, k0: float, rho13: float, rho24: float) -> np.ndarray:
    _, g, _, _ = bc_from_g(r, k0)
    return cross_zero_covariance_from_g(g, rho13, rho24)


def cross_gradient_covariance_blocks_from_stats(
    r: float,
    a: float,
    g: float,
    gp: float,
    gpp: float,
    rho13: float,
    rho24: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Sigma_YY and Sigma_YZ for the 12D cross-gradient vector."""

    a, _, b, c = bc_from_stats(r, a, g, gp, gpp)
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))
    D = np.diag([b, b, c])
    q = np.array([[0.0], [0.0], [gp]])
    z = np.zeros((3, 1))
    yy = np.block(
        [
            [a * I3, Z3, rho13 * D, Z3],
            [Z3, a * I3, Z3, rho24 * D],
            [rho13 * D, Z3, a * I3, Z3],
            [Z3, rho24 * D, Z3, a * I3],
        ]
    )
    yz = np.block(
        [
            [z, z, -rho13 * q, z],
            [z, z, z, -rho24 * q],
            [rho13 * q, z, z, z],
            [z, rho24 * q, z, z],
        ]
    )
    return yy, yz


def cross_gradient_covariance_blocks(
    r: float,
    k0: float,
    rho13: float,
    rho24: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Sigma_YY and Sigma_YZ for a monochromatic wave."""

    a = float(k0) ** 2 / 3.0
    g = _as_float(g_mono(r, k0))
    gp = _as_float(gp_mono(r, k0))
    gpp = _as_float(gpp_mono(r, k0))
    return cross_gradient_covariance_blocks_from_stats(r, a, g, gp, gpp, rho13, rho24)


def cross_conditional_covariance_from_stats(
    r: float,
    a: float,
    g: float,
    gp: float,
    gpp: float,
    rho13: float,
    rho24: float,
) -> np.ndarray:
    zz = cross_zero_covariance_from_g(g, rho13, rho24)
    yy, yz = cross_gradient_covariance_blocks_from_stats(r, a, g, gp, gpp, rho13, rho24)
    return yy - yz @ np.linalg.solve(zz, yz.T)


def cross_conditional_covariance(r: float, k0: float, rho13: float, rho24: float) -> np.ndarray:
    a = float(k0) ** 2 / 3.0
    g = _as_float(g_mono(r, k0))
    gp = _as_float(gp_mono(r, k0))
    gpp = _as_float(gpp_mono(r, k0))
    zz = cross_zero_covariance_from_g(g, rho13, rho24)
    yy, yz = cross_gradient_covariance_blocks_from_stats(r, a, g, gp, gpp, rho13, rho24)
    return yy - yz @ np.linalg.solve(zz, yz.T)


def pzeros_self_from_g(g: np.ndarray | float) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    return 1.0 / (2.0 * np.pi * np.sqrt(np.maximum(1.0 - g * g, np.finfo(float).tiny)))


def pzeros_self(r: np.ndarray | float, k0: float = DEFAULT_K0) -> np.ndarray:
    return pzeros_self_from_g(g_mono(r, k0))


def pzeros_cross_from_g(g: np.ndarray | float, rho13: float, rho24: float) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    d13 = np.maximum(1.0 - (rho13 * g) ** 2, np.finfo(float).tiny)
    d24 = np.maximum(1.0 - (rho24 * g) ** 2, np.finfo(float).tiny)
    return 1.0 / ((2.0 * np.pi) ** 2 * np.sqrt(d13 * d24))


def pzeros_cross(r: np.ndarray | float, k0: float, rho13: float, rho24: float) -> np.ndarray:
    return pzeros_cross_from_g(g_mono(r, k0), rho13, rho24)


def estimate_M_self_from_stats(r: float, a: float, g: float, gp: float, gpp: float, xi12: np.ndarray) -> float:
    sigma = self_conditional_covariance_from_stats(r, a, g, gp, gpp)
    zero = np.zeros_like(sigma)
    sigma_w = np.block([[sigma, zero], [zero, sigma]])
    samples = xi12 @ covariance_sqrt(sigma_w).T
    u0 = samples[:, 0:3]
    ur = samples[:, 3:6]
    v0 = samples[:, 6:9]
    vr = samples[:, 9:12]
    return float(np.mean(np.linalg.norm(np.cross(u0, v0), axis=1) * np.linalg.norm(np.cross(ur, vr), axis=1)))


def estimate_M_self(r: float, k0: float, xi12: np.ndarray) -> float:
    sigma = self_conditional_covariance(r, k0)
    zero = np.zeros_like(sigma)
    sigma_w = np.block([[sigma, zero], [zero, sigma]])
    samples = xi12 @ covariance_sqrt(sigma_w).T
    u0 = samples[:, 0:3]
    ur = samples[:, 3:6]
    v0 = samples[:, 6:9]
    vr = samples[:, 9:12]
    return float(np.mean(np.linalg.norm(np.cross(u0, v0), axis=1) * np.linalg.norm(np.cross(ur, vr), axis=1)))


def estimate_M_cross_from_stats(
    r: float,
    a: float,
    g: float,
    gp: float,
    gpp: float,
    rho13: float,
    rho24: float,
    xi12: np.ndarray,
) -> float:
    sigma = cross_conditional_covariance_from_stats(r, a, g, gp, gpp, rho13, rho24)
    samples = xi12 @ covariance_sqrt(sigma).T
    g10 = samples[:, 0:3]
    g20 = samples[:, 3:6]
    g3r = samples[:, 6:9]
    g4r = samples[:, 9:12]
    return float(np.mean(np.linalg.norm(np.cross(g10, g20), axis=1) * np.linalg.norm(np.cross(g3r, g4r), axis=1)))


def estimate_M_cross(r: float, k0: float, rho13: float, rho24: float, xi12: np.ndarray) -> float:
    sigma = cross_conditional_covariance(r, k0, rho13, rho24)
    samples = xi12 @ covariance_sqrt(sigma).T
    g10 = samples[:, 0:3]
    g20 = samples[:, 3:6]
    g3r = samples[:, 6:9]
    g4r = samples[:, 9:12]
    return float(np.mean(np.linalg.norm(np.cross(g10, g20), axis=1) * np.linalg.norm(np.cross(g3r, g4r), axis=1)))


def compute_self_correlation(
    r_grid: np.ndarray,
    k0: float,
    xi12: np.ndarray,
    *,
    progress: bool = True,
) -> LineCorrelationResult:
    r_grid = np.asarray(r_grid, dtype=float)
    M = np.empty_like(r_grid)
    t0 = time.perf_counter()
    report_every = max(1, len(r_grid) // 20)
    for idx, r in enumerate(r_grid):
        M[idx] = estimate_M_self(float(r), k0, xi12)
        if progress and ((idx + 1) % report_every == 0 or idx + 1 == len(r_grid)):
            print(f"self: {idx + 1}/{len(r_grid)} r values ({time.perf_counter() - t0:.1f}s)")
    pz_pair = pzeros_self(r_grid, k0) ** 2
    return LineCorrelationResult(C=pz_pair * M, M=M, pzeros=pz_pair)


def compute_cross_correlation(
    r_grid: np.ndarray,
    k0: float,
    rho13: float,
    rho24: float,
    xi12: np.ndarray,
    *,
    progress: bool = True,
) -> LineCorrelationResult:
    r_grid = np.asarray(r_grid, dtype=float)
    M = np.empty_like(r_grid)
    t0 = time.perf_counter()
    report_every = max(1, len(r_grid) // 20)
    for idx, r in enumerate(r_grid):
        M[idx] = estimate_M_cross(float(r), k0, rho13, rho24, xi12)
        if progress and ((idx + 1) % report_every == 0 or idx + 1 == len(r_grid)):
            print(
                f"cross rho13={rho13:g}, rho24={rho24:g}: "
                f"{idx + 1}/{len(r_grid)} r values ({time.perf_counter() - t0:.1f}s)"
            )
    pz = pzeros_cross(r_grid, k0, rho13, rho24)
    return LineCorrelationResult(C=pz * M, M=M, pzeros=pz)


def compute_wave_stats_for_r_grid(
    r_grid: np.ndarray,
    k_radii: np.ndarray,
    *,
    k_weights: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Return radial covariance statistics needed by the four-field line model."""

    r_grid = np.asarray(r_grid, dtype=float)
    a = gradient_variance_from_k_radii(k_radii, k_weights=k_weights)
    return {
        "a": float(a),
        "rho0": float(a / np.pi),
        "k_eff": float(np.sqrt(3.0 * a)),
        "g": g_radial_numeric(r_grid, k_radii, k_weights=k_weights),
        "gp": gp_radial_numeric(r_grid, k_radii, k_weights=k_weights),
        "gpp": gpp_radial_numeric(r_grid, k_radii, k_weights=k_weights),
    }


def compute_self_correlation_from_wave_stats(
    r_grid: np.ndarray,
    a: float,
    g: np.ndarray,
    gp: np.ndarray,
    gpp: np.ndarray,
    xi12: np.ndarray,
    *,
    progress: bool = True,
) -> LineCorrelationResult:
    """Compute the four-field auto term from precomputed radial wave statistics."""

    r_grid = np.asarray(r_grid, dtype=float)
    g = np.asarray(g, dtype=float)
    gp = np.asarray(gp, dtype=float)
    gpp = np.asarray(gpp, dtype=float)
    M = np.empty_like(r_grid)
    t0 = time.perf_counter()
    report_every = max(1, len(r_grid) // 20)
    for idx, r in enumerate(r_grid):
        M[idx] = estimate_M_self_from_stats(float(r), float(a), float(g[idx]), float(gp[idx]), float(gpp[idx]), xi12)
        if progress and ((idx + 1) % report_every == 0 or idx + 1 == len(r_grid)):
            print(f"self: {idx + 1}/{len(r_grid)} r values ({time.perf_counter() - t0:.1f}s)")
    pz_pair = pzeros_self_from_g(g) ** 2
    return LineCorrelationResult(C=pz_pair * M, M=M, pzeros=pz_pair)


def compute_cross_correlation_from_wave_stats(
    r_grid: np.ndarray,
    a: float,
    g: np.ndarray,
    gp: np.ndarray,
    gpp: np.ndarray,
    rho13: float,
    rho24: float,
    xi12: np.ndarray,
    *,
    progress: bool = True,
) -> LineCorrelationResult:
    """Compute the four-field cross term from precomputed radial wave statistics."""

    r_grid = np.asarray(r_grid, dtype=float)
    g = np.asarray(g, dtype=float)
    gp = np.asarray(gp, dtype=float)
    gpp = np.asarray(gpp, dtype=float)
    M = np.empty_like(r_grid)
    t0 = time.perf_counter()
    report_every = max(1, len(r_grid) // 20)
    for idx, r in enumerate(r_grid):
        M[idx] = estimate_M_cross_from_stats(
            float(r),
            float(a),
            float(g[idx]),
            float(gp[idx]),
            float(gpp[idx]),
            rho13,
            rho24,
            xi12,
        )
        if progress and ((idx + 1) % report_every == 0 or idx + 1 == len(r_grid)):
            print(
                f"cross rho13={rho13:g}, rho24={rho24:g}: "
                f"{idx + 1}/{len(r_grid)} r values ({time.perf_counter() - t0:.1f}s)"
            )
    pz = pzeros_cross_from_g(g, rho13, rho24)
    return LineCorrelationResult(C=pz * M, M=M, pzeros=pz)


def default_rho_pairs() -> list[tuple[float, float]]:
    diag = [(rho, rho) for rho in (0.0, 0.25, 0.5, 0.75, 0.95, 1.0)]
    extra = [(1.0, 0.0), (0.0, 1.0), (1.0, 0.5), (0.5, 1.0)]
    return diag + [pair for pair in extra if pair not in diag]


def run_four_field_scan(
    *,
    k0: float = DEFAULT_K0,
    r_min: float | None = None,
    r_max: float | None = None,
    Nr: int = 3000,
    Q_min: float | None = None,
    Q_max: float | None = None,
    NQ: int = 300,
    n_samp: int = DEFAULT_N_SAMP,
    qmc_power: int | None = None,
    seed: int = DEFAULT_RANDOM_SEED,
    rho_pairs: Sequence[tuple[float, float]] | None = None,
    r_taper_start: float | None = None,
    progress: bool = True,
) -> FourFieldScanResult:
    if qmc_power is not None:
        # Backward-compatible alias for older callers. New notebooks and APIs
        # should pass n_samp directly, matching the two-wave sampling helpers.
        n_samp = 2 ** int(qmc_power)
    if r_min is None:
        r_min = 1.0e-3 / k0
    if r_max is None:
        r_max = 300.0 / k0
    if Q_min is None:
        Q_min = 1.0e-2 * k0
    if Q_max is None:
        Q_max = 1.0e2 * k0
    r_grid = np.linspace(r_min, r_max, int(Nr))
    Q_grid = np.logspace(np.log10(Q_min), np.log10(Q_max), int(NQ))
    if r_taper_start is None:
        r_taper_start = 0.75 * r_max
    xi12 = make_qmc_normals(12, n_samp, seed)
    rho0 = k0**2 / (3.0 * np.pi)
    w = tail_window(r_grid, r_taper_start, r_max)

    self_res = compute_self_correlation(r_grid, k0, xi12, progress=progress)
    C_self_coherent = self_res.C - rho0**2
    I_self = hankel_transform(r_grid, C_self_coherent * w, Q_grid)

    pairs = list(default_rho_pairs() if rho_pairs is None else rho_pairs)
    C_cross = np.empty((len(pairs), len(r_grid)))
    C_cross_coherent = np.empty_like(C_cross)
    I_cross = np.empty((len(pairs), len(Q_grid)))
    M_cross = np.empty_like(C_cross)
    pz_cross = np.empty_like(C_cross)
    for idx, (rho13, rho24) in enumerate(pairs):
        res = compute_cross_correlation(r_grid, k0, rho13, rho24, xi12, progress=progress)
        C_cross[idx] = res.C
        C_cross_coherent[idx] = res.C - rho0**2
        I_cross[idx] = hankel_transform(r_grid, C_cross_coherent[idx] * w, Q_grid)
        M_cross[idx] = res.M
        pz_cross[idx] = res.pzeros

    return FourFieldScanResult(
        r_grid=r_grid,
        Q_grid=Q_grid,
        k0=k0,
        rho0=rho0,
        C_self=self_res.C,
        C_self_coherent=C_self_coherent,
        I_self=I_self,
        M_self=self_res.M,
        pzeros_self=self_res.pzeros,
        rho_pairs=pairs,
        C_cross=C_cross,
        C_cross_coherent=C_cross_coherent,
        I_cross=I_cross,
        M_cross=M_cross,
        pzeros_cross=pz_cross,
    )


def save_four_field_scan_outputs(result: FourFieldScanResult, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rho_pairs = np.array(result.rho_pairs, dtype=float)
    np.savez_compressed(
        output_dir / "four_field_line_scan.npz",
        r_grid=result.r_grid,
        Q_grid=result.Q_grid,
        k0=result.k0,
        rho0=result.rho0,
        rho_pairs=rho_pairs,
        C_self=result.C_self,
        C_self_coherent=result.C_self_coherent,
        I_self=result.I_self,
        M_self=result.M_self,
        pzeros_self=result.pzeros_self,
        C_cross=result.C_cross,
        C_cross_coherent=result.C_cross_coherent,
        I_cross=result.I_cross,
        M_cross=result.M_cross,
        pzeros_cross=result.pzeros_cross,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k0", type=float, default=DEFAULT_K0)
    parser.add_argument("--k-distribution", choices=["single_shell", "gaussian_radial", "uniform_band"], default="single_shell")
    parser.add_argument("--num-modes", type=int, default=4096)
    parser.add_argument("--r-sigma-k", type=float, default=0.15)
    parser.add_argument("--r-k-min", type=float, default=0.7)
    parser.add_argument("--r-k-max", type=float, default=1.3)
    parser.add_argument("--shared-k-vectors", action="store_true")
    parser.add_argument("--r-min", type=float, default=None)
    parser.add_argument("--r-max", type=float, default=None)
    parser.add_argument("--Nr", type=int, default=DEFAULT_NR)
    parser.add_argument("--r-grid", choices=["mixed", "linear"], default="mixed")
    parser.add_argument("--r-split", type=float, default=None)
    parser.add_argument("--Nr-small", type=int, default=None)
    parser.add_argument("--Q-min", type=float, default=None)
    parser.add_argument("--Q-max", type=float, default=None)
    parser.add_argument("--NQ", type=int, default=DEFAULT_NQ)
    parser.add_argument("--N-samp", type=int, default=DEFAULT_N_SAMP)
    parser.add_argument("--jacobian-method", choices=["direct_12d", "conditional_6d_2d"], default="direct_12d")
    parser.add_argument("--N-samp-U", type=int, default=None)
    parser.add_argument("--N-samp-st", type=int, default=2**8)
    parser.add_argument("--U-sampling", choices=["qmc", "random"], default=None)
    parser.add_argument("--st-sampling", choices=["quadrature", "qmc"], default="quadrature")
    parser.add_argument("--st-transform", choices=["rational", "logistic"], default="rational")
    parser.add_argument("--st-tau", type=float, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--no-qmc", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tail-start", type=float, default=None)
    parser.add_argument("--no-singular-split", action="store_true")
    parser.add_argument("--singular-rc", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k0 = float(args.k0)
    if args.k_distribution != "single_shell":
        k_rng = np.random.default_rng(int(args.seed))
        k_sets = make_field_k_sets(
            int(args.num_modes),
            args.k_distribution,
            k_rng,
            k0=k0,
            r_sigma_k=float(args.r_sigma_k),
            r_k_min=float(args.r_k_min),
            r_k_max=float(args.r_k_max),
            shared_k_vectors=bool(args.shared_k_vectors),
        )
        k0_1 = effective_k0_from_k_vectors(k_sets.psi1)
        k0_2 = effective_k0_from_k_vectors(k_sets.psi2)
        k0 = float(np.sqrt(0.5 * (k0_1 * k0_1 + k0_2 * k0_2)))
        print(
            f"k-distribution = {args.k_distribution}; "
            f"effective k0 = {k0:.12g} from sqrt(<|k|^2>)"
        )
    r_min = DEFAULT_R_MIN_FACTOR / k0 if args.r_min is None else float(args.r_min)
    r_max = DEFAULT_R_MAX_FACTOR / k0 if args.r_max is None else float(args.r_max)
    q_min = DEFAULT_Q_MIN_FACTOR * k0 if args.Q_min is None else float(args.Q_min)
    q_max = DEFAULT_Q_MAX_FACTOR * k0 if args.Q_max is None else float(args.Q_max)
    output_dir = Path(args.output_dir)

    if r_min <= 0.0 or r_max <= r_min:
        raise ValueError("Require 0 < r_min < r_max.")
    if q_min <= 0.0 or q_max <= q_min:
        raise ValueError("Require 0 < Q_min < Q_max.")

    r_grid = make_r_grid(
        r_min,
        r_max,
        int(args.Nr),
        mode=args.r_grid,
        r_split=args.r_split,
        n_small=args.Nr_small,
    )
    q_grid = np.logspace(np.log10(q_min), np.log10(q_max), int(args.NQ))
    a = k0**2 / 3.0
    rho0 = a / np.pi

    print(f"a = k0^2/3 = {a:.12g}")
    print(f"rho0 = a/pi = {rho0:.12g}")
    print(f"rho0^2 background plateau = {rho0**2:.12g}")
    print(f"expected M_J(0+) = 2*a^2 = {2.0 * a**2:.12g}")
    print(f"expected M_J(infinity) = 4*a^2 = {4.0 * a**2:.12g}")
    print(f"expected Q*I(Q) plateau = k0^2/3 = {k0**2 / 3.0:.12g}")

    m_j, c_l = compute_CL(
        r_grid,
        k0,
        int(args.N_samp),
        use_qmc=not args.no_qmc,
        random_seed=int(args.seed),
        progress=not args.quiet,
        jacobian_method=args.jacobian_method,
        N_samp_U=args.N_samp_U,
        N_samp_st=int(args.N_samp_st),
        U_sampling=args.U_sampling,
        st_sampling=args.st_sampling,
        st_transform=args.st_transform,
        st_tau=args.st_tau,
        n_jobs=int(args.n_jobs),
    )
    use_singular_split = not args.no_singular_split
    r_taper_start = 0.75 * r_max if args.tail_start is None else float(args.tail_start)
    diagnostics = compute_transform_diagnostics(
        r_grid,
        c_l,
        q_grid,
        k0,
        r_taper_start=r_taper_start,
        use_singular_split=use_singular_split,
        rc=args.singular_rc,
    )
    c_coherent = diagnostics["CL_coherent"]
    c_transform = diagnostics["CL_reg"] * diagnostics["w_tail"] if use_singular_split else c_coherent * diagnostics["w_tail"]
    i_q = diagnostics["I_total"]

    if use_singular_split:
        mode = "singular split: I_sing + H[(C_L-rho0^2-C_sing)*w_tail]"
    else:
        mode = "H[(C_L-rho0^2)*w_tail]"
    print(f"transform mode = {mode}")
    print(f"tail window starts at r = {diagnostics['r_taper_start']:.12g}")
    if use_singular_split:
        print(f"singular cutoff rc = {diagnostics['rc']:.12g}")

    high_q = (q_grid / k0 >= 5.0) & (q_grid / k0 <= 50.0)
    plateau_median = float(np.median(q_grid[high_q] * i_q[high_q])) if np.any(high_q) else float("nan")

    print(f"empirical M_J(r_min={r_grid[0]:.6g}) = {m_j[0]:.12g}")
    print(f"empirical M_J(r_max={r_grid[-1]:.6g}) = {m_j[-1]:.12g}")
    print(f"CL[-1] = {c_l[-1]:.12g}")
    print(f"CL[-1] - rho0^2 = {c_l[-1] - rho0**2:.12g}")
    print(f"median Q*I(Q), Q/k0 in [5, 50] = {plateau_median:.12g}")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "rw_line_scattering_data.npz",
        r_grid=r_grid,
        Q_grid=q_grid,
        M_J=m_j,
        C_L=c_l,
        C_L_coherent=c_coherent,
        C_L_transform=c_transform,
        CL_coherent=diagnostics["CL_coherent"],
        CL_sing=diagnostics["CL_sing"],
        CL_reg=diagnostics["CL_reg"],
        w_tail=diagnostics["w_tail"],
        I_Q=i_q,
        I_total=diagnostics["I_total"],
        I_coherent_windowed=diagnostics["I_coherent_windowed"],
        I_plateau_windowed=diagnostics["I_plateau_windowed"],
        I_full_windowed=diagnostics["I_full_windowed"],
        I_sing=diagnostics["I_sing"],
        I_reg_windowed=diagnostics["I_reg_windowed"],
        rho0=rho0,
        k0=k0,
        k_distribution=args.k_distribution,
        num_modes=int(args.num_modes),
        r_sigma_k=float(args.r_sigma_k),
        r_k_min=float(args.r_k_min),
        r_k_max=float(args.r_k_max),
        shared_k_vectors=bool(args.shared_k_vectors),
        r_min=r_min,
        r_max=r_max,
        Nr=int(args.Nr),
        r_grid_mode=args.r_grid,
        r_split=float(r_grid[np.searchsorted(r_grid, 1.0, side="left")]) if args.r_split is None else float(args.r_split),
        Nr_small=-1 if args.Nr_small is None else int(args.Nr_small),
        Q_min=q_min,
        Q_max=q_max,
        NQ=int(args.NQ),
        N_samp=int(args.N_samp),
        jacobian_method=args.jacobian_method,
        N_samp_U=-1 if args.N_samp_U is None else int(args.N_samp_U),
        N_samp_st=int(args.N_samp_st),
        U_sampling="" if args.U_sampling is None else args.U_sampling,
        st_sampling=args.st_sampling,
        st_transform=args.st_transform,
        st_tau=np.nan if args.st_tau is None else float(args.st_tau),
        n_jobs=int(args.n_jobs),
        use_qmc=not args.no_qmc,
        random_seed=int(args.seed),
        subtract_background=True,
        tail_window=True,
        tail_start=float(diagnostics["r_taper_start"]),
        r_taper_start=float(diagnostics["r_taper_start"]),
        rc=float(diagnostics["rc"]),
        use_singular_split=use_singular_split,
        quadrature_method="diagnostic_windowed_hankel",
    )
    save_plots(output_dir, r_grid, q_grid, k0, m_j, c_l, i_q)
    print(f"Saved data and figures under: {output_dir}")


if __name__ == "__main__":
    main()
