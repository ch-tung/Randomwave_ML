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
SMALL_X = 1.0e-4
DEFAULT_RADIAL_CHUNK_SIZE = 256
KDistribution = Literal["single_shell", "gaussian_radial", "uniform_band"]


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
        m = int(np.log2(n_samp))
        if 2**m != n_samp:
            raise ValueError("Sobol QMC requires n_samp to be a power of two.")
        sampler = qmc.Sobol(d=12, scramble=True, seed=random_seed)
        uniform = sampler.random_base2(m=m)
        uniform = np.clip(uniform, 1.0e-12, 1.0 - 1.0e-12)
        z = norm.ppf(uniform)
    else:
        rng = np.random.default_rng(random_seed)
        z = rng.standard_normal((n_samp, 12))
    return z[:, :6], z[:, 6:]


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


def compute_CL(
    r_grid: np.ndarray,
    k0: float,
    n_samp: int,
    *,
    use_qmc: bool = True,
    random_seed: int = DEFAULT_RANDOM_SEED,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute M_J(r) and C_L(r) over the supplied positive r grid."""

    z_u, z_v = standard_normal_samples(n_samp, use_qmc=use_qmc, random_seed=random_seed)
    m_j = np.empty_like(r_grid, dtype=float)
    t0 = time.perf_counter()
    report_every = max(1, len(r_grid) // 20)
    for idx, r_value in enumerate(r_grid):
        m_j[idx] = sample_MJ_for_r(float(r_value), k0, z_u, z_v)
        if progress and ((idx + 1) % report_every == 0 or idx + 1 == len(r_grid)):
            elapsed = time.perf_counter() - t0
            print(f"M_J samples: {idx + 1}/{len(r_grid)} r values ({elapsed:.1f}s)")

    g = g_mono(r_grid, k0)
    denom = 4.0 * np.pi**2 * np.maximum(1.0 - g * g, np.finfo(float).tiny)
    c_l = m_j / denom
    return m_j, c_l


def compute_CL_general(
    r_grid: np.ndarray,
    k_radii: np.ndarray,
    n_samp: int,
    *,
    k_weights: np.ndarray | None = None,
    use_qmc: bool = True,
    random_seed: int = DEFAULT_RANDOM_SEED,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute M_J(r) and C_L(r) using a numerical radial k-spectrum covariance."""

    z_u, z_v = standard_normal_samples(n_samp, use_qmc=use_qmc, random_seed=random_seed)
    m_j = np.empty_like(r_grid, dtype=float)
    t0 = time.perf_counter()
    report_every = max(1, len(r_grid) // 20)
    for idx, r_value in enumerate(r_grid):
        m_j[idx] = sample_MJ_for_r_general(float(r_value), k_radii, z_u, z_v, k_weights=k_weights)
        if progress and ((idx + 1) % report_every == 0 or idx + 1 == len(r_grid)):
            elapsed = time.perf_counter() - t0
            print(f"M_J samples: {idx + 1}/{len(r_grid)} r values ({elapsed:.1f}s)")

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

    return {
        "CL": c_l,
        "CL_coherent": cl_coherent,
        "CL_transform_raw": cl_transform_raw,
        "CL_transform": cl_transform,
        "w_tail": w_tail,
        "I_coherent_raw": hankel_transform(r_grid, cl_transform_raw, q_grid),
        "I_coherent_windowed": hankel_transform(r_grid, cl_transform, q_grid),
        "I_full_windowed": hankel_transform(r_grid, c_l * w_tail, q_grid),
        "I_plateau_windowed": hankel_transform(r_grid, plateau * w_tail, q_grid),
        "rho0": rho0,
        "rho0_squared": float(rho0**2),
        "r_max": r_max,
        "r_taper_start": float(r_taper_start),
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


def make_qmc_normals(dim: int, qmc_power: int, seed: int = DEFAULT_RANDOM_SEED) -> np.ndarray:
    """Sobol normal samples with shape (2**qmc_power, dim)."""

    sampler = qmc.Sobol(d=int(dim), scramble=True, seed=seed)
    u = sampler.random_base2(m=int(qmc_power))
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
    qmc_power: int = 14,
    seed: int = DEFAULT_RANDOM_SEED,
    rho_pairs: Sequence[tuple[float, float]] | None = None,
    r_taper_start: float | None = None,
    progress: bool = True,
) -> FourFieldScanResult:
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
    xi12 = make_qmc_normals(12, qmc_power, seed)
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
