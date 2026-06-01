"""
Scattering from retained random-wave line networks.

This module reuses ``rw_line_network.py`` to generate the retained
line families Gamma_12 and Gamma_13, samples those curves as point scatterers,
and evaluates

    A(q) = sum_j w_j exp(i q dot r_j)
    I(q) = |A(q)|^2

The reported 1D ``I(Q)`` is an orientational average over random q-directions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pyvista as pv

import rw_line_network as r


# =============================================================================
# Editable parameters
# =============================================================================

Q_MIN = 0.05
Q_MAX = 2.5
NUM_Q = 80
Q_SPACING = "linear"  # "linear" or "log"
Q_AXIS_SCALE = "mean_k"  # "mean_k" or "raw"; total line length is an intensity scale.
NUM_Q_DIRECTIONS = 96
SCATTERING_SEED = 20260529
# Number of q-vectors handled per matrix multiply. The full set has
# NUM_Q * NUM_Q_DIRECTIONS vectors; batching keeps memory bounded.
SCATTERING_BATCH_SIZE = 32768
SCATTERING_FLATTEN_Q_DIRECTIONS = False
SCATTERING_AMPLITUDE_METHOD = "points"  # "points" or "line_segments"
SCATTERING_WINDOW = "none"  # "none", "tukey_box", "hann_box", or "gaussian"
SCATTERING_WINDOW_TAPER = 0.15
SUBTRACT_WINDOWED_MEAN = True
WINDOW_MEAN_METHOD = "numeric_1d"  # "numeric_1d" or "grid_fft" (reserved)
WINDOW_NORMALIZATION = "windowed_measure"
SUBTRACT_EXPLICIT_BOX_MEAN = False
ANALYTIC_MEAN_BUFFER_BLOCKS = 0
ANALYTIC_MEAN_BUFFER_MODE = "none"  # "none", "incoherent", or "coherent"
ANALYTIC_MEAN_BUFFER_NORMALIZE_TOTAL = True
INTENSITY_NORMALIZATION = "i0"  # "i0", "length_density", or "none"
NORMALIZE_I0 = True  # Backward-compatible shorthand for INTENSITY_NORMALIZATION.
NUM_SEED_AVERAGE = 1
STRUCTURE_SEED_START = 894894
STRUCTURE_SEED_STRIDE = 1009
PRINT_TIMING = True

# Point sampling along vortex lines, in the same grid-coordinate units as the
# PyVista geometry. A smaller spacing gives a better curve integral but costs
# more in the q-sum.
LINE_SAMPLE_SPACING = 0.75
DYNAMIC_LINE_SAMPLE_SPACING = False
DYNAMIC_LINE_SAMPLE_EXPONENT = 1.0
DYNAMIC_LINE_SAMPLE_POWER2_SUBSETS = True
S12_SCATTERING_WEIGHT = 1.0
S13_SCATTERING_WEIGHT = 1.0

INCLUDE_CROSSLINK_POINTS = False
CROSSLINK_SCATTERING_WEIGHT = 1.0

SAVE_SCATTERING_DATA = True
SCATTERING_OUTPUT_PATH = "rw_line_scattering_iq.csv"

SAVE_REAL_SPACE_SCREENSHOT = True
REAL_SPACE_SCREENSHOT_PATH = "rw_line_scattering_structure.png"

SHOW_STRUCTURE_LINES = True
SHOW_SCATTERING_SAMPLE_POINTS = False
SCATTERING_SAMPLE_POINT_COLOR = "black"
SCATTERING_SAMPLE_POINT_SIZE = 5.0
SCATTERING_SAMPLE_POINT_OPACITY = 0.85
SCATTERING_SAMPLE_POINTS_AS_BALLS = True
SCATTERING_SAMPLE_BALL_RADIUS = 0.35


@dataclass(frozen=True)
class LineStructure:
    """Line structure used for scattering."""

    points: np.ndarray
    weights: np.ndarray
    segment_starts: np.ndarray
    segment_ends: np.ndarray
    segment_weights: np.ndarray
    s12_lines: pv.PolyData
    s13_lines: pv.PolyData
    crosslink_points: np.ndarray


def q_values(
    q_min: float | None = None,
    q_max: float | None = None,
    num_q: int | None = None,
    spacing: str | None = None,
) -> np.ndarray:
    """Return Q values for the 1D scattering curve."""

    if q_min is None:
        q_min = Q_MIN
    if q_max is None:
        q_max = Q_MAX
    if num_q is None:
        num_q = NUM_Q
    if spacing is None:
        spacing = Q_SPACING
    if num_q <= 0:
        raise ValueError("NUM_Q must be positive.")
    if q_min <= 0.0 and spacing == "log":
        raise ValueError("Q_MIN must be positive for log spacing.")
    if spacing == "linear":
        return np.linspace(float(q_min), float(q_max), int(num_q))
    if spacing == "log":
        return np.geomspace(float(q_min), float(q_max), int(num_q))
    raise ValueError("Q_SPACING must be 'linear' or 'log'.")


def isotropic_directions(count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample isotropic unit vectors."""

    if count <= 0:
        raise ValueError("direction count must be positive.")
    directions = rng.normal(size=(int(count), 3))
    norms = np.linalg.norm(directions, axis=1)
    while np.any(norms == 0.0):
        bad = norms == 0.0
        directions[bad] = rng.normal(size=(np.count_nonzero(bad), 3))
        norms = np.linalg.norm(directions, axis=1)
    return directions / norms[:, None]


def structure_seed_values(
    count: int | None = None,
    *,
    start: int | None = None,
    stride: int | None = None,
) -> list[int]:
    """Return deterministic random-wave seeds for ensemble averaging."""

    if count is None:
        count = NUM_SEED_AVERAGE
    if start is None:
        start = STRUCTURE_SEED_START
    if stride is None:
        stride = STRUCTURE_SEED_STRIDE
    if int(count) <= 0:
        raise ValueError("NUM_SEED_AVERAGE must be positive.")
    return [int(start) + int(stride) * i for i in range(int(count))]


def active_box_size_l() -> float:
    """Return the displayed cubic box side in grid-coordinate units."""

    if r.BOX_SIZE_L is not None:
        return float(r.BOX_SIZE_L)
    return float(r.expanded_grid_size(r.GRID_SIZE, r.NUM_BLOCK) - 1)


def active_box_center() -> np.ndarray:
    """Return the center of the active cubic observation box."""

    half_l = 0.5 * active_box_size_l()
    return np.array([half_l, half_l, half_l], dtype=float)


def scattering_window_type(window_type: str | None = None) -> str:
    """Resolve and validate the scattering window type."""

    if window_type is None:
        window_type = SCATTERING_WINDOW
    window_type = str(window_type).lower()
    if window_type not in {"none", "tukey_box", "hann_box", "gaussian"}:
        raise ValueError("SCATTERING_WINDOW must be 'none', 'tukey_box', 'hann_box', or 'gaussian'.")
    return window_type


def _window_1d_from_u(u: np.ndarray, window_type: str, taper_fraction: float) -> np.ndarray:
    """Evaluate a symmetric 1D window as a function of u=abs(x-center)/(L/2)."""

    u = np.asarray(u, dtype=float)
    window_type = scattering_window_type(window_type)
    if window_type == "none":
        return np.ones_like(u)

    inside = u < 1.0
    values = np.zeros_like(u)
    if window_type == "tukey_box":
        taper = float(np.clip(taper_fraction, 1.0e-6, 1.0))
        plateau_edge = max(0.0, 1.0 - taper)
        values[(u <= plateau_edge) & inside] = 1.0
        taper_region = (u > plateau_edge) & inside
        if np.any(taper_region):
            t = (u[taper_region] - plateau_edge) / max(1.0 - plateau_edge, 1.0e-12)
            values[taper_region] = 0.5 * (1.0 + np.cos(np.pi * t))
        return values

    if window_type == "hann_box":
        values[inside] = 0.5 * (1.0 + np.cos(np.pi * u[inside]))
        return values

    edge_value = float(np.clip(taper_fraction, 1.0e-4, 0.95))
    sigma_u = 1.0 / np.sqrt(2.0 * np.log(1.0 / edge_value))
    values[inside] = np.exp(-0.5 * (u[inside] / sigma_u) ** 2)
    return values


def window_function(
    points: np.ndarray,
    box_center: Sequence[float] | None = None,
    box_size: float | Sequence[float] | None = None,
    window_type: str | None = None,
    taper_fraction: float | None = None,
) -> np.ndarray:
    """Evaluate the smooth observation window at point coordinates."""

    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    window_type = scattering_window_type(window_type)
    if taper_fraction is None:
        taper_fraction = SCATTERING_WINDOW_TAPER
    if box_size is None:
        box_size = active_box_size_l()
    box_size = np.asarray(box_size, dtype=float)
    if box_size.ndim == 0:
        box_size = np.full(3, float(box_size))
    if box_center is None:
        box_center = 0.5 * box_size
    box_center = np.asarray(box_center, dtype=float)

    if window_type == "none":
        return np.ones(len(points), dtype=float)

    half_size = 0.5 * box_size
    u = np.abs(points - box_center[None, :]) / half_size[None, :]
    one_d = _window_1d_from_u(u, window_type, float(taper_fraction))
    return np.prod(one_d, axis=1)


@lru_cache(maxsize=64)
def window_volume_1d(
    box_size: float,
    window_type: str,
    taper_fraction: float,
    samples: int = 4096,
) -> float:
    """Numerically integrate the 1D window over one box side."""

    window_type = scattering_window_type(window_type)
    if window_type == "none":
        return float(box_size)
    x = np.linspace(0.0, float(box_size), int(samples))
    center = 0.5 * float(box_size)
    u = np.abs(x - center) / center
    w = _window_1d_from_u(u, window_type, float(taper_fraction))
    return float(np.trapz(w, x))


def window_volume(
    box_size: float | None = None,
    window_type: str | None = None,
    taper_fraction: float | None = None,
) -> float:
    """Return integral W(r) dr over the active observation box."""

    if box_size is None:
        box_size = active_box_size_l()
    if taper_fraction is None:
        taper_fraction = SCATTERING_WINDOW_TAPER
    window_type = scattering_window_type(window_type)
    v1 = window_volume_1d(float(box_size), window_type, float(taper_fraction))
    return v1**3


@lru_cache(maxsize=200000)
def window_hat_1d_numeric(
    q_value_rounded: float,
    box_size: float,
    window_type: str,
    taper_fraction: float,
    samples: int = 4096,
) -> complex:
    """Numerically compute int w(x) exp(i q x) dx for one coordinate."""

    q_value = float(q_value_rounded)
    window_type = scattering_window_type(window_type)
    if window_type == "none":
        center = 0.5 * float(box_size)
        return complex(float(box_size) * np.exp(1j * q_value * center) * np.sinc(q_value * float(box_size) / (2.0 * np.pi)))
    x = np.linspace(0.0, float(box_size), int(samples))
    center = 0.5 * float(box_size)
    u = np.abs(x - center) / center
    w = _window_1d_from_u(u, window_type, float(taper_fraction))
    return complex(np.trapz(w * np.exp(1j * q_value * x), x))


def window_hat(
    q_vectors: np.ndarray,
    *,
    box_size: float | None = None,
    window_type: str | None = None,
    taper_fraction: float | None = None,
    method: str | None = None,
) -> np.ndarray:
    """Return W_hat(q)=int W(r) exp(i q.r) dr using separable 1D quadrature."""

    if method is None:
        method = WINDOW_MEAN_METHOD
    method = str(method).lower()
    if method != "numeric_1d":
        raise NotImplementedError("Only WINDOW_MEAN_METHOD='numeric_1d' is currently implemented.")
    if box_size is None:
        box_size = active_box_size_l()
    if taper_fraction is None:
        taper_fraction = SCATTERING_WINDOW_TAPER
    window_type = scattering_window_type(window_type)
    q_vectors = np.asarray(q_vectors, dtype=float)
    if q_vectors.ndim != 2 or q_vectors.shape[1] != 3:
        raise ValueError("q_vectors must have shape (N, 3).")

    result = np.ones(len(q_vectors), dtype=complex)
    for axis in range(3):
        components = q_vectors[:, axis]
        values = np.empty(len(components), dtype=complex)
        for idx, q_component in enumerate(components):
            values[idx] = window_hat_1d_numeric(
                round(float(q_component), 12),
                float(box_size),
                window_type,
                float(taper_fraction),
            )
        result *= values
    return result


def normalization_mode(normalize_i0: bool | None = None, normalization: str | None = None) -> str:
    """Resolve the requested intensity normalization mode."""

    if normalization is None:
        normalization = INTENSITY_NORMALIZATION
    if normalize_i0 is not None:
        normalization = "i0" if normalize_i0 else "none"
    normalization = str(normalization).lower()
    aliases = {
        "i_0": "i0",
        "i(q=0)": "i0",
        "true": "i0",
        "false": "none",
        "off": "none",
        "length*scatter density": "length_density",
        "length_density": "length_density",
        "measure_density": "length_density",
    }
    normalization = aliases.get(normalization, normalization)
    if normalization not in {"i0", "length_density", "none"}:
        raise ValueError("INTENSITY_NORMALIZATION must be 'i0', 'length_density', or 'none'.")
    return normalization


def intensity_normalization_factor(
    total_measure: float,
    mode: str,
    *,
    effective_volume: float | None = None,
) -> float:
    """
    Return the divisor for I(Q).

    ``total_measure`` is weighted point count for the point method, or weighted
    line length plus optional point weights for the line-segment method.
    ``length_density`` divides by total_measure * density, where density is
    total_measure / box volume.
    """

    total_measure = float(total_measure)
    if mode == "none" or total_measure == 0.0:
        return 1.0
    if mode == "i0":
        return total_measure * total_measure
    box_volume = active_box_size_l() ** 3 if effective_volume is None else float(effective_volume)
    if box_volume <= 0.0:
        return total_measure * total_measure
    return total_measure * (total_measure / box_volume)


def rectangular_box_amplitude(q_vectors: np.ndarray, box_size_l: float | None = None) -> np.ndarray:
    """
    Analytic Fourier amplitude of a uniform box from 0..L in x,y,z.

    This is equivalent to assembling the box volume from tetrahedra, but the
    product form is more stable around repeated ``q dot r`` values:

        A(q) = prod_i int_0^L exp(i q_i x_i) dx_i
             = L^3 exp(i q dot c) prod_i sinc(q_i L/2)
    """

    q_vectors = np.asarray(q_vectors, dtype=float)
    if q_vectors.ndim != 2 or q_vectors.shape[1] != 3:
        raise ValueError("q_vectors must have shape (N, 3).")
    if box_size_l is None:
        box_size_l = active_box_size_l()
    l_value = float(box_size_l)
    half_phase = 0.5 * l_value * q_vectors
    phase_center = np.sum(half_phase, axis=1)
    sinc_product = np.prod(np.sinc(half_phase / np.pi), axis=1)
    return (l_value**3) * np.exp(1j * phase_center) * sinc_product


def centered_box_amplitude(
    q_vectors: np.ndarray,
    box_size_l: float,
    center: Sequence[float],
) -> np.ndarray:
    """Analytic Fourier amplitude of a uniform cube with arbitrary center."""

    q_vectors = np.asarray(q_vectors, dtype=float)
    if q_vectors.ndim != 2 or q_vectors.shape[1] != 3:
        raise ValueError("q_vectors must have shape (N, 3).")
    l_value = float(box_size_l)
    center = np.asarray(center, dtype=float)
    half_phase = 0.5 * l_value * q_vectors
    phase_center = q_vectors @ center
    sinc_product = np.prod(np.sinc(half_phase / np.pi), axis=1)
    return (l_value**3) * np.exp(1j * phase_center) * sinc_product


def analytic_mean_buffer_amplitude(
    q_vectors: np.ndarray,
    explicit_measure: float,
    *,
    buffer_blocks: int | None = None,
    box_size_l: float | None = None,
) -> np.ndarray:
    """
    Analytic amplitude of a mean-SLD buffer around the explicit box.

    ``buffer_blocks=B`` means the explicit L^3 box is surrounded by a larger
    cube of side (2B+1)L. The returned amplitude is rho_mean times
    A_big_box - A_inner_box, with both boxes sharing the same center.
    """

    if buffer_blocks is None:
        buffer_blocks = ANALYTIC_MEAN_BUFFER_BLOCKS
    buffer_blocks = int(buffer_blocks)
    if buffer_blocks <= 0 or explicit_measure == 0.0:
        return np.zeros(len(q_vectors), dtype=complex)
    if box_size_l is None:
        box_size_l = active_box_size_l()
    inner_l = float(box_size_l)
    outer_l = (2 * buffer_blocks + 1) * inner_l
    center = np.array([0.5 * inner_l, 0.5 * inner_l, 0.5 * inner_l], dtype=float)
    rho_mean = float(explicit_measure) / (inner_l**3)
    return rho_mean * (
        centered_box_amplitude(q_vectors, outer_l, center)
        - centered_box_amplitude(q_vectors, inner_l, center)
    )


def explicit_box_mean_amplitude(
    q_vectors: np.ndarray,
    explicit_measure: float,
    *,
    box_size_l: float | None = None,
) -> np.ndarray:
    """Analytic amplitude of the central box filled with the mean SLD."""

    if explicit_measure == 0.0:
        return np.zeros(len(q_vectors), dtype=complex)
    if box_size_l is None:
        box_size_l = active_box_size_l()
    inner_l = float(box_size_l)
    center = np.array([0.5 * inner_l, 0.5 * inner_l, 0.5 * inner_l], dtype=float)
    rho_mean = float(explicit_measure) / (inner_l**3)
    return rho_mean * centered_box_amplitude(q_vectors, inner_l, center)


def analytic_mean_background_amplitude(
    q_vectors: np.ndarray,
    explicit_measure: float,
    *,
    buffer_blocks: int | None = None,
) -> np.ndarray:
    """
    Return the analytic mean-background amplitude to combine with the network.

    If ``SUBTRACT_EXPLICIT_BOX_MEAN`` is true, the central mean box is
    subtracted. If buffer blocks are requested, the surrounding mean shell is
    added. Thus with B=0 and mean subtraction enabled, this returns only
    ``-rho_mean * A_inner_box``.
    """

    amplitude = np.zeros(len(q_vectors), dtype=complex)
    if SUBTRACT_EXPLICIT_BOX_MEAN:
        amplitude -= explicit_box_mean_amplitude(q_vectors, explicit_measure)
    if analytic_mean_buffer_mode() != "none":
        amplitude += analytic_mean_buffer_amplitude(
            q_vectors,
            explicit_measure,
            buffer_blocks=buffer_blocks,
        )
    return amplitude


def analytic_buffer_total_measure(explicit_measure: float, buffer_blocks: int | None = None) -> float:
    """Return explicit plus mean-buffer measure for normalization."""

    if buffer_blocks is None:
        buffer_blocks = ANALYTIC_MEAN_BUFFER_BLOCKS
    buffer_blocks = int(buffer_blocks)
    if buffer_blocks <= 0:
        return float(explicit_measure)
    scale = float((2 * buffer_blocks + 1) ** 3)
    return float(explicit_measure) * scale


def analytic_mean_buffer_mode(mode: str | None = None) -> str:
    """Resolve analytic mean-buffer mode."""

    if mode is None:
        mode = ANALYTIC_MEAN_BUFFER_MODE
    mode = str(mode).lower()
    if mode not in {"none", "incoherent", "coherent"}:
        raise ValueError("ANALYTIC_MEAN_BUFFER_MODE must be 'none', 'incoherent', or 'coherent'.")
    if int(ANALYTIC_MEAN_BUFFER_BLOCKS) <= 0:
        return "none"
    return mode


def analytic_mean_background_mode() -> str:
    """Return how analytic mean-background amplitude should be combined."""

    buffer_mode = analytic_mean_buffer_mode()
    if buffer_mode != "none":
        return buffer_mode
    if SUBTRACT_EXPLICIT_BOX_MEAN:
        return "coherent"
    return "none"


def use_scattering_window() -> bool:
    """Return whether a nontrivial smooth scattering window is active."""

    return scattering_window_type() != "none"


def windowed_mean_enabled() -> bool:
    """Return whether the windowed mean amplitude should be subtracted."""

    return bool(SUBTRACT_WINDOWED_MEAN)


def scattering_effective_volume() -> float:
    """Return the volume used for density normalization."""

    if use_scattering_window() or WINDOW_NORMALIZATION == "windowed_measure":
        return window_volume()
    return active_box_size_l() ** 3


def normalization_volume(buffer_mode: str = "none") -> float:
    """Return the volume paired with the current normalization measure."""

    volume = scattering_effective_volume()
    if buffer_mode != "none" and ANALYTIC_MEAN_BUFFER_NORMALIZE_TOTAL:
        volume *= float((2 * int(ANALYTIC_MEAN_BUFFER_BLOCKS) + 1) ** 3)
    return volume


def maybe_print_window_diagnostics(total_measure: float, *, prefix: str = "scattering") -> None:
    """Print compact diagnostics for the active scattering window."""

    v_window = scattering_effective_volume()
    rho = float(total_measure) / v_window if v_window > 0.0 else 0.0
    print(
        f"{prefix}: scattering_window={SCATTERING_WINDOW}; "
        f"scattering_window_taper={SCATTERING_WINDOW_TAPER}; "
        f"M_windowed={float(total_measure):.6g}; "
        f"V_windowed={v_window:.6g}; "
        f"rho_windowed={rho:.6g}; "
        f"subtract_windowed_mean={bool(SUBTRACT_WINDOWED_MEAN)}"
    )


def current_window_diagnostics(structure: LineStructure | None = None) -> dict[str, float | str | bool]:
    """Return current window diagnostics, optionally for a built structure."""

    m_windowed = scattering_measure_for_structure(structure) if structure is not None else float("nan")
    v_windowed = scattering_effective_volume()
    rho_windowed = m_windowed / v_windowed if v_windowed > 0.0 and np.isfinite(m_windowed) else float("nan")
    return {
        "scattering_window": SCATTERING_WINDOW,
        "scattering_window_taper": float(SCATTERING_WINDOW_TAPER),
        "M_windowed": float(m_windowed),
        "V_windowed": float(v_windowed),
        "rho_windowed": float(rho_windowed),
        "subtract_windowed_mean": bool(SUBTRACT_WINDOWED_MEAN),
        "window_mean_method": WINDOW_MEAN_METHOD,
        "window_normalization": WINDOW_NORMALIZATION,
    }


def box_scattering_intensity_1d(
    q: Sequence[float],
    *,
    box_size_l: float | None = None,
    num_directions: int | None = None,
    seed: int | None = None,
    normalize_i0: bool | None = None,
    normalization: str | None = None,
) -> np.ndarray:
    """Orientationally averaged normalized scattering from the finite cubic box."""

    if num_directions is None:
        num_directions = NUM_Q_DIRECTIONS
    if seed is None:
        seed = SCATTERING_SEED
    if box_size_l is None:
        box_size_l = active_box_size_l()

    q = np.asarray(q, dtype=float)
    rng = np.random.default_rng(seed)
    directions = isotropic_directions(num_directions, rng)
    volume = float(box_size_l) ** 3
    mode = normalization_mode(normalize_i0, normalization)
    norm = intensity_normalization_factor(volume, mode)
    intensity = np.empty(len(q), dtype=float)
    for iq, q_value in enumerate(q):
        amplitudes = rectangular_box_amplitude(float(q_value) * directions, box_size_l)
        intensity[iq] = float(np.mean(np.abs(amplitudes) ** 2) / norm)
    return intensity


def effective_scatterer_count(
    *,
    weights: np.ndarray | None = None,
    point_counts: Sequence[int] | None = None,
) -> float:
    """
    Estimate the effective number of independent scatterers.

    For weighted point scatterers this uses ``(sum w)^2/sum(w^2)``. For an
    ensemble where only point counts are available, it uses the mean point
    count. This is a simple finite-density estimate for the incoherent floor.
    """

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        denom = float(np.sum(weights * weights))
        if denom <= 0.0:
            raise ValueError("weights must not all be zero.")
        return float(np.sum(weights) ** 2 / denom)
    if point_counts is not None:
        counts = np.asarray(point_counts, dtype=float)
        if len(counts) == 0 or np.any(counts <= 0.0):
            raise ValueError("point_counts must contain positive values.")
        return float(np.mean(counts))
    raise ValueError("supply either weights or point_counts.")


def density_floor_constant(
    *,
    weights: np.ndarray | None = None,
    point_counts: Sequence[int] | None = None,
) -> float:
    """Return c = 1/N_eff for the finite-density denominator floor."""

    return 1.0 / effective_scatterer_count(weights=weights, point_counts=point_counts)


def _line_cells(poly: pv.PolyData) -> list[np.ndarray]:
    """Return ordered point arrays for each line cell in a PolyData."""

    if poly.n_points == 0 or poly.n_lines == 0:
        return []

    points = np.asarray(poly.points, dtype=float)
    lines = np.asarray(poly.lines)
    cursor = 0
    cells: list[np.ndarray] = []
    while cursor < len(lines):
        n_in_cell = int(lines[cursor])
        ids = lines[cursor + 1 : cursor + 1 + n_in_cell].astype(int)
        cursor += n_in_cell + 1
        if n_in_cell >= 2:
            cells.append(points[ids])
    return cells


def _sample_line_cell_by_arclength(cell_points: np.ndarray, spacing: float) -> np.ndarray:
    """Sample one ordered traced line cell by cumulative arclength."""

    spacing = float(spacing)
    if spacing <= 0.0:
        raise ValueError("LINE_SAMPLE_SPACING must be positive.")
    segments = np.diff(cell_points, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    valid = segment_lengths > 0.0
    if not np.any(valid):
        return np.empty((0, 3), dtype=float)

    clean_points = [cell_points[0]]
    for keep, point in zip(valid, cell_points[1:]):
        if keep:
            clean_points.append(point)
    clean_points = np.asarray(clean_points, dtype=float)
    segment_lengths = np.linalg.norm(np.diff(clean_points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length == 0.0:
        return np.empty((0, 3), dtype=float)

    distances = np.arange(0.0, total_length, spacing)
    if len(distances) == 0 or not np.isclose(distances[-1], total_length):
        distances = np.append(distances, total_length)

    segment_ids = np.searchsorted(cumulative, distances, side="right") - 1
    segment_ids = np.clip(segment_ids, 0, len(segment_lengths) - 1)
    local_lengths = distances - cumulative[segment_ids]
    t_values = local_lengths / segment_lengths[segment_ids]
    return clean_points[segment_ids] + t_values[:, None] * (
        clean_points[segment_ids + 1] - clean_points[segment_ids]
    )


def sample_polyline_points(poly: pv.PolyData, spacing: float) -> np.ndarray:
    """Sample approximately uniform arclength points along ordered line cells.

    The line cells are produced by the vertex/vortex tracing step. PyVista is
    only used as a convenient container for those ordered polylines and for
    rendering. Sampling is therefore done by walking along each traced line's
    cumulative arclength, not by forcing one sample on every rendered segment.
    """

    sampled: list[np.ndarray] = []
    for cell_points in _line_cells(poly):
        points = _sample_line_cell_by_arclength(cell_points, spacing)
        sampled.extend(points)

    if not sampled:
        return np.empty((0, 3), dtype=float)
    return np.asarray(sampled, dtype=float)


def sample_polyline_power2_subset(poly: pv.PolyData, spacing: float, stride: int) -> np.ndarray:
    """
    Sample the finest chain once per line cell and keep every ``stride`` bead.

    ``stride`` is normally a power of two. Endpoints are kept for each line cell
    so the coarse chain still covers the full traced curve.
    """

    stride = max(1, int(stride))
    sampled: list[np.ndarray] = []
    for cell_points in _line_cells(poly):
        points = _sample_line_cell_by_arclength(cell_points, spacing)
        if len(points) == 0:
            continue
        subset = points[::stride]
        if len(subset) == 0 or not np.allclose(subset[-1], points[-1]):
            subset = np.vstack((subset, points[-1]))
        sampled.extend(subset)
    if not sampled:
        return np.empty((0, 3), dtype=float)
    return np.asarray(sampled, dtype=float)


def polyline_segments(poly: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
    """Return start/end arrays for all nonzero segments in ordered line cells."""

    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for cell_points in _line_cells(poly):
        if len(cell_points) < 2:
            continue
        cell_starts = cell_points[:-1]
        cell_ends = cell_points[1:]
        lengths = np.linalg.norm(cell_ends - cell_starts, axis=1)
        keep = lengths > 0.0
        if np.any(keep):
            starts.append(cell_starts[keep])
            ends.append(cell_ends[keep])
    if not starts:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty
    return np.vstack(starts), np.vstack(ends)


def total_line_length(structure: LineStructure, *, weighted: bool = True) -> float:
    """Return the total traced line length used by the segment-integral method."""

    if len(structure.segment_starts) == 0:
        return 0.0
    lengths = np.linalg.norm(structure.segment_ends - structure.segment_starts, axis=1)
    if weighted:
        return float(np.sum(lengths * structure.segment_weights))
    return float(np.sum(lengths))


def scattering_measure_for_structure(
    structure: LineStructure,
    *,
    amplitude_method: str | None = None,
) -> float:
    """Return the current windowed scattering measure for diagnostics."""

    if amplitude_method is None:
        amplitude_method = SCATTERING_AMPLITUDE_METHOD
    if amplitude_method == "points":
        if len(structure.points) == 0:
            return 0.0
        return float(np.sum(structure.weights * window_function(structure.points)))
    if amplitude_method == "line_segments":
        if len(structure.segment_starts) == 0:
            return 0.0
        lengths = np.linalg.norm(structure.segment_ends - structure.segment_starts, axis=1)
        midpoints = 0.5 * (structure.segment_starts + structure.segment_ends)
        measure = float(np.sum(lengths * structure.segment_weights * window_function(midpoints)))
        if len(structure.crosslink_points):
            measure += float(
                np.sum(
                    np.full(len(structure.crosslink_points), float(CROSSLINK_SCATTERING_WEIGHT))
                    * window_function(structure.crosslink_points)
                )
            )
        return measure
    raise ValueError("SCATTERING_AMPLITUDE_METHOD must be 'points' or 'line_segments'.")


def _build_fields() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build phi1, phi2, phi3 using the current parameters in the rendering module."""

    rng = np.random.default_rng(r.RANDOM_SEED)
    k_sets = r.make_field_k_sets(
        r.NUM_MODES,
        r.K_DISTRIBUTION,
        rng,
        shared_k_vectors=r.SHARED_K_VECTORS,
    )
    phi1 = r.build_random_wave_field(
        r.GRID_SIZE,
        k_sets.phi1,
        rng,
        num_block=r.NUM_BLOCK,
        block_overlap=r.BLOCK_OVERLAP,
    )
    if r.COUPLE_PHI2_PHI3:
        phi2, phi3 = r.build_correlated_phi2_phi3_fields(
            r.GRID_SIZE,
            k_sets.phi2,
            k_sets.phi3,
            rng,
            coupling_c=r.PHI23_COUPLING_C,
        )
    else:
        phi2 = r.build_random_wave_field(
            r.GRID_SIZE,
            k_sets.phi2,
            rng,
            num_block=r.NUM_BLOCK,
            block_overlap=r.BLOCK_OVERLAP,
        )
        phi3 = r.build_random_wave_field(
            r.GRID_SIZE,
            k_sets.phi3,
            rng,
            num_block=r.NUM_BLOCK,
            block_overlap=r.BLOCK_OVERLAP,
        )
    return phi1, phi2, phi3


def build_line_structure(
    *,
    line_sample_spacing: float | None = None,
    include_crosslinks: bool | None = None,
    print_timing: bool | None = None,
    timing_label: str = "structure",
) -> LineStructure:
    """Generate the retained line network and sample it for scattering."""

    if line_sample_spacing is None:
        line_sample_spacing = LINE_SAMPLE_SPACING
    if include_crosslinks is None:
        include_crosslinks = INCLUDE_CROSSLINK_POINTS
    if print_timing is None:
        print_timing = False

    t0 = time.perf_counter()
    phi1, phi2, phi3 = _build_fields()
    t_fields = time.perf_counter()
    s12_raw = r.trace_vortex_segments(phi1, phi2)
    s13_raw = r.trace_vortex_segments(phi1, phi3)
    t_trace = time.perf_counter()
    s12_lines = r.smooth_vortex_polydata(s12_raw) if r.SMOOTH_VORTEX_LINES else s12_raw
    s13_lines = r.smooth_vortex_polydata(s13_raw) if r.SMOOTH_VORTEX_LINES else s13_raw
    t_smooth = time.perf_counter()

    s12_segment_starts, s12_segment_ends = polyline_segments(s12_lines)
    s13_segment_starts, s13_segment_ends = polyline_segments(s13_lines)
    segment_blocks: list[np.ndarray] = []
    segment_end_blocks: list[np.ndarray] = []
    segment_weight_blocks: list[np.ndarray] = []
    if len(s12_segment_starts):
        segment_blocks.append(s12_segment_starts)
        segment_end_blocks.append(s12_segment_ends)
        segment_weight_blocks.append(np.full(len(s12_segment_starts), float(S12_SCATTERING_WEIGHT)))
    if len(s13_segment_starts):
        segment_blocks.append(s13_segment_starts)
        segment_end_blocks.append(s13_segment_ends)
        segment_weight_blocks.append(np.full(len(s13_segment_starts), float(S13_SCATTERING_WEIGHT)))

    s12_points = sample_polyline_points(s12_lines, line_sample_spacing)
    s13_points = sample_polyline_points(s13_lines, line_sample_spacing)
    t_sample = time.perf_counter()
    point_blocks = []
    weight_blocks = []
    crosslink_points = np.empty((0, 3), dtype=float)

    if len(s12_points):
        point_blocks.append(s12_points)
        weight_blocks.append(np.full(len(s12_points), float(S12_SCATTERING_WEIGHT)))
    if len(s13_points):
        point_blocks.append(s13_points)
        weight_blocks.append(np.full(len(s13_points), float(S13_SCATTERING_WEIGHT)))

    if include_crosslinks:
        candidates = r.find_crosslink_candidate_midpoints(
            s12_raw,
            s13_raw,
            search_radius=r.CROSSLINK_SEARCH_RADIUS,
        )
        crosslink_points = (
            r._cluster_points(candidates, radius=float(r.CROSSLINK_MERGE_RADIUS))
            if len(candidates)
            else np.empty((0, 3), dtype=float)
        )
        if len(crosslink_points):
            point_blocks.append(crosslink_points)
            weight_blocks.append(np.full(len(crosslink_points), float(CROSSLINK_SCATTERING_WEIGHT)))
    t_crosslinks = time.perf_counter()

    if print_timing:
        print(
            f"{timing_label}: fields {t_fields - t0:.3f}s; "
            f"vortex trace {t_trace - t_fields:.3f}s; "
            f"smoothing {t_smooth - t_trace:.3f}s; "
            f"line point sampling {t_sample - t_smooth:.3f}s; "
            f"crosslinks {t_crosslinks - t_sample:.3f}s"
        )

    if not point_blocks:
        empty = np.empty((0, 3), dtype=float)
        return LineStructure(
            empty,
            np.empty(0, dtype=float),
            empty,
            empty,
            np.empty(0, dtype=float),
            s12_lines,
            s13_lines,
            crosslink_points,
        )

    points = np.vstack(point_blocks)
    weights = np.concatenate(weight_blocks)
    if segment_blocks:
        segment_starts = np.vstack(segment_blocks)
        segment_ends = np.vstack(segment_end_blocks)
        segment_weights = np.concatenate(segment_weight_blocks)
    else:
        segment_starts = np.empty((0, 3), dtype=float)
        segment_ends = np.empty((0, 3), dtype=float)
        segment_weights = np.empty(0, dtype=float)
    if print_timing:
        print(
            f"{timing_label}: sampled points {len(points)} "
            f"line segments {len(segment_starts)}; "
            f"(S12 {len(s12_points)}, S13 {len(s13_points)}, crosslinks {len(crosslink_points)}); "
            f"structure total {time.perf_counter() - t0:.3f}s"
        )
    return LineStructure(
        points,
        weights,
        segment_starts,
        segment_ends,
        segment_weights,
        s12_lines,
        s13_lines,
        crosslink_points,
    )


def scattering_intensity_1d(
    points: np.ndarray,
    q: Sequence[float],
    *,
    weights: np.ndarray | None = None,
    num_directions: int | None = None,
    seed: int | None = None,
    batch_size: int | None = None,
    flatten_q_directions: bool | None = None,
    normalize_i0: bool | None = None,
    normalization: str | None = None,
) -> np.ndarray:
    """
    Compute orientationally averaged I(Q) from point scatterers.

    The result is normalized by ``sum(weights)^2`` when ``normalize_i0`` is True,
    so I(Q=0) would be 1 for a single-component positive-density structure.
    """

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if len(points) == 0:
        raise ValueError("no scattering points were supplied.")

    if num_directions is None:
        num_directions = NUM_Q_DIRECTIONS
    if seed is None:
        seed = SCATTERING_SEED
    if batch_size is None:
        batch_size = SCATTERING_BATCH_SIZE
    if flatten_q_directions is None:
        flatten_q_directions = SCATTERING_FLATTEN_Q_DIRECTIONS
    q = np.asarray(q, dtype=float)
    if weights is None:
        weights = np.ones(len(points), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (len(points),):
            raise ValueError("weights must have shape (N_points,).")

    rng = np.random.default_rng(seed)
    directions = isotropic_directions(num_directions, rng)
    window_values = window_function(points)
    windowed_weights = weights * window_values
    weighted_norm = float(np.sum(windowed_weights))
    v_window = scattering_effective_volume()
    rho_windowed = weighted_norm / v_window if v_window > 0.0 else 0.0
    buffer_mode = analytic_mean_background_mode()
    norm_measure = (
        analytic_buffer_total_measure(weighted_norm)
        if buffer_mode != "none" and ANALYTIC_MEAN_BUFFER_NORMALIZE_TOTAL
        else weighted_norm
    )
    mode = normalization_mode(normalize_i0, normalization)
    norm = intensity_normalization_factor(
        norm_measure,
        mode,
        effective_volume=normalization_volume(buffer_mode),
    )
    intensities = np.empty(len(q), dtype=float)

    if flatten_q_directions:
        q_vectors = (q[:, None, None] * directions[None, :, :]).reshape(-1, 3)
        q_shells = np.repeat(np.arange(len(q)), len(directions))
        shell_sums = np.zeros(len(q), dtype=float)
        batch_size = max(1, int(batch_size))
        for start in range(0, len(q_vectors), batch_size):
            stop = min(start + batch_size, len(q_vectors))
            q_batch = q_vectors[start:stop]
            shell_batch = q_shells[start:stop]
            phase = points @ q_batch.T
            amplitudes = np.exp(1j * phase).T @ windowed_weights
            if windowed_mean_enabled():
                amplitudes = amplitudes - rho_windowed * window_hat(q_batch)
            if buffer_mode != "none":
                buffer_amplitudes = analytic_mean_background_amplitude(q_batch, weighted_norm)
                if buffer_mode == "coherent":
                    amplitudes = amplitudes + buffer_amplitudes
                    buffer_extra = 0.0
                else:
                    buffer_extra = np.abs(buffer_amplitudes) ** 2
            else:
                buffer_extra = 0.0
            shell_sums += np.bincount(
                shell_batch,
                weights=np.abs(amplitudes) ** 2 + buffer_extra,
                minlength=len(q),
            )
        return shell_sums / (len(directions) * norm)

    for iq, q_value in enumerate(q):
        q_vectors = float(q_value) * directions
        shell_intensity = 0.0
        for start in range(0, len(q_vectors), int(batch_size)):
            q_batch = q_vectors[start : start + int(batch_size)]
            phase = points @ q_batch.T
            amplitudes = np.exp(1j * phase).T @ windowed_weights
            if windowed_mean_enabled():
                amplitudes = amplitudes - rho_windowed * window_hat(q_batch)
            if buffer_mode != "none":
                buffer_amplitudes = analytic_mean_background_amplitude(q_batch, weighted_norm)
                if buffer_mode == "coherent":
                    amplitudes = amplitudes + buffer_amplitudes
                    shell_intensity += float(np.sum(np.abs(amplitudes) ** 2))
                else:
                    shell_intensity += float(
                        np.sum(np.abs(amplitudes) ** 2)
                        + np.sum(np.abs(buffer_amplitudes) ** 2)
                    )
            else:
                shell_intensity += float(np.sum(np.abs(amplitudes) ** 2))
        intensities[iq] = shell_intensity / (len(q_vectors) * norm)

    return intensities


def dynamic_sample_spacing_for_q(
    q: Sequence[float],
    *,
    base_spacing: float | None = None,
    exponent: float | None = None,
    reference_q: float | None = None,
) -> np.ndarray:
    """
    Return Q-dependent line-sampling spacings.

    ``base_spacing`` is interpreted as the spacing suitable for the largest Q.
    With exponent n, Q^n * spacing is held constant:

        spacing(Q) = base_spacing * (reference_q / Q)^n.
    """

    if base_spacing is None:
        base_spacing = LINE_SAMPLE_SPACING
    if exponent is None:
        exponent = DYNAMIC_LINE_SAMPLE_EXPONENT
    q = np.asarray(q, dtype=float)
    positive_q = q[q > 0.0]
    if len(positive_q) == 0:
        raise ValueError("dynamic line spacing requires at least one positive Q.")
    if reference_q is None:
        reference_q = float(np.max(positive_q))
    if reference_q <= 0.0:
        raise ValueError("reference_q must be positive.")
    safe_q = np.maximum(q, np.min(positive_q))
    return float(base_spacing) * (float(reference_q) / safe_q) ** float(exponent)


def dynamic_sample_strides_for_q(
    q: Sequence[float],
    *,
    exponent: float | None = None,
    reference_q: float | None = None,
) -> np.ndarray:
    """
    Return power-of-two subset strides for dynamic point sampling.

    The finest chain is used at the largest Q. For lower Q, if the requested
    spacing has grown past a 2^m level, every 2^m bead is kept.
    """

    if exponent is None:
        exponent = DYNAMIC_LINE_SAMPLE_EXPONENT
    q = np.asarray(q, dtype=float)
    positive_q = q[q > 0.0]
    if len(positive_q) == 0:
        raise ValueError("dynamic line spacing requires at least one positive Q.")
    if reference_q is None:
        reference_q = float(np.max(positive_q))
    safe_q = np.maximum(q, np.min(positive_q))
    ratio = (float(reference_q) / safe_q) ** float(exponent)
    levels = np.floor(np.log2(np.maximum(ratio, 1.0))).astype(int)
    return (2 ** levels).astype(int)


def _points_from_structure_lines(structure: LineStructure, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample the retained line families at a given spacing."""

    point_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    s12_points = sample_polyline_points(structure.s12_lines, spacing)
    s13_points = sample_polyline_points(structure.s13_lines, spacing)
    if len(s12_points):
        point_blocks.append(s12_points)
        weight_blocks.append(np.full(len(s12_points), float(S12_SCATTERING_WEIGHT)))
    if len(s13_points):
        point_blocks.append(s13_points)
        weight_blocks.append(np.full(len(s13_points), float(S13_SCATTERING_WEIGHT)))
    if len(structure.crosslink_points):
        point_blocks.append(structure.crosslink_points)
        weight_blocks.append(np.full(len(structure.crosslink_points), float(CROSSLINK_SCATTERING_WEIGHT)))
    if not point_blocks:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
    return np.vstack(point_blocks), np.concatenate(weight_blocks)


def _points_from_structure_lines_power2_subset(
    structure: LineStructure,
    base_spacing: float,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reuse the finest chain and keep every power-of-two subset per line."""

    point_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    s12_points = sample_polyline_power2_subset(structure.s12_lines, base_spacing, stride)
    s13_points = sample_polyline_power2_subset(structure.s13_lines, base_spacing, stride)
    if len(s12_points):
        point_blocks.append(s12_points)
        weight_blocks.append(np.full(len(s12_points), float(S12_SCATTERING_WEIGHT)))
    if len(s13_points):
        point_blocks.append(s13_points)
        weight_blocks.append(np.full(len(s13_points), float(S13_SCATTERING_WEIGHT)))
    if len(structure.crosslink_points):
        point_blocks.append(structure.crosslink_points)
        weight_blocks.append(np.full(len(structure.crosslink_points), float(CROSSLINK_SCATTERING_WEIGHT)))
    if not point_blocks:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
    return np.vstack(point_blocks), np.concatenate(weight_blocks)


def dynamic_point_scattering_intensity_1d(
    structure: LineStructure,
    q: Sequence[float],
    *,
    base_spacing: float | None = None,
    exponent: float | None = None,
    normalization: str | None = None,
) -> np.ndarray:
    """
    Compute point-scatterer I(Q) with Q-dependent bead spacing along lines.

    This keeps the user-provided ``LINE_SAMPLE_SPACING`` as the high-Q spacing.
    At lower Q the spacing grows so Q^n * spacing is constant.
    """

    q = np.asarray(q, dtype=float)
    if base_spacing is None:
        base_spacing = LINE_SAMPLE_SPACING
    intensities = np.empty(len(q), dtype=float)
    point_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if DYNAMIC_LINE_SAMPLE_POWER2_SUBSETS:
        strides = dynamic_sample_strides_for_q(q, exponent=exponent)
        for iq, (q_value, stride) in enumerate(zip(q, strides)):
            stride = int(stride)
            if stride not in point_cache:
                point_cache[stride] = _points_from_structure_lines_power2_subset(
                    structure,
                    float(base_spacing),
                    stride,
                )
            points, weights = point_cache[stride]
            intensities[iq] = scattering_intensity_1d(
                points,
                np.asarray([q_value], dtype=float),
                weights=weights,
                normalization=normalization,
            )[0]
        return intensities

    spacings = dynamic_sample_spacing_for_q(q, base_spacing=base_spacing, exponent=exponent)
    for iq, (q_value, spacing) in enumerate(zip(q, spacings)):
        points, weights = _points_from_structure_lines(structure, float(spacing))
        intensities[iq] = scattering_intensity_1d(
            points,
            np.asarray([q_value], dtype=float),
            weights=weights,
            normalization=normalization,
        )[0]
    return intensities


def segment_integral_scattering_intensity_1d(
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    q: Sequence[float],
    *,
    segment_weights: np.ndarray | None = None,
    point_positions: np.ndarray | None = None,
    point_weights: np.ndarray | None = None,
    num_directions: int | None = None,
    seed: int | None = None,
    batch_size: int | None = None,
    normalize_i0: bool | None = None,
    normalization: str | None = None,
) -> np.ndarray:
    """
    Compute I(Q) using analytic amplitudes of straight line segments.

    For each segment from r1 to r2 with length L, the contribution is

        integral exp(i q.r) ds = L * (exp(i qr2) - exp(i qr1)) / (i * (qr2 - qr1)).

    The small-denominator limit is handled as L * exp(i qr1). Optional point
    scatterers, such as crosslink nodes, can be added to the same amplitude.
    With a nontrivial scattering window, the window is evaluated at the segment
    midpoint. Very long segments crossing the taper may need subdivision in a
    later refinement.
    """

    starts = np.asarray(segment_starts, dtype=float)
    ends = np.asarray(segment_ends, dtype=float)
    if starts.ndim != 2 or starts.shape[1] != 3:
        raise ValueError("segment_starts must have shape (N_segments, 3).")
    if ends.shape != starts.shape:
        raise ValueError("segment_ends must have the same shape as segment_starts.")
    if len(starts) == 0:
        raise ValueError("no line segments were supplied.")

    if segment_weights is None:
        segment_weights = np.ones(len(starts), dtype=float)
    else:
        segment_weights = np.asarray(segment_weights, dtype=float)
        if segment_weights.shape != (len(starts),):
            raise ValueError("segment_weights must have shape (N_segments,).")

    if point_positions is None:
        point_positions = np.empty((0, 3), dtype=float)
    else:
        point_positions = np.asarray(point_positions, dtype=float)
        if point_positions.ndim != 2 or point_positions.shape[1] != 3:
            raise ValueError("point_positions must have shape (N_points, 3).")
    if point_weights is None:
        point_weights = np.ones(len(point_positions), dtype=float)
    else:
        point_weights = np.asarray(point_weights, dtype=float)
        if point_weights.shape != (len(point_positions),):
            raise ValueError("point_weights must have shape (N_points,).")

    if num_directions is None:
        num_directions = NUM_Q_DIRECTIONS
    if seed is None:
        seed = SCATTERING_SEED
    if batch_size is None:
        batch_size = SCATTERING_BATCH_SIZE
    q = np.asarray(q, dtype=float)
    rng = np.random.default_rng(seed)
    directions = isotropic_directions(num_directions, rng)
    lengths = np.linalg.norm(ends - starts, axis=1)
    midpoints = 0.5 * (starts + ends)
    window_mid = window_function(midpoints)
    weighted_lengths = lengths * segment_weights * window_mid
    point_windows = window_function(point_positions) if len(point_positions) else np.empty(0, dtype=float)
    windowed_point_weights = point_weights * point_windows
    weighted_norm = float(np.sum(weighted_lengths) + np.sum(windowed_point_weights))
    v_window = scattering_effective_volume()
    rho_windowed = weighted_norm / v_window if v_window > 0.0 else 0.0
    buffer_mode = analytic_mean_background_mode()
    norm_measure = (
        analytic_buffer_total_measure(weighted_norm)
        if buffer_mode != "none" and ANALYTIC_MEAN_BUFFER_NORMALIZE_TOTAL
        else weighted_norm
    )
    mode = normalization_mode(normalize_i0, normalization)
    norm = intensity_normalization_factor(
        norm_measure,
        mode,
        effective_volume=normalization_volume(buffer_mode),
    )
    intensities = np.empty(len(q), dtype=float)
    batch_size = max(1, int(batch_size))

    for iq, q_value in enumerate(q):
        q_vectors = float(q_value) * directions
        shell_intensity = 0.0
        for start in range(0, len(q_vectors), batch_size):
            q_batch = q_vectors[start : start + batch_size]
            qr1 = starts @ q_batch.T
            qr2 = ends @ q_batch.T
            delta = qr2 - qr1
            midpoint_phase = np.exp(0.5j * (qr1 + qr2))
            sinc_factor = np.sinc(delta / (2.0 * np.pi))
            segment_amplitudes = (weighted_lengths[:, None] * midpoint_phase * sinc_factor).sum(axis=0)
            if len(point_positions):
                point_phase = point_positions @ q_batch.T
                segment_amplitudes += np.exp(1j * point_phase).T @ windowed_point_weights
            if windowed_mean_enabled():
                segment_amplitudes = segment_amplitudes - rho_windowed * window_hat(q_batch)
            if buffer_mode != "none":
                buffer_amplitudes = analytic_mean_background_amplitude(q_batch, weighted_norm)
                if buffer_mode == "coherent":
                    segment_amplitudes = segment_amplitudes + buffer_amplitudes
                    shell_intensity += float(np.sum(np.abs(segment_amplitudes) ** 2))
                else:
                    shell_intensity += float(
                        np.sum(np.abs(segment_amplitudes) ** 2)
                        + np.sum(np.abs(buffer_amplitudes) ** 2)
                    )
            else:
                shell_intensity += float(np.sum(np.abs(segment_amplitudes) ** 2))
        intensities[iq] = shell_intensity / (len(q_vectors) * norm)

    return intensities


def structure_scattering_intensity_1d(
    structure: LineStructure,
    q: Sequence[float],
    *,
    amplitude_method: str | None = None,
    normalization: str | None = None,
) -> np.ndarray:
    """Compute I(Q) from a built structure using the selected amplitude method."""

    if amplitude_method is None:
        amplitude_method = SCATTERING_AMPLITUDE_METHOD
    if amplitude_method == "points":
        if DYNAMIC_LINE_SAMPLE_SPACING:
            return dynamic_point_scattering_intensity_1d(
                structure,
                q,
                exponent=DYNAMIC_LINE_SAMPLE_EXPONENT,
                normalization=normalization,
            )
        return scattering_intensity_1d(
            structure.points,
            q,
            weights=structure.weights,
            normalization=normalization,
        )
    if amplitude_method == "line_segments":
        crosslink_weights = (
            np.full(len(structure.crosslink_points), float(CROSSLINK_SCATTERING_WEIGHT))
            if len(structure.crosslink_points)
            else np.empty(0, dtype=float)
        )
        return segment_integral_scattering_intensity_1d(
            structure.segment_starts,
            structure.segment_ends,
            q,
            segment_weights=structure.segment_weights,
            point_positions=structure.crosslink_points,
            point_weights=crosslink_weights,
            normalization=normalization,
        )
    raise ValueError("SCATTERING_AMPLITUDE_METHOD must be 'points' or 'line_segments'.")


def compute_current_scattering() -> tuple[np.ndarray, np.ndarray, LineStructure]:
    """Build the current structure and compute its 1D scattering curve."""

    structure = build_line_structure()
    q = q_values()
    intensity = structure_scattering_intensity_1d(structure, q)
    return q, intensity, structure


def compute_seed_averaged_scattering(
    seeds: Sequence[int] | None = None,
    *,
    print_timing: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Average I(Q) over multiple random-wave seeds.

    Returns Q, mean intensity, sample standard deviation, finite-box intensity,
    and the point count for each generated structure.
    """

    if seeds is None:
        seeds = structure_seed_values()
    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("at least one seed is required.")
    if print_timing is None:
        print_timing = PRINT_TIMING

    original_seed = r.RANDOM_SEED
    t_all = time.perf_counter()
    q = q_values()
    t_q = time.perf_counter()
    curves: list[np.ndarray] = []
    point_counts: list[int] = []
    try:
        for seed_index, seed in enumerate(seeds, start=1):
            t_seed = time.perf_counter()
            r.RANDOM_SEED = seed
            structure = build_line_structure(
                print_timing=print_timing,
                timing_label=f"seed {seed_index}/{len(seeds)} ({seed})",
            )
            t_structure = time.perf_counter()
            point_counts.append(len(structure.points))
            if print_timing:
                maybe_print_window_diagnostics(
                    scattering_measure_for_structure(structure),
                    prefix=f"seed {seed_index}/{len(seeds)} ({seed})",
                )
            curves.append(structure_scattering_intensity_1d(structure, q))
            t_scatter = time.perf_counter()
            if print_timing:
                print(
                    f"seed {seed_index}/{len(seeds)} ({seed}): "
                    f"structure {t_structure - t_seed:.3f}s; "
                    f"scattering sum {t_scatter - t_structure:.3f}s; "
                    f"seed total {t_scatter - t_seed:.3f}s"
                )
    finally:
        r.RANDOM_SEED = original_seed

    t_reduce0 = time.perf_counter()
    curve_array = np.vstack(curves)
    mean_intensity = np.mean(curve_array, axis=0)
    std_intensity = (
        np.std(curve_array, axis=0, ddof=1)
        if len(curves) > 1
        else np.zeros_like(mean_intensity)
    )
    t_reduce = time.perf_counter()
    box_intensity = box_scattering_intensity_1d(q)
    t_box = time.perf_counter()
    if print_timing:
        print(
            f"scattering average: q setup {t_q - t_all:.3f}s; "
            f"curve reduction {t_reduce - t_reduce0:.3f}s; "
            f"box reference {t_box - t_reduce:.3f}s; "
            f"total {t_box - t_all:.3f}s"
        )
    return q, mean_intensity, std_intensity, box_intensity, point_counts


def save_scattering_curve(
    path: str | Path,
    q: np.ndarray,
    intensity: np.ndarray,
    *,
    std: np.ndarray | None = None,
    box_intensity: np.ndarray | None = None,
    corrected_intensity: np.ndarray | None = None,
    correction_floor: float | None = None,
) -> Path:
    """Save Q and I(Q) as CSV, with optional uncertainty and box reference."""

    out_path = Path(path)
    columns = [np.asarray(q), np.asarray(intensity)]
    names = ["Q", "I(Q)"]
    if std is not None:
        columns.append(np.asarray(std))
        names.append("I_std")
    if box_intensity is not None:
        columns.append(np.asarray(box_intensity))
        names.append("I_box(Q)")
    if corrected_intensity is not None:
        columns.append(np.asarray(corrected_intensity))
        names.append("I_density_floor_corrected(Q)")
    if correction_floor is not None:
        columns.append(np.full_like(np.asarray(q, dtype=float), float(correction_floor)))
        names.append("density_floor_c")
    data = np.column_stack(columns)
    np.savetxt(out_path, data, delimiter=",", header=",".join(names), comments="")
    return out_path


def make_structure_plotter(structure: LineStructure) -> pv.Plotter:
    """Create a PyVista preview of the line structure used for scattering."""

    plotter = pv.Plotter(window_size=r.WINDOW_SIZE)
    plotter.set_background(r.BACKGROUND_COLOR)
    if SHOW_STRUCTURE_LINES and structure.s12_lines.n_points:
        plotter.add_mesh(
            structure.s12_lines.tube(radius=r.VORTEX_TUBE_RADIUS),
            color=r.S12_TUBE_COLOR,
            opacity=r.S12_TUBE_OPACITY,
            smooth_shading=True,
            name="S12",
        )
    if SHOW_STRUCTURE_LINES and structure.s13_lines.n_points:
        plotter.add_mesh(
            structure.s13_lines.tube(radius=r.VORTEX_TUBE_RADIUS),
            color=r.S13_TUBE_COLOR,
            opacity=r.S13_TUBE_OPACITY,
            smooth_shading=True,
            name="S13",
        )
    if len(structure.crosslink_points):
        plotter.add_mesh(
            r.make_crosslink_node_mesh(structure.crosslink_points, radius=r.CROSSLINK_BALL_RADIUS),
            color=r.CROSSLINK_COLOR,
            smooth_shading=True,
            name="crosslinks",
        )
    if SHOW_SCATTERING_SAMPLE_POINTS and len(structure.points):
        sample_cloud = pv.PolyData(structure.points)
        if SCATTERING_SAMPLE_POINTS_AS_BALLS:
            sample_mesh = sample_cloud.glyph(
                geom=pv.Sphere(
                    radius=float(SCATTERING_SAMPLE_BALL_RADIUS),
                    theta_resolution=12,
                    phi_resolution=12,
                ),
                scale=False,
            )
            plotter.add_mesh(
                sample_mesh,
                color=SCATTERING_SAMPLE_POINT_COLOR,
                opacity=SCATTERING_SAMPLE_POINT_OPACITY,
                smooth_shading=True,
                name="scattering_sample_balls",
            )
        else:
            plotter.add_mesh(
                sample_cloud,
                color=SCATTERING_SAMPLE_POINT_COLOR,
                opacity=SCATTERING_SAMPLE_POINT_OPACITY,
                point_size=SCATTERING_SAMPLE_POINT_SIZE,
                render_points_as_spheres=True,
                name="scattering_sample_points",
            )
    if r.SHOW_BOUNDING_BOX:
        r.add_fixed_box_outline(plotter)
    plotter.add_axes(line_width=2, labels_off=False)
    active_grid_size = r.expanded_grid_size(r.GRID_SIZE, r.NUM_BLOCK)
    r.set_camera_from_angles(plotter, active_grid_size)
    plotter.camera.zoom(r.CAMERA_ZOOM)
    plotter.camera.reset_clipping_range()
    r.add_camera_and_top_lights(plotter, active_grid_size)
    return plotter


def main() -> None:
    if int(NUM_SEED_AVERAGE) > 1:
        q, intensity, std, box_intensity, point_counts = compute_seed_averaged_scattering()
        print(f"Averaged seeds: {NUM_SEED_AVERAGE}")
        print(f"Scattering points per seed: {point_counts}")
        structure = None
    else:
        q, intensity, structure = compute_current_scattering()
        std = None
        box_intensity = box_scattering_intensity_1d(q)
        print(f"Scattering points: {len(structure.points)}")
    print(f"Q range: {q[0]:.6g} to {q[-1]:.6g}; {len(q)} bins")

    if SAVE_SCATTERING_DATA:
        out_path = save_scattering_curve(
            SCATTERING_OUTPUT_PATH,
            q,
            intensity,
            std=std,
            box_intensity=box_intensity,
        )
        print(f"Saved scattering curve: {out_path}")

    if SAVE_REAL_SPACE_SCREENSHOT and structure is not None:
        plotter = make_structure_plotter(structure)
        plotter.show(screenshot=REAL_SPACE_SCREENSHOT_PATH)
        print(f"Saved structure screenshot: {REAL_SPACE_SCREENSHOT_PATH}")


if __name__ == "__main__":
    main()
