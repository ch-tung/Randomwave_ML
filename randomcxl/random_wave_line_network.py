"""
Three-field random-wave line network in a periodic cubic box.

Geometry:
    A single scalar zero set phi_a = 0 is a surface in 3D.
    A pairwise zero set phi_a = 0 and phi_b = 0 is a line in 3D.
    Therefore Gamma_12 and Gamma_13 define two random line families.

    Their common triple-zero points,

        phi_1 = phi_2 = phi_3 = 0,

    act as crosslink nodes between the two retained line families. Since only
    the (1, 2) and (1, 3) line families are retained here, a generic triple-zero
    node has four local branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pyvista as pv
from scipy.interpolate import splev, splrep
from scipy.spatial import cKDTree


# =============================================================================
# Editable parameters
# =============================================================================

GRID_SIZE = 128
RANDOM_SEED = 20260526

# Number of modes per field. Use one integer for all fields or a 3-tuple.
NUM_MODES: int | tuple[int, int, int] = (180, 180, 180)

# K-distribution choices:
#   "single_shell"   : |k| = K0 with isotropic directions
#   "gaussian_radial": |k| ~ Normal(K0, r_SIGMA_K*K0) with isotropic directions
#   "uniform_band"   : |k| in [r_K_MIN*K0, r_K_MAX*K0]
#   "user_list"      : use USER_K_VECTORS below
K_DISTRIBUTION: Literal[
    "single_shell", "gaussian_radial", "uniform_band", "user_list"
] = "single_shell"

# Use one float for all fields or a 3-tuple for (phi_1, phi_2, phi_3).
K0: float | tuple[float, float, float] = 10.0

# Relative k-distribution controls. Use one float for all fields or a 3-tuple
# for (phi_1, phi_2, phi_3). The absolute values passed to the sampler are:
#   sigma_k = r_SIGMA_K*K0
#   k_min   = r_K_MIN*K0
#   k_max   = r_K_MAX*K0
r_SIGMA_K: float | tuple[float, float, float] = 0.15
r_K_MIN: float | tuple[float, float, float] = 0.7
r_K_MAX: float | tuple[float, float, float] = 1.3

# Optional correlation control for phi_2 and phi_3. If enabled, two independent
# base waves Sa and Sb are generated, then:
#   phi_2 = normalize(Sa + c*Sb)
#   phi_3 = normalize(Sa - c*Sb)
# For iid Gaussian base waves, corr(phi_2, phi_3) = (1-c^2)/(1+c^2).
COUPLE_PHI2_PHI3 = False
PHI23_COUPLING_C = 1.0

# Used only when K_DISTRIBUTION = "user_list".
# If SHARED_K_VECTORS is True, provide one list of (kx, ky, kz) vectors.
# If SHARED_K_VECTORS is False, provide a 3-tuple of such lists, one per field.
USER_K_VECTORS: Sequence[Sequence[float]] | tuple[
    Sequence[Sequence[float]], Sequence[Sequence[float]], Sequence[Sequence[float]]
] = (
    (8.0, 0.0, 0.0),
    (0.0, 8.0, 0.0),
    (0.0, 0.0, 8.0),
)

# Share the same sampled k-vectors among phi_1, phi_2, phi_3, while keeping
# amplitudes and phases independent. If False, each field gets its own k-set.
SHARED_K_VECTORS = True

# Tube-domain thresholds for visualizing pairwise zero lines:
#   S_12 = {phi_1^2 + phi_2^2 < EPSILON_12^2}
#   S_13 = {phi_1^2 + phi_3^2 < EPSILON_13^2}
EPSILON_12 = 0.18
EPSILON_13 = 0.18

# Context zero surfaces.
SHOW_PHI1_ZERO_SURFACE = True
SHOW_PHI2_ZERO_SURFACE = False
SHOW_PHI3_ZERO_SURFACE = False

# Rendering and output.
WINDOW_SIZE = (1000, 1000)
BACKGROUND_COLOR = "white"
CAMERA_ZOOM = 0.82
CAMERA_AZIMUTH_DEGREES = 30.0
CAMERA_POLAR_DEGREES = 60.0
USE_CUSTOM_LIGHTING = True
HEADLIGHT_INTENSITY = 0.55
TOP_LIGHT_INTENSITY = 0.75
AMBIENT_LIGHT_INTENSITY = 0.18

SLACK_AUBERGINE = "#4A154B"
SLACK_BLUE = "#64C3EB"
SLACK_GREEN = "#5BB381"
SLACK_YELLOW = "#E3B34C"
SLACK_RED = "#CE375C"

S12_TUBE_COLOR = SLACK_RED
S13_TUBE_COLOR = SLACK_GREEN
S12_TUBE_OPACITY = 0.95
S13_TUBE_OPACITY = 0.95
LINE_DOMAIN_OPACITY = 0.62

PHI1_SURFACE_COLOR = "gray"
PHI2_SURFACE_COLOR = SLACK_RED
PHI3_SURFACE_COLOR = SLACK_GREEN
PHI1_SURFACE_OPACITY = 0.13
PHI2_SURFACE_OPACITY = 0.07
PHI3_SURFACE_OPACITY = 0.07
ZERO_SURFACE_OPACITY = 0.10
SHOW_PHI_SURFACE_EDGES = False
PHI_SURFACE_EDGE_COLOR = "black"
SHOW_TUBE_EDGES = False
TUBE_EDGE_COLOR = "black"
SHOW_CROSSLINK_EDGES = False
CROSSLINK_EDGE_COLOR = "black"
SHOW_BOUNDING_BOX = True
BOX_SIZE_L: float | None = None
BOUNDING_BOX_COLOR = "black"
BOUNDING_BOX_LINE_WIDTH = 1.0

# Vortex-line rendering traces phase windings of psi_12=phi1+i*phi2 and
# psi_13=phi1+i*phi3. This gives line geometry directly, avoiding the bulged
# tube-boundary isosurfaces from phi_a^2+phi_b^2=epsilon^2.
USE_VORTEX_TRACING = True
VORTEX_TUBE_RADIUS = 0.35
MIN_VORTEX_SEGMENT_LENGTH = 0.0
SMOOTH_VORTEX_LINES = True
VORTEX_SMOOTHING_SCALE = 4
VORTEX_MIN_SMOOTH_POINTS = 5
# If None, use the same style as Vortex.py: s = n_points + sqrt(2*n_points).
VORTEX_SPLINE_SMOOTHING: float | None = None

SHOW_CROSSLINK_NODES = True
CROSSLINK_SEARCH_RADIUS = 1.25
CROSSLINK_MERGE_RADIUS = 0.75
CROSSLINK_BALL_RADIUS = 0.85
CROSSLINK_COLOR = SLACK_YELLOW
CROSSLINK_RENDER_CONSECUTIVE_AS_TUBES = False
CROSSLINK_TUBE_CONNECT_RADIUS: float | None = None
ORDER_CROSSLINK_BALLS = False
CROSSLINK_TRACE_LINES = True
CROSSLINK_ORDER_LINK_RADIUS = 2.0
CROSSLINK_ORDER_SPACING = 1.5
CROSSLINK_ORDER_MIN_NODES = 3
CROSSLINK_ADJUST_TO_SMOOTHED_LINES = True
CROSSLINK_SEGMENT_CANDIDATES = 12
CROSSLINK_ADJUST_CENTER_WEIGHT = 0.25
CROSSLINK_ADJUST_CLUSTER_RADIUS = CROSSLINK_MERGE_RADIUS

SAVE_SCREENSHOT = True
SCREENSHOT_PATH = "random_wave_line_network.png"

SAVE_MESHES = False
MESH_OUTPUT_DIR = "random_wave_line_meshes"
S12_MESH_NAME = "S12_red_domain.vtp"
S13_MESH_NAME = "S13_blue_domain.vtp"


# =============================================================================
# Random-wave construction
# =============================================================================


KDistribution = Literal["single_shell", "gaussian_radial", "uniform_band", "user_list"]


@dataclass(frozen=True)
class FieldKSets:
    """Container for the wavevector sets used by the three fields."""

    phi1: np.ndarray
    phi2: np.ndarray
    phi3: np.ndarray


def _mode_counts(num_modes: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(num_modes, int):
        return (num_modes, num_modes, num_modes)
    if len(num_modes) != 3:
        raise ValueError("NUM_MODES must be an integer or a length-3 sequence.")
    return tuple(int(n) for n in num_modes)


def _field_parameter_values(value: float | Sequence[float], name: str) -> tuple[float, float, float]:
    """Return one parameter value per field from a scalar or length-3 tuple."""

    if isinstance(value, (int, float)):
        scalar = float(value)
        return (scalar, scalar, scalar)
    if len(value) != 3:
        raise ValueError(f"{name} must be a float or a length-3 sequence.")
    return tuple(float(item) for item in value)


def _isotropic_unit_vectors(count: int, rng: np.random.Generator) -> np.ndarray:
    directions = rng.normal(size=(count, 3))
    norms = np.linalg.norm(directions, axis=1)
    while np.any(norms == 0.0):
        bad = norms == 0.0
        directions[bad] = rng.normal(size=(np.count_nonzero(bad), 3))
        norms = np.linalg.norm(directions, axis=1)
    return directions / norms[:, None]


def sample_k_vectors(
    count: int,
    distribution: KDistribution,
    rng: np.random.Generator,
    *,
    k0: float = K0,
    sigma_k: float | None = None,
    k_min: float | None = None,
    k_max: float | None = None,
    user_vectors: Iterable[Sequence[float]] | None = None,
) -> np.ndarray:
    """
    Sample wavevectors normalized relative to the box size.

    The fields use cos(2*pi*k dot r_tilde + theta), where r_tilde=(x/N,y/N,z/N).
    Integer-valued k components make the waves exactly periodic on the grid.
    Isotropic radial choices below allow non-integer components to give continuous
    directional control; increase/decrease k magnitudes to tune feature scale.
    """

    if count <= 0:
        raise ValueError("count must be positive.")

    if distribution == "user_list":
        if user_vectors is None:
            raise ValueError("user_vectors must be supplied for 'user_list'.")
        vectors = np.asarray(list(user_vectors), dtype=float)
        if vectors.ndim != 2 or vectors.shape[1] != 3:
            raise ValueError("user_vectors must have shape (M, 3).")
        if len(vectors) < count:
            raise ValueError("user_vectors contains fewer vectors than requested.")
        return vectors[:count].copy()

    directions = _isotropic_unit_vectors(count, rng)

    if distribution == "single_shell":
        radii = np.full(count, float(k0))
    elif distribution == "gaussian_radial":
        if sigma_k is None:
            sigma_k = float(k0) * 0.15
        if sigma_k <= 0.0:
            raise ValueError("sigma_k must be positive for 'gaussian_radial'.")
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
            raise ValueError("k_max must exceed k_min for 'uniform_band'.")
        radii = rng.uniform(float(k_min), float(k_max), size=count)
    else:
        raise ValueError(f"Unknown k-distribution: {distribution!r}")

    return radii[:, None] * directions


def make_field_k_sets(
    num_modes: int | Sequence[int],
    distribution: KDistribution,
    rng: np.random.Generator,
    *,
    shared_k_vectors: bool,
) -> FieldKSets:
    """Create the k-vector sets for phi_1, phi_2, and phi_3."""

    counts = _mode_counts(num_modes)
    k0_values = _field_parameter_values(K0, "K0")
    r_sigma_k_values = _field_parameter_values(r_SIGMA_K, "r_SIGMA_K")
    r_k_min_values = _field_parameter_values(r_K_MIN, "r_K_MIN")
    r_k_max_values = _field_parameter_values(r_K_MAX, "r_K_MAX")
    sigma_k_values = tuple(r_sigma * k0 for r_sigma, k0 in zip(r_sigma_k_values, k0_values))
    k_min_values = tuple(r_k_min * k0 for r_k_min, k0 in zip(r_k_min_values, k0_values))
    k_max_values = tuple(r_k_max * k0 for r_k_max, k0 in zip(r_k_max_values, k0_values))

    if distribution == "user_list" and not shared_k_vectors:
        if len(USER_K_VECTORS) != 3:
            raise ValueError(
                "For independent user_list k-sets, USER_K_VECTORS must be a 3-tuple."
            )
        return FieldKSets(
            sample_k_vectors(counts[0], distribution, rng, user_vectors=USER_K_VECTORS[0]),
            sample_k_vectors(counts[1], distribution, rng, user_vectors=USER_K_VECTORS[1]),
            sample_k_vectors(counts[2], distribution, rng, user_vectors=USER_K_VECTORS[2]),
        )

    if shared_k_vectors:
        base_count = max(counts)
        if (
            len(set(k0_values)) > 1
            or len(set(sigma_k_values)) > 1
            or len(set(k_min_values)) > 1
            or len(set(k_max_values)) > 1
        ):
            raise ValueError(
                "SHARED_K_VECTORS=True requires scalar k-distribution parameters. "
                "Use SHARED_K_VECTORS=False for per-field K0/r_SIGMA_K/r_K_MIN/r_K_MAX tuples."
            )
        base_vectors = sample_k_vectors(
            base_count,
            distribution,
            rng,
            k0=k0_values[0],
            sigma_k=sigma_k_values[0],
            k_min=k_min_values[0],
            k_max=k_max_values[0],
            user_vectors=USER_K_VECTORS,
        )
        return FieldKSets(
            base_vectors[: counts[0]].copy(),
            base_vectors[: counts[1]].copy(),
            base_vectors[: counts[2]].copy(),
        )

    return FieldKSets(
        sample_k_vectors(
            counts[0],
            distribution,
            rng,
            k0=k0_values[0],
            sigma_k=sigma_k_values[0],
            k_min=k_min_values[0],
            k_max=k_max_values[0],
        ),
        sample_k_vectors(
            counts[1],
            distribution,
            rng,
            k0=k0_values[1],
            sigma_k=sigma_k_values[1],
            k_min=k_min_values[1],
            k_max=k_max_values[1],
        ),
        sample_k_vectors(
            counts[2],
            distribution,
            rng,
            k0=k0_values[2],
            sigma_k=sigma_k_values[2],
            k_min=k_min_values[2],
            k_max=k_max_values[2],
        ),
    )


def build_random_wave_field(
    grid_size: int,
    k_vectors: np.ndarray,
    rng: np.random.Generator,
    *,
    amplitude_scale: float = 1.0,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Build one real random-wave field and normalize it to zero mean/unit std.

    The returned array is indexed as field[x, y, z]. This convention matches
    PyVista point-data insertion when flattened with order="F".
    """

    n = int(grid_size)
    coords = np.arange(n, dtype=np.float64) / float(n)
    x = coords[:, None, None]
    y = coords[None, :, None]
    z = coords[None, None, :]

    field = np.zeros((n, n, n), dtype=np.float64)
    amplitudes = rng.normal(loc=0.0, scale=amplitude_scale, size=len(k_vectors))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(k_vectors))

    for (kx, ky, kz), amplitude, phase0 in zip(k_vectors, amplitudes, phases):
        phase = 2.0 * np.pi * (kx * x + ky * y + kz * z) + phase0
        field += amplitude * np.cos(phase)

    return normalize_field(field, dtype=dtype)


def normalize_field(field: np.ndarray, *, dtype: np.dtype = np.float32) -> np.ndarray:
    """Normalize a field to zero mean and unit standard deviation."""

    field = np.asarray(field, dtype=np.float64)
    field -= np.mean(field)
    std = np.std(field)
    if std == 0.0:
        raise RuntimeError("Field has zero standard deviation.")
    field /= std
    return field.astype(dtype, copy=False)


def build_correlated_phi2_phi3_fields(
    grid_size: int,
    k_vectors_a: np.ndarray,
    k_vectors_b: np.ndarray,
    rng: np.random.Generator,
    *,
    coupling_c: float = PHI23_COUPLING_C,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build phi_2 and phi_3 from two independent base waves Sa and Sb.

    The construction is:
        phi_2 = normalize(Sa + c*Sb)
        phi_3 = normalize(Sa - c*Sb)

    If Sa and Sb are iid Gaussian random fields, the resulting pointwise
    correlation is (1-c^2)/(1+c^2). Thus c=0 gives complete overlap and c=1
    gives zero correlation, which is independence in the Gaussian limit.
    """

    c = float(coupling_c)
    if c < 0.0:
        raise ValueError("PHI23_COUPLING_C must be non-negative.")

    sa = build_random_wave_field(grid_size, k_vectors_a, rng, dtype=np.float64)
    sb = build_random_wave_field(grid_size, k_vectors_b, rng, dtype=np.float64)
    phi2 = normalize_field(sa + c * sb, dtype=dtype)
    phi3 = normalize_field(sa - c * sb, dtype=dtype)
    return phi2, phi3


def theoretical_phi23_correlation(coupling_c: float) -> float:
    """Correlation from phi2=Sa+c*Sb and phi3=Sa-c*Sb for iid base waves."""

    c = float(coupling_c)
    return (1.0 - c * c) / (1.0 + c * c)


def make_line_scalar(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    """Return phi_a^2 + phi_b^2, whose small values approximate a zero line."""

    return np.square(phi_a, dtype=np.float32) + np.square(phi_b, dtype=np.float32)


def _wrapped_phase_difference(delta: np.ndarray) -> np.ndarray:
    """Wrap phase differences to (-pi, pi]."""

    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _plaquette_winding(
    p00: np.ndarray,
    p10: np.ndarray,
    p11: np.ndarray,
    p01: np.ndarray,
) -> np.ndarray:
    """Integer phase winding around oriented plaquettes."""

    total = (
        _wrapped_phase_difference(p10 - p00)
        + _wrapped_phase_difference(p11 - p10)
        + _wrapped_phase_difference(p01 - p11)
        + _wrapped_phase_difference(p00 - p01)
    )
    return np.rint(total / (2.0 * np.pi)).astype(np.int8)


def phase_winding_faces(phi_real: np.ndarray, phi_imag: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect phase winding on grid plaquettes for psi = phi_real + i*phi_imag.

    Returns:
        wx: winding on yz faces, indexed as wx[x_face, y_cell, z_cell]
        wy: winding on xz faces, indexed as wy[x_cell, y_face, z_cell]
        wz: winding on xy faces, indexed as wz[x_cell, y_cell, z_face]

    The implementation uses non-periodic boundary cells. Periodic wrapping can
    be added later by replacing the endpoint slices with rolled arrays and
    connecting segments across opposite box faces.
    """

    phase = np.angle(phi_real + 1j * phi_imag)

    wx = _plaquette_winding(
        phase[:, :-1, :-1],
        phase[:, 1:, :-1],
        phase[:, 1:, 1:],
        phase[:, :-1, 1:],
    )
    wy = _plaquette_winding(
        phase[:-1, :, :-1],
        phase[:-1, :, 1:],
        phase[1:, :, 1:],
        phase[1:, :, :-1],
    )
    wz = _plaquette_winding(
        phase[:-1, :-1, :],
        phase[1:, :-1, :],
        phase[1:, 1:, :],
        phase[:-1, 1:, :],
    )
    return wx, wy, wz


def trace_vortex_segments(
    phi_real: np.ndarray,
    phi_imag: np.ndarray,
    *,
    min_segment_length: float = MIN_VORTEX_SEGMENT_LENGTH,
) -> pv.PolyData:
    """
    Trace vortex-line segments from phase-winding plaquettes.

    Each cell collects the centers of its pierced faces. Cells with two pierced
    faces become one segment; cells with more pierced faces connect each pierced
    face to the local centroid. This is a compact marching-vortex construction
    for the pairwise zero set Re(psi)=0, Im(psi)=0.
    """

    wx, wy, wz = phase_winding_faces(phi_real, phi_imag)
    nx, ny, nz = phi_real.shape

    points: list[tuple[float, float, float]] = []
    lines: list[int] = []

    def add_segment(a: np.ndarray, b: np.ndarray) -> None:
        if np.linalg.norm(b - a) < min_segment_length:
            return
        start = len(points)
        points.append(tuple(a))
        points.append(tuple(b))
        lines.extend((2, start, start + 1))

    face_count = np.zeros((nx - 1, ny - 1, nz - 1), dtype=np.uint8)
    face_count += (wx[:-1, :, :] != 0)
    face_count += (wx[1:, :, :] != 0)
    face_count += (wy[:, :-1, :] != 0)
    face_count += (wy[:, 1:, :] != 0)
    face_count += (wz[:, :, :-1] != 0)
    face_count += (wz[:, :, 1:] != 0)

    active_cells = np.argwhere(face_count >= 2)
    for i, j, k in active_cells:
        face_points: list[np.ndarray] = []

        if wx[i, j, k] != 0:
            face_points.append(np.array([i, j + 0.5, k + 0.5], dtype=float))
        if wx[i + 1, j, k] != 0:
            face_points.append(np.array([i + 1, j + 0.5, k + 0.5], dtype=float))
        if wy[i, j, k] != 0:
            face_points.append(np.array([i + 0.5, j, k + 0.5], dtype=float))
        if wy[i, j + 1, k] != 0:
            face_points.append(np.array([i + 0.5, j + 1, k + 0.5], dtype=float))
        if wz[i, j, k] != 0:
            face_points.append(np.array([i + 0.5, j + 0.5, k], dtype=float))
        if wz[i, j, k + 1] != 0:
            face_points.append(np.array([i + 0.5, j + 0.5, k + 1], dtype=float))

        if len(face_points) == 2:
            add_segment(face_points[0], face_points[1])
        elif len(face_points) > 2:
            center = np.mean(face_points, axis=0)
            for face_point in face_points:
                add_segment(face_point, center)

    poly = pv.PolyData()
    if points:
        poly.points = np.asarray(points)
        poly.lines = np.asarray(lines, dtype=np.int_)
    return poly


def _polydata_segments_to_paths(poly: pv.PolyData) -> list[np.ndarray]:
    """Convert a segment-only PolyData graph into ordered point paths."""

    if poly.n_points == 0 or poly.n_lines == 0:
        return []

    raw_points = np.asarray(poly.points)
    point_ids: dict[tuple[float, float, float], int] = {}
    points: list[np.ndarray] = []
    adjacency: dict[int, set[int]] = {}

    def merged_id(point: np.ndarray) -> int:
        key = tuple(np.round(point.astype(float), 6))
        if key not in point_ids:
            point_ids[key] = len(points)
            points.append(np.asarray(point, dtype=float))
            adjacency[point_ids[key]] = set()
        return point_ids[key]

    lines = np.asarray(poly.lines)
    cursor = 0
    edges: set[tuple[int, int]] = set()
    while cursor < len(lines):
        n_in_cell = int(lines[cursor])
        ids = lines[cursor + 1 : cursor + 1 + n_in_cell]
        cursor += n_in_cell + 1
        if n_in_cell != 2:
            continue
        a = merged_id(raw_points[int(ids[0])])
        b = merged_id(raw_points[int(ids[1])])
        if a == b:
            continue
        edge = tuple(sorted((a, b)))
        if edge in edges:
            continue
        edges.add(edge)
        adjacency[a].add(b)
        adjacency[b].add(a)

    visited_edges: set[tuple[int, int]] = set()
    paths: list[np.ndarray] = []

    def walk(start: int, nxt: int) -> list[int]:
        path = [start, nxt]
        visited_edges.add(tuple(sorted((start, nxt))))
        prev, cur = start, nxt

        while True:
            candidates = [
                node
                for node in adjacency[cur]
                if node != prev and tuple(sorted((cur, node))) not in visited_edges
            ]
            if len(candidates) != 1 or len(adjacency[cur]) != 2:
                break
            nxt_node = candidates[0]
            visited_edges.add(tuple(sorted((cur, nxt_node))))
            path.append(nxt_node)
            prev, cur = cur, nxt_node
        return path

    branch_or_end_nodes = [node for node, neighbors in adjacency.items() if len(neighbors) != 2]
    for node in branch_or_end_nodes:
        for neighbor in adjacency[node]:
            edge = tuple(sorted((node, neighbor)))
            if edge not in visited_edges:
                paths.append(np.array([points[idx] for idx in walk(node, neighbor)]))

    for a, b in edges:
        edge = tuple(sorted((a, b)))
        if edge in visited_edges:
            continue
        loop = walk(a, b)
        paths.append(np.array([points[idx] for idx in loop]))

    return [path for path in paths if len(path) >= 2]


def _smooth_path(path: np.ndarray) -> np.ndarray:
    """Spline-smooth one ordered vortex path, following the approach in Vortex.py."""

    if len(path) < VORTEX_MIN_SMOOTH_POINTS:
        return path

    x = np.arange(len(path), dtype=float)
    x_fine = np.linspace(0.0, len(path) - 1.0, max(len(path) * VORTEX_SMOOTHING_SCALE, len(path)))
    smoothing = (
        len(path) + np.sqrt(2.0 * len(path))
        if VORTEX_SPLINE_SMOOTHING is None
        else float(VORTEX_SPLINE_SMOOTHING)
    )
    k = min(3, len(path) - 1)
    splines = [splrep(x, path[:, axis], s=smoothing, k=k) for axis in range(3)]
    return np.array([splev(x_fine, spline) for spline in splines]).T


def smooth_vortex_polydata(poly: pv.PolyData) -> pv.PolyData:
    """Merge traced vortex segments into ordered paths and spline-smooth them."""

    paths = _polydata_segments_to_paths(poly)
    if not paths:
        return poly

    out = pv.PolyData()
    points: list[np.ndarray] = []
    lines: list[int] = []

    for path in paths:
        smooth_path = _smooth_path(path) if SMOOTH_VORTEX_LINES else path
        if len(smooth_path) < 2:
            continue
        start = len(points)
        points.extend(smooth_path)
        lines.extend([len(smooth_path), *range(start, start + len(smooth_path))])

    if points:
        out.points = np.asarray(points)
        out.lines = np.asarray(lines, dtype=np.int_)
    return out


def _cluster_points(points: np.ndarray, radius: float) -> np.ndarray:
    """Cluster nearby points and return one centroid per connected cluster."""

    if len(points) == 0:
        return np.empty((0, 3), dtype=float)

    tree = cKDTree(points)
    pairs = tree.query_pairs(r=radius)
    parent = np.arange(len(points))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    for i, j in pairs:
        union(i, j)

    roots = np.array([find(i) for i in range(len(points))])
    centers = []
    for root in np.unique(roots):
        centers.append(np.mean(points[roots == root], axis=0))
    return np.asarray(centers)


def find_crosslink_nodes(
    s12_lines: pv.PolyData,
    s13_lines: pv.PolyData,
    *,
    search_radius: float = CROSSLINK_SEARCH_RADIUS,
    merge_radius: float | None = None,
) -> np.ndarray:
    """
    Estimate triple-zero crosslink nodes from close contacts of two line families.

    The retained line families are Gamma_12=(phi1=phi2=0) and
    Gamma_13=(phi1=phi3=0). Their intersections are triple zeros. On a grid, the
    two independently traced line sets rarely share exactly identical vertices,
    so this function finds close vertex pairs and clusters their midpoints.
    Candidate midpoint clusters farther apart than merge_radius remain separate.
    """

    if merge_radius is None:
        merge_radius = CROSSLINK_MERGE_RADIUS

    candidate_midpoints = find_crosslink_candidate_midpoints(
        s12_lines,
        s13_lines,
        search_radius=search_radius,
    )
    if len(candidate_midpoints) == 0:
        return np.empty((0, 3), dtype=float)

    return _cluster_points(candidate_midpoints, radius=float(merge_radius))


def find_crosslink_candidate_midpoints(
    s12_lines: pv.PolyData,
    s13_lines: pv.PolyData,
    *,
    search_radius: float = CROSSLINK_SEARCH_RADIUS,
) -> np.ndarray:
    """
    Return dense close-contact midpoint candidates before crosslink merging.

    This is useful when phi_2 and phi_3 are strongly coupled. Then Gamma_12 and
    Gamma_13 can overlap for extended distances, and clustering all candidates
    immediately would collapse many visible contact sites into one node.
    """

    if s12_lines.n_points == 0 or s13_lines.n_points == 0:
        return np.empty((0, 3), dtype=float)

    p12 = np.asarray(s12_lines.points)
    p13 = np.asarray(s13_lines.points)
    tree13 = cKDTree(p13)
    distances, ids = tree13.query(p12, k=1, distance_upper_bound=float(search_radius))
    valid = np.isfinite(distances) & (ids < len(p13))
    if not np.any(valid):
        return np.empty((0, 3), dtype=float)

    return 0.5 * (p12[valid] + p13[ids[valid]])


def _polydata_line_segments(poly: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
    """Return start/end arrays for every consecutive segment in a line PolyData."""

    if poly.n_points == 0 or poly.n_lines == 0:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty

    points = np.asarray(poly.points)
    lines = np.asarray(poly.lines)
    starts = []
    ends = []
    cursor = 0
    while cursor < len(lines):
        n_in_cell = int(lines[cursor])
        ids = lines[cursor + 1 : cursor + 1 + n_in_cell].astype(int)
        cursor += n_in_cell + 1
        if n_in_cell < 2:
            continue
        for a_id, b_id in zip(ids[:-1], ids[1:]):
            starts.append(points[a_id])
            ends.append(points[b_id])

    if not starts:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty
    return np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)


def _closest_points_between_segments(
    p1: np.ndarray,
    q1: np.ndarray,
    p2: np.ndarray,
    q2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Closest points on two 3D line segments."""

    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    eps = 1e-12

    if a <= eps and e <= eps:
        c1 = p1
        c2 = p2
        return c1, c2, float(np.linalg.norm(c1 - c2))

    if a <= eps:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = float(np.dot(d1, r))
        if e <= eps:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            if denom != 0.0:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0

            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    c1 = p1 + s * d1
    c2 = p2 + t * d2
    return c1, c2, float(np.linalg.norm(c1 - c2))


def adjust_crosslink_nodes_to_smoothed_lines(
    raw_centers: np.ndarray,
    s12_lines: pv.PolyData,
    s13_lines: pv.PolyData,
    *,
    candidate_count: int = CROSSLINK_SEGMENT_CANDIDATES,
    center_weight: float = CROSSLINK_ADJUST_CENTER_WEIGHT,
    cluster_radius: float | None = None,
) -> np.ndarray:
    """
    Move raw crosslink estimates onto the intersection of the smoothed lines.

    For each raw node, nearby smoothed segments from both retained line families
    are considered together. The adjusted node is the midpoint between the
    closest points on the best pair of smoothed segments.
    """

    if len(raw_centers) == 0:
        return raw_centers

    s12_starts, s12_ends = _polydata_line_segments(s12_lines)
    s13_starts, s13_ends = _polydata_line_segments(s13_lines)
    if len(s12_starts) == 0 or len(s13_starts) == 0:
        return raw_centers

    s12_midpoints = 0.5 * (s12_starts + s12_ends)
    s13_midpoints = 0.5 * (s13_starts + s13_ends)
    tree12 = cKDTree(s12_midpoints)
    tree13 = cKDTree(s13_midpoints)
    k12 = min(int(candidate_count), len(s12_midpoints))
    k13 = min(int(candidate_count), len(s13_midpoints))

    adjusted = []
    for center in raw_centers:
        ids12 = np.atleast_1d(tree12.query(center, k=k12)[1]).astype(int)
        ids13 = np.atleast_1d(tree13.query(center, k=k13)[1]).astype(int)

        best_score = np.inf
        best_point = center
        for id12 in ids12:
            for id13 in ids13:
                c12, c13, segment_distance = _closest_points_between_segments(
                    s12_starts[id12],
                    s12_ends[id12],
                    s13_starts[id13],
                    s13_ends[id13],
                )
                midpoint = 0.5 * (c12 + c13)
                score = segment_distance + center_weight * float(np.linalg.norm(midpoint - center))
                if score < best_score:
                    best_score = score
                    best_point = midpoint
        adjusted.append(best_point)

    adjusted_points = np.asarray(adjusted)
    if cluster_radius is None or cluster_radius <= 0:
        return adjusted_points
    return _cluster_points(adjusted_points, radius=float(cluster_radius))


def make_crosslink_node_mesh(
    centers: np.ndarray,
    *,
    radius: float | None = None,
) -> pv.PolyData:
    """Create a merged sphere mesh for crosslink nodes."""

    if radius is None:
        radius = CROSSLINK_BALL_RADIUS

    merged = pv.PolyData()
    for center in centers:
        sphere = pv.Sphere(radius=radius, center=tuple(center), theta_resolution=20, phi_resolution=20)
        merged = merged.merge(sphere)
    return merged


def make_capped_crosslink_tube_mesh(
    centers: np.ndarray,
    *,
    radius: float | None = None,
    connect_radius: float | None = None,
) -> tuple[pv.PolyData, pv.PolyData]:
    """
    Render consecutive crosslink centers as tubes with spherical end caps.

    Returned meshes are (tube_mesh, cap_mesh). Components with only one center
    are returned as a single sphere in cap_mesh.
    """

    if radius is None:
        radius = CROSSLINK_BALL_RADIUS
    if connect_radius is None:
        connect_radius = 1.25 * float(CROSSLINK_ORDER_SPACING)

    tube_mesh = pv.PolyData()
    cap_centers = []
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 0:
        return tube_mesh, make_crosslink_node_mesh(np.empty((0, 3)), radius=radius)
    if len(centers) == 1:
        return tube_mesh, make_crosslink_node_mesh(centers, radius=radius)

    tree = cKDTree(centers)
    pairs = [(int(i), int(j)) for i, j in tree.query_pairs(r=float(connect_radius))]
    if not pairs:
        return tube_mesh, make_crosslink_node_mesh(centers, radius=radius)

    adjacency = _minimum_spanning_adjacency(centers, pairs)
    visited: set[int] = set()
    points: list[np.ndarray] = []
    lines: list[int] = []

    for start in range(len(centers)):
        if start in visited:
            continue
        stack = [start]
        component = []
        visited.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor, _ in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        if len(component) == 1:
            cap_centers.append(centers[component[0]])
            continue

        path_ids = _tree_farthest_path(adjacency, component)
        path_points = centers[path_ids]
        if len(path_points) < 2:
            cap_centers.extend(centers[component])
            continue

        path_start = len(points)
        points.extend(path_points)
        lines.extend([len(path_points), *range(path_start, path_start + len(path_points))])
        cap_centers.append(path_points[0])
        cap_centers.append(path_points[-1])

    if points:
        line_mesh = pv.PolyData()
        line_mesh.points = np.asarray(points, dtype=float)
        line_mesh.lines = np.asarray(lines, dtype=np.int_)
        tube_mesh = line_mesh.tube(radius=float(radius), capping=True)

    cap_mesh = make_crosslink_node_mesh(np.asarray(cap_centers), radius=radius)
    return tube_mesh, cap_mesh


def _minimum_spanning_adjacency(
    centers: np.ndarray,
    pairs: list[tuple[int, int]],
) -> dict[int, list[tuple[int, float]]]:
    """Build a sparse nearest-neighbor tree from candidate close pairs."""

    parent = np.arange(len(centers))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    adjacency: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(centers))}
    weighted_pairs = [
        (float(np.linalg.norm(centers[i] - centers[j])), int(i), int(j))
        for i, j in pairs
    ]
    for distance, i, j in sorted(weighted_pairs):
        ri = find(i)
        rj = find(j)
        if ri == rj:
            continue
        parent[rj] = ri
        adjacency[i].append((j, distance))
        adjacency[j].append((i, distance))
    return adjacency


def _tree_farthest_path(
    adjacency: dict[int, list[tuple[int, float]]],
    component: list[int],
) -> list[int]:
    """Return the weighted-diameter path through one tree component."""

    def farthest_from(start: int) -> tuple[int, dict[int, int | None]]:
        distances = {start: 0.0}
        parents: dict[int, int | None] = {start: None}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor, weight in adjacency[node]:
                if neighbor in distances:
                    continue
                distances[neighbor] = distances[node] + weight
                parents[neighbor] = node
                stack.append(neighbor)
        farthest = max(distances, key=distances.get)
        return farthest, parents

    a, _ = farthest_from(component[0])
    b, parents = farthest_from(a)
    path = [b]
    while path[-1] != a:
        parent = parents[path[-1]]
        if parent is None:
            break
        path.append(parent)
    path.reverse()
    return path


def _resample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    """Place evenly spaced points along an ordered polyline."""

    if len(points) < 2 or spacing <= 0.0:
        return points

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length == 0.0:
        return points[:1]

    sample_distances = list(np.arange(0.0, total_length, float(spacing)))
    if not np.isclose(sample_distances[-1], total_length):
        sample_distances.append(total_length)

    samples = []
    for distance in sample_distances:
        idx = int(np.searchsorted(cumulative, distance, side="right") - 1)
        idx = min(idx, len(points) - 2)
        span = cumulative[idx + 1] - cumulative[idx]
        fraction = 0.0 if span == 0.0 else (distance - cumulative[idx]) / span
        samples.append(points[idx] + fraction * (points[idx + 1] - points[idx]))
    return np.asarray(samples)


def order_crosslink_centers_with_spacing(
    centers: np.ndarray,
    *,
    link_radius: float = CROSSLINK_ORDER_LINK_RADIUS,
    spacing: float = CROSSLINK_ORDER_SPACING,
    min_nodes: int = CROSSLINK_ORDER_MIN_NODES,
) -> np.ndarray:
    """
    Redistribute close crosslink centers as evenly spaced balls along local paths.

    Nearby centers are grouped with link_radius. Components with at least
    min_nodes are ordered along their nearest-neighbor tree diameter and
    redistributed using the requested spacing.
    """

    if len(centers) < max(2, int(min_nodes)):
        return centers

    centers = np.asarray(centers, dtype=float)
    tree = cKDTree(centers)
    pairs = [(int(i), int(j)) for i, j in tree.query_pairs(r=float(link_radius))]
    if not pairs:
        return centers

    adjacency = _minimum_spanning_adjacency(centers, pairs)
    visited: set[int] = set()
    ordered_centers = []

    for start in range(len(centers)):
        if start in visited:
            continue
        stack = [start]
        component = []
        visited.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor, _ in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        if len(component) < int(min_nodes):
            ordered_centers.extend(centers[component])
            continue

        path_ids = _tree_farthest_path(adjacency, component)
        path_points = centers[path_ids]
        ordered_centers.extend(_resample_polyline(path_points, float(spacing)))

    return np.asarray(ordered_centers, dtype=float)


def trace_crosslink_lines(
    candidate_centers: np.ndarray,
    *,
    link_radius: float = CROSSLINK_ORDER_LINK_RADIUS,
    min_nodes: int = CROSSLINK_ORDER_MIN_NODES,
) -> pv.PolyData:
    """
    Trace dense crosslink contacts into line-like overlap paths.

    This follows the same idea as vortex-line rendering: local vertices are
    connected into ordered paths, optionally smoothed, and represented as a
    PyVista line mesh. Ball sites can then be sampled from this mesh.

    Note for future improvement: this point-cloud path reconstruction keeps a
    graph-diameter path in each connected component. That is stable for partial
    crosslinks, but when phi_2 and phi_3 are identical or nearly identical, the
    overlap may contain loops/branches and some portions can be underrepresented.
    A future version should preserve original vortex-line connectivity while
    still evaluating sampled/display ball centers robustly for non-identical
    phi_2 and phi_3.
    """

    out = pv.PolyData()
    if len(candidate_centers) < max(2, int(min_nodes)):
        return out

    centers = np.asarray(candidate_centers, dtype=float)
    tree = cKDTree(centers)
    pairs = [(int(i), int(j)) for i, j in tree.query_pairs(r=float(link_radius))]
    if not pairs:
        return out

    adjacency = _minimum_spanning_adjacency(centers, pairs)
    visited: set[int] = set()
    points: list[np.ndarray] = []
    lines: list[int] = []

    for start in range(len(centers)):
        if start in visited:
            continue
        stack = [start]
        component = []
        visited.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor, _ in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        if len(component) < int(min_nodes):
            continue

        path_ids = _tree_farthest_path(adjacency, component)
        path_points = centers[path_ids]
        if SMOOTH_VORTEX_LINES:
            path_points = _smooth_path(path_points)
        if len(path_points) < 2:
            continue

        path_start = len(points)
        points.extend(path_points)
        lines.extend([len(path_points), *range(path_start, path_start + len(path_points))])

    if points:
        out.points = np.asarray(points, dtype=float)
        out.lines = np.asarray(lines, dtype=np.int_)
    return out


def sample_crosslink_line_sites(
    crosslink_lines: pv.PolyData,
    *,
    spacing: float = CROSSLINK_ORDER_SPACING,
) -> np.ndarray:
    """Sample evenly spaced ball centers from traced crosslink line paths."""

    if crosslink_lines.n_points == 0 or crosslink_lines.n_lines == 0:
        return np.empty((0, 3), dtype=float)

    points = np.asarray(crosslink_lines.points)
    lines = np.asarray(crosslink_lines.lines)
    samples = []
    cursor = 0
    while cursor < len(lines):
        n_in_cell = int(lines[cursor])
        ids = lines[cursor + 1 : cursor + 1 + n_in_cell].astype(int)
        cursor += n_in_cell + 1
        if n_in_cell < 2:
            continue
        samples.extend(_resample_polyline(points[ids], float(spacing)))

    if not samples:
        return np.empty((0, 3), dtype=float)
    return np.asarray(samples, dtype=float)


def sample_crosslink_line_site_mesh(
    crosslink_lines: pv.PolyData,
    *,
    spacing: float = CROSSLINK_ORDER_SPACING,
) -> pv.PolyData:
    """Sample displayed crosslink sites while preserving their line order."""

    out = pv.PolyData()
    if crosslink_lines.n_points == 0 or crosslink_lines.n_lines == 0:
        return out

    points = np.asarray(crosslink_lines.points)
    raw_lines = np.asarray(crosslink_lines.lines)
    sampled_points: list[np.ndarray] = []
    sampled_lines: list[int] = []
    cursor = 0
    while cursor < len(raw_lines):
        n_in_cell = int(raw_lines[cursor])
        ids = raw_lines[cursor + 1 : cursor + 1 + n_in_cell].astype(int)
        cursor += n_in_cell + 1
        if n_in_cell < 2:
            continue

        path_samples = _resample_polyline(points[ids], float(spacing))
        if len(path_samples) == 0:
            continue

        start = len(sampled_points)
        sampled_points.extend(path_samples)
        if len(path_samples) >= 2:
            sampled_lines.extend([len(path_samples), *range(start, start + len(path_samples))])

    if sampled_points:
        out.points = np.asarray(sampled_points, dtype=float)
        if sampled_lines:
            out.lines = np.asarray(sampled_lines, dtype=np.int_)
    return out


def adjust_crosslink_site_mesh_to_smoothed_lines(
    site_mesh: pv.PolyData,
    s12_lines: pv.PolyData,
    s13_lines: pv.PolyData,
) -> pv.PolyData:
    """Adjust sampled site positions while keeping their existing path order."""

    if site_mesh.n_points == 0:
        return site_mesh

    adjusted_points = adjust_crosslink_nodes_to_smoothed_lines(
        np.asarray(site_mesh.points),
        s12_lines,
        s13_lines,
        cluster_radius=None,
    )
    out = pv.PolyData()
    out.points = adjusted_points
    if site_mesh.n_lines:
        out.lines = np.asarray(site_mesh.lines, dtype=np.int_)
    return out


def break_crosslink_site_mesh_by_spacing(
    site_mesh: pv.PolyData,
    *,
    max_segment_length: float | None = None,
) -> pv.PolyData:
    """Break display-site paths where consecutive centers are too far apart."""

    if site_mesh.n_points == 0 or site_mesh.n_lines == 0:
        return site_mesh
    if max_segment_length is None:
        max_segment_length = CROSSLINK_TUBE_CONNECT_RADIUS
    if max_segment_length is None:
        max_segment_length = 1.25 * float(CROSSLINK_ORDER_SPACING)

    points = np.asarray(site_mesh.points)
    raw_lines = np.asarray(site_mesh.lines)
    out_points: list[np.ndarray] = []
    out_lines: list[int] = []
    cursor = 0
    while cursor < len(raw_lines):
        n_in_cell = int(raw_lines[cursor])
        ids = raw_lines[cursor + 1 : cursor + 1 + n_in_cell].astype(int)
        cursor += n_in_cell + 1
        if n_in_cell == 0:
            continue

        run: list[int] = [int(ids[0])]
        for point_id in ids[1:]:
            previous_id = run[-1]
            distance = float(np.linalg.norm(points[int(point_id)] - points[previous_id]))
            if distance <= float(max_segment_length):
                run.append(int(point_id))
            else:
                if len(run) >= 2:
                    start = len(out_points)
                    out_points.extend(points[run])
                    out_lines.extend([len(run), *range(start, start + len(run))])
                elif len(run) == 1:
                    out_points.append(points[run[0]])
                run = [int(point_id)]

        if len(run) >= 2:
            start = len(out_points)
            out_points.extend(points[run])
            out_lines.extend([len(run), *range(start, start + len(run))])
        elif len(run) == 1:
            out_points.append(points[run[0]])

    out = pv.PolyData()
    if out_points:
        out.points = np.asarray(out_points, dtype=float)
        if out_lines:
            out.lines = np.asarray(out_lines, dtype=np.int_)
    return out


def bridge_crosslink_site_mesh_endpoints(
    site_mesh: pv.PolyData,
    *,
    connect_radius: float | None = None,
) -> pv.PolyData:
    """Join separate sampled-site paths when their endpoints are close enough."""

    if site_mesh.n_points == 0:
        return site_mesh
    if connect_radius is None:
        connect_radius = CROSSLINK_TUBE_CONNECT_RADIUS
    if connect_radius is None:
        connect_radius = 1.25 * float(CROSSLINK_ORDER_SPACING)
    connect_radius = float(connect_radius)
    if connect_radius <= 0:
        return site_mesh

    points = np.asarray(site_mesh.points)
    paths: list[np.ndarray] = []
    used_ids: set[int] = set()

    if site_mesh.n_lines:
        raw_lines = np.asarray(site_mesh.lines)
        cursor = 0
        while cursor < len(raw_lines):
            n_in_cell = int(raw_lines[cursor])
            ids = raw_lines[cursor + 1 : cursor + 1 + n_in_cell].astype(int)
            cursor += n_in_cell + 1
            if n_in_cell == 0:
                continue
            used_ids.update(int(i) for i in ids)
            paths.append(points[ids].copy())

    for point_id in range(site_mesh.n_points):
        if point_id not in used_ids:
            paths.append(points[[point_id]].copy())

    if len(paths) < 2:
        return site_mesh

    while True:
        best: tuple[float, int, int, str] | None = None
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                endpoints = (
                    (paths[i][0], paths[j][0], "ss"),
                    (paths[i][0], paths[j][-1], "se"),
                    (paths[i][-1], paths[j][0], "es"),
                    (paths[i][-1], paths[j][-1], "ee"),
                )
                for a, b, mode in endpoints:
                    distance = float(np.linalg.norm(a - b))
                    if distance <= connect_radius and (best is None or distance < best[0]):
                        best = (distance, i, j, mode)

        if best is None:
            break

        _, i, j, mode = best
        path_i = paths[i]
        path_j = paths[j]
        if mode == "ss":
            merged = np.vstack((path_i[::-1], path_j))
        elif mode == "se":
            merged = np.vstack((path_j, path_i))
        elif mode == "es":
            merged = np.vstack((path_i, path_j))
        else:
            merged = np.vstack((path_i, path_j[::-1]))

        paths[i] = merged
        del paths[j]

    out_points: list[np.ndarray] = []
    out_lines: list[int] = []
    for path in paths:
        if len(path) == 0:
            continue
        start = len(out_points)
        out_points.extend(path)
        if len(path) >= 2:
            out_lines.extend([len(path), *range(start, start + len(path))])

    out = pv.PolyData()
    if out_points:
        out.points = np.asarray(out_points, dtype=float)
        if out_lines:
            out.lines = np.asarray(out_lines, dtype=np.int_)
    return out


def _line_endpoint_centers(line_mesh: pv.PolyData) -> np.ndarray:
    """Return endpoint centers for every line cell plus isolated points."""

    if line_mesh.n_points == 0:
        return np.empty((0, 3), dtype=float)

    points = np.asarray(line_mesh.points)
    if line_mesh.n_lines == 0:
        return points.copy()

    raw_lines = np.asarray(line_mesh.lines)
    endpoint_ids: list[int] = []
    used_ids: set[int] = set()
    cursor = 0
    while cursor < len(raw_lines):
        n_in_cell = int(raw_lines[cursor])
        ids = raw_lines[cursor + 1 : cursor + 1 + n_in_cell].astype(int)
        cursor += n_in_cell + 1
        if n_in_cell == 0:
            continue
        used_ids.update(int(i) for i in ids)
        endpoint_ids.append(int(ids[0]))
        if n_in_cell > 1:
            endpoint_ids.append(int(ids[-1]))

    for point_id in range(line_mesh.n_points):
        if point_id not in used_ids:
            endpoint_ids.append(point_id)

    if not endpoint_ids:
        return np.empty((0, 3), dtype=float)
    return points[np.asarray(endpoint_ids, dtype=int)]


def enforce_crosslink_site_spacing(
    centers: np.ndarray,
    *,
    min_spacing: float = CROSSLINK_ORDER_SPACING,
) -> np.ndarray:
    """Greedily remove displayed crosslink balls closer than min_spacing."""

    if len(centers) < 2 or min_spacing <= 0:
        return centers

    centers = np.asarray(centers, dtype=float)
    kept: list[np.ndarray] = []
    for center in centers:
        if not kept:
            kept.append(center)
            continue
        distances = np.linalg.norm(np.asarray(kept) - center, axis=1)
        if np.all(distances >= float(min_spacing)):
            kept.append(center)
    return np.asarray(kept, dtype=float)


# =============================================================================
# PyVista visualization
# =============================================================================


def make_image_data(grid_size: int) -> pv.ImageData:
    """Create a point-centered ImageData grid with coordinates 0..N-1."""

    grid = pv.ImageData()
    grid.dimensions = (grid_size, grid_size, grid_size)
    grid.origin = (0.0, 0.0, 0.0)
    grid.spacing = (1.0, 1.0, 1.0)
    return grid


def add_scalar_field(grid: pv.ImageData, name: str, values: np.ndarray) -> None:
    """Attach field[x,y,z] to PyVista point data with correct VTK ordering."""

    grid.point_data[name] = np.asarray(values).ravel(order="F")


def add_isosurface_to_plotter(
    plotter: pv.Plotter,
    grid: pv.ImageData,
    scalar_name: str,
    *,
    value: float = 0.0,
    color: str = "lightgray",
    opacity: float = ZERO_SURFACE_OPACITY,
    smooth_shading: bool = True,
    show_edges: bool = SHOW_PHI_SURFACE_EDGES,
    edge_color: str = PHI_SURFACE_EDGE_COLOR,
) -> pv.PolyData:
    """Extract and render an isosurface for a scalar field."""

    surface = grid.contour(isosurfaces=[value], scalars=scalar_name)
    plotter.add_mesh(
        surface,
        color=color,
        opacity=opacity,
        smooth_shading=smooth_shading,
        show_edges=show_edges,
        edge_color=edge_color,
        specular=0.12,
        name=f"{scalar_name}_isosurface",
    )
    return surface


def extract_line_domain(
    grid: pv.ImageData,
    scalar_name: str,
    epsilon: float,
) -> pv.PolyData:
    """Extract the epsilon tube boundary as scalar_name = epsilon^2."""

    return grid.contour(isosurfaces=[float(epsilon) ** 2], scalars=scalar_name)


def add_line_domain_to_plotter(
    plotter: pv.Plotter,
    mesh: pv.PolyData,
    *,
    color: str,
    opacity: float = LINE_DOMAIN_OPACITY,
    name: str,
    show_edges: bool = SHOW_TUBE_EDGES,
    edge_color: str = TUBE_EDGE_COLOR,
) -> None:
    plotter.add_mesh(
        mesh,
        color=color,
        opacity=opacity,
        smooth_shading=True,
        show_edges=show_edges,
        edge_color=edge_color,
        specular=0.25,
        name=name,
    )


def add_camera_and_top_lights(plotter: pv.Plotter, grid_size: int) -> None:
    """Add a headlight from the camera and a top light over the box."""

    if not USE_CUSTOM_LIGHTING:
        return

    plotter.remove_all_lights()

    camera_position = np.asarray(plotter.camera.position, dtype=float)
    camera_focal_point = np.asarray(plotter.camera.focal_point, dtype=float)
    view_direction = camera_focal_point - camera_position
    norm = np.linalg.norm(view_direction)
    if norm == 0.0:
        view_direction = np.array([-1.0, -1.0, -1.0])
    else:
        view_direction /= norm

    headlight = pv.Light(
        position=tuple(camera_position),
        focal_point=tuple(camera_focal_point),
        color="white",
        light_type="scene light",
        intensity=float(HEADLIGHT_INTENSITY),
    )
    plotter.add_light(headlight)

    center = np.array([0.5 * (grid_size - 1)] * 3, dtype=float)
    top_position = center + np.array([0.0, 0.0, 2.4 * grid_size])
    top_focal_point = center + 0.15 * grid_size * view_direction
    top_light = pv.Light(
        position=tuple(top_position),
        focal_point=tuple(top_focal_point),
        color="white",
        light_type="scene light",
        intensity=float(TOP_LIGHT_INTENSITY),
    )
    plotter.add_light(top_light)

    if AMBIENT_LIGHT_INTENSITY > 0:
        ambient = pv.Light(
            color="white",
            light_type="headlight",
            intensity=float(AMBIENT_LIGHT_INTENSITY),
        )
        plotter.add_light(ambient)


def set_camera_from_angles(plotter: pv.Plotter, grid_size: int) -> None:
    """Set camera by azimuth from +x and polar angle down from +z."""

    center = np.array([0.5 * (grid_size - 1)] * 3, dtype=float)
    azimuth = np.deg2rad(float(CAMERA_AZIMUTH_DEGREES))
    polar = np.deg2rad(float(CAMERA_POLAR_DEGREES))
    direction = np.array(
        [
            np.sin(polar) * np.cos(azimuth),
            np.sin(polar) * np.sin(azimuth),
            np.cos(polar),
        ],
        dtype=float,
    )
    distance = 1.7 * np.sqrt(3.0) * max(1.0, float(grid_size - 1))
    position = center + distance * direction
    view_up = (0.0, 0.0, 1.0)
    if abs(float(np.dot(direction, view_up))) > 0.98:
        view_up = (0.0, 1.0, 0.0)

    plotter.camera.position = tuple(position)
    plotter.camera.focal_point = tuple(center)
    plotter.camera.up = view_up


def add_fixed_box_outline(plotter: pv.Plotter, *, box_size_l: float | None = None) -> pv.PolyData:
    """Add a fixed box outline from (0,0,0) to (L,L,L), independent of contents."""

    if box_size_l is None:
        box_size_l = BOX_SIZE_L
    if box_size_l is None:
        box_size_l = float(GRID_SIZE - 1)
    l_value = float(box_size_l)
    box = pv.Box(bounds=(0.0, l_value, 0.0, l_value, 0.0, l_value)).outline()
    plotter.add_mesh(
        box,
        color=BOUNDING_BOX_COLOR,
        line_width=BOUNDING_BOX_LINE_WIDTH,
        name="fixed_box_outline",
    )
    return box


def save_mesh(mesh: pv.PolyData, output_dir: str | Path, file_name: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out_path = path / file_name
    mesh.save(out_path)
    return out_path


def build_scene() -> tuple[pv.Plotter, dict[str, pv.PolyData]]:
    rng = np.random.default_rng(RANDOM_SEED)

    k_sets = make_field_k_sets(
        NUM_MODES,
        K_DISTRIBUTION,
        rng,
        shared_k_vectors=SHARED_K_VECTORS,
    )

    print("Building phi_1...")
    phi1 = build_random_wave_field(GRID_SIZE, k_sets.phi1, rng)
    if COUPLE_PHI2_PHI3:
        print(
            "Building coupled phi_2/phi_3 "
            f"(c={PHI23_COUPLING_C}, theoretical corr={theoretical_phi23_correlation(PHI23_COUPLING_C):.3f})..."
        )
        phi2, phi3 = build_correlated_phi2_phi3_fields(
            GRID_SIZE,
            k_sets.phi2,
            k_sets.phi3,
            rng,
            coupling_c=PHI23_COUPLING_C,
        )
    else:
        print("Building phi_2...")
        phi2 = build_random_wave_field(GRID_SIZE, k_sets.phi2, rng)
        print("Building phi_3...")
        phi3 = build_random_wave_field(GRID_SIZE, k_sets.phi3, rng)

    s12 = make_line_scalar(phi1, phi2)
    s13 = make_line_scalar(phi1, phi3)

    grid = make_image_data(GRID_SIZE)
    add_scalar_field(grid, "phi1", phi1)
    add_scalar_field(grid, "phi2", phi2)
    add_scalar_field(grid, "phi3", phi3)
    add_scalar_field(grid, "S12_scalar", s12)
    add_scalar_field(grid, "S13_scalar", s13)

    print("Extracting S_12 and S_13 tube-domain meshes...")
    if USE_VORTEX_TRACING:
        s12_raw_mesh = trace_vortex_segments(phi1, phi2)
        s13_raw_mesh = trace_vortex_segments(phi1, phi3)
        crosslink_candidate_centers = find_crosslink_candidate_midpoints(
            s12_raw_mesh,
            s13_raw_mesh,
            search_radius=CROSSLINK_SEARCH_RADIUS,
        )
        raw_crosslink_centers = (
            _cluster_points(crosslink_candidate_centers, radius=float(CROSSLINK_MERGE_RADIUS))
            if len(crosslink_candidate_centers)
            else np.empty((0, 3), dtype=float)
        )
        crosslink_line_mesh = trace_crosslink_lines(
            crosslink_candidate_centers,
            link_radius=CROSSLINK_ORDER_LINK_RADIUS,
            min_nodes=CROSSLINK_ORDER_MIN_NODES,
        )
        s12_mesh = smooth_vortex_polydata(s12_raw_mesh)
        s13_mesh = smooth_vortex_polydata(s13_raw_mesh)
        if CROSSLINK_ADJUST_TO_SMOOTHED_LINES:
            crosslink_centers = adjust_crosslink_nodes_to_smoothed_lines(
                raw_crosslink_centers,
                s12_mesh,
                s13_mesh,
                cluster_radius=CROSSLINK_ADJUST_CLUSTER_RADIUS,
            )
        else:
            crosslink_centers = raw_crosslink_centers
    else:
        s12_mesh = extract_line_domain(grid, "S12_scalar", EPSILON_12)
        s13_mesh = extract_line_domain(grid, "S13_scalar", EPSILON_13)
        crosslink_candidate_centers = np.empty((0, 3), dtype=float)
        raw_crosslink_centers = np.empty((0, 3), dtype=float)
        crosslink_centers = np.empty((0, 3), dtype=float)
        crosslink_line_mesh = pv.PolyData()

    display_crosslink_site_mesh = pv.PolyData()
    if ORDER_CROSSLINK_BALLS:
        if USE_VORTEX_TRACING and CROSSLINK_TRACE_LINES:
            display_crosslink_site_mesh = sample_crosslink_line_site_mesh(
                crosslink_line_mesh,
                spacing=CROSSLINK_ORDER_SPACING,
            )
            if CROSSLINK_ADJUST_TO_SMOOTHED_LINES:
                display_crosslink_site_mesh = adjust_crosslink_site_mesh_to_smoothed_lines(
                    display_crosslink_site_mesh,
                    s12_mesh,
                    s13_mesh,
                )
            display_crosslink_site_mesh = break_crosslink_site_mesh_by_spacing(
                display_crosslink_site_mesh,
                max_segment_length=CROSSLINK_TUBE_CONNECT_RADIUS,
            )
            display_crosslink_site_mesh = bridge_crosslink_site_mesh_endpoints(
                display_crosslink_site_mesh,
                connect_radius=CROSSLINK_TUBE_CONNECT_RADIUS,
            )
            display_crosslink_centers = (
                np.asarray(display_crosslink_site_mesh.points)
                if display_crosslink_site_mesh.n_points
                else np.empty((0, 3), dtype=float)
            )
        else:
            display_crosslink_centers = order_crosslink_centers_with_spacing(
                crosslink_centers,
                link_radius=CROSSLINK_ORDER_LINK_RADIUS,
                spacing=CROSSLINK_ORDER_SPACING,
                min_nodes=CROSSLINK_ORDER_MIN_NODES,
            )
            display_crosslink_centers = enforce_crosslink_site_spacing(
                display_crosslink_centers,
                min_spacing=CROSSLINK_ORDER_SPACING,
            )
    else:
        display_crosslink_centers = crosslink_centers

    plotter = pv.Plotter(window_size=WINDOW_SIZE)
    plotter.set_background(BACKGROUND_COLOR)

    if USE_VORTEX_TRACING:
        s12_display = s12_mesh.tube(radius=VORTEX_TUBE_RADIUS) if s12_mesh.n_points else s12_mesh
        s13_display = s13_mesh.tube(radius=VORTEX_TUBE_RADIUS) if s13_mesh.n_points else s13_mesh
        add_line_domain_to_plotter(
            plotter,
            s12_display,
            color=S12_TUBE_COLOR,
            opacity=S12_TUBE_OPACITY,
            name="S12",
            show_edges=SHOW_TUBE_EDGES,
            edge_color=TUBE_EDGE_COLOR,
        )
        add_line_domain_to_plotter(
            plotter,
            s13_display,
            color=S13_TUBE_COLOR,
            opacity=S13_TUBE_OPACITY,
            name="S13",
            show_edges=SHOW_TUBE_EDGES,
            edge_color=TUBE_EDGE_COLOR,
        )
    else:
        add_line_domain_to_plotter(
            plotter,
            s12_mesh,
            color=S12_TUBE_COLOR,
            opacity=LINE_DOMAIN_OPACITY,
            name="S12",
            show_edges=SHOW_TUBE_EDGES,
            edge_color=TUBE_EDGE_COLOR,
        )
        add_line_domain_to_plotter(
            plotter,
            s13_mesh,
            color=S13_TUBE_COLOR,
            opacity=LINE_DOMAIN_OPACITY,
            name="S13",
            show_edges=SHOW_TUBE_EDGES,
            edge_color=TUBE_EDGE_COLOR,
        )

    if SHOW_PHI1_ZERO_SURFACE:
        add_isosurface_to_plotter(
            plotter,
            grid,
            "phi1",
            color=PHI1_SURFACE_COLOR,
            opacity=PHI1_SURFACE_OPACITY,
            show_edges=SHOW_PHI_SURFACE_EDGES,
            edge_color=PHI_SURFACE_EDGE_COLOR,
        )
    if SHOW_PHI2_ZERO_SURFACE:
        add_isosurface_to_plotter(
            plotter,
            grid,
            "phi2",
            color=PHI2_SURFACE_COLOR,
            opacity=PHI2_SURFACE_OPACITY,
            show_edges=SHOW_PHI_SURFACE_EDGES,
            edge_color=PHI_SURFACE_EDGE_COLOR,
        )
    if SHOW_PHI3_ZERO_SURFACE:
        add_isosurface_to_plotter(
            plotter,
            grid,
            "phi3",
            color=PHI3_SURFACE_COLOR,
            opacity=PHI3_SURFACE_OPACITY,
            show_edges=SHOW_PHI_SURFACE_EDGES,
            edge_color=PHI_SURFACE_EDGE_COLOR,
        )

    if SHOW_CROSSLINK_NODES and len(display_crosslink_centers) > 0:
        if CROSSLINK_RENDER_CONSECUTIVE_AS_TUBES:
            if USE_VORTEX_TRACING and CROSSLINK_TRACE_LINES and display_crosslink_site_mesh.n_lines:
                crosslink_tubes = display_crosslink_site_mesh.tube(
                    radius=float(CROSSLINK_BALL_RADIUS),
                    capping=True,
                )
                crosslink_caps = make_crosslink_node_mesh(
                    _line_endpoint_centers(display_crosslink_site_mesh),
                    radius=CROSSLINK_BALL_RADIUS,
                )
            else:
                crosslink_tubes, crosslink_caps = make_capped_crosslink_tube_mesh(
                    display_crosslink_centers,
                    radius=CROSSLINK_BALL_RADIUS,
                    connect_radius=CROSSLINK_TUBE_CONNECT_RADIUS,
                )
            if crosslink_tubes.n_points:
                plotter.add_mesh(
                    crosslink_tubes,
                    color=CROSSLINK_COLOR,
                    opacity=1.0,
                    smooth_shading=True,
                    show_edges=SHOW_CROSSLINK_EDGES,
                    edge_color=CROSSLINK_EDGE_COLOR,
                    specular=0.35,
                    name="crosslink_tubes",
                )
            if crosslink_caps.n_points:
                plotter.add_mesh(
                    crosslink_caps,
                    color=CROSSLINK_COLOR,
                    opacity=1.0,
                    smooth_shading=True,
                    show_edges=SHOW_CROSSLINK_EDGES,
                    edge_color=CROSSLINK_EDGE_COLOR,
                    specular=0.35,
                    name="crosslink_caps",
                )
        else:
            crosslink_mesh = make_crosslink_node_mesh(
                display_crosslink_centers,
                radius=CROSSLINK_BALL_RADIUS,
            )
            plotter.add_mesh(
                crosslink_mesh,
                color=CROSSLINK_COLOR,
                opacity=1.0,
                smooth_shading=True,
                show_edges=SHOW_CROSSLINK_EDGES,
                edge_color=CROSSLINK_EDGE_COLOR,
                specular=0.35,
                name="crosslink_nodes",
            )

    if SHOW_BOUNDING_BOX:
        add_fixed_box_outline(plotter)
    plotter.add_axes(line_width=2, labels_off=False)
    plotter.reset_camera()
    set_camera_from_angles(plotter, GRID_SIZE)
    plotter.camera.zoom(CAMERA_ZOOM)
    plotter.camera.reset_clipping_range()
    add_camera_and_top_lights(plotter, GRID_SIZE)

    meshes = {
        "S12": s12_mesh,
        "S13": s13_mesh,
        "crosslink_centers": crosslink_centers,
        "display_crosslink_centers": display_crosslink_centers,
        "display_crosslink_site_mesh": display_crosslink_site_mesh,
        "crosslink_candidate_centers": crosslink_candidate_centers,
        "crosslink_lines": crosslink_line_mesh,
        "raw_crosslink_centers": raw_crosslink_centers,
    }
    return plotter, meshes


def main() -> None:
    plotter, meshes = build_scene()

    if SAVE_MESHES:
        s12_path = save_mesh(meshes["S12"], MESH_OUTPUT_DIR, S12_MESH_NAME)
        s13_path = save_mesh(meshes["S13"], MESH_OUTPUT_DIR, S13_MESH_NAME)
        print(f"Saved S_12 mesh: {s12_path}")
        print(f"Saved S_13 mesh: {s13_path}")

    if SAVE_SCREENSHOT:
        plotter.show(screenshot=SCREENSHOT_PATH)
        print(f"Saved screenshot: {SCREENSHOT_PATH}")
    else:
        plotter.show()

    print(
        "\nMorphology tuning note:\n"
        "  Larger k values produce finer line-network features; smaller k values\n"
        "  produce coarser features. A single shell gives a narrow wavelength,\n"
        "  Gaussian radial sampling broadens the length-scale distribution, and a\n"
        "  uniform band limits features to a controlled k interval. More modes make\n"
        "  the random waves closer to Gaussian and usually smooth out directional\n"
        "  artifacts. Increasing epsilon thickens the displayed tube domains around\n"
        "  Gamma_12 and Gamma_13. Shared k-sets correlate morphology across fields,\n"
        "  while independent k-sets decorrelate them. Additional correlation between\n"
        "  phi_2 and phi_3 can be introduced by partially sharing amplitudes/phases\n"
        "  or by mixing fields, which changes how often the two retained line\n"
        "  families approach the same triple-zero crosslink nodes."
    )


if __name__ == "__main__":
    main()
