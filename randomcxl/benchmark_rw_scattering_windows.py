"""
Benchmark smooth scattering windows using notebook parameters.

This script reads the main parameter cell from
``interactive_rw_line_scattering.ipynb``, builds the same line-network
structures used by the notebook, then compares scattering curves for:

* no smooth window;
* smooth window with taper fractions 0.05, 0.10, and 0.20.

All random-wave, tracing, Q-grid, normalization, seed-average, and plotting
scale parameters are inherited from the notebook unless overridden by the
command line.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import rw_line_network as r
import rw_line_scattering as s


DEFAULT_NOTEBOOK = "interactive_rw_line_scattering.ipynb"
PARAMETER_MARKER = "# Random-wave sampling settings"
DEFAULT_TAPERS = (0.05, 0.10, 0.20)


@dataclass(frozen=True)
class WindowCase:
    """One scattering-window benchmark case."""

    label: str
    window_type: str
    taper: float


def load_notebook(path: Path) -> dict[str, Any]:
    """Load a notebook JSON document."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_code_cell_source(notebook: dict[str, Any], marker: str) -> str:
    """Return the first code-cell source containing ``marker``."""

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if marker in source:
            return source
    raise RuntimeError(f"Could not find notebook code cell containing: {marker!r}")


def apply_notebook_parameters(parameter_source: str) -> None:
    """Execute the notebook parameter cell against the imported modules."""

    global r, s
    r = importlib.reload(r)
    s = importlib.reload(s)
    namespace = {
        "np": np,
        "r": r,
        "s": s,
    }
    exec(parameter_source, namespace)


def resolve_window_type(requested: str) -> str:
    """Resolve the smooth-window type used for nonzero taper cases."""

    requested = str(requested).lower()
    if requested != "auto":
        return requested
    notebook_window = s.scattering_window_type()
    if notebook_window == "none":
        return "tukey_box"
    return notebook_window


def window_cases(window_type: str, tapers: tuple[float, ...]) -> list[WindowCase]:
    """Build the ordered benchmark cases."""

    cases = [WindowCase("none", "none", 0.0)]
    for taper in tapers:
        compact = f"{taper:g}".replace(".", "p")
        cases.append(WindowCase(f"{window_type}_{compact}", window_type, float(taper)))
    return cases


def structure_seed_values(limit: int | None = None) -> list[int]:
    """Return notebook seed list, optionally truncated for quick tests."""

    seeds = [int(seed) for seed in s.structure_seed_values()]
    if limit is not None:
        seeds = seeds[: int(limit)]
    if not seeds:
        raise ValueError("at least one seed is required.")
    return seeds


def build_structures(seeds: list[int]) -> list[s.LineStructure]:
    """Build each seed structure once, independent of scattering window."""

    original_seed = r.RANDOM_SEED
    structures: list[s.LineStructure] = []
    try:
        for seed_index, seed in enumerate(seeds, start=1):
            print(f"Building structure {seed_index}/{len(seeds)} for seed {seed}...")
            t0 = time.perf_counter()
            r.RANDOM_SEED = int(seed)
            structure = s.build_line_structure(
                print_timing=s.PRINT_TIMING,
                timing_label=f"seed {seed_index}/{len(seeds)} ({seed})",
            )
            structures.append(structure)
            print(
                f"  points={len(structure.points)}; "
                f"segments={len(structure.segment_starts)}; "
                f"crosslinks={len(structure.crosslink_points)}; "
                f"time={time.perf_counter() - t0:.3f}s"
            )
    finally:
        r.RANDOM_SEED = original_seed
    return structures


def q_scaled_axis(q: np.ndarray) -> tuple[np.ndarray, str, dict[str, float]]:
    """Return the Q axis used by the notebook plot."""

    k_mean = float(np.mean(r._field_parameter_values(r.K0, "K0")))
    block_grid_side = float(r.GRID_SIZE - 1)
    q_block_angular = q * block_grid_side
    k_wave_angular_mean = 2.0 * np.pi * k_mean
    q_scaled_mean_k = q_block_angular / k_wave_angular_mean

    if s.Q_AXIS_SCALE == "mean_k":
        q_scaled = q_scaled_mean_k
        q_axis_label = r"$Q_{block}/(2\pi\langle k\rangle)$"
    elif s.Q_AXIS_SCALE == "raw":
        q_scaled = q
        q_axis_label = "Q"
    else:
        raise ValueError("s.Q_AXIS_SCALE must be 'mean_k' or 'raw'.")

    metadata = {
        "k_mean": k_mean,
        "block_grid_side": block_grid_side,
        "k_wave_angular_mean": k_wave_angular_mean,
    }
    return q_scaled, q_axis_label, metadata


def compute_case_curves(
    structures: list[s.LineStructure],
    cases: list[WindowCase],
    q: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Compute seed-averaged scattering for each window case."""

    original_window = s.SCATTERING_WINDOW
    original_taper = s.SCATTERING_WINDOW_TAPER
    results: dict[str, dict[str, Any]] = {}
    try:
        for case in cases:
            s.SCATTERING_WINDOW = case.window_type
            s.SCATTERING_WINDOW_TAPER = case.taper
            print(
                f"Computing {case.label}: "
                f"SCATTERING_WINDOW={case.window_type}, "
                f"SCATTERING_WINDOW_TAPER={case.taper}"
            )
            t0 = time.perf_counter()
            curves = [
                s.structure_scattering_intensity_1d(structure, q)
                for structure in structures
            ]
            curve_array = np.vstack(curves)
            mean = np.mean(curve_array, axis=0)
            std = (
                np.std(curve_array, axis=0, ddof=1)
                if len(curves) > 1
                else np.zeros_like(mean)
            )
            diagnostics = s.current_window_diagnostics(structures[0])
            results[case.label] = {
                "case": case,
                "curves": curve_array,
                "mean": mean,
                "std": std,
                "diagnostics": diagnostics,
                "elapsed_s": time.perf_counter() - t0,
            }
            print(f"  scattering time={results[case.label]['elapsed_s']:.3f}s")
    finally:
        s.SCATTERING_WINDOW = original_window
        s.SCATTERING_WINDOW_TAPER = original_taper
    return results


def save_curves(path: Path, q: np.ndarray, q_scaled: np.ndarray, results: dict[str, dict[str, Any]]) -> None:
    """Save all benchmark curves to a wide CSV table."""

    columns = [q, q_scaled]
    header = ["q", "q_scaled"]
    for label, result in results.items():
        columns.extend([result["mean"], result["std"]])
        header.extend([f"{label}_mean", f"{label}_std"])
    data = np.column_stack(columns)
    np.savetxt(path, data, delimiter=",", header=",".join(header), comments="")


def plot_curves(
    path: Path,
    q_scaled: np.ndarray,
    q_axis_label: str,
    results: dict[str, dict[str, Any]],
) -> None:
    """Plot the benchmark scattering curves."""

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.0), constrained_layout=True)
    colors = {
        "none": "black",
    }
    fallback_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for index, (label, result) in enumerate(results.items()):
        color = colors.get(label, fallback_colors[(index - 1) % len(fallback_colors)])
        mean = result["mean"]
        std = result["std"]
        case = result["case"]
        if case.window_type == "none":
            curve_label = "no window"
        else:
            curve_label = f"{case.window_type}, taper={case.taper:g}"
        ax.plot(q_scaled, mean, lw=1.5, color=color, label=curve_label)
        if len(result["curves"]) > 1:
            ax.fill_between(q_scaled, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)

    ax.set_xlabel(q_axis_label)
    ax.set_ylabel("I(Q)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_settings(
    path: Path,
    parameter_source: str,
    seeds: list[int],
    cases: list[WindowCase],
    q_metadata: dict[str, float],
    structures: list[s.LineStructure],
    results: dict[str, dict[str, Any]],
) -> None:
    """Save notebook parameters and benchmark diagnostics."""

    lines = [
        "RandomCXL scattering-window benchmark",
        "",
        "Notebook parameter cell:",
        parameter_source.rstrip(),
        "",
        "Benchmark cases:",
    ]
    for case in cases:
        lines.append(
            f"- {case.label}: SCATTERING_WINDOW={case.window_type}, "
            f"SCATTERING_WINDOW_TAPER={case.taper}"
        )
    lines.extend(
        [
            "",
            "Diagnostics:",
            f"seeds = {seeds}",
            f"point_counts = {[len(structure.points) for structure in structures]}",
            f"segment_counts = {[len(structure.segment_starts) for structure in structures]}",
            f"crosslink_counts = {[len(structure.crosslink_points) for structure in structures]}",
            f"Q_AXIS_SCALE = {s.Q_AXIS_SCALE}",
            f"k_mean = {q_metadata['k_mean']}",
            f"k_wave_angular_mean = {q_metadata['k_wave_angular_mean']}",
            f"SCATTERING_AMPLITUDE_METHOD = {s.SCATTERING_AMPLITUDE_METHOD}",
            f"INTENSITY_NORMALIZATION = {s.INTENSITY_NORMALIZATION}",
            f"SUBTRACT_WINDOWED_MEAN = {s.SUBTRACT_WINDOWED_MEAN}",
            f"WINDOW_NORMALIZATION = {s.WINDOW_NORMALIZATION}",
            "",
            "Window diagnostics from first seed:",
        ]
    )
    for label, result in results.items():
        lines.append(f"- {label}: {result['diagnostics']}; elapsed_s={result['elapsed_s']:.6g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default=DEFAULT_NOTEBOOK, help="Notebook to read settings from.")
    parser.add_argument("--output-dir", default="rw_scattering_window_benchmark", help="Folder for outputs.")
    parser.add_argument(
        "--window-type",
        default="auto",
        choices=["auto", "tukey_box", "hann_box", "gaussian"],
        help="Smooth window type for taper cases. 'auto' uses the notebook window, or tukey_box if it is none.",
    )
    parser.add_argument(
        "--tapers",
        nargs="+",
        type=float,
        default=list(DEFAULT_TAPERS),
        help="Taper fractions/window widths to compare.",
    )
    parser.add_argument("--max-seeds", type=int, default=None, help="Optional seed limit for quick tests.")
    parser.add_argument("--plot-name", default="window_benchmark.png", help="Saved comparison plot filename.")
    parser.add_argument("--data-name", default="window_benchmark_curves.csv", help="Saved wide CSV filename.")
    parser.add_argument("--settings-name", default="window_benchmark_settings.txt", help="Saved settings filename.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    notebook_path = Path(args.notebook)
    if not notebook_path.is_absolute():
        notebook_path = script_dir / notebook_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    notebook = load_notebook(notebook_path)
    parameter_source = find_code_cell_source(notebook, PARAMETER_MARKER)
    apply_notebook_parameters(parameter_source)

    window_type = resolve_window_type(args.window_type)
    cases = window_cases(window_type, tuple(float(taper) for taper in args.tapers))
    seeds = structure_seed_values(args.max_seeds)
    q = s.q_values()
    q_scaled, q_axis_label, q_metadata = q_scaled_axis(q)

    print("benchmark window type:", window_type)
    print("benchmark tapers:", [case.taper for case in cases if case.window_type != "none"])
    print("seeds:", seeds)

    t_all = time.perf_counter()
    structures = build_structures(seeds)
    results = compute_case_curves(structures, cases, q)

    plot_path = output_dir / args.plot_name
    plot_curves(plot_path, q_scaled, q_axis_label, results)
    print(f"Saved benchmark plot: {plot_path}")

    data_path = output_dir / args.data_name
    save_curves(data_path, q, q_scaled, results)
    print(f"Saved benchmark data: {data_path}")

    settings_path = output_dir / args.settings_name
    save_settings(settings_path, parameter_source, seeds, cases, q_metadata, structures, results)
    print(f"Saved benchmark settings: {settings_path}")
    print(f"Benchmark total time: {time.perf_counter() - t_all:.3f}s")


if __name__ == "__main__":
    main()
