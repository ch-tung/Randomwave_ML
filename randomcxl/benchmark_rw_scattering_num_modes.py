"""
Benchmark random-wave mode count using notebook parameters.

This script reads the main parameter cell from
``interactive_rw_line_scattering.ipynb`` and compares scattering curves for
different random-wave mode counts, by default ``NUM_MODES = 16, 64, 256``.

All other random-wave, tracing, scattering-window, Q-grid, normalization, and
seed-average settings are inherited from the notebook unless overridden by the
command line. Each mode-count value is applied as ``(M, M, M)`` for the three
fields.
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
DEFAULT_NUM_MODES = (16, 64, 256)


@dataclass(frozen=True)
class ModeCase:
    """One random-wave mode-count benchmark case."""

    label: str
    num_modes: tuple[int, int, int]


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


def structure_seed_values(limit: int | None = None) -> list[int]:
    """Return notebook seed list, optionally truncated for quick tests."""

    seeds = [int(seed) for seed in s.structure_seed_values()]
    if limit is not None:
        seeds = seeds[: int(limit)]
    if not seeds:
        raise ValueError("at least one seed is required.")
    return seeds


def mode_cases(values: tuple[int, ...]) -> list[ModeCase]:
    """Build the ordered benchmark cases."""

    cases: list[ModeCase] = []
    for value in values:
        m = int(value)
        if m <= 0:
            raise ValueError("mode counts must be positive.")
        cases.append(ModeCase(f"modes_{m}", (m, m, m)))
    return cases


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


def build_case_structures(case: ModeCase, seeds: list[int]) -> list[s.LineStructure]:
    """Build all seed structures for one mode-count case."""

    original_seed = r.RANDOM_SEED
    structures: list[s.LineStructure] = []
    try:
        for seed_index, seed in enumerate(seeds, start=1):
            print(
                f"Building {case.label}, seed {seed_index}/{len(seeds)} ({seed}); "
                f"NUM_MODES={case.num_modes}"
            )
            t0 = time.perf_counter()
            r.RANDOM_SEED = int(seed)
            structure = s.build_line_structure(
                print_timing=s.PRINT_TIMING,
                timing_label=f"{case.label} seed {seed_index}/{len(seeds)} ({seed})",
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


def compute_case_curve(case: ModeCase, structures: list[s.LineStructure], q: np.ndarray) -> dict[str, Any]:
    """Compute the seed-averaged scattering curve for one case."""

    print(f"Computing scattering for {case.label}")
    t0 = time.perf_counter()
    curves = [s.structure_scattering_intensity_1d(structure, q) for structure in structures]
    curve_array = np.vstack(curves)
    mean = np.mean(curve_array, axis=0)
    std = np.std(curve_array, axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
    elapsed = time.perf_counter() - t0
    print(f"  scattering time={elapsed:.3f}s")
    return {
        "case": case,
        "curves": curve_array,
        "mean": mean,
        "std": std,
        "diagnostics": s.current_window_diagnostics(structures[0]),
        "elapsed_s": elapsed,
        "point_counts": [len(structure.points) for structure in structures],
        "segment_counts": [len(structure.segment_starts) for structure in structures],
        "crosslink_counts": [len(structure.crosslink_points) for structure in structures],
        "total_line_lengths": [s.total_line_length(structure, weighted=False) for structure in structures],
    }


def run_mode_cases(cases: list[ModeCase], seeds: list[int], q: np.ndarray) -> dict[str, dict[str, Any]]:
    """Build and scatter each mode-count case."""

    original_num_modes = r.NUM_MODES
    results: dict[str, dict[str, Any]] = {}
    try:
        for case in cases:
            r.NUM_MODES = case.num_modes
            t0 = time.perf_counter()
            structures = build_case_structures(case, seeds)
            results[case.label] = compute_case_curve(case, structures, q)
            results[case.label]["case_elapsed_s"] = time.perf_counter() - t0
            print(f"{case.label} total time={results[case.label]['case_elapsed_s']:.3f}s")
    finally:
        r.NUM_MODES = original_num_modes
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


def plot_curves(path: Path, q_scaled: np.ndarray, q_axis_label: str, results: dict[str, dict[str, Any]]) -> None:
    """Plot the mode-count benchmark curves."""

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.0), constrained_layout=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for index, (label, result) in enumerate(results.items()):
        color = colors[index % len(colors)]
        case = result["case"]
        mean = result["mean"]
        std = result["std"]
        curve_label = f"NUM_MODES={case.num_modes[0]}"
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
    cases: list[ModeCase],
    q_metadata: dict[str, float],
    notebook_num_modes: int | tuple[int, int, int],
    results: dict[str, dict[str, Any]],
) -> None:
    """Save notebook parameters and benchmark diagnostics."""

    lines = [
        "RandomCXL random-wave mode-count benchmark",
        "",
        "Notebook parameter cell:",
        parameter_source.rstrip(),
        "",
        "Benchmark controls:",
        f"NUM_MODES values = {[case.num_modes for case in cases]}",
        f"notebook NUM_MODES = {notebook_num_modes}",
        f"SCATTERING_WINDOW = {s.SCATTERING_WINDOW}",
        f"SCATTERING_WINDOW_TAPER = {s.SCATTERING_WINDOW_TAPER}",
        "",
        "Diagnostics:",
        f"seeds = {seeds}",
        f"Q_AXIS_SCALE = {s.Q_AXIS_SCALE}",
        f"k_mean = {q_metadata['k_mean']}",
        f"k_wave_angular_mean = {q_metadata['k_wave_angular_mean']}",
        f"GRID_SIZE = {r.GRID_SIZE}",
        f"NUM_BLOCK = {r.NUM_BLOCK}",
        f"BOX_SIZE_L = {r.BOX_SIZE_L}",
        f"SCATTERING_AMPLITUDE_METHOD = {s.SCATTERING_AMPLITUDE_METHOD}",
        f"INTENSITY_NORMALIZATION = {s.INTENSITY_NORMALIZATION}",
        f"SUBTRACT_WINDOWED_MEAN = {s.SUBTRACT_WINDOWED_MEAN}",
        f"WINDOW_NORMALIZATION = {s.WINDOW_NORMALIZATION}",
        "",
        "Case diagnostics:",
    ]
    for label, result in results.items():
        case = result["case"]
        lines.extend(
            [
                f"- {label}:",
                f"  NUM_MODES = {case.num_modes}",
                f"  point_counts = {result['point_counts']}",
                f"  segment_counts = {result['segment_counts']}",
                f"  crosslink_counts = {result['crosslink_counts']}",
                f"  total_line_lengths = {result['total_line_lengths']}",
                f"  window_diagnostics_first_seed = {result['diagnostics']}",
                f"  scattering_elapsed_s = {result['elapsed_s']:.6g}",
                f"  case_elapsed_s = {result['case_elapsed_s']:.6g}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default=DEFAULT_NOTEBOOK, help="Notebook to read settings from.")
    parser.add_argument("--output-dir", default="rw_scattering_num_modes_benchmark", help="Folder for outputs.")
    parser.add_argument("--num-modes", nargs="+", type=int, default=list(DEFAULT_NUM_MODES), help="Mode counts.")
    parser.add_argument("--max-seeds", type=int, default=None, help="Optional seed limit for quick tests.")
    parser.add_argument("--plot-name", default="num_modes_benchmark.png", help="Saved comparison plot filename.")
    parser.add_argument("--data-name", default="num_modes_benchmark_curves.csv", help="Saved wide CSV filename.")
    parser.add_argument("--settings-name", default="num_modes_benchmark_settings.txt", help="Saved settings filename.")
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

    notebook_num_modes = r.NUM_MODES
    cases = mode_cases(tuple(int(value) for value in args.num_modes))
    seeds = structure_seed_values(args.max_seeds)
    q = s.q_values()
    q_scaled, q_axis_label, q_metadata = q_scaled_axis(q)

    print("NUM_MODES benchmark values:", [case.num_modes for case in cases])
    print("notebook NUM_MODES:", notebook_num_modes)
    print("SCATTERING_WINDOW:", s.SCATTERING_WINDOW)
    print("SCATTERING_WINDOW_TAPER:", s.SCATTERING_WINDOW_TAPER)
    print("seeds:", seeds)

    t_all = time.perf_counter()
    results = run_mode_cases(cases, seeds, q)

    plot_path = output_dir / args.plot_name
    plot_curves(plot_path, q_scaled, q_axis_label, results)
    print(f"Saved mode-count benchmark plot: {plot_path}")

    data_path = output_dir / args.data_name
    save_curves(data_path, q, q_scaled, results)
    print(f"Saved mode-count benchmark data: {data_path}")

    settings_path = output_dir / args.settings_name
    save_settings(settings_path, parameter_source, seeds, cases, q_metadata, notebook_num_modes, results)
    print(f"Saved mode-count benchmark settings: {settings_path}")
    print(f"Benchmark total time: {time.perf_counter() - t_all:.3f}s")


if __name__ == "__main__":
    main()
