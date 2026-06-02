"""
Run the random-wave line scattering workflow from notebook settings.

This script reads the parameter cell from ``interactive_rw_line_scattering.ipynb``,
applies those settings to ``rw_line_network`` and ``rw_line_scattering``, computes
the scattering curve, and executes the notebook plotting cell so the saved figure
matches the notebook instructions.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import rw_line_network as r
import rw_line_scattering as s


DEFAULT_NOTEBOOK = "interactive_rw_line_scattering.ipynb"
PARAMETER_MARKER = "# Random-wave sampling settings"
PLOT_MARKER = "fig, ax = plt.subplots"


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
        "pv": pv,
        "plt": plt,
        "r": r,
        "s": s,
    }
    exec(parameter_source, namespace)


def build_structure() -> s.LineStructure:
    """Build the real-space line structure with current notebook settings."""

    structure = s.build_line_structure(
        line_sample_spacing=s.LINE_SAMPLE_SPACING,
        include_crosslinks=s.INCLUDE_CROSSLINK_POINTS,
    )
    print("sampled scattering points:", len(structure.points))
    print("continuous line segments:", len(structure.segment_starts))
    print("S12 line points/lines:", structure.s12_lines.n_points, structure.s12_lines.n_lines)
    print("S13 line points/lines:", structure.s13_lines.n_points, structure.s13_lines.n_lines)
    print("crosslink point scatterers:", len(structure.crosslink_points))
    return structure


def compute_scattering(structure: s.LineStructure) -> dict[str, Any]:
    """Compute the scattering curve and scaled Q axis used by the notebook."""

    seeds = s.structure_seed_values()
    q, iq, iq_std, iq_box, point_counts = s.compute_seed_averaged_scattering(seeds)
    k_mean = float(np.mean(r._field_parameter_values(r.K0, "K0")))
    block_grid_side = float(r.GRID_SIZE - 1)
    q_block_angular = q * block_grid_side
    k_wave_angular_mean = 2.0 * np.pi * k_mean
    total_line_length = s.total_line_length(structure, weighted=False)
    q_scaled_mean_k = q_block_angular / k_wave_angular_mean

    if s.Q_AXIS_SCALE == "mean_k":
        q_scaled = q_scaled_mean_k
        q_axis_label = r"$Q_{block}/(2\pi\langle k\rangle)$"
    elif s.Q_AXIS_SCALE == "raw":
        q_scaled = q
        q_axis_label = "Q"
    else:
        raise ValueError("s.Q_AXIS_SCALE must be 'mean_k' or 'raw'.")

    return {
        "seeds": seeds,
        "q": q,
        "iq": iq,
        "iq_std": iq_std,
        "iq_box": iq_box,
        "point_counts": point_counts,
        "k_mean": k_mean,
        "block_grid_side": block_grid_side,
        "q_block_angular": q_block_angular,
        "k_wave_angular_mean": k_wave_angular_mean,
        "total_line_length": total_line_length,
        "q_scaled_mean_k": q_scaled_mean_k,
        "q_scaled": q_scaled,
        "q_axis_label": q_axis_label,
    }


def execute_notebook_plot(plot_source: str, structure: s.LineStructure, values: dict[str, Any]) -> plt.Figure:
    """Execute the notebook plotting cell and return the generated figure."""

    namespace = {
        "np": np,
        "pv": pv,
        "plt": plt,
        "r": r,
        "s": s,
        "structure": structure,
    }
    namespace.update(values)
    plt.close("all")
    exec(plot_source, namespace)
    figure = plt.gcf()
    if figure is None:
        raise RuntimeError("Notebook plot cell did not create a matplotlib figure.")
    return figure


def save_settings(
    path: Path,
    parameter_source: str,
    values: dict[str, Any],
    structure: s.LineStructure,
) -> None:
    """Save the executed notebook parameter source and compact diagnostics."""

    diagnostics = s.current_window_diagnostics(structure)
    lines = [
        "RandomCXL scattering run from notebook settings",
        "",
        "Notebook parameter cell:",
        parameter_source.rstrip(),
        "",
        "Diagnostics:",
        f"seeds = {values['seeds']}",
        f"point_counts = {values['point_counts']}",
        f"k_mean = {values['k_mean']}",
        f"k_wave_angular_mean = {values['k_wave_angular_mean']}",
        f"total_line_length = {values['total_line_length']}",
        f"Q_AXIS_SCALE = {s.Q_AXIS_SCALE}",
        f"SCATTERING_WINDOW = {s.SCATTERING_WINDOW}",
        f"SUBTRACT_WINDOWED_MEAN = {s.SUBTRACT_WINDOWED_MEAN}",
        f"WINDOW_NORMALIZATION = {s.WINDOW_NORMALIZATION}",
        f"window_diagnostics = {diagnostics}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default=DEFAULT_NOTEBOOK, help="Notebook to read settings/plotting code from.")
    parser.add_argument("--output-dir", default="rw_line_scattering_run", help="Folder for outputs.")
    parser.add_argument("--plot-name", default="rw_line_scattering_plot.png", help="Saved scattering plot filename.")
    parser.add_argument("--data-name", default="rw_line_scattering_iq.csv", help="Saved scattering CSV filename.")
    parser.add_argument("--settings-name", default="settings_from_notebook.txt", help="Saved settings text filename.")
    parser.add_argument("--show-structure", action="store_true", help="Also save a PyVista real-space preview.")
    parser.add_argument("--structure-name", default="rw_line_scattering_structure.png", help="Saved structure screenshot.")
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
    plot_source = find_code_cell_source(notebook, PLOT_MARKER)

    apply_notebook_parameters(parameter_source)
    structure = build_structure()
    values = compute_scattering(structure)
    figure = execute_notebook_plot(plot_source, structure, values)

    plot_path = output_dir / args.plot_name
    figure.savefig(plot_path, dpi=220)
    print(f"Saved scattering plot: {plot_path}")

    data_path = output_dir / args.data_name
    s.save_scattering_curve(
        data_path,
        values["q"],
        values["iq"],
        std=values["iq_std"],
        box_intensity=values["iq_box"],
    )
    print(f"Saved scattering data: {data_path}")

    settings_path = output_dir / args.settings_name
    save_settings(settings_path, parameter_source, values, structure)
    print(f"Saved settings: {settings_path}")

    if args.show_structure:
        structure_path = output_dir / args.structure_name
        plotter = s.make_structure_plotter(structure)
        plotter.show(screenshot=str(structure_path))
        print(f"Saved structure screenshot: {structure_path}")


if __name__ == "__main__":
    main()
