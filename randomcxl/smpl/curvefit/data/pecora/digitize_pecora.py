"""Digitize Figure 1 from Borsali, Nguyen, and Pecora (1998).

The script detects the five printed marker shapes by normalized template
matching.  Axis calibration is taken from the plot frame and major ticks.
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage, signal


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "pecora_1998_figure.png"

# Pixel coordinates of the plot frame, determined from its one-pixel edges.
X_LEFT, X_RIGHT = 174.0, 1011.0
Y_TOP, Y_BOTTOM = 125.0, 684.0
Q_MIN, Q_MAX = 0.0, 0.22
I_MIN, I_MAX = 0.0, 0.50

# Clean examples span the sub-pixel rasterization variants found in the scan.
TEMPLATE_CENTERS = {
    "salt_free": [(312, 173), (407, 443), (417, 445), (427, 449)],
    "10mM": [(311, 199), (407, 473), (417, 474), (427, 478), (437, 486)],
    "50mM": [(311, 226), (406, 505), (417, 505), (427, 507), (437, 512)],
    "0.5M": [(312, 253), (215, 204), (223, 260), (239, 323), (255, 367)],
    "1M": [(312, 280), (229, 143), (231, 164), (234, 176), (237, 199), (248, 253)],
}

COLORS = {
    "salt_free": "#d62728",
    "10mM": "#1f77b4",
    "50mM": "#2ca02c",
    "0.5M": "#ff7f0e",
    "1M": "#9467bd",
}

# Shared detector-bin positions inferred from the repeated horizontal spacing.
X_GRID = np.array(
    [
        202, 205, 207, 210, 213, 215, 219, 223, 228, 233, 239, 245, 250,
        255, 260, 266, 271, 278, 287, 297, 307, 317, 327, 337, 347, 357,
        367, 377, 387, 397, 407, 417, 427, 437, 447, 457, 467, 477, 487,
        497, 507, 517, 526, 536, 546, 555, 565, 575, 585, 595, 604, 614,
        624, 634, 644, 654, 664, 674, 683, 693, 703, 713, 723, 732, 742,
        751, 761, 771, 781, 790, 800, 810, 819, 829, 839, 848, 858, 868,
        877, 887, 896, 906, 916, 925, 935, 945, 954, 964, 973, 983, 992,
    ],
    dtype=int,
)

# Hand-checked centerline anchors constrain matching when symbols overlap.
PATH_ANCHORS = {
    "salt_free": [
        (202, 252), (205, 361), (207, 407), (210, 447), (213, 484),
        (215, 541), (220, 575), (230, 596), (240, 607), (260, 610),
        (280, 610), (290, 596), (300, 574), (320, 542), (340, 526),
        (360, 500), (380, 456), (400, 443), (420, 445), (440, 460),
        (460, 490), (480, 522), (500, 545), (540, 582), (580, 607),
        (620, 624), (680, 636), (740, 648), (800, 658), (860, 664),
        (920, 668), (992, 666),
    ],
    "10mM": [
        (202, 315), (205, 397), (208, 435), (210, 470), (213, 502),
        (215, 542), (220, 564), (230, 575), (250, 579), (270, 575),
        (290, 572), (310, 550), (330, 526), (350, 511), (370, 489),
        (390, 475), (410, 474), (430, 479), (450, 494), (470, 518),
        (490, 543), (520, 568), (560, 594), (600, 614), (650, 630),
        (700, 641), (750, 649), (800, 655), (850, 658), (900, 658),
        (950, 651), (992, 658),
    ],
    "50mM": [
        (205, 165), (207, 242), (210, 297), (213, 337), (215, 388),
        (220, 414), (230, 438), (250, 457), (270, 477), (290, 490),
        (320, 498), (350, 502), (380, 504), (410, 505), (440, 513),
        (470, 533), (500, 556), (540, 584), (580, 607), (620, 624),
        (680, 642), (740, 653), (800, 666), (860, 671), (920, 678),
        (992, 675),
    ],
    "0.5M": [
        (213, 153), (215, 204), (219, 243), (223, 260), (228, 282),
        (233, 303), (239, 323), (245, 345), (255, 367), (270, 400),
        (290, 432), (310, 455), (330, 478), (350, 497), (370, 511),
        (390, 524), (410, 538), (430, 550), (450, 562), (470, 574),
        (500, 586), (540, 600), (580, 609), (620, 616), (680, 628),
        (740, 637), (800, 644), (860, 648), (920, 653), (960, 644),
        (992, 645),
    ],
    "1M": [
        (229, 143), (231, 164), (234, 176), (237, 199), (245, 232),
        (248, 253), (255, 280), (260, 303), (270, 344), (280, 380),
        (290, 409), (310, 443), (330, 469), (350, 490), (370, 511),
        (390, 532), (410, 548), (430, 560), (450, 575), (470, 588),
        (500, 600), (540, 610), (580, 617), (620, 625), (680, 640),
        (740, 650), (800, 661), (860, 668), (920, 675), (992, 679),
    ],
}


def normalized_match(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Return a same-size normalized cross-correlation map."""
    template = template - template.mean()
    n = template.size
    numerator = signal.correlate2d(image, template, mode="same")
    local_sum = ndimage.uniform_filter(image, template.shape, mode="constant") * n
    local_sum_sq = (
        ndimage.uniform_filter(image * image, template.shape, mode="constant") * n
    )
    local_variance_sum = np.maximum(local_sum_sq - local_sum * local_sum / n, 1e-12)
    return numerator / np.sqrt(local_variance_sum * np.sum(template * template))


def pixel_to_data(x: float, y: float) -> tuple[float, float]:
    q = Q_MIN + (x - X_LEFT) * (Q_MAX - Q_MIN) / (X_RIGHT - X_LEFT)
    intensity = I_MIN + (Y_BOTTOM - y) * (I_MAX - I_MIN) / (Y_BOTTOM - Y_TOP)
    return q, intensity


def follow_marker_path(name: str, score: np.ndarray):
    anchors = np.asarray(PATH_ANCHORS[name], dtype=float)
    min_x, max_x = anchors[0, 0], anchors[-1, 0]
    x_grid = X_GRID[(X_GRID >= min_x - 2) & (X_GRID <= max_x + 2)]
    prior_y = np.interp(x_grid, anchors[:, 0], anchors[:, 1])
    points = []
    used = set()
    for nominal_x, expected_y in zip(x_grid, prior_y):
        x0, x1 = max(190, nominal_x - 1), min(1002, nominal_x + 1)
        y0 = max(135, int(round(expected_y)) - 7)
        y1 = min(675, int(round(expected_y)) + 7)
        patch = score[y0 : y1 + 1, x0 : x1 + 1].copy()
        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        objective = patch - 0.018 * np.abs(yy - expected_y) - 0.012 * np.abs(
            xx - nominal_x
        )
        index = np.unravel_index(np.argmax(objective), objective.shape)
        y = y0 + index[0]
        x = x0 + index[1]
        match = float(score[y, x])
        if match < 0.32 or (x, y) in used:
            continue
        used.add((x, y))
        q, intensity = pixel_to_data(x, y)
        points.append(
            {
                "series": name,
                "q_A^-1": q,
                "I_q": intensity,
                "pixel_x": x,
                "pixel_y": y,
                "match_score": match,
            }
        )
    return points


def write_csvs(points) -> None:
    fields = ["series", "q_A^-1", "I_q", "pixel_x", "pixel_y", "match_score"]
    with (HERE / "pecora_1998_digitized.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in points:
            row = point.copy()
            row["q_A^-1"] = f"{row['q_A^-1']:.7f}"
            row["I_q"] = f"{row['I_q']:.6f}"
            row["match_score"] = f"{row['match_score']:.4f}"
            writer.writerow(row)

    for name in TEMPLATE_CENTERS:
        subset = [point for point in points if point["series"] == name]
        with (HERE / f"pecora_1998_{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["q_A^-1", "I_q"])
            for point in subset:
                writer.writerow([f"{point['q_A^-1']:.7f}", f"{point['I_q']:.6f}"])


def main() -> None:
    gray = np.asarray(Image.open(SOURCE).convert("L"), dtype=float) / 255.0
    ink = 1.0 - gray

    # A 13 x 13 window includes each glyph plus a narrow white margin.
    radius = 6
    score_maps = {}
    for name, centers in TEMPLATE_CENTERS.items():
        matches = []
        for cx, cy in centers:
            template = ink[
                cy - radius : cy + radius + 1, cx - radius : cx + radius + 1
            ]
            matches.append(normalized_match(ink, template))
        score_maps[name] = np.max(matches, axis=0)

    all_points = []
    for name, score in score_maps.items():
        all_points.extend(follow_marker_path(name, score))
    write_csvs(all_points)

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
    for name in TEMPLATE_CENTERS:
        subset = [point for point in all_points if point["series"] == name]
        ax.scatter(
            [point["pixel_x"] for point in subset],
            [point["pixel_y"] for point in subset],
            s=44,
            facecolors="none",
            edgecolors=COLORS[name],
            linewidths=0.9,
            label=f"{name} ({len(subset)})",
        )
    ax.set_xlim(185, 1005)
    ax.set_ylim(676, 135)
    ax.legend(loc="upper right")
    ax.set_title("Selected digitized marker centers")
    fig.tight_layout()
    fig.savefig(HERE / "pecora_1998_digitization_overlay.png", dpi=180)
    plt.close(fig)

    marker_styles = {
        "salt_free": "o",
        "10mM": "o",
        "50mM": "s",
        "0.5M": "s",
        "1M": "^",
    }
    filled = {"salt_free", "50mM", "1M"}
    fig, ax = plt.subplots(figsize=(9, 6))
    for name in TEMPLATE_CENTERS:
        subset = [point for point in all_points if point["series"] == name]
        ax.plot(
            [point["q_A^-1"] for point in subset],
            [point["I_q"] for point in subset],
            linestyle="none",
            marker=marker_styles[name],
            markersize=4,
            markerfacecolor="black" if name in filled else "white",
            markeredgecolor="black",
            label=name,
        )
    ax.set(xlabel=r"$q$ ($\AA^{-1}$)", ylabel=r"$I(q)$", xlim=(0, 0.22), ylim=(0, 0.5))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "pecora_1998_digitized_plot.png", dpi=180)
    plt.close(fig)
    print(f"wrote {len(all_points)} digitized points")


if __name__ == "__main__":
    main()
