"""Create the publication scatter for GT-mask depth/visibility trade-offs.

Reads ``mask_depth_tradeoff_results.csv`` and writes a 600 dpi PNG plus a
vector PDF. Existing research files are not modified.
"""

import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


INPUT_CSV = "mask_depth_tradeoff_results.csv"
OUTPUT_PNG = "figure2b_mask_depth_tradeoff_scatter.png"
OUTPUT_PDF = "figure2b_mask_depth_tradeoff_scatter.pdf"

DEMO_SEQUENCE = "402"
DEMO_FRAME = "1600938539283987"

EXPECTED_ROWS = 118
EXPECTED_MEDIAN_DEPTH = 11.5
EXPECTED_MEDIAN_VISIBILITY = 0.1834
EXPECTED_POSITIVE_VISIBILITY = 113


def load_data():
    with open(INPUT_CSV, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    required_columns = {
        "sequence_id",
        "frame_id",
        "delta_depth_mm",
        "delta_visibility",
    }
    if not rows:
        raise RuntimeError(f"No rows found in {INPUT_CSV}")
    missing = required_columns - set(rows[0])
    if missing:
        raise RuntimeError(f"Missing required CSV columns: {sorted(missing)}")

    delta_depth = np.asarray(
        [float(row["delta_depth_mm"]) for row in rows],
        dtype=float,
    )
    delta_visibility = np.asarray(
        [float(row["delta_visibility"]) for row in rows],
        dtype=float,
    )
    demo_indices = [
        index
        for index, row in enumerate(rows)
        if row["sequence_id"] == DEMO_SEQUENCE
        and row["frame_id"] == DEMO_FRAME
    ]
    return rows, delta_depth, delta_visibility, demo_indices


def validate(rows, delta_depth, delta_visibility, demo_indices):
    row_count = len(rows)
    median_depth = float(np.median(delta_depth))
    median_visibility = float(np.median(delta_visibility))
    positive_visibility = int(np.count_nonzero(delta_visibility > 0))

    checks = [
        ("loaded rows", row_count, EXPECTED_ROWS),
        ("median delta_depth", median_depth, EXPECTED_MEDIAN_DEPTH),
        (
            "delta_visibility > 0 count",
            positive_visibility,
            EXPECTED_POSITIVE_VISIBILITY,
        ),
        ("Demo frame count", len(demo_indices), 1),
    ]
    failed = False
    for _, actual, expected in checks:
        failed = failed or actual != expected
    if abs(median_visibility - EXPECTED_MEDIAN_VISIBILITY) > 5e-4:
        failed = True
    if len(demo_indices) == 1:
        demo_index = demo_indices[0]
        if abs(delta_depth[demo_index] - 13.0) > 1e-9:
            failed = True
        if abs(delta_visibility[demo_index] - 0.1573) > 5e-4:
            failed = True
    if failed:
        raise RuntimeError("Scatter input validation failed")

    return median_depth, median_visibility, positive_visibility


def make_figure(
    delta_depth,
    delta_visibility,
    demo_index,
    median_depth,
    median_visibility,
    positive_visibility,
):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axis = plt.subplots(figsize=(6.5, 4.8))

    axis.scatter(
        delta_depth,
        delta_visibility,
        s=34,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.45,
        alpha=0.82,
        zorder=3,
    )

    axis.axhline(
        0,
        color="#555555",
        linewidth=0.9,
        zorder=1,
    )
    axis.axvline(
        median_depth,
        color="#D07C35",
        linestyle="--",
        linewidth=1.1,
        zorder=2,
    )
    axis.axhline(
        median_visibility,
        color="#2A9D8F",
        linestyle="--",
        linewidth=1.1,
        zorder=2,
    )

    demo_depth = delta_depth[demo_index]
    demo_visibility = delta_visibility[demo_index]
    axis.scatter(
        [demo_depth],
        [demo_visibility],
        marker="D",
        s=78,
        color="#C73E3A",
        edgecolor="white",
        linewidth=0.9,
        zorder=5,
    )
    axis.annotate(
        "Demo",
        xy=(demo_depth, demo_visibility),
        xytext=(10, -18),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="#9E2F2C",
        arrowprops={
            "arrowstyle": "-",
            "color": "#9E2F2C",
            "linewidth": 0.8,
        },
        zorder=6,
    )

    depth_range = float(np.ptp(delta_depth))
    visibility_range = float(np.ptp(delta_visibility))
    x_padding = max(3.0, depth_range * 0.045)
    y_padding = max(0.025, visibility_range * 0.065)
    axis.set_xlim(max(0.0, float(delta_depth.min()) - x_padding), float(delta_depth.max()) + x_padding)
    axis.set_ylim(
        float(delta_visibility.min()) - y_padding,
        float(delta_visibility.max()) + y_padding,
    )

    axis.set_xlabel("ΔDepth (mm)")
    axis.set_ylabel("ΔVisibility")
    axis.grid(
        True,
        color="#D9D9D9",
        linewidth=0.55,
        alpha=0.55,
        zorder=0,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    annotation = (
        f"n = {len(delta_depth)}\n"
        f"Median ΔDepth = {median_depth:.1f} mm\n"
        f"Median ΔVisibility = {median_visibility:.3f}\n"
        f"ΔVisibility > 0: {positive_visibility}/{len(delta_depth)} "
        f"({positive_visibility / len(delta_depth) * 100:.1f}%)"
    )
    axis.text(
        0.975,
        0.045,
        annotation,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#B8B8B8",
            "linewidth": 0.7,
            "alpha": 0.94,
        },
        zorder=7,
    )

    figure.tight_layout(pad=0.7)
    figure.savefig(
        OUTPUT_PNG,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main():
    rows, delta_depth, delta_visibility, demo_indices = load_data()
    median_depth, median_visibility, positive_visibility = validate(
        rows,
        delta_depth,
        delta_visibility,
        demo_indices,
    )
    make_figure(
        delta_depth,
        delta_visibility,
        demo_indices[0],
        median_depth,
        median_visibility,
        positive_visibility,
    )

    print(f"loaded rows = {len(rows)}")
    print(f"median delta_depth = {median_depth:.4f} mm")
    print(f"median delta_visibility = {median_visibility:.7f}")
    print(
        f"delta_visibility > 0 = {positive_visibility}/{len(rows)} "
        f"({positive_visibility / len(rows) * 100:.1f}%)"
    )
    print(
        f"Demo frame exists = {len(demo_indices) == 1} "
        f"({DEMO_SEQUENCE}/{DEMO_FRAME})"
    )
    print(
        f"Demo values = ΔDepth {delta_depth[demo_indices[0]]:.1f} mm, "
        f"ΔVisibility {delta_visibility[demo_indices[0]]:.4f}"
    )
    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
