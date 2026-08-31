"""Create the v3 absolute depth-visibility scatter and its plot-data CSV."""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from compare_candidate_pool_top1 import (
    build_pure_candidates,
    load_frame_inputs,
    reference_frames,
)
from load_detections import load_detections_csv
from mask_depth_reanalysis import (
    FULL_WEIGHTS,
    MODE,
    select_top1_precomputed,
)
from priority import score_visibility


DETECTION_CSV = "eval_detections_fixed.csv"
OUTPUT_CSV = "v3_scatter_plot_data.csv"
OUTPUT_PNG = "v3_depth_visibility_scatter.png"

EXPECTED_FRAMES = 364
EXPECTED_CANDIDATES = 2728
EXPECTED_DISAGREEMENTS = 123
EXPECTED_VISIBILITY_HIGHER = 115
EXPECTED_MEDIAN_DEPTH = 13.0
EXPECTED_MEDIAN_VISIBILITY = 0.17393529386893647

FIELDS = [
    "sequence",
    "frame_id",
    "nearest_pred_id",
    "nearest_gt_id",
    "nearest_depth_mm",
    "nearest_visibility",
    "full_pred_id",
    "full_gt_id",
    "full_depth_mm",
    "full_visibility",
    "delta_depth_mm",
    "delta_visibility",
]


def reconstruct_rows():
    detections = load_detections_csv(DETECTION_CSV)
    frames = reference_frames()
    if len(frames) != EXPECTED_FRAMES:
        raise RuntimeError(
            f"Reference frames={len(frames)}, expected={EXPECTED_FRAMES}"
        )

    rows = []
    candidate_total = 0
    for sequence, frame_id in sorted(frames):
        predictions, ground_truth, depth, image_size = load_frame_inputs(
            detections, sequence, frame_id
        )
        candidates = build_pure_candidates(
            [dict(item) for item in predictions],
            ground_truth,
            depth,
            image_size,
        )
        if len(candidates) < 2:
            raise RuntimeError(
                f"Pure comparison frame has fewer than two candidates: "
                f"{sequence}/{frame_id}"
            )
        candidate_total += len(candidates)

        full_top1, _ = select_top1_precomputed(
            candidates,
            "mask_depth",
            FULL_WEIGHTS,
            image_size,
            MODE,
        )
        if full_top1 is None:
            raise RuntimeError(f"No Full Top-1: {sequence}/{frame_id}")

        minimum_depth = min(
            float(candidate["mask_depth"]) for candidate in candidates
        )
        nearest_set = [
            candidate for candidate in candidates
            if float(candidate["mask_depth"]) == minimum_depth
        ]
        nearest_ids = {int(candidate["id"]) for candidate in nearest_set}
        if int(full_top1["id"]) in nearest_ids:
            continue

        # Match v3 trade-off's conservative comparison: the nearest-set
        # representative has maximum visibility. Prediction ID resolves only
        # an exact visibility tie and cannot change the plotted coordinate.
        nearest = min(
            nearest_set,
            key=lambda candidate: (
                -float(score_visibility(candidate)),
                int(candidate["id"]),
            ),
        )
        nearest_visibility = float(score_visibility(nearest))
        full_visibility = float(score_visibility(full_top1))
        nearest_depth = float(nearest["mask_depth"])
        full_depth = float(full_top1["mask_depth"])

        rows.append({
            "sequence": sequence,
            "frame_id": frame_id,
            "nearest_pred_id": int(nearest["id"]),
            "nearest_gt_id": int(nearest["matched_gt_id"]),
            "nearest_depth_mm": nearest_depth,
            "nearest_visibility": nearest_visibility,
            "full_pred_id": int(full_top1["id"]),
            "full_gt_id": int(full_top1["matched_gt_id"]),
            "full_depth_mm": full_depth,
            "full_visibility": full_visibility,
            "delta_depth_mm": full_depth - nearest_depth,
            "delta_visibility": full_visibility - nearest_visibility,
        })

    if candidate_total != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"Pure candidates={candidate_total}, expected={EXPECTED_CANDIDATES}"
        )
    return rows


def validate(rows):
    delta_depth = np.asarray(
        [row["delta_depth_mm"] for row in rows], dtype=float
    )
    delta_visibility = np.asarray(
        [row["delta_visibility"] for row in rows], dtype=float
    )
    positive_depth = int(np.count_nonzero(delta_depth > 0))
    visibility_higher = int(np.count_nonzero(delta_visibility > 0))
    median_depth = float(np.median(delta_depth))
    median_visibility = float(np.median(delta_visibility))

    errors = []
    if len(rows) != EXPECTED_DISAGREEMENTS:
        errors.append(
            f"disagreement rows={len(rows)}, expected={EXPECTED_DISAGREEMENTS}"
        )
    if positive_depth != len(rows):
        errors.append(
            f"positive Delta Depth={positive_depth}, expected={len(rows)}"
        )
    if visibility_higher != EXPECTED_VISIBILITY_HIGHER:
        errors.append(
            f"visibility higher={visibility_higher}, "
            f"expected={EXPECTED_VISIBILITY_HIGHER}"
        )
    if abs(median_depth - EXPECTED_MEDIAN_DEPTH) > 1e-9:
        errors.append(
            f"median Delta Depth={median_depth}, "
            f"expected={EXPECTED_MEDIAN_DEPTH}"
        )
    if abs(median_visibility - EXPECTED_MEDIAN_VISIBILITY) > 1e-9:
        errors.append(
            f"median Delta Visibility={median_visibility}, "
            f"expected={EXPECTED_MEDIAN_VISIBILITY}"
        )
    if errors:
        raise RuntimeError(
            "v3 scatter validation failed before outputs: "
            + " | ".join(errors)
        )
    return {
        "positive_depth": positive_depth,
        "visibility_higher": visibility_higher,
        "median_depth": median_depth,
        "median_visibility": median_visibility,
    }


def write_csv(rows):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows, stats):
    nearest_depth = np.asarray(
        [row["nearest_depth_mm"] for row in rows], dtype=float
    )
    nearest_visibility = np.asarray(
        [row["nearest_visibility"] for row in rows], dtype=float
    )
    full_depth = np.asarray(
        [row["full_depth_mm"] for row in rows], dtype=float
    )
    full_visibility = np.asarray(
        [row["full_visibility"] for row in rows], dtype=float
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.labelsize": 11,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
    })

    figure, axis = plt.subplots(figsize=(7.2, 5.4))

    for n_depth, n_vis, f_depth, f_vis in zip(
        nearest_depth,
        nearest_visibility,
        full_depth,
        full_visibility,
    ):
        axis.plot(
            [n_depth, f_depth],
            [n_vis, f_vis],
            color="#A8A8A8",
            linewidth=0.55,
            alpha=0.24,
            zorder=1,
        )

    axis.scatter(
        nearest_depth,
        nearest_visibility,
        s=27,
        marker="o",
        facecolors="#2C7FB8",
        edgecolors="white",
        linewidths=0.35,
        alpha=0.72,
        label="Nearest",
        zorder=3,
    )
    axis.scatter(
        full_depth,
        full_visibility,
        s=31,
        marker="^",
        facecolors="#D84A4A",
        edgecolors="white",
        linewidths=0.35,
        alpha=0.72,
        label="Full Priority",
        zorder=4,
    )

    all_depth = np.concatenate([nearest_depth, full_depth])
    all_visibility = np.concatenate([nearest_visibility, full_visibility])
    x_padding = max(8.0, float(np.ptp(all_depth)) * 0.035)
    y_padding = max(0.02, float(np.ptp(all_visibility)) * 0.05)
    axis.set_xlim(
        float(np.min(all_depth)) - x_padding,
        float(np.max(all_depth)) + x_padding,
    )
    axis.set_ylim(
        max(0.0, float(np.min(all_visibility)) - y_padding),
        min(1.0, float(np.max(all_visibility)) + y_padding),
    )

    axis.set_xlabel("Representative depth (mm)")
    axis.set_ylabel("Visibility score")
    axis.set_title(
        "Depth–Visibility distribution of selected targets "
        "(v3, disagreement frames)"
    )
    axis.grid(
        color="#C8C8C8",
        linestyle="-",
        linewidth=0.55,
        alpha=0.28,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="lower right")

    annotation = (
        f"Frames = {len(rows)}\n"
        f"Median Δdepth = +{stats['median_depth']:.0f} mm\n"
        f"Median Δvisibility = +{stats['median_visibility']:.4f}\n"
        f"Full visibility higher: "
        f"{stats['visibility_higher']}/{len(rows)} "
        f"({stats['visibility_higher'] / len(rows) * 100:.1f}%)"
    )
    axis.text(
        0.025,
        0.975,
        annotation,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#C7C7C7",
            "linewidth": 0.6,
            "alpha": 0.90,
        },
        zorder=6,
    )

    figure.tight_layout(pad=0.8)
    figure.savefig(
        OUTPUT_PNG,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main():
    rows = reconstruct_rows()
    stats = validate(rows)
    write_csv(rows)
    make_figure(rows, stats)

    print(f"disagreement frames = {len(rows)}")
    print(f"data rows in CSV = {len(rows)}")
    print(
        "all delta_depth_mm > 0 = "
        f"{stats['positive_depth'] == len(rows)}"
    )
    print(
        "Full visibility > Nearest visibility count = "
        f"{stats['visibility_higher']}/{len(rows)}"
    )


if __name__ == "__main__":
    main()
