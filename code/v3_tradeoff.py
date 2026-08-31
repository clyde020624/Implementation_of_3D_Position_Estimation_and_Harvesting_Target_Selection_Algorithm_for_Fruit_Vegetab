"""v3 trade-off analysis using the pure GT-mask-depth candidate pool."""

from __future__ import annotations

import csv

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
from priority import score_center, score_visibility


DETECTION_CSV = "eval_detections_fixed.csv"
RESULTS_CSV = "v3_tradeoff_results.csv"
SUMMARY_CSV = "v3_tradeoff_summary.csv"

EXPECTED_FRAMES = 364
EXPECTED_CANDIDATES = 2728
EXPECTED_DISAGREEMENTS = 123

METRICS = (
    "delta_depth_mm",
    "delta_visibility",
    "delta_confidence",
    "delta_center",
)

RESULT_FIELDS = [
    "sequence_id",
    "frame_id",
    "candidate_count",
    "full_prediction_id",
    "full_gt_id",
    "full_mask_depth",
    "nearest_min_mask_depth",
    "nearest_set_size",
    "nearest_set_ids",
    "full_visibility",
    "nearest_set_max_visibility",
    "full_confidence",
    "nearest_set_max_confidence",
    "full_center_score",
    "nearest_set_max_center_score",
    "delta_depth_mm",
    "delta_visibility",
    "delta_confidence",
    "delta_center",
]

SUMMARY_FIELDS = [
    "metric",
    "frames",
    "median",
    "q1",
    "q3",
    "iqr",
    "mean",
    "min",
    "max",
    "positive_count",
    "positive_rate",
]


def percentage(count, total):
    return 100.0 * count / total if total else 0.0


def candidate_metrics(candidate, image_size):
    image_center = (image_size[0] / 2, image_size[1] / 2)
    return {
        "visibility": float(score_visibility(candidate)),
        "confidence": float(candidate["confidence"]),
        "center": float(score_center(candidate["bbox"], image_center)),
    }


def analyze():
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
        nearest = [
            candidate for candidate in candidates
            if float(candidate["mask_depth"]) == minimum_depth
        ]
        nearest_ids = {int(candidate["id"]) for candidate in nearest}
        if int(full_top1["id"]) in nearest_ids:
            continue

        full_metrics = candidate_metrics(full_top1, image_size)
        nearest_metrics = [
            candidate_metrics(candidate, image_size) for candidate in nearest
        ]
        nearest_max_visibility = max(
            item["visibility"] for item in nearest_metrics
        )
        nearest_max_confidence = max(
            item["confidence"] for item in nearest_metrics
        )
        nearest_max_center = max(
            item["center"] for item in nearest_metrics
        )
        full_depth = float(full_top1["mask_depth"])

        rows.append({
            "sequence_id": sequence,
            "frame_id": frame_id,
            "candidate_count": len(candidates),
            "full_prediction_id": int(full_top1["id"]),
            "full_gt_id": int(full_top1["matched_gt_id"]),
            "full_mask_depth": full_depth,
            "nearest_min_mask_depth": minimum_depth,
            "nearest_set_size": len(nearest),
            "nearest_set_ids": ";".join(
                str(value) for value in sorted(nearest_ids)
            ),
            "full_visibility": full_metrics["visibility"],
            "nearest_set_max_visibility": nearest_max_visibility,
            "full_confidence": full_metrics["confidence"],
            "nearest_set_max_confidence": nearest_max_confidence,
            "full_center_score": full_metrics["center"],
            "nearest_set_max_center_score": nearest_max_center,
            "delta_depth_mm": full_depth - minimum_depth,
            "delta_visibility": (
                full_metrics["visibility"] - nearest_max_visibility
            ),
            "delta_confidence": (
                full_metrics["confidence"] - nearest_max_confidence
            ),
            "delta_center": full_metrics["center"] - nearest_max_center,
        })

    if candidate_total != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"Pure candidates={candidate_total}, expected={EXPECTED_CANDIDATES}"
        )
    if len(rows) != EXPECTED_DISAGREEMENTS:
        raise RuntimeError(
            "v3 disagreement validation failed before CSV write: "
            f"actual={len(rows)}/{len(frames)}, "
            f"expected={EXPECTED_DISAGREEMENTS}/{EXPECTED_FRAMES}"
        )
    positive_depth = sum(row["delta_depth_mm"] > 0 for row in rows)
    if positive_depth != len(rows):
        raise RuntimeError(
            "Delta Depth validation failed before CSV write: "
            f"positive={positive_depth}, disagreement={len(rows)}"
        )
    return rows, candidate_total


def summarize(rows):
    summaries = []
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        positive_count = int(np.sum(values > 0))
        summaries.append({
            "metric": metric,
            "frames": len(values),
            "median": float(np.median(values)),
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "positive_count": positive_count,
            "positive_rate": percentage(positive_count, len(values)),
        })
    return summaries


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(summaries):
    print(
        "Metric | Median | Q1 | Q3 | Mean | Min | Max | "
        "Positive count | Positive %"
    )
    for row in summaries:
        print(
            f"{row['metric']} | "
            f"{row['median']:.4f} | "
            f"{row['q1']:.4f} | "
            f"{row['q3']:.4f} | "
            f"{row['mean']:.4f} | "
            f"{row['min']:.4f} | "
            f"{row['max']:.4f} | "
            f"{row['positive_count']} | "
            f"{row['positive_rate']:.1f}%"
        )


def main():
    rows, _ = analyze()
    summaries = summarize(rows)
    write_csv(RESULTS_CSV, rows, RESULT_FIELDS)
    write_csv(SUMMARY_CSV, summaries, SUMMARY_FIELDS)
    print_table(summaries)


if __name__ == "__main__":
    main()
