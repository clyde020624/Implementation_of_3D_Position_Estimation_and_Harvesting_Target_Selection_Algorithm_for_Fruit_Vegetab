"""Trade-off analysis for GT-mask-depth Full Priority disagreements.

This standalone experiment reuses the existing data loading, greedy GT
matching, candidate preparation, and Priority Score definitions. It does not
modify the existing research files. Nearest evaluation is tie-aware.
"""

import csv
import glob
import os
from collections import defaultdict

import numpy as np

from data_loader import clean_depth, load_bupst20_annotation, load_depth
from load_detections import load_detections_csv
from mask_depth_reanalysis import (
    FULL_WEIGHTS,
    MODE,
    prepare_candidates,
    select_top1_precomputed,
)
from priority import score_center, score_visibility


DATA_DIR = "dataset_bulk"
DETECTION_CSV = "eval_detections_fixed.csv"
OUTPUT_CSV = "mask_depth_tradeoff_results.csv"

SEQUENCES = {str(i) for i in range(400, 410)}
CONF_THRESHOLD = 0.30

EXPECTED_COMPARABLE_FRAMES = 364
EXPECTED_DISAGREEMENT_FRAMES = 118

DELTA_KEYS = (
    "delta_depth_mm",
    "delta_visibility",
    "delta_confidence",
    "delta_center",
)


def percentage(count, total):
    return count / total * 100 if total else 0.0


def mask_nearest_candidates(candidates):
    """Return every candidate at the exact minimum GT-mask median depth."""
    minimum = min(candidate["mask_depth"] for candidate in candidates)
    nearest = [
        candidate
        for candidate in candidates
        if candidate["mask_depth"] == minimum
    ]
    return nearest, minimum


def object_metrics(candidate, image_size):
    image_center = (image_size[0] / 2, image_size[1] / 2)
    return {
        "visibility": float(score_visibility(candidate)),
        "confidence": float(candidate["confidence"]),
        "center": float(score_center(candidate["bbox"], image_center)),
    }


def analyze():
    detections = load_detections_csv(DETECTION_CSV)
    depth_root = os.path.join(DATA_DIR, "depth")
    annotation_root = os.path.join(DATA_DIR, "annotations")

    rows = []
    comparable_frames = 0
    comparable_per_sequence = defaultdict(int)

    sequences = [
        sequence
        for sequence in sorted(os.listdir(depth_root))
        if sequence in SEQUENCES
    ]

    for sequence in sequences:
        depth_files = {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in glob.glob(
                os.path.join(depth_root, sequence, "*.tif*")
            )
        }
        annotation_files = {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in glob.glob(
                os.path.join(annotation_root, sequence, "*.pkl")
            )
        }

        for frame_id in sorted(set(depth_files) & set(annotation_files)):
            info = detections.get((sequence, frame_id))
            if info is None:
                continue

            predictions = [
                dict(obj)
                for obj in info["objects"]
                if obj["confidence"] >= CONF_THRESHOLD
            ]
            if len(predictions) < 2:
                continue

            depth = clean_depth(load_depth(depth_files[frame_id]))
            ground_truth = load_bupst20_annotation(
                annotation_files[frame_id]
            )
            candidates = prepare_candidates(
                predictions,
                ground_truth,
                depth,
            )
            if len(candidates) < 2:
                continue

            image_size = (
                info["img_size"]
                if info["img_size"] is not None
                else (720, 1280)
            )
            full_top1, _ = select_top1_precomputed(
                candidates,
                "mask_depth",
                FULL_WEIGHTS,
                image_size,
                MODE,
            )
            if full_top1 is None:
                continue

            comparable_frames += 1
            comparable_per_sequence[sequence] += 1

            nearest, nearest_minimum = mask_nearest_candidates(candidates)
            nearest_ids = frozenset(candidate["id"] for candidate in nearest)

            # Agreement frames are outside the trade-off analysis population.
            if full_top1["id"] in nearest_ids:
                continue

            full_metrics = object_metrics(full_top1, image_size)
            nearest_metrics = [
                object_metrics(candidate, image_size)
                for candidate in nearest
            ]
            nearest_max_visibility = max(
                metrics["visibility"] for metrics in nearest_metrics
            )
            nearest_max_confidence = max(
                metrics["confidence"] for metrics in nearest_metrics
            )
            nearest_max_center = max(
                metrics["center"] for metrics in nearest_metrics
            )

            full_mask_depth = float(full_top1["mask_depth"])
            delta_depth = full_mask_depth - float(nearest_minimum)
            delta_visibility = (
                full_metrics["visibility"] - nearest_max_visibility
            )
            delta_confidence = (
                full_metrics["confidence"] - nearest_max_confidence
            )
            delta_center = full_metrics["center"] - nearest_max_center

            rows.append(
                {
                    "sequence_id": sequence,
                    "frame_id": frame_id,
                    "full_prediction_id": full_top1["id"],
                    "full_mask_depth": full_mask_depth,
                    "nearest_min_mask_depth": float(nearest_minimum),
                    "nearest_set_size": len(nearest),
                    "nearest_set_ids": ";".join(
                        str(candidate_id)
                        for candidate_id in sorted(nearest_ids)
                    ),
                    "full_visibility": full_metrics["visibility"],
                    "nearest_set_max_visibility": nearest_max_visibility,
                    "full_confidence": full_metrics["confidence"],
                    "nearest_set_max_confidence": nearest_max_confidence,
                    "full_center_score": full_metrics["center"],
                    "nearest_set_max_center_score": nearest_max_center,
                    "delta_depth_mm": delta_depth,
                    "delta_visibility": delta_visibility,
                    "delta_confidence": delta_confidence,
                    "delta_center": delta_center,
                }
            )

    return rows, comparable_frames, comparable_per_sequence


def write_csv(rows):
    fieldnames = [
        "sequence_id",
        "frame_id",
        "full_prediction_id",
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
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def format_median(rows, key):
    if not rows:
        return "n/a"
    return f"{np.median([row[key] for row in rows]):.4f}"


def print_results(rows, comparable_frames, comparable_per_sequence):
    disagreement_frames = len(rows)

    print("=" * 92)
    print("GT-mask Full Priority vs tie-aware nearest trade-off")
    print("=" * 92)
    print(f"Comparable frames : {comparable_frames}")
    print(
        f"Trade-off frames  : {disagreement_frames}/{comparable_frames} "
        f"({percentage(disagreement_frames, comparable_frames):.1f}%)"
    )

    print()
    print("Delta summary over disagreement frames")
    print(
        f"{'metric':<22}{'mean':>12}{'median':>12}{'Q1':>12}"
        f"{'Q3':>12}{'min':>12}{'max':>12}"
    )
    for key in DELTA_KEYS:
        stats = summarize([row[key] for row in rows])
        print(
            f"{key:<22}{stats['mean']:>12.4f}{stats['median']:>12.4f}"
            f"{stats['q1']:>12.4f}{stats['q3']:>12.4f}"
            f"{stats['min']:>12.4f}{stats['max']:>12.4f}"
        )

    visibility_higher = sum(row["delta_visibility"] > 0 for row in rows)
    confidence_higher = sum(row["delta_confidence"] > 0 for row in rows)
    center_higher = sum(row["delta_center"] > 0 for row in rows)
    farther_and_visibility_higher = sum(
        row["delta_depth_mm"] > 0 and row["delta_visibility"] > 0
        for row in rows
    )

    print()
    print("Positive trade-off rates")
    rate_rows = [
        ("Full visibility > nearest-set max", visibility_higher),
        ("Full confidence > nearest-set max", confidence_higher),
        ("Full center > nearest-set max", center_higher),
        (
            "Full farther and visibility higher",
            farther_and_visibility_higher,
        ),
    ]
    for label, count in rate_rows:
        print(
            f"  {label:<42}: {count}/{disagreement_frames} "
            f"({percentage(count, disagreement_frames):.1f}%)"
        )

    print()
    print("Sequence-wise disagreement trade-off")
    print(
        f"{'seq':<6}{'comparable':>12}{'disagree':>12}"
        f"{'median dDepth':>18}{'median dVis':>16}"
    )
    for sequence in sorted(SEQUENCES):
        sequence_rows = [
            row for row in rows if row["sequence_id"] == sequence
        ]
        print(
            f"{sequence:<6}{comparable_per_sequence[sequence]:>12}"
            f"{len(sequence_rows):>12}"
            f"{format_median(sequence_rows, 'delta_depth_mm'):>18}"
            f"{format_median(sequence_rows, 'delta_visibility'):>16}"
        )

    positive_depth_count = sum(
        row["delta_depth_mm"] > 0 for row in rows
    )
    checks = [
        (
            "Comparable frames",
            comparable_frames,
            EXPECTED_COMPARABLE_FRAMES,
        ),
        (
            "Trade-off disagreement frames",
            disagreement_frames,
            EXPECTED_DISAGREEMENT_FRAMES,
        ),
        (
            "Frames with delta_depth_mm > 0",
            positive_depth_count,
            disagreement_frames,
        ),
    ]

    print()
    print("Validation")
    failed = False
    for label, actual, expected in checks:
        passed = actual == expected
        failed = failed or not passed
        print(
            f"  [{'PASS' if passed else 'FAIL'}] {label}: "
            f"actual={actual}, expected={expected}"
        )

    print()
    print(f"Per-frame CSV: {OUTPUT_CSV}")
    if failed:
        raise RuntimeError("One or more trade-off validation checks failed")


def main():
    rows, comparable_frames, comparable_per_sequence = analyze()
    write_csv(rows)
    print_results(rows, comparable_frames, comparable_per_sequence)


if __name__ == "__main__":
    main()
