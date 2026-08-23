"""Visibility-weight sensitivity with GT-mask median selection distance.

This standalone experiment reuses the existing loading, matching, candidate,
and Priority Score implementations without modifying any research module.
Nearest agreement is evaluated against the complete exact-minimum mask-depth
set, so prediction order is never used to break nearest ties.
"""

import csv
import glob
import os
from collections import defaultdict

from data_loader import clean_depth, load_bupst20_annotation, load_depth
from load_detections import load_detections_csv
from mask_depth_reanalysis import (
    MODE,
    prepare_candidates,
    select_top1_precomputed,
)


DATA_DIR = "dataset_bulk"
DETECTION_CSV = "eval_detections_fixed.csv"
OUTPUT_CSV = "mask_depth_sensitivity_results.csv"
SEQUENCE_OUTPUT_CSV = "mask_depth_sensitivity_by_sequence.csv"

SEQUENCES = {str(i) for i in range(400, 410)}
CONF_THRESHOLD = 0.30
VISIBILITY_WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
BASELINE_VISIBILITY_WEIGHT = 0.3

EXPECTED_COMPARABLE_FRAMES = 364
EXPECTED_V0_DISAGREEMENTS = 63
EXPECTED_V03_DISAGREEMENTS = 118


def percentage(count, total):
    return count / total * 100 if total else 0.0


def make_weights(visibility_weight):
    """Preserve the 0.35:0.20:0.15 ratio in the non-visibility remainder."""
    remainder = 1.0 - visibility_weight
    return {
        "proximity": remainder * 0.35 / 0.70,
        "visibility": visibility_weight,
        "confidence": remainder * 0.20 / 0.70,
        "center": remainder * 0.15 / 0.70,
    }


def mask_nearest_set(candidates):
    minimum = min(candidate["mask_depth"] for candidate in candidates)
    return frozenset(
        candidate["id"]
        for candidate in candidates
        if candidate["mask_depth"] == minimum
    )


def new_stats(weights):
    return {
        "weights": weights,
        "frames": 0,
        "disagreement": 0,
        "full_agreement": 0,
        "per_sequence": defaultdict(
            lambda: {
                "frames": 0,
                "disagreement": 0,
                "full_agreement": 0,
            }
        ),
    }


def analyze():
    weights_by_visibility = {
        visibility: make_weights(visibility)
        for visibility in VISIBILITY_WEIGHTS
    }
    for visibility, weights in weights_by_visibility.items():
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-12:
            raise ValueError(
                f"Weights for visibility={visibility} sum to {weight_sum}"
            )

    results = {
        visibility: new_stats(weights)
        for visibility, weights in weights_by_visibility.items()
    }

    detections = load_detections_csv(DETECTION_CSV)
    depth_root = os.path.join(DATA_DIR, "depth")
    annotation_root = os.path.join(DATA_DIR, "annotations")
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
            nearest_ids = mask_nearest_set(candidates)

            selections = {}
            for visibility, weights in weights_by_visibility.items():
                top1, _ = select_top1_precomputed(
                    candidates,
                    "mask_depth",
                    weights,
                    image_size,
                    MODE,
                )
                if top1 is None:
                    raise RuntimeError(
                        f"No Top-1 for {sequence}/{frame_id}, V={visibility}"
                    )
                selections[visibility] = top1

            baseline_id = selections[BASELINE_VISIBILITY_WEIGHT]["id"]

            for visibility, top1 in selections.items():
                disagreement = top1["id"] not in nearest_ids
                full_agreement = top1["id"] == baseline_id
                stats = results[visibility]
                sequence_stats = stats["per_sequence"][sequence]

                stats["frames"] += 1
                stats["disagreement"] += int(disagreement)
                stats["full_agreement"] += int(full_agreement)
                sequence_stats["frames"] += 1
                sequence_stats["disagreement"] += int(disagreement)
                sequence_stats["full_agreement"] += int(full_agreement)

    return results


def summary_rows(results):
    rows = []
    for visibility in VISIBILITY_WEIGHTS:
        stats = results[visibility]
        weights = stats["weights"]
        rows.append(
            {
                "visibility_weight": visibility,
                "distance_weight": weights["proximity"],
                "confidence_weight": weights["confidence"],
                "center_weight": weights["center"],
                "weight_sum": sum(weights.values()),
                "comparable_frames": stats["frames"],
                "disagreement_count": stats["disagreement"],
                "disagreement_rate": percentage(
                    stats["disagreement"], stats["frames"]
                ),
                "full_agreement_count": stats["full_agreement"],
                "full_agreement_rate": percentage(
                    stats["full_agreement"], stats["frames"]
                ),
            }
        )
    return rows


def sequence_rows(results):
    rows = []
    for visibility in VISIBILITY_WEIGHTS:
        stats = results[visibility]
        for sequence in sorted(SEQUENCES):
            sequence_stats = stats["per_sequence"][sequence]
            rows.append(
                {
                    "visibility_weight": visibility,
                    "sequence_id": sequence,
                    "comparable_frames": sequence_stats["frames"],
                    "disagreement_count": sequence_stats["disagreement"],
                    "disagreement_rate": percentage(
                        sequence_stats["disagreement"],
                        sequence_stats["frames"],
                    ),
                    "full_agreement_count": sequence_stats["full_agreement"],
                    "full_agreement_rate": percentage(
                        sequence_stats["full_agreement"],
                        sequence_stats["frames"],
                    ),
                }
            )
    return rows


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_results(results):
    print("=" * 108)
    print("GT-mask Priority visibility-weight sensitivity")
    print("=" * 108)
    print(
        f"{'V':>5}{'Distance':>11}{'Confidence':>13}{'Center':>10}"
        f"{'frames':>10}{'disagree':>14}{'rate':>10}"
        f"{'same Full':>14}{'rate':>10}"
    )
    for visibility in VISIBILITY_WEIGHTS:
        stats = results[visibility]
        weights = stats["weights"]
        print(
            f"{visibility:>5.1f}{weights['proximity']:>11.4f}"
            f"{weights['confidence']:>13.4f}{weights['center']:>10.4f}"
            f"{stats['frames']:>10}{stats['disagreement']:>9}/{stats['frames']:<4}"
            f"{percentage(stats['disagreement'], stats['frames']):>9.1f}%"
            f"{stats['full_agreement']:>9}/{stats['frames']:<4}"
            f"{percentage(stats['full_agreement'], stats['frames']):>9.1f}%"
        )

    print()
    print("Sequence-wise tie-aware nearest disagreement")
    print(
        f"{'seq':<6}{'frames':>8}"
        + "".join(f"{'V=' + str(v):>14}" for v in VISIBILITY_WEIGHTS)
    )
    for sequence in sorted(SEQUENCES):
        frame_count = results[BASELINE_VISIBILITY_WEIGHT][
            "per_sequence"
        ][sequence]["frames"]
        values = []
        for visibility in VISIBILITY_WEIGHTS:
            sequence_stats = results[visibility]["per_sequence"][sequence]
            values.append(
                f"{sequence_stats['disagreement']}/{sequence_stats['frames']} "
                f"{percentage(sequence_stats['disagreement'], sequence_stats['frames']):.1f}%"
            )
        print(
            f"{sequence:<6}{frame_count:>8}"
            + "".join(f"{value:>14}" for value in values)
        )

    weights_valid = all(
        abs(sum(results[v]["weights"].values()) - 1.0) <= 1e-12
        for v in VISIBILITY_WEIGHTS
    )
    checks = [
        (
            "Comparable frames",
            results[BASELINE_VISIBILITY_WEIGHT]["frames"],
            EXPECTED_COMPARABLE_FRAMES,
        ),
        (
            "V=0.0 disagreement",
            results[0.0]["disagreement"],
            EXPECTED_V0_DISAGREEMENTS,
        ),
        (
            "V=0.3 disagreement",
            results[0.3]["disagreement"],
            EXPECTED_V03_DISAGREEMENTS,
        ),
        ("All weight sums equal 1.0", weights_valid, True),
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
    print(f"Summary CSV : {OUTPUT_CSV}")
    print(f"Sequence CSV: {SEQUENCE_OUTPUT_CSV}")
    if failed:
        raise RuntimeError("One or more sensitivity validation checks failed")


def main():
    results = analyze()
    write_csv(OUTPUT_CSV, summary_rows(results))
    write_csv(SEQUENCE_OUTPUT_CSV, sequence_rows(results))
    print_results(results)


if __name__ == "__main__":
    main()
