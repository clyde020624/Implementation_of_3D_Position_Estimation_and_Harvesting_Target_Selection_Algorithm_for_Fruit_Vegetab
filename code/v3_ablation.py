"""Ablation evaluation using the validated pure GT-mask-depth candidate pool."""

from __future__ import annotations

import csv
import os

from compare_candidate_pool_top1 import (
    build_pure_candidates,
    load_frame_inputs,
    reference_frames,
)
from load_detections import load_detections_csv
from mask_depth_ablation import ABLATIONS
from mask_depth_reanalysis import MODE, select_top1_precomputed


DETECTION_CSV = "eval_detections_fixed.csv"
OUTPUT_CSV = "pure_mask_ablation_results.csv"

EXPECTED_FRAMES = 364
EXPECTED_CANDIDATES = 2728
EXPECTED_FULL_DISAGREEMENT = 123
EXPECTED_DISTANCE_DISAGREEMENT = 0

CONDITION_LABELS = {
    "D": "Distance only",
    "D+V": "Distance + Visibility",
    "D+V+Conf": "Distance + Visibility + Confidence",
    "Full": "Full",
    "Full-V": "Full - Visibility",
}

FIELDS = [
    "condition",
    "distance_weight",
    "visibility_weight",
    "confidence_weight",
    "center_weight",
    "comparable_frames",
    "candidate_objects",
    "disagreement_count",
    "disagreement_rate",
    "agreement_with_full_count",
    "agreement_with_full_rate",
]


def percentage(count, total):
    return 100.0 * count / total if total else 0.0


def validate_weights():
    for name, weights in ABLATIONS.items():
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-12:
            raise RuntimeError(f"{name} weights sum to {total}, expected 1.0")


def nearest_set(candidates):
    minimum = min(float(candidate["mask_depth"]) for candidate in candidates)
    return {
        int(candidate["id"]) for candidate in candidates
        if float(candidate["mask_depth"]) == minimum
    }


def evaluate():
    validate_weights()
    detections = load_detections_csv(DETECTION_CSV)
    frames = reference_frames()
    if len(frames) != EXPECTED_FRAMES:
        raise RuntimeError(
            f"Reference frames={len(frames)}, expected={EXPECTED_FRAMES}"
        )

    totals = {
        name: {"disagreement": 0, "same_as_full": 0}
        for name in ABLATIONS
    }
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
        nearest_ids = nearest_set(candidates)

        selected_ids = {}
        for name, weights in ABLATIONS.items():
            top1, _ = select_top1_precomputed(
                candidates,
                "mask_depth",
                weights,
                image_size,
                MODE,
            )
            if top1 is None:
                raise RuntimeError(
                    f"No Top-1 for {sequence}/{frame_id}, condition={name}"
                )
            selected_ids[name] = int(top1["id"])

        full_id = selected_ids["Full"]
        for name, selected_id in selected_ids.items():
            totals[name]["disagreement"] += int(
                selected_id not in nearest_ids
            )
            totals[name]["same_as_full"] += int(selected_id == full_id)

    if candidate_total != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"Pure candidates={candidate_total}, expected={EXPECTED_CANDIDATES}"
        )

    full_disagreement = totals["Full"]["disagreement"]
    if full_disagreement != EXPECTED_FULL_DISAGREEMENT:
        raise RuntimeError(
            "Full validation failed before CSV write: "
            f"actual={full_disagreement}/{len(frames)}, "
            f"expected={EXPECTED_FULL_DISAGREEMENT}/{EXPECTED_FRAMES}"
        )
    if totals["D"]["disagreement"] != EXPECTED_DISTANCE_DISAGREEMENT:
        raise RuntimeError(
            "Distance-only validation failed before CSV write: "
            f"actual={totals['D']['disagreement']}, expected=0"
        )

    rows = []
    for name, weights in ABLATIONS.items():
        disagreement = totals[name]["disagreement"]
        same_as_full = totals[name]["same_as_full"]
        rows.append({
            "condition": CONDITION_LABELS[name],
            "distance_weight": weights["proximity"],
            "visibility_weight": weights["visibility"],
            "confidence_weight": weights["confidence"],
            "center_weight": weights["center"],
            "comparable_frames": len(frames),
            "candidate_objects": candidate_total,
            "disagreement_count": disagreement,
            "disagreement_rate": percentage(disagreement, len(frames)),
            "agreement_with_full_count": same_as_full,
            "agreement_with_full_rate": percentage(
                same_as_full, len(frames)
            ),
        })
    return rows


def write_output(rows):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    print(
        "Condition | Disagreement count | Disagreement % | "
        "Agreement with Full count | Agreement with Full %"
    )
    for row in rows:
        print(
            f"{row['condition']} | "
            f"{row['disagreement_count']} | "
            f"{row['disagreement_rate']:.1f}% | "
            f"{row['agreement_with_full_count']} | "
            f"{row['agreement_with_full_rate']:.1f}%"
        )


def main():
    rows = evaluate()
    write_output(rows)
    print_table(rows)


if __name__ == "__main__":
    main()
