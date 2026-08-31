"""v3 visibility-weight sensitivity using the pure mask-depth pool."""

from __future__ import annotations

import csv

from compare_candidate_pool_top1 import (
    build_pure_candidates,
    load_frame_inputs,
    reference_frames,
)
from load_detections import load_detections_csv
from mask_depth_reanalysis import MODE, select_top1_precomputed


DETECTION_CSV = "eval_detections_fixed.csv"
OUTPUT_CSV = "v3_sensitivity_results.csv"

VISIBILITY_WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
BASELINE_VISIBILITY = 0.3
EXPECTED_FRAMES = 364
EXPECTED_CANDIDATES = 2728
EXPECTED_BASELINE_DISAGREEMENT = 123

FIELDS = [
    "visibility_weight",
    "distance_weight",
    "confidence_weight",
    "center_weight",
    "weight_sum",
    "comparable_frames",
    "candidate_objects",
    "disagreement_count",
    "disagreement_rate",
    "baseline_agreement_count",
    "baseline_agreement_rate",
]


def percentage(count, total):
    return 100.0 * count / total if total else 0.0


def make_weights(visibility):
    remainder = 1.0 - visibility
    return {
        "proximity": remainder * 0.35 / 0.70,
        "visibility": visibility,
        "confidence": remainder * 0.20 / 0.70,
        "center": remainder * 0.15 / 0.70,
    }


def nearest_set(candidates):
    minimum = min(float(item["mask_depth"]) for item in candidates)
    return {
        int(item["id"]) for item in candidates
        if float(item["mask_depth"]) == minimum
    }


def evaluate():
    weights_by_visibility = {
        visibility: make_weights(visibility)
        for visibility in VISIBILITY_WEIGHTS
    }
    for visibility, weights in weights_by_visibility.items():
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-12:
            raise RuntimeError(
                f"V={visibility} weights sum to {total}, expected 1.0"
            )

    detections = load_detections_csv(DETECTION_CSV)
    frames = reference_frames()
    if len(frames) != EXPECTED_FRAMES:
        raise RuntimeError(
            f"Reference frames={len(frames)}, expected={EXPECTED_FRAMES}"
        )

    totals = {
        visibility: {"disagreement": 0, "baseline_agreement": 0}
        for visibility in VISIBILITY_WEIGHTS
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
            selected_ids[visibility] = int(top1["id"])

        baseline_id = selected_ids[BASELINE_VISIBILITY]
        for visibility, selected_id in selected_ids.items():
            totals[visibility]["disagreement"] += int(
                selected_id not in nearest_ids
            )
            totals[visibility]["baseline_agreement"] += int(
                selected_id == baseline_id
            )

    if candidate_total != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"Pure candidates={candidate_total}, expected={EXPECTED_CANDIDATES}"
        )
    baseline_disagreement = totals[BASELINE_VISIBILITY]["disagreement"]
    if baseline_disagreement != EXPECTED_BASELINE_DISAGREEMENT:
        raise RuntimeError(
            "V=0.3 validation failed before CSV write: "
            f"actual={baseline_disagreement}/{len(frames)}, "
            f"expected={EXPECTED_BASELINE_DISAGREEMENT}/{EXPECTED_FRAMES}"
        )

    rows = []
    for visibility in VISIBILITY_WEIGHTS:
        weights = weights_by_visibility[visibility]
        disagreement = totals[visibility]["disagreement"]
        baseline_agreement = totals[visibility]["baseline_agreement"]
        rows.append({
            "visibility_weight": visibility,
            "distance_weight": weights["proximity"],
            "confidence_weight": weights["confidence"],
            "center_weight": weights["center"],
            "weight_sum": sum(weights.values()),
            "comparable_frames": len(frames),
            "candidate_objects": candidate_total,
            "disagreement_count": disagreement,
            "disagreement_rate": percentage(disagreement, len(frames)),
            "baseline_agreement_count": baseline_agreement,
            "baseline_agreement_rate": percentage(
                baseline_agreement, len(frames)
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
        "V | D | Conf | Center | Disagreement count | Rate | "
        "Agreement with V=0.3"
    )
    for row in rows:
        print(
            f"{row['visibility_weight']:.1f} | "
            f"{row['distance_weight']:.4f} | "
            f"{row['confidence_weight']:.4f} | "
            f"{row['center_weight']:.4f} | "
            f"{row['disagreement_count']} | "
            f"{row['disagreement_rate']:.1f}% | "
            f"{row['baseline_agreement_count']}/"
            f"{row['comparable_frames']} "
            f"({row['baseline_agreement_rate']:.1f}%)"
        )


def main():
    rows = evaluate()
    write_output(rows)
    print_table(rows)


if __name__ == "__main__":
    main()
