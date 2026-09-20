"""Audit frame-wise proximity min-max normalization in 69 affected frames."""

from __future__ import annotations

import csv
import os

from compare_candidate_pool_top1 import (
    build_pure_candidates,
    load_frame_inputs,
)
from load_detections import load_detections_csv
from mask_depth_reanalysis import prepare_candidates
from trace_changed_disagreement_frames import score_pool


COMPARISON_CSV = "candidate_pool_top1_comparison.csv"
DETECTION_CSV = "eval_detections_fixed.csv"
OUTPUT_CSV = "proximity_normalization_impact_audit.csv"

EXPECTED_AFFECTED_FRAMES = 69
EXPECTED_ADDED_CANDIDATES = 84
EXPECTED_NEAREST_CHANGED = 18
EXPECTED_FULL_CHANGED = 17

FIELDS = [
    "sequence", "frame_id",
    "legacy_candidate_count", "pure_candidate_count",
    "added_candidate_count", "added_pred_gt_depth",
    "legacy_depth_min", "legacy_depth_max", "legacy_depth_range",
    "pure_depth_min", "pure_depth_max", "pure_depth_range",
    "depth_min_changed", "depth_max_changed", "depth_range_changed",
    "legacy_nearest_set", "pure_nearest_set",
    "nearest_changed", "nearest_change_class",
    "added_candidate_is_pure_nearest",
    "legacy_full_pred", "pure_full_pred",
    "full_changed", "full_change_class",
    "added_candidate_is_pure_full",
    "legacy_full_legacy_proximity", "legacy_full_legacy_score",
    "legacy_full_pure_proximity", "legacy_full_pure_score",
    "pure_full_legacy_proximity", "pure_full_legacy_score",
    "pure_full_pure_proximity", "pure_full_pure_score",
]


def as_bool(value):
    return str(value).strip().lower() == "true"


def parse_id_set(value):
    text = str(value).strip()
    return {int(item) for item in text.split(";")} if text else set()


def serialize_ids(values):
    return ";".join(str(value) for value in sorted(values))


def metric_value(pool, pred_id, key):
    metric = pool["metrics"].get(pred_id)
    return "" if metric is None else metric[key]


def load_affected_reference_rows():
    with open(COMPARISON_CSV, newline="", encoding="utf-8-sig") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if int(row["added_candidate_count"]) > 0
        ]
    added_total = sum(int(row["added_candidate_count"]) for row in rows)
    nearest_changed = sum(as_bool(row["nearest_changed"]) for row in rows)
    full_changed = sum(as_bool(row["full_changed"]) for row in rows)
    if (
        len(rows) != EXPECTED_AFFECTED_FRAMES
        or added_total != EXPECTED_ADDED_CANDIDATES
        or nearest_changed != EXPECTED_NEAREST_CHANGED
        or full_changed != EXPECTED_FULL_CHANGED
    ):
        raise RuntimeError(
            "Affected reference mismatch: "
            f"frames={len(rows)}, added={added_total}, "
            f"nearest_changed={nearest_changed}, full_changed={full_changed}"
        )
    return rows


def validate_reference(reference, legacy, pure, added_ids):
    actual = {
        "legacy_count": legacy["count"],
        "pure_count": pure["count"],
        "added_count": len(added_ids),
        "legacy_nearest": legacy["nearest_ids"],
        "pure_nearest": pure["nearest_ids"],
        "legacy_full": legacy["full_id"],
        "pure_full": pure["full_id"],
        "nearest_changed": legacy["nearest_ids"] != pure["nearest_ids"],
        "full_changed": legacy["full_id"] != pure["full_id"],
    }
    expected = {
        "legacy_count": int(reference["legacy_candidate_count"]),
        "pure_count": int(reference["pure_candidate_count"]),
        "added_count": int(reference["added_candidate_count"]),
        "legacy_nearest": parse_id_set(reference["legacy_nearest_set"]),
        "pure_nearest": parse_id_set(reference["pure_nearest_set"]),
        "legacy_full": int(reference["legacy_full_pred"]),
        "pure_full": int(reference["pure_full_pred"]),
        "nearest_changed": as_bool(reference["nearest_changed"]),
        "full_changed": as_bool(reference["full_changed"]),
    }
    if actual != expected:
        raise RuntimeError(
            f"Comparison mismatch at {reference['sequence']}/"
            f"{reference['frame_id']}: expected={expected}, actual={actual}"
        )


def audit_frames(reference_rows):
    detections = load_detections_csv(DETECTION_CSV)
    rows = []
    for reference in reference_rows:
        sequence, frame_id = reference["sequence"], reference["frame_id"]
        predictions, ground_truth, depth, image_size = load_frame_inputs(
            detections, sequence, frame_id
        )
        legacy_candidates = prepare_candidates(
            [dict(item) for item in predictions], ground_truth, depth
        )
        pure_candidates = build_pure_candidates(
            [dict(item) for item in predictions],
            ground_truth,
            depth,
            image_size,
        )
        legacy = score_pool(legacy_candidates, image_size)
        pure = score_pool(pure_candidates, image_size)
        added_ids = pure["ids"] - legacy["ids"]
        validate_reference(reference, legacy, pure, added_ids)

        legacy_depths = [
            item["mask_median_depth"] for item in legacy["metrics"].values()
        ]
        pure_depths = [
            item["mask_median_depth"] for item in pure["metrics"].values()
        ]
        legacy_min, legacy_max = min(legacy_depths), max(legacy_depths)
        pure_min, pure_max = min(pure_depths), max(pure_depths)
        min_changed = legacy_min != pure_min
        max_changed = legacy_max != pure_max
        range_changed = min_changed or max_changed

        added_nearest_ids = added_ids & pure["nearest_ids"]
        nearest_changed = legacy["nearest_ids"] != pure["nearest_ids"]
        if nearest_changed and not added_nearest_ids:
            raise RuntimeError(
                f"Nearest changed without a new nearest: {sequence}/{frame_id}"
            )
        nearest_class = (
            "NEW_CANDIDATE_NEAREST"
            if added_nearest_ids else "NEAREST_UNCHANGED"
        )

        full_changed = legacy["full_id"] != pure["full_id"]
        added_is_full = pure["full_id"] in added_ids
        if not full_changed:
            full_class = "FULL_UNCHANGED"
        elif added_is_full:
            full_class = "DIRECT_NEW_FULL"
        elif pure["full_id"] in legacy["ids"] and range_changed:
            full_class = "NORMALIZATION_ONLY_FULL_CHANGE"
        else:
            full_class = "OTHER"

        added_mapping = ";".join(
            f"{pred_id}:{pure['metrics'][pred_id]['gt_id']}:"
            f"{pure['metrics'][pred_id]['mask_median_depth']:.1f}"
            for pred_id in sorted(added_ids)
        )
        legacy_full = legacy["full_id"]
        pure_full = pure["full_id"]
        rows.append({
            "sequence": sequence,
            "frame_id": frame_id,
            "legacy_candidate_count": legacy["count"],
            "pure_candidate_count": pure["count"],
            "added_candidate_count": len(added_ids),
            "added_pred_gt_depth": added_mapping,
            "legacy_depth_min": legacy_min,
            "legacy_depth_max": legacy_max,
            "legacy_depth_range": legacy_max - legacy_min,
            "pure_depth_min": pure_min,
            "pure_depth_max": pure_max,
            "pure_depth_range": pure_max - pure_min,
            "depth_min_changed": min_changed,
            "depth_max_changed": max_changed,
            "depth_range_changed": range_changed,
            "legacy_nearest_set": serialize_ids(legacy["nearest_ids"]),
            "pure_nearest_set": serialize_ids(pure["nearest_ids"]),
            "nearest_changed": nearest_changed,
            "nearest_change_class": nearest_class,
            "added_candidate_is_pure_nearest": serialize_ids(added_nearest_ids),
            "legacy_full_pred": legacy_full,
            "pure_full_pred": pure_full,
            "full_changed": full_changed,
            "full_change_class": full_class,
            "added_candidate_is_pure_full": pure_full if added_is_full else "",
            "legacy_full_legacy_proximity": metric_value(
                legacy, legacy_full, "proximity"
            ),
            "legacy_full_legacy_score": metric_value(
                legacy, legacy_full, "full_score"
            ),
            "legacy_full_pure_proximity": metric_value(
                pure, legacy_full, "proximity"
            ),
            "legacy_full_pure_score": metric_value(
                pure, legacy_full, "full_score"
            ),
            "pure_full_legacy_proximity": metric_value(
                legacy, pure_full, "proximity"
            ),
            "pure_full_legacy_score": metric_value(
                legacy, pure_full, "full_score"
            ),
            "pure_full_pure_proximity": metric_value(
                pure, pure_full, "proximity"
            ),
            "pure_full_pure_score": metric_value(
                pure, pure_full, "full_score"
            ),
        })
    return rows


def write_output(rows):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_normalization_only(rows):
    selected = [
        row for row in rows
        if row["full_change_class"] == "NORMALIZATION_ONLY_FULL_CHANGE"
    ]
    print("\n[NORMALIZATION_ONLY_FULL_CHANGE DETAILS]")
    if not selected:
        print("(none)")
        return
    for row in selected:
        print(
            f"{row['sequence']}/{row['frame_id']} "
            f"added={row['added_pred_gt_depth']} "
            f"range L=[{row['legacy_depth_min']:.1f},"
            f"{row['legacy_depth_max']:.1f}] "
            f"P=[{row['pure_depth_min']:.1f},"
            f"{row['pure_depth_max']:.1f}] "
            f"Full {row['legacy_full_pred']}->{row['pure_full_pred']}"
        )
        print(
            f"  legacy-Full pred {row['legacy_full_pred']}: "
            f"proximity L={row['legacy_full_legacy_proximity']:.6f}, "
            f"P={row['legacy_full_pure_proximity']:.6f}; "
            f"score L={row['legacy_full_legacy_score']:.6f}, "
            f"P={row['legacy_full_pure_score']:.6f}"
        )
        print(
            f"  pure-Full pred {row['pure_full_pred']}: "
            f"proximity L={row['pure_full_legacy_proximity']:.6f}, "
            f"P={row['pure_full_pure_proximity']:.6f}; "
            f"score L={row['pure_full_legacy_score']:.6f}, "
            f"P={row['pure_full_pure_score']:.6f}"
        )


def validate_and_print(rows):
    affected = len(rows)
    added = sum(row["added_candidate_count"] for row in rows)
    depth_changed = sum(row["depth_range_changed"] for row in rows)
    nearest_new = sum(
        row["nearest_change_class"] == "NEW_CANDIDATE_NEAREST"
        for row in rows
    )
    full_changed = sum(row["full_changed"] for row in rows)
    direct = sum(
        row["full_change_class"] == "DIRECT_NEW_FULL" for row in rows
    )
    normalization_only = sum(
        row["full_change_class"] == "NORMALIZATION_ONLY_FULL_CHANGE"
        for row in rows
    )
    other = sum(row["full_change_class"] == "OTHER" for row in rows)
    unchanged = sum(
        row["full_change_class"] == "FULL_UNCHANGED" for row in rows
    )
    if (
        affected != EXPECTED_AFFECTED_FRAMES
        or added != EXPECTED_ADDED_CANDIDATES
        or nearest_new != EXPECTED_NEAREST_CHANGED
        or full_changed != EXPECTED_FULL_CHANGED
        or direct + normalization_only + other != full_changed
        or unchanged + full_changed != affected
    ):
        raise RuntimeError(
            "Final aggregation mismatch: "
            f"affected={affected}, added={added}, nearest_new={nearest_new}, "
            f"full_changed={full_changed}, direct={direct}, "
            f"normalization_only={normalization_only}, other={other}, "
            f"unchanged={unchanged}"
        )

    print("[PROXIMITY NORMALIZATION IMPACT]")
    print(f"Affected frames: {affected}")
    print(f"Depth range changed: {depth_changed}")
    print(f"Nearest changed by new candidate: {nearest_new}")
    print(f"Full changed total: {full_changed}")
    print(f"- DIRECT_NEW_FULL: {direct}")
    print(f"- NORMALIZATION_ONLY_FULL_CHANGE: {normalization_only}")
    print(f"- Other: {other}")
    print_normalization_only(rows)

    label = (
        "isolated (1-2 cases)" if normalization_only <= 2
        else "repeated across multiple frames"
    )
    print("\n[THREE ANSWERS]")
    print(f"1. Direct new Full: {direct}/{full_changed} frames")
    print(
        "2. Normalization-only existing-candidate rank change: "
        f"{normalization_only}/{full_changed} frames"
    )
    print(f"3. Pattern: {label} ({normalization_only} frames)")
    print(f"\nCSV: {os.path.abspath(OUTPUT_CSV)}")


def main():
    reference_rows = load_affected_reference_rows()
    rows = audit_frames(reference_rows)
    validate_and_print(rows)
    write_output(rows)


if __name__ == "__main__":
    main()
