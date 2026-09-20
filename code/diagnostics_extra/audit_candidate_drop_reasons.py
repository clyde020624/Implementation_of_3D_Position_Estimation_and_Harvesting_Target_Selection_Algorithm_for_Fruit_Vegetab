"""Trace why 2,729 GT-matched detections become 2,644 candidates.

Independent diagnostic only. It reuses the exact 364-frame set, matching,
depth cleaning, and candidate builder from the existing research pipeline.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

from data_loader import clean_depth, load_bupst20_annotation, load_depth
from load_detections import attach_mask_from_pkl, load_detections_csv
from localization import depth_from_bbox_center
from mask_depth_reanalysis import get_mask_median_depth, prepare_candidates
from priority import score_center, score_visibility


DATA_DIR = "dataset_bulk"
DETECTION_CSV = "eval_detections_fixed.csv"
REFERENCE_FRAME_CSV = "mask_depth_reanalysis_results.csv"
CONF_THRESHOLD = 0.30
IOU_THRESHOLD = 0.50
EXPECTED_FRAMES = 364
EXPECTED_MATCHED = 2729
EXPECTED_CANDIDATES = 2644

OUTPUT_AUDIT = "candidate_drop_audit.csv"
OUTPUT_SUMMARY = "candidate_drop_summary.csv"


def bbox_text(bbox):
    return json.dumps([round(float(v), 6) for v in bbox], separators=(",", ":"))


def finite_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def bbox_depth_stats(bbox, depth):
    x1, y1, x2, y2 = bbox
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    height, width = depth.shape
    ys, ye = max(0, cy - 2), min(height, cy + 3)
    xs, xe = max(0, cx - 2), min(width, cx + 3)
    valid = depth[ys:ye, xs:xe]
    valid = valid[valid > 0]
    median, center = depth_from_bbox_center(bbox, depth, region_half=2)
    calculated = None if valid.size == 0 else float(np.median(valid))
    if median != calculated:
        raise RuntimeError(
            f"bbox depth implementation mismatch: helper={median}, direct={calculated}"
        )
    return int(valid.size), median, center


def mask_depth_stats(mask, depth):
    if mask is None:
        return 0, 0, None, False
    mask = np.asarray(mask, dtype=bool)
    mask_pixels = int(np.count_nonzero(mask))
    if mask.shape != depth.shape:
        return mask_pixels, 0, None, False
    valid = depth[mask]
    valid = valid[valid > 0]
    median = get_mask_median_depth(mask, depth)
    calculated = None if valid.size == 0 else float(np.median(valid))
    if median != calculated:
        raise RuntimeError(
            f"mask depth implementation mismatch: helper={median}, direct={calculated}"
        )
    return mask_pixels, int(valid.size), median, True


def reference_frames():
    with open(REFERENCE_FRAME_CSV, newline="", encoding="utf-8-sig") as handle:
        frames = {
            (row["sequence_id"], row["frame_id"])
            for row in csv.DictReader(handle)
        }
    if len(frames) != EXPECTED_FRAMES:
        raise RuntimeError(
            f"Reference set has {len(frames)} frames, expected {EXPECTED_FRAMES}"
        )
    return frames


def input_paths(sequence, frame_id):
    depth_paths = glob.glob(
        os.path.join(DATA_DIR, "depth", sequence, frame_id + ".tif*")
    )
    annotation_path = os.path.join(
        DATA_DIR, "annotations", sequence, frame_id + ".pkl"
    )
    if not depth_paths or not os.path.isfile(annotation_path):
        raise FileNotFoundError(
            f"Missing frame input for {sequence}/{frame_id}: "
            f"depth={depth_paths}, annotation={annotation_path}"
        )
    return depth_paths[0], annotation_path


def required_field_failures(pred, gt):
    failures = []
    bbox = pred.get("bbox")
    if (
        bbox is None or len(bbox) < 4
        or not all(finite_number(value) for value in bbox[:4])
    ):
        failures.append("MISSING_OR_INVALID_BBOX")
    if not finite_number(pred.get("confidence")):
        failures.append("MISSING_OR_INVALID_CONFIDENCE")
    if gt is None:
        failures.append("MISSING_MATCHED_GT")
    return failures


def reproduce_matched_count(detections, frames):
    matched_count = 0
    for sequence, frame_id in sorted(frames):
        info = detections.get((sequence, frame_id))
        if info is None:
            raise RuntimeError(f"Detection frame missing: {sequence}/{frame_id}")
        _, annotation_path = input_paths(sequence, frame_id)
        ground_truth = load_bupst20_annotation(annotation_path)
        high = [
            dict(item) for item in info["objects"]
            if item["confidence"] >= CONF_THRESHOLD
        ]
        attach_mask_from_pkl(high, ground_truth, IOU_THRESHOLD)
        matched_count += sum(item.get("gt_matched", False) for item in high)
    return matched_count


def trace_objects(detections, frames):
    rows = []
    actual_candidate_count = 0
    old_candidate_count = 0
    mask_only_possible_count = 0

    for sequence, frame_id in sorted(frames):
        info = detections[(sequence, frame_id)]
        depth_path, annotation_path = input_paths(sequence, frame_id)
        depth = clean_depth(load_depth(depth_path))
        ground_truth = load_bupst20_annotation(annotation_path)
        gt_by_id = {gt["id"]: gt for gt in ground_truth}
        high = [
            dict(item) for item in info["objects"]
            if item["confidence"] >= CONF_THRESHOLD
        ]
        attach_mask_from_pkl(high, ground_truth, IOU_THRESHOLD)
        matched = [item for item in high if item.get("gt_matched", False)]

        actual_candidates = prepare_candidates(
            [dict(item) for item in high], ground_truth, depth
        )
        actual_candidate_ids = {item["id"] for item in actual_candidates}
        actual_candidate_count += len(actual_candidates)

        image_size = info["img_size"] if info["img_size"] is not None else (720, 1280)
        image_center = (image_size[0] / 2, image_size[1] / 2)

        for pred in matched:
            gt_id = pred.get("matched_gt_id")
            gt = gt_by_id.get(gt_id)
            field_failures = required_field_failures(pred, gt)

            bbox_valid_pixels = 0
            bbox_median = None
            center = (None, None)
            if "MISSING_OR_INVALID_BBOX" not in field_failures:
                bbox_valid_pixels, bbox_median, center = bbox_depth_stats(
                    pred["bbox"], depth
                )
            bbox_depth_valid = bbox_median is not None

            mask_pixels = 0
            mask_valid_pixels = 0
            mask_median = None
            mask_shape_valid = False
            if gt is not None:
                (
                    mask_pixels, mask_valid_pixels,
                    mask_median, mask_shape_valid,
                ) = mask_depth_stats(gt.get("instance_mask"), depth)
            mask_depth_valid = mask_median is not None

            visibility_value = None
            visibility_valid = False
            try:
                visibility_value = float(score_visibility(pred))
                visibility_valid = math.isfinite(visibility_value)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                pass

            center_value = None
            center_valid = False
            try:
                center_value = float(score_center(pred["bbox"], image_center))
                center_valid = math.isfinite(center_value)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                pass

            confidence_valid = finite_number(pred.get("confidence"))
            bbox_proximity_valid = bbox_depth_valid
            mask_proximity_valid = mask_depth_valid

            old_candidate = (
                bbox_depth_valid and visibility_valid
                and center_valid and confidence_valid and not field_failures
            )
            followup_candidate = pred["id"] in actual_candidate_ids
            mask_only_candidate_possible = (
                mask_depth_valid and visibility_valid
                and center_valid and confidence_valid and not field_failures
            )
            old_candidate_count += int(old_candidate)
            mask_only_possible_count += int(mask_only_candidate_possible)

            failed = list(field_failures)
            if not bbox_depth_valid:
                failed.append("INVALID_BBOX_CENTER_DEPTH")
            if not mask_shape_valid:
                failed.append("INVALID_GT_MASK_OR_SHAPE")
            if not mask_depth_valid:
                failed.append("INVALID_GT_MASK_DEPTH")
            if not visibility_valid:
                failed.append("INVALID_VISIBILITY")
            if not center_valid:
                failed.append("INVALID_CENTER_SCORE")
            if not confidence_valid:
                failed.append("INVALID_CONFIDENCE_SCORE")
            if not followup_candidate and not failed:
                failed.append("OTHER_PIPELINE_FILTER")

            if followup_candidate:
                primary_reason = "INCLUDED"
            else:
                priority = (
                    "MISSING_MATCHED_GT", "MISSING_OR_INVALID_BBOX",
                    "MISSING_OR_INVALID_CONFIDENCE",
                    "INVALID_BBOX_CENTER_DEPTH",
                    "INVALID_GT_MASK_OR_SHAPE", "INVALID_GT_MASK_DEPTH",
                    "INVALID_VISIBILITY", "INVALID_CENTER_SCORE",
                    "INVALID_CONFIDENCE_SCORE", "OTHER_PIPELINE_FILTER",
                )
                primary_reason = next(
                    (reason for reason in priority if reason in failed), "OTHER"
                )

            if not bbox_depth_valid and mask_depth_valid:
                critical_case = "A_BBOX_INVALID_MASK_VALID"
            elif bbox_depth_valid and not mask_depth_valid:
                critical_case = "B_BBOX_VALID_MASK_INVALID"
            elif not bbox_depth_valid and not mask_depth_valid:
                critical_case = "C_BOTH_INVALID"
            elif not followup_candidate:
                critical_case = "D_BOTH_VALID_BUT_EXCLUDED"
            else:
                critical_case = "BOTH_VALID_INCLUDED"

            rows.append({
                "sequence": sequence, "frame_id": frame_id,
                "pred_id": pred["id"], "gt_id": gt_id,
                "confidence": pred["confidence"], "iou": pred["match_iou"],
                "pred_bbox": bbox_text(pred["bbox"]),
                "bbox_center_x": center[0], "bbox_center_y": center[1],
                "bbox_depth_valid": bbox_depth_valid,
                "bbox_valid_pixels": bbox_valid_pixels,
                "bbox_median_depth": bbox_median,
                "mask_depth_valid": mask_depth_valid,
                "mask_pixel_count": mask_pixels,
                "mask_valid_pixels": mask_valid_pixels,
                "mask_median_depth": mask_median,
                "visibility_valid": visibility_valid,
                "visibility": visibility_value,
                "bbox_proximity_valid": bbox_proximity_valid,
                "mask_proximity_valid": mask_proximity_valid,
                "center_score_valid": center_valid,
                "center_score": center_value,
                "confidence_score_valid": confidence_valid,
                "old_candidate": old_candidate,
                "followup_candidate": followup_candidate,
                "mask_only_candidate_possible": mask_only_candidate_possible,
                "primary_drop_reason": primary_reason,
                "all_failed_conditions": ";".join(failed),
                "critical_case": critical_case,
            })

    return rows, old_candidate_count, actual_candidate_count, mask_only_possible_count


AUDIT_FIELDS = [
    "sequence", "frame_id", "pred_id", "gt_id", "confidence", "iou",
    "pred_bbox", "bbox_center_x", "bbox_center_y", "bbox_depth_valid",
    "bbox_valid_pixels", "bbox_median_depth", "mask_depth_valid",
    "mask_pixel_count", "mask_valid_pixels", "mask_median_depth",
    "visibility_valid", "visibility", "bbox_proximity_valid",
    "mask_proximity_valid", "center_score_valid", "center_score",
    "confidence_score_valid", "old_candidate", "followup_candidate",
    "mask_only_candidate_possible", "primary_drop_reason",
    "all_failed_conditions", "critical_case",
]


def build_summary(rows):
    dropped = [row for row in rows if not row["followup_candidate"]]
    summary = []

    primary_counts = Counter(row["primary_drop_reason"] for row in dropped)
    for reason, count in sorted(primary_counts.items()):
        frames = len({
            (row["sequence"], row["frame_id"])
            for row in dropped if row["primary_drop_reason"] == reason
        })
        summary.append({
            "reason": f"PRIMARY:{reason}", "count": count, "frames": frames
        })

    condition_names = sorted({
        condition
        for row in dropped
        for condition in row["all_failed_conditions"].split(";")
        if condition
    })
    for condition in condition_names:
        members = [
            row for row in dropped
            if condition in row["all_failed_conditions"].split(";")
        ]
        summary.append({
            "reason": f"CONDITION:{condition}",
            "count": len(members),
            "frames": len({(row["sequence"], row["frame_id"]) for row in members}),
        })

    for case in (
        "A_BBOX_INVALID_MASK_VALID", "B_BBOX_VALID_MASK_INVALID",
        "C_BOTH_INVALID", "D_BOTH_VALID_BUT_EXCLUDED",
    ):
        members = [row for row in rows if row["critical_case"] == case]
        summary.append({
            "reason": f"CRITICAL:{case}",
            "count": len(members),
            "frames": len({(row["sequence"], row["frame_id"]) for row in members}),
        })
    return summary


def write_results(rows, summary):
    with open(OUTPUT_AUDIT, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(OUTPUT_SUMMARY, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reason", "count", "frames"])
        writer.writeheader()
        writer.writerows(summary)


def count_case(rows, case):
    members = [row for row in rows if row["critical_case"] == case]
    frames = len({(row["sequence"], row["frame_id"]) for row in members})
    return len(members), frames


def print_report(
    reproduced_matched, rows, old_count, candidate_count, mask_only_count, summary
):
    dropped = [row for row in rows if not row["followup_candidate"]]
    primary = [
        item for item in summary if item["reason"].startswith("PRIMARY:")
    ]
    case_a = count_case(rows, "A_BBOX_INVALID_MASK_VALID")
    case_b = count_case(rows, "B_BBOX_VALID_MASK_INVALID")
    case_c = count_case(rows, "C_BOTH_INVALID")
    case_d = count_case(rows, "D_BOTH_VALID_BUT_EXCLUDED")
    verdict = "NEEDS_REVIEW" if case_a[0] > 0 or case_d[0] > 0 else "NO"

    print("[MATCHED SET]")
    print(f"Expected matched: {EXPECTED_MATCHED}")
    print(f"Reproduced matched: {reproduced_matched}")

    print("\n[CANDIDATE SET]")
    print(f"Expected candidates: {EXPECTED_CANDIDATES}")
    print(f"Reproduced candidates: {candidate_count}")
    print(f"Difference: {reproduced_matched - candidate_count}")
    print(f"Existing-study bbox-depth candidates: {old_count}")
    print(f"Mask-depth-valid candidates without bbox-depth dependency: {mask_only_count}")

    print("\n[DROP REASONS]")
    for item in primary:
        print(
            f"{item['reason'].removeprefix('PRIMARY:')}: "
            f"{item['count']} objects / {item['frames']} frames"
        )
    condition_items = [
        item for item in summary if item["reason"].startswith("CONDITION:")
    ]
    print("All failed conditions (non-exclusive):")
    for item in condition_items:
        print(
            f"- {item['reason'].removeprefix('CONDITION:')}: "
            f"{item['count']} objects / {item['frames']} frames"
        )

    print("\n[CRITICAL CASES]")
    print(f"A. bbox invalid / mask valid: {case_a[0]} objects / {case_a[1]} frames")
    print(f"B. bbox valid / mask invalid: {case_b[0]} objects / {case_b[1]} frames")
    print(f"C. both invalid: {case_c[0]} objects / {case_c[1]} frames")
    print(f"D. both valid but excluded: {case_d[0]} objects / {case_d[1]} frames")

    print("\n[INTERPRETATION]")
    print(
        f"1. The {len(dropped)} objects were dropped because the current follow-up "
        "prepare_candidates() checks bbox-center depth first and mask depth second. "
        f"Primary invalid-bbox drops: {sum(not row['bbox_depth_valid'] for row in dropped)}; "
        f"among them both depths invalid: {case_c[0]}."
    )
    print(
        "2. Does the follow-up candidate pool still depend on bbox-center depth "
        f"validity? {'YES' if case_a[0] > 0 else 'NO'}."
    )
    print(
        "3. Do bbox-invalid but mask-valid matched objects exist? "
        f"{'YES' if case_a[0] > 0 else 'NO'}."
    )
    print(
        f"4. Count: {case_a[0]} objects in {case_a[1]} frames."
    )
    print(
        f"5. Recalculation possibility for 118/364: {verdict}. "
        "The current result is reproducible for the existing shared candidate pool, "
        "but mask-valid matched objects are excluded solely by bbox-depth invalidity; "
        "whether Top-1/disagreement changes cannot be determined without a separate "
        "recalculation, which this audit does not perform."
    )
    print(f"\nAudit CSV: {os.path.abspath(OUTPUT_AUDIT)}")
    print(f"Summary CSV: {os.path.abspath(OUTPUT_SUMMARY)}")


def main():
    detections = load_detections_csv(DETECTION_CSV)
    frames = reference_frames()
    reproduced_matched = reproduce_matched_count(detections, frames)
    if reproduced_matched != EXPECTED_MATCHED:
        raise RuntimeError(
            f"Matched-set reproduction failed: expected {EXPECTED_MATCHED}, "
            f"got {reproduced_matched}. Candidate tracing stopped."
        )

    rows, old_count, candidate_count, mask_only_count = trace_objects(
        detections, frames
    )
    if len(rows) != EXPECTED_MATCHED:
        raise RuntimeError(
            f"Object trace count mismatch: expected {EXPECTED_MATCHED}, got {len(rows)}"
        )
    if candidate_count != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"Candidate-set reproduction failed: expected {EXPECTED_CANDIDATES}, "
            f"got {candidate_count}. CSV output stopped."
        )

    summary = build_summary(rows)
    write_results(rows, summary)
    print_report(
        reproduced_matched, rows, old_count,
        candidate_count, mask_only_count, summary,
    )


if __name__ == "__main__":
    main()
