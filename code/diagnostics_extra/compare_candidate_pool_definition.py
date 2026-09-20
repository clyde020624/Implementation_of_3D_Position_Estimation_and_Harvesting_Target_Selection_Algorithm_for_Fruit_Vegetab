"""Reproduce legacy and pure GT-mask-depth candidate pool definitions only.

No Top-1, nearest, disagreement, ablation, trade-off, or sensitivity result is
computed. Existing research code and result files are read-only inputs.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os

import numpy as np

from data_loader import clean_depth, load_bupst20_annotation, load_depth
from load_detections import attach_mask_from_pkl, load_detections_csv
from localization import depth_from_bbox_center
from mask_depth_reanalysis import (
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    get_mask_median_depth,
    prepare_candidates,
)
from priority import score_center, score_visibility


DATA_DIR = "dataset_bulk"
DETECTION_CSV = "eval_detections_fixed.csv"
REFERENCE_FRAME_CSV = "mask_depth_reanalysis_results.csv"
OUTPUT_CSV = "candidate_pool_definition_audit.csv"

EXPECTED_FRAMES = 364
EXPECTED_MATCHED = 2729
EXPECTED_LEGACY = 2644
EXPECTED_PURE = 2728
EXPECTED_ADDED = 84
EXPECTED_AFFECTED_FRAMES = 69


def bbox_text(bbox):
    return json.dumps([round(float(v), 6) for v in bbox], separators=(",", ":"))


def finite_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def reference_frames():
    with open(REFERENCE_FRAME_CSV, newline="", encoding="utf-8-sig") as handle:
        frames = {
            (row["sequence_id"], row["frame_id"])
            for row in csv.DictReader(handle)
        }
    if len(frames) != EXPECTED_FRAMES:
        raise RuntimeError(
            f"Reference frame set has {len(frames)}, expected {EXPECTED_FRAMES}"
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
            f"Missing input for {sequence}/{frame_id}: "
            f"depth={depth_paths}, annotation={annotation_path}"
        )
    return depth_paths[0], annotation_path


def bbox_depth_statistics(bbox, depth):
    x1, y1, x2, y2 = bbox
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    height, width = depth.shape
    region = depth[
        max(0, cy - 2):min(height, cy + 3),
        max(0, cx - 2):min(width, cx + 3),
    ]
    valid = region[region > 0]
    helper_depth, _ = depth_from_bbox_center(bbox, depth, region_half=2)
    direct_depth = None if valid.size == 0 else float(np.median(valid))
    if helper_depth != direct_depth:
        raise RuntimeError(
            f"bbox-depth mismatch: helper={helper_depth}, direct={direct_depth}"
        )
    return int(valid.size), helper_depth


def mask_depth_statistics(mask, depth):
    if mask is None:
        return 0, 0, None, False
    mask = np.asarray(mask, dtype=bool)
    mask_pixels = int(np.count_nonzero(mask))
    if mask.shape != depth.shape:
        return mask_pixels, 0, None, False
    valid = depth[mask]
    valid = valid[valid > 0]
    helper_depth = get_mask_median_depth(mask, depth)
    direct_depth = None if valid.size == 0 else float(np.median(valid))
    if helper_depth != direct_depth:
        raise RuntimeError(
            f"mask-depth mismatch: helper={helper_depth}, direct={direct_depth}"
        )
    return mask_pixels, int(valid.size), helper_depth, True


def component_validity(prediction, image_size):
    visibility_valid = False
    center_valid = False
    confidence_valid = finite_number(prediction.get("confidence"))
    try:
        visibility = float(score_visibility(prediction))
        visibility_valid = math.isfinite(visibility)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        visibility = None
    try:
        center = float(
            score_center(
                prediction["bbox"],
                (image_size[0] / 2, image_size[1] / 2),
            )
        )
        center_valid = math.isfinite(center)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        center = None
    return (
        visibility_valid, visibility,
        confidence_valid, center_valid, center,
    )


FIELDS = [
    "sequence", "frame_id", "pred_id", "gt_id", "confidence", "iou",
    "pred_bbox", "bbox_valid_pixels", "bbox_median_depth",
    "bbox_depth_valid", "mask_pixel_count", "mask_valid_pixels",
    "mask_median_depth", "mask_depth_valid", "mask_shape_valid",
    "visibility_valid", "visibility", "confidence_valid",
    "center_score_valid", "center_score", "legacy_candidate",
    "pure_mask_candidate", "newly_added_in_pure_pool",
]


def audit():
    detections = load_detections_csv(DETECTION_CSV)
    reference = reference_frames()
    rows = []
    matched_total = 0
    legacy_total = 0
    pure_total = 0
    legacy_frame_set = set()
    pure_frame_set = set()
    affected_frames = set()

    for sequence, frame_id in sorted(reference):
        info = detections.get((sequence, frame_id))
        if info is None:
            raise RuntimeError(f"Detection frame missing: {sequence}/{frame_id}")
        depth_path, annotation_path = input_paths(sequence, frame_id)
        depth = clean_depth(load_depth(depth_path))
        ground_truth = load_bupst20_annotation(annotation_path)
        gt_by_id = {gt["id"]: gt for gt in ground_truth}
        predictions = [
            dict(item) for item in info["objects"]
            if item["confidence"] >= CONF_THRESHOLD
        ]
        image_size = info["img_size"] if info["img_size"] is not None else (720, 1280)

        matched_predictions = [dict(item) for item in predictions]
        attach_mask_from_pkl(
            matched_predictions, ground_truth, IOU_THRESHOLD
        )
        matched = [
            item for item in matched_predictions
            if item.get("gt_matched", False)
        ]
        matched_total += len(matched)

        legacy_candidates = prepare_candidates(
            [dict(item) for item in predictions], ground_truth, depth
        )
        legacy_ids = {item["id"] for item in legacy_candidates}
        legacy_total += len(legacy_candidates)
        if len(legacy_candidates) >= 2:
            legacy_frame_set.add((sequence, frame_id))

        pure_ids = set()
        frame_rows = []
        for prediction in matched:
            gt_id = prediction["matched_gt_id"]
            gt = gt_by_id.get(gt_id)
            if gt is None:
                raise RuntimeError(
                    f"Matched GT missing: {sequence}/{frame_id}/GT{gt_id}"
                )
            bbox_valid_pixels, bbox_median = bbox_depth_statistics(
                prediction["bbox"], depth
            )
            (
                mask_pixels, mask_valid_pixels,
                mask_median, mask_shape_valid,
            ) = mask_depth_statistics(gt.get("instance_mask"), depth)
            (
                visibility_valid, visibility,
                confidence_valid, center_valid, center,
            ) = component_validity(prediction, image_size)

            pure_candidate = (
                mask_median is not None
                and visibility_valid
                and confidence_valid
                and center_valid
            )
            if pure_candidate:
                pure_ids.add(prediction["id"])

            frame_rows.append({
                "sequence": sequence,
                "frame_id": frame_id,
                "pred_id": prediction["id"],
                "gt_id": gt_id,
                "confidence": prediction["confidence"],
                "iou": prediction["match_iou"],
                "pred_bbox": bbox_text(prediction["bbox"]),
                "bbox_valid_pixels": bbox_valid_pixels,
                "bbox_median_depth": bbox_median,
                "bbox_depth_valid": bbox_median is not None,
                "mask_pixel_count": mask_pixels,
                "mask_valid_pixels": mask_valid_pixels,
                "mask_median_depth": mask_median,
                "mask_depth_valid": mask_median is not None,
                "mask_shape_valid": mask_shape_valid,
                "visibility_valid": visibility_valid,
                "visibility": visibility,
                "confidence_valid": confidence_valid,
                "center_score_valid": center_valid,
                "center_score": center,
                "legacy_candidate": prediction["id"] in legacy_ids,
                "pure_mask_candidate": pure_candidate,
                "newly_added_in_pure_pool": (
                    pure_candidate and prediction["id"] not in legacy_ids
                ),
            })

        pure_total += len(pure_ids)
        if len(pure_ids) >= 2:
            pure_frame_set.add((sequence, frame_id))
        added_ids = pure_ids - legacy_ids
        if added_ids:
            affected_frames.add((sequence, frame_id))
        if not legacy_ids.issubset(pure_ids):
            raise RuntimeError(
                f"Legacy pool is not a subset of pure pool: {sequence}/{frame_id}"
            )
        rows.extend(frame_rows)

    added_total = sum(row["newly_added_in_pure_pool"] for row in rows)
    checks = (
        ("matched detections", matched_total, EXPECTED_MATCHED),
        ("legacy candidates", legacy_total, EXPECTED_LEGACY),
        ("pure mask candidates", pure_total, EXPECTED_PURE),
        ("added candidates", added_total, EXPECTED_ADDED),
        ("affected frames", len(affected_frames), EXPECTED_AFFECTED_FRAMES),
        ("legacy frame set", len(legacy_frame_set), EXPECTED_FRAMES),
        ("pure frame set", len(pure_frame_set), EXPECTED_FRAMES),
    )
    failures = [
        (name, actual, expected)
        for name, actual, expected in checks if actual != expected
    ]
    if legacy_frame_set != reference:
        failures.append(
            ("legacy frame IDs exact", len(legacy_frame_set ^ reference), 0)
        )
    if pure_frame_set != reference:
        failures.append(
            ("pure frame IDs exact", len(pure_frame_set ^ reference), 0)
        )
    if failures:
        print("[VALIDATION FAILED]")
        for name, actual, expected in failures:
            print(f"{name}: actual={actual}, expected={expected}")
        raise RuntimeError("Candidate-pool validation failed; CSV output stopped")

    return {
        "rows": rows,
        "matched_total": matched_total,
        "legacy_total": legacy_total,
        "pure_total": pure_total,
        "added_total": added_total,
        "affected_frames": affected_frames,
        "legacy_frame_set": legacy_frame_set,
        "pure_frame_set": pure_frame_set,
        "reference": reference,
    }


def write_output(rows):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(result):
    print("[REFERENCE FRAME SET]")
    print(f"Existing frame IDs: {len(result['reference'])}")
    print(f"Matched detections: {result['matched_total']}")

    print("\n[A] LEGACY FOLLOW-UP POOL")
    print(f"Expected candidate objects: {EXPECTED_LEGACY}")
    print(f"Reproduced candidate objects: {result['legacy_total']}")
    print(f"Frames with >=2 candidates: {len(result['legacy_frame_set'])}")
    print(
        "Frame IDs exactly match existing 364: "
        f"{result['legacy_frame_set'] == result['reference']}"
    )

    print("\n[B] PURE MASK-DEPTH POOL")
    print(f"Expected candidate objects: {EXPECTED_PURE}")
    print(f"Reproduced candidate objects: {result['pure_total']}")
    print(f"Frames with >=2 candidates: {len(result['pure_frame_set'])}")
    print(
        "Frame IDs exactly match existing 364: "
        f"{result['pure_frame_set'] == result['reference']}"
    )

    print("\n[POOL DIFFERENCE]")
    print(f"Expected newly added candidates: {EXPECTED_ADDED}")
    print(f"Reproduced newly added candidates: {result['added_total']}")
    print(f"Expected affected frames: {EXPECTED_AFFECTED_FRAMES}")
    print(f"Reproduced affected frames: {len(result['affected_frames'])}")
    print("Validation: PASS")
    print(f"\nCSV: {os.path.abspath(OUTPUT_CSV)}")
    print("Nearest / Full / disagreement computed: NO")


def main():
    result = audit()
    write_output(result["rows"])
    print_summary(result)


if __name__ == "__main__":
    main()
