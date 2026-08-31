"""Compare Top-1 selection for legacy and pure GT-mask-depth pools only."""

from __future__ import annotations

import csv
import glob
import math
import os

from data_loader import clean_depth, load_bupst20_annotation, load_depth
from load_detections import attach_mask_from_pkl, load_detections_csv
from mask_depth_reanalysis import (
    CONF_THRESHOLD,
    FULL_WEIGHTS,
    IOU_THRESHOLD,
    MODE,
    get_mask_median_depth,
    prepare_candidates,
    select_top1_precomputed,
)
from priority import score_center, score_visibility


DATA_DIR = "dataset_bulk"
DETECTION_CSV = "eval_detections_fixed.csv"
REFERENCE_FRAME_CSV = "mask_depth_reanalysis_results.csv"
OUTPUT_CSV = "candidate_pool_top1_comparison.csv"

EXPECTED_FRAMES = 364
EXPECTED_LEGACY_CANDIDATES = 2644
EXPECTED_LEGACY_DISAGREEMENT = 118
EXPECTED_PURE_CANDIDATES = 2728
EXPECTED_AFFECTED_FRAMES = 69


FIELDS = [
    "sequence", "frame_id", "legacy_candidate_count",
    "pure_candidate_count", "added_candidate_count",
    "legacy_nearest_set", "pure_nearest_set",
    "legacy_full_pred", "pure_full_pred",
    "legacy_disagreement", "pure_disagreement",
    "nearest_changed", "full_changed", "disagreement_status_changed",
]


def pct(count, total):
    return 100.0 * count / total if total else 0.0


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


def finite_components(candidate, image_size):
    try:
        values = (
            float(score_visibility(candidate)),
            float(candidate["confidence"]),
            float(
                score_center(
                    candidate["bbox"],
                    (image_size[0] / 2, image_size[1] / 2),
                )
            ),
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return all(math.isfinite(value) for value in values)


def select_pool(candidates, image_size):
    if len(candidates) < 2:
        raise RuntimeError("A comparison frame has fewer than two candidates")
    full, _ = select_top1_precomputed(
        candidates, "mask_depth", FULL_WEIGHTS, image_size, MODE
    )
    minimum = min(candidate["mask_depth"] for candidate in candidates)
    nearest_ids = {
        candidate["id"]
        for candidate in candidates
        if candidate["mask_depth"] == minimum
    }
    return {
        "count": len(candidates),
        "ids": {candidate["id"] for candidate in candidates},
        "nearest_ids": nearest_ids,
        "full_id": full["id"],
        "disagreement": full["id"] not in nearest_ids,
    }


def load_frame_inputs(detections, sequence, frame_id):
    info = detections.get((sequence, frame_id))
    if info is None:
        raise RuntimeError(f"Detection frame missing: {sequence}/{frame_id}")
    depth_path, annotation_path = input_paths(sequence, frame_id)
    depth = clean_depth(load_depth(depth_path))
    ground_truth = load_bupst20_annotation(annotation_path)
    predictions = [
        dict(item) for item in info["objects"]
        if item["confidence"] >= CONF_THRESHOLD
    ]
    image_size = info["img_size"] if info["img_size"] is not None else (720, 1280)
    return predictions, ground_truth, depth, image_size


def reproduce_legacy(detections, frames):
    results = {}
    candidate_total = 0
    for sequence, frame_id in sorted(frames):
        predictions, ground_truth, depth, image_size = load_frame_inputs(
            detections, sequence, frame_id
        )
        candidates = prepare_candidates(
            [dict(item) for item in predictions], ground_truth, depth
        )
        selected = select_pool(candidates, image_size)
        candidate_total += selected["count"]
        results[(sequence, frame_id)] = selected

    disagreement = sum(
        result["disagreement"] for result in results.values()
    )
    failures = []
    if len(results) != EXPECTED_FRAMES:
        failures.append(
            f"frames actual={len(results)}, expected={EXPECTED_FRAMES}"
        )
    if candidate_total != EXPECTED_LEGACY_CANDIDATES:
        failures.append(
            "legacy candidates "
            f"actual={candidate_total}, expected={EXPECTED_LEGACY_CANDIDATES}"
        )
    if disagreement != EXPECTED_LEGACY_DISAGREEMENT:
        failures.append(
            "legacy disagreement "
            f"actual={disagreement}, expected={EXPECTED_LEGACY_DISAGREEMENT}"
        )
    if failures:
        print("[LEGACY VALIDATION FAILED]")
        for failure in failures:
            print(failure)
        raise RuntimeError(
            "Legacy 2644 and 118/364 were not reproduced; pure calculation stopped"
        )
    return results, candidate_total, disagreement


def build_pure_candidates(predictions, ground_truth, depth, image_size):
    attached = [dict(item) for item in predictions]
    attach_mask_from_pkl(attached, ground_truth, IOU_THRESHOLD)
    gt_by_id = {gt["id"]: gt for gt in ground_truth}
    candidates = []
    for prediction in attached:
        if not prediction.get("gt_matched", False):
            continue
        gt = gt_by_id.get(prediction.get("matched_gt_id"))
        if gt is None:
            continue
        mask_depth = get_mask_median_depth(gt.get("instance_mask"), depth)
        if mask_depth is None:
            continue
        candidate = dict(prediction)
        candidate["mask_depth"] = float(mask_depth)
        if finite_components(candidate, image_size):
            candidates.append(candidate)
    return candidates


def compare_pure(detections, frames, legacy_results):
    rows = []
    pure_candidate_total = 0
    affected_frames = 0

    for sequence, frame_id in sorted(frames):
        predictions, ground_truth, depth, image_size = load_frame_inputs(
            detections, sequence, frame_id
        )
        pure_candidates = build_pure_candidates(
            predictions, ground_truth, depth, image_size
        )
        pure = select_pool(pure_candidates, image_size)
        legacy = legacy_results[(sequence, frame_id)]
        if not legacy["ids"].issubset(pure["ids"]):
            raise RuntimeError(
                f"Legacy pool is not a pure-pool subset: {sequence}/{frame_id}"
            )
        added_count = len(pure["ids"] - legacy["ids"])
        if added_count:
            affected_frames += 1
        pure_candidate_total += pure["count"]
        nearest_changed = legacy["nearest_ids"] != pure["nearest_ids"]
        full_changed = legacy["full_id"] != pure["full_id"]
        status_changed = legacy["disagreement"] != pure["disagreement"]
        rows.append({
            "sequence": sequence,
            "frame_id": frame_id,
            "legacy_candidate_count": legacy["count"],
            "pure_candidate_count": pure["count"],
            "added_candidate_count": added_count,
            "legacy_nearest_set": ";".join(
                str(value) for value in sorted(legacy["nearest_ids"])
            ),
            "pure_nearest_set": ";".join(
                str(value) for value in sorted(pure["nearest_ids"])
            ),
            "legacy_full_pred": legacy["full_id"],
            "pure_full_pred": pure["full_id"],
            "legacy_disagreement": legacy["disagreement"],
            "pure_disagreement": pure["disagreement"],
            "nearest_changed": nearest_changed,
            "full_changed": full_changed,
            "disagreement_status_changed": status_changed,
        })

    if pure_candidate_total != EXPECTED_PURE_CANDIDATES:
        raise RuntimeError(
            "Pure candidate validation failed: "
            f"actual={pure_candidate_total}, expected={EXPECTED_PURE_CANDIDATES}"
        )
    if affected_frames != EXPECTED_AFFECTED_FRAMES:
        raise RuntimeError(
            "Affected-frame validation failed: "
            f"actual={affected_frames}, expected={EXPECTED_AFFECTED_FRAMES}"
        )
    return rows, pure_candidate_total


def write_output(rows):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    rows, legacy_candidates, legacy_disagreement, pure_candidates
):
    total = len(rows)
    pure_disagreement = sum(row["pure_disagreement"] for row in rows)
    affected = [
        row for row in rows if row["added_candidate_count"] > 0
    ]
    nearest_changed = sum(row["nearest_changed"] for row in rows)
    full_changed = sum(row["full_changed"] for row in rows)
    either_changed = sum(
        row["nearest_changed"] or row["full_changed"] for row in rows
    )
    agreement_to_disagreement = sum(
        not row["legacy_disagreement"] and row["pure_disagreement"]
        for row in rows
    )
    disagreement_to_agreement = sum(
        row["legacy_disagreement"] and not row["pure_disagreement"]
        for row in rows
    )
    no_top1_change_affected = sum(
        not row["nearest_changed"] and not row["full_changed"]
        for row in affected
    )

    print("[LEGACY]")
    print(f"Candidates: {legacy_candidates}")
    print(f"Disagreement: {legacy_disagreement}/{total}")
    print(f"Rate: {pct(legacy_disagreement, total):.1f}%")

    print("\n[PURE MASK]")
    print(f"Candidates: {pure_candidates}")
    print(f"Disagreement: {pure_disagreement}/{total}")
    print(f"Rate: {pct(pure_disagreement, total):.1f}%")

    print("\n[TOP1 IMPACT]")
    print(f"Frames with added candidates: {len(affected)}")
    print(f"Nearest changed: {nearest_changed}")
    print(f"Full changed: {full_changed}")
    print(f"Either changed: {either_changed}")
    print(f"Agreement -> Disagreement: {agreement_to_disagreement}")
    print(f"Disagreement -> Agreement: {disagreement_to_agreement}")
    print(
        "No Top-1 change among affected 69 frames: "
        f"{no_top1_change_affected}"
    )
    print(f"\nCSV: {os.path.abspath(OUTPUT_CSV)}")


def main():
    detections = load_detections_csv(DETECTION_CSV)
    frames = reference_frames()
    legacy_results, legacy_candidates, legacy_disagreement = (
        reproduce_legacy(detections, frames)
    )
    rows, pure_candidates = compare_pure(
        detections, frames, legacy_results
    )
    write_output(rows)
    print_summary(
        rows, legacy_candidates, legacy_disagreement, pure_candidates
    )


if __name__ == "__main__":
    main()
