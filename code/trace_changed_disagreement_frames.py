"""Trace the seven frames whose tie-aware agreement status changed."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter

from compare_candidate_pool_top1 import (
    build_pure_candidates,
    input_paths,
)
from data_loader import clean_depth, load_bupst20_annotation, load_depth
from load_detections import load_detections_csv
from mask_depth_reanalysis import (
    CONF_THRESHOLD,
    FULL_WEIGHTS,
    MODE,
    prepare_candidates,
    select_top1_precomputed,
)
from priority import normalize_proximity, score_center, score_visibility


COMPARISON_CSV = "candidate_pool_top1_comparison.csv"
DETECTION_CSV = "eval_detections_fixed.csv"
OUTPUT_CSV = "changed_disagreement_frames.csv"
EXPECTED_CHANGED = 7

FIELDS = [
    "sequence", "frame_id", "legacy_candidate_count",
    "pure_candidate_count", "added_candidate_count",
    "added_pred_ids", "added_gt_ids", "added_pred_gt_pairs",
    "legacy_nearest_set", "pure_nearest_set",
    "legacy_full_pred", "pure_full_pred",
    "legacy_disagreement", "pure_disagreement",
    "nearest_changed", "full_changed", "change_direction",
    "cause_classification", "cause_detail",
    "added_used_as_nearest", "added_used_as_full",
    "influential_pred_ids", "candidate_metrics_json", "tie_check_pass",
]


def as_bool(value):
    return str(value).strip().lower() == "true"


def parse_id_set(value):
    text = str(value).strip()
    return {int(item) for item in text.split(";")} if text else set()


def serialize_ids(values):
    return ";".join(str(value) for value in sorted(values))


def load_changed_reference_rows():
    with open(COMPARISON_CSV, newline="", encoding="utf-8-sig") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if as_bool(row["disagreement_status_changed"])
        ]
    a_to_d = sum(
        not as_bool(row["legacy_disagreement"])
        and as_bool(row["pure_disagreement"])
        for row in rows
    )
    d_to_a = sum(
        as_bool(row["legacy_disagreement"])
        and not as_bool(row["pure_disagreement"])
        for row in rows
    )
    if len(rows) != EXPECTED_CHANGED or (a_to_d, d_to_a) != (6, 1):
        raise RuntimeError(
            f"Reference changed set mismatch: rows={len(rows)}, "
            f"A->D={a_to_d}, D->A={d_to_a}"
        )
    return rows


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


def score_pool(candidates, image_size):
    if len(candidates) < 2:
        raise RuntimeError("A comparison frame has fewer than two candidates")
    top1, scored = select_top1_precomputed(
        candidates, "mask_depth", FULL_WEIGHTS, image_size, MODE
    )
    depths = [float(candidate["mask_depth"]) for candidate in candidates]
    min_depth, max_depth = min(depths), max(depths)
    nearest_ids = {
        int(candidate["id"]) for candidate in candidates
        if float(candidate["mask_depth"]) == min_depth
    }
    scored_by_id = {int(candidate["id"]): candidate for candidate in scored}
    img_center = (image_size[0] / 2, image_size[1] / 2)
    metrics = {}
    for candidate in candidates:
        pred_id = int(candidate["id"])
        metrics[pred_id] = {
            "gt_id": int(candidate["matched_gt_id"]),
            "mask_median_depth": float(candidate["mask_depth"]),
            "visibility": float(score_visibility(candidate)),
            "confidence": float(candidate["confidence"]),
            "center_score": float(score_center(candidate["bbox"], img_center)),
            "proximity": float(normalize_proximity(
                float(candidate["mask_depth"]), min_depth, max_depth
            )),
            "full_score": float(scored_by_id[pred_id]["score"]),
        }
    exact_min_ids = {
        pred_id for pred_id, item in metrics.items()
        if item["mask_median_depth"] == min_depth
    }
    full_id = int(top1["id"])
    return {
        "count": len(candidates),
        "ids": set(metrics),
        "nearest_ids": nearest_ids,
        "full_id": full_id,
        "disagreement": full_id not in nearest_ids,
        "metrics": metrics,
        "tie_check": nearest_ids == exact_min_ids,
    }


def validate_against_reference(reference, legacy, pure):
    expected = {
        "legacy_count": int(reference["legacy_candidate_count"]),
        "pure_count": int(reference["pure_candidate_count"]),
        "added_count": int(reference["added_candidate_count"]),
        "legacy_nearest": parse_id_set(reference["legacy_nearest_set"]),
        "pure_nearest": parse_id_set(reference["pure_nearest_set"]),
        "legacy_full": int(reference["legacy_full_pred"]),
        "pure_full": int(reference["pure_full_pred"]),
        "legacy_disagreement": as_bool(reference["legacy_disagreement"]),
        "pure_disagreement": as_bool(reference["pure_disagreement"]),
    }
    actual = {
        "legacy_count": legacy["count"],
        "pure_count": pure["count"],
        "added_count": len(pure["ids"] - legacy["ids"]),
        "legacy_nearest": legacy["nearest_ids"],
        "pure_nearest": pure["nearest_ids"],
        "legacy_full": legacy["full_id"],
        "pure_full": pure["full_id"],
        "legacy_disagreement": legacy["disagreement"],
        "pure_disagreement": pure["disagreement"],
    }
    if actual != expected:
        raise RuntimeError(
            f"Reference mismatch at {reference['sequence']}/"
            f"{reference['frame_id']}: expected={expected}, actual={actual}"
        )


def classify_change(added_ids, legacy, pure):
    added_nearest = added_ids & pure["nearest_ids"]
    added_full = pure["full_id"] in added_ids
    if added_nearest and added_full:
        classification = "C_NEW_CANDIDATE_BECAME_NEAREST_AND_FULL"
        detail = (
            f"added pred {serialize_ids(added_nearest)} entered nearest set "
            f"and pure Full selected added pred {pure['full_id']}"
        )
    elif added_nearest:
        classification = "A_NEW_CANDIDATE_BECAME_NEAREST"
        detail = (
            f"added pred {serialize_ids(added_nearest)} entered nearest set; "
            f"pure Full selected pred {pure['full_id']}"
        )
    elif added_full:
        classification = "B_NEW_CANDIDATE_BECAME_FULL_TOP1"
        detail = f"pure Full directly selected added pred {pure['full_id']}"
    elif legacy["nearest_ids"] != pure["nearest_ids"]:
        classification = "D_TIE_AWARE_RELATION_CHANGED"
        detail = "nearest set changed without an added candidate becoming nearest"
    else:
        classification = "E_OTHER"
        detail = (
            "added candidates changed proximity normalization/competition, "
            "but no added candidate was directly selected"
        )
    return classification, detail, added_nearest, added_full


def compact_metrics(influential_ids, legacy, pure, added_ids):
    result = {}
    for pred_id in sorted(influential_ids):
        roles = []
        if pred_id in added_ids:
            roles.append("ADDED")
        if pred_id in legacy["nearest_ids"]:
            roles.append("LEGACY_NEAREST")
        if pred_id in pure["nearest_ids"]:
            roles.append("PURE_NEAREST")
        if pred_id == legacy["full_id"]:
            roles.append("LEGACY_FULL")
        if pred_id == pure["full_id"]:
            roles.append("PURE_FULL")
        result[str(pred_id)] = {
            "roles": roles,
            "legacy": legacy["metrics"].get(pred_id),
            "pure": pure["metrics"].get(pred_id),
        }
    return result


def trace_rows(reference_rows):
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
        validate_against_reference(reference, legacy, pure)
        if not legacy["ids"].issubset(pure["ids"]):
            raise RuntimeError(f"Legacy is not a subset: {sequence}/{frame_id}")

        added_ids = pure["ids"] - legacy["ids"]
        classification, detail, added_nearest, added_full = classify_change(
            added_ids, legacy, pure
        )
        influential_ids = (
            added_ids | legacy["nearest_ids"] | pure["nearest_ids"]
            | {legacy["full_id"], pure["full_id"]}
        )
        added_gt_ids = {
            pure["metrics"][pred_id]["gt_id"] for pred_id in added_ids
        }
        added_pairs = ";".join(
            f"{pred_id}:{pure['metrics'][pred_id]['gt_id']}"
            for pred_id in sorted(added_ids)
        )
        metrics = compact_metrics(influential_ids, legacy, pure, added_ids)
        direction = (
            "AGREEMENT_TO_DISAGREEMENT"
            if not legacy["disagreement"] and pure["disagreement"]
            else "DISAGREEMENT_TO_AGREEMENT"
        )
        rows.append({
            "sequence": sequence,
            "frame_id": frame_id,
            "legacy_candidate_count": legacy["count"],
            "pure_candidate_count": pure["count"],
            "added_candidate_count": len(added_ids),
            "added_pred_ids": serialize_ids(added_ids),
            "added_gt_ids": serialize_ids(added_gt_ids),
            "added_pred_gt_pairs": added_pairs,
            "legacy_nearest_set": serialize_ids(legacy["nearest_ids"]),
            "pure_nearest_set": serialize_ids(pure["nearest_ids"]),
            "legacy_full_pred": legacy["full_id"],
            "pure_full_pred": pure["full_id"],
            "legacy_disagreement": legacy["disagreement"],
            "pure_disagreement": pure["disagreement"],
            "nearest_changed": legacy["nearest_ids"] != pure["nearest_ids"],
            "full_changed": legacy["full_id"] != pure["full_id"],
            "change_direction": direction,
            "cause_classification": classification,
            "cause_detail": detail,
            "added_used_as_nearest": serialize_ids(added_nearest),
            "added_used_as_full": pure["full_id"] if added_full else "",
            "influential_pred_ids": serialize_ids(influential_ids),
            "candidate_metrics_json": json.dumps(
                metrics, sort_keys=True, separators=(",", ":")
            ),
            "tie_check_pass": legacy["tie_check"] and pure["tie_check"],
        })
    return rows


def short_cause(value):
    return {
        "A_NEW_CANDIDATE_BECAME_NEAREST": "A:new nearest",
        "B_NEW_CANDIDATE_BECAME_FULL_TOP1": "B:new Full",
        "C_NEW_CANDIDATE_BECAME_NEAREST_AND_FULL": "C:new nearest+Full",
        "D_TIE_AWARE_RELATION_CHANGED": "D:tie relation",
        "E_OTHER": "E:normalization/competition",
    }[value]


def write_output(rows):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_console(rows):
    print("[CHANGED AGREEMENT/DISAGREEMENT FRAMES]")
    print(
        "seq frame_id         L/Pcand added(pred/gt) "
        "L nearest/Full P nearest/Full dir  cause"
    )
    for row in rows:
        direction = (
            "A->D" if row["change_direction"].startswith("AGREEMENT") else "D->A"
        )
        print(
            f"{row['sequence']:>3} {row['frame_id']} "
            f"{row['legacy_candidate_count']:>2}/{row['pure_candidate_count']:<2} "
            f"{row['added_pred_gt_pairs']:<24} "
            f"{row['legacy_nearest_set']}/{row['legacy_full_pred']:<4} "
            f"{row['pure_nearest_set']}/{row['pure_full_pred']:<4} "
            f"{direction:<4} {short_cause(row['cause_classification'])}"
        )
        metrics = json.loads(row["candidate_metrics_json"])
        for pred_id, item in metrics.items():
            roles = "+".join(item["roles"])
            for pool_name in ("legacy", "pure"):
                values = item[pool_name]
                if values is None:
                    continue
                print(
                    f"    pred {pred_id:>2} [{pool_name[0].upper()}] "
                    f"gt={values['gt_id']} roles={roles} "
                    f"depth={values['mask_median_depth']:.1f} "
                    f"vis={values['visibility']:.6f} "
                    f"conf={values['confidence']:.6f} "
                    f"center={values['center_score']:.6f} "
                    f"prox={values['proximity']:.6f} "
                    f"Full={values['full_score']:.6f}"
                )

    a_to_d = [
        row for row in rows
        if row["change_direction"] == "AGREEMENT_TO_DISAGREEMENT"
    ]
    d_to_a = [
        row for row in rows
        if row["change_direction"] == "DISAGREEMENT_TO_AGREEMENT"
    ]
    causes = Counter(row["cause_classification"] for row in a_to_d)
    nearest_direct = [row for row in rows if row["added_used_as_nearest"]]
    full_direct = [row for row in rows if row["added_used_as_full"] != ""]
    either_direct = {
        (row["sequence"], row["frame_id"])
        for row in nearest_direct + full_direct
    }
    tie_failures = [row for row in rows if not row["tie_check_pass"]]

    print("\n[FOUR ANSWERS]")
    print("1. Agreement -> Disagreement (6):")
    for cause, count in sorted(causes.items()):
        print(f"   {short_cause(cause)}: {count} frames")
    row = d_to_a[0]
    print(
        "2. Disagreement -> Agreement (1): "
        f"seq {row['sequence']} / frame {row['frame_id']} / "
        f"{short_cause(row['cause_classification'])}; "
        f"nearest {row['legacy_nearest_set']}->{row['pure_nearest_set']}, "
        f"Full {row['legacy_full_pred']}->{row['pure_full_pred']}"
    )
    print(
        "3. Added candidate directly used: "
        f"nearest={len(nearest_direct)} frames, "
        f"Full={len(full_direct)} frames, either={len(either_direct)} frames"
    )
    print(
        "4. Tie/consistency anomalies: "
        f"{len(tie_failures)}; exact-depth nearest-set checks "
        f"{'passed' if not tie_failures else 'failed'}"
    )
    print(f"\nCSV: {os.path.abspath(OUTPUT_CSV)}")


def main():
    reference_rows = load_changed_reference_rows()
    rows = trace_rows(reference_rows)
    if len(rows) != EXPECTED_CHANGED:
        raise RuntimeError(f"Output rows={len(rows)}, expected {EXPECTED_CHANGED}")
    if not all(row["tie_check_pass"] for row in rows):
        raise RuntimeError("Tie-aware nearest-set validation failed")
    write_output(rows)
    print_console(rows)


if __name__ == "__main__":
    main()
