"""Trace the exact frame/object difference between old 3544 and audit 3542."""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np

from data_loader import load_bupst20_annotation
from load_detections import load_detections_csv


DATA_DIR = Path("dataset_bulk")
AUDIT_FRAME_CSV = Path("gt_coverage_frame_summary.csv")
DETECTION_CSV = "eval_detections_fixed.csv"
OUTPUT_BY_FRAME = Path("gt_count_3544_vs_3542_by_frame.csv")
OUTPUT_MISMATCH = Path("gt_count_3544_vs_3542_mismatch.csv")

EXPECTED_OLD_FRAMES = 366
EXPECTED_AUDIT_FRAMES = 364
EXPECTED_OLD_TOTAL = 3544
EXPECTED_AUDIT_TOTAL = 3542

OLD_PATH_DESCRIPTION = (
    "check_reference.py: depth/annotation intersection with detection frame; "
    "no final 364-frame candidate filter"
)
AUDIT_PATH_DESCRIPTION = (
    "audit_gt_coverage_seq400_409.py -> "
    "data_loader.load_bupst20_annotation() -> len(ground_truth)"
)

BY_FRAME_FIELDS = [
    "sequence",
    "frame_id",
    "old_frame_included",
    "audit_frame_included",
    "old_gt_count",
    "audit_gt_count",
    "difference",
    "audit_csv_gt_count",
    "audit_csv_matches_recalculation",
    "old_count_path",
    "audit_count_path",
]

MISMATCH_FIELDS = [
    "sequence",
    "frame_id",
    "old_frame_included",
    "audit_frame_included",
    "old_gt_count",
    "audit_gt_count",
    "difference",
    "object_side",
    "gt_id",
    "bbox",
    "semantic_label",
    "mask_exists",
    "raw_record_type",
    "count_exclusion_condition",
]


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def audit_frame_counts():
    rows = read_csv(AUDIT_FRAME_CSV)
    result = {
        (row["sequence"], row["frame_id"]): int(row["gt_objects"])
        for row in rows
    }
    if len(result) != EXPECTED_AUDIT_FRAMES:
        raise RuntimeError(
            f"Audit frame count={len(result)}, expected={EXPECTED_AUDIT_FRAMES}"
        )
    return result


def old_frame_set():
    detections = load_detections_csv(DETECTION_CSV)
    frames = set()
    depth_root = DATA_DIR / "depth"
    annotation_root = DATA_DIR / "annotations"
    for sequence_path in sorted(
        path for path in depth_root.iterdir() if path.is_dir()
    ):
        sequence = sequence_path.name
        depth_ids = {
            Path(path).stem
            for path in glob.glob(str(sequence_path / "*.tif*"))
        }
        annotation_ids = {
            path.stem
            for path in (annotation_root / sequence).glob("*.pkl")
        }
        for frame_id in sorted(depth_ids & annotation_ids):
            if detections.get((sequence, frame_id)) is not None:
                frames.add((sequence, frame_id))
    if len(frames) != EXPECTED_OLD_FRAMES:
        raise RuntimeError(
            f"Old check_reference frame count={len(frames)}, "
            f"expected={EXPECTED_OLD_FRAMES}"
        )
    return frames


def verify_source_paths():
    old_source = Path("check_reference.py")
    audit_source = Path("audit_gt_coverage_seq400_409.py")
    loader_source = Path("data_loader.py")
    for path in (old_source, audit_source, loader_source):
        if not path.is_file():
            raise FileNotFoundError(path)

    old_text = old_source.read_text(encoding="utf-8", errors="replace")
    audit_text = audit_source.read_text(encoding="utf-8", errors="replace")
    loader_text = loader_source.read_text(encoding="utf-8", errors="replace")
    required = (
        ("old depth/annotation intersection", "set(depth_files) & set(ann_files)" in old_text),
        ("old detection-frame check", "if info is None" in old_text),
        ("old GT increment", "n_obj += 1" in old_text),
        ("audit loader call", "load_bupst20_annotation" in audit_text),
        ("audit len ground_truth", "len(ground_truth)" in audit_text),
        ("shared annotation loader", "load_bupst20_annotation" in loader_text),
    )
    missing = [label for label, present in required if not present]
    if missing:
        raise RuntimeError(f"Counting-path evidence missing: {missing}")


def annotation_path(sequence, frame_id):
    path = DATA_DIR / "annotations" / sequence / f"{frame_id}.pkl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def bbox_json(value):
    if value is None:
        return ""
    try:
        return json.dumps(np.asarray(value).tolist(), ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)


def record_details(gt_id, record):
    if not isinstance(record, dict):
        return {
            "gt_id": str(gt_id),
            "bbox": "",
            "semantic_label": "",
            "mask_exists": False,
            "raw_record_type": type(record).__name__,
        }
    return {
        "gt_id": str(gt_id),
        "bbox": bbox_json(record.get("bbox")),
        "semantic_label": (
            "" if record.get("semantic_label") is None
            else str(record.get("semantic_label"))
        ),
        "mask_exists": (
            "instance_mask" in record
            and record.get("instance_mask") is not None
        ),
        "raw_record_type": type(record).__name__,
    }


def compare_frames():
    verify_source_paths()
    old_frames = old_frame_set()
    audit_counts = audit_frame_counts()
    audit_frames = set(audit_counts)
    if not audit_frames.issubset(old_frames):
        raise RuntimeError(
            f"Audit frames outside old path: {sorted(audit_frames - old_frames)}"
        )

    by_frame = []
    mismatches = []
    old_total = 0
    audit_total = 0
    for sequence, frame_id in sorted(old_frames | audit_frames):
        path = annotation_path(sequence, frame_id)
        loaded = load_bupst20_annotation(str(path))
        old_included = (sequence, frame_id) in old_frames
        audit_included = (sequence, frame_id) in audit_frames
        old_count = len(loaded) if old_included else 0
        audit_count = len(loaded) if audit_included else 0
        audit_csv_count = (
            audit_counts[(sequence, frame_id)] if audit_included else 0
        )
        if audit_included and audit_count != audit_csv_count:
            raise RuntimeError(
                f"Audit CSV mismatch at {sequence}/{frame_id}: "
                f"recalculated={audit_count}, CSV={audit_csv_count}"
            )

        difference = old_count - audit_count
        by_frame.append({
            "sequence": sequence,
            "frame_id": frame_id,
            "old_frame_included": old_included,
            "audit_frame_included": audit_included,
            "old_gt_count": old_count,
            "audit_gt_count": audit_count,
            "difference": difference,
            "audit_csv_gt_count": audit_csv_count,
            "audit_csv_matches_recalculation": (
                not audit_included or audit_count == audit_csv_count
            ),
            "old_count_path": OLD_PATH_DESCRIPTION,
            "audit_count_path": AUDIT_PATH_DESCRIPTION,
        })
        old_total += old_count
        audit_total += audit_count

        old_by_id = (
            {item["id"]: item for item in loaded} if old_included else {}
        )
        audit_by_id = (
            {item["id"]: item for item in loaded} if audit_included else {}
        )
        for gt_id in sorted(set(old_by_id) - set(audit_by_id), key=str):
            details = record_details(gt_id, old_by_id[gt_id])
            mismatches.append({
                "sequence": sequence,
                "frame_id": frame_id,
                "old_frame_included": old_included,
                "audit_frame_included": audit_included,
                "old_gt_count": old_count,
                "audit_gt_count": audit_count,
                "difference": difference,
                "object_side": "OLD_3544_ONLY",
                **details,
                "count_exclusion_condition": (
                    "FRAME_NOT_IN_AUDIT_364"
                ),
            })
        for gt_id in sorted(set(audit_by_id) - set(old_by_id), key=str):
            details = record_details(gt_id, audit_by_id[gt_id])
            mismatches.append({
                "sequence": sequence,
                "frame_id": frame_id,
                "old_frame_included": old_included,
                "audit_frame_included": audit_included,
                "old_gt_count": old_count,
                "audit_gt_count": audit_count,
                "difference": difference,
                "object_side": "AUDIT_3542_ONLY",
                **details,
                "count_exclusion_condition": "FRAME_NOT_IN_OLD_PATH",
            })

    if old_total != EXPECTED_OLD_TOTAL:
        raise RuntimeError(
            f"Old path total={old_total}, expected={EXPECTED_OLD_TOTAL}"
        )
    if audit_total != EXPECTED_AUDIT_TOTAL:
        raise RuntimeError(
            f"Audit path total={audit_total}, expected={EXPECTED_AUDIT_TOTAL}"
        )
    if old_total - audit_total != 2:
        raise RuntimeError(
            f"Total difference={old_total - audit_total}, expected=2"
        )
    if sum(row["difference"] for row in by_frame) != 2:
        raise RuntimeError("Frame-wise differences do not sum to 2")
    if len(old_frames - audit_frames) != 2:
        raise RuntimeError(
            f"Old-only frames={len(old_frames - audit_frames)}, expected=2"
        )
    if audit_frames - old_frames:
        raise RuntimeError("Unexpected audit-only frames")
    return by_frame, mismatches, old_total, audit_total


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_console(by_frame, mismatches, old_total, audit_total):
    mismatch_frames = [
        row for row in by_frame if row["old_gt_count"] != row["audit_gt_count"]
    ]
    print("[COUNT]")
    print(f"Old path total: {old_total}")
    print(f"Audit path total: {audit_total}")
    print(f"Difference: {old_total - audit_total}")

    print("\n[MISMATCH FRAMES]")
    for row in mismatch_frames:
        print(
            f"{row['sequence']}/{row['frame_id']}: "
            f"old={row['old_gt_count']}, "
            f"audit={row['audit_gt_count']}"
        )

    print("\n[OBJECT DIFFERENCE]")
    for row in mismatches:
        print(
            f"{row['object_side']} | "
            f"{row['sequence']}/{row['frame_id']} | "
            f"gt_id={row['gt_id']} | "
            f"bbox={row['bbox'] or 'None'} | "
            f"semantic_label={row['semantic_label'] or 'None'} | "
            f"mask_exists={row['mask_exists']} | "
            f"record_type={row['raw_record_type']} | "
            f"condition={row['count_exclusion_condition']}"
        )


def main():
    by_frame, mismatches, old_total, audit_total = compare_frames()
    write_csv(OUTPUT_BY_FRAME, by_frame, BY_FRAME_FIELDS)
    write_csv(OUTPUT_MISMATCH, mismatches, MISMATCH_FIELDS)
    print_console(by_frame, mismatches, old_total, audit_total)


if __name__ == "__main__":
    main()
