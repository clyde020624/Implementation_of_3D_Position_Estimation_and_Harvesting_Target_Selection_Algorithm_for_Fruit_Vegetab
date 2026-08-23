"""Priority Score ablation using matched GT-mask median depth.

The existing research modules are imported and reused without modification.
Nearest agreement is tie-aware: a selected Top-1 agrees when its candidate ID
belongs to the complete set of candidates at the exact minimum mask depth.
"""

import csv
import glob
import os
from collections import defaultdict

from data_loader import clean_depth, load_bupst20_annotation, load_depth
from load_detections import load_detections_csv
from mask_depth_reanalysis import (
    FULL_WEIGHTS,
    MODE,
    prepare_candidates,
    select_top1_precomputed,
)


DATA_DIR = "dataset_bulk"
DETECTION_CSV = "eval_detections_fixed.csv"
OUTPUT_CSV = "mask_depth_ablation_results.csv"

SEQUENCES = {str(i) for i in range(400, 410)}
CONF_THRESHOLD = 0.30

EXPECTED_FRAMES = 364
EXPECTED_D_DISAGREEMENTS = 0
EXPECTED_FULL_DISAGREEMENTS = 118


ABLATIONS = {
    "D": {
        "proximity": 1.0,
        "visibility": 0.0,
        "confidence": 0.0,
        "center": 0.0,
    },
    "D+V": {
        "proximity": 0.35 / (0.35 + 0.30),
        "visibility": 0.30 / (0.35 + 0.30),
        "confidence": 0.0,
        "center": 0.0,
    },
    "D+V+Conf": {
        "proximity": 0.35 / 0.85,
        "visibility": 0.30 / 0.85,
        "confidence": 0.20 / 0.85,
        "center": 0.0,
    },
    "Full": dict(FULL_WEIGHTS),
    "Full-V": {
        "proximity": 0.35 / 0.70,
        "visibility": 0.0,
        "confidence": 0.20 / 0.70,
        "center": 0.15 / 0.70,
    },
}

CSV_PREFIX = {
    "D": "d",
    "D+V": "d_v",
    "D+V+Conf": "d_v_conf",
    "Full": "full",
    "Full-V": "full_no_v",
}


def percentage(count, total):
    return count / total * 100 if total else 0.0


def mask_nearest_set(candidates):
    """Return all IDs at the exact minimum GT-mask median depth."""
    minimum = min(candidate["mask_depth"] for candidate in candidates)
    nearest_ids = frozenset(
        candidate["id"]
        for candidate in candidates
        if candidate["mask_depth"] == minimum
    )
    return nearest_ids, minimum


def validate_weights():
    for name, weights in ABLATIONS.items():
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-12:
            raise ValueError(f"{name} weights sum to {total}, not 1.0")


def analyze():
    validate_weights()
    detections = load_detections_csv(DETECTION_CSV)
    depth_root = os.path.join(DATA_DIR, "depth")
    annotation_root = os.path.join(DATA_DIR, "annotations")

    rows = []
    totals = {
        name: {
            "disagreement": 0,
            "same_as_full": 0,
            "per_sequence": defaultdict(
                lambda: {"frames": 0, "disagreement": 0}
            ),
        }
        for name in ABLATIONS
    }

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
            nearest_ids, minimum_depth = mask_nearest_set(candidates)

            selections = {}
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
                        f"No Top-1 for {sequence}/{frame_id} in {name}"
                    )
                selections[name] = top1

            full_top1_id = selections["Full"]["id"]
            row = {
                "sequence_id": sequence,
                "frame_id": frame_id,
                "n_candidates": len(candidates),
                "mask_min_depth": minimum_depth,
                "mask_nearest_set": ";".join(
                    str(candidate_id)
                    for candidate_id in sorted(nearest_ids)
                ),
                "full_top1_id": full_top1_id,
            }

            for name, top1 in selections.items():
                disagreement = top1["id"] not in nearest_ids
                same_as_full = top1["id"] == full_top1_id
                stats = totals[name]

                stats["disagreement"] += int(disagreement)
                stats["same_as_full"] += int(same_as_full)
                stats["per_sequence"][sequence]["frames"] += 1
                stats["per_sequence"][sequence]["disagreement"] += int(
                    disagreement
                )

                prefix = CSV_PREFIX[name]
                row[f"{prefix}_top1_id"] = top1["id"]
                row[f"{prefix}_top1_score"] = top1["score"]
                row[f"{prefix}_nearest_disagreement"] = int(disagreement)
                row[f"{prefix}_same_as_full"] = int(same_as_full)

            rows.append(row)

    return rows, totals


def write_csv(rows):
    fieldnames = [
        "sequence_id",
        "frame_id",
        "n_candidates",
        "mask_min_depth",
        "mask_nearest_set",
    ]
    for name in ABLATIONS:
        prefix = CSV_PREFIX[name]
        fieldnames.extend(
            [
                f"{prefix}_top1_id",
                f"{prefix}_top1_score",
                f"{prefix}_nearest_disagreement",
                f"{prefix}_same_as_full",
            ]
        )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_results(rows, totals):
    total_frames = len(rows)

    print("=" * 86)
    print("GT-mask median depth Priority Score ablation")
    print("=" * 86)
    print(f"Comparable frames: {total_frames}")
    print()
    print(
        f"{'condition':<12}{'disagreement':>16}{'rate':>10}"
        f"{'same as Full':>18}{'rate':>10}"
    )
    for name, stats in totals.items():
        disagreement = stats["disagreement"]
        same_as_full = stats["same_as_full"]
        print(
            f"{name:<12}{disagreement:>9}/{total_frames:<6}"
            f"{percentage(disagreement, total_frames):>9.1f}%"
            f"{same_as_full:>11}/{total_frames:<6}"
            f"{percentage(same_as_full, total_frames):>9.1f}%"
        )

    print()
    print("Sequence-wise tie-aware mask-nearest disagreement rate")
    header = f"{'seq':<6}{'frames':>8}" + "".join(
        f"{name:>14}" for name in ABLATIONS
    )
    print(header)
    for sequence in sorted(SEQUENCES):
        frame_count = totals["Full"]["per_sequence"][sequence]["frames"]
        values = []
        for name in ABLATIONS:
            sequence_stats = totals[name]["per_sequence"][sequence]
            values.append(
                f"{sequence_stats['disagreement']:>4}/"
                f"{sequence_stats['frames']:<4} "
                f"{percentage(sequence_stats['disagreement'], sequence_stats['frames']):>4.1f}%"
            )
        print(
            f"{sequence:<6}{frame_count:>8}"
            + "".join(f"{value:>14}" for value in values)
        )

    checks = [
        (
            "Comparable frames",
            total_frames,
            EXPECTED_FRAMES,
        ),
        (
            "D tie-aware disagreement",
            totals["D"]["disagreement"],
            EXPECTED_D_DISAGREEMENTS,
        ),
        (
            "Full tie-aware disagreement",
            totals["Full"]["disagreement"],
            EXPECTED_FULL_DISAGREEMENTS,
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
        raise RuntimeError("One or more ablation validation checks failed")


def main():
    rows, totals = analyze()
    write_csv(rows)
    print_results(rows, totals)


if __name__ == "__main__":
    main()
