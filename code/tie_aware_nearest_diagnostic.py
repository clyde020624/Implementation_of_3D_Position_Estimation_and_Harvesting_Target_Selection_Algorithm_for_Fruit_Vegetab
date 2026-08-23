"""Tie-aware comparison of bbox-depth and GT-mask-depth nearest sets.

This diagnostic does not choose one candidate when minimum depths are tied.
It does not modify the existing selection/reanalysis implementations and does
not write result files.
"""

import glob
import os

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
SEQUENCES = {str(i) for i in range(400, 410)}
CONF_THRESHOLD = 0.30


def nearest_set(candidates, depth_key):
    """Return all candidate IDs whose depth is exactly the minimum depth."""
    minimum = min(candidate[depth_key] for candidate in candidates)
    ids = frozenset(
        candidate["id"]
        for candidate in candidates
        if candidate[depth_key] == minimum
    )
    return ids, minimum


def analyze():
    detections = load_detections_csv(DETECTION_CSV)
    depth_root = os.path.join(DATA_DIR, "depth")
    annotation_root = os.path.join(DATA_DIR, "annotations")
    rows = []

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
            bbox_full, _ = select_top1_precomputed(
                candidates,
                "bbox_depth",
                FULL_WEIGHTS,
                image_size,
                MODE,
            )
            mask_full, _ = select_top1_precomputed(
                candidates,
                "mask_depth",
                FULL_WEIGHTS,
                image_size,
                MODE,
            )
            if bbox_full is None or mask_full is None:
                continue

            bbox_set, bbox_min = nearest_set(candidates, "bbox_depth")
            mask_set, mask_min = nearest_set(candidates, "mask_depth")

            # Historical order-dependent single-min selections, retained only
            # for comparison with the tie-aware set membership results.
            bbox_single_nearest = min(
                candidates,
                key=lambda candidate: candidate["bbox_depth"],
            )
            mask_single_nearest = min(
                candidates,
                key=lambda candidate: candidate["mask_depth"],
            )

            if bbox_set == mask_set:
                relation = "equal"
            elif bbox_set & mask_set:
                relation = "overlap"
            else:
                relation = "disjoint"

            rows.append(
                {
                    "sequence": sequence,
                    "frame": frame_id,
                    "bbox_set": bbox_set,
                    "mask_set": mask_set,
                    "bbox_min": bbox_min,
                    "mask_min": mask_min,
                    "relation": relation,
                    "bbox_full_in_set": bbox_full["id"] in bbox_set,
                    "mask_full_in_set": mask_full["id"] in mask_set,
                    "bbox_single_full_disagreement": (
                        bbox_single_nearest["id"] != bbox_full["id"]
                    ),
                    "mask_single_full_disagreement": (
                        mask_single_nearest["id"] != mask_full["id"]
                    ),
                }
            )

    return rows


def percentage(count, total):
    return count / total * 100 if total else 0.0


def main():
    rows = analyze()
    total = len(rows)

    equal = [row for row in rows if row["relation"] == "equal"]
    overlap = [row for row in rows if row["relation"] == "overlap"]
    disjoint = [row for row in rows if row["relation"] == "disjoint"]
    bbox_ties = [row for row in rows if len(row["bbox_set"]) > 1]
    mask_ties = [row for row in rows if len(row["mask_set"]) > 1]
    either_ties = [
        row
        for row in rows
        if len(row["bbox_set"]) > 1 or len(row["mask_set"]) > 1
    ]
    bbox_full_disagreements = [
        row for row in rows if not row["bbox_full_in_set"]
    ]
    mask_full_disagreements = [
        row for row in rows if not row["mask_full_in_set"]
    ]
    bbox_single_disagreements = [
        row for row in rows if row["bbox_single_full_disagreement"]
    ]
    mask_single_disagreements = [
        row for row in rows if row["mask_single_full_disagreement"]
    ]

    print("=" * 72)
    print("Tie-aware bbox-depth vs GT-mask-depth nearest-set diagnostic")
    print("=" * 72)
    print(f"Comparable frames          : {total}")
    print(
        f"Sets exactly equal         : {len(equal)}/{total} "
        f"({percentage(len(equal), total):.1f}%)"
    )
    print(
        f"Different but overlapping  : {len(overlap)}/{total} "
        f"({percentage(len(overlap), total):.1f}%)"
    )
    print(
        f"Disjoint (stable mismatch) : {len(disjoint)}/{total} "
        f"({percentage(len(disjoint), total):.1f}%)"
    )
    print(
        f"Sets not equal             : {len(overlap) + len(disjoint)}/{total} "
        f"({percentage(len(overlap) + len(disjoint), total):.1f}%)"
    )
    print()
    print(f"Bbox-depth tie frames      : {len(bbox_ties)}/{total}")
    print(f"Mask-depth tie frames      : {len(mask_ties)}/{total}")
    print(f"Either depth has a tie     : {len(either_ties)}/{total}")

    print()
    print("=" * 72)
    print("Tie-aware nearest-set vs Full Top-1")
    print("=" * 72)
    print(
        f"Bbox nearest set vs bbox Full disagreement : "
        f"{len(bbox_full_disagreements)}/{total} "
        f"({percentage(len(bbox_full_disagreements), total):.1f}%)"
    )
    print(
        f"  Historical single min() vs bbox Full      : "
        f"{len(bbox_single_disagreements)}/{total} "
        f"({percentage(len(bbox_single_disagreements), total):.1f}%)"
    )
    print(
        f"Mask nearest set vs GT-mask Full disagreement: "
        f"{len(mask_full_disagreements)}/{total} "
        f"({percentage(len(mask_full_disagreements), total):.1f}%)"
    )
    print(
        f"  Historical single min() vs GT-mask Full   : "
        f"{len(mask_single_disagreements)}/{total} "
        f"({percentage(len(mask_single_disagreements), total):.1f}%)"
    )

    print()
    print("Sequence-wise tie-aware mask nearest set vs GT-mask Full")
    print(f"{'seq':<6}{'frames':>8}{'disagree':>12}{'rate':>10}")
    for sequence in sorted(SEQUENCES):
        sequence_rows = [
            row for row in rows if row["sequence"] == sequence
        ]
        sequence_disagreements = sum(
            not row["mask_full_in_set"] for row in sequence_rows
        )
        print(
            f"{sequence:<6}{len(sequence_rows):>8}"
            f"{sequence_disagreements:>12}"
            f"{percentage(sequence_disagreements, len(sequence_rows)):>9.1f}%"
        )

    print()
    print("Frames with different but overlapping nearest sets")
    for row in overlap:
        print(
            f"  {row['sequence']}/{row['frame']} | "
            f"bbox={sorted(row['bbox_set'])} at {row['bbox_min']:.0f} | "
            f"mask={sorted(row['mask_set'])} at {row['mask_min']:.0f}"
        )


if __name__ == "__main__":
    main()
