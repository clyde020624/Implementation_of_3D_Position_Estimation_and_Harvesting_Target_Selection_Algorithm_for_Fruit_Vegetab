"""Select and render 12 representative v3 disagreement demo frames."""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from compare_candidate_pool_top1 import (
    build_pure_candidates,
    load_frame_inputs,
)
from load_detections import load_detections_csv
from mask_depth_reanalysis import (
    FULL_WEIGHTS,
    MODE,
    select_top1_precomputed,
)
from priority import score_center, score_visibility


DETECTION_CSV = "eval_detections_fixed.csv"
TRADEOFF_CSV = "v3_tradeoff_results.csv"
SCATTER_CSV = "v3_scatter_plot_data.csv"
SUMMARY_CSV = "v3_tradeoff_summary.csv"
OUTPUT_CSV = "v3_demo_candidates.csv"
SIMPLE_CONTACT = "v3_demo_contact_sheet.png"
AUDIT_CONTACT = "v3_demo_audit_contact_sheet.png"

EXPECTED_DISAGREEMENTS = 123
EXPECTED_CANDIDATES = 2728
DEMO_COUNT = 12
MAX_PER_SEQUENCE = 2

NEAREST_COLOR = (214, 48, 49)
FULL_COLOR = (35, 103, 176)
OTHER_COLOR = (118, 118, 118)
EXTENSIONS = (".tiff", ".tif", ".png", ".jpg", ".jpeg")

CSV_FIELDS = [
    "rank", "sequence", "frame_id",
    "nearest_pred_id", "nearest_gt_id",
    "full_pred_id", "full_gt_id",
    "nearest_depth", "full_depth", "delta_depth",
    "nearest_visibility", "full_visibility", "delta_visibility",
    "nearest_confidence", "full_confidence",
    "nearest_center", "full_center",
    "nearest_full_score", "full_full_score",
    "nearest_set_size", "candidate_count",
    "representativeness_score",
    "rgb_path", "simple_image", "audit_image",
    "candidate_pool_definition",
]


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def tradeoff_reference():
    rows = read_csv(TRADEOFF_CSV)
    if len(rows) != EXPECTED_DISAGREEMENTS:
        raise RuntimeError(
            f"v3 disagreement rows={len(rows)}, expected={EXPECTED_DISAGREEMENTS}"
        )
    return {
        (row["sequence_id"], row["frame_id"]): row for row in rows
    }


def scatter_reference():
    rows = read_csv(SCATTER_CSV)
    if len(rows) != EXPECTED_DISAGREEMENTS:
        raise RuntimeError(
            f"v3 scatter rows={len(rows)}, expected={EXPECTED_DISAGREEMENTS}"
        )
    return {
        (row["sequence"], row["frame_id"]): row for row in rows
    }


def robust_reference_stats():
    summary = {row["metric"]: row for row in read_csv(SUMMARY_CSV)}
    depth = summary["delta_depth_mm"]
    visibility = summary["delta_visibility"]
    values = {
        "median_depth": float(depth["median"]),
        "iqr_depth": float(depth["iqr"]),
        "median_visibility": float(visibility["median"]),
        "iqr_visibility": float(visibility["iqr"]),
    }
    if (
        abs(values["median_depth"] - 13.0) > 1e-9
        or abs(values["median_visibility"] - 0.17393529386893647) > 1e-9
        or values["iqr_depth"] <= 0
        or values["iqr_visibility"] <= 0
    ):
        raise RuntimeError(f"Unexpected v3 summary statistics: {values}")
    return values


def rank_frames(tradeoff, stats):
    ranked = []
    for key, source in tradeoff.items():
        delta_depth = float(source["delta_depth_mm"])
        delta_visibility = float(source["delta_visibility"])
        score = (
            abs(delta_depth - stats["median_depth"]) / stats["iqr_depth"]
            + abs(delta_visibility - stats["median_visibility"])
            / stats["iqr_visibility"]
        )
        ranked.append({
            "sequence": key[0],
            "frame_id": key[1],
            "delta_depth": delta_depth,
            "delta_visibility": delta_visibility,
            "representativeness_score": score,
        })
    return sorted(
        ranked,
        key=lambda row: (
            row["representativeness_score"],
            row["sequence"],
            row["frame_id"],
        ),
    )


def balanced_selection(ranked):
    selected = []
    per_sequence = Counter()
    for row in ranked:
        if per_sequence[row["sequence"]] >= MAX_PER_SEQUENCE:
            continue
        selected.append(dict(row))
        per_sequence[row["sequence"]] += 1
        if len(selected) == DEMO_COUNT:
            break
    if len(selected) != DEMO_COUNT:
        raise RuntimeError(f"Selected {len(selected)}, expected {DEMO_COUNT}")
    for rank, row in enumerate(selected, 1):
        row["rank"] = rank
    return selected


def validate_global_pool_count():
    rows = read_csv("candidate_pool_top1_comparison.csv")
    pure_total = sum(int(row["pure_candidate_count"]) for row in rows)
    if len(rows) != 364 or pure_total != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"Pure pool reference mismatch: frames={len(rows)}, "
            f"candidates={pure_total}"
        )


def metrics(candidate, image_size):
    center = (image_size[0] / 2, image_size[1] / 2)
    return {
        "visibility": float(score_visibility(candidate)),
        "confidence": float(candidate["confidence"]),
        "center": float(score_center(candidate["bbox"], center)),
    }


def close_enough(actual, expected, label, key):
    if abs(float(actual) - float(expected)) > 1e-9:
        raise RuntimeError(
            f"{label} mismatch at {key}: actual={actual}, expected={expected}"
        )


def find_rgb(info, sequence, frame_id):
    candidates = []
    image_path = info.get("image_path")
    if image_path:
        candidates.append(Path(image_path))
    for extension in EXTENSIONS:
        candidates.extend([
            Path("dataset_bulk") / "images" / sequence
            / f"{frame_id}{extension}",
            Path("dataset_bulk") / "rgb" / sequence
            / f"{frame_id}{extension}",
            Path(f"{frame_id}{extension}"),
        ])
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"RGB not found for {sequence}/{frame_id}: {candidates}"
    )


def reconstruct(selected, tradeoff, scatter):
    detections = load_detections_csv(DETECTION_CSV)
    for row in selected:
        key = (row["sequence"], row["frame_id"])
        info = detections.get(key)
        if info is None:
            raise RuntimeError(f"Detection frame missing: {key}")
        predictions, ground_truth, depth, image_size = load_frame_inputs(
            detections, *key
        )
        candidates = build_pure_candidates(
            [dict(item) for item in predictions],
            ground_truth,
            depth,
            image_size,
        )
        full, scored = select_top1_precomputed(
            candidates,
            "mask_depth",
            FULL_WEIGHTS,
            image_size,
            MODE,
        )
        if full is None:
            raise RuntimeError(f"No Full Top-1: {key}")

        minimum = min(float(item["mask_depth"]) for item in candidates)
        nearest_set = [
            item for item in candidates
            if float(item["mask_depth"]) == minimum
        ]
        nearest_ids = {int(item["id"]) for item in nearest_set}
        if int(full["id"]) in nearest_ids:
            raise RuntimeError(f"Selected frame is not a disagreement: {key}")
        nearest = min(
            nearest_set,
            key=lambda item: (
                -metrics(item, image_size)["visibility"],
                int(item["id"]),
            ),
        )

        scored_by_id = {int(item["id"]): item for item in scored}
        candidate_records = []
        for candidate in candidates:
            pred_id = int(candidate["id"])
            item_metrics = metrics(candidate, image_size)
            candidate_records.append({
                "pred_id": pred_id,
                "gt_id": int(candidate["matched_gt_id"]),
                "bbox": list(candidate["bbox"]),
                "depth": float(candidate["mask_depth"]),
                "visibility": item_metrics["visibility"],
                "confidence": item_metrics["confidence"],
                "center": item_metrics["center"],
                "full_score": float(scored_by_id[pred_id]["score"]),
            })
        records_by_id = {
            item["pred_id"]: item for item in candidate_records
        }
        nearest_record = records_by_id[int(nearest["id"])]
        full_record = records_by_id[int(full["id"])]

        source = tradeoff[key]
        scatter_row = scatter[key]
        delta_depth = full_record["depth"] - nearest_record["depth"]
        delta_visibility = (
            full_record["visibility"] - nearest_record["visibility"]
        )
        checks = (
            (nearest_record["depth"], source["nearest_min_mask_depth"], "near depth"),
            (full_record["depth"], source["full_mask_depth"], "full depth"),
            (nearest_record["visibility"], source["nearest_set_max_visibility"], "near visibility"),
            (full_record["visibility"], source["full_visibility"], "full visibility"),
            (delta_depth, source["delta_depth_mm"], "delta depth"),
            (delta_visibility, source["delta_visibility"], "delta visibility"),
            (nearest_record["pred_id"], scatter_row["nearest_pred_id"], "near pred"),
            (nearest_record["gt_id"], scatter_row["nearest_gt_id"], "near GT"),
            (full_record["pred_id"], scatter_row["full_pred_id"], "full pred"),
            (full_record["gt_id"], scatter_row["full_gt_id"], "full GT"),
        )
        for actual, expected, label in checks:
            close_enough(actual, expected, label, key)

        if (
            delta_depth <= 0
            or nearest_record["pred_id"] == full_record["pred_id"]
        ):
            raise RuntimeError(f"Invalid demo selection at {key}")
        if any(
            item["depth"] is None
            or not np.isfinite(item["depth"])
            for item in candidate_records
        ):
            raise RuntimeError(f"Invalid Pure mask depth at {key}")

        row.update({
            "nearest_pred_id": nearest_record["pred_id"],
            "nearest_gt_id": nearest_record["gt_id"],
            "full_pred_id": full_record["pred_id"],
            "full_gt_id": full_record["gt_id"],
            "nearest_depth": nearest_record["depth"],
            "full_depth": full_record["depth"],
            "delta_depth": delta_depth,
            "nearest_visibility": nearest_record["visibility"],
            "full_visibility": full_record["visibility"],
            "delta_visibility": delta_visibility,
            "nearest_confidence": nearest_record["confidence"],
            "full_confidence": full_record["confidence"],
            "nearest_center": nearest_record["center"],
            "full_center": full_record["center"],
            "nearest_full_score": nearest_record["full_score"],
            "full_full_score": full_record["full_score"],
            "nearest_set_size": len(nearest_set),
            "candidate_count": len(candidate_records),
            "nearest_bbox": nearest_record["bbox"],
            "full_bbox": full_record["bbox"],
            "candidate_records": candidate_records,
            "rgb_path": str(find_rgb(info, *key)),
            "candidate_pool_definition": "PURE_MASK_NO_BBOX_DEPTH_FILTER",
        })
    return selected


def load_rgb(path):
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        array = np.asarray(tifffile.imread(str(path)))
    else:
        array = np.asarray(Image.open(path).convert("RGB"))
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if (
        array.ndim == 3
        and array.shape[0] in (3, 4)
        and array.shape[-1] not in (3, 4)
    ):
        array = np.moveaxis(array, 0, -1)
    array = array[:, :, :3]
    if array.dtype != np.uint8:
        values = array.astype(np.float32)
        low, high = np.percentile(values, [1, 99])
        array = (
            np.clip((values - low) / max(high - low, 1e-6), 0, 1) * 255
        ).astype(np.uint8)
    return Image.fromarray(array)


def font(size, bold=False):
    paths = (
        ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/calibrib.ttf")
        if bold
        else ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf")
    )
    for path in paths:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def clipped_bbox(image, bbox):
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    return (
        max(0, min(image.width - 1, x1)),
        max(0, min(image.height - 1, y1)),
        max(0, min(image.width - 1, x2)),
        max(0, min(image.height - 1, y2)),
    )


def draw_box_label(
    image,
    bbox,
    color,
    text,
    line_width,
    font_size,
    prefer_below=False,
    label_alpha=215,
):
    draw = ImageDraw.Draw(image, "RGBA")
    x1, y1, x2, y2 = clipped_bbox(image, bbox)
    draw.rectangle(
        (x1, y1, x2, y2),
        outline=color + (255,),
        width=line_width,
    )
    label_font = font(font_size, bold=True)
    bounds = draw.multiline_textbbox(
        (0, 0), text, font=label_font, spacing=2
    )
    label_width = bounds[2] - bounds[0] + 10
    label_height = bounds[3] - bounds[1] + 8
    tx = max(0, min(image.width - label_width, x1))
    if prefer_below:
        ty = min(image.height - label_height, y2 + 2)
    else:
        ty = y1 - label_height - 2
        if ty < 0:
            ty = min(image.height - label_height, y2 + 2)
    draw.rectangle(
        (tx, ty, tx + label_width, ty + label_height),
        fill=color + (label_alpha,),
    )
    draw.multiline_text(
        (tx + 5, ty + 4),
        text,
        fill=(255, 255, 255, 255),
        font=label_font,
        spacing=2,
    )


def add_header(image, text):
    draw = ImageDraw.Draw(image, "RGBA")
    header_font = font(max(18, image.width // 55), bold=True)
    bounds = draw.textbbox((0, 0), text, font=header_font)
    height = bounds[3] - bounds[1] + 16
    draw.rectangle((0, 0, image.width, height), fill=(0, 0, 0, 195))
    draw.text(
        (9, 7),
        text,
        fill=(255, 255, 255, 255),
        font=header_font,
    )


def selected_label(title, row, prefix):
    return (
        f"{title}\n"
        f"Depth {row[prefix + '_depth']:.0f} mm\n"
        f"Visibility {row[prefix + '_visibility']:.3f}\n"
        f"Confidence {row[prefix + '_confidence']:.3f}"
    )


def render_simple(row):
    image = load_rgb(row["rgb_path"])
    font_size = max(15, image.width // 70)
    draw_box_label(
        image,
        row["nearest_bbox"],
        NEAREST_COLOR,
        selected_label("Nearest", row, "nearest"),
        line_width=5,
        font_size=font_size,
        prefer_below=False,
    )
    draw_box_label(
        image,
        row["full_bbox"],
        FULL_COLOR,
        selected_label("Full", row, "full"),
        line_width=5,
        font_size=font_size,
        prefer_below=True,
    )
    add_header(
        image,
        f"Rank {row['rank']:02d} | Seq {row['sequence']} | "
        f"Frame {row['frame_id']}",
    )
    output = Path(f"v3_demo_rank{row['rank']:02d}.png")
    image.save(output)
    row["simple_image"] = str(output.resolve())
    return output


def render_audit(row):
    image = load_rgb(row["rgb_path"])
    selected_ids = {row["nearest_pred_id"], row["full_pred_id"]}
    gray_font_size = max(10, image.width // 105)
    for item in row["candidate_records"]:
        if item["pred_id"] in selected_ids:
            continue
        draw_box_label(
            image,
            item["bbox"],
            OTHER_COLOR,
            (
                f"P{item['pred_id']} Z{item['depth']:.0f} "
                f"V{item['visibility']:.2f} C{item['confidence']:.2f}"
            ),
            line_width=2,
            font_size=gray_font_size,
            prefer_below=item["pred_id"] % 2 == 1,
            label_alpha=170,
        )

    near = next(
        item for item in row["candidate_records"]
        if item["pred_id"] == row["nearest_pred_id"]
    )
    full = next(
        item for item in row["candidate_records"]
        if item["pred_id"] == row["full_pred_id"]
    )
    selected_font_size = max(13, image.width // 82)
    draw_box_label(
        image,
        near["bbox"],
        NEAREST_COLOR,
        (
            f"Nearest P{near['pred_id']} Z{near['depth']:.0f} "
            f"V{near['visibility']:.2f} C{near['confidence']:.2f}"
        ),
        line_width=5,
        font_size=selected_font_size,
        prefer_below=False,
    )
    draw_box_label(
        image,
        full["bbox"],
        FULL_COLOR,
        (
            f"Full P{full['pred_id']} Z{full['depth']:.0f} "
            f"V{full['visibility']:.2f} C{full['confidence']:.2f}"
        ),
        line_width=5,
        font_size=selected_font_size,
        prefer_below=True,
    )
    add_header(
        image,
        f"Rank {row['rank']:02d} audit | Seq {row['sequence']} | "
        f"Frame {row['frame_id']} | Pure candidates {row['candidate_count']}",
    )
    output = Path(f"v3_demo_rank{row['rank']:02d}_audit.png")
    image.save(output)
    row["audit_image"] = str(output.resolve())
    return output


def contact_sheet(rows, paths, output_path, audit=False):
    columns, rows_count = 3, 4
    cell_width, cell_height = 480, 330
    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows_count * cell_height),
        (246, 247, 249),
    )
    draw = ImageDraw.Draw(sheet)
    title_font = font(16, bold=True)
    metric_font = font(14)

    for index, (row, path) in enumerate(zip(rows, paths)):
        column, panel_row = index % columns, index // columns
        origin_x = column * cell_width
        origin_y = panel_row * cell_height
        draw.rectangle(
            (
                origin_x + 3,
                origin_y + 3,
                origin_x + cell_width - 4,
                origin_y + cell_height - 4,
            ),
            outline=(205, 208, 212),
            width=1,
        )
        label = (
            f"Rank {row['rank']:02d} | Seq {row['sequence']} | "
            f"Frame {row['frame_id']}"
        )
        delta = (
            f"ΔDepth +{row['delta_depth']:.1f} mm | "
            f"ΔVisibility {row['delta_visibility']:+.4f}"
        )
        draw.text(
            (origin_x + 10, origin_y + 8),
            label,
            fill=(28, 31, 35),
            font=title_font,
        )
        draw.text(
            (origin_x + 10, origin_y + 29),
            delta,
            fill=(60, 64, 70),
            font=metric_font,
        )

        panel = Image.open(path).convert("RGB")
        panel = ImageOps.contain(
            panel,
            (cell_width - 18, cell_height - 62),
            method=Image.Resampling.LANCZOS,
        )
        paste_x = origin_x + (cell_width - panel.width) // 2
        paste_y = origin_y + 56 + (cell_height - 60 - panel.height) // 2
        sheet.paste(panel, (paste_x, paste_y))

    sheet.save(output_path)
    return Path(output_path)


def write_candidates(rows):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_outputs(rows, simple_paths, audit_paths):
    if len(rows) != DEMO_COUNT:
        raise RuntimeError(f"Demo rows={len(rows)}, expected={DEMO_COUNT}")
    sequence_counts = Counter(row["sequence"] for row in rows)
    if max(sequence_counts.values()) > MAX_PER_SEQUENCE:
        raise RuntimeError(f"Sequence cap failed: {sequence_counts}")
    for row in rows:
        if row["nearest_pred_id"] == row["full_pred_id"]:
            raise RuntimeError(
                f"Nearest equals Full: {row['sequence']}/{row['frame_id']}"
            )
        if row["candidate_pool_definition"] != "PURE_MASK_NO_BBOX_DEPTH_FILTER":
            raise RuntimeError("Unexpected candidate-pool definition")
    outputs = (
        list(simple_paths)
        + list(audit_paths)
        + [Path(SIMPLE_CONTACT), Path(AUDIT_CONTACT), Path(OUTPUT_CSV)]
    )
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing outputs: {missing}")


def print_table(rows):
    print(
        "Rank | Seq | Frame | Nearest | Full | ΔDepth | ΔVisibility"
    )
    for row in rows:
        print(
            f"{row['rank']:02d} | {row['sequence']} | "
            f"{row['frame_id']} | "
            f"P{row['nearest_pred_id']}/GT{row['nearest_gt_id']} | "
            f"P{row['full_pred_id']}/GT{row['full_gt_id']} | "
            f"{row['delta_depth']:+.1f} mm | "
            f"{row['delta_visibility']:+.4f}"
        )


def main():
    validate_global_pool_count()
    tradeoff = tradeoff_reference()
    scatter = scatter_reference()
    stats = robust_reference_stats()
    selected = balanced_selection(rank_frames(tradeoff, stats))
    selected = reconstruct(selected, tradeoff, scatter)

    simple_paths = [render_simple(row) for row in selected]
    audit_paths = [render_audit(row) for row in selected]
    contact_sheet(selected, simple_paths, SIMPLE_CONTACT)
    contact_sheet(selected, audit_paths, AUDIT_CONTACT, audit=True)
    write_candidates(selected)
    validate_outputs(selected, simple_paths, audit_paths)
    print_table(selected)


if __name__ == "__main__":
    main()
