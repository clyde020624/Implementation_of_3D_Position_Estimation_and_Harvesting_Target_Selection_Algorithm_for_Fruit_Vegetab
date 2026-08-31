"""Audit GT coverage, greedy matching, and GT-mask depth quality on seq400-409.

Independent read-only audit of the exact 364-frame set used by
mask_depth_reanalysis.py. Core research files and prior results are untouched.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data_loader import clean_depth, load_bupst20_annotation, load_depth
from load_detections import attach_mask_from_pkl, load_detections_csv
from localization import depth_from_bbox_center, iou
from mask_depth_reanalysis import prepare_candidates


DATA_DIR = "dataset_bulk"
DETECTION_CSV = "eval_detections_fixed.csv"
REFERENCE_FRAME_CSV = "mask_depth_reanalysis_results.csv"
SEQUENCES = tuple(str(value) for value in range(400, 410))
CONF_THRESHOLD = 0.30
IOU_THRESHOLD = 0.50
NO_OVERLAP_EPS = 1e-12
EXPECTED_FRAMES = 364

DETECTION_AUDIT_CSV = "gt_coverage_detection_audit.csv"
FRAME_SUMMARY_CSV = "gt_coverage_frame_summary.csv"
SEQUENCE_SUMMARY_CSV = "gt_coverage_sequence_summary.csv"
GT_OBJECT_AUDIT_CSV = "gt_object_coverage_audit.csv"
MASK_DEPTH_CSV = "gt_mask_depth_quality.csv"
MASK_DEPTH_SUMMARY_CSV = "gt_mask_depth_quality_summary.csv"
TOP_UNMATCHED_CSV = "top_unmatched_frames.csv"
POOR_MASK_DEPTH_CSV = "poor_mask_depth_frames.csv"
VIS_ROOT = "gt_coverage_visualizations"
TOP_VIS_DIR = os.path.join(VIS_ROOT, "top_unmatched")
POOR_VIS_DIR = os.path.join(VIS_ROOT, "poor_mask_depth")
TOP_CONTACT_PREFIX = "gt_coverage_contact_sheet_"
POOR_CONTACT = "poor_mask_depth_contact_sheet.png"

MATCHED_COLOR = (35, 166, 88)
UNMATCHED_COLOR = (235, 137, 35)
LOW_CONF_COLOR = (125, 125, 125)
GT_COLOR = (40, 170, 220)
POOR_COLOR = (220, 55, 55)
MASK_COLORS = (
    (230, 57, 70), (37, 150, 190), (48, 170, 95), (240, 160, 40),
    (145, 92, 190), (30, 180, 175), (220, 95, 170), (130, 155, 45),
    (95, 120, 205), (210, 115, 45),
)


def pct(count, total):
    return 100.0 * count / total if total else 0.0


def bbox_text(bbox):
    return json.dumps([round(float(v), 6) for v in bbox], separators=(",", ":"))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def best_gt(prediction, ground_truth):
    if not ground_truth:
        return None, 0.0
    return max(
        ((gt, float(iou(prediction["bbox"], gt["bbox"]))) for gt in ground_truth),
        key=lambda pair: pair[1],
    )


def best_prediction(gt, predictions):
    if not predictions:
        return None, 0.0
    return max(
        ((pred, float(iou(gt["bbox"], pred["bbox"]))) for pred in predictions),
        key=lambda pair: pair[1],
    )


def classify_detection(prediction, matched_record, best_iou):
    if prediction["confidence"] < CONF_THRESHOLD:
        return "LOW_CONF"
    if matched_record.get("gt_matched", False):
        return "MATCHED"
    if best_iou <= NO_OVERLAP_EPS:
        return "NO_GT_OVERLAP"
    if best_iou < IOU_THRESHOLD:
        return "IOU_BELOW_0_5"
    return "GREEDY_MATCH_CONFLICT"


def classify_gt(gt, predictions, high_predictions, matched_pred):
    best_all, best_all_iou = best_prediction(gt, predictions)
    best_high, best_high_iou = best_prediction(gt, high_predictions)
    if matched_pred is not None:
        status = "MATCHED"
    elif best_high is not None and best_high_iou >= IOU_THRESHOLD:
        status = "GREEDY_MATCH_CONFLICT"
    elif best_all is not None and best_all_iou >= IOU_THRESHOLD:
        status = "ONLY_LOW_CONF_IOU_GE_0_5"
    elif best_all is not None and best_all_iou > NO_OVERLAP_EPS:
        status = "DETECTION_IOU_BELOW_0_5"
    else:
        status = "NO_PREDICTION_OVERLAP"
    return status, best_all, best_all_iou, best_high, best_high_iou


def mask_depth_statistics(gt, depth):
    empty = {
        "valid_depth_pixels": 0, "valid_depth_ratio": 0.0,
        "mask_median_depth": None, "minimum_valid_depth": None,
        "maximum_valid_depth": None, "depth_q1": None,
        "depth_q3": None, "depth_iqr": None,
    }
    mask = gt.get("instance_mask")
    if mask is None:
        return {"mask_pixel_count": 0, **empty, "mask_status": "MISSING_MASK"}
    mask = np.asarray(mask, dtype=bool)
    mask_pixels = int(np.count_nonzero(mask))
    if mask.shape != depth.shape:
        return {
            "mask_pixel_count": mask_pixels, **empty,
            "mask_status": "MASK_DEPTH_SHAPE_MISMATCH",
        }
    valid = depth[mask]
    valid = valid[valid > 0]
    ratio = float(valid.size / mask_pixels) if mask_pixels else 0.0
    if valid.size == 0:
        return {
            "mask_pixel_count": mask_pixels, **empty,
            "valid_depth_ratio": ratio, "mask_status": "NO_VALID_DEPTH",
        }
    q1, median, q3 = np.percentile(valid.astype(float), [25, 50, 75])
    return {
        "mask_pixel_count": mask_pixels,
        "valid_depth_pixels": int(valid.size),
        "valid_depth_ratio": ratio,
        "mask_median_depth": float(median),
        "minimum_valid_depth": float(np.min(valid)),
        "maximum_valid_depth": float(np.max(valid)),
        "depth_q1": float(q1), "depth_q3": float(q3),
        "depth_iqr": float(q3 - q1), "mask_status": "VALID",
    }


def discover_frame_inputs(detections):
    records = []
    for sequence in SEQUENCES:
        depth_files = {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in glob.glob(os.path.join(DATA_DIR, "depth", sequence, "*.tif*"))
        }
        annotation_files = {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in glob.glob(os.path.join(DATA_DIR, "annotations", sequence, "*.pkl"))
        }
        for frame_id in sorted(set(depth_files) & set(annotation_files)):
            info = detections.get((sequence, frame_id))
            if info is not None:
                records.append({
                    "sequence": sequence, "frame_id": frame_id,
                    "depth_path": depth_files[frame_id],
                    "annotation_path": annotation_files[frame_id], "info": info,
                })
    return records


def reference_frame_set():
    if not os.path.isfile(REFERENCE_FRAME_CSV):
        raise FileNotFoundError(f"Reference frame CSV is required: {REFERENCE_FRAME_CSV}")
    with open(REFERENCE_FRAME_CSV, newline="", encoding="utf-8-sig") as handle:
        return {
            (row["sequence_id"], row["frame_id"])
            for row in csv.DictReader(handle)
        }


def find_rgb_path(sequence, frame_id, info):
    candidates = [
        os.path.join(DATA_DIR, "images", sequence, frame_id + extension)
        for extension in (".tiff", ".tif", ".png", ".jpg", ".jpeg")
    ]
    if info.get("image_path"):
        candidates.append(info["image_path"])
    return next((path for path in candidates if os.path.isfile(path)), None)


def audit_dataset():
    detections = load_detections_csv(DETECTION_CSV)
    reference = reference_frame_set()
    if len(reference) != EXPECTED_FRAMES:
        raise RuntimeError(
            f"Reference CSV has {len(reference)} unique frames, expected {EXPECTED_FRAMES}"
        )

    detection_rows, frame_rows, gt_rows, depth_rows = [], [], [], []
    visual_inputs, derived = {}, set()
    for source in discover_frame_inputs(detections):
        sequence, frame_id, info = (
            source["sequence"], source["frame_id"], source["info"]
        )
        predictions = [dict(item) for item in info["objects"]]
        high_predictions = [
            dict(item) for item in predictions
            if item["confidence"] >= CONF_THRESHOLD
        ]
        if len(high_predictions) < 2:
            continue
        depth = clean_depth(load_depth(source["depth_path"]))
        ground_truth = load_bupst20_annotation(source["annotation_path"])
        candidate_probe = prepare_candidates(
            [dict(item) for item in high_predictions], ground_truth, depth
        )
        if len(candidate_probe) < 2:
            continue

        key = (sequence, frame_id)
        derived.add(key)
        candidate_ids = {item["id"] for item in candidate_probe}
        matched_predictions = [dict(item) for item in high_predictions]
        attach_mask_from_pkl(matched_predictions, ground_truth, IOU_THRESHOLD)
        matched_by_id = {item["id"]: item for item in matched_predictions}
        matched_by_gt = {
            item["matched_gt_id"]: item for item in matched_predictions
            if item.get("gt_matched", False)
        }

        current_detection_rows = []
        for prediction in predictions:
            best, best_iou = best_gt(prediction, ground_truth)
            matched_record = matched_by_id.get(prediction["id"], {})
            status = classify_detection(prediction, matched_record, best_iou)
            row = {
                "sequence": sequence, "frame_id": frame_id,
                "pred_id": prediction["id"],
                "confidence": prediction["confidence"],
                "pred_bbox": bbox_text(prediction["bbox"]),
                "best_gt_id": None if best is None else best["id"],
                "best_iou": best_iou,
                "matched_gt_id": matched_record.get("matched_gt_id"),
                "matched": bool(matched_record.get("gt_matched", False)),
                "status": status, "reason": status,
                "selection_candidate": prediction["id"] in candidate_ids,
            }
            detection_rows.append(row)
            current_detection_rows.append(row)

        status_counts = Counter(row["status"] for row in current_detection_rows)
        conf_pass = len(high_predictions)
        matched_count = status_counts["MATCHED"]
        unmatched = conf_pass - matched_count
        frame_rows.append({
            "sequence": sequence, "frame_id": frame_id,
            "gt_objects": len(ground_truth),
            "yolo_detections": len(predictions),
            "conf_pass": conf_pass,
            "conf_fail": len(predictions) - conf_pass,
            "matched_candidate_count": matched_count,
            "selection_candidate_count": len(candidate_probe),
            "unmatched_high_conf": unmatched,
            "iou_below_0_5": status_counts["IOU_BELOW_0_5"],
            "no_gt_overlap": status_counts["NO_GT_OVERLAP"],
            "greedy_conflict": status_counts["GREEDY_MATCH_CONFLICT"],
            "unmatched_high_conf_rate": unmatched / conf_pass if conf_pass else 0.0,
        })

        for gt in ground_truth:
            matched_pred = matched_by_gt.get(gt["id"])
            status, best_all, best_all_iou, best_high, best_high_iou = classify_gt(
                gt, predictions, high_predictions, matched_pred
            )
            gt_rows.append({
                "sequence": sequence, "frame_id": frame_id, "gt_id": gt["id"],
                "gt_bbox": bbox_text(gt["bbox"]),
                "mask_pixel_count": int(gt.get("mask_area") or 0),
                "best_pred_id": None if best_all is None else best_all["id"],
                "best_pred_confidence": (
                    None if best_all is None else best_all["confidence"]
                ),
                "best_iou": best_all_iou,
                "best_high_conf_pred_id": (
                    None if best_high is None else best_high["id"]
                ),
                "best_high_conf_iou": best_high_iou,
                "matched": matched_pred is not None,
                "matched_pred_id": None if matched_pred is None else matched_pred["id"],
                "matched_iou": (
                    None if matched_pred is None else matched_pred["match_iou"]
                ),
                "coverage_status": status,
            })

            stats = mask_depth_statistics(gt, depth)
            bbox_depth = None
            if matched_pred is not None:
                bbox_depth, _ = depth_from_bbox_center(
                    matched_pred["bbox"], depth, region_half=2
                )
            stats.update({
                "sequence": sequence, "frame_id": frame_id, "gt_id": gt["id"],
                "gt_bbox": bbox_text(gt["bbox"]),
                "matched": matched_pred is not None,
                "matched_pred_id": None if matched_pred is None else matched_pred["id"],
                "matched_pred_confidence": (
                    None if matched_pred is None else matched_pred["confidence"]
                ),
                "matched_iou": (
                    None if matched_pred is None else matched_pred["match_iou"]
                ),
                "selection_candidate": (
                    matched_pred is not None and matched_pred["id"] in candidate_ids
                ),
                "bbox_center_depth": bbox_depth,
                "bbox_depth_valid": bbox_depth is not None,
                "mask_depth_valid": stats["mask_median_depth"] is not None,
                "mask_only_depth_exclusion": (
                    matched_pred is not None
                    and bbox_depth is not None
                    and stats["mask_median_depth"] is None
                ),
            })
            depth_rows.append(stats)

        visual_inputs[key] = {
            "depth_path": source["depth_path"],
            "annotation_path": source["annotation_path"],
            "rgb_path": find_rgb_path(sequence, frame_id, info),
            "info": info,
        }
        if len(derived) % 50 == 0:
            print(f"  audited usable frames: {len(derived)}/{EXPECTED_FRAMES}")

    if derived != reference:
        missing, extra = sorted(reference - derived), sorted(derived - reference)
        print("[FRAME SET MISMATCH]")
        print(f"reference={len(reference)}, derived={len(derived)}")
        print(f"missing from derived ({len(missing)}): {missing[:20]}")
        print(f"extra in derived ({len(extra)}): {extra[:20]}")
        raise RuntimeError(
            "Derived usable-frame set differs from existing selection frame set"
        )
    return detection_rows, frame_rows, gt_rows, depth_rows, visual_inputs


def sequence_summary(frame_rows):
    rows = []
    sum_keys = (
        "gt_objects", "yolo_detections", "conf_pass", "conf_fail",
        "matched_candidate_count", "selection_candidate_count",
        "unmatched_high_conf", "no_gt_overlap", "iou_below_0_5",
        "greedy_conflict",
    )
    for sequence in SEQUENCES:
        members = [row for row in frame_rows if row["sequence"] == sequence]
        summed = {key: sum(row[key] for row in members) for key in sum_keys}
        rows.append({
            "sequence": sequence, "frames": len(members), **summed,
            "matched": summed["matched_candidate_count"],
            "unmatched_rate": (
                summed["unmatched_high_conf"] / summed["conf_pass"]
                if summed["conf_pass"] else 0.0
            ),
        })
    return rows


def mask_depth_summary(depth_rows):
    summary = []
    scopes = {
        "ALL_GT": depth_rows,
        "MATCHED_GT": [row for row in depth_rows if row["matched"]],
        "SELECTION_CANDIDATE_GT": [
            row for row in depth_rows if row["selection_candidate"]
        ],
    }
    count_groups = (
        ("0", lambda n: n == 0), ("1-4", lambda n: 1 <= n <= 4),
        ("5-9", lambda n: 5 <= n <= 9),
        ("10-19", lambda n: 10 <= n <= 19), (">=20", lambda n: n >= 20),
    )
    ratio_groups = (
        ("<1%", lambda r: r < 0.01), ("<5%", lambda r: r < 0.05),
        ("<10%", lambda r: r < 0.10), ("<25%", lambda r: r < 0.25),
        (">=25%", lambda r: r >= 0.25),
    )
    for scope, rows in scopes.items():
        total = len(rows)
        for category, predicate in count_groups:
            count = sum(predicate(row["valid_depth_pixels"]) for row in rows)
            summary.append({
                "scope": scope, "metric": "valid_depth_pixels",
                "category": category, "count": count, "total": total,
                "rate": count / total if total else 0.0,
            })
        for category, predicate in ratio_groups:
            count = sum(predicate(row["valid_depth_ratio"]) for row in rows)
            summary.append({
                "scope": scope, "metric": "valid_depth_ratio",
                "category": category, "count": count, "total": total,
                "rate": count / total if total else 0.0,
            })
    return summary


def get_font(size, bold=False):
    candidates = (
        ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/calibrib.ttf")
        if bold else
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf")
    )
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_rgb(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def overlay_masks(rgb, ground_truth, alpha=0.26, selected_gt_ids=None):
    output = rgb.astype(np.float32).copy()
    height, width = output.shape[:2]
    for index, gt in enumerate(ground_truth):
        if selected_gt_ids is not None and gt["id"] not in selected_gt_ids:
            continue
        mask = gt.get("instance_mask")
        if mask is None:
            continue
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (height, width):
            continue
        color = np.asarray(MASK_COLORS[index % len(MASK_COLORS)], dtype=np.float32)
        output[mask] = (1.0 - alpha) * output[mask] + alpha * color
    return np.clip(output, 0, 255).astype(np.uint8)


def draw_labeled_box(image, bbox, color, label, width=3, font_size=13):
    draw = ImageDraw.Draw(image, "RGBA")
    image_width, image_height = image.size
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1, x2 = max(0, min(image_width - 1, x1)), max(0, min(image_width - 1, x2))
    y1, y2 = max(0, min(image_height - 1, y1)), max(0, min(image_height - 1, y2))
    draw.rectangle((x1, y1, x2, y2), outline=color + (255,), width=width)
    font = get_font(font_size, bold=True)
    bounds = draw.textbbox((0, 0), label, font=font)
    label_width = min(image_width, bounds[2] - bounds[0] + 8)
    label_height = bounds[3] - bounds[1] + 6
    tx = max(0, min(image_width - label_width, x1))
    ty = y1 - label_height
    if ty < 0:
        ty = min(image_height - label_height, y2 + 1)
    draw.rectangle(
        (tx, ty, tx + label_width, ty + label_height), fill=color + (210,)
    )
    draw.text((tx + 4, ty + 3), label, fill=(255, 255, 255, 255), font=font)


def add_header(image, text):
    draw = ImageDraw.Draw(image, "RGBA")
    font = get_font(18, bold=True)
    height = draw.textbbox((0, 0), text, font=font)[3] + 12
    draw.rectangle((0, 0, image.width, height), fill=(0, 0, 0, 190))
    draw.text((7, 5), text, fill=(255, 255, 255, 255), font=font)


def render_unmatched_frame(sequence, frame_id, visual, output_path):
    rgb = load_rgb(visual["rgb_path"])
    ground_truth = load_bupst20_annotation(visual["annotation_path"])
    predictions = [dict(item) for item in visual["info"]["objects"]]
    high = [dict(item) for item in predictions if item["confidence"] >= CONF_THRESHOLD]
    attach_mask_from_pkl(high, ground_truth, IOU_THRESHOLD)
    matched_by_id = {item["id"]: item for item in high}
    image = Image.fromarray(overlay_masks(rgb, ground_truth))
    for gt in ground_truth:
        draw_labeled_box(image, gt["bbox"], GT_COLOR, f"GT{gt['id']}", 2, 11)
    for prediction in predictions:
        _, best_iou = best_gt(prediction, ground_truth)
        record = matched_by_id.get(prediction["id"], {})
        status = classify_detection(prediction, record, best_iou)
        color = (
            LOW_CONF_COLOR if status == "LOW_CONF" else
            MATCHED_COLOR if status == "MATCHED" else UNMATCHED_COLOR
        )
        draw_labeled_box(
            image, prediction["bbox"], color,
            f"P{prediction['id']} C{prediction['confidence']:.2f} IoU{best_iou:.2f}",
            3 if status != "LOW_CONF" else 2, 11,
        )
    add_header(
        image,
        f"seq{sequence}/{frame_id}: green matched, orange unmatched, gray low conf",
    )
    image.save(output_path)
    return output_path


def render_poor_depth_frame(sequence, frame_id, visual, poor_rows, output_path):
    rgb = load_rgb(visual["rgb_path"])
    ground_truth = load_bupst20_annotation(visual["annotation_path"])
    selected_ids = {row["gt_id"] for row in poor_rows}
    image = Image.fromarray(
        overlay_masks(rgb, ground_truth, alpha=0.38, selected_gt_ids=selected_ids)
    )
    by_id = {row["gt_id"]: row for row in poor_rows}
    for gt in ground_truth:
        if gt["id"] not in selected_ids:
            continue
        row = by_id[gt["id"]]
        median = row["mask_median_depth"]
        median_text = "NA" if median is None else f"{median:.1f}mm"
        label = (
            f"GT{gt['id']} valid {row['valid_depth_pixels']}/"
            f"{row['mask_pixel_count']} ({100*row['valid_depth_ratio']:.2f}%) "
            f"med {median_text}"
        )
        draw_labeled_box(image, gt["bbox"], POOR_COLOR, label, 4, 11)
    add_header(image, f"seq{sequence}/{frame_id}: poor matched-candidate mask depth")
    image.save(output_path)
    return output_path


def make_contact_sheet(image_paths, output_path, columns=5, tile_size=(260, 470)):
    if not image_paths:
        placeholder = Image.new("RGB", (800, 180), "white")
        draw = ImageDraw.Draw(placeholder)
        draw.text(
            (20, 70), "No frames satisfied this condition",
            fill="black", font=get_font(22, True),
        )
        placeholder.save(output_path)
        return
    rows = math.ceil(len(image_paths) / columns)
    sheet = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "white")
    for index, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail(tile_size, Image.Resampling.LANCZOS)
        x = index % columns * tile_size[0] + (tile_size[0] - image.width) // 2
        y = index // columns * tile_size[1] + (tile_size[1] - image.height) // 2
        sheet.paste(image, (x, y))
    sheet.save(output_path)


def create_visual_outputs(frame_rows, depth_rows, visual_inputs):
    os.makedirs(TOP_VIS_DIR, exist_ok=True)
    os.makedirs(POOR_VIS_DIR, exist_ok=True)
    top_rows = sorted(
        frame_rows,
        key=lambda row: (
            -row["unmatched_high_conf"], -row["unmatched_high_conf_rate"],
            -row["conf_pass"], row["sequence"], row["frame_id"],
        ),
    )[:20]
    top_paths = []
    for rank, row in enumerate(top_rows, start=1):
        key = (row["sequence"], row["frame_id"])
        output_path = os.path.join(
            TOP_VIS_DIR, f"{rank:02d}_seq{row['sequence']}_{row['frame_id']}.png"
        )
        visual = visual_inputs[key]
        if visual["rgb_path"] is None:
            row["preview_path"] = "RGB_NOT_FOUND"
        else:
            render_unmatched_frame(*key, visual, output_path)
            row["preview_path"] = output_path
            top_paths.append(output_path)
        row["rank"] = rank

    contact_paths = []
    for page, start in enumerate(range(0, len(top_paths), 10), start=1):
        output = f"{TOP_CONTACT_PREFIX}{page:02d}.png"
        make_contact_sheet(top_paths[start:start + 10], output)
        contact_paths.append(output)

    poor_instances = [
        row for row in depth_rows
        if row["selection_candidate"]
        and (row["valid_depth_pixels"] < 5 or row["valid_depth_ratio"] < 0.05)
    ]
    poor_by_frame = defaultdict(list)
    for row in poor_instances:
        poor_by_frame[(row["sequence"], row["frame_id"])].append(row)
    poor_frames = sorted(
        poor_by_frame,
        key=lambda key: (
            min(row["valid_depth_pixels"] for row in poor_by_frame[key]),
            min(row["valid_depth_ratio"] for row in poor_by_frame[key]),
            -len(poor_by_frame[key]), key[0], key[1],
        ),
    )[:20]
    poor_paths = []
    frame_rank = {key: rank for rank, key in enumerate(poor_frames, start=1)}
    for key in poor_frames:
        rank = frame_rank[key]
        output_path = os.path.join(
            POOR_VIS_DIR, f"{rank:02d}_seq{key[0]}_{key[1]}.png"
        )
        visual = visual_inputs[key]
        if visual["rgb_path"] is not None:
            render_poor_depth_frame(*key, visual, poor_by_frame[key], output_path)
            poor_paths.append(output_path)
    make_contact_sheet(poor_paths, POOR_CONTACT, columns=4, tile_size=(300, 535))

    for row in poor_instances:
        key = (row["sequence"], row["frame_id"])
        row["contact_sheet_rank"] = frame_rank.get(key)
        row["selected_for_contact_sheet"] = key in frame_rank
    return top_rows, poor_instances, contact_paths


def write_outputs(
    detection_rows, frame_rows, gt_rows, depth_rows,
    sequence_rows, top_rows, poor_rows,
):
    write_csv(DETECTION_AUDIT_CSV, detection_rows, [
        "sequence", "frame_id", "pred_id", "confidence", "pred_bbox",
        "best_gt_id", "best_iou", "matched_gt_id", "matched", "status",
        "reason", "selection_candidate",
    ])
    write_csv(FRAME_SUMMARY_CSV, frame_rows, [
        "sequence", "frame_id", "gt_objects", "yolo_detections", "conf_pass",
        "conf_fail", "matched_candidate_count", "selection_candidate_count",
        "unmatched_high_conf", "iou_below_0_5", "no_gt_overlap",
        "greedy_conflict", "unmatched_high_conf_rate",
    ])
    write_csv(SEQUENCE_SUMMARY_CSV, sequence_rows, [
        "sequence", "frames", "gt_objects", "yolo_detections", "conf_pass",
        "conf_fail", "matched", "matched_candidate_count",
        "selection_candidate_count", "unmatched_high_conf", "unmatched_rate",
        "no_gt_overlap", "iou_below_0_5", "greedy_conflict",
    ])
    write_csv(GT_OBJECT_AUDIT_CSV, gt_rows, [
        "sequence", "frame_id", "gt_id", "gt_bbox", "mask_pixel_count",
        "best_pred_id", "best_pred_confidence", "best_iou",
        "best_high_conf_pred_id", "best_high_conf_iou", "matched",
        "matched_pred_id", "matched_iou", "coverage_status",
    ])
    write_csv(MASK_DEPTH_CSV, depth_rows, [
        "sequence", "frame_id", "gt_id", "gt_bbox", "matched",
        "matched_pred_id", "matched_pred_confidence", "matched_iou",
        "selection_candidate", "mask_pixel_count", "valid_depth_pixels",
        "bbox_center_depth", "bbox_depth_valid", "mask_depth_valid",
        "mask_only_depth_exclusion",
        "valid_depth_ratio", "mask_median_depth", "minimum_valid_depth",
        "maximum_valid_depth", "depth_q1", "depth_q3", "depth_iqr",
        "mask_status",
    ])
    write_csv(MASK_DEPTH_SUMMARY_CSV, mask_depth_summary(depth_rows), [
        "scope", "metric", "category", "count", "total", "rate",
    ])
    write_csv(TOP_UNMATCHED_CSV, top_rows, [
        "rank", "sequence", "frame_id", "gt_objects", "yolo_detections",
        "conf_pass", "conf_fail", "matched_candidate_count",
        "selection_candidate_count", "unmatched_high_conf",
        "unmatched_high_conf_rate", "no_gt_overlap", "iou_below_0_5",
        "greedy_conflict", "preview_path",
    ])
    write_csv(POOR_MASK_DEPTH_CSV, poor_rows, [
        "sequence", "frame_id", "gt_id", "gt_bbox", "matched_pred_id",
        "matched_pred_confidence", "matched_iou", "mask_pixel_count",
        "valid_depth_pixels", "valid_depth_ratio", "mask_median_depth",
        "minimum_valid_depth", "maximum_valid_depth", "depth_iqr",
        "contact_sheet_rank", "selected_for_contact_sheet",
    ])


def print_report(
    detection_rows, frame_rows, gt_rows, depth_rows,
    sequence_rows, top_rows, contact_paths,
):
    total_detections = len(detection_rows)
    conf_pass = sum(row["confidence"] >= CONF_THRESHOLD for row in detection_rows)
    conf_fail = total_detections - conf_pass
    statuses = Counter(row["status"] for row in detection_rows)
    matched = statuses["MATCHED"]
    unmatched = conf_pass - matched
    total_gt = len(gt_rows)
    matched_gt = sum(row["matched"] for row in gt_rows)
    matched_depth = [row for row in depth_rows if row["matched"]]
    candidate_depth = [row for row in depth_rows if row["selection_candidate"]]
    no_overlap_rows = [
        row for row in detection_rows
        if row["confidence"] >= CONF_THRESHOLD
        and row["best_iou"] <= NO_OVERLAP_EPS
    ]
    boundary_rows = [
        row for row in detection_rows
        if row["confidence"] >= CONF_THRESHOLD
        and 0.4 <= row["best_iou"] < 0.5
    ]
    poor_all = [row for row in depth_rows if row["valid_depth_pixels"] < 5]
    poor_matched = [
        row for row in matched_depth if row["valid_depth_pixels"] < 5
    ]
    poor_candidate = [
        row for row in candidate_depth if row["valid_depth_pixels"] < 5
    ]
    mask_only_exclusions = [
        row for row in matched_depth if row["mask_only_depth_exclusion"]
    ]

    def unique_frames(rows):
        return len({(row["sequence"], row["frame_id"]) for row in rows})

    print("\n[DATASET]")
    print(f"Frames: {len(frame_rows)} (exact existing 364-frame set: PASS)")
    print(f"GT objects: {total_gt}")
    print(f"YOLO detections: {total_detections}")
    print("\n[CONFIDENCE]")
    print(f"Conf >= 0.30: {conf_pass}")
    print(f"Conf < 0.30: {conf_fail}")
    print("\n[GT MATCHING]")
    print(f"Matched: {matched}")
    print(f"Unmatched high-confidence: {unmatched}")
    print(f"Unmatched rate: {pct(unmatched, conf_pass):.2f}%")
    print(f"- No GT overlap: {statuses['NO_GT_OVERLAP']}")
    print(f"- IoU below 0.5: {statuses['IOU_BELOW_0_5']}")
    print(f"- Greedy conflict: {statuses['GREEDY_MATCH_CONFLICT']}")

    print("\n[GT OBJECT COVERAGE]")
    print(f"GT matched: {matched_gt}")
    print(f"GT unmatched: {total_gt - matched_gt}")
    print(f"GT matching rate: {pct(matched_gt, total_gt):.2f}%")
    gt_statuses = Counter(row["coverage_status"] for row in gt_rows)
    for name in (
        "DETECTION_IOU_BELOW_0_5", "NO_PREDICTION_OVERLAP",
        "ONLY_LOW_CONF_IOU_GE_0_5", "GREEDY_MATCH_CONFLICT",
    ):
        print(f"- {name}: {gt_statuses[name]}")

    print("\n[MASK DEPTH QUALITY]")
    print(f"Matched GT masks: {len(matched_depth)}")
    print(
        "Valid depth = 0: "
        f"{sum(row['valid_depth_pixels'] == 0 for row in matched_depth)}"
    )
    print(f"Valid depth pixels < 5: {len(poor_matched)}")
    print(
        "Valid depth pixels < 10: "
        f"{sum(row['valid_depth_pixels'] < 10 for row in matched_depth)}"
    )
    print(
        "Valid depth ratio < 1%: "
        f"{sum(row['valid_depth_ratio'] < .01 for row in matched_depth)}"
    )
    print(
        "Valid depth ratio < 5%: "
        f"{sum(row['valid_depth_ratio'] < .05 for row in matched_depth)}"
    )
    print(
        "Valid depth ratio < 10%: "
        f"{sum(row['valid_depth_ratio'] < .10 for row in matched_depth)}"
    )
    bins = Counter(
        "0" if row["valid_depth_pixels"] == 0 else
        "1-4" if row["valid_depth_pixels"] < 5 else
        "5-9" if row["valid_depth_pixels"] < 10 else
        "10-19" if row["valid_depth_pixels"] < 20 else ">=20"
        for row in matched_depth
    )
    print(
        "Valid-depth bins: "
        + ", ".join(
            f"{key}: {bins[key]}"
            for key in ("0", "1-4", "5-9", "10-19", ">=20")
        )
    )

    print("\n[SEQUENCE]")
    for row in sequence_rows:
        print(
            f"{row['sequence']}: frames={row['frames']} GT={row['gt_objects']} "
            f"YOLO={row['yolo_detections']} conf_pass={row['conf_pass']} "
            f"matched={row['matched']} unmatched={row['unmatched_high_conf']} "
            f"rate={100*row['unmatched_rate']:.2f}% "
            f"no_overlap={row['no_gt_overlap']} "
            f"low_iou={row['iou_below_0_5']} conflict={row['greedy_conflict']}"
        )

    print("\n[TOP PROBLEM FRAMES]")
    for row in top_rows:
        print(
            f"{row['rank']:02d}. seq{row['sequence']}/{row['frame_id']} "
            f"unmatched={row['unmatched_high_conf']}/{row['conf_pass']} "
            f"({100*row['unmatched_high_conf_rate']:.1f}%)"
        )

    print("\n[IMPORTANT CASES]")
    print(
        "Case A - high-confidence detections with no GT overlap: "
        f"{len(no_overlap_rows)} detections / {unique_frames(no_overlap_rows)} frames"
    )
    print(
        "Case B - threshold boundary 0.4 <= best IoU < 0.5: "
        f"{len(boundary_rows)} detections / {unique_frames(boundary_rows)} frames"
    )
    print(
        "Case C - GT valid depth pixels < 5: "
        f"all GT {len(poor_all)}/{len(depth_rows)}, "
        f"matched GT {len(poor_matched)}/{len(matched_depth)}, "
        f"selection-candidate GT {len(poor_candidate)}/{len(candidate_depth)}"
    )

    no_overlap_frames = unique_frames(no_overlap_rows)
    unmatched_frames = sum(row["unmatched_high_conf"] > 0 for row in frame_rows)
    poor_combined = [
        row for row in candidate_depth
        if row["valid_depth_pixels"] < 5 or row["valid_depth_ratio"] < 0.05
    ]
    print("\n[INTERPRETATION]")
    print(
        "1. seq408-like signal: high-confidence/no-GT-overlap detections occur in "
        f"{no_overlap_frames}/{len(frame_rows)} frames "
        f"({pct(no_overlap_frames, len(frame_rows)):.2f}%). "
        "They may be annotation omissions or YOLO false positives; review contact sheets."
    )
    print(
        "2. Candidate-pool exclusion by GT matching: "
        f"{unmatched}/{conf_pass} high-confidence detections "
        f"({pct(unmatched, conf_pass):.2f}%), affecting "
        f"{unmatched_frames}/{len(frame_rows)} frames "
        f"({pct(unmatched_frames, len(frame_rows)):.2f}%)."
    )
    print(
        "3. GT-matching exclusion affects both studies because both use the same "
        "GT-matched candidate pool and GT-mask occupancy visibility."
    )
    print(
        "4. Follow-up-only representative-depth risk: "
        f"{len(poor_candidate)}/{len(candidate_depth)} selection candidates have "
        "<5 valid pixels; (<5 pixels or <5% ratio) affects "
        f"{unique_frames(poor_combined)} frames."
    )
    print(
        "   Matched objects with valid bbox-center depth but invalid mask depth: "
        f"{len(mask_only_exclusions)}/{len(matched_depth)}."
    )
    print(
        "5. The 103/364 and 118/364 values remain reproducible conditional results "
        "for this fixed GT-matchable pool. The audit does not relabel data; the rates "
        "above determine whether broader scene-level claims require protocol review."
    )
    print(
        "Original-only issue: bbox-center 5x5 depth contamination remains possible "
        "in the original study; GT-mask median controls that contamination in follow-up."
    )

    print("\n[OUTPUTS]")
    for path in (
        DETECTION_AUDIT_CSV, FRAME_SUMMARY_CSV, SEQUENCE_SUMMARY_CSV,
        GT_OBJECT_AUDIT_CSV, MASK_DEPTH_CSV, MASK_DEPTH_SUMMARY_CSV,
        TOP_UNMATCHED_CSV, POOR_MASK_DEPTH_CSV, *contact_paths, POOR_CONTACT,
    ):
        print(os.path.abspath(path))


def main():
    print("Building the exact existing selection frame set and running the audit...")
    detection_rows, frame_rows, gt_rows, depth_rows, visual_inputs = audit_dataset()
    sequence_rows = sequence_summary(frame_rows)
    top_rows, poor_rows, contact_paths = create_visual_outputs(
        frame_rows, depth_rows, visual_inputs
    )
    write_outputs(
        detection_rows, frame_rows, gt_rows, depth_rows,
        sequence_rows, top_rows, poor_rows,
    )
    print_report(
        detection_rows, frame_rows, gt_rows, depth_rows,
        sequence_rows, top_rows, contact_paths,
    )


if __name__ == "__main__":
    main()
