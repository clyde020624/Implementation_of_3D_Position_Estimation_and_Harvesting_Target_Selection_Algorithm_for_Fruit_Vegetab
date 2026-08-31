"""Diagnose GT coverage/alignment and YOLO localization for the seq408 frame.

Creates GT-mask overlays and CSV diagnostics without modifying core code.
"""

import csv
import glob
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data_loader import clean_depth, load_bupst20_annotation, load_depth
from load_detections import attach_mask_from_pkl, load_detections_csv
from localization import iou


SEQUENCE = "408"
FRAME_ID = "1600938562015928"
DATA_DIR = "dataset_bulk"
DETECTION_CSV = "eval_detections_fixed.csv"
CONF_THRESHOLD = 0.30
IOU_THRESHOLD = 0.50

OUTPUT_GT = "seq408_all_gt_instances.png"
OUTPUT_COMBINED = "seq408_gt_and_yolo_overlay.png"
OUTPUT_GT_CSV = "seq408_gt_summary.csv"
OUTPUT_UNMATCHED_CSV = "seq408_unmatched_detection_analysis.csv"

MATCHED_COLOR = (34, 166, 88)
UNMATCHED_COLOR = (235, 137, 35)
LOW_CONF_COLOR = (125, 125, 125)
MASK_COLORS = (
    (230, 57, 70),
    (37, 150, 190),
    (48, 170, 95),
    (240, 160, 40),
    (145, 92, 190),
    (30, 180, 175),
    (220, 95, 170),
    (130, 155, 45),
    (95, 120, 205),
    (210, 115, 45),
)


def bbox_text(bbox):
    return "[" + ",".join(f"{value:.3f}" for value in bbox) + "]"


def bbox_center(bbox):
    return (
        int((bbox[0] + bbox[2]) / 2),
        int((bbox[1] + bbox[3]) / 2),
    )


def bbox_area(bbox):
    return max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def get_font(size, bold=False):
    paths = (
        ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/calibrib.ttf")
        if bold
        else ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf")
    )
    for path in paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_rgb(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def find_inputs(info):
    rgb_candidates = [
        os.path.join(DATA_DIR, "images", SEQUENCE, FRAME_ID + ext)
        for ext in (".tiff", ".tif", ".png", ".jpg", ".jpeg")
    ]
    if info.get("image_path"):
        rgb_candidates.append(info["image_path"])
    rgb_path = next((path for path in rgb_candidates if os.path.isfile(path)), None)
    depth_paths = glob.glob(
        os.path.join(DATA_DIR, "depth", SEQUENCE, FRAME_ID + ".tif*")
    )
    annotation_path = os.path.join(
        DATA_DIR, "annotations", SEQUENCE, FRAME_ID + ".pkl"
    )
    if rgb_path is None or not depth_paths or not os.path.isfile(annotation_path):
        raise FileNotFoundError(
            f"Missing input: rgb={rgb_path}, depth={depth_paths}, ann={annotation_path}"
        )
    return rgb_path, depth_paths[0], annotation_path


def mask_stats(gt, depth):
    mask = gt.get("instance_mask")
    if mask is None or getattr(mask, "shape", None) != depth.shape:
        return 0, 0, None
    area = int(np.count_nonzero(mask))
    values = depth[mask]
    valid = values[values > 0]
    median = None if valid.size == 0 else float(np.median(valid))
    return area, int(valid.size), median


def best_prediction(gt, predictions):
    if not predictions:
        return None, 0.0
    pairs = [
        (pred, float(iou(gt["bbox"], pred["bbox"])))
        for pred in predictions
    ]
    return max(pairs, key=lambda pair: pair[1])


def best_gt(prediction, gts):
    if not gts:
        return None, 0.0
    pairs = [
        (gt, float(iou(prediction["bbox"], gt["bbox"])))
        for gt in gts
    ]
    return max(pairs, key=lambda pair: pair[1])


def classify_detection(prediction, matched_record, best_iou):
    if prediction["confidence"] < CONF_THRESHOLD:
        return "LOW_CONF"
    if matched_record.get("gt_matched", False):
        return "MATCHED"
    if best_iou <= 0:
        return "NO_GT_OVERLAP"
    if best_iou < IOU_THRESHOLD:
        return "IOU_BELOW_0_5"
    return "GREEDY_1TO1_CONFLICT"


def classify_gt(best_pred, best_iou, matched_pred_id):
    if matched_pred_id is not None:
        return "MATCHED"
    if best_pred is None or best_iou <= 0:
        return "NO_OVERLAPPING_DETECTION"
    if best_pred["confidence"] < CONF_THRESHOLD and best_iou >= IOU_THRESHOLD:
        return "ONLY_LOW_CONF_DETECTION"
    if best_iou < IOU_THRESHOLD:
        return "YOLO_LOCALIZATION_IOU_BELOW_0_5"
    return "GREEDY_1TO1_CONFLICT"


def build_diagnostics(predictions, gts, depth):
    high_predictions = [
        dict(pred)
        for pred in predictions
        if pred["confidence"] >= CONF_THRESHOLD
    ]
    attach_mask_from_pkl(high_predictions, gts, IOU_THRESHOLD)
    high_by_id = {pred["id"]: pred for pred in high_predictions}

    matched_pred_by_gt = {
        pred["matched_gt_id"]: pred
        for pred in high_predictions
        if pred.get("gt_matched", False)
    }

    detection_rows = []
    detection_state = {}
    for pred in predictions:
        best, best_iou = best_gt(pred, gts)
        matched = high_by_id.get(pred["id"], {})
        reason = classify_detection(pred, matched, best_iou)
        row = {
            "pred_id": pred["id"],
            "conf": pred["confidence"],
            "pred_bbox": bbox_text(pred["bbox"]),
            "bbox_center_x": bbox_center(pred["bbox"])[0],
            "bbox_center_y": bbox_center(pred["bbox"])[1],
            "best_gt_id": None if best is None else best["id"],
            "best_iou": best_iou,
            "matched_gt_id": matched.get("matched_gt_id"),
            "matched": bool(matched.get("gt_matched", False)),
            "reason": reason,
        }
        detection_state[pred["id"]] = row
        if reason != "MATCHED":
            detection_rows.append(row)

    gt_rows = []
    for gt in gts:
        area, valid_count, median_depth = mask_stats(gt, depth)
        best_pred, best_iou = best_prediction(gt, predictions)
        matched_pred = matched_pred_by_gt.get(gt["id"])
        visibility = ""
        if best_pred is not None:
            visibility = min(1.0, area / bbox_area(best_pred["bbox"]))
        diagnosis = classify_gt(
            best_pred,
            best_iou,
            None if matched_pred is None else matched_pred["id"],
        )
        gt_rows.append(
            {
                "gt_id": gt["id"],
                "gt_bbox": bbox_text(gt["bbox"]),
                "gt_area": area,
                "visibility": visibility,
                "best_pred_id": None if best_pred is None else best_pred["id"],
                "best_iou": best_iou,
                "best_conf": None if best_pred is None else best_pred["confidence"],
                "matched": matched_pred is not None,
                "matched_pred_id": None if matched_pred is None else matched_pred["id"],
                "matched_iou": None if matched_pred is None else matched_pred["match_iou"],
                "semantic_label": gt.get("semantic_label"),
                "valid_mask_depth_pixels": valid_count,
                "mask_median_depth": median_depth,
                "diagnosis": diagnosis,
            }
        )
    return gt_rows, detection_rows, detection_state


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def overlay_gt_masks(rgb, gts):
    output = rgb.astype(np.float32).copy()
    alpha = 0.34
    for index, gt in enumerate(gts):
        mask = gt.get("instance_mask")
        if mask is None or getattr(mask, "shape", None) != rgb.shape[:2]:
            continue
        color = np.asarray(MASK_COLORS[index % len(MASK_COLORS)], dtype=np.float32)
        output[mask] = output[mask] * (1 - alpha) + color * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def draw_label(draw, bbox, text, color, line_width=3, font_size=13):
    width, height = draw._image.size
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
    y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
    draw.rectangle((x1, y1, x2, y2), outline=color + (255,), width=line_width)
    font = get_font(font_size, bold=True)
    bounds = draw.textbbox((0, 0), text, font=font)
    label_width = bounds[2] - bounds[0] + 8
    label_height = bounds[3] - bounds[1] + 6
    tx = max(0, min(width - label_width, x1))
    ty = y1 - label_height
    if ty < 0:
        ty = min(height - label_height, y2 + 1)
    draw.rectangle(
        (tx, ty, tx + label_width, ty + label_height),
        fill=color + (205,),
    )
    draw.text((tx + 4, ty + 3), text, fill=(255, 255, 255, 255), font=font)


def add_header(image, text):
    draw = ImageDraw.Draw(image, "RGBA")
    font = get_font(19, bold=True)
    height = draw.textbbox((0, 0), text, font=font)[3] + 13
    draw.rectangle((0, 0, image.width, height), fill=(0, 0, 0, 190))
    draw.text((8, 5), text, fill=(255, 255, 255, 255), font=font)


def make_images(rgb, gts, predictions, detection_state):
    gt_overlay = Image.fromarray(overlay_gt_masks(rgb, gts), "RGB")
    draw = ImageDraw.Draw(gt_overlay, "RGBA")
    for index, gt in enumerate(gts):
        color = MASK_COLORS[index % len(MASK_COLORS)]
        draw_label(
            draw,
            gt["bbox"],
            f"GT{gt['id']} {gt.get('semantic_label')}",
            color,
            line_width=3,
            font_size=13,
        )
    add_header(gt_overlay, "All GT instance masks and GT boxes")
    gt_overlay.save(OUTPUT_GT)

    combined = Image.fromarray(overlay_gt_masks(rgb, gts), "RGB")
    draw = ImageDraw.Draw(combined, "RGBA")
    for index, gt in enumerate(gts):
        color = MASK_COLORS[index % len(MASK_COLORS)]
        draw.rectangle(
            tuple(int(round(value)) for value in gt["bbox"]),
            outline=color + (220,),
            width=2,
        )
    for pred in predictions:
        state = detection_state[pred["id"]]
        if state["reason"] == "MATCHED":
            color, short = MATCHED_COLOR, "M"
        elif state["reason"] == "LOW_CONF":
            color, short = LOW_CONF_COLOR, "LOW"
        else:
            color, short = UNMATCHED_COLOR, "U"
        draw_label(
            draw,
            pred["bbox"],
            f"P{pred['id']} {pred['confidence']:.3f} {short}",
            color,
            line_width=2,
            font_size=12,
        )
    add_header(
        combined,
        "GT masks + YOLO: green=matched, orange=unmatched/low-IoU, gray=low confidence",
    )
    combined.save(OUTPUT_COMBINED)


def print_summary(rgb_path, gt_rows, detection_rows, predictions):
    print("[seq408 GT coverage and alignment diagnosis]")
    print(f"sequence/frame : {SEQUENCE}/{FRAME_ID}")
    print(f"RGB            : {rgb_path}")
    print(f"GT instances   : {len(gt_rows)}")
    print(f"YOLO detections: {len(predictions)}")
    print()
    print("GT summary")
    for row in gt_rows:
        print(
            f"GT{row['gt_id']} bbox={row['gt_bbox']} area={row['gt_area']} "
            f"best=P{row['best_pred_id']} IoU={row['best_iou']:.6f} "
            f"conf={row['best_conf']} matched={row['matched']} "
            f"diagnosis={row['diagnosis']}"
        )
    print()
    print("Unmatched / low-confidence detections")
    for row in detection_rows:
        print(
            f"P{row['pred_id']} center=({row['bbox_center_x']},{row['bbox_center_y']}) "
            f"conf={row['conf']:.6f} best_gt={row['best_gt_id']} "
            f"best_iou={row['best_iou']:.6f} reason={row['reason']}"
        )

    central_right = [
        row
        for row in detection_rows
        if row["bbox_center_x"] >= 240 and row["bbox_center_y"] >= 400
    ]
    print()
    print("Central/right visually relevant unmatched detections")
    for row in central_right:
        print(
            f"P{row['pred_id']} center=({row['bbox_center_x']},{row['bbox_center_y']}) "
            f"conf={row['conf']:.3f} best_iou={row['best_iou']:.3f} "
            f"reason={row['reason']}"
        )

    matched_ious = [row["matched_iou"] for row in gt_rows if row["matched"]]
    zero_overlap_central = [
        row for row in central_right if row["reason"] == "NO_GT_OVERLAP"
    ]
    localization_cases = [
        row for row in detection_rows if row["reason"] == "IOU_BELOW_0_5"
    ]
    print()
    print("Automatic conclusion")
    print(
        f"- Global RGB-GT alignment issue: unlikely; {len(matched_ious)}/{len(gt_rows)} "
        f"GTs matched, matched IoU range={min(matched_ious):.3f}-{max(matched_ious):.3f}."
    )
    print(
        f"- Annotation omission evidence: {len(zero_overlap_central)} central/right "
        "detections have exactly zero overlap with every GT."
    )
    print(
        f"- YOLO localization/threshold evidence: {len(localization_cases)} "
        "high-confidence detections overlap a GT but remain below IoU 0.5."
    )
    print(
        "Final visual judgment must use the overlay: a visible fruit with an "
        "orange zero-overlap box and no colored mask indicates annotation omission; "
        "a colored mask shifted away from the fruit indicates alignment; an orange "
        "box overlapping the correct mask below 0.5 indicates localization mismatch."
    )
    print()
    print(f"GT overlay CSV   : {OUTPUT_GT_CSV}")
    print(f"Unmatched CSV    : {OUTPUT_UNMATCHED_CSV}")
    print(f"GT image         : {OUTPUT_GT}")
    print(f"Combined overlay : {OUTPUT_COMBINED}")


def main():
    detections = load_detections_csv(DETECTION_CSV)
    info = detections.get((SEQUENCE, FRAME_ID))
    if info is None:
        raise RuntimeError("Target frame missing from detection CSV")
    rgb_path, depth_path, annotation_path = find_inputs(info)
    rgb = load_rgb(rgb_path)
    depth = clean_depth(load_depth(depth_path))
    gts = load_bupst20_annotation(annotation_path)
    predictions = [dict(pred) for pred in info["objects"]]

    gt_rows, detection_rows, detection_state = build_diagnostics(
        predictions, gts, depth
    )
    write_csv(
        OUTPUT_GT_CSV,
        gt_rows,
        [
            "gt_id",
            "gt_bbox",
            "gt_area",
            "visibility",
            "best_pred_id",
            "best_iou",
            "best_conf",
            "matched",
            "matched_pred_id",
            "matched_iou",
            "semantic_label",
            "valid_mask_depth_pixels",
            "mask_median_depth",
            "diagnosis",
        ],
    )
    write_csv(
        OUTPUT_UNMATCHED_CSV,
        detection_rows,
        [
            "pred_id",
            "conf",
            "pred_bbox",
            "bbox_center_x",
            "bbox_center_y",
            "best_gt_id",
            "best_iou",
            "matched_gt_id",
            "matched",
            "reason",
        ],
    )
    make_images(rgb, gts, predictions, detection_state)
    print_summary(rgb_path, gt_rows, detection_rows, predictions)


if __name__ == "__main__":
    main()
