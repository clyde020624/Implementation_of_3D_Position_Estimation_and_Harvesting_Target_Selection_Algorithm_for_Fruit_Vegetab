"""Trace every selection stage for sequence 408 / frame 1600938562015928.

This diagnostic reuses the current detection loader, greedy GT matcher,
candidate preparation, depth extraction, and Priority Score implementation.
It does not modify any core research file.
"""

import glob
import os
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data_loader import clean_depth, load_bupst20_annotation, load_depth
from load_detections import attach_mask_from_pkl, load_detections_csv
from localization import depth_from_bbox_center, iou
from mask_depth_reanalysis import (
    FULL_WEIGHTS,
    IOU_THRESHOLD,
    MODE,
    get_mask_median_depth,
    prepare_candidates,
    select_top1_precomputed,
)
from priority import normalize_proximity, score_center, score_visibility


SEQUENCE = "408"
FRAME_ID = "1600938562015928"
DATA_DIR = "dataset_bulk"
DETECTION_CSV = "eval_detections_fixed.csv"
CONF_THRESHOLD = 0.30

OUTPUT_ALL_DETECTIONS = "seq408_all_yolo_detections.png"
OUTPUT_ALL_CANDIDATES = "seq408_all_selection_candidates.png"
OUTPUT_FINAL = "seq408_nearest_vs_full.png"

COLOR_HIGH = (45, 160, 90)
COLOR_LOW = (125, 125, 125)
COLOR_UNMATCHED = (230, 135, 30)
COLOR_CANDIDATE = (224, 164, 35)
COLOR_NEAREST = (210, 48, 40)
COLOR_FULL = (31, 119, 180)


def fmt(value, digits=3):
    if value is None:
        return "None"
    return f"{float(value):.{digits}f}"


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def find_paths(info):
    depth_paths = glob.glob(
        os.path.join(DATA_DIR, "depth", SEQUENCE, FRAME_ID + ".tif*")
    )
    annotation_path = os.path.join(
        DATA_DIR, "annotations", SEQUENCE, FRAME_ID + ".pkl"
    )
    rgb_candidates = [
        os.path.join(DATA_DIR, "images", SEQUENCE, FRAME_ID + ext)
        for ext in (".tiff", ".tif", ".png", ".jpg", ".jpeg")
    ]
    if info.get("image_path"):
        rgb_candidates.append(info["image_path"])
    rgb_path = next((path for path in rgb_candidates if os.path.isfile(path)), None)
    if not depth_paths:
        raise FileNotFoundError("Depth file not found")
    if not os.path.isfile(annotation_path):
        raise FileNotFoundError("Annotation file not found")
    if rgb_path is None:
        raise FileNotFoundError(f"RGB file not found; checked {rgb_candidates}")
    return rgb_path, depth_paths[0], annotation_path


def mask_statistics(gt, depth):
    mask = gt.get("instance_mask")
    if mask is None or getattr(mask, "shape", None) != depth.shape:
        return 0, 0, None
    mask_pixels = int(np.count_nonzero(mask))
    values = depth[mask]
    valid = values[values > 0]
    median = None if valid.size == 0 else float(np.median(valid))
    return mask_pixels, int(valid.size), median


def best_gt_for_prediction(pred, gts):
    if not gts:
        return None, 0.0
    pairs = [(gt, float(iou(pred["bbox"], gt["bbox"]))) for gt in gts]
    return max(pairs, key=lambda pair: pair[1])


def best_prediction_for_gt(gt, predictions):
    if not predictions:
        return None, 0.0
    pairs = [
        (pred, float(iou(pred["bbox"], gt["bbox"])))
        for pred in predictions
    ]
    return max(pairs, key=lambda pair: pair[1])


def vertical_region(bbox, image_height):
    cy = (bbox[1] + bbox[3]) / 2
    if cy >= image_height * 0.60:
        return "LOWER"
    if cy >= image_height * 0.25:
        return "MIDDLE"
    return "UPPER"


def match_and_filter(all_predictions, gts, depth):
    high_predictions = [
        dict(pred)
        for pred in all_predictions
        if pred["confidence"] >= CONF_THRESHOLD
    ]
    attach_mask_from_pkl(high_predictions, gts, IOU_THRESHOLD)
    matched_by_id = {pred["id"]: pred for pred in high_predictions}

    prepared = prepare_candidates(
        [dict(pred) for pred in all_predictions if pred["confidence"] >= CONF_THRESHOLD],
        gts,
        depth,
    )
    candidate_by_id = {candidate["id"]: candidate for candidate in prepared}
    gt_by_id = {gt["id"]: gt for gt in gts}

    trace = {}
    for pred in all_predictions:
        pred_id = pred["id"]
        best_gt, best_iou = best_gt_for_prediction(pred, gts)
        record = {
            "pred": pred,
            "best_gt_id": None if best_gt is None else best_gt["id"],
            "best_iou": best_iou,
            "matched": False,
            "matched_gt_id": None,
            "match_iou": 0.0,
            "candidate": candidate_by_id.get(pred_id),
            "reason": None,
        }

        if pred["confidence"] < CONF_THRESHOLD:
            record["reason"] = "LOW_CONF"
        else:
            matched = matched_by_id[pred_id]
            record["matched"] = bool(matched.get("gt_matched", False))
            record["matched_gt_id"] = matched.get("matched_gt_id")
            record["match_iou"] = float(matched.get("match_iou", 0.0))
            if not record["matched"]:
                record["reason"] = (
                    "BEST_IOU_BELOW_0.50"
                    if best_iou < IOU_THRESHOLD
                    else "GREEDY_1TO1_CONFLICT"
                )
            elif record["candidate"] is None:
                gt = gt_by_id.get(record["matched_gt_id"])
                bbox_depth, _ = depth_from_bbox_center(
                    pred["bbox"], depth, region_half=2
                )
                if gt is None:
                    record["reason"] = "MATCHED_GT_NOT_FOUND"
                elif bbox_depth is None:
                    record["reason"] = "INVALID_BBOX_DEPTH"
                elif get_mask_median_depth(
                    gt.get("instance_mask"), depth
                ) is None:
                    record["reason"] = "INVALID_MASK_DEPTH"
                else:
                    record["reason"] = "OTHER_PREPARE_CANDIDATE_EXCLUSION"
            else:
                record["reason"] = "CANDIDATE"
        trace[pred_id] = record

    return high_predictions, prepared, trace


def score_candidates(candidates, image_size):
    full, scored = select_top1_precomputed(
        candidates,
        "mask_depth",
        FULL_WEIGHTS,
        image_size,
        MODE,
    )
    if full is None:
        raise RuntimeError("No Full Top-1")
    min_depth = min(candidate["mask_depth"] for candidate in candidates)
    max_depth = max(candidate["mask_depth"] for candidate in candidates)
    nearest_set = [
        candidate for candidate in candidates if candidate["mask_depth"] == min_depth
    ]
    image_center = (image_size[0] / 2, image_size[1] / 2)

    details = {}
    for candidate in scored:
        proximity = float(
            normalize_proximity(candidate["mask_depth"], min_depth, max_depth)
        )
        visibility = float(score_visibility(candidate))
        center = float(score_center(candidate["bbox"], image_center))
        confidence = float(candidate["confidence"])
        contributions = {
            "distance": FULL_WEIGHTS["proximity"] * proximity,
            "visibility": FULL_WEIGHTS["visibility"] * visibility,
            "confidence": FULL_WEIGHTS["confidence"] * confidence,
            "center": FULL_WEIGHTS["center"] * center,
        }
        details[candidate["id"]] = {
            "candidate": candidate,
            "proximity": proximity,
            "visibility": visibility,
            "center": center,
            "confidence": confidence,
            "contributions": contributions,
            "score": float(candidate["score"]),
        }
    return full, nearest_set, details


def gt_coverage_status(gt, all_predictions, trace, candidate_ids):
    best_pred, best_iou = best_prediction_for_gt(gt, all_predictions)
    if best_pred is None or best_iou < IOU_THRESHOLD:
        return "NO_YOLO_DETECTION_IOU_GE_0.50", best_pred, best_iou
    pred_trace = trace[best_pred["id"]]
    if best_pred["confidence"] < CONF_THRESHOLD:
        return "DETECTED_BUT_LOW_CONF", best_pred, best_iou
    matched_to_gt = [
        record
        for record in trace.values()
        if record["matched_gt_id"] == gt["id"] and record["matched"]
    ]
    if not matched_to_gt:
        return "NOT_SELECTED_BY_GREEDY_MATCHING", best_pred, best_iou
    matched_pred_id = matched_to_gt[0]["pred"]["id"]
    if matched_pred_id not in candidate_ids:
        return trace[matched_pred_id]["reason"], best_pred, best_iou
    return "VALID_SELECTION_CANDIDATE", best_pred, best_iou


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
    return Image.fromarray(image)


def get_font(size, bold=False):
    candidates = (
        ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/calibrib.ttf")
        if bold
        else ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf")
    )
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_box_label(image, bbox, color, label, line_width=3, font_size=14):
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
    y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
    draw.rectangle((x1, y1, x2, y2), outline=color + (255,), width=line_width)
    font = get_font(font_size, bold=True)
    bounds = draw.multiline_textbbox((0, 0), label, font=font, spacing=2)
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
    draw.multiline_text(
        (tx + 4, ty + 3), label, fill=(255, 255, 255, 255), font=font, spacing=2
    )


def add_header(image, text):
    draw = ImageDraw.Draw(image, "RGBA")
    font = get_font(20, bold=True)
    bounds = draw.textbbox((0, 0), text, font=font)
    header_height = bounds[3] - bounds[1] + 14
    draw.rectangle((0, 0, image.width, header_height), fill=(0, 0, 0, 190))
    draw.text((8, 6), text, fill=(255, 255, 255, 255), font=font)


def make_visualizations(rgb_path, all_predictions, trace, candidates, nearest_set, full):
    all_image = load_rgb(rgb_path)
    for pred in all_predictions:
        record = trace[pred["id"]]
        if pred["confidence"] < CONF_THRESHOLD:
            color, status = COLOR_LOW, "LOW"
        elif not record["matched"]:
            color, status = COLOR_UNMATCHED, "UNMATCHED"
        else:
            color, status = COLOR_HIGH, "PASS"
        draw_box_label(
            all_image,
            pred["bbox"],
            color,
            f"P{pred['id']} {pred['confidence']:.3f} {status}",
            line_width=2,
            font_size=12,
        )
    add_header(all_image, "All YOLO detections: green=matched, orange=unmatched, gray=low confidence")
    all_image.save(OUTPUT_ALL_DETECTIONS)

    candidate_image = load_rgb(rgb_path)
    for candidate in candidates:
        draw_box_label(
            candidate_image,
            candidate["bbox"],
            COLOR_CANDIDATE,
            (
                f"P{candidate['id']} / GT{candidate['matched_gt_id']}\n"
                f"C {candidate['confidence']:.3f} Z {candidate['mask_depth']:.0f} "
                f"V {score_visibility(candidate):.3f}"
            ),
            line_width=3,
            font_size=12,
        )
    add_header(candidate_image, "All valid selection candidates")
    candidate_image.save(OUTPUT_ALL_CANDIDATES)

    final_image = load_rgb(rgb_path)
    for candidate in nearest_set:
        draw_box_label(
            final_image,
            candidate["bbox"],
            COLOR_NEAREST,
            (
                f"Nearest P{candidate['id']} / GT{candidate['matched_gt_id']}\n"
                f"Z {candidate['mask_depth']:.0f} V {score_visibility(candidate):.3f}"
            ),
            line_width=6,
            font_size=17,
        )
    draw_box_label(
        final_image,
        full["bbox"],
        COLOR_FULL,
        (
            f"Full P{full['id']} / GT{full['matched_gt_id']}\n"
            f"Z {full['mask_depth']:.0f} V {score_visibility(full):.3f}"
        ),
        line_width=6,
        font_size=17,
    )
    add_header(final_image, "GT-mask nearest vs Full Priority")
    final_image.save(OUTPUT_FINAL)


def print_report(
    rgb_path,
    depth_path,
    annotation_path,
    all_predictions,
    gts,
    depth,
    trace,
    candidates,
    full,
    nearest_set,
    score_details,
    image_size,
):
    print("\n[1] Frame basic info")
    print(f"sequence             : {SEQUENCE}")
    print(f"frame_id             : {FRAME_ID}")
    print(f"RGB path             : {rgb_path}")
    print(f"depth path           : {depth_path}")
    print(f"GT annotation exists : {os.path.isfile(annotation_path)}")
    print(f"YOLO prediction count: {len(all_predictions)}")

    print("\n[2] All YOLO detections")
    print("pred bbox center confidence conf>=0.30")
    for pred in all_predictions:
        print(
            f"P{pred['id']:>2} bbox={pred['bbox']} center={bbox_center(pred['bbox'])} "
            f"conf={pred['confidence']:.6f} pass={pred['confidence'] >= CONF_THRESHOLD}"
        )

    print("\n[3] GT objects")
    print("gt bbox mask_pixels semantic valid_depth_pixels mask_median_depth")
    for gt in gts:
        mask_pixels, valid_count, median = mask_statistics(gt, depth)
        print(
            f"GT{gt['id']} bbox={gt['bbox']} mask_pixels={mask_pixels} "
            f"semantic={gt.get('semantic_label')} valid_depth={valid_count} "
            f"maskZ={fmt(median, 1)}"
        )

    print("\n[4] YOLO-GT greedy 1:1 matching")
    print("pred matched_gt match_iou matched best_gt best_iou reason")
    for pred in all_predictions:
        record = trace[pred["id"]]
        print(
            f"P{pred['id']:>2} matched_gt={record['matched_gt_id']} "
            f"match_iou={record['match_iou']:.6f} matched={record['matched']} "
            f"best_gt={record['best_gt_id']} best_iou={record['best_iou']:.6f} "
            f"reason={record['reason']}"
        )

    print("\n[4b] GT coverage diagnosis")
    candidate_ids = {candidate["id"] for candidate in candidates}
    gt_status = {}
    for gt in gts:
        status, best_pred, best_iou = gt_coverage_status(
            gt, all_predictions, trace, candidate_ids
        )
        gt_status[gt["id"]] = status
        print(
            f"GT{gt['id']} region={vertical_region(gt['bbox'], image_size[1])} "
            f"center={bbox_center(gt['bbox'])} status={status} "
            f"best_pred={None if best_pred is None else best_pred['id']} "
            f"best_conf={None if best_pred is None else fmt(best_pred['confidence'])} "
            f"best_iou={best_iou:.6f}"
        )

    print("\n[5] Selection filtering")
    for pred in all_predictions:
        record = trace[pred["id"]]
        print(
            f"P{pred['id']:>2} candidate={record['candidate'] is not None} "
            f"reason={record['reason']}"
        )

    print("\n[6] Final candidate table")
    header = (
        "pred gt conf center in_mask bboxZ maskZ diff visibility proximity "
        "center_score conf_score D_con V_con C_con X_con FullScore nearest Full"
    )
    print(header)
    nearest_ids = {candidate["id"] for candidate in nearest_set}
    for candidate in candidates:
        detail = score_details[candidate["id"]]
        contributions = detail["contributions"]
        print(
            f"P{candidate['id']} GT{candidate['matched_gt_id']} "
            f"{candidate['confidence']:.6f} {candidate['bbox_center']} "
            f"{candidate['center_in_mask']} {candidate['bbox_depth']:.1f} "
            f"{candidate['mask_depth']:.1f} "
            f"{candidate['depth_difference']:+.1f} {detail['visibility']:.6f} "
            f"{detail['proximity']:.6f} {detail['center']:.6f} "
            f"{detail['confidence']:.6f} {contributions['distance']:.6f} "
            f"{contributions['visibility']:.6f} "
            f"{contributions['confidence']:.6f} {contributions['center']:.6f} "
            f"{detail['score']:.6f} {candidate['id'] in nearest_ids} "
            f"{candidate['id'] == full['id']}"
        )

    representative = min(
        nearest_set,
        key=lambda candidate: (-score_visibility(candidate), candidate["id"]),
    )
    near_detail = score_details[representative["id"]]
    full_detail = score_details[full["id"]]
    print("\n[7] Selection result")
    print(
        "nearest set: "
        + ", ".join(
            f"P{candidate['id']}/GT{candidate['matched_gt_id']}"
            for candidate in nearest_set
        )
    )
    print(
        f"nearest representative: P{representative['id']}/GT{representative['matched_gt_id']} "
        f"depth={representative['mask_depth']:.1f} "
        f"visibility={near_detail['visibility']:.6f} "
        f"confidence={near_detail['confidence']:.6f} "
        f"center={near_detail['center']:.6f} score={near_detail['score']:.6f}"
    )
    print(
        f"Full selected: P{full['id']}/GT{full['matched_gt_id']} "
        f"depth={full['mask_depth']:.1f} visibility={full_detail['visibility']:.6f} "
        f"confidence={full_detail['confidence']:.6f} "
        f"center={full_detail['center']:.6f} score={full_detail['score']:.6f}"
    )
    print("score contributions (weighted)")
    for name, detail in (("Nearest", near_detail), ("Full", full_detail)):
        c = detail["contributions"]
        print(
            f"  {name}: Distance={c['distance']:.6f} "
            f"Visibility={c['visibility']:.6f} "
            f"Confidence={c['confidence']:.6f} Center={c['center']:.6f} "
            f"Total={detail['score']:.6f}"
        )
    print("Full minus Nearest weighted contributions")
    for component in ("distance", "visibility", "confidence", "center"):
        difference = (
            full_detail["contributions"][component]
            - near_detail["contributions"][component]
        )
        print(f"  {component}: {difference:+.6f}")
    print(
        "Explanation: Full accepts the distance-score loss when gains from "
        "visibility/confidence/center make its total Priority Score larger."
    )

    print("\n[8] Visualization files")
    print(f"all YOLO detections      : {OUTPUT_ALL_DETECTIONS}")
    print(f"all selection candidates : {OUTPUT_ALL_CANDIDATES}")
    print(f"nearest vs Full          : {OUTPUT_FINAL}")

    print("\n[9] Final diagnostic summary")
    reason_counts = Counter(record["reason"] for record in trace.values())
    print(f"prediction flow counts: {dict(reason_counts)}")
    print(f"valid selection candidates: {len(candidates)}")
    status_counts = Counter(gt_status.values())
    print(f"GT coverage counts: {dict(status_counts)}")
    print("Middle/lower GT objects and their pipeline outcome:")
    for gt in gts:
        region = vertical_region(gt["bbox"], image_size[1])
        if region in ("MIDDLE", "LOWER"):
            print(
                f"  GT{gt['id']} region={region} center={bbox_center(gt['bbox'])} "
                f"-> {gt_status[gt['id']]}"
            )
    print(
        "Interpretation: GTs marked NO_YOLO_DETECTION_IOU_GE_0.50 have no "
        "corresponding detection box at IoU>=0.5; DETECTED_BUT_LOW_CONF are "
        "threshold losses; matching/depth reasons are reported explicitly; "
        "VALID_SELECTION_CANDIDATE objects were considered but may lose on score."
    )


def main():
    detections = load_detections_csv(DETECTION_CSV)
    info = detections.get((SEQUENCE, FRAME_ID))
    if info is None:
        raise RuntimeError("Frame missing from detection CSV")
    rgb_path, depth_path, annotation_path = find_paths(info)
    depth = clean_depth(load_depth(depth_path))
    gts = load_bupst20_annotation(annotation_path)
    all_predictions = [dict(pred) for pred in info["objects"]]

    _, candidates, trace = match_and_filter(all_predictions, gts, depth)
    full, nearest_set, score_details = score_candidates(
        candidates, info["img_size"] or (720, 1280)
    )
    make_visualizations(
        rgb_path, all_predictions, trace, candidates, nearest_set, full
    )
    print_report(
        rgb_path,
        depth_path,
        annotation_path,
        all_predictions,
        gts,
        depth,
        trace,
        candidates,
        full,
        nearest_set,
        score_details,
        info["img_size"] or (720, 1280),
    )


if __name__ == "__main__":
    main()
