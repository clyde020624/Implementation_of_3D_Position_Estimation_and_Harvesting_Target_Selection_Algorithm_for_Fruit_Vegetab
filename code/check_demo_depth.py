import os
import csv
import pickle
import numpy as np

from data_loader import load_depth, clean_depth
from localization import depth_from_bbox_center


SEQ = "406"
FRAME_ID = "1600938551617024"

CSV_PATH = "eval_detections_fixed.csv"

DEPTH_PATH = os.path.join(
    "dataset_bulk", "depth", SEQ, FRAME_ID + ".tiff"
)

ANN_PATH = os.path.join(
    "dataset_bulk", "annotations", SEQ, FRAME_ID + ".pkl"
)

CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


# ------------------------------------------------------------
# depth 로드
# ------------------------------------------------------------
depth = clean_depth(load_depth(DEPTH_PATH))


# ------------------------------------------------------------
# GT annotation 직접 로드
# ------------------------------------------------------------
with open(ANN_PATH, "rb") as f:
    ann = pickle.load(f)

gt_objects = []

for gt_id, info in ann.items():

    if not isinstance(info, dict):
        continue

    bbox = info.get("bbox")
    mask = info.get("instance_mask")

    if bbox is None or mask is None:
        continue

    # GT bbox는 [x, y, w, h]
    x, y, w, h = map(float, bbox[:4])

    gt_objects.append({
        "id": gt_id,
        "bbox": [x, y, x + w, y + h],
        "mask": mask,
    })


# ------------------------------------------------------------
# YOLO 후보 읽기
# ------------------------------------------------------------
objects = []

with open(CSV_PATH, encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:

        if row["sequence_id"] != SEQ:
            continue

        if row["frame_id"] != FRAME_ID:
            continue

        conf = float(row["confidence"])

        if conf < CONF_THRESHOLD:
            continue

        objects.append({
            "id": int(row["candidate_id"]),
            "bbox": [
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            ],
            "confidence": conf,
        })


# ------------------------------------------------------------
# 후보별 GT 매칭 + center-in-mask 검사
# ------------------------------------------------------------
for obj in objects:

    bbox = obj["bbox"]

    z, (cx, cy) = depth_from_bbox_center(
        bbox, depth, region_half=2
    )

    # 가장 IoU가 높은 GT 찾기
    best_gt = None
    best_iou = 0.0

    for gt in gt_objects:

        score = iou(bbox, gt["bbox"])

        if score > best_iou:
            best_iou = score
            best_gt = gt

    print()
    print("=" * 70)
    print(f"candidate ID : {obj['id']}")
    print(f"confidence   : {obj['confidence']:.3f}")
    print(f"bbox center  : ({cx}, {cy})")
    print(f"center depth : {z:.1f} mm")

    if best_gt is None or best_iou < IOU_THRESHOLD:

        print("matched GT   : 없음")
        print("center in GT mask : 확인 불가")

        continue

    mask = best_gt["mask"]

    # 좌표 범위 체크
    if (
        0 <= cy < mask.shape[0]
        and
        0 <= cx < mask.shape[1]
    ):
        center_in_mask = bool(mask[cy, cx])
    else:
        center_in_mask = False

    # GT mask 전체 depth median도 같이 계산
    mask_depth = depth[mask]

    valid_mask_depth = mask_depth[
        mask_depth > 0
    ]

    if len(valid_mask_depth) > 0:
        gt_mask_median = float(
            np.median(valid_mask_depth)
        )
    else:
        gt_mask_median = None

    print(f"matched GT ID : {best_gt['id']}")
    print(f"IoU           : {best_iou:.3f}")
    print(f"center in GT mask : {center_in_mask}")

    if gt_mask_median is not None:
        print(f"GT mask depth median : {gt_mask_median:.1f} mm")
        print(
            f"depth difference     : "
            f"{z - gt_mask_median:+.1f} mm"
        )