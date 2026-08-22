import os
import glob
import csv
import pickle
from collections import defaultdict

import numpy as np

from data_loader import load_depth, clean_depth
from localization import depth_from_bbox_center


# ============================================================
# 설정
# ============================================================

DATA_DIR = "dataset_bulk"
DETECTION_CSV = "eval_detections_fixed.csv"

SEQUENCES = {str(i) for i in range(400, 410)}

CONF_THRESHOLD = 0.30
IOU_THRESHOLD = 0.50

# 기존 Full Priority Score
W_DISTANCE = 0.35
W_VISIBILITY = 0.30
W_CONFIDENCE = 0.20
W_CENTER = 0.15


# ============================================================
# IoU
# ============================================================

def calc_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area1 = max(0.0, box1[2] - box1[0]) * max(
        0.0, box1[3] - box1[1]
    )
    area2 = max(0.0, box2[2] - box2[0]) * max(
        0.0, box2[3] - box2[1]
    )

    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


# ============================================================
# Detection CSV 로드
# ============================================================

def load_detections():
    frames = defaultdict(list)

    with open(DETECTION_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:

            seq = str(row["sequence_id"]).strip()
            frame_id = str(row["frame_id"]).strip()

            if seq not in SEQUENCES:
                continue

            conf = float(row["confidence"])

            if conf < CONF_THRESHOLD:
                continue

            obj = {
                "id": int(float(row["candidate_id"])),
                "bbox": [
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                ],
                "confidence": conf,
                "image_width": int(float(row["image_width"])),
                "image_height": int(float(row["image_height"])),
            }

            frames[(seq, frame_id)].append(obj)

    return frames


# ============================================================
# GT pkl 직접 로드
# ============================================================

def load_gt(pkl_path):

    with open(pkl_path, "rb") as f:
        ann = pickle.load(f)

    gts = []

    for gt_id, info in ann.items():

        if not isinstance(info, dict):
            continue

        bbox = info.get("bbox")
        mask = info.get("instance_mask")

        if bbox is None or mask is None:
            continue

        x, y, w, h = map(float, bbox[:4])

        gts.append({
            "id": gt_id,
            "bbox": [
                x,
                y,
                x + w,
                y + h,
            ],
            "mask": mask,
        })

    return gts


# ============================================================
# prediction ↔ GT greedy 1:1 matching
# ============================================================

def greedy_match(preds, gts):

    pairs = []

    for pi, pred in enumerate(preds):

        for gi, gt in enumerate(gts):

            value = calc_iou(
                pred["bbox"],
                gt["bbox"]
            )

            if value >= IOU_THRESHOLD:
                pairs.append(
                    (value, pi, gi)
                )

    pairs.sort(reverse=True)

    used_pred = set()
    used_gt = set()

    matches = []

    for value, pi, gi in pairs:

        if pi in used_pred:
            continue

        if gi in used_gt:
            continue

        used_pred.add(pi)
        used_gt.add(gi)

        matches.append(
            (preds[pi], gts[gi], value)
        )

    return matches


# ============================================================
# GT mask median depth
# ============================================================

def get_mask_depth(mask, depth):

    if mask is None:
        return None

    if mask.shape != depth.shape:
        return None

    values = depth[mask]

    values = values[
        values > 0
    ]

    if values.size == 0:
        return None

    return float(
        np.median(values)
    )


# ============================================================
# 후보 하나 진단
# ============================================================

def analyze_candidate(pred, gt, depth):

    # 현재 연구에서 사용하는 bbox center 5x5 depth
    z_center, center = depth_from_bbox_center(
        pred["bbox"],
        depth,
        region_half=2
    )

    if z_center is None:
        return None

    cx, cy = center

    mask = gt["mask"]

    if (
        0 <= cy < mask.shape[0]
        and
        0 <= cx < mask.shape[1]
    ):
        center_in_mask = bool(
            mask[cy, cx]
        )
    else:
        center_in_mask = False

    z_mask = get_mask_depth(
        mask,
        depth
    )

    if z_mask is None:
        return None

    x1, y1, x2, y2 = pred["bbox"]

    bbox_area = max(
        1.0,
        (x2 - x1) * (y2 - y1)
    )

    mask_area = float(
        np.count_nonzero(mask)
    )

    # 기존 visibility proxy
    visibility = min(
        1.0,
        mask_area / bbox_area
    )

    return {
        "pred": pred,
        "gt": gt,

        "z_center": float(z_center),
        "z_mask": float(z_mask),

        "depth_diff": float(
            z_center - z_mask
        ),

        "center_in_mask": center_in_mask,

        "visibility": visibility,

        "cx": cx,
        "cy": cy,
    }


# ============================================================
# 기존 Full Priority Score 계산
# ============================================================

def select_full(candidates):

    if not candidates:
        return None

    z_values = [
        c["z_center"]
        for c in candidates
    ]

    z_min = min(z_values)
    z_max = max(z_values)

    for c in candidates:

        # ----------------------------------
        # 거리 점수
        # ----------------------------------

        if z_max > z_min:
            proximity = (
                z_max - c["z_center"]
            ) / (z_max - z_min)
        else:
            proximity = 1.0

        # ----------------------------------
        # confidence
        # ----------------------------------

        confidence = c["pred"][
            "confidence"
        ]

        # ----------------------------------
        # 화면 중심성
        # ----------------------------------

        img_w = c["pred"][
            "image_width"
        ]

        img_h = c["pred"][
            "image_height"
        ]

        x1, y1, x2, y2 = c["pred"][
            "bbox"
        ]

        obj_x = (x1 + x2) / 2
        obj_y = (y1 + y2) / 2

        img_cx = img_w / 2
        img_cy = img_h / 2

        dist = np.sqrt(
            (obj_x - img_cx) ** 2
            +
            (obj_y - img_cy) ** 2
        )

        center_score = (
            1.0 / (1.0 + dist / 100.0)
        )

        # ----------------------------------
        # 최종 Score
        # ----------------------------------

        score = (
            W_DISTANCE * proximity
            +
            W_VISIBILITY * c["visibility"]
            +
            W_CONFIDENCE * confidence
            +
            W_CENTER * center_score
        )

        c["priority_score"] = score

    return max(
        candidates,
        key=lambda x: x["priority_score"]
    )


# ============================================================
# depth / annotation 파일 찾기
# ============================================================

def find_file(root, seq, frame_id, extensions):

    folder = os.path.join(
        root,
        seq
    )

    for ext in extensions:

        path = os.path.join(
            folder,
            frame_id + ext
        )

        if os.path.exists(path):
            return path

    return None


# ============================================================
# MAIN
# ============================================================

detections = load_detections()

total_matched = 0
total_valid = 0

outside_count = 0

diff20 = 0
diff50 = 0
diff100 = 0

usable_frames = 0

nearest_outside = 0
full_outside = 0

nearest_order_changed = 0

old_nearest_vs_full_diff = 0

problem_frames = []


for (seq, frame_id), preds in detections.items():

    depth_path = find_file(
        os.path.join(DATA_DIR, "depth"),
        seq,
        frame_id,
        [".tiff", ".tif"]
    )

    ann_path = find_file(
        os.path.join(DATA_DIR, "annotations"),
        seq,
        frame_id,
        [".pkl"]
    )

    if depth_path is None:
        continue

    if ann_path is None:
        continue

    depth = clean_depth(
        load_depth(depth_path)
    )

    gts = load_gt(
        ann_path
    )

    matches = greedy_match(
        preds,
        gts
    )

    total_matched += len(matches)

    candidates = []

    for pred, gt, match_iou in matches:

        result = analyze_candidate(
            pred,
            gt,
            depth
        )

        if result is None:
            continue

        result["iou"] = match_iou

        candidates.append(result)

        total_valid += 1

        if not result[
            "center_in_mask"
        ]:
            outside_count += 1

        abs_diff = abs(
            result["depth_diff"]
        )

        if abs_diff > 20:
            diff20 += 1

        if abs_diff > 50:
            diff50 += 1

        if abs_diff > 100:
            diff100 += 1


    # Top-1 비교는 후보 2개 이상
    if len(candidates) < 2:
        continue

    usable_frames += 1


    # --------------------------------------------------------
    # ① 현재 bbox-center 기준 nearest
    # --------------------------------------------------------

    nearest_center = min(
        candidates,
        key=lambda x: x["z_center"]
    )


    # --------------------------------------------------------
    # ② GT mask depth를 이용한 진단용 nearest
    # --------------------------------------------------------

    nearest_mask = min(
        candidates,
        key=lambda x: x["z_mask"]
    )


    # --------------------------------------------------------
    # ③ 기존 Full
    # --------------------------------------------------------

    full = select_full(
        candidates
    )


    # --------------------------------------------------------
    # 통계
    # --------------------------------------------------------

    if not nearest_center[
        "center_in_mask"
    ]:
        nearest_outside += 1

    if not full[
        "center_in_mask"
    ]:
        full_outside += 1


    # 현재 nearest와 mask-depth nearest 순위가 바뀌었는지
    if (
        nearest_center["pred"]["id"]
        !=
        nearest_mask["pred"]["id"]
    ):

        nearest_order_changed += 1

        problem_frames.append({
            "seq": seq,
            "frame": frame_id,

            "old_id":
                nearest_center["pred"]["id"],

            "old_z":
                nearest_center["z_center"],

            "old_mask_z":
                nearest_center["z_mask"],

            "old_inside":
                nearest_center["center_in_mask"],

            "mask_nearest_id":
                nearest_mask["pred"]["id"],

            "mask_nearest_z":
                nearest_mask["z_mask"],
        })


    # 기존 실험 검산
    if (
        nearest_center["pred"]["id"]
        !=
        full["pred"]["id"]
    ):
        old_nearest_vs_full_diff += 1


# ============================================================
# 결과 출력
# ============================================================

print()
print("=" * 72)
print(" 전체 거리 추정 품질 진단")
print("=" * 72)

print()
print("[1] matched candidate 전체")
print(f"  IoU 매칭 수             : {total_matched}")
print(f"  depth까지 유효한 수     : {total_valid}")

if total_valid > 0:

    print(
        f"  center_in_mask=False    : "
        f"{outside_count} "
        f"({outside_count / total_valid * 100:.1f}%)"
    )

    print(
        f"  |center-mask depth| >20 : "
        f"{diff20} "
        f"({diff20 / total_valid * 100:.1f}%)"
    )

    print(
        f"  |center-mask depth| >50 : "
        f"{diff50} "
        f"({diff50 / total_valid * 100:.1f}%)"
    )

    print(
        f"  |center-mask depth|>100 : "
        f"{diff100} "
        f"({diff100 / total_valid * 100:.1f}%)"
    )


print()
print("[2] Top-1 분석")

print(
    f"  비교 가능 프레임       : "
    f"{usable_frames}"
)

if usable_frames > 0:

    print(
        f"  nearest Top-1 center outside : "
        f"{nearest_outside}/{usable_frames} "
        f"({nearest_outside / usable_frames * 100:.1f}%)"
    )

    print(
        f"  Full Top-1 center outside    : "
        f"{full_outside}/{usable_frames} "
        f"({full_outside / usable_frames * 100:.1f}%)"
    )

    print(
        f"  bbox-depth nearest와 "
        f"mask-depth nearest 불일치 : "
        f"{nearest_order_changed}/{usable_frames} "
        f"({nearest_order_changed / usable_frames * 100:.1f}%)"
    )

    print(
        f"  기존 nearest vs Full 불일치 : "
        f"{old_nearest_vs_full_diff}/{usable_frames} "
        f"({old_nearest_vs_full_diff / usable_frames * 100:.1f}%)"
    )


print()
print("[3] 기존 실험 검산")

print(
    "  기존 기록 예상: "
    "비교 가능 364프레임 / nearest-Full 차이 103프레임"
)

print(
    f"  이번 진단 결과: "
    f"{usable_frames}프레임 / "
    f"{old_nearest_vs_full_diff}프레임"
)


print()
print("[4] 거리 순위가 바뀐 예시 — 최대 20개")

for r in problem_frames[:20]:

    print(
        f"  seq={r['seq']} "
        f"frame={r['frame']} | "
        f"기존 nearest ID={r['old_id']} "
        f"centerZ={r['old_z']:.0f} "
        f"maskZ={r['old_mask_z']:.0f} "
        f"inMask={r['old_inside']} | "
        f"mask-nearest ID={r['mask_nearest_id']} "
        f"maskZ={r['mask_nearest_z']:.0f}"
    )