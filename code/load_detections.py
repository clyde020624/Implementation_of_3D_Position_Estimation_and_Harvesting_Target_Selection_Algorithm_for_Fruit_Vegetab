# -*- coding: utf-8 -*-
"""
====================================================================
 load_detections.py - 민혁의 YOLO 검출 CSV 불러오기
====================================================================

민혁의 06_export_detections.py가 만든 eval_detections.csv를
priority.py가 쓰는 형식으로 변환합니다.

[CSV 형식] (06번 스크립트 기준)
  sequence_id, frame_id, image_path, candidate_id, class_id,
  class_name, confidence, x1, y1, x2, y2, center_x, center_y,
  box_width, box_height, image_width, image_height

[핵심]
  - CSV 한 파일에 '모든 프레임의 모든 검출'이 들어있음
  - sequence_id + frame_id 로 같은 프레임의 검출들을 묶어야 함
  - 검출형(박스만)이라 마스크 없음 → 가시성은 별도 처리 (아래 참고)

[마스크(가시성) 문제]
  민혁이 검출형(박스만)으로 학습 → YOLO 결과에 mask_area 없음
  → 두 가지 방법:
    방법 A: 원본 pkl의 instance_mask를 매칭 (attach_mask_from_pkl 사용)
    방법 B: 가시성 빼고 거리+신뢰도+중심만 (MODE 조정)
====================================================================
"""

import csv
from collections import defaultdict


# ====================================================================
# 1) 검출 CSV를 프레임별로 묶어서 불러오기
# ====================================================================
def load_detections_csv(csv_path):
    """
    민혁의 검출 CSV를 프레임별 dict로 변환.
    반환: { (sequence_id, frame_id): {
              "objects": [priority.py용 객체 리스트],
              "img_size": (W, H),
              "image_path": ...,
           } }
    """
    frames = defaultdict(lambda: {"objects": [], "img_size": None, "image_path": None})

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["sequence_id"], row["frame_id"])

            # 프레임 메타정보 (한 번만 설정)
            if frames[key]["img_size"] is None:
                W = int(row["image_width"])
                H = int(row["image_height"])
                frames[key]["img_size"] = (W, H)
                frames[key]["image_path"] = row["image_path"]

            # 검출 후보 하나 → priority.py용 객체로 변환
            obj = {
                "id": int(row["candidate_id"]),
                "bbox": [
                    float(row["x1"]), float(row["y1"]),
                    float(row["x2"]), float(row["y2"]),
                ],
                "confidence": float(row["confidence"]),
                "class": row["class_name"],   # 'pepper' (1클래스)
            }
            frames[key]["objects"].append(obj)

    return dict(frames)


# ====================================================================
# 2) 검출 CSV → main.py의 dataset 형식으로 변환
#    (depth 맵과 연결해서 평가 가능한 형태로)
# ====================================================================
def build_dataset_from_csv(csv_path, depth_dir, load_depth_fn, clean_depth_fn,
                           depth_suffix=".tiff"):
    """
    검출 CSV + depth 폴더를 매칭해서 dataset 리스트 생성.
    depth 파일 경로 규칙: depth_dir / sequence_id / frame_id.tiff
    (BUP-ST20 실제 구조 기준. 다르면 이 부분만 수정)
    """
    import os

    frames = load_detections_csv(csv_path)
    dataset = []

    for (seq_id, frame_id), info in frames.items():
        # depth 파일 경로 조립 (실제 구조에 맞게 조정 가능)
        depth_path = os.path.join(depth_dir, seq_id, frame_id + depth_suffix)
        if not os.path.exists(depth_path):
            continue  # depth 없는 프레임은 건너뜀

        depth = clean_depth_fn(load_depth_fn(depth_path))
        dataset.append({
            "image_id": f"{seq_id}_{frame_id}",
            "sequence_id": seq_id,
            "frame_id": frame_id,
            "objects": info["objects"],
            "depth_map": depth,
            "img_size": info["img_size"],
        })

    return dataset


# ====================================================================
# 3) (방법 A) 원본 pkl의 instance_mask를 검출 결과에 매칭
# --------------------------------------------------------------------
# 검출형 YOLO엔 마스크가 없으므로, 가시성을 쓰려면 원본 GT mask가 필요.
# YOLO 검출 박스와 pkl의 GT 박스를 IoU로 매칭해서 mask_area를 붙임.
# ====================================================================
def attach_mask_from_pkl(objects, gt_objects, iou_threshold=0.5):
    """
    YOLO prediction과 GT를 IoU 기준 1:1로 매칭하여
    GT의 mask_area를 prediction에 부여한다.

    gt_matched:
      True  = GT와 정상 매칭
      False = GT와 매칭되지 않음
    """

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

    # 모든 prediction 초기화
    for obj in objects:
        obj.pop("mask_area", None)
        obj["gt_matched"] = False
        obj["matched_gt_id"] = None
        obj["match_iou"] = 0.0

    # 가능한 모든 IoU pair 생성
    pairs = []

    for pi, pred in enumerate(objects):
        for gi, gt in enumerate(gt_objects):
            value = iou(pred["bbox"], gt["bbox"])

            if value >= iou_threshold:
                pairs.append((value, pi, gi))

    # IoU 높은 순
    pairs.sort(reverse=True)

    used_pred = set()
    used_gt = set()

    # 1:1 greedy matching
    for value, pi, gi in pairs:

        if pi in used_pred or gi in used_gt:
            continue

        pred = objects[pi]
        gt = gt_objects[gi]

        pred["mask_area"] = gt.get("mask_area")
        pred["gt_matched"] = True
        pred["matched_gt_id"] = gt.get("id")
        pred["match_iou"] = value

        used_pred.add(pi)
        used_gt.add(gi)

    return objects


# ====================================================================
# 4) 테스트: 가짜 CSV 만들어서 불러오기 확인
# ====================================================================
if __name__ == "__main__":
    import os

    # 민혁 형식의 가짜 CSV 생성
    fake_csv = "_test_det.csv"
    with open(fake_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sequence_id","frame_id","image_path","candidate_id","class_id",
            "class_name","confidence","x1","y1","x2","y2","center_x","center_y",
            "box_width","box_height","image_width","image_height"])
        writer.writeheader()
        # 프레임 400의 파프리카 2개
        writer.writerow({"sequence_id":"400","frame_id":"1600","image_path":"a.tiff",
            "candidate_id":0,"class_id":0,"class_name":"pepper","confidence":0.91,
            "x1":100,"y1":50,"x2":180,"y2":140,"center_x":140,"center_y":95,
            "box_width":80,"box_height":90,"image_width":720,"image_height":1280})
        writer.writerow({"sequence_id":"400","frame_id":"1600","image_path":"a.tiff",
            "candidate_id":1,"class_id":0,"class_name":"pepper","confidence":0.75,
            "x1":300,"y1":200,"x2":360,"y2":270,"center_x":330,"center_y":235,
            "box_width":60,"box_height":70,"image_width":720,"image_height":1280})
        # 프레임 401의 파프리카 1개
        writer.writerow({"sequence_id":"400","frame_id":"1601","image_path":"b.tiff",
            "candidate_id":0,"class_id":0,"class_name":"pepper","confidence":0.88,
            "x1":380,"y1":100,"x2":460,"y2":190,"center_x":420,"center_y":145,
            "box_width":80,"box_height":90,"image_width":720,"image_height":1280})

    print(">> CSV 불러오기 테스트")
    frames = load_detections_csv(fake_csv)
    for key, info in frames.items():
        print(f"\n 프레임 {key}: 검출 {len(info['objects'])}개, "
              f"이미지크기 {info['img_size']}")
        for o in info["objects"]:
            print(f"    id={o['id']} bbox={o['bbox']} conf={o['confidence']} "
                  f"class={o['class']}")

    print("\n>> 방법 A: pkl 마스크 매칭 테스트")
    # 검출 객체 (마스크 없음)
    dets = frames[("400","1600")]["objects"]
    # GT 객체 (마스크 있음) - 첫 번째와 위치 비슷
    gts = [
        {"bbox": [102, 52, 182, 142], "mask_area": 5200},  # id=0과 겹침
        {"bbox": [305, 205, 365, 275], "mask_area": 1800},  # id=1과 겹침
    ]
    attach_mask_from_pkl(dets, gts)
    for o in dets:
        print(f"    id={o['id']}: mask_area={o.get('mask_area', '없음(1.0처리)')}")

    os.remove(fake_csv)
    print("\n테스트 완료")
