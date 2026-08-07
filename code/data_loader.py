# -*- coding: utf-8 -*-
"""
====================================================================
 실제 데이터 불러오기 - BUP-ST20 파프리카 데이터셋 전용
====================================================================

BUP-ST20 실측 형식 (직접 확인한 구조):
  - depth      : .tiff, uint16, RealSense (65535 = invalid → 제외)
  - annotation : .pkl, object_id별 dict
                 각 객체: bbox([x,y,w,h]), area, instance_mask, semantic_label
  - cam_params : cam_params.yaml (intrinsics: fx,fy,cx,cy)

이 파일은 위 형식을 priority.py의 select_top1()이 이해하는
객체 리스트 형태로 변환합니다.

[변환 결과 객체 형식]
  {
    "id": 3,
    "bbox": [x1, y1, x2, y2],   # 픽셀 (좌상단, 우하단)
    "confidence": 1.0,          # GT는 1.0, YOLO 검출이면 실제 conf
    "mask_area": 358,           # instance_mask의 픽셀 수 (가시성 계산용)
    "semantic_label": "red",    # 색상 라벨 (성숙도 아님! 보조 정보로만)
  }
====================================================================
"""

import numpy as np
import pickle
import yaml

# tiff 읽기: tifffile 우선, 없으면 cv2로 대체
try:
    import tifffile
    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False
import cv2


# BUP-ST20 depth의 invalid 값 (uint16 최대값)
INVALID_DEPTH = 65535


# ====================================================================
# 1) depth 파일 불러오기 (.tiff / .png / .npy 모두 지원)
# ====================================================================
def load_depth(depth_path):
    """
    depth 파일을 numpy 2D 배열(uint16)로 불러옵니다.
    BUP-ST20은 .tiff 형식입니다.
    """
    if depth_path.endswith(".npy"):
        depth = np.load(depth_path)
    elif depth_path.endswith((".tif", ".tiff")):
        if _HAS_TIFFFILE:
            depth = tifffile.imread(depth_path)
        else:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    else:  # .png 등
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise FileNotFoundError(f"depth 파일을 못 읽음: {depth_path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth


def clean_depth(depth):
    """
    BUP-ST20 depth 정제: invalid 값(65535)과 0을 무효 처리.
    무효 픽셀은 0으로 바꿔서, 이후 extract_depth에서 자동 제외되게 함.
    (priority.py의 extract_depth는 0을 무효로 보고 median에서 제외함)
    """
    depth = depth.copy()
    depth[depth == INVALID_DEPTH] = 0
    return depth


def inspect_depth(depth):
    """
    ★ 데이터 확인용 진단 함수 ★
    depth 값 범위를 보고 metric인지 상대값인지 판단.
    BUP-ST20이면 uint16 + 수백~수천 값 → mm 단위 metric으로 판정됨.
    """
    valid = depth[(depth > 0) & (depth != INVALID_DEPTH)]
    print("─" * 52)
    print(" [depth 진단]")
    print(f"   shape        : {depth.shape}")
    print(f"   dtype        : {depth.dtype}")
    if len(valid) == 0:
        print("   유효값 없음 (전부 0 또는 invalid)")
        print("─" * 52)
        return None, None
    p1, p50, p99 = np.percentile(valid, [1, 50, 99])
    print(f"   유효 최소값  : {valid.min()}")
    print(f"   유효 최대값  : {valid.max()}")
    print(f"   percentile 1/50/99 : [{p1:.0f}, {p50:.0f}, {p99:.0f}]")
    if valid.max() <= 255:
        print("   판정: 0~255 → 상대 depth (visibility 모드 권장)")
    else:
        print("   판정: uint16 metric depth 추정 (거리 기반 주장 가능)")
        print(f"          중앙값 {p50:.0f} → mm로 보면 약 {p50/1000:.2f}m (온실 거리로 타당)")
    print("─" * 52)
    return valid.min(), valid.max()


# ====================================================================
# 2) 카메라 파라미터(cam_params.yaml) 불러오기
# ====================================================================
def load_cam_params(yaml_path):
    """
    cam_params.yaml에서 intrinsic(fx,fy,cx,cy)을 딕셔너리로 반환.
    BUP-ST20 형식: intrinsics가 3x3 행렬로 저장됨.
      [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    K = data["intrinsics"]
    return {
        "fx": float(K[0][0]),
        "fy": float(K[1][1]),
        "cx": float(K[0][2]),
        "cy": float(K[1][2]),
    }


def pixel_to_3d(u, v, depth_value, cam):
    """
    픽셀(u,v)+depth를 카메라 좌표계 3D 위치(X,Y,Z)로 변환.
    (선택 기능: 3D 위치 점수를 쓰고 싶을 때 사용)
      X = (u - cx) * Z / fx
      Y = (v - cy) * Z / fy
      Z = depth
    """
    Z = depth_value
    X = (u - cam["cx"]) * Z / cam["fx"]
    Y = (v - cam["cy"]) * Z / cam["fy"]
    return X, Y, Z


# ====================================================================
# 3) BUP-ST20 annotation(.pkl) 불러오기
# --------------------------------------------------------------------
# pkl 구조: object_id별 dict. 각 객체는
#   { "bbox": [x, y, w, h], "area": ..., "instance_mask": bool array,
#     "semantic_label": "red" 등 }
# 이걸 priority.py용 객체 리스트로 변환.
# ====================================================================
def load_bupst20_annotation(pkl_path):
    """
    BUP-ST20 pkl annotation을 priority.py용 객체 리스트로 변환.
    - bbox는 [x, y, w, h] → [x1, y1, x2, y2]로 변환
    - instance_mask의 True 픽셀 수를 mask_area로 (가시성 계산용)
    - confidence는 GT라 1.0
    - semantic_label은 '색상' 정보로만 보관 (성숙도 아님)
    """
    with open(pkl_path, "rb") as f:
        ann = pickle.load(f)

    objects = []
    for obj_id, info in ann.items():
        # --- 방어: 비정상 레코드는 건너뜀 (IndexError 방지) ---
        if not isinstance(info, dict):
            continue
        bbox = info.get("bbox")
        if bbox is None or not hasattr(bbox, "__len__") or len(bbox) < 4:
            continue
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

        # mask 픽셀 수 (가시성 계산용). area 키가 있으면 그것도 활용
        if "instance_mask" in info and info["instance_mask"] is not None:
            mask_area = int(np.count_nonzero(info["instance_mask"]))
        else:
            mask_area = int(info.get("area", w * h))

        objects.append({
            "id": obj_id,
            "bbox": [x, y, x + w, y + h],        # x,y,w,h → x1,y1,x2,y2
            "confidence": 1.0,                    # GT 라벨
            "mask_area": mask_area,
               # ★ 추가
            "instance_mask": info.get("instance_mask", None),
            "semantic_label": info.get("semantic_label", None),  # 색상(보조)
        })
    return objects


# ====================================================================
# 4) YOLO 검출 결과(txt) 불러오기 - 민혁이 YOLO 학습 후 쓸 경우
#    (BUP-ST20 GT 대신 실제 YOLO 예측을 쓸 때)
# ====================================================================
def load_yolo_detections(label_path, img_w, img_h, class_names=None):
    objects = []
    with open(label_path, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) > 5 else 1.0
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            obj = {"id": i + 1, "bbox": [x1, y1, x2, y2], "confidence": conf}
            if class_names and cls_id in class_names:
                obj["class"] = class_names[cls_id]
            objects.append(obj)
    return objects


# ====================================================================
# 5) 테스트: BUP-ST20 형식의 가짜 파일을 만들어 로딩 확인
# ====================================================================
if __name__ == "__main__":
    import os

    print(">> [1] BUP-ST20 형식 depth(.tiff) 로딩 + 정제 테스트")
    # 실측과 비슷하게: 대부분 500~2000, 일부 invalid(65535), 일부 0
    fake = np.random.randint(500, 2000, (1280, 720)).astype(np.uint16)
    fake[0:50, 0:50] = 65535     # invalid 영역
    fake[100:120, 100:120] = 0   # 구멍
    if _HAS_TIFFFILE:
        tifffile.imwrite("_t.tiff", fake)
    else:
        cv2.imwrite("_t.tiff", fake)
    d = load_depth("_t.tiff")
    inspect_depth(d)
    d_clean = clean_depth(d)
    print(f"   정제 후 invalid(65535) 개수: {np.count_nonzero(d_clean == 65535)} (0이어야 정상)")

    print("\n>> [2] cam_params.yaml 로딩 테스트")
    fake_yaml = {
        "intrinsics": [[919.456, 0.0, 361.72], [0.0, 920.65, 636.79], [0.0, 0.0, 1.0]],
        "extrinsics": [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],
    }
    with open("_cam.yaml", "w") as f:
        yaml.dump(fake_yaml, f)
    cam = load_cam_params("_cam.yaml")
    print(f"   intrinsic: {cam}")
    X, Y, Z = pixel_to_3d(400, 600, 849, cam)
    print(f"   픽셀(400,600) depth=849 → 3D: X={X:.1f}, Y={Y:.1f}, Z={Z:.1f} (mm)")

    print("\n>> [3] BUP-ST20 annotation(.pkl) 로딩 테스트")
    fake_ann = {
        1: {"bbox": [635, 1051, 13, 39], "area": 358,
            "instance_mask": np.zeros((1280, 720), dtype=bool),
            "semantic_label": "mixed_red"},
        2: {"bbox": [100, 200, 40, 60], "area": 1500,
            "instance_mask": np.ones((60, 40), dtype=bool),
            "semantic_label": "green"},
    }
    fake_ann[1]["instance_mask"][1051:1090, 635:648] = True  # 일부 True
    with open("_ann.pkl", "wb") as f:
        pickle.dump(fake_ann, f)
    objs = load_bupst20_annotation("_ann.pkl")
    for o in objs:
        print(f"   id={o['id']} bbox={o['bbox']} mask_area={o['mask_area']} "
              f"color={o['semantic_label']}")

    # 정리
    for fn in ["_t.tiff", "_cam.yaml", "_ann.pkl"]:
        if os.path.exists(fn):
            os.remove(fn)
    print("\n테스트 완료 (임시 파일 삭제됨)")