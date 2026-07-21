# -*- coding: utf-8 -*-
"""
====================================================================
 실제 데이터 불러오기 (depth 파일 + YOLO 검출 결과)
====================================================================

이 파일은 '실제 데이터셋'을 priority.py가 이해하는 형식으로 바꿔줍니다.

데이터셋마다 파일 형식이 다르므로, 흔한 경우들을 모두 대응합니다:
  - depth 파일: .png(16bit), .npy 둘 다 지원
  - 검출 라벨: YOLO txt 형식 지원 (민혁의 YOLO 출력)

[중요] 데이터셋이 확정되면 아래 함수들이 실제 파일 경로를 받아
       priority.py의 select_top1()에 바로 넣을 수 있는 형태로 변환합니다.
====================================================================
"""

import numpy as np
import cv2   # OpenCV: 이미지/depth 파일 읽기용


# ====================================================================
# 1) depth 파일 불러오기 (.png 또는 .npy 자동 판별)
# ====================================================================
def load_depth(depth_path):
    """
    depth 파일을 numpy 2D 배열로 불러옵니다.
      - .npy  : np.load 로 그대로
      - .png  : 16bit depth 이미지 (RealSense가 보통 이 형식)
    반환: (H, W) 형태의 numpy 배열
    """
    if depth_path.endswith(".npy"):
        depth = np.load(depth_path)
    else:
        # cv2.IMREAD_UNCHANGED: 16bit 값을 그대로 읽음 (안 하면 8bit로 뭉개짐)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"depth 파일을 못 읽음: {depth_path}")

    # 혹시 3채널로 읽히면 1채널로
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth


def inspect_depth(depth):
    """
    ★ 데이터셋 확정 시 가장 먼저 실행할 함수 ★
    depth 값의 범위를 출력해서 metric(mm)인지 상대값(0~255)인지 판단합니다.
      - 값이 0~255 범위면  -> 상대 depth (visibility 모드 권장)
      - 값이 수백~수천이면 -> mm 단위 metric depth (거리 기반 주장 가능)
    """
    valid = depth[depth > 0]
    print("─" * 50)
    print(" [depth 진단]")
    print(f"   shape       : {depth.shape}")
    print(f"   dtype       : {depth.dtype}")
    print(f"   최소값(0제외): {valid.min()}")
    print(f"   최대값      : {valid.max()}")
    print(f"   중앙값      : {np.median(valid):.1f}")
    if valid.max() <= 255:
        print("   판정: 0~255 범위 → 상대 depth 가능성 높음 (visibility 모드 권장)")
    else:
        print("   판정: 255 초과 → mm 단위 metric depth 가능성 (거리 기반 주장 가능)")
    print("─" * 50)
    return valid.min(), valid.max()


# ====================================================================
# 2) YOLO 검출 결과(txt) 불러오기
# --------------------------------------------------------------------
# YOLO는 보통 이런 형식의 txt를 뱉습니다 (한 줄 = 객체 하나):
#   class_id  x_center  y_center  width  height  [confidence]
#   (좌표는 0~1로 정규화된 값)
# 이걸 priority.py가 쓰는 픽셀 bbox [x1,y1,x2,y2] 형식으로 변환합니다.
# ====================================================================
def load_yolo_detections(label_path, img_w, img_h, class_names=None):
    """
    YOLO txt 라벨을 priority.py용 객체 리스트로 변환.
    class_names: {0: "unripe", 1: "ripe"} 같은 매핑 (성숙도 모드용, 없으면 생략)
    """
    objects = []
    with open(label_path, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) > 5 else 1.0

            # 정규화 좌표(0~1) → 픽셀 좌표로 변환
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h

            obj = {
                "id": i + 1,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
            }
            # 성숙도 라벨이 있으면 class 이름 붙이기
            if class_names and cls_id in class_names:
                obj["class"] = class_names[cls_id]

            objects.append(obj)
    return objects


# ====================================================================
# 3) 테스트: 가짜 파일을 만들어 로딩이 되는지 확인
# ====================================================================
if __name__ == "__main__":
    # --- 가짜 depth 파일 2종 만들어서 로딩 테스트 ---
    fake_depth = np.random.randint(300, 2000, (480, 640)).astype(np.uint16)  # mm 가정
    np.save("_test_depth.npy", fake_depth)
    cv2.imwrite("_test_depth.png", fake_depth)

    print(">> .npy 로딩 테스트")
    d1 = load_depth("_test_depth.npy")
    inspect_depth(d1)

    print("\n>> .png 로딩 테스트")
    d2 = load_depth("_test_depth.png")
    inspect_depth(d2)

    # --- 가짜 YOLO 라벨 파일 만들어서 로딩 테스트 ---
    with open("_test_label.txt", "w") as f:
        f.write("1 0.3 0.4 0.1 0.2 0.91\n")   # ripe
        f.write("0 0.7 0.6 0.08 0.15 0.75\n")  # unripe
    print("\n>> YOLO 라벨 로딩 테스트")
    objs = load_yolo_detections("_test_label.txt", 640, 480,
                                class_names={0: "unripe", 1: "ripe"})
    for o in objs:
        print("  ", o)

    # 정리
    import os
    for fn in ["_test_depth.npy", "_test_depth.png", "_test_label.txt"]:
        os.remove(fn)
    print("\n테스트 완료 (임시 파일 삭제됨)")
