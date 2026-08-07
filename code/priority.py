# -*- coding: utf-8 -*-
"""
====================================================================
 파프리카 수확 대상 선정 - Priority Score 알고리즘 (유연 버전)
====================================================================

실행: python priority.py

[핵심] 데이터셋에 성숙도(ripe/unripe) 라벨이 있는지 아직 확정 전이므로,
       스위치 하나(MODE)로 두 경우를 전환할 수 있게 만들었습니다.

  MODE = "ripeness"    -> 성숙도 라벨이 있을 때
         Priority = 거리 + 신뢰도 + 중심 + 성숙도

  MODE = "visibility"  -> 성숙도 라벨이 없을 때
         Priority = 거리 + 신뢰도 + 중심 + 가시성(가림 정도)

데이터셋이 확정되면 MODE 값만 바꾸면 됩니다. 코드 본체는 안 바꿔도 됩니다.

[입력 형식] 객체(파프리카) 하나 = 딕셔너리 (민혁의 YOLO 출력과 맞출 형식)
  {
    "id": 1,
    "bbox": [x1, y1, x2, y2],   # 박스 좌표 (픽셀)
    "confidence": 0.87,          # YOLO 검출 신뢰도 0~1
    "class": "ripe",             # (성숙도 모드일 때만 사용) ripe/unripe
    "mask_area": 3200,           # (가시성 모드일 때만 사용) 마스크 픽셀 수
  }
  * 사용하지 않는 키는 없어도 됩니다. 모드에 따라 필요한 것만 쓰입니다.
====================================================================
"""

import numpy as np


# ====================================================================
# 설정: 여기만 바꾸면 됩니다
# ====================================================================
# 성숙도 라벨이 있으면 "ripeness", 없으면 "visibility"
MODE = "visibility"     # <- 데이터셋 확정되면 이 값만 변경

# 각 모드별 가중치 (합이 1이 되도록 맞춤)
WEIGHTS = {
    "ripeness": {   # 성숙도 라벨이 있을 때
        "proximity":  0.40,   # 거리 근접도 (가까울수록 우선)
        "confidence": 0.20,   # 검출 신뢰도
        "center":     0.20,   # 화면 중심 근접
        "ripeness":   0.20,   # 성숙도 (ripe=1, unripe=0)
    },
    "visibility": {  # 성숙도 라벨이 없을 때 (성숙도 자리에 가시성)
        "proximity":  0.35,   # 거리 근접도
        "confidence": 0.20,   # 검출 신뢰도
        "center":     0.15,   # 화면 중심 근접
        "visibility": 0.30,   # 가시성 (가려지지 않고 잘 보일수록 우선)
    },
}


# ====================================================================
# 거리(Depth) 추출: 박스 중심 주변 영역의 중앙값(median) 사용
# (중심 한 점은 구멍/노이즈일 수 있어 영역의 median이 안정적)
# ====================================================================
def extract_depth(bbox, depth_map, region_half=2):
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    h, w = depth_map.shape
    y_start, y_end = max(0, cy - region_half), min(h, cy + region_half + 1)
    x_start, x_end = max(0, cx - region_half), min(w, cx + region_half + 1)
    region = depth_map[y_start:y_end, x_start:x_end]

    valid = region[region > 0]      # 0(구멍) 제외
    if len(valid) == 0:
        return None
    return float(np.median(valid))


# ====================================================================
# 정규화: 값을 0~1로 맞춤. 거리는 '가까울수록 높게' 뒤집음.
# depth가 실제 미터든 상대값(0~255)이든 상관없이 작동
# (한 이미지 안에서 '상대적으로 누가 더 가까운가'만 보기 때문)
# ====================================================================
def normalize_proximity(depth_value, min_d, max_d):
    if max_d == min_d:
        return 1.0
    return (max_d - depth_value) / (max_d - min_d)


# ====================================================================
# 개별 점수 계산 함수들 (항목별로 분리 -> 읽기 쉽고 실험하기 좋음)
# ====================================================================
def score_center(bbox, img_center):
    """화면 중앙에 가까울수록 1에 가까움"""
    x1, y1, x2, y2 = bbox
    obj_cx, obj_cy = (x1 + x2) / 2, (y1 + y2) / 2
    dist = ((obj_cx - img_center[0]) ** 2 +
            (obj_cy - img_center[1]) ** 2) ** 0.5
    return 1 / (1 + dist / 100)


def score_ripeness(obj):
    """익음: ripe=1, 그 외=0"""
    return 1.0 if obj.get("class") == "ripe" else 0.0


def score_visibility(obj):
    """
    GT mask 기반 occupancy proxy.
    mask_area가 없는 객체는 0으로 처리한다.
    """
    if "mask_area" not in obj or obj["mask_area"] is None:
        return 0.0

    x1, y1, x2, y2 = obj["bbox"]
    bbox_area = max(1, (x2 - x1) * (y2 - y1))

    return min(1.0, obj["mask_area"] / bbox_area)


# ====================================================================
# Priority Score: MODE에 따라 항목 구성이 자동으로 바뀜
# ====================================================================
def priority_score(obj, min_d, max_d, weights, img_center, mode):
    p_score = normalize_proximity(obj["depth_value"], min_d, max_d)
    c_score = obj["confidence"]
    center_score = score_center(obj["bbox"], img_center)

    # 공통 항목
    score = (weights["proximity"]  * p_score
           + weights["confidence"] * c_score
           + weights["center"]     * center_score)

    detail = {
        "proximity": round(p_score, 3),
        "confidence": round(c_score, 3),
        "center": round(center_score, 3),
    }

    # 모드별 항목 (성숙도 or 가시성)
    if mode == "ripeness":
        r_score = score_ripeness(obj)
        score += weights["ripeness"] * r_score
        detail["ripeness"] = round(r_score, 3)
    elif mode == "visibility":
        v_score = score_visibility(obj)
        score += weights["visibility"] * v_score
        detail["visibility"] = round(v_score, 3)

    obj["_detail"] = detail
    return score


# ====================================================================
# Top-1 선정: 모든 객체에 점수 매기고 최고점 하나 선택
# ====================================================================
def select_top1(objects, depth_map, weights, img_size, mode):
    img_center = (img_size[0] / 2, img_size[1] / 2)

    for obj in objects:
        obj["depth_value"] = extract_depth(obj["bbox"], depth_map)

    valid = [o for o in objects if o["depth_value"] is not None]
    if not valid:
        return None, []

    depths = [o["depth_value"] for o in valid]
    min_d, max_d = min(depths), max(depths)

    for obj in valid:
        obj["score"] = priority_score(obj, min_d, max_d, weights, img_center, mode)

    top1 = max(valid, key=lambda o: o["score"])
    return top1, valid


# ====================================================================
# 결과 출력 헬퍼
# ====================================================================
def print_result(top1, scored, mode):
    print("=" * 62)
    print(f" Priority Score 결과   (MODE = '{mode}')")
    print("=" * 62)
    for o in sorted(scored, key=lambda x: x["score"], reverse=True):
        d = o["_detail"]
        extra_key = "ripeness" if mode == "ripeness" else "visibility"
        print(f" 객체 {o['id']}  |  총점 {o['score']:.3f}")
        print(f"    depth={o['depth_value']:.0f}  conf={o['confidence']}")
        detail_str = (f"근접도={d['proximity']}  신뢰도={d['confidence']}  "
                      f"중심={d['center']}  {extra_key}={d[extra_key]}")
        print(f"    세부: {detail_str}")
        print("-" * 62)
    print(f"\n  >>> 1순위 수확 대상: 객체 {top1['id']} "
          f"(총점 {top1['score']:.3f}) <<<\n")


# ====================================================================
# 더미 데이터로 실행 (실제 데이터가 오면 이 부분만 교체)
# ====================================================================
if __name__ == "__main__":
    np.random.seed(42)
    dummy_depth = np.random.randint(50, 200, (480, 640))
    dummy_depth[100:150, 380:460] = 70    # 3번: 가까움
    dummy_depth[50:140,  100:180] = 150   # 1번: 중간
    dummy_depth[200:270, 300:360] = 190   # 2번: 멂

    # 더미 검출 결과 (성숙도용 class, 가시성용 mask_area 둘 다 넣어둠)
    # -> MODE에 따라 필요한 것만 자동으로 쓰임
    dummy_objects = [
        {"id": 1, "bbox": [100,  50, 180, 140], "confidence": 0.91,
         "class": "ripe",   "mask_area": 6500},   # 박스대비 마스크 큼(잘 보임)
        {"id": 2, "bbox": [300, 200, 360, 270], "confidence": 0.75,
         "class": "ripe",   "mask_area": 2000},   # 마스크 작음(많이 가려짐)
        {"id": 3, "bbox": [380, 100, 460, 190], "confidence": 0.88,
         "class": "unripe", "mask_area": 6000},   # 잘 보이지만 안 익음
    ]

    weights = WEIGHTS[MODE]

    top1, scored = select_top1(
        dummy_objects, dummy_depth, weights, img_size=(640, 480), mode=MODE
    )
    print_result(top1, scored, MODE)

    # ----------------------------------------------------------------
    # 참고: 아래 주석을 풀면 두 모드를 한 번에 비교해볼 수 있습니다.
    # ----------------------------------------------------------------
    # for m in ("ripeness", "visibility"):
    #     objs = [dict(o) for o in dummy_objects]   # 원본 보존용 복사
    #     t1, sc = select_top1(objs, dummy_depth, WEIGHTS[m],
    #                          img_size=(640, 480), mode=m)
    #     print_result(t1, sc, m)