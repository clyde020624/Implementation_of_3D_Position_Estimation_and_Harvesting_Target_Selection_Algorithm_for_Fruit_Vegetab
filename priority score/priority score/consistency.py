# -*- coding: utf-8 -*-
"""
====================================================================
 consistency.py - 위치 추정 일관성 평가 (새 연구 방향)
====================================================================

[방향 전환]
  기존: 알고리즘 Top-1 vs 사람 Top-1 비교 (evaluate.py)
  변경: 같은 파프리카의 3D 위치 추정이 프레임 간 얼마나 일관적인가

[핵심 아이디어]
  로봇이 움직이며 같은 파프리카(temporal ID 동일)를 여러 프레임에서 봄
  → 각 프레임에서 그 파프리카의 3D 위치(또는 거리)를 추정
  → 추정값들이 얼마나 흔들리는지(표준편차) 측정
  → 흔들림이 작을수록 "위치 추정이 신뢰할 만하다"

[사람 정답 필요 없음] → human_top1_id 삭제됨
====================================================================
"""

import numpy as np
from collections import defaultdict
from data_loader import pixel_to_3d


# ====================================================================
# 1) 프레임 시퀀스에서 '같은 파프리카별' 3D 위치를 모음
# --------------------------------------------------------------------
# frames: 같은 시퀀스의 프레임 리스트. 각 프레임은
#   { "objects": [...], "depth_map": ..., "img_size": (W,H) }
#   각 object는 temporal id를 "id"로 가짐 (프레임 넘어 같은 파프리카면 같은 id)
# cam: intrinsic (fx,fy,cx,cy)
# ====================================================================
def collect_positions_by_id(frames, cam, extract_depth_fn):
    """
    같은 id(파프리카)별로 여러 프레임의 3D 위치를 모아서 반환.
    반환: { id: [(X1,Y1,Z1), (X2,Y2,Z2), ...] }
    """
    positions = defaultdict(list)

    for frame in frames:
        depth_map = frame["depth_map"]
        for obj in frame["objects"]:
            # 박스 중심의 depth(Z) 추출
            Z = extract_depth_fn(obj["bbox"], depth_map)
            if Z is None:
                continue
            # 박스 중심 픽셀 좌표
            x1, y1, x2, y2 = obj["bbox"]
            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            # 3D 위치 계산 (intrinsic 사용)
            X, Y, Z = pixel_to_3d(u, v, Z, cam)
            positions[obj["id"]].append((X, Y, Z))

    return positions


# ====================================================================
# 2) 각 파프리카의 위치 추정 일관성(흔들림) 계산
# ====================================================================
def compute_consistency(positions, min_observations=2):
    """
    각 파프리카별로 3D 위치의 흔들림(표준편차)을 계산.
    - min_observations: 최소 몇 번 관찰돼야 일관성을 잴지 (2번 이상)
    반환: 파프리카별 결과 + 전체 요약
    """
    results = []
    for obj_id, pos_list in positions.items():
        if len(pos_list) < min_observations:
            continue  # 한 번만 보인 파프리카는 일관성 못 잼

        arr = np.array(pos_list)  # shape (관찰수, 3)  [X,Y,Z]

        # 각 축(X,Y,Z)의 표준편차 = 흔들림 정도
        std_x, std_y, std_z = arr.std(axis=0)
        # 3D 위치 전체의 흔들림 (유클리드)
        std_3d = np.sqrt(std_x**2 + std_y**2 + std_z**2)

        results.append({
            "id": obj_id,
            "관찰수": len(pos_list),
            "std_Z(거리)": round(std_z, 1),
            "std_3D": round(std_3d, 1),
        })

    return results


# ====================================================================
# 3) 전체 요약 통계
# ====================================================================
def summarize(results):
    if not results:
        return None
    std_3d_all = [r["std_3D"] for r in results]
    std_z_all = [r["std_Z(거리)"] for r in results]
    return {
        "분석된 파프리카 수": len(results),
        "평균 3D 흔들림(mm)": round(np.mean(std_3d_all), 1),
        "평균 거리 흔들림(mm)": round(np.mean(std_z_all), 1),
        "가장 불안정한 파프리카": max(results, key=lambda r: r["std_3D"]),
        "가장 안정적인 파프리카": min(results, key=lambda r: r["std_3D"]),
    }


# ====================================================================
# 4) 더미 데이터로 시연
# ====================================================================
if __name__ == "__main__":
    from priority import extract_depth

    # 가짜 intrinsic (BUP-ST20 실측값)
    cam = {"fx": 919.46, "fy": 920.65, "cx": 361.72, "cy": 636.79}

    # 같은 시퀀스의 3개 프레임 (로봇이 움직이며 같은 파프리카 관찰)
    # 파프리카 id=5는 안정적으로 추적됨, id=8은 3번째에 depth가 튐
    np.random.seed(3)
    def make_depth(base_vals):
        """특정 위치에 특정 depth를 심은 가짜 depth map"""
        d = np.random.randint(500, 2000, (1280, 720)).astype(np.uint16)
        for (cx, cy, val) in base_vals:
            d[cy-10:cy+10, cx-10:cx+10] = val
        return d

    frames = [
        {  # 프레임 1
            "objects": [
                {"id": 5, "bbox": [100, 200, 160, 280]},   # 중심 (130,240)
                {"id": 8, "bbox": [400, 300, 460, 380]},   # 중심 (430,340)
            ],
            "depth_map": make_depth([(130, 240, 850), (430, 340, 1200)]),
            "img_size": (720, 1280),
        },
        {  # 프레임 2 (파프리카 조금 이동, depth 비슷)
            "objects": [
                {"id": 5, "bbox": [110, 205, 170, 285]},   # 중심 (140,245)
                {"id": 8, "bbox": [410, 305, 470, 385]},   # 중심 (440,345)
            ],
            "depth_map": make_depth([(140, 245, 855), (440, 345, 1210)]),
            "img_size": (720, 1280),
        },
        {  # 프레임 3 (id=8의 depth가 튐 → 불안정)
            "objects": [
                {"id": 5, "bbox": [120, 210, 180, 290]},   # 중심 (150,250)
                {"id": 8, "bbox": [420, 310, 480, 390]},   # 중심 (450,350)
            ],
            "depth_map": make_depth([(150, 250, 860), (450, 350, 1650)]),  # 1650! 튐
            "img_size": (720, 1280),
        },
    ]

    # 1) 같은 파프리카별 3D 위치 모으기
    positions = collect_positions_by_id(frames, cam, extract_depth)

    # 2) 일관성 계산
    results = compute_consistency(positions)

    # 출력
    print("=" * 62)
    print(" 위치 추정 일관성 분석")
    print("=" * 62)
    for r in results:
        print(f" 파프리카 id={r['id']}: {r['관찰수']}번 관찰")
        print(f"    거리(Z) 흔들림: {r['std_Z(거리)']}mm  |  3D 흔들림: {r['std_3D']}mm")
        stable = "안정적 ✅" if r["std_3D"] < 100 else "불안정 ⚠️"
        print(f"    → {stable}")
        print("-" * 62)

    # 3) 요약
    s = summarize(results)
    print(f"\n [요약]")
    print(f"   분석된 파프리카: {s['분석된 파프리카 수']}개")
    print(f"   평균 거리 흔들림: {s['평균 거리 흔들림(mm)']}mm")
    print(f"   평균 3D 흔들림: {s['평균 3D 흔들림(mm)']}mm")
    print(f"   가장 불안정: id={s['가장 불안정한 파프리카']['id']} "
          f"({s['가장 불안정한 파프리카']['std_3D']}mm)")
    print(f"   가장 안정적: id={s['가장 안정적인 파프리카']['id']} "
          f"({s['가장 안정적인 파프리카']['std_3D']}mm)")