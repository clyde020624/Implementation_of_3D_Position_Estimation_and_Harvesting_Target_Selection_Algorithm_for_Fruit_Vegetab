# -*- coding: utf-8 -*-
"""
====================================================================
 main.py - 전체 파이프라인 (BUP-ST20, 위치 추정 일관성 방향)
====================================================================

파일 구성:
   data_loader.py  (데이터 불러오기: depth/pkl/cam_params)
   priority.py     (Priority Score + Top-1 선정)
   consistency.py  (위치 추정 일관성 분석) ← 현재 연구 방향

[사용 흐름]
  1. USE_REAL_DATA = False → 더미로 파이프라인 확인 (지금)
  2. 실제 데이터 오면 경로 채우고 True로

[방향 참고]
  '사람 정답 비교(evaluate.py)'는 옛 방향이라 제거됨.
  현재는 '위치 추정 일관성(consistency.py)' 중심.
====================================================================
"""

import os
import glob
import numpy as np

from data_loader import (
    load_depth, clean_depth, inspect_depth,
    load_cam_params, load_bupst20_annotation,
)
from priority import select_top1, print_result, WEIGHTS
from consistency import (
    collect_positions_by_id, compute_consistency, summarize,
)


# ====================================================================
# ★ 설정 ★
# ====================================================================
USE_REAL_DATA = False          # 실제 데이터 준비되면 True
MODE = "visibility"            # BUP-ST20: 성숙도 없음 → visibility

DEPTH_DIR = r"C:\path\to\depth"
ANN_DIR   = r"C:\path\to\annotations"
CAM_PATH  = r"C:\path\to\cam_params.yaml"

IMG_SIZE = (720, 1280)


# ====================================================================
# 더미 데이터 (실제 데이터 없이 작동 확인용)
# ====================================================================
def build_dummy_frames(n=3):
    """같은 시퀀스의 연속 프레임 더미 (일관성 분석용)"""
    np.random.seed(1)
    cam = {"fx": 919.46, "fy": 920.65, "cx": 361.72, "cy": 636.79}

    def make_depth(vals):
        d = np.random.randint(500, 2000, (1280, 720)).astype(np.uint16)
        d[0:40, 0:40] = 65535  # invalid 섞기
        for (cx, cy, val) in vals:
            d[cy-10:cy+10, cx-10:cx+10] = val
        return clean_depth(d)

    frames = []
    for i in range(n):
        frames.append({
            "objects": [
                {"id": 5, "bbox": [100+i*10, 200+i*5, 160+i*10, 280+i*5],
                 "confidence": 0.9, "mask_area": 6000},
                {"id": 8, "bbox": [400+i*10, 300+i*5, 460+i*10, 380+i*5],
                 "confidence": 0.8, "mask_area": 3000},
            ],
            # id=8은 마지막 프레임에서 depth가 튐 (불안정 시연)
            "depth_map": make_depth([
                (130+i*10, 240+i*5, 850+i*5),
                (430+i*10, 340+i*5, 1200 if i < n-1 else 1650),
            ]),
            "img_size": (720, 1280),
        })
    return frames, cam


# ====================================================================
# 메인
# ====================================================================
def main():
    weights = WEIGHTS[MODE]
    print("=" * 62)
    print(f" BUP-ST20 파이프라인   (MODE='{MODE}', 실제데이터={USE_REAL_DATA})")
    print("=" * 62)

    if USE_REAL_DATA:
        print("\n[1] 카메라 파라미터 로드")
        cam = load_cam_params(CAM_PATH)
        print(f"    intrinsic: {cam}")

        print("\n[2] depth 샘플 진단")
        samples = sorted(glob.glob(os.path.join(DEPTH_DIR, "**", "*.tiff"),
                                   recursive=True))
        if samples:
            inspect_depth(load_depth(samples[0]))

        # 실제 데이터 연결은 민혁 CSV/폴더구조 확정 후 build 함수 작성
        print("\n[3] (실제 데이터 dataset 구성 — 민혁 데이터 확정 후 연결)")

    else:
        print("\n[더미 모드] 파이프라인 동작 확인\n")

        # --- (A) Top-1 선정 시연 ---
        frames, cam = build_dummy_frames(n=3)
        f0 = frames[0]
        top1, scored = select_top1(f0["objects"], f0["depth_map"],
                                   weights, f0["img_size"], MODE)
        print("[A] Top-1 선정 (첫 프레임)")
        print_result(top1, scored, MODE)

        # --- (B) 위치 추정 일관성 분석 (현재 연구 방향) ---
        print("[B] 위치 추정 일관성 분석 (여러 프레임)")
        from priority import extract_depth
        positions = collect_positions_by_id(frames, cam, extract_depth)
        results = compute_consistency(positions)
        for r in results:
            stable = "안정적 ✅" if r["std_3D"] < 100 else "불안정 ⚠️"
            print(f"  파프리카 id={r['id']}: 3D 흔들림 {r['std_3D']}mm → {stable}")
        s = summarize(results)
        if s:
            print(f"\n  평균 3D 흔들림: {s['평균 3D 흔들림(mm)']}mm")
            print(f"  가장 불안정: id={s['가장 불안정한 파프리카']['id']}, "
                  f"가장 안정: id={s['가장 안정적인 파프리카']['id']}")


if __name__ == "__main__":
    main()