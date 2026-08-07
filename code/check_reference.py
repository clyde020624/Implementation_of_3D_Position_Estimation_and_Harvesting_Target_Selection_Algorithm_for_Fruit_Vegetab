# -*- coding: utf-8 -*-
"""
====================================================================
 check_reference.py - GT reference가 정말 'mask 기반'인지 검증
====================================================================

[왜 필요한가]
  localization.py의 depth_from_mask()는 mask가 없거나 shape가 다르면
  조용히 None을 반환하고, evaluate_localization()은 그때
  GT도 '박스 중심 5x5'로 대체한다.

  이 대체가 많이 일어나면
    예측 = 박스중심 5x5,  reference = 박스중심 5x5
  가 되어 오차가 실제보다 훨씬 작게 나온다.
  (논문의 3D 중앙 2.9mm가 이것 때문인지 반드시 확인해야 함)

실행: python check_reference.py
====================================================================
"""

import os
import glob
import numpy as np

from data_loader import (
    load_depth, clean_depth, load_cam_params, load_bupst20_annotation,
)
from localization import depth_from_bbox_center, depth_from_mask, match_by_iou
from load_detections import load_detections_csv

DATA_DIR      = "dataset_bulk"
CAM_PATH      = "cam_params.yaml"
DETECTION_CSV = "eval_detections_fixed.csv"
IOU_THRESHOLD = 0.5


def main():
    det = load_detections_csv(DETECTION_CSV)
    depth_root = os.path.join(DATA_DIR, "depth")
    ann_root   = os.path.join(DATA_DIR, "annotations")

    n_obj = 0
    dtype_counter = {}
    shape_ok = 0
    shape_bad = 0
    mask_none = 0

    fallback = 0          # reference가 박스중심으로 대체된 횟수
    mask_used = 0         # reference가 정상적으로 mask로 계산된 횟수
    pix_offsets = []      # 예측 중심 <-> mask 무게중심 픽셀 거리
    depth_diffs = []      # 박스중심 median <-> mask median 차이(mm)
    depth_none = 0        # 예측 depth 자체가 None이라 통째로 버려진 수

    for seq in sorted(os.listdir(depth_root)):
        d_dir = os.path.join(depth_root, seq)
        a_dir = os.path.join(ann_root, seq)
        depth_files = {os.path.splitext(os.path.basename(p))[0]: p
                       for p in glob.glob(os.path.join(d_dir, "*.tif*"))}
        ann_files = {os.path.splitext(os.path.basename(p))[0]: p
                     for p in glob.glob(os.path.join(a_dir, "*.pkl"))}

        for ts in sorted(set(depth_files) & set(ann_files)):
            info = det.get((seq, ts))
            if info is None:
                continue
            depth = clean_depth(load_depth(depth_files[ts]))
            gts = load_bupst20_annotation(ann_files[ts])

            # --- GT mask 자체 점검 ---
            for g in gts:
                n_obj += 1
                m = g.get("instance_mask")
                if m is None:
                    mask_none += 1
                    continue
                dt = str(getattr(m, "dtype", type(m).__name__))
                dtype_counter[dt] = dtype_counter.get(dt, 0) + 1
                if getattr(m, "shape", None) == depth.shape:
                    shape_ok += 1
                else:
                    shape_bad += 1

            # --- 매칭된 쌍에서 reference 계산 경로 추적 ---
            matched, _, _ = match_by_iou(info["objects"], gts, IOU_THRESHOLD)
            for mm in matched:
                zp, cp = depth_from_bbox_center(mm["pred"]["bbox"], depth)
                if zp is None:
                    depth_none += 1
                    continue
                zg, cg = depth_from_mask(mm["gt"].get("instance_mask"), depth)
                if zg is None:
                    fallback += 1
                    continue
                mask_used += 1
                pix_offsets.append(float(np.hypot(cp[0] - cg[0], cp[1] - cg[1])))
                depth_diffs.append(float(zp - zg))

    print("=" * 66)
    print(" [1] GT instance_mask 자체 점검")
    print("=" * 66)
    print(f"  GT 객체 총 {n_obj}개")
    print(f"  mask 없음        : {mask_none}개")
    print(f"  mask dtype 분포  : {dtype_counter}   <- bool 이어야 정상")
    print(f"  depth와 shape 일치: {shape_ok}개 / 불일치: {shape_bad}개")
    if shape_bad:
        print("  ※ 불일치가 있으면 그 객체는 전부 박스중심으로 대체됩니다!")

    print()
    print("=" * 66)
    print(" [2] reference 계산 경로 (매칭된 쌍 기준)")
    print("=" * 66)
    total = mask_used + fallback
    print(f"  mask로 계산   : {mask_used}개"
          f" ({mask_used/total*100:.1f}%)" if total else "  (매칭 없음)")
    print(f"  박스중심 대체 : {fallback}개"
          f" ({fallback/total*100:.1f}%)  <- 0에 가까워야 정상" if total else "")
    print(f"  예측 depth 없어 제외: {depth_none}개")

    if pix_offsets:
        po = np.array(pix_offsets)
        dd = np.array(depth_diffs)
        print()
        print("=" * 66)
        print(" [3] 예측 기준점 vs reference 기준점 차이 (오차의 실제 성분)")
        print("=" * 66)
        print(f"  픽셀 거리(박스중심 <-> mask무게중심)")
        print(f"     평균 {po.mean():.2f}px | 중앙값 {np.median(po):.2f}px "
              f"| 최대 {po.max():.2f}px")
        print(f"  depth 차이(박스중심 median <-> mask median)")
        print(f"     절대평균 {np.abs(dd).mean():.1f}mm "
              f"| 중앙값 {np.median(np.abs(dd)):.1f}mm "
              f"| 최대 {np.abs(dd).max():.1f}mm")
        print()
        print("  해석: 픽셀 거리 중앙값이 1px 안팎이면 두 기준점이 사실상 같은 곳이라")
        print("        오차가 작게 나오는 것이 당연하다는 뜻입니다.")
        print("        (논문에서 '기준점 정의 차이' 한계를 수치로 쓸 수 있는 자리)")


if __name__ == "__main__":
    main()
