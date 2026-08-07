# -*- coding: utf-8 -*-
"""
====================================================================
 sensitivity.py - 가시성 가중치 민감도 분석
====================================================================

[목적]
  Priority Score의 가중치(0.35/0.30/0.20/0.15)는 경험적으로 설정된
  값이다. "가시성 가중치가 0.30이 아니었으면 결론이 달라지는가?"
  라는 질문에 답하기 위해 가시성 비중을 바꿔가며 결과를 관찰한다.

[방법]
  가시성 가중치 w_v를 0.00 ~ 0.60으로 변화시키고,
  나머지 세 항목(거리 0.35 / 신뢰도 0.20 / 중심 0.15)은
  원래 비율(0.5 : 0.286 : 0.214)을 유지한 채 (1 - w_v)를 나눠 갖는다.

  예) w_v = 0.30 → 거리 0.350 / 신뢰도 0.200 / 중심 0.150  (원래 설정)
      w_v = 0.50 → 거리 0.250 / 신뢰도 0.143 / 중심 0.107

[읽는 법]
  차이비율이 완만하게 변하면
     → 결론이 특정 가중치 선택에 민감하지 않다 (논문에 유리)
  급격하게 변하면
     → 정직한 한계로 보고하되, 가시성의 지배력 자체가 발견이 됨

실행: python sensitivity.py
====================================================================
"""

import os
import glob
import numpy as np

from data_loader import load_depth, clean_depth, load_bupst20_annotation
from priority import select_top1
from load_detections import load_detections_csv, attach_mask_from_pkl


# ====================================================================
# ★ 설정 ★
# ====================================================================
DATA_DIR       = "dataset_bulk"
DETECTION_CSV  = "eval_detections_fixed.csv"
MODE           = "visibility"
IMG_SIZE       = (720, 1280)
IOU_THRESHOLD  = 0.5
CONF_THRESHOLD = 0.3

# 실험할 가시성 가중치
V_WEIGHTS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

BASELINE_SETTING = 0.30      # 논문에서 사용한 값 (표에 표시용)


def make_weights(w_v):
    """가시성 가중치 w_v를 고정하고 나머지는 원래 비율로 (1-w_v)를 분배"""
    rest = {"proximity": 0.35, "confidence": 0.20, "center": 0.15}
    total_rest = sum(rest.values())          # 0.70
    scale = (1.0 - w_v) / total_rest
    w = {k: v * scale for k, v in rest.items()}
    w["visibility"] = w_v
    return w


def main():
    det = load_detections_csv(DETECTION_CSV)
    depth_root = os.path.join(DATA_DIR, "depth")
    ann_root   = os.path.join(DATA_DIR, "annotations")
    sequences  = sorted(os.listdir(depth_root))

    weight_sets = [(w_v, make_weights(w_v)) for w_v in V_WEIGHTS]
    ref_idx = V_WEIGHTS.index(BASELINE_SETTING)

    diff_count   = {w_v: 0 for w_v in V_WEIGHTS}
    same_as_ref  = {w_v: 0 for w_v in V_WEIGHTS}
    per_seq      = {seq: {w_v: 0 for w_v in V_WEIGHTS} for seq in sequences}
    per_seq_n    = {seq: 0 for seq in sequences}
    usable = 0

    print(f"가시성 가중치 {len(V_WEIGHTS)}종 평가 중 "
          f"(confidence {CONF_THRESHOLD} 고정)...")

    for seq in sequences:
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
            preds = [dict(o) for o in info["objects"]
                     if o["confidence"] >= CONF_THRESHOLD]
            if len(preds) < 2:
                continue

            gts = load_bupst20_annotation(ann_files[ts])
            attach_mask_from_pkl(preds, gts, IOU_THRESHOLD)
            preds = [p for p in preds if p.get("gt_matched", False)]
            del gts
            if len(preds) < 2:
                continue

            depth = clean_depth(load_depth(depth_files[ts]))
            img_size = info["img_size"] or IMG_SIZE

            picks = {}
            nearest_key = None
            ok = True
            for w_v, w in weight_sets:
                objs = [dict(o) for o in preds]
                top1, scored = select_top1(objs, depth, w, img_size, MODE)
                if top1 is None or len(scored) < 2:
                    ok = False
                    break
                picks[w_v] = tuple(top1["bbox"])
                if nearest_key is None:
                    nearest = min(scored, key=lambda o: o["depth_value"])
                    nearest_key = tuple(nearest["bbox"])
            if not ok:
                continue

            usable += 1
            per_seq_n[seq] += 1
            ref_key = picks[V_WEIGHTS[ref_idx]]
            for w_v in V_WEIGHTS:
                if picks[w_v] != nearest_key:
                    diff_count[w_v] += 1
                    per_seq[seq][w_v] += 1
                if picks[w_v] == ref_key:
                    same_as_ref[w_v] += 1

    if usable == 0:
        print("비교 가능한 프레임이 없습니다.")
        return

    print(f"\n비교 가능 프레임 {usable}개\n")

    print("=" * 78)
    print(" [표 S1] 가시성 가중치별 결과")
    print("=" * 78)
    print(f" {'w_v':>6}{'거리':>8}{'신뢰도':>8}{'중심':>8}"
          f"{'차이':>8}{'차이비율':>11}{'w_v=0.30과 일치':>17}")
    for w_v, w in weight_sets:
        mark = " *" if w_v == BASELINE_SETTING else "  "
        d = diff_count[w_v]
        s = same_as_ref[w_v]
        print(f"{mark}{w_v:>4.2f}{w['proximity']:>8.3f}"
              f"{w['confidence']:>8.3f}{w['center']:>8.3f}"
              f"{d:>8}{d/usable*100:>10.1f}%{s/usable*100:>16.1f}%")
    print("\n * = 논문에서 사용한 설정")

    print("\n" + "=" * 78)
    print(" [표 S2] 시퀀스별 차이비율 (가시성 가중치별)")
    print("=" * 78)
    header = f" {'시퀀스':<8}{'프레임':>7}"
    for w_v in V_WEIGHTS:
        header += f"{w_v:>9.2f}"
    print(header)
    for seq in sequences:
        n = per_seq_n[seq]
        if n == 0:
            continue
        line = f" {seq:<8}{n:>7}"
        for w_v in V_WEIGHTS:
            line += f"{per_seq[seq][w_v]/n*100:>8.1f}%"
        print(line)

    # ---- 민감도 요약 ----
    ratios = np.array([diff_count[w] / usable * 100 for w in V_WEIGHTS])
    span = ratios.max() - ratios.min()
    # 0.20~0.40 구간(설정값 주변)에서의 변동폭
    near = [w for w in V_WEIGHTS if 0.20 <= w <= 0.40]
    near_r = np.array([diff_count[w] / usable * 100 for w in near])
    near_span = near_r.max() - near_r.min()

    print("\n" + "=" * 78)
    print(" [요약] 민감도")
    print("=" * 78)
    print(f"  전 구간(0.00~0.60) 차이비율 범위 : "
          f"{ratios.min():.1f}% ~ {ratios.max():.1f}%  (폭 {span:.1f}%p)")
    print(f"  설정값 주변(0.20~0.40) 변동폭     : {near_span:.1f}%p")
    print()
    if near_span < 5.0:
        print("  → 설정값 주변에서 변동이 작다. 결론이 특정 가중치 선택에")
        print("     민감하지 않다고 서술할 수 있다.")
    else:
        print("  → 설정값 주변에서도 변동이 크다. 한계로 명시하고,")
        print("     가시성 비중이 의사결정을 지배한다는 점을 함께 보고할 것.")


if __name__ == "__main__":
    main()
