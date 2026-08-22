# -*- coding: utf-8 -*-
"""
====================================================================
 diag_depth.py - bbox 중심 depth vs mask 기반 depth 비교 진단
====================================================================

[문제]
  데모 그림에서 육안으로 더 가까워 보이는 대상이
  오히려 큰 depth 값을 갖는 현상이 반복 관찰되었다.

[가설]
  priority.py의 extract_depth()는 bbox 중심 5×5 영역만 본다.
  가려진 대상은 bbox 내부 대부분이 잎이므로, 중심 5×5가
  과실이 아닌 앞쪽 잎의 depth를 잡을 수 있다.
  → 가려진 대상의 depth가 실제보다 가깝게 추정된다.

[검증 방법]
  같은 객체에 대해 두 가지 depth를 비교한다.
    (a) bbox 중심 5×5 median   ← 현재 사용 방식
    (b) GT instance mask 내부 median  ← 과실 픽셀만
  (a) < (b) 이면 중심 영역이 앞쪽 물체를 잡았다는 뜻이다.

[출력]
  1) 지정 프레임의 객체별 상세 비교
  2) 전체 프레임 통계 — 가시성 구간별 depth 편차

실행: python diag_depth.py
====================================================================
"""

import os
import glob
import numpy as np

from data_loader import load_depth, clean_depth, load_bupst20_annotation
from priority import select_top1, extract_depth, WEIGHTS
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

# 상세 진단할 프레임 (데모 후보 3개)
TARGET_FRAMES = [
    ("406", "1600938551617024"),
    ("405", "1600938547284129"),
    ("401", "1600938537684754"),
]

RUN_GLOBAL_STATS = True     # 전체 프레임 통계도 낼지


def depth_from_mask_median(mask, depth_map):
    """GT mask 내부 유효 픽셀의 median depth (과실 픽셀만)"""
    if mask is None:
        return None
    m = np.asarray(mask).astype(bool)
    if m.shape != depth_map.shape:
        return None
    vals = depth_map[m & (depth_map > 0)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def load_frame(seq, ts):
    ann_path = os.path.join(DATA_DIR, "annotations", seq, ts + ".pkl")
    dcands = glob.glob(os.path.join(DATA_DIR, "depth", seq, ts + ".tif*"))
    if not os.path.exists(ann_path) or not dcands:
        return None
    depth = clean_depth(load_depth(dcands[0]))
    gts = load_bupst20_annotation(ann_path)
    return depth, gts


def analyze_frame(seq, ts, det):
    info = det.get((seq, ts))
    if info is None:
        print(f"  검출 CSV에 없음: {seq}/{ts}")
        return
    loaded = load_frame(seq, ts)
    if loaded is None:
        print(f"  depth/annotation 없음: {seq}/{ts}")
        return
    depth, gts = loaded

    preds = [dict(o) for o in info["objects"]
             if o["confidence"] >= CONF_THRESHOLD]
    attach_mask_from_pkl(preds, gts, IOU_THRESHOLD)
    preds = [p for p in preds if p.get("gt_matched", False)]
    if len(preds) < 2:
        print(f"  후보 부족: {seq}/{ts}")
        return

    # GT id → mask 매핑
    gt_by_id = {g.get("id"): g for g in gts}

    top1, scored = select_top1(preds, depth, WEIGHTS[MODE],
                              info["img_size"] or IMG_SIZE, MODE)
    if top1 is None:
        print(f"  선정 불가: {seq}/{ts}")
        return
    nearest = min(scored, key=lambda o: o["depth_value"])

    print(f"\n{'='*88}")
    print(f" 시퀀스 {seq} / {ts}   (후보 {len(scored)}개)")
    print(f"{'='*88}")
    print(f" {'':4}{'가시성':>8}{'중심5x5':>10}{'mask median':>13}"
          f"{'차이':>9}{'해석':>18}")

    rows = []
    for o in sorted(scored, key=lambda x: x["depth_value"]):
        gt = gt_by_id.get(o.get("matched_gt_id"))
        z_mask = depth_from_mask_median(
            gt.get("instance_mask") if gt else None, depth)
        z_ctr = o["depth_value"]
        vis = o["_detail"]["visibility"]

        mark = "  "
        if o is nearest and o is top1:
            mark = "AB"
        elif o is nearest:
            mark = "A "
        elif o is top1:
            mark = "B "

        if z_mask is None:
            print(f" {mark:4}{vis:>8.3f}{z_ctr:>9.0f}mm{'-':>13}"
                  f"{'-':>9}{'mask 없음':>18}")
            continue

        diff = z_ctr - z_mask
        if diff < -30:
            interp = "중심이 앞쪽 잡음"
        elif diff > 30:
            interp = "중심이 뒤쪽 잡음"
        else:
            interp = "일치"
        print(f" {mark:4}{vis:>8.3f}{z_ctr:>9.0f}mm{z_mask:>11.0f}mm"
              f"{diff:>+8.0f}mm{interp:>18}")
        rows.append((vis, z_ctr, z_mask, diff, mark.strip()))

    # --- A/B 재비교 ---
    a = [r for r in rows if r[4] == "A"]
    b = [r for r in rows if r[4] == "B"]
    if a and b:
        va, za_c, za_m, _, _ = a[0]
        vb, zb_c, zb_m, _, _ = b[0]
        print()
        print(f"  [A] 가시성 {va:.3f}  중심5x5 {za_c:.0f}mm  "
              f"mask median {za_m:.0f}mm")
        print(f"  [B] 가시성 {vb:.3f}  중심5x5 {zb_c:.0f}mm  "
              f"mask median {zb_m:.0f}mm")
        print()
        print(f"  중심5x5 기준     : A {za_c:.0f}mm vs B {zb_c:.0f}mm  "
              f"→ {'A가 더 가까움' if za_c < zb_c else 'B가 더 가까움'}")
        print(f"  mask median 기준 : A {za_m:.0f}mm vs B {zb_m:.0f}mm  "
              f"→ {'A가 더 가까움' if za_m < zb_m else 'B가 더 가까움'}")
        if (za_c < zb_c) != (za_m < zb_m):
            print()
            print("  ★ 두 기준의 대소 관계가 뒤집힘 →")
            print("     bbox 중심 5×5가 과실이 아닌 다른 물체를 잡은 것으로")
            print("     판단됩니다. 육안 관찰과 depth 값이 어긋난 원인입니다.")


def global_stats(det):
    """가시성 구간별로 중심5x5와 mask median의 편차 통계"""
    depth_root = os.path.join(DATA_DIR, "depth")
    ann_root   = os.path.join(DATA_DIR, "annotations")
    buckets = {"0.0-0.3": [], "0.3-0.5": [], "0.5-0.7": [], "0.7-1.0": []}

    for seq in sorted(os.listdir(depth_root)):
        dfiles = {os.path.splitext(os.path.basename(p))[0]: p
                  for p in glob.glob(os.path.join(depth_root, seq, "*.tif*"))}
        afiles = {os.path.splitext(os.path.basename(p))[0]: p
                  for p in glob.glob(os.path.join(ann_root, seq, "*.pkl"))}
        for ts in sorted(set(dfiles) & set(afiles)):
            info = det.get((seq, ts))
            if info is None:
                continue
            preds = [dict(o) for o in info["objects"]
                     if o["confidence"] >= CONF_THRESHOLD]
            if not preds:
                continue
            gts = load_bupst20_annotation(afiles[ts])
            attach_mask_from_pkl(preds, gts, IOU_THRESHOLD)
            preds = [p for p in preds if p.get("gt_matched", False)]
            if not preds:
                del gts
                continue
            depth = clean_depth(load_depth(dfiles[ts]))
            gt_by_id = {g.get("id"): g for g in gts}

            for p in preds:
                gt = gt_by_id.get(p.get("matched_gt_id"))
                if gt is None:
                    continue
                z_mask = depth_from_mask_median(gt.get("instance_mask"), depth)
                z_ctr = extract_depth(p["bbox"], depth)
                if z_mask is None or z_ctr is None:
                    continue
                x1, y1, x2, y2 = p["bbox"]
                area = max(1, (x2 - x1) * (y2 - y1))
                ma = gt.get("mask_area")
                if ma is None:
                    continue
                vis = min(1.0, ma / area)
                d = z_ctr - z_mask
                if vis < 0.3:   buckets["0.0-0.3"].append(d)
                elif vis < 0.5: buckets["0.3-0.5"].append(d)
                elif vis < 0.7: buckets["0.5-0.7"].append(d)
                else:           buckets["0.7-1.0"].append(d)
            del gts

    print(f"\n{'='*88}")
    print(" [전체 통계] 가시성 구간별 (중심5×5 − mask median) 편차")
    print(f"{'='*88}")
    print(f" {'가시성 구간':>12}{'객체 수':>9}{'중앙값':>10}{'절대 중앙값':>13}"
          f"{'앞쪽 오인 비율':>15}")
    for k in ["0.0-0.3", "0.3-0.5", "0.5-0.7", "0.7-1.0"]:
        v = np.array(buckets[k], dtype=float)
        if v.size == 0:
            print(f" {k:>12}{'0':>9}")
            continue
        ahead = float(np.mean(v < -30) * 100)
        print(f" {k:>12}{v.size:>9}{np.median(v):>+9.0f}mm"
              f"{np.median(np.abs(v)):>12.0f}mm{ahead:>14.1f}%")
    print()
    print(" '앞쪽 오인 비율' = 중심5×5가 mask보다 30mm 이상 가깝게 나온 비율")
    print(" 가시성이 낮은 구간에서 이 비율이 높으면 가설이 확인됩니다.")


def main():
    det = load_detections_csv(DETECTION_CSV)

    print("=" * 88)
    print(" bbox 중심 5×5 depth vs GT mask median depth 비교")
    print("=" * 88)
    print(" A = 거리 기준 baseline 선정, B = 우선순위 점수 선정")

    for seq, ts in TARGET_FRAMES:
        analyze_frame(seq, ts, det)

    if RUN_GLOBAL_STATS:
        print("\n전체 프레임 통계 계산 중... (시간이 걸립니다)")
        global_stats(det)


if __name__ == "__main__":
    main()
