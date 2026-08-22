# -*- coding: utf-8 -*-
"""
====================================================================
 pick_demo_frames.py - 데모 시각화용 프레임 후보 선정
====================================================================

[목적]
  거리 단일 기준 baseline과 우선순위 점수 기반 선정이 서로 다른
  대상을 고른 프레임 중, 시각적으로 차이가 잘 드러나는 것을 뽑는다.

  선정된 프레임의 시퀀스·타임스탬프를 출력하므로,
  해당 RGB 이미지만 별도로 확보하면 시각화가 가능하다.

[정렬 기준]
  가시성 차이(Full 선택 − baseline 선택)가 큰 순서.
  가시성 차이가 크면 "가려진 것 vs 잘 보이는 것"의 대비가 뚜렷하다.

[출력]
  1) 가시성 차이 상위 프레임 목록 (시퀀스, 타임스탬프, 좌표, 속성)
  2) RGB 이미지 요청용 파일 경로 목록  → demo_frame_list.txt
  3) 후보 프레임의 상세 정보          → demo_frame_detail.csv

실행: python pick_demo_frames.py
====================================================================
"""

import os
import csv
import glob
import numpy as np

from data_loader import load_depth, clean_depth, load_bupst20_annotation
from priority import select_top1, WEIGHTS
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

TOP_N          = 12       # 출력할 후보 수
MIN_CANDIDATES = 3        # 후보가 이 개수 이상인 프레임만 (장면이 너무 단순하면 제외)

OUT_LIST   = "demo_frame_list.txt"
OUT_DETAIL = "demo_frame_detail.csv"


def main():
    det = load_detections_csv(DETECTION_CSV)
    depth_root = os.path.join(DATA_DIR, "depth")
    ann_root   = os.path.join(DATA_DIR, "annotations")
    sequences  = sorted(os.listdir(depth_root))
    weights    = WEIGHTS[MODE]

    rows = []
    print(f"프레임 후보 탐색 중 (confidence {CONF_THRESHOLD} 고정)...")

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
            if len(preds) < MIN_CANDIDATES:
                continue

            depth = clean_depth(load_depth(depth_files[ts]))
            objs = [dict(o) for o in preds]
            top1, scored = select_top1(objs, depth, weights,
                                       info["img_size"] or IMG_SIZE, MODE)
            if top1 is None or len(scored) < MIN_CANDIDATES:
                continue

            nearest = min(scored, key=lambda o: o["depth_value"])
            if tuple(nearest["bbox"]) == tuple(top1["bbox"]):
                continue                    # 같은 대상 → 데모 부적합

            dn, dt = nearest["_detail"], top1["_detail"]
            rows.append({
                "seq": seq,
                "timestamp": ts,
                "n_candidates": len(scored),
                "vis_diff": round(dt["visibility"] - dn["visibility"], 3),
                "dist_diff": round(top1["depth_value"] - nearest["depth_value"], 1),
                "near_bbox": [round(v, 1) for v in nearest["bbox"]],
                "near_dist": round(nearest["depth_value"], 1),
                "near_vis": dn["visibility"],
                "near_conf": dn["confidence"],
                "near_score": round(nearest["score"], 3),
                "full_bbox": [round(v, 1) for v in top1["bbox"]],
                "full_dist": round(top1["depth_value"], 1),
                "full_vis": dt["visibility"],
                "full_conf": dt["confidence"],
                "full_score": round(top1["score"], 3),
            })

    if not rows:
        print("조건을 만족하는 프레임이 없습니다.")
        return

    rows.sort(key=lambda r: -r["vis_diff"])
    top = rows[:TOP_N]

    # ---------------- 화면 출력 ----------------
    print(f"\n선택이 갈린 프레임 {len(rows)}개 중 가시성 차이 상위 {len(top)}개\n")
    print("=" * 96)
    print(f" {'#':>2} {'시퀀스':>6} {'타임스탬프':>18} {'후보':>4}"
          f" {'가시성차':>8} {'거리차':>8}"
          f" {'baseline(거리/가시성)':>22} {'제안(거리/가시성)':>20}")
    print("=" * 96)
    for i, r in enumerate(top, 1):
        print(f" {i:>2} {r['seq']:>6} {r['timestamp']:>18} {r['n_candidates']:>4}"
              f" {r['vis_diff']:>+8.3f} {r['dist_diff']:>+7.0f}mm"
              f"   {r['near_dist']:>6.0f}mm / {r['near_vis']:<6.3f}"
              f"   {r['full_dist']:>6.0f}mm / {r['full_vis']:<6.3f}")

    # ---------------- 추천 ----------------
    print("\n" + "=" * 96)
    print(" 추천")
    print("=" * 96)
    best = top[0]
    print(f"  1순위: 시퀀스 {best['seq']} / {best['timestamp']}")
    print(f"     후보 {best['n_candidates']}개, 가시성 차이 {best['vis_diff']:+.3f}, "
          f"거리 차이 {best['dist_diff']:+.0f}mm")
    print(f"     baseline 선택: 거리 {best['near_dist']:.0f}mm, "
          f"가시성 {best['near_vis']:.3f}, 점수 {best['near_score']:.3f}")
    print(f"     제안 선택    : 거리 {best['full_dist']:.0f}mm, "
          f"가시성 {best['full_vis']:.3f}, 점수 {best['full_score']:.3f}")
    print()
    print("  ※ 후보 수가 너무 많으면 그림이 복잡해집니다.")
    print("     4~7개 정도인 프레임이 시각화에 적합합니다.")
    mid = [r for r in top if 4 <= r["n_candidates"] <= 7]
    if mid:
        m = mid[0]
        print(f"  대안(후보 4~7개): 시퀀스 {m['seq']} / {m['timestamp']} "
              f"(후보 {m['n_candidates']}개, 가시성 차이 {m['vis_diff']:+.3f})")

    # ---------------- RGB 요청용 목록 ----------------
    with open(OUT_LIST, "w", encoding="utf-8") as f:
        f.write("# 데모 시각화용 RGB 이미지 요청 목록\n")
        f.write("# 형식: images/<시퀀스>/<타임스탬프>.<확장자>\n")
        f.write(f"# 가시성 차이 상위 {len(top)}개 프레임\n\n")
        for r in top:
            f.write(f"images/{r['seq']}/{r['timestamp']}\n")
    print(f"\n RGB 요청용 목록 저장: {OUT_LIST}")

    # ---------------- 상세 CSV ----------------
    with open(OUT_DETAIL, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(top[0].keys()))
        w.writeheader()
        for r in top:
            row = dict(r)
            row["near_bbox"] = " ".join(str(v) for v in r["near_bbox"])
            row["full_bbox"] = " ".join(str(v) for v in r["full_bbox"])
            w.writerow(row)
    print(f" 상세 정보 저장    : {OUT_DETAIL}")
    print("\n 이 목록의 타임스탬프를 민혁님께 전달하여 해당 RGB 이미지를 요청하세요.")


if __name__ == "__main__":
    main()
