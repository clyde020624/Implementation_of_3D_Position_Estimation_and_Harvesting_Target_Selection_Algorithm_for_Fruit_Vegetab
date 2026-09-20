# -*- coding: utf-8 -*-
"""
====================================================================
 tradeoff.py - Full vs nearest 선택 대상의 trade-off 정량화
====================================================================

[목적]
  두 방식이 서로 다른 대상을 선택했을 때,
  '무엇을 얻고 무엇을 잃는지'를 수치로 제시한다.

  우열은 판단할 수 없으나(파지 성공률·정답 우선순위 부재),
  선택된 대상의 속성 차이는 정량화할 수 있다.

[통계]
  outlier의 영향을 받지 않도록 평균이 아닌
  중앙값(median)과 IQR(사분위 범위)을 사용한다.

[출력]
  표 T1  선택이 갈린 프레임에서 각 방식이 고른 대상의 속성
  표 T2  프레임별 차이(Full - nearest)의 분포
  표 T3  시퀀스별 거리/가시성 차이 중앙값
  그림   tradeoff_scatter.png (거리-가시성 평면 산점도)

실행: python tradeoff.py
====================================================================
"""

import os
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

PLOT_PATH = "v1_depth_visibility_scatter.png"
MAKE_PLOT = True          # matplotlib 없으면 False


def iqr_stats(values):
    """중앙값과 사분위수 반환"""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return None
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "median": med, "q1": q1, "q3": q3,
            "iqr": q3 - q1, "min": v.min(), "max": v.max()}


def fmt(st, unit="", digits=1):
    if st is None:
        return "-"
    return (f"{st['median']:.{digits}f}{unit} "
            f"[{st['q1']:.{digits}f}, {st['q3']:.{digits}f}]")


def main():
    det = load_detections_csv(DETECTION_CSV)
    depth_root = os.path.join(DATA_DIR, "depth")
    ann_root   = os.path.join(DATA_DIR, "annotations")
    sequences  = sorted(os.listdir(depth_root))
    weights    = WEIGHTS[MODE]

    # 선택이 갈린 프레임의 기록
    rec = {
        "near_dist": [], "full_dist": [],
        "near_vis":  [], "full_vis":  [],
        "near_conf": [], "full_conf": [],
        "near_cent": [], "full_cent": [],
        "d_dist": [], "d_vis": [],
        "seq": [],
    }
    n_usable = 0
    n_diff = 0

    print(f"trade-off 분석 중 (confidence {CONF_THRESHOLD} 고정)...")

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
            objs = [dict(o) for o in preds]
            top1, scored = select_top1(objs, depth, weights,
                                       info["img_size"] or IMG_SIZE, MODE)
            if top1 is None or len(scored) < 2:
                continue

            n_usable += 1
            nearest = min(scored, key=lambda o: o["depth_value"])
            if tuple(nearest["bbox"]) == tuple(top1["bbox"]):
                continue                       # 같은 대상 선택 → 제외

            n_diff += 1
            dn, dt = nearest["_detail"], top1["_detail"]
            rec["near_dist"].append(nearest["depth_value"])
            rec["full_dist"].append(top1["depth_value"])
            rec["near_vis"].append(dn["visibility"])
            rec["full_vis"].append(dt["visibility"])
            rec["near_conf"].append(dn["confidence"])
            rec["full_conf"].append(dt["confidence"])
            rec["near_cent"].append(dn["center"])
            rec["full_cent"].append(dt["center"])
            rec["d_dist"].append(top1["depth_value"] - nearest["depth_value"])
            rec["d_vis"].append(dt["visibility"] - dn["visibility"])
            rec["seq"].append(seq)

    if n_diff == 0:
        print("선택이 갈린 프레임이 없습니다.")
        return

    print(f"\n비교 가능 {n_usable}프레임 | 선택이 갈린 프레임 {n_diff}개 "
          f"({n_diff/n_usable*100:.1f}%)\n")

    # ---------------- 표 T1 ----------------
    print("=" * 78)
    print(" [표 T1] 선택이 갈린 프레임에서 각 방식이 선택한 대상의 속성")
    print("         값은 중앙값 [Q1, Q3]")
    print("=" * 78)
    print(f" {'속성':<12}{'nearest':>26}{'Full':>26}")
    rows = [
        ("거리(mm)",  "near_dist", "full_dist", "", 0),
        ("가시성",    "near_vis",  "full_vis",  "", 3),
        ("신뢰도",    "near_conf", "full_conf", "", 3),
        ("중심성",    "near_cent", "full_cent", "", 3),
    ]
    for label, kn, kf, unit, dg in rows:
        sn = iqr_stats(rec[kn])
        sf = iqr_stats(rec[kf])
        print(f" {label:<12}{fmt(sn, unit, dg):>26}{fmt(sf, unit, dg):>26}")

    # ---------------- 표 T2 ----------------
    print("\n" + "=" * 78)
    print(" [표 T2] 프레임별 차이 (Full 선택 − nearest 선택)")
    print("=" * 78)
    sd = iqr_stats(rec["d_dist"])
    sv = iqr_stats(rec["d_vis"])
    print(f" 거리 차이  : 중앙값 {sd['median']:+.0f}mm  "
          f"IQR [{sd['q1']:+.0f}, {sd['q3']:+.0f}]  "
          f"범위 [{sd['min']:+.0f}, {sd['max']:+.0f}]")
    print(f" 가시성 차이: 중앙값 {sv['median']:+.3f}  "
          f"IQR [{sv['q1']:+.3f}, {sv['q3']:+.3f}]  "
          f"범위 [{sv['min']:+.3f}, {sv['max']:+.3f}]")
    print()
    gain = sum(1 for x in rec["d_vis"] if x > 0)
    print(f" 가시성이 더 높은 대상을 선택한 경우: {gain}/{n_diff} "
          f"({gain/n_diff*100:.1f}%)")
    farther = sum(1 for x in rec["d_dist"] if x > 0)
    print(f" 더 먼 대상을 선택한 경우           : {farther}/{n_diff} "
          f"({farther/n_diff*100:.1f}%)")

    # ---------------- 표 T3 ----------------
    print("\n" + "=" * 78)
    print(" [표 T3] 시퀀스별 차이 중앙값")
    print("=" * 78)
    print(f" {'시퀀스':<8}{'갈린 프레임':>12}{'거리 차이':>14}{'가시성 차이':>14}")
    for seq in sequences:
        idx = [i for i, s in enumerate(rec["seq"]) if s == seq]
        if not idx:
            continue
        dd = np.median([rec["d_dist"][i] for i in idx])
        dv = np.median([rec["d_vis"][i] for i in idx])
        print(f" {seq:<8}{len(idx):>12}{dd:>+13.0f}mm{dv:>+14.3f}")

    # ---------------- 산점도 ----------------
    if MAKE_PLOT:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 5.5))
            ax.scatter(rec["near_dist"], rec["near_vis"],
                       s=34, alpha=0.55, marker="o",
                       facecolors="none", edgecolors="#c0392b",
                       linewidths=1.2, label="nearest baseline")
            ax.scatter(rec["full_dist"], rec["full_vis"],
                       s=34, alpha=0.55, marker="^",
                       facecolors="none", edgecolors="#1f6f8b",
                       linewidths=1.2, label="Proposed (Full)")

            ax.set_xlabel("Representative depth (mm)")
            ax.set_ylabel("Visibility (mask area / bbox area)")
            ax.set_title(f"Selected target: nearest vs proposed "
                         f"(n={n_diff} frames)")
            ax.legend(frameon=False, loc="lower right")
            ax.grid(alpha=0.25, linewidth=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            fig.tight_layout()
            fig.savefig(PLOT_PATH, dpi=200)
            print(f"\n 산점도 저장: {PLOT_PATH}")
        except ImportError:
            print("\n matplotlib이 없어 산점도를 건너뜁니다.")
            print(" pip install matplotlib 후 다시 실행하세요.")

    print("\n" + "=" * 78)
    print(" 해석 가이드")
    print("=" * 78)
    print(" - 거리 차이가 +면 제안 방법이 더 먼 대상을 선택했다는 뜻")
    print(" - 가시성 차이가 +면 덜 가려진 대상을 선택했다는 뜻")
    print(" - 두 값이 함께 +이면 '거리를 내주고 가시성을 얻는' trade-off")
    print(" - 우열 판단이 아니라 선택 특성의 차이를 기술하는 것임에 유의")


if __name__ == "__main__":
    main()
