# -*- coding: utf-8 -*-
"""
====================================================================
 diag_seq.py - ablation.py에서 특정 시퀀스가 왜 빠지는지 추적
====================================================================

ablation.py는 프레임을 여러 관문(gate)에서 걸러낸다.
어느 관문에서 몇 개가 떨어지는지 시퀀스별로 세어 출력한다.

실행: python diag_seq.py
====================================================================
"""

import os
import glob

from data_loader import load_depth, clean_depth, load_bupst20_annotation
from priority import select_top1
from load_detections import load_detections_csv, attach_mask_from_pkl

DATA_DIR       = "dataset_bulk"
DETECTION_CSV  = "eval_detections_fixed.csv"
MODE           = "visibility"
IMG_SIZE       = (720, 1280)
IOU_THRESHOLD  = 0.5
CONF_THRESHOLD = 0.3

WEIGHTS_FULL = {"proximity": 0.35, "visibility": 0.30,
                "confidence": 0.20, "center": 0.15}


def main():
    det = load_detections_csv(DETECTION_CSV)
    depth_root = os.path.join(DATA_DIR, "depth")
    ann_root   = os.path.join(DATA_DIR, "annotations")
    sequences  = sorted(os.listdir(depth_root))

    print(f"CSV에 있는 시퀀스: {sorted({s for s, _ in det.keys()})}")
    print(f"depth 폴더 시퀀스: {sequences}")
    print(f"annotations 폴더 : {sorted(os.listdir(ann_root))}")
    print()

    print("=" * 92)
    print(" 관문별 잔존 프레임 수")
    print("=" * 92)
    print(f" {'시퀀스':<8}{'depth∩pkl':>11}{'CSV있음':>9}"
          f"{'conf>=임계':>11}{'GT매칭>=2':>11}{'depth유효>=2':>13}{'최종':>7}")

    for seq in sequences:
        d_dir = os.path.join(depth_root, seq)
        a_dir = os.path.join(ann_root, seq)
        depth_files = {os.path.splitext(os.path.basename(p))[0]: p
                       for p in glob.glob(os.path.join(d_dir, "*.tif*"))}
        ann_files = {os.path.splitext(os.path.basename(p))[0]: p
                     for p in glob.glob(os.path.join(a_dir, "*.pkl"))}

        g0 = g1 = g2 = g3 = g4 = g5 = 0

        for ts in sorted(set(depth_files) & set(ann_files)):
            g0 += 1
            info = det.get((seq, ts))
            if info is None:
                continue
            g1 += 1

            preds = [dict(o) for o in info["objects"]
                     if o["confidence"] >= CONF_THRESHOLD]
            if len(preds) < 2:
                continue
            g2 += 1

            gts = load_bupst20_annotation(ann_files[ts])
            attach_mask_from_pkl(preds, gts, IOU_THRESHOLD)
            preds = [p for p in preds if p.get("gt_matched", False)]
            if len(preds) < 2:
                continue
            g3 += 1

            depth = clean_depth(load_depth(depth_files[ts]))
            objs = [dict(o) for o in preds]
            top1, scored = select_top1(objs, depth, WEIGHTS_FULL,
                                       info["img_size"] or IMG_SIZE, MODE)
            if top1 is None or len(scored) < 2:
                continue
            g4 += 1
            g5 += 1

        print(f" {seq:<8}{g0:>11}{g1:>9}{g2:>11}{g3:>11}{g4:>13}{g5:>7}")

    print()
    print(" 열 설명")
    print("  depth∩pkl   : depth와 pkl이 짝을 이루는 프레임")
    print("  CSV있음     : 민혁 CSV에 해당 프레임 검출이 존재")
    print("  conf>=임계  : 임계값 통과 검출이 2개 이상")
    print("  GT매칭>=2   : 그중 GT와 IoU 매칭된 것이 2개 이상")
    print("  depth유효>=2: 박스 중심 depth가 유효한 후보가 2개 이상")
    print()
    print(" 숫자가 급감하는 열이 원인 지점이다.")


if __name__ == "__main__":
    main()
