# -*- coding: utf-8 -*-
"""
====================================================================
 ablation.py - Priority Score 구성요소 ablation (연구 5단계)
====================================================================

[목적]
  Priority Score를 이루는 4개 항목 중 무엇이 실제로 선택을 바꾸는지
  하나씩 빼가며 확인한다.

[구성]
  D          거리만            (= nearest baseline과 동일해야 함. 검산용)
  D+V        거리 + 가시성
  D+V+C      거리 + 가시성 + 신뢰도
  Full       전체 (거리+가시성+신뢰도+중심)
  가시성제거  거리 + 신뢰도 + 중심   ★ 핵심 비교

  각 구성의 가중치는 원본 비율을 유지한 채 합이 1이 되도록 재정규화.

[메모리]
  instance_mask는 프레임 하나씩 읽고 바로 버린다.
  (threshold_experiment.py처럼 전체를 들고 있지 않음)

실행: python ablation.py
====================================================================
"""

import os
import glob
import numpy as np

from data_loader import (
    load_depth, clean_depth, load_cam_params, load_bupst20_annotation,
)
from priority import select_top1
from load_detections import load_detections_csv, attach_mask_from_pkl


# ====================================================================
# ★ 설정 ★
# ====================================================================
DATA_DIR      = "dataset_bulk"
DETECTION_CSV = "eval_detections_fixed.csv"
MODE          = "visibility"
IMG_SIZE      = (720, 1280)
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.3          # 안정 구간에서 고정


def make_weights(use_proximity=True, use_visibility=True,
                 use_confidence=True, use_center=True):
    """원본 비율을 유지한 채 사용할 항목만 남기고 합=1로 재정규화"""
    base = {"proximity": 0.35, "visibility": 0.30,
            "confidence": 0.20, "center": 0.15}
    keep = {"proximity": use_proximity, "visibility": use_visibility,
            "confidence": use_confidence, "center": use_center}
    total = sum(v for k, v in base.items() if keep[k])
    return {k: (base[k] / total if keep[k] else 0.0) for k in base}


CONFIGS = [
    ("D",         make_weights(True, False, False, False)),
    ("D+V",       make_weights(True, True,  False, False)),
    ("D+V+C",     make_weights(True, True,  True,  False)),
    ("Full",      make_weights(True, True,  True,  True)),
    ("가시성제거", make_weights(True, False, True,  True)),
]


# ====================================================================
# 한 프레임에서 모든 구성의 Top-1을 구함
# ====================================================================
def top1_per_config(preds, depth_map, img_size):
    """반환: {구성이름: 선택된 객체의 식별키}, nearest 식별키, 후보 수"""
    picks = {}
    nearest_key = None
    n_cand = 0

    for name, w in CONFIGS:
        objs = [dict(o) for o in preds]
        top1, scored = select_top1(objs, depth_map, w, img_size, MODE)
        if top1 is None or len(scored) < 2:
            return None, None, 0
        n_cand = len(scored)
        # 객체를 bbox로 식별 (dict 복사본이라 is 비교 불가)
        picks[name] = tuple(top1["bbox"])
        if nearest_key is None:
            nearest = min(scored, key=lambda o: o["depth_value"])
            nearest_key = tuple(nearest["bbox"])

    return picks, nearest_key, n_cand


def main():
    det = load_detections_csv(DETECTION_CSV)
    depth_root = os.path.join(DATA_DIR, "depth")
    ann_root   = os.path.join(DATA_DIR, "annotations")
    sequences = sorted(os.listdir(depth_root))

    names = [n for n, _ in CONFIGS]
    diff_vs_nearest = {n: 0 for n in names}
    agree_with_full = {n: 0 for n in names}
    per_seq = {}
    usable = 0
    cand_sum = 0

    print(f"confidence 임계값 {CONF_THRESHOLD} 고정, 구성 {len(CONFIGS)}개 평가 중...")

    for seq in sequences:
        d_dir = os.path.join(depth_root, seq)
        a_dir = os.path.join(ann_root, seq)
        depth_files = {os.path.splitext(os.path.basename(p))[0]: p
                       for p in glob.glob(os.path.join(d_dir, "*.tif*"))}
        ann_files = {os.path.splitext(os.path.basename(p))[0]: p
                     for p in glob.glob(os.path.join(a_dir, "*.pkl"))}
        per_seq[seq] = {n: 0 for n in names}
        per_seq[seq]["usable"] = 0

        for ts in sorted(set(depth_files) & set(ann_files)):
            info = det.get((seq, ts))
            if info is None:
                continue
            preds = [dict(o) for o in info["objects"]
                     if o["confidence"] >= CONF_THRESHOLD]
            if len(preds) < 2:
                continue

            depth = clean_depth(load_depth(depth_files[ts]))
            gts = load_bupst20_annotation(ann_files[ts])
            attach_mask_from_pkl(preds, gts, IOU_THRESHOLD)
            preds = [p for p in preds if p.get("gt_matched", False)]
            del gts                      # 마스크 즉시 해제
            if len(preds) < 2:
                continue

            picks, nearest_key, n_cand = top1_per_config(
                preds, depth, info["img_size"] or IMG_SIZE)
            if picks is None:
                continue

            usable += 1
            cand_sum += n_cand
            per_seq[seq]["usable"] += 1
            for n in names:
                if picks[n] != nearest_key:
                    diff_vs_nearest[n] += 1
                    per_seq[seq][n] += 1
                if picks[n] == picks["Full"]:
                    agree_with_full[n] += 1

    if usable == 0:
        print("비교 가능한 프레임이 없습니다.")
        return

    print(f"\n비교 가능 프레임 {usable}개 | 평균 후보 {cand_sum/usable:.1f}개\n")

    print("=" * 74)
    print(" [표 A] 구성별 가중치")
    print("=" * 74)
    print(f" {'구성':<10}{'거리':>10}{'가시성':>10}{'신뢰도':>10}{'중심':>10}")
    for name, w in CONFIGS:
        print(f" {name:<10}{w['proximity']:>10.3f}{w['visibility']:>10.3f}"
              f"{w['confidence']:>10.3f}{w['center']:>10.3f}")

    print("\n" + "=" * 74)
    print(" [표 B] 구성별 nearest baseline 대비 선택 차이 / Full과의 일치율")
    print("=" * 74)
    print(f" {'구성':<10}{'차이':>8}{'차이비율':>12}{'Full일치':>10}{'일치율':>10}")
    for n in names:
        d = diff_vs_nearest[n]
        a = agree_with_full[n]
        print(f" {n:<10}{d:>8}{d/usable*100:>11.1f}%"
              f"{a:>10}{a/usable*100:>9.1f}%")

    print("\n" + "=" * 74)
    print(" [표 C] 시퀀스별 nearest 대비 차이비율 (구성별)")
    print("=" * 74)
    header = f" {'시퀀스':<8}{'프레임':>7}"
    for n in names:
        header += f"{n:>12}"
    print(header)
    for seq in sequences:
        d = per_seq.get(seq)
        if not d or d["usable"] == 0:
            continue
        line = f" {seq:<8}{d['usable']:>7}"
        for n in names:
            line += f"{d[n]/d['usable']*100:>11.1f}%"
        print(line)

    print("\n" + "=" * 74)
    print(" 해석 가이드")
    print("=" * 74)
    print(" - D의 차이비율은 0.0%여야 정상 (nearest와 동일한 정의)")
    print(" - D -> D+V 에서 차이비율이 크게 뛰면 가시성이 선택을 바꾸는 주역")
    print(" - '가시성제거'의 Full 일치율이 낮을수록 가시성 기여가 큼")


if __name__ == "__main__":
    main()
