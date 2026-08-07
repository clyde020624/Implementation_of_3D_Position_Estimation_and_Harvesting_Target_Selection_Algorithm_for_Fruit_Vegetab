# -*- coding: utf-8 -*-
"""
====================================================================
 threshold_experiment.py - confidence 임계값 실험
====================================================================

[목적]
  검출 CSV가 conf 0.05로 넓게 저장되어 저품질 후보가 섞여 있다.
  임계값을 바꿔가며 아래가 어떻게 변하는지 분석한다.

    1) 매칭/미매칭 수  (오검출이 얼마나 걸러지는지)
    2) 3D localization error
    3) nearest baseline 대비 Top-1 선택 차이
    4) 선택이 갈린 원인 (어떤 항목 때문인지)

[중요]
  이 실험은 '경향 파악'이 목적이다.
  최종 임계값은 validation split에서 결정해 eval에 고정 적용해야 한다.
  (eval 결과를 보고 고르면 test-set tuning이 된다)

실행: python threshold_experiment.py
====================================================================
"""

import os
import glob
import numpy as np

from data_loader import (
    load_depth, clean_depth, load_cam_params, load_bupst20_annotation,
)
from priority import select_top1, WEIGHTS
from localization import evaluate_localization, summarize_errors
from load_detections import load_detections_csv, attach_mask_from_pkl


# ====================================================================
# ★ 설정 ★
# ====================================================================
DATA_DIR      = "dataset_bulk"
CAM_PATH      = "cam_params.yaml"
DETECTION_CSV = "eval_detections_fixed.csv"
MODE          = "visibility"
IMG_SIZE      = (720, 1280)
IOU_THRESHOLD = 0.5

# 실험할 임계값들
THRESHOLDS = [0.05, 0.10, 0.25, 0.30, 0.374, 0.50, 0.70]


# ====================================================================
# 데이터 로드 (한 번만 읽어서 재사용)
# ====================================================================
def load_all_frames():
    cam = load_cam_params(CAM_PATH)
    det = load_detections_csv(DETECTION_CSV)

    depth_root = os.path.join(DATA_DIR, "depth")
    ann_root = os.path.join(DATA_DIR, "annotations")
    sequences = sorted(os.listdir(depth_root))

    all_frames = []
    for seq in sequences:
        d_dir = os.path.join(depth_root, seq)
        a_dir = os.path.join(ann_root, seq)
        depth_files = {os.path.splitext(os.path.basename(p))[0]: p
                       for p in glob.glob(os.path.join(d_dir, "*.tif*"))}
        ann_files = {os.path.splitext(os.path.basename(p))[0]: p
                     for p in glob.glob(os.path.join(a_dir, "*.pkl"))}

        for (s, ts), info in det.items():
            if s != seq or ts not in depth_files or ts not in ann_files:
                continue
            gts = load_bupst20_annotation(ann_files[ts])
            preds_all = info["objects"]
            attach_mask_from_pkl(preds_all, gts)     # 가시성 부여
            all_frames.append({
                "seq": seq,
                "image_id": ts,
                "preds_all": preds_all,               # 필터 전 전체
                "gts": gts,
                "depth_map": clean_depth(load_depth(depth_files[ts])),
                "img_size": info["img_size"] or IMG_SIZE,
            })
    return all_frames, cam


# ====================================================================
# 특정 임계값으로 필터링해서 평가
# ====================================================================
def run_at_threshold(all_frames, cam, th, weights):
    # 임계값 적용
    frames = []
    n_pred = 0
    for f in all_frames:
        preds = [dict(p) for p in f["preds_all"] if p["confidence"] >= th]
        attach_mask_from_pkl(preds, f["gts"])   # ← 임계값마다 재매칭
        n_pred += len(preds)
        frames.append({**f, "preds": preds})

    # --- localization error ---
    res, um_p, um_g = evaluate_localization(frames, cam, IOU_THRESHOLD)

    # --- Top-1 선정 & baseline 비교 ---
    same = diff = usable = 0
    diff_reasons = {"거리": 0, "가시성": 0, "신뢰도": 0, "중심": 0}
    cand_counts = []
    per_seq = {}      # 시퀀스별 집계

    for f in frames:
        seq = f["seq"]
        if seq not in per_seq:
            per_seq[seq] = {"usable": 0, "same": 0, "diff": 0, "cand": [],
                            "reasons": {"거리": 0, "가시성": 0,
                                        "신뢰도": 0, "중심": 0}}
        objs = [dict(o) for o in f["preds"]
                if o.get("gt_matched", False)]
        top1, scored = select_top1(objs, f["depth_map"], weights,
                                   f["img_size"], MODE)
        if top1 is None or len(scored) < 2:
            continue                      # 후보 1개면 비교 불가
        usable += 1
        cand_counts.append(len(scored))
        per_seq[seq]["usable"] += 1
        per_seq[seq]["cand"].append(len(scored))

        nearest = min(scored, key=lambda o: o["depth_value"])
        if nearest is top1:
            same += 1
            per_seq[seq]["same"] += 1
        else:
            diff += 1
            per_seq[seq]["diff"] += 1
            # 어떤 항목에서 top1이 nearest보다 우세했는지
            dt, dn = top1["_detail"], nearest["_detail"]
            gaps = {
                "거리":   (dt["proximity"] - dn["proximity"]) * weights["proximity"],
                "가시성": (dt["visibility"] - dn["visibility"]) * weights["visibility"],
                "신뢰도": (dt["confidence"] - dn["confidence"]) * weights["confidence"],
                "중심":   (dt["center"] - dn["center"]) * weights["center"],
            }
            key = max(gaps, key=gaps.get)
            diff_reasons[key] += 1
            per_seq[seq]["reasons"][key] += 1

    return {
        "th": th,
        "예측수": n_pred,
        "매칭": len(res),
        "미매칭예측": um_p,
        "미매칭GT": um_g,
        "err": summarize_errors(res),
        "비교가능프레임": usable,
        "동일": same,
        "차이": diff,
        "차이비율": (diff / usable * 100) if usable else 0.0,
        "평균후보수": float(np.mean(cand_counts)) if cand_counts else 0.0,
        "차이원인": diff_reasons,
        "시퀀스별": per_seq,
    }


# ====================================================================
# 메인
# ====================================================================
def main():
    print("데이터 로딩 중...")
    all_frames, cam = load_all_frames()
    weights = WEIGHTS[MODE]
    print(f"프레임 {len(all_frames)}개 로드 완료\n")

    results = []
    for th in THRESHOLDS:
        print(f"  임계값 {th} 처리 중...")
        results.append(run_at_threshold(all_frames, cam, th, weights))

    # ---------------- 표 1: 검출/매칭 변화 ----------------
    print("\n" + "=" * 78)
    print(" [표 1] confidence 임계값별 검출·매칭 변화")
    print("=" * 78)
    print(f" {'임계값':>7}{'예측수':>9}{'매칭':>8}{'미매칭예측':>11}"
          f"{'미매칭비율':>11}{'미매칭GT':>10}")
    for r in results:
        tot = r["매칭"] + r["미매칭예측"]
        ratio = r["미매칭예측"] / tot * 100 if tot else 0
        print(f" {r['th']:>7}{r['예측수']:>9}{r['매칭']:>8}"
              f"{r['미매칭예측']:>11}{ratio:>10.1f}%{r['미매칭GT']:>10}")

    # ---------------- 표 2: localization error ----------------
    print("\n" + "=" * 78)
    print(" [표 2] confidence 임계값별 3D localization error")
    print("=" * 78)
    print(f" {'임계값':>7}{'매칭':>8}{'3D평균':>10}{'3D중앙':>10}"
          f"{'3D RMSE':>10}{'Z RMSE':>10}{'10mm이내':>10}")
    for r in results:
        e = r["err"]
        if e is None:
            print(f" {r['th']:>7}   (매칭 없음)")
            continue
        print(f" {r['th']:>7}{r['매칭']:>8}{e['3D']['평균']:>9}mm"
              f"{e['3D']['중앙값']:>9}mm{e['3D']['RMSE']:>9}mm"
              f"{e['Z']['RMSE']:>9}mm{s(e)}")

    # ---------------- 표 3: baseline 비교 ----------------
    print("\n" + "=" * 78)
    print(" [표 3] confidence 임계값별 nearest baseline 대비 선택 차이")
    print("=" * 78)
    print(f" {'임계값':>7}{'비교가능':>10}{'평균후보':>10}{'동일':>8}"
          f"{'차이':>8}{'차이비율':>10}")
    for r in results:
        print(f" {r['th']:>7}{r['비교가능프레임']:>10}{r['평균후보수']:>9.1f}"
              f"{r['동일']:>8}{r['차이']:>8}{r['차이비율']:>9.1f}%")

    # ---------------- 표 4: 선택이 갈린 원인 ----------------
    print("\n" + "=" * 78)
    print(" [표 4] 선택이 갈린 주된 원인 (기여도가 가장 큰 항목)")
    print("=" * 78)
    print(f" {'임계값':>7}{'차이':>8}{'거리':>8}{'가시성':>8}{'신뢰도':>8}{'중심':>8}")
    for r in results:
        d = r["차이원인"]
        print(f" {r['th']:>7}{r['차이']:>8}{d['거리']:>8}{d['가시성']:>8}"
              f"{d['신뢰도']:>8}{d['중심']:>8}")

    # ---------------- 표 5: 시퀀스별 차이비율 ----------------
    seqs = sorted(results[0]["시퀀스별"].keys())
    print("\n" + "=" * 78)
    print(" [표 5] 시퀀스별 nearest 대비 선택 차이비율 (임계값별)")
    print("=" * 78)
    header = f" {'임계값':>7}"
    for sq in seqs:
        header += f"{'seq ' + sq:>14}"
    print(header)
    for r in results:
        line = f" {r['th']:>7}"
        for sq in seqs:
            d = r["시퀀스별"].get(sq)
            if not d or d["usable"] == 0:
                line += f"{'-':>14}"
            else:
                ratio = d["diff"] / d["usable"] * 100
                line += f"{d['diff']}/{d['usable']} ({ratio:.0f}%)".rjust(14)
        print(line)

    # ---------------- 표 6: 시퀀스별 평균 후보 수 ----------------
    print("\n" + "=" * 78)
    print(" [표 6] 시퀀스별 평균 후보 수 (임계값별)")
    print("=" * 78)
    header = f" {'임계값':>7}"
    for sq in seqs:
        header += f"{'seq ' + sq:>12}"
    print(header)
    for r in results:
        line = f" {r['th']:>7}"
        for sq in seqs:
            d = r["시퀀스별"].get(sq)
            v = np.mean(d["cand"]) if d and d["cand"] else 0.0
            line += f"{v:>12.1f}"
        print(line)

    # ---------------- 표 7: 시퀀스별 차이 원인 ----------------
    print("\n" + "=" * 78)
    print(" [표 7] 시퀀스별 선택 차이 원인 (거리/가시성/신뢰도/중심)")
    print("=" * 78)
    for sq in seqs:
        print(f"\n  [시퀀스 {sq}]")
        print(f"   {'임계값':>7}{'차이':>7}{'거리':>7}{'가시성':>8}"
              f"{'신뢰도':>8}{'중심':>7}")
        for r in results:
            d = r["시퀀스별"].get(sq)
            if not d:
                continue
            rs = d["reasons"]
            print(f"   {r['th']:>7}{d['diff']:>7}{rs['거리']:>7}"
                  f"{rs['가시성']:>8}{rs['신뢰도']:>8}{rs['중심']:>7}")

    print("\n" + "=" * 78)
    print(" 해석 가이드")
    print("=" * 78)
    print(" - 임계값을 올리면 미매칭(오검출)이 줄어드는지 확인")
    print(" - 차이비율이 크게 떨어지면 → 저품질 검출이 차이의 원인이었음")
    print(" - 차이비율이 유지되면 → 후보 수·장면 복잡도가 원인")
    print(" - 최종 임계값은 validation split에서 결정해 고정 적용할 것")


def s(e):
    """10mm 이내 비율 문자열"""
    return f"{e['10mm이내(%)']:>9.1f}%"


if __name__ == "__main__":
    main()