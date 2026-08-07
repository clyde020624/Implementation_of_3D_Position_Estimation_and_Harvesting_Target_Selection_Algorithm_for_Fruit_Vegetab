# -*- coding: utf-8 -*-
"""
====================================================================
 diagnose_z.py - 거리(Z)가 로봇 이동에 오염됐는지 확인
====================================================================

앞선 진단에서 3D 흔들림(72mm)의 대부분이 X축(60mm)이었고,
이는 로봇의 옆이동 때문으로 판별되었습니다.

이제 Z(거리)만 따로 봅니다. Z를 일관성 지표로 쓸 수 있는지 판단:

  [판별 기준]
  1) Z와 관찰횟수의 상관 → 높으면 로봇 전진이동에 오염(X), 낮으면 깨끗(O)
     (참고: X축은 상관 0.903으로 명백히 오염됨)
  2) Z 궤적의 형태 → 한 방향으로 계속 증가/감소하면 이동, 위아래 흔들리면 노이즈
  3) 추세 제거(detrend) 후 Z 흔들림 → 이동 성분을 뺀 '순수 추정 오차'

실행: python diagnose_z.py
====================================================================
"""

import os
import glob
import numpy as np
from collections import defaultdict

from data_loader import (
    load_depth, clean_depth, load_cam_params, load_bupst20_annotation,
)
from priority import extract_depth


DATA_DIR = "dataset_bulk"
CAM_PATH = "cam_params.yaml"   # ★ 실제 경로로


def load_sequence(seq):
    d_dir = os.path.join(DATA_DIR, "depth", seq)
    a_dir = os.path.join(DATA_DIR, "annotations", seq)
    depth_files = {os.path.splitext(os.path.basename(p))[0]: p
                   for p in glob.glob(os.path.join(d_dir, "*.tif*"))}
    ann_files = {os.path.splitext(os.path.basename(p))[0]: p
                 for p in glob.glob(os.path.join(a_dir, "*.pkl"))}
    common = sorted(set(depth_files) & set(ann_files))
    frames = []
    for ts in common:
        frames.append({
            "ts": ts,
            "objects": load_bupst20_annotation(ann_files[ts]),
            "depth_map": clean_depth(load_depth(depth_files[ts])),
        })
    return frames


def collect_z_tracks(frames):
    """객체별 Z 시계열 수집"""
    tracks = defaultdict(list)
    for fi, f in enumerate(frames):
        for obj in f["objects"]:
            Z = extract_depth(obj["bbox"], f["depth_map"])
            if Z is not None:
                tracks[obj["id"]].append((fi, Z))
    return tracks


def detrend_std(fi_list, z_list):
    """직선 추세(=로봇 이동)를 뺀 뒤의 표준편차 = 순수 흔들림"""
    if len(z_list) < 3:
        return None, None
    x = np.array(fi_list, dtype=float)
    y = np.array(z_list, dtype=float)
    # 1차 직선 피팅 (이동 성분)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    return float(np.std(residual)), float(slope)


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"'{DATA_DIR}' 폴더 없음")
        return

    sequences = sorted(os.listdir(os.path.join(DATA_DIR, "depth")))

    rows = []   # (seq, id, n, std_z, detrend_std, slope)

    for seq in sequences:
        frames = load_sequence(seq)
        if len(frames) < 3:
            continue
        tracks = collect_z_tracks(frames)
        for oid, pts in tracks.items():
            if len(pts) < 3:
                continue
            fi = [p[0] for p in pts]
            zs = [p[1] for p in pts]
            std_raw = float(np.std(zs))
            std_dt, slope = detrend_std(fi, zs)
            if std_dt is None:
                continue
            rows.append({"seq": seq, "id": oid, "n": len(pts),
                         "std_raw": std_raw, "std_dt": std_dt, "slope": slope})

    if not rows:
        print("분석할 데이터 없음")
        return

    ns      = np.array([r["n"] for r in rows], float)
    raw     = np.array([r["std_raw"] for r in rows], float)
    dt      = np.array([r["std_dt"] for r in rows], float)
    slopes  = np.array([r["slope"] for r in rows], float)

    # ---------------- 진단 1: Z와 관찰횟수 상관 ----------------
    print("=" * 68)
    print(" [Z 진단 1] 거리(Z) 흔들림 vs 관찰 횟수 상관")
    print("=" * 68)
    corr = np.corrcoef(ns, raw)[0, 1] if ns.std() > 0 else 0
    print(f"   Z 흔들림과 관찰횟수 상관계수: {corr:+.3f}")
    print(f"   (참고: X축은 +0.903 으로 명백히 로봇 이동에 오염되었음)")
    if corr > 0.6:
        print("   판정: Z도 로봇 이동(전진/후진)에 오염됨 ⚠️")
    elif corr > 0.3:
        print("   판정: Z에 이동 성분이 일부 섞임 (부분 오염)")
    else:
        print("   판정: Z는 관찰횟수와 무관 → 로봇 이동 영향 작음 ✅")

    print("\n   관찰횟수 구간별 Z 흔들림:")
    for lo, hi in [(3, 5), (6, 10), (11, 20), (21, 100)]:
        m = [r["std_raw"] for r in rows if lo <= r["n"] <= hi]
        if m:
            print(f"     {lo:>2}~{hi:>3}회: {np.mean(m):7.1f}mm  ({len(m)}개)")

    # ---------------- 진단 2: 추세(이동) 성분 크기 ----------------
    print("\n" + "=" * 68)
    print(" [Z 진단 2] Z 궤적의 추세 = 로봇 전진/후진 성분")
    print("=" * 68)
    print(f"   프레임당 Z 변화(기울기) 평균: {np.mean(slopes):+.2f} mm/frame")
    print(f"   기울기 절대값 평균          : {np.mean(np.abs(slopes)):.2f} mm/frame")
    if np.mean(np.abs(slopes)) > 3:
        print("   → 프레임마다 Z가 꾸준히 변함 = 로봇이 앞뒤로 이동 중")
    else:
        print("   → Z 추세가 약함 = 로봇이 주로 옆으로만 이동 (Z는 비교적 유지)")

    # ---------------- 진단 3: 추세 제거 후 순수 흔들림 ----------------
    print("\n" + "=" * 68)
    print(" [Z 진단 3] 추세 제거 후 '순수 추정 흔들림'")
    print("=" * 68)
    print(f"   원본 Z 흔들림      : 평균 {raw.mean():6.1f}mm | 중앙값 {np.median(raw):6.1f}mm")
    print(f"   추세 제거 후       : 평균 {dt.mean():6.1f}mm | 중앙값 {np.median(dt):6.1f}mm")
    reduction = (1 - dt.mean() / raw.mean()) * 100 if raw.mean() > 0 else 0
    print(f"   이동 성분이 차지한 비율: {reduction:.1f}%")

    stable20 = np.sum(dt < 20)
    stable10 = np.sum(dt < 10)
    print(f"\n   추세 제거 후 안정 비율:")
    print(f"     10mm 이내: {stable10}/{len(dt)} ({stable10/len(dt)*100:.1f}%)")
    print(f"     20mm 이내: {stable20}/{len(dt)} ({stable20/len(dt)*100:.1f}%)")

    # ---------------- 종합 권고 ----------------
    print("\n" + "=" * 68)
    print(" 종합 판단")
    print("=" * 68)
    if corr < 0.3 and np.mean(np.abs(slopes)) < 3:
        print(" ✅ Z(거리)는 로봇 이동 영향이 작아 일관성 지표로 사용 가능")
        print(f"    → 논문 수치: 거리 추정 흔들림 평균 {raw.mean():.1f}mm")
    else:
        print(" ⚠️ Z에도 로봇 이동 성분이 있음")
        print(f"    → '추세 제거 후' 값을 쓰는 것을 권장: 평균 {dt.mean():.1f}mm")
        print("       (로봇 이동을 뺀 순수 추정 흔들림)")


if __name__ == "__main__":
    main()
