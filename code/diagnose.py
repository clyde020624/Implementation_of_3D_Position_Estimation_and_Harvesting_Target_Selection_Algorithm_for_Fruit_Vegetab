# -*- coding: utf-8 -*-
"""
====================================================================
 diagnose.py - 일관성 수치 원인 진단
====================================================================

run_bulk.py 결과(3D 흔들림 평균 72mm)가
  (A) 진짜 '추정 오차'인지
  (B) 로봇 이동에 따른 '자연스러운 위치 변화'인지
를 판별합니다.

[진단 3가지]
  1) X / Y / Z 축별 흔들림 분리
     → X,Y가 크고 Z가 작으면 = 로봇 옆이동 영향(B)
     → 세 축이 비슷하면 = 추정 오차(A)
  2) 관찰 횟수 vs 흔들림 상관관계
     → 강한 양의 상관 = 긴 추적일수록 이동 누적(B)
  3) 극단값 객체의 프레임별 위치 추적
     → 위치가 서서히 변하면 이동(B), 튀면 오차(A)

실행: python diagnose.py
====================================================================
"""

import os
import glob
import numpy as np
from collections import defaultdict

from data_loader import (
    load_depth, clean_depth, load_cam_params, load_bupst20_annotation, pixel_to_3d,
)
from priority import extract_depth


DATA_DIR = "dataset_bulk"
CAM_PATH = r"C:\Users\82109\Downloads\cam_params.yaml"   # ★ 실제 경로로
IMG_SIZE = (720, 1280)


def load_sequence(seq):
    d_dir = os.path.join(DATA_DIR, "depth", seq)
    a_dir = os.path.join(DATA_DIR, "annotations", seq)
    depth_files = {os.path.splitext(os.path.basename(p))[0]: p
                   for p in glob.glob(os.path.join(d_dir, "*.tif*"))}
    ann_files = {os.path.splitext(os.path.basename(p))[0]: p
                 for p in glob.glob(os.path.join(a_dir, "*.pkl"))}
    common = sorted(set(depth_files) & set(ann_files))   # 시간순 정렬
    frames = []
    for ts in common:
        frames.append({
            "ts": ts,
            "objects": load_bupst20_annotation(ann_files[ts]),
            "depth_map": clean_depth(load_depth(depth_files[ts])),
        })
    return frames


def collect_xyz(frames, cam):
    """객체별로 (프레임순서, ts, X, Y, Z, u, v) 기록"""
    tracks = defaultdict(list)
    for fi, f in enumerate(frames):
        for obj in f["objects"]:
            Z = extract_depth(obj["bbox"], f["depth_map"])
            if Z is None:
                continue
            x1, y1, x2, y2 = obj["bbox"]
            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            X, Y, Zc = pixel_to_3d(u, v, Z, cam)
            tracks[obj["id"]].append({
                "fi": fi, "ts": f["ts"], "X": X, "Y": Y, "Z": Zc, "u": u, "v": v
            })
    return tracks


def main():
    if not os.path.exists(CAM_PATH):
        print(f"cam_params 경로 확인: {CAM_PATH}")
        return
    cam = load_cam_params(CAM_PATH)

    depth_root = os.path.join(DATA_DIR, "depth")
    sequences = sorted(os.listdir(depth_root))

    all_stats = []
    worst = None   # 가장 불안정한 객체 기록

    print("=" * 68)
    print(" [진단 1] 축별(X/Y/Z) 흔들림 분리")
    print("=" * 68)

    for seq in sequences:
        frames = load_sequence(seq)
        if len(frames) < 2:
            continue
        tracks = collect_xyz(frames, cam)

        sx, sy, sz, s3, obs = [], [], [], [], []
        for oid, pts in tracks.items():
            if len(pts) < 2:
                continue
            X = np.array([p["X"] for p in pts])
            Y = np.array([p["Y"] for p in pts])
            Z = np.array([p["Z"] for p in pts])
            stdx, stdy, stdz = X.std(), Y.std(), Z.std()
            std3 = np.sqrt(stdx**2 + stdy**2 + stdz**2)
            sx.append(stdx); sy.append(stdy); sz.append(stdz)
            s3.append(std3); obs.append(len(pts))
            all_stats.append({"seq": seq, "id": oid, "n": len(pts),
                              "sx": stdx, "sy": stdy, "sz": stdz, "s3": std3,
                              "pts": pts})
            if worst is None or std3 > worst["s3"]:
                worst = all_stats[-1]

        print(f"\n[시퀀스 {seq}] 추적 객체 {len(sx)}개")
        print(f"   X축 흔들림 평균: {np.mean(sx):7.1f} mm")
        print(f"   Y축 흔들림 평균: {np.mean(sy):7.1f} mm")
        print(f"   Z축 흔들림 평균: {np.mean(sz):7.1f} mm  ← 거리")
        print(f"   3D 합성       : {np.mean(s3):7.1f} mm")

    # ---------- 전체 축별 요약 ----------
    if not all_stats:
        print("분석할 데이터가 없습니다.")
        return

    SX = np.mean([s["sx"] for s in all_stats])
    SY = np.mean([s["sy"] for s in all_stats])
    SZ = np.mean([s["sz"] for s in all_stats])
    print("\n" + "-" * 68)
    print(f" 전체 평균:  X {SX:.1f}mm | Y {SY:.1f}mm | Z {SZ:.1f}mm")
    dominant = max([("X", SX), ("Y", SY), ("Z", SZ)], key=lambda t: t[1])
    print(f" 가장 큰 축: {dominant[0]} ({dominant[1]:.1f}mm)")
    if (SX + SY) / 2 > SZ * 1.5:
        print(" 판정: X,Y가 Z보다 훨씬 큼 → 로봇 이동(옆/위아래) 영향이 지배적 (B)")
        print("       (추정 오차가 아니라 카메라가 움직여서 생긴 위치 변화)")
    elif SZ > (SX + SY) / 2 * 1.5:
        print(" 판정: Z(거리)가 X,Y보다 큼 → depth 추정 자체가 불안정 (A)")
        print("       (가림/노이즈로 거리값이 튀는 것으로 의심)")
    else:
        print(" 판정: 세 축이 비슷한 수준 → 이동과 오차가 섞여 있음")

    # ---------- 진단 2: 관찰 횟수 vs 흔들림 ----------
    print("\n" + "=" * 68)
    print(" [진단 2] 관찰 횟수 vs 흔들림 상관관계")
    print("=" * 68)
    ns = np.array([s["n"] for s in all_stats], dtype=float)
    s3s = np.array([s["s3"] for s in all_stats], dtype=float)
    if len(ns) > 2 and ns.std() > 0:
        corr = np.corrcoef(ns, s3s)[0, 1]
        print(f"   상관계수: {corr:+.3f}")
        if corr > 0.4:
            print("   판정: 관찰이 길수록 흔들림 큼 → 로봇 이동 누적 영향(B)")
        elif corr < -0.4:
            print("   판정: 관찰이 길수록 흔들림 작음 (이례적)")
        else:
            print("   판정: 뚜렷한 관계 없음 → 이동 누적만으로 설명 안 됨(A 가능성)")
        # 구간별 평균
        print("\n   관찰횟수 구간별 평균 흔들림:")
        for lo, hi in [(2, 5), (6, 10), (11, 20), (21, 100)]:
            m = [s["s3"] for s in all_stats if lo <= s["n"] <= hi]
            if m:
                print(f"     {lo:>2}~{hi:>3}회: {np.mean(m):7.1f}mm  ({len(m)}개)")

    # ---------- 진단 3: 최악 객체의 궤적 ----------
    print("\n" + "=" * 68)
    print(f" [진단 3] 가장 불안정한 객체 추적 (id={worst['id']}, "
          f"seq={worst['seq']}, 3D흔들림 {worst['s3']:.1f}mm)")
    print("=" * 68)
    pts = worst["pts"]
    print(f"   {'프레임':>5} {'화면u':>7} {'화면v':>7} {'X(mm)':>9} "
          f"{'Y(mm)':>9} {'Z(mm)':>8}")
    step = max(1, len(pts) // 12)   # 너무 길면 솎아서 출력
    for p in pts[::step]:
        print(f"   {p['fi']:>5} {p['u']:>7.0f} {p['v']:>7.0f} "
              f"{p['X']:>9.1f} {p['Y']:>9.1f} {p['Z']:>8.0f}")

    # 연속 변화량 확인
    Zs = np.array([p["Z"] for p in pts])
    dZ = np.abs(np.diff(Zs))
    med = np.median(dZ) if len(dZ) else 0.0
    print(f"\n   프레임 간 Z 변화: 중앙값 {med:.1f}mm, 평균 {dZ.mean():.1f}mm, "
          f"최대 {dZ.max():.1f}mm")
    # 중앙값 대비 최대 점프가 크면 '튀는' 것 (평균은 튐 자체에 오염되므로 중앙값 사용)
    if len(dZ) and dZ.max() > max(50.0, 5 * (med + 1e-9)):
        print("   → 특정 구간에서 급격히 튐 = 추정 오류(가림/노이즈) 의심 (A)")
    else:
        print("   → 서서히 변함 = 로봇 이동에 따른 자연스러운 변화 (B)")


if __name__ == "__main__":
    main()
