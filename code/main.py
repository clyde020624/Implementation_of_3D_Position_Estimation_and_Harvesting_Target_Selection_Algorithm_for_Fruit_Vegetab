# -*- coding: utf-8 -*-
"""
====================================================================
 main.py - 전체 파이프라인 (BUP-ST20 파프리카 수확 우선순위)
====================================================================

[확정된 연구 흐름]
  1단계  YOLO 파프리카 검출 성능 평가            (민혁 파트, 별도 완료)
  2단계  YOLO bbox + Depth + intrinsic → 3D 위치 추정
  3단계  동일 프레임 GT(mask 기반) reference와 비교
         → X / Y / Z 및 3D localization error 평가
  4단계  Priority Score(거리·가시성·confidence·중심성) → Top-1 선정
  5단계  거리 단독 baseline 및 ablation 비교

  * 시간적 일관성 분석은 연구 목적과 무관하여 제외 (멘토 피드백)
  * 같은 프레임 내 비교이므로 로봇 이동 영향 없음

--------------------------------------------------------------------
[실행 모드]
  MODE_SOURCE = "gt"   : GT 라벨로 실행 (CSV 도착 전 테스트용)
                "yolo" : 민혁의 YOLO 검출 CSV로 실행 (본 실험)
====================================================================
"""

import os
import glob
import numpy as np

from data_loader import (
    load_depth, clean_depth, load_cam_params, load_bupst20_annotation,
)
from priority import select_top1, WEIGHTS
from localization import (
    evaluate_localization, summarize_errors,
    print_summary as print_loc_summary,
)


# ====================================================================
# ★ 설정 ★
# ====================================================================
MODE_SOURCE = "gt"          # "gt" 또는 "yolo"
MODE        = "visibility"  # Priority Score 구성

DATA_DIR = "dataset_bulk"
CAM_PATH = r"C:\Users\82109\Downloads\cam_params.yaml"   # ★ 실제 경로
IMG_SIZE = (720, 1280)

DETECTION_CSV = "eval_detections.csv"    # "yolo" 모드에서 사용
IOU_THRESHOLD = 0.5                      # 예측↔GT 매칭 기준
SHOW_TOP1_FRAMES = 2                     # Top-1 예시로 보여줄 프레임 수


# ====================================================================
# 데이터 로드
# ====================================================================
def load_gt_frames(seq):
    """GT(pkl) + depth 로 프레임 구성"""
    d_dir = os.path.join(DATA_DIR, "depth", seq)
    a_dir = os.path.join(DATA_DIR, "annotations", seq)
    depth_files = {os.path.splitext(os.path.basename(p))[0]: p
                   for p in glob.glob(os.path.join(d_dir, "*.tif*"))}
    ann_files = {os.path.splitext(os.path.basename(p))[0]: p
                 for p in glob.glob(os.path.join(a_dir, "*.pkl"))}

    frames = []
    for ts in sorted(set(depth_files) & set(ann_files)):
        frames.append({
            "image_id": ts,
            "gts": load_bupst20_annotation(ann_files[ts]),
            "depth_map": clean_depth(load_depth(depth_files[ts])),
            "img_size": IMG_SIZE,
        })
    return frames


def attach_predictions(frames, seq):
    """
    각 프레임에 예측(preds)을 채운다.
      - "yolo" 모드: 민혁 CSV의 검출 결과
      - "gt"   모드: GT를 예측 자리에 넣어 파이프라인 점검
                    (오차는 0에 가깝게 나오는 것이 정상)
    """
    if MODE_SOURCE == "yolo":
        from load_detections import load_detections_csv, attach_mask_from_pkl
        det = load_detections_csv(DETECTION_CSV)
        by_ts = {ts: info for (s, ts), info in det.items() if s == seq}

        kept = []
        for f in frames:
            info = by_ts.get(f["image_id"])
            if info is None:
                continue
            preds = info["objects"]
            # 가시성용 mask는 YOLO에 없으므로 GT에서 IoU 매칭해 부여
            attach_mask_from_pkl(preds, f["gts"])
            f["preds"] = preds
            kept.append(f)
        return kept
    else:
        for f in frames:
            # GT를 예측처럼 사용 (bbox/confidence만)
            f["preds"] = [{"bbox": g["bbox"], "confidence": 1.0,
                           "mask_area": g.get("mask_area"), "id": g.get("id")}
                          for g in f["gts"]]
        return frames


# ====================================================================
# 4단계: Top-1 선정 예시 출력
# ====================================================================
def show_top1(frames, weights):
    print(f"\n ▶ [4단계] Priority Score → Top-1 선정 "
          f"(앞 {SHOW_TOP1_FRAMES}개 프레임 예시)")
    for f in frames[:SHOW_TOP1_FRAMES]:
        objs = [dict(o) for o in f["preds"]]      # 원본 보존
        top1, scored = select_top1(objs, f["depth_map"],
                                   weights, f["img_size"], MODE)
        if top1 is None:
            print(f"   [{f['image_id']}] depth 유효 객체 없음")
            continue
        print(f"\n   [프레임 {f['image_id']}] 후보 {len(scored)}개")
        for o in sorted(scored, key=lambda x: x["score"], reverse=True)[:3]:
            d = o["_detail"]
            mark = "★" if o is top1 else " "
            print(f"    {mark} 총점 {o['score']:.3f} | 거리 {o['depth_value']:.0f}mm "
                  f"| 근접 {d['proximity']} 가시성 {d['visibility']} "
                  f"신뢰도 {d['confidence']}")
        print(f"      → 1순위 선정 (총점 {top1['score']:.3f})")


# ====================================================================
# 5단계: baseline 비교 (거리 단독 vs Priority Score)
# ====================================================================
def compare_baseline(frames, weights):
    """거리만 보는 baseline과 Priority Score가 같은 대상을 고르는지 비교"""
    same, diff, total = 0, 0, 0
    diff_cases = []

    for f in frames:
        objs = [dict(o) for o in f["preds"]]
        top1, scored = select_top1(objs, f["depth_map"],
                                   weights, f["img_size"], MODE)
        if top1 is None or not scored:
            continue
        # baseline: 가장 가까운 것
        nearest = min(scored, key=lambda o: o["depth_value"])
        total += 1
        if nearest is top1:
            same += 1
        else:
            diff += 1
            if len(diff_cases) < 3:
                diff_cases.append((f["image_id"], nearest, top1))

    print(f"\n ▶ [5단계] 거리 단독 baseline 대비 비교")
    if total == 0:
        print("   비교 가능한 프레임 없음")
        return
    print(f"   전체 {total}프레임 | 동일 선택 {same} ({same/total*100:.1f}%) "
          f"| 다른 선택 {diff} ({diff/total*100:.1f}%)")
    for img_id, nb, ours in diff_cases:
        print(f"\n   [{img_id}] 선택이 갈린 예시")
        print(f"     거리 baseline : 거리 {nb['depth_value']:.0f}mm, "
              f"가시성 {nb['_detail']['visibility']}")
        print(f"     Priority Score: 거리 {ours['depth_value']:.0f}mm, "
              f"가시성 {ours['_detail']['visibility']}")


# ====================================================================
# 메인
# ====================================================================
def main():
    weights = WEIGHTS[MODE]

    print("=" * 66)
    print(" 파프리카 수확 우선순위 파이프라인")
    print(f"   입력: {MODE_SOURCE.upper()}  |  Priority 구성: {MODE}")
    print("=" * 66)

    if not os.path.isdir(DATA_DIR):
        print(f"\n'{DATA_DIR}' 폴더 없음. extract_bulk.py를 먼저 실행하세요.")
        return
    if not os.path.exists(CAM_PATH):
        print(f"\ncam_params 경로 확인 필요: {CAM_PATH}")
        return

    cam = load_cam_params(CAM_PATH)
    sequences = sorted(os.listdir(os.path.join(DATA_DIR, "depth")))

    all_loc_results = []
    tot_um_pred = tot_um_gt = 0

    for seq in sequences:
        frames = load_gt_frames(seq)
        frames = attach_predictions(frames, seq)
        if not frames:
            print(f"\n[시퀀스 {seq}] 사용할 프레임 없음")
            continue

        n_pred = sum(len(f["preds"]) for f in frames)
        n_gt = sum(len(f["gts"]) for f in frames)
        print(f"\n{'─'*66}")
        print(f" [시퀀스 {seq}] 프레임 {len(frames)}개 | "
              f"예측 {n_pred}개 / GT {n_gt}개")
        print(f"{'─'*66}")

        # --- 2~3단계: 3D 위치 추정 + 오차 평가 ---
        res, um_p, um_g = evaluate_localization(frames, cam, IOU_THRESHOLD)
        all_loc_results.extend(res)
        tot_um_pred += um_p
        tot_um_gt += um_g

        if res:
            e3d = np.array([r["err_3D"] for r in res], dtype=float)
            ez = np.array([abs(r["err_Z"]) for r in res], dtype=float)
            print(f"\n ▶ [2~3단계] 3D 위치 추정 오차")
            print(f"   매칭 {len(res)}개 (미매칭: 예측 {um_p} / GT {um_g})")
            print(f"   3D 오차 평균 {e3d.mean():.1f}mm | 중앙값 {np.median(e3d):.1f}mm")
            print(f"   Z(깊이) 오차 평균 {ez.mean():.1f}mm")

        # --- 4단계: Top-1 선정 ---
        show_top1(frames, weights)

        # --- 5단계: baseline 비교 ---
        compare_baseline(frames, weights)

    # ---------------- 전체 요약 ----------------
    print("\n")
    print_loc_summary(summarize_errors(all_loc_results), tot_um_pred, tot_um_gt)


if __name__ == "__main__":
    main()