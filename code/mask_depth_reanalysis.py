# -*- coding: utf-8 -*-
"""
====================================================================
mask_depth_reanalysis.py

목적
--------------------------------------------------------------------
기존 Top-1 selection에서 사용하는 거리:

    YOLO bbox center 5x5 median depth

를

    matched GT instance mask 내부 median depth

로 바꾸었을 때 selection 결과가 얼마나 달라지는지 확인한다.

중요
--------------------------------------------------------------------
- 기존 priority.py / localization.py / ablation.py 등은 수정하지 않는다.
- 3D localization 결과(3D median 8.7 mm)는 건드리지 않는다.
- 이 스크립트는 "selection 거리 정의에 따른 영향"만 확인한다.
- GT mask depth는 현재 단계에서는 최종 배포 방법이 아니라
  대조/재분석용 reference distance로 사용한다.

검산 목표
--------------------------------------------------------------------
1) 비교 가능 프레임 = 364
2) 기존 bbox-depth nearest vs 기존 Full = 103/364 부근 재현
3) bbox-depth nearest vs mask-depth nearest = 108/364 부근 재현
4) D-only(mask depth) vs mask-depth nearest = 0/364
5) sequence 406 데모:
       candidate 6 mask depth ≈ 955 mm
       candidate 0 mask depth ≈ 755 mm
       new nearest = candidate 0

실행
--------------------------------------------------------------------
python mask_depth_reanalysis.py
====================================================================
"""

import os
import glob
import csv

import numpy as np

from data_loader import (
    load_depth,
    clean_depth,
    load_bupst20_annotation,
)

from load_detections import (
    load_detections_csv,
    attach_mask_from_pkl,
)

from localization import depth_from_bbox_center

from priority import (
    priority_score,
    WEIGHTS,
)


# ====================================================================
# 설정
# ====================================================================

DATA_DIR = "dataset_bulk"
DETECTION_CSV = "eval_detections_fixed.csv"

SEQUENCES = {str(i) for i in range(400, 410)}

CONF_THRESHOLD = 0.30
IOU_THRESHOLD = 0.50

MODE = "visibility"

FULL_WEIGHTS = WEIGHTS[MODE]

# 거리만 사용하는 D-only
D_ONLY_WEIGHTS = {
    "proximity": 1.0,
    "confidence": 0.0,
    "center": 0.0,
    "visibility": 0.0,
}

# 406 데모 검산용
CHECK_SEQ = "406"
CHECK_FRAME = "1600938551617024"

# 프레임별 결과 저장
OUTPUT_CSV = "mask_depth_reanalysis_results.csv"


# ====================================================================
# GT mask 내부 median depth
# ====================================================================

def get_mask_median_depth(mask, depth_map):
    """
    GT instance mask에 해당하는 depth 픽셀만 추출하여
    유효값(>0)의 median을 반환한다.
    """

    if mask is None:
        return None

    if mask.shape != depth_map.shape:
        return None

    values = depth_map[mask]

    # clean_depth()에서 65535는 이미 0으로 치환됨
    valid = values[values > 0]

    if valid.size == 0:
        return None

    return float(np.median(valid))


# ====================================================================
# precomputed depth를 이용한 Priority Top-1
# ====================================================================

def select_top1_precomputed(
    objects,
    depth_key,
    weights,
    img_size,
    mode="visibility",
):
    """
    priority.select_top1()과 차이점:

    기존 select_top1():
        bbox center 5x5에서 depth를 새로 계산

    이 함수:
        객체에 이미 저장된 depth_key 값을 그대로 사용

    예:
        depth_key="bbox_depth"
        depth_key="mask_depth"
    """

    objs = [dict(o) for o in objects]

    valid = []

    for obj in objs:

        depth_value = obj.get(depth_key)

        if depth_value is None:
            continue

        obj["depth_value"] = float(depth_value)

        valid.append(obj)

    if not valid:
        return None, []

    depths = [
        obj["depth_value"]
        for obj in valid
    ]

    min_d = min(depths)
    max_d = max(depths)

    img_center = (
        img_size[0] / 2,
        img_size[1] / 2,
    )

    for obj in valid:

        obj["score"] = priority_score(
            obj,
            min_d,
            max_d,
            weights,
            img_center,
            mode,
        )

    top1 = max(
        valid,
        key=lambda x: x["score"]
    )

    return top1, valid


# ====================================================================
# candidate 준비
# ====================================================================

def prepare_candidates(preds, gts, depth_map):
    """
    같은 후보 집합에서 bbox depth와 mask depth를 모두 계산한다.

    중요:
    기존 방식과 새 방식의 공정한 비교를 위해
    두 depth가 모두 유효한 candidate만 사용한다.
    """

    # 기존 방식과 동일:
    # confidence filtering 이후 GT matching 수행
    attach_mask_from_pkl(
        preds,
        gts,
        IOU_THRESHOLD
    )

    matched = [
        p for p in preds
        if p.get("gt_matched", False)
    ]

    gt_by_id = {
        gt["id"]: gt
        for gt in gts
    }

    candidates = []

    for pred in matched:

        gt_id = pred.get("matched_gt_id")

        gt = gt_by_id.get(gt_id)

        if gt is None:
            continue

        # ------------------------------------------------------------
        # 기존 거리:
        # YOLO bbox center 5x5 median
        # ------------------------------------------------------------

        bbox_depth, center = depth_from_bbox_center(
            pred["bbox"],
            depth_map,
            region_half=2,
        )

        if bbox_depth is None:
            continue

        # ------------------------------------------------------------
        # 새 대조 거리:
        # matched GT mask median
        # ------------------------------------------------------------

        mask = gt.get("instance_mask")

        mask_depth = get_mask_median_depth(
            mask,
            depth_map
        )

        if mask_depth is None:
            continue

        obj = dict(pred)

        obj["bbox_depth"] = float(bbox_depth)
        obj["mask_depth"] = float(mask_depth)

        obj["depth_difference"] = (
            float(bbox_depth)
            -
            float(mask_depth)
        )

        obj["bbox_center"] = center

        # bbox center가 GT mask 내부인지도 기록
        cx, cy = center

        if (
            mask is not None
            and
            0 <= cy < mask.shape[0]
            and
            0 <= cx < mask.shape[1]
        ):
            obj["center_in_mask"] = bool(
                mask[cy, cx]
            )
        else:
            obj["center_in_mask"] = False

        candidates.append(obj)

    return candidates


# ====================================================================
# MAIN
# ====================================================================

def main():

    detections = load_detections_csv(
        DETECTION_CSV
    )

    depth_root = os.path.join(
        DATA_DIR,
        "depth"
    )

    ann_root = os.path.join(
        DATA_DIR,
        "annotations"
    )

    sequences = [
        seq for seq in sorted(
            os.listdir(depth_root)
        )
        if seq in SEQUENCES
    ]

    # ------------------------------------------------------------
    # 통계
    # ------------------------------------------------------------

    usable = 0

    old_nearest_vs_old_full = 0

    old_nearest_vs_new_nearest = 0

    new_nearest_vs_new_full = 0

    d_only_vs_new_nearest = 0

    candidate_sum = 0

    per_seq = {}

    rows = []

    check_frame_found = False


    print("=" * 78)
    print(" GT-mask median depth 기반 Top-1 재분석")
    print("=" * 78)

    print(
        f"confidence >= {CONF_THRESHOLD}, "
        f"IoU >= {IOU_THRESHOLD}"
    )

    print()


    # ================================================================
    # sequence loop
    # ================================================================

    for seq in sequences:

        per_seq[seq] = {
            "usable": 0,
            "old_new_nearest_diff": 0,
            "new_full_diff": 0,
        }

        depth_dir = os.path.join(
            depth_root,
            seq
        )

        ann_dir = os.path.join(
            ann_root,
            seq
        )

        depth_files = {
            os.path.splitext(
                os.path.basename(p)
            )[0]: p
            for p in glob.glob(
                os.path.join(
                    depth_dir,
                    "*.tif*"
                )
            )
        }

        ann_files = {
            os.path.splitext(
                os.path.basename(p)
            )[0]: p
            for p in glob.glob(
                os.path.join(
                    ann_dir,
                    "*.pkl"
                )
            )
        }


        # ============================================================
        # frame loop
        # ============================================================

        for frame_id in sorted(
            set(depth_files)
            &
            set(ann_files)
        ):

            info = detections.get(
                (seq, frame_id)
            )

            if info is None:
                continue


            # --------------------------------------------------------
            # confidence >= 0.3
            # --------------------------------------------------------

            preds = [
                dict(o)
                for o in info["objects"]
                if o["confidence"]
                >= CONF_THRESHOLD
            ]

            if len(preds) < 2:
                continue


            # --------------------------------------------------------
            # depth / GT
            # --------------------------------------------------------

            depth = clean_depth(
                load_depth(
                    depth_files[frame_id]
                )
            )

            gts = load_bupst20_annotation(
                ann_files[frame_id]
            )


            # --------------------------------------------------------
            # candidate 준비
            # --------------------------------------------------------

            candidates = prepare_candidates(
                preds,
                gts,
                depth,
            )

            if len(candidates) < 2:
                continue


            img_size = (
                info["img_size"]
                if info["img_size"] is not None
                else (720, 1280)
            )


            # ========================================================
            # ① 기존 bbox-center nearest
            # ========================================================

            old_nearest = min(
                candidates,
                key=lambda x: x[
                    "bbox_depth"
                ]
            )


            # ========================================================
            # ② GT-mask depth nearest
            # ========================================================

            new_nearest = min(
                candidates,
                key=lambda x: x[
                    "mask_depth"
                ]
            )


            # ========================================================
            # ③ 기존 bbox-depth Full
            # ========================================================

            old_full, _ = (
                select_top1_precomputed(
                    candidates,
                    "bbox_depth",
                    FULL_WEIGHTS,
                    img_size,
                    MODE,
                )
            )


            # ========================================================
            # ④ 새로운 mask-depth Full
            # ========================================================

            new_full, _ = (
                select_top1_precomputed(
                    candidates,
                    "mask_depth",
                    FULL_WEIGHTS,
                    img_size,
                    MODE,
                )
            )


            # ========================================================
            # ⑤ mask-depth D-only
            # ========================================================

            d_only, _ = (
                select_top1_precomputed(
                    candidates,
                    "mask_depth",
                    D_ONLY_WEIGHTS,
                    img_size,
                    MODE,
                )
            )


            if (
                old_full is None
                or
                new_full is None
                or
                d_only is None
            ):
                continue


            # --------------------------------------------------------
            # 실제 분석 대상 frame
            # --------------------------------------------------------

            usable += 1
            candidate_sum += len(candidates)

            per_seq[seq]["usable"] += 1


            # ========================================================
            # 검산/비교
            # ========================================================

            if (
                old_nearest["id"]
                !=
                old_full["id"]
            ):
                old_nearest_vs_old_full += 1


            if (
                old_nearest["id"]
                !=
                new_nearest["id"]
            ):
                old_nearest_vs_new_nearest += 1

                per_seq[seq][
                    "old_new_nearest_diff"
                ] += 1


            if (
                new_nearest["id"]
                !=
                new_full["id"]
            ):
                new_nearest_vs_new_full += 1

                per_seq[seq][
                    "new_full_diff"
                ] += 1


            if (
                d_only["id"]
                !=
                new_nearest["id"]
            ):
                d_only_vs_new_nearest += 1


            # ========================================================
            # CSV 기록
            # ========================================================

            rows.append({
                "sequence_id": seq,
                "frame_id": frame_id,
                "n_candidates": len(
                    candidates
                ),

                "old_nearest_id":
                    old_nearest["id"],

                "old_nearest_depth":
                    old_nearest[
                        "bbox_depth"
                    ],

                "new_nearest_id":
                    new_nearest["id"],

                "new_nearest_depth":
                    new_nearest[
                        "mask_depth"
                    ],

                "old_full_id":
                    old_full["id"],

                "new_full_id":
                    new_full["id"],

                "old_new_nearest_diff":
                    int(
                        old_nearest["id"]
                        !=
                        new_nearest["id"]
                    ),

                "new_nearest_full_diff":
                    int(
                        new_nearest["id"]
                        !=
                        new_full["id"]
                    ),
            })


            # ========================================================
            # sequence 406 demo 상세 검산
            # ========================================================

            if (
                seq == CHECK_SEQ
                and
                frame_id == CHECK_FRAME
            ):

                check_frame_found = True

                print()
                print("=" * 78)
                print(
                    " [406 데모 프레임 검산]"
                )
                print("=" * 78)

                print(
                    f" sequence={seq}, "
                    f"frame={frame_id}"
                )

                print()

                for c in sorted(
                    candidates,
                    key=lambda x:
                        x["id"]
                ):

                    print(
                        f" candidate {c['id']:>2} | "
                        f"bboxZ={c['bbox_depth']:>6.0f} mm | "
                        f"maskZ={c['mask_depth']:>6.0f} mm | "
                        f"inMask={c['center_in_mask']}"
                    )

                print()

                print(
                    " 기존 bbox-depth nearest : "
                    f"candidate "
                    f"{old_nearest['id']}"
                )

                print(
                    " GT-mask-depth nearest    : "
                    f"candidate "
                    f"{new_nearest['id']}"
                )

                print(
                    " GT-mask-depth Full       : "
                    f"candidate "
                    f"{new_full['id']}"
                )

                print()


    # =================================================================
    # 결과 출력
    # =================================================================

    print()
    print("=" * 78)
    print(" [1] 기본 검산")
    print("=" * 78)

    print(
        f" 비교 가능 프레임 : "
        f"{usable}"
    )

    if usable > 0:

        print(
            f" 평균 후보 수      : "
            f"{candidate_sum/usable:.2f}"
        )

    print()

    print(
        " 기존 기록 예상:"
        " 364프레임"
    )

    print()


    # -----------------------------------------------------------------

    print("=" * 78)
    print(
        " [2] 기존 방식 재현 검산"
    )
    print("=" * 78)

    print(
        " bbox-depth nearest vs "
        "bbox-depth Full"
    )

    if usable > 0:

        print(
            f" 차이 : "
            f"{old_nearest_vs_old_full}"
            f"/{usable} "
            f"("
            f"{old_nearest_vs_old_full/usable*100:.1f}%"
            f")"
        )

    print(
        " 기존 기록 예상:"
        " 103/364 = 28.3%"
    )

    print()


    # -----------------------------------------------------------------

    print("=" * 78)
    print(
        " [3] 거리 정의 변경 영향"
    )
    print("=" * 78)

    print(
        " bbox-depth nearest vs "
        "GT-mask-depth nearest"
    )

    if usable > 0:

        print(
            f" 차이 : "
            f"{old_nearest_vs_new_nearest}"
            f"/{usable} "
            f"("
            f"{old_nearest_vs_new_nearest/usable*100:.1f}%"
            f")"
        )

    print(
        " 기존 진단 예상:"
        " 108/364 = 29.7%"
    )

    print()


    # -----------------------------------------------------------------

    print("=" * 78)
    print(
        " [4] ★ 새로운 핵심 결과 ★"
    )
    print("=" * 78)

    print(
        " GT-mask-depth nearest vs "
        "GT-mask-depth Full"
    )

    if usable > 0:

        print(
            f" 차이 : "
            f"{new_nearest_vs_new_full}"
            f"/{usable} "
            f"("
            f"{new_nearest_vs_new_full/usable*100:.1f}%"
            f")"
        )

    print()

    print(
        " 이 값이 기존 28.3%와 "
        "비교할 핵심 재분석 결과입니다."
    )

    print()


    # -----------------------------------------------------------------

    print("=" * 78)
    print(
        " [5] D-only 검산"
    )
    print("=" * 78)

    print(
        " mask-depth D-only vs "
        "mask-depth nearest"
    )

    if usable > 0:

        print(
            f" 차이 : "
            f"{d_only_vs_new_nearest}"
            f"/{usable} "
            f"("
            f"{d_only_vs_new_nearest/usable*100:.1f}%"
            f")"
        )

    print(
        " 정상이라면 반드시"
        " 0/364 = 0.0%"
    )

    print()


    # -----------------------------------------------------------------

    print("=" * 78)
    print(
        " [6] 시퀀스별 결과"
    )
    print("=" * 78)

    print(
        f" {'seq':<6}"
        f"{'frames':>8}"
        f"{'old-new near':>16}"
        f"{'new near-Full':>16}"
    )

    for seq in sequences:

        d = per_seq[seq]

        n = d["usable"]

        if n == 0:
            continue

        r1 = (
            d["old_new_nearest_diff"]
            /
            n
            *
            100
        )

        r2 = (
            d["new_full_diff"]
            /
            n
            *
            100
        )

        print(
            f" {seq:<6}"
            f"{n:>8}"
            f"{r1:>15.1f}%"
            f"{r2:>15.1f}%"
        )


    # =================================================================
    # CSV 저장
    # =================================================================

    if rows:

        with open(
            OUTPUT_CSV,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    rows[0].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(rows)

        print()
        print(
            f"프레임별 결과 저장: "
            f"{OUTPUT_CSV}"
        )


    # =================================================================
    # 마지막 체크
    # =================================================================

    print()
    print("=" * 78)
    print(" 최종 체크")
    print("=" * 78)

    if usable != 364:

        print(
            f" [주의] usable={usable}"
            " → 기존 364와 다릅니다."
        )

        print(
            " 후보 필터링 / depth 유효성 "
            "차이를 먼저 확인하세요."
        )

    else:

        print(
            " [PASS] 비교 가능 프레임 "
            "364개 재현"
        )


    if d_only_vs_new_nearest == 0:

        print(
            " [PASS] D-only와 "
            "mask-depth nearest 완전 일치"
        )

    else:

        print(
            " [FAIL] D-only와 nearest가 "
            "일치하지 않습니다."
        )


    if (
        old_nearest_vs_old_full
        == 103
    ):

        print(
            " [PASS] 기존 103/364 "
            "selection 결과 재현"
        )

    else:

        print(
            " [주의] 기존 Full 결과가 "
            "103과 다릅니다."
        )


    if (
        old_nearest_vs_new_nearest
        == 108
    ):

        print(
            " [PASS] 기존 거리 진단 "
            "108/364 재현"
        )

    else:

        print(
            " [주의] 기존 거리 진단 "
            "108과 다릅니다."
        )


    if not check_frame_found:

        print(
            " [주의] 406 데모 frame을 "
            "찾지 못했습니다."
        )


    print()
    print("=" * 78)


# ====================================================================
# 실행
# ====================================================================

if __name__ == "__main__":
    main()