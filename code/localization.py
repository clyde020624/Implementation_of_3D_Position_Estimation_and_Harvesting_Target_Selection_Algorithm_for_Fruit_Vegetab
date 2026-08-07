# -*- coding: utf-8 -*-
"""
====================================================================
 localization.py - 3D 위치 추정 정확도 평가 (연구 3단계)
====================================================================

[목적]
  같은 프레임 안에서
     예측 3D 좌표 (YOLO bbox 기반)
     정답 3D 좌표 (GT instance mask 기반)
  를 비교하여 X / Y / Z 및 3D 거리 오차를 측정한다.

  * 같은 프레임끼리 비교하므로 로봇 이동의 영향이 없다.
    (멘토 피드백 반영: 시간적 일관성이 아닌 모델 위치 추정 정확도)

[핵심 함수]
  depth_from_bbox_center() : 박스 중심 영역의 median depth  (예측 방식)
  depth_from_mask()        : 마스크 영역의 median depth      (정답/reference)
  match_by_iou()           : 예측 ↔ GT 짝짓기
  evaluate_localization()  : 오차 계산
  summarize_errors()       : 논문용 통계 (평균/중앙값/RMSE)

[부가 활용]
  예측 쪽 depth 추출 방식을 바꿔가며 호출하면
  '박스 중심 vs 마스크 기반' 추정 방식 비교 실험도 가능하다.
====================================================================
"""

import numpy as np
from data_loader import pixel_to_3d


# ====================================================================
# 1) depth 추출 방식 두 가지
# ====================================================================
def depth_from_bbox_center(bbox, depth_map, region_half=2):
    """박스 중심 주변 (2*region_half+1)^2 영역의 median depth.
       YOLO 검출은 마스크가 없으므로 이 방식을 쓴다."""
    x1, y1, x2, y2 = bbox
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    h, w = depth_map.shape
    ys, ye = max(0, cy - region_half), min(h, cy + region_half + 1)
    xs, xe = max(0, cx - region_half), min(w, cx + region_half + 1)
    region = depth_map[ys:ye, xs:xe]
    valid = region[region > 0]
    if len(valid) == 0:
        return None, (cx, cy)
    return float(np.median(valid)), (cx, cy)


def depth_from_mask(mask, depth_map):
    """instance mask 영역 전체의 median depth.
       파프리카 픽셀만 사용하므로 배경/잎 오염이 적어 reference로 적합."""
    if mask is None:
        return None, None
    mask = np.asarray(mask).astype(bool)     # ← 추가
    if mask.shape != depth_map.shape:
        return None, None
    vals = depth_map[mask & (depth_map > 0)]
    if vals.size == 0:
        return None, None
    # 마스크의 무게중심을 대표 픽셀 좌표로 사용
    ys, xs = np.nonzero(mask)
    cx, cy = float(xs.mean()), float(ys.mean())
    return float(np.median(vals)), (cx, cy)


# ====================================================================
# 2) 예측 ↔ GT 매칭 (IoU 기준)
# ====================================================================
def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def match_by_iou(preds, gts, iou_threshold=0.5):
    """
    예측(preds)과 정답(gts)을 IoU 기준으로 1:1 매칭.
    IoU가 높은 쌍부터 그리디하게 짝지어 중복 매칭을 방지한다.
    반환: (매칭쌍 리스트, 매칭 안 된 예측 수, 매칭 안 된 GT 수)
    """
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            v = iou(p["bbox"], g["bbox"])
            if v >= iou_threshold:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)          # IoU 높은 순

    used_p, used_g, matched = set(), set(), []
    for v, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        matched.append({"pred": preds[i], "gt": gts[j], "iou": v})

    return matched, len(preds) - len(used_p), len(gts) - len(used_g)


# ====================================================================
# 3) localization error 계산
# ====================================================================
def evaluate_localization(frames, cam, iou_threshold=0.5):
    """
    frames: [{ "image_id", "preds":[...], "gts":[...], "depth_map" }, ...]
      preds: YOLO 검출 (bbox, confidence)
      gts  : GT (bbox, instance_mask)
    반환: 매칭된 객체별 오차 리스트
    """
    results = []
    total_unmatched_pred = 0
    total_unmatched_gt = 0

    for f in frames:
        depth_map = f["depth_map"]
        matched, um_p, um_g = match_by_iou(f["preds"], f["gts"], iou_threshold)
        total_unmatched_pred += um_p
        total_unmatched_gt += um_g

        for m in matched:
            # --- 예측 3D: 박스 중심 기반 ---
            zp, cp = depth_from_bbox_center(m["pred"]["bbox"], depth_map)
            if zp is None:
                continue
            Xp, Yp, Zp = pixel_to_3d(cp[0], cp[1], zp, cam)

            # --- 정답 3D: GT mask 기반 (없으면 GT 박스 중심) ---
            zg, cg = depth_from_mask(m["gt"].get("instance_mask"), depth_map)
            if zg is None:
                zg, cg = depth_from_bbox_center(m["gt"]["bbox"], depth_map)
            if zg is None:
                continue
            Xg, Yg, Zg = pixel_to_3d(cg[0], cg[1], zg, cam)

            ex, ey, ez = Xp - Xg, Yp - Yg, Zp - Zg
            e3d = float(np.sqrt(ex**2 + ey**2 + ez**2))

            results.append({
                "image_id": f["image_id"],
                "gt_id": m["gt"].get("id"),
                "iou": round(m["iou"], 3),
                "conf": m["pred"].get("confidence"),
                "err_X": round(ex, 1), "err_Y": round(ey, 1),
                "err_Z": round(ez, 1), "err_3D": round(e3d, 1),
                "pred_Z": round(Zp, 1), "gt_Z": round(Zg, 1),
            })

    return results, total_unmatched_pred, total_unmatched_gt


# ====================================================================
# 4) 논문용 통계
# ====================================================================
def summarize_errors(results):
    if not results:
        return None

    def stats(key):
        v = np.array([r[key] for r in results], dtype=float)
        return {
            "평균": round(float(np.mean(v)), 1),
            "절대평균": round(float(np.mean(np.abs(v))), 1),
            "중앙값": round(float(np.median(np.abs(v))), 1),
            "표준편차": round(float(np.std(v)), 1),
            "RMSE": round(float(np.sqrt(np.mean(v ** 2))), 1),
        }

    e3d = np.array([r["err_3D"] for r in results], dtype=float)
    return {
        "매칭 객체 수": len(results),
        "X": stats("err_X"), "Y": stats("err_Y"), "Z": stats("err_Z"),
        "3D": {
            "평균": round(float(np.mean(e3d)), 1),
            "중앙값": round(float(np.median(e3d)), 1),
            "RMSE": round(float(np.sqrt(np.mean(e3d ** 2))), 1),
            "최대": round(float(np.max(e3d)), 1),
        },
        "10mm이내(%)": round(float(np.mean(e3d < 10) * 100), 1),
        "20mm이내(%)": round(float(np.mean(e3d < 20) * 100), 1),
        "50mm이내(%)": round(float(np.mean(e3d < 50) * 100), 1),
    }


def print_summary(s, unmatched_pred=0, unmatched_gt=0):
    if s is None:
        print(" 매칭된 객체가 없습니다.")
        return
    print("=" * 66)
    print(" 3D 위치 추정 정확도 (localization error)")
    print("=" * 66)
    print(f"  매칭 객체 수 : {s['매칭 객체 수']}개")
    print(f"  미매칭       : 예측 {unmatched_pred}개 / GT {unmatched_gt}개")
    print()
    print(f"  {'축':<4}{'절대평균':>10}{'중앙값':>10}{'RMSE':>10}")
    for ax in ("X", "Y", "Z"):
        st = s[ax]
        print(f"  {ax:<4}{st['절대평균']:>9}mm{st['중앙값']:>9}mm{st['RMSE']:>9}mm")
    t = s["3D"]
    print(f"  {'3D':<4}{t['평균']:>9}mm{t['중앙값']:>9}mm{t['RMSE']:>9}mm")
    print(f"\n  3D 오차 최대: {t['최대']}mm")
    print(f"  오차 분포: 10mm 이내 {s['10mm이내(%)']}% | "
          f"20mm 이내 {s['20mm이내(%)']}% | 50mm 이내 {s['50mm이내(%)']}%")


# ====================================================================
# 5) 더미 데이터로 동작 확인
#    (실제로는 preds = YOLO CSV, gts = BUP-ST20 pkl)
# ====================================================================
if __name__ == "__main__":
    np.random.seed(2)
    cam = {"fx": 919.46, "fy": 920.65, "cx": 361.72, "cy": 636.79}

    frames = []
    for fi in range(3):
        depth = np.random.randint(500, 2000, (1280, 720)).astype(np.uint16)
        gts, preds = [], []
        for oi in range(5):
            x, y, w, h = 80 + oi * 120, 200 + (oi % 3) * 300, 60, 70
            true_z = 700 + oi * 80
            depth[y:y+h, x:x+w] = true_z

            mask = np.zeros((1280, 720), dtype=bool)
            mask[y:y+h, x:x+w] = True
            gts.append({"id": 1000 + oi, "bbox": [x, y, x+w, y+h],
                        "instance_mask": mask})

            # 예측은 GT에서 살짝 어긋나게 (검출 오차 모사)
            dx, dy = np.random.randint(-8, 9), np.random.randint(-8, 9)
            preds.append({"bbox": [x+dx, y+dy, x+w+dx, y+h+dy],
                          "confidence": round(np.random.uniform(0.4, 0.95), 2)})

        frames.append({"image_id": f"f{fi}", "preds": preds, "gts": gts,
                       "depth_map": depth})

    results, um_p, um_g = evaluate_localization(frames, cam, iou_threshold=0.5)

    print("=" * 66)
    print(" 객체별 오차 (앞 8개)")
    print("=" * 66)
    for r in results[:8]:
        print(f"  {r['image_id']} id={r['gt_id']} IoU={r['iou']} "
              f"| X {r['err_X']:>6}  Y {r['err_Y']:>6}  Z {r['err_Z']:>6}  "
              f"→ 3D {r['err_3D']}mm")
    print()
    print_summary(summarize_errors(results), um_p, um_g)
