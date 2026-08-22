# -*- coding: utf-8 -*-
"""
====================================================================
 make_demo_figure.py - 선정 결과 RGB 시각화 (논문 데모 그림)
====================================================================

[목적]
  거리 단일 기준 baseline과 우선순위 점수 기반 선정이 각각 고른
  대상을 RGB 이미지 위에 표시하여, 두 방식의 선택 차이를
  시각적으로 보여준다.

[표시 규칙]
  회색 실선  : 검출된 전체 후보
  빨간 실선  : baseline(최근접)이 선정한 대상
  파란 실선  : 우선순위 점수가 선정한 대상  ← 최종 선정
  각 선정 대상 옆에 거리 / 가시성 / 점수 표기

[준비물]
  1) RGB 이미지 (민혁님으로부터 확보)
  2) dataset_bulk/annotations, dataset_bulk/depth
  3) eval_detections_fixed.csv

[사용법]
  아래 SEQ, TIMESTAMP, RGB_PATH 를 채운 뒤 실행.

    SEQ       = "406"
    TIMESTAMP = "1600938551617024"
    RGB_PATH  = "1600938551617024.tiff"

  python make_demo_figure.py

[출력]
  demo_<시퀀스>_<타임스탬프>.png  (300 dpi)
====================================================================
"""

import os
import glob
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm

from data_loader import load_depth, clean_depth, load_bupst20_annotation
from priority import select_top1, WEIGHTS
from load_detections import load_detections_csv, attach_mask_from_pkl


# ====================================================================
# ★ 설정 — 여기만 채우면 됩니다 ★
# ====================================================================
SEQ       = "406"
TIMESTAMP = "1600938551617024"
RGB_PATH  = "1600938551617024.tiff"

LANG      = "ko"        # "ko" 한글 라벨 / "en" 영문 라벨
SHOW_ALL_CANDIDATES = True   # 나머지 후보도 회색으로 표시
DPI       = 300

# 논문 조판용 자르기 — 선정 대상 주변만 남겨 세로 길이를 줄인다.
# None이면 원본 전체. AUTO_CROP=True면 A/B 박스를 기준으로 자동 계산.
AUTO_CROP   = True
CROP_MARGIN = 230     # 박스 바깥으로 남길 여백(픽셀)

# 공통 설정
DATA_DIR       = "dataset_bulk"
DETECTION_CSV  = "eval_detections_fixed.csv"
MODE           = "visibility"
IMG_SIZE       = (720, 1280)
IOU_THRESHOLD  = 0.5
CONF_THRESHOLD = 0.3

# 색상
C_OTHER = "#9aa5ad"
C_NEAR  = "#c0392b"
C_FULL  = "#1f6f8b"


# 한글 폰트
for cand in ["Malgun Gothic", "NanumGothic", "AppleGothic",
             "Noto Sans CJK KR", "Noto Sans CJK JP"]:
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

LABEL = {
    "ko": {"near": "거리 기준 baseline", "full": "제안 방법 (Top-1)",
           "other": "검출 후보", "dist": "거리", "vis": "가시성", "score": "점수"},
    "en": {"near": "Distance-only baseline", "full": "Proposed (Top-1)",
           "other": "Candidates", "dist": "d", "vis": "v", "score": "S"},
}[LANG]


def load_rgb(path):
    """RGB 이미지 로드 (.tiff/.png/.jpg 지원)"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"RGB 이미지를 찾을 수 없습니다: {path}\n"
            f"RGB_PATH 설정을 확인하세요."
        )
    try:
        import tifffile
        if path.lower().endswith((".tif", ".tiff")):
            img = tifffile.imread(path)
        else:
            raise ImportError
    except Exception:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
        img = img[:, :, ::-1]          # BGR → RGB
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.dtype != np.uint8:          # 16bit 등은 8bit로 정규화
        v = img.astype(np.float32)
        v = (v - v.min()) / max(1e-6, v.max() - v.min())
        img = (v * 255).astype(np.uint8)
    return img[:, :, :3]


def draw_box(ax, bbox, color, lw, tag=None, dashed=False):
    x1, y1, x2, y2 = bbox
    ax.add_patch(patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=lw, edgecolor=color, facecolor="none",
        linestyle="--" if dashed else "-", zorder=3))
    if tag:
        ax.text(x2 + 5, (y1 + y2) / 2, tag, fontsize=11, color="white",
                va="center", ha="left", zorder=4, fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.28", facecolor=color,
                          edgecolor="white", linewidth=1.0))


def main():
    # ---- 데이터 로드 ----
    ann_path = os.path.join(DATA_DIR, "annotations", SEQ, TIMESTAMP + ".pkl")
    depth_cands = glob.glob(
        os.path.join(DATA_DIR, "depth", SEQ, TIMESTAMP + ".tif*"))
    if not os.path.exists(ann_path) or not depth_cands:
        raise FileNotFoundError(
            f"해당 프레임의 annotation/depth를 찾을 수 없습니다.\n"
            f"  annotation: {ann_path}\n"
            f"  depth     : {os.path.join(DATA_DIR, 'depth', SEQ, TIMESTAMP + '.tiff')}"
        )

    det = load_detections_csv(DETECTION_CSV)
    info = det.get((SEQ, TIMESTAMP))
    if info is None:
        raise ValueError(f"검출 CSV에 해당 프레임이 없습니다: {SEQ}/{TIMESTAMP}")

    depth = clean_depth(load_depth(depth_cands[0]))
    gts = load_bupst20_annotation(ann_path)

    preds = [dict(o) for o in info["objects"]
             if o["confidence"] >= CONF_THRESHOLD]
    attach_mask_from_pkl(preds, gts, IOU_THRESHOLD)
    preds = [p for p in preds if p.get("gt_matched", False)]

    top1, scored = select_top1(preds, depth, WEIGHTS[MODE],
                              info["img_size"] or IMG_SIZE, MODE)
    if top1 is None or len(scored) < 2:
        raise ValueError("후보가 부족하여 시각화할 수 없습니다.")

    nearest = min(scored, key=lambda o: o["depth_value"])
    same = tuple(nearest["bbox"]) == tuple(top1["bbox"])

    # ---- 그림 ----
    img = load_rgb(RGB_PATH)
    h, w = img.shape[:2]
    if AUTO_CROP:
        bs = [nearest["bbox"], top1["bbox"]]
        _x1 = max(0, min(b[0] for b in bs) - CROP_MARGIN)
        _x2 = min(w, max(b[2] for b in bs) + CROP_MARGIN)
        _y1 = max(0, min(b[1] for b in bs) - CROP_MARGIN)
        _y2 = min(h, max(b[3] for b in bs) + CROP_MARGIN)
        fw, fh = (_x2 - _x1) / 120, (_y2 - _y1) / 120
    else:
        fw, fh = w / 120, h / 120
    fig, ax = plt.subplots(figsize=(fw, fh))
    ax.imshow(img)
    # ---- 자르기 범위 결정 ----
    if AUTO_CROP:
        bs = [nearest["bbox"], top1["bbox"]]
        x1 = min(b[0] for b in bs) - CROP_MARGIN
        x2 = max(b[2] for b in bs) + CROP_MARGIN
        y1 = min(b[1] for b in bs) - CROP_MARGIN
        y2 = max(b[3] for b in bs) + CROP_MARGIN
        cx1, cx2 = max(0, x1), min(w, x2)
        cy1, cy2 = max(0, y1), min(h, y2)
    else:
        cx1, cy1, cx2, cy2 = 0, 0, w, h
    ax.set_xlim(cx1, cx2); ax.set_ylim(cy2, cy1); ax.axis("off")

    # 나머지 후보
    if SHOW_ALL_CANDIDATES:
        for o in scored:
            if o is nearest or o is top1:
                continue
            draw_box(ax, o["bbox"], C_OTHER, 1.6, dashed=True)

    dn, dt = nearest["_detail"], top1["_detail"]

    if same:
        draw_box(ax, top1["bbox"], C_FULL, 3.0, "B")
    else:
        draw_box(ax, nearest["bbox"], C_NEAR, 2.8, "A")
        draw_box(ax, top1["bbox"], C_FULL, 3.0, "B")

    # ---- 정보 패널 (좌상단) ----
    if LANG == "ko":
        lines = [
            f"A  {LABEL['near']}   {LABEL['vis']} {dn['visibility']:.3f}",
            f"B  {LABEL['full']}   {LABEL['vis']} {dt['visibility']:.3f}",
        ]
    else:
        lines = [
            f"A  {LABEL['near']}   visibility {dn['visibility']:.3f}",
            f"B  {LABEL['full']}   visibility {dt['visibility']:.3f}",
        ]
    ax.text(cx1 + 12, cy1 + 12, "\n".join(lines), fontsize=8.5, color="#1b2631",
            va="top", ha="left", zorder=5, linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#8a9299", alpha=0.93, linewidth=0.9))

    fig.tight_layout(pad=0.1)
    out = f"demo_{SEQ}_{TIMESTAMP}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")

    # ---- 콘솔 요약 ----
    print("=" * 66)
    print(f" 시퀀스 {SEQ} / {TIMESTAMP}")
    print("=" * 66)
    print(f"  후보 수 : {len(scored)}개")
    print()
    print(f"  {'':14}{'거리':>10}{'가시성':>10}{'신뢰도':>10}{'점수':>10}")
    print(f"  {'baseline':14}{nearest['depth_value']:>8.0f}mm"
          f"{dn['visibility']:>10.3f}{dn['confidence']:>10.3f}"
          f"{nearest['score']:>10.3f}")
    print(f"  {'제안 방법':14}{top1['depth_value']:>8.0f}mm"
          f"{dt['visibility']:>10.3f}{dt['confidence']:>10.3f}"
          f"{top1['score']:>10.3f}")
    print()
    if same:
        print("  ※ 두 방식이 동일한 대상을 선정한 프레임입니다.")
        print("     데모용으로는 선정이 갈린 프레임을 쓰세요.")
        print("     (pick_demo_frames.py 결과 참고)")
    else:
        print(f"  차이: 거리 {top1['depth_value'] - nearest['depth_value']:+.0f}mm, "
              f"가시성 {dt['visibility'] - dn['visibility']:+.3f}")
    print()
    print(f"  저장: {out}")


if __name__ == "__main__":
    main()