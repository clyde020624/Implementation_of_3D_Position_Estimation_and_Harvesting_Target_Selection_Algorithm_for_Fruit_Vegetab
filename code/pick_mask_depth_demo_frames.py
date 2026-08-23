"""Pick 10 representative mask-depth trade-off frames and make previews.

RGB files are resolved from the detection CSV or --rgb-root (also available as
BUPST20_IMAGE_ROOT). Missing RGBs produce explicit placeholders, not fake RGB.
"""
import argparse, csv, glob, os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data_loader import clean_depth, load_bupst20_annotation, load_depth
from load_detections import load_detections_csv
from mask_depth_reanalysis import FULL_WEIGHTS, MODE, prepare_candidates, select_top1_precomputed
from priority import score_center, score_visibility

DATA_DIR = "dataset_bulk"
DETECTION_CSV = "eval_detections_fixed.csv"
TRADEOFF_CSV = "mask_depth_tradeoff_results.csv"
OUTPUT_CSV = "mask_depth_demo_candidates.csv"
OUTPUT_DIR = Path("demo_candidates")
CONF_THRESHOLD = 0.30
MAX_PER_SEQUENCE = 2
MEDIAN_DEPTH, MEDIAN_VIS = 11.5, 0.1834
DEPTH_Q1, DEPTH_Q3 = 5.25, 29.25
VIS_Q1, VIS_Q3 = 0.0801, 0.2931
EXCLUDED_FRAME = ("406", "1600938551617024")
EXTENSIONS = (".tiff", ".tif", ".png", ".jpg", ".jpeg")
NEAR_COLOR, FULL_COLOR = (220, 50, 47), (31, 119, 180)


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--rgb-root", default=os.environ.get("BUPST20_IMAGE_ROOT"))
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args()


def rank_tradeoff():
    with open(TRADEOFF_CSV, newline="", encoding="utf-8-sig") as f:
        raw = list(csv.DictReader(f))
    if len(raw) != 118:
        raise RuntimeError(f"Expected 118 trade-off rows, found {len(raw)}")
    ranked = []
    for r in raw:
        key = (r["sequence_id"], r["frame_id"])
        dd, dv, size = float(r["delta_depth_mm"]), float(r["delta_visibility"]), int(r["nearest_set_size"])
        if key == EXCLUDED_FRAME or dd <= 0 or dv <= 0:
            continue
        score = abs(dd - MEDIAN_DEPTH) / (DEPTH_Q3 - DEPTH_Q1) + abs(dv - MEDIAN_VIS) / (VIS_Q3 - VIS_Q1)
        in_iqr = DEPTH_Q1 <= dd <= DEPTH_Q3 and VIS_Q1 <= dv <= VIS_Q3
        tier = 0 if size == 1 and in_iqr else 1 if size == 1 else 2 if in_iqr else 3
        ranked.append(dict(sequence_id=key[0], frame_id=key[1], delta_depth_mm=dd,
                           delta_visibility=dv, nearest_set_size=size,
                           representativeness_score=score, selection_tier=tier))
    return sorted(ranked, key=lambda r: (r["selection_tier"], r["representativeness_score"], r["sequence_id"], r["frame_id"]))


def balanced_top(ranked, count):
    selected, per_seq = [], Counter()
    for r in ranked:
        if per_seq[r["sequence_id"]] >= MAX_PER_SEQUENCE:
            continue
        selected.append(r); per_seq[r["sequence_id"]] += 1
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)} balanced candidates available")
    for i, r in enumerate(selected, 1):
        r["rank"] = i
    return selected


def metrics(candidate, image_size):
    center = (image_size[0] / 2, image_size[1] / 2)
    return dict(visibility=float(score_visibility(candidate)),
                confidence=float(candidate["confidence"]),
                center=float(score_center(candidate["bbox"], center)))


def reconstruct(selected):
    detections = load_detections_csv(DETECTION_CSV)
    for row in selected:
        seq, frame = row["sequence_id"], row["frame_id"]
        info = detections.get((seq, frame))
        depths = glob.glob(os.path.join(DATA_DIR, "depth", seq, frame + ".tif*"))
        ann = os.path.join(DATA_DIR, "annotations", seq, frame + ".pkl")
        if info is None or not depths or not os.path.exists(ann):
            raise RuntimeError(f"Missing source data: {seq}/{frame}")
        preds = [dict(o) for o in info["objects"] if o["confidence"] >= CONF_THRESHOLD]
        depth = clean_depth(load_depth(depths[0]))
        candidates = prepare_candidates(preds, load_bupst20_annotation(ann), depth)
        image_size = info["img_size"] or (720, 1280)
        full, _ = select_top1_precomputed(candidates, "mask_depth", FULL_WEIGHTS, image_size, MODE)
        min_depth = min(c["mask_depth"] for c in candidates)
        nearest_set = [c for c in candidates if c["mask_depth"] == min_depth]
        nearest_ids = {c["id"] for c in nearest_set}
        if full is None or full["id"] in nearest_ids:
            raise RuntimeError(f"Not a disagreement: {seq}/{frame}")
        nearest = max(nearest_set, key=lambda c: metrics(c, image_size)["visibility"])
        nm, fm = metrics(nearest, image_size), metrics(full, image_size)
        dd = float(full["mask_depth"] - min_depth)
        dv = fm["visibility"] - max(metrics(c, image_size)["visibility"] for c in nearest_set)
        if abs(dd - row["delta_depth_mm"]) > 1e-9 or abs(dv - row["delta_visibility"]) > 1e-9:
            raise RuntimeError(f"CSV reconstruction mismatch: {seq}/{frame}")
        if nm["confidence"] < CONF_THRESHOLD or fm["confidence"] < CONF_THRESHOLD:
            raise RuntimeError(f"Confidence check failed: {seq}/{frame}")
        row.update(nearest_prediction_id=nearest["id"], full_prediction_id=full["id"],
                   nearest_mask_depth=float(min_depth), full_mask_depth=float(full["mask_depth"]),
                   nearest_visibility=nm["visibility"], full_visibility=fm["visibility"],
                   nearest_confidence=nm["confidence"], full_confidence=fm["confidence"],
                   nearest_center_score=nm["center"], full_center_score=fm["center"],
                   nearest_set_size=len(nearest_set),
                   nearest_set_ids=";".join(str(i) for i in sorted(nearest_ids)),
                   nearest_bbox=list(nearest["bbox"]), full_bbox=list(full["bbox"]),
                   csv_image_path=info.get("image_path") or "")
    return selected


def rgb_path(row, root):
    paths = [Path(row["csv_image_path"])] if row["csv_image_path"] else []
    roots = ([Path(root)] if root else []) + [Path(DATA_DIR) / "images", Path(DATA_DIR) / "rgb", Path(".")]
    for base in roots:
        for ext in EXTENSIONS:
            paths += [base / row["sequence_id"] / (row["frame_id"] + ext),
                      base / "images" / row["sequence_id"] / (row["frame_id"] + ext),
                      base / "rgb" / row["sequence_id"] / (row["frame_id"] + ext),
                      base / (row["frame_id"] + ext)]
    seen = set()
    for p in paths:
        key = str(p).lower()
        if key not in seen and p.is_file():
            return p
        seen.add(key)
    return None


def load_rgb(path):
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        a = np.asarray(tifffile.imread(str(path)))
    else:
        a = np.asarray(Image.open(path).convert("RGB"))
    if a.ndim == 2: a = np.stack([a] * 3, axis=-1)
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4): a = np.moveaxis(a, 0, -1)
    a = a[:, :, :3]
    if a.dtype != np.uint8:
        v = a.astype(np.float32); lo, hi = np.percentile(v, [1, 99])
        a = (np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(a, "RGB")


def get_font(size):
    for p in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"):
        if os.path.exists(p): return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_candidate(draw, row, prefix, title, bbox, color, fnt):
    x1, y1, x2, y2 = map(lambda v: int(round(v)), bbox)
    draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
    label = (f"{title}\nDepth {row[prefix + '_mask_depth']:.0f} mm\n"
             f"Visibility {row[prefix + '_visibility']:.3f}\nConfidence {row[prefix + '_confidence']:.3f}")
    b = draw.multiline_textbbox((0, 0), label, font=fnt, spacing=3)
    w, h = b[2] - b[0] + 12, b[3] - b[1] + 8
    tx, ty = max(0, x1), max(0, y1 - h)
    draw.rectangle((tx, ty, tx + w, ty + h), fill=color)
    draw.multiline_text((tx + 6, ty + 4), label, fill="white", font=fnt, spacing=3)


def placeholder(row):
    im = Image.new("RGB", (900, 620), (238, 241, 244)); d = ImageDraw.Draw(im)
    d.text((35, 30), "RGB IMAGE NOT FOUND", fill=(150, 35, 35), font=get_font(28))
    text = (f"Rank {row['rank']} | {row['sequence_id']}/{row['frame_id']}\n\n"
            f"Nearest ID {row['nearest_prediction_id']}: Depth {row['nearest_mask_depth']:.0f}, Vis {row['nearest_visibility']:.3f}, Conf {row['nearest_confidence']:.3f}\n"
            f"Full ID {row['full_prediction_id']}: Depth {row['full_mask_depth']:.0f}, Vis {row['full_visibility']:.3f}, Conf {row['full_confidence']:.3f}\n\n"
            f"Rerun with --rgb-root or BUPST20_IMAGE_ROOT.\nCSV path: {row['csv_image_path']}")
    d.multiline_text((35, 95), text, fill=(35, 45, 55), font=get_font(20), spacing=10)
    return im


def preview(row, root):
    source = rgb_path(row, root); im = load_rgb(source) if source else placeholder(row)
    if source:
        d = ImageDraw.Draw(im); fnt = get_font(max(16, im.width // 38))
        draw_candidate(d, row, "nearest", "Nearest", row["nearest_bbox"], NEAR_COLOR, fnt)
        draw_candidate(d, row, "full", "Full", row["full_bbox"], FULL_COLOR, fnt)
        header = f"Rank {row['rank']} | {row['sequence_id']}/{row['frame_id']} | score {row['representativeness_score']:.3f}"
        hf = get_font(max(18, im.width // 32)); h = d.textbbox((0, 0), header, font=hf)[3] + 12
        d.rectangle((0, 0, im.width, h), fill="black"); d.text((8, 5), header, fill="white", font=hf)
    if max(im.size) > 1400: im.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
    out = OUTPUT_DIR / f"rank_{row['rank']:02d}_seq_{row['sequence_id']}_frame_{row['frame_id']}.png"
    im.save(out); row["rgb_image_path"] = str(source) if source else ""; row["preview_path"] = str(out)
    return out, source is not None


def contact_sheet(selected, paths):
    cw, ch, cols = 360, 500, 5
    sheet = Image.new("RGB", (cw * cols, ch * 2), "white"); d = ImageDraw.Draw(sheet)
    for n, (row, path) in enumerate(zip(selected, paths)):
        im = Image.open(path).convert("RGB"); im.thumbnail((cw - 16, ch - 48), Image.Resampling.LANCZOS)
        col, rr = n % cols, n // cols; x = col * cw + (cw - im.width) // 2; y = rr * ch + 36
        sheet.paste(im, (x, y)); d.text((col * cw + 8, rr * ch + 8), f"#{row['rank']} {row['sequence_id']}/{row['frame_id']}", fill=(20, 25, 30), font=get_font(18))
    out = OUTPUT_DIR / "contact_sheet.png"; sheet.save(out); return out


def write_results(selected):
    fields = ["rank", "sequence_id", "frame_id", "nearest_prediction_id", "full_prediction_id",
              "nearest_mask_depth", "full_mask_depth", "delta_depth_mm", "nearest_visibility",
              "full_visibility", "delta_visibility", "nearest_confidence", "full_confidence",
              "nearest_center_score", "full_center_score", "nearest_set_size",
              "representativeness_score", "nearest_set_ids", "selection_tier",
              "rgb_image_path", "preview_path"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(selected)


def main():
    a = args(); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = reconstruct(balanced_top(rank_tradeoff(), a.top_k))
    paths, actual = [], 0
    for row in selected:
        path, ok = preview(row, a.rgb_root); paths.append(path); actual += int(ok)
    sheet = contact_sheet(selected, paths); write_results(selected)
    print(
        "rank seq frame nearest full nearZ fullZ dDepth nearVis fullVis "
        "dVis nearConf fullConf nearCenter fullCenter setSize score"
    )
    for r in selected:
        print(
            r["rank"], r["sequence_id"], r["frame_id"],
            r["nearest_prediction_id"], r["full_prediction_id"],
            f"{r['nearest_mask_depth']:.1f}", f"{r['full_mask_depth']:.1f}",
            f"{r['delta_depth_mm']:.1f}", f"{r['nearest_visibility']:.4f}",
            f"{r['full_visibility']:.4f}", f"{r['delta_visibility']:.4f}",
            f"{r['nearest_confidence']:.4f}", f"{r['full_confidence']:.4f}",
            f"{r['nearest_center_score']:.4f}", f"{r['full_center_score']:.4f}",
            r["nearest_set_size"], f"{r['representativeness_score']:.4f}",
        )
    print(f"CSV: {OUTPUT_CSV}\nContact sheet: {sheet}\nActual RGB previews: {actual}/{len(selected)}")
    if actual < len(selected): print("Missing RGB files produced placeholders; rerun with --rgb-root.")


if __name__ == "__main__":
    main()
