"""Read-only V1 localization audit; all artifacts go to a new diagnostic folder.

Run: py -3 -X utf8 -B audit_localization_outliers.py
No core/source/data/previous-result writes. No V3 selection logic.
Paper statistics and ranking retain localization.py's per-object 0.1-mm rounding.
"""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageOps

import main as loc_config
import tradeoff as v1_config
from data_loader import (clean_depth, load_bupst20_annotation, load_cam_params,
                         load_depth, pixel_to_3d)
from export_v1_ppt_stage2 import font, rgb_path
from load_detections import attach_mask_from_pkl, load_detections_csv
from localization import (depth_from_bbox_center, depth_from_mask,
                          evaluate_localization, match_by_iou, summarize_errors)
from priority import WEIGHTS, select_top1

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/localization_outlier_audit"
EXPECTED = {"objects": 2959, "median": 8.7, "RMSE": 32.0, "max": 296.8}
# Diagnostic tags ONLY: never applied to evaluation or candidate inclusion.
FLAG_RULES = {
    "CENTER_OUTSIDE_GT_MASK": "integer predicted bbox center outside matched GT mask",
    "LOW_5X5_MASK_COVERAGE": "GT-mask pixels / actual clipped window pixels < 0.50",
    "LARGE_DEPTH_DIFFERENCE": "abs(pred_Z - gt_Z) > 50 mm",
    "LARGE_CENTER_OFFSET": "bbox center to all-mask-pixel centroid distance > 20 px",
    "LOW_IOU": "matched IoU < 0.60 (diagnostic only; matching threshold stays 0.50)",
    "NEAR_IMAGE_EDGE": "predicted bbox minimum edge margin < 8 px",
    "LOW_CONFIDENCE": "confidence < 0.30 (localization threshold stays 0.05)",
}
PRED = (225, 61, 57)
GT = (32, 170, 210)
WINDOW = (255, 215, 45)
BG = (246, 248, 251)
INK = (25, 34, 48)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def protect_existing():
    paths = [p for p in ROOT.iterdir() if p.is_file()]
    paths += list((ROOT / "tools").glob("*.py"))
    paths += [p for p in (ROOT / "results").rglob("*") if p.is_file()]
    return {p: digest(p) for p in paths}


def check_protected(protected):
    for path, original in protected.items():
        if not path.is_file() or digest(path) != original:
            raise RuntimeError(f"Protected input changed: {path}")


def csv_read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, default=native)
                             if isinstance(v, (list, dict, tuple)) else v for k, v in row.items()})


def specs_for(det):
    specs = {}
    for seq in map(str, range(400, 410)):
        depths = {p.stem: p for p in (ROOT / loc_config.DATA_DIR / "depth" / seq).glob("*.tif*")}
        anns = {p.stem: p for p in (ROOT / loc_config.DATA_DIR / "annotations" / seq).glob("*.pkl")}
        for fid in sorted(depths.keys() & anns.keys()):
            if (seq, fid) in det:
                specs[(seq, fid)] = (depths[fid], anns[fid])
    return specs


def frame_data(paths):
    dp, ap = paths
    return clean_depth(load_depth(str(dp))), load_bupst20_annotation(str(ap))


def reproduce(det, specs):
    rows, n_gt, n_below = [], 0, 0
    for (seq, fid), paths in specs.items():
        depth, gts = frame_data(paths)
        raw = det[(seq, fid)]["objects"]
        preds = [dict(p) for p in raw if p["confidence"] >= .05]
        n_below += len(raw) - len(preds)
        result, _, _ = evaluate_localization(
            [dict(image_id=fid, preds=preds, gts=gts, depth_map=depth)], CAM, .5)
        rows.extend(dict(sequence_id=seq, **r) for r in result)
        n_gt += len(gts)
    summary = summarize_errors(rows)
    if summary is None:
        raise RuntimeError("Reproduction returned no error-evaluated matches")
    s = summary["3D"]
    actual = dict(objects=len(rows), median=s["중앙값"], RMSE=s["RMSE"], max=s["최대"])
    return rows, actual, dict(input_frames=len(specs), gt_objects=n_gt,
                              detections_below_005=n_below, core_summary=summary)


def quantiles(values, prefix):
    q = np.percentile(values, [0, 25, 50, 75, 100])
    return {prefix + "_" + name: float(v)
            for name, v in zip(("min", "q1", "median", "q3", "max"), q)}


def trace_all(det, specs, core_rows):
    core = {(r["sequence_id"], r["image_id"], r["gt_id"]): r for r in core_rows}
    assert len(core) == len(core_rows)
    rows, fallback = [], 0
    for (seq, fid), paths in specs.items():
        depth, gts = frame_data(paths)
        preds = [dict(p) for p in det[(seq, fid)]["objects"] if p["confidence"] >= .05]
        matches, _, _ = match_by_iou(preds, gts, .5)
        h, w = depth.shape
        for m in matches:
            p, g = m["pred"], m["gt"]
            key = seq, fid, g["id"]
            if key not in core:
                continue
            result = core[key]
            zp, cp = depth_from_bbox_center(p["bbox"], depth)
            zg, cg = depth_from_mask(g.get("instance_mask"), depth)
            if zg is None:
                fallback += 1
                raise RuntimeError(f"Mask reference fallback in evaluated object: {key}")
            mask = np.asarray(g["instance_mask"], dtype=bool)
            u, v = cp
            xs, xe, ys, ye = max(0, u-2), min(w, u+3), max(0, v-2), min(h, v+3)
            window = depth[ys:ye, xs:xe]
            vals = window[window > 0]
            mvals = depth[mask & (depth > 0)]
            in_window = mask[ys:ye, xs:xe]
            inside = bool(0 <= u < w and 0 <= v < h and mask[v, u])
            xp = np.array(pixel_to_3d(*cp, zp, CAM))
            xg = np.array(pixel_to_3d(*cg, zg, CAM))
            error = xp - xg
            e3d = float(np.linalg.norm(error))
            for k, value in zip(("err_X", "err_Y", "err_Z", "err_3D", "pred_Z", "gt_Z"),
                                (*error, e3d, zp, zg)):
                if round(float(value), 1) != result[k]:
                    raise RuntimeError(f"Trace/core mismatch {key} {k}: {value}, {result[k]}")
            assert round(m["iou"], 3) == result["iou"] and p["confidence"] == result["conf"]
            edge = float(min(p["bbox"][0], p["bbox"][1], w-p["bbox"][2], h-p["bbox"][3]))
            offset = float(np.linalg.norm(np.asarray(cp)-cg))
            r = dict(sequence_id=seq, frame_id=fid, pred_id=p["id"], gt_id=g["id"],
                     confidence=p["confidence"], iou=m["iou"], iou_core_rounded=result["iou"],
                     pred_bbox=p["bbox"], gt_bbox=g["bbox"], semantic_label=g["semantic_label"],
                     u_pred=u, v_pred=v, u_gt=cg[0], v_gt=cg[1],
                     pixel_center_distance=offset, center_in_gt_mask=inside,
                     window_xyxy_exclusive=[xs, ys, xe, ye], window_total_pixels=int(window.size),
                     window_valid_pixels=int(vals.size), window_valid_raw_values=vals.tolist(),
                     window_depth_grid_cleaned=window.tolist(),
                     window_gt_mask_grid=in_window.astype(int).tolist(),
                     window_mask_pixels=int(in_window.sum()),
                     window_mask_ratio=float(in_window.mean()),
                     window_valid_inside_mask_pixels=int((in_window & (window > 0)).sum()),
                     gt_mask_pixels=int(mask.sum()), gt_mask_valid_pixels=int(mvals.size),
                     gt_mask_valid_ratio=float(mvals.size / mask.sum()),
                     pred_Z=zp, gt_Z=zg, delta_Z=zp-zg, abs_delta_Z=abs(zp-zg),
                     err_X=result["err_X"], err_Y=result["err_Y"], err_Z=result["err_Z"],
                     err_3D=result["err_3D"], err_X_raw=float(error[0]),
                     err_Y_raw=float(error[1]), err_Z_raw=float(error[2]), err_3D_raw=e3d,
                     image_width=w, image_height=h, bbox_touches_image_edge=edge <= 0,
                     bbox_edge_margin_px=edge, depth_path=str(paths[0]), annotation_path=str(paths[1]))
            # Exact additive decomposition at reference depth; norms themselves are not additive.
            shared = np.array(pixel_to_3d(*cp, zg, CAM))
            r["depth_difference_error_vector"] = (xp - shared).tolist()
            r["center_offset_error_vector"] = (shared - xg).tolist()
            r["depth_difference_vector_norm"] = float(np.linalg.norm(xp-shared))
            r["center_offset_vector_norm"] = float(np.linalg.norm(shared-xg))
            assert np.allclose((xp-shared)+(shared-xg), error)
            r.update(quantiles(vals, "bbox_depth"))
            r.update(quantiles(mvals, "mask_depth"))
            flags = dict(CENTER_OUTSIDE_GT_MASK=not inside, LOW_5X5_MASK_COVERAGE=r["window_mask_ratio"] < .5,
                         LARGE_DEPTH_DIFFERENCE=abs(zp-zg) > 50, LARGE_CENTER_OFFSET=offset > 20,
                         LOW_IOU=m["iou"] < .6, NEAR_IMAGE_EDGE=edge < 8, LOW_CONFIDENCE=p["confidence"] < .3)
            r.update(flags)
            r["flags"] = [k for k, value in flags.items() if value]
            rows.append(r)
    assert len(rows) == 2959
    return rows, fallback


def v1_crosscheck(row, det, paths, eligible, stage1):
    key = row["sequence_id"], row["frame_id"]
    info = det[key]
    depth, gts = frame_data(paths)
    preds = [dict(p) for p in info["objects"] if p["confidence"] >= .30]
    attach_mask_from_pkl(preds, gts, .5)  # Redo matching after V1 confidence filtering.
    matched = [p for p in preds if p["gt_matched"]]
    full, scored = select_top1(matched, depth, WEIGHTS["visibility"],
                              info["img_size"] or v1_config.IMG_SIZE, "visibility")
    actual_eligible = full is not None and len(scored) >= 2
    assert actual_eligible == (key in eligible), f"V1 frame membership mismatch: {key}"
    baseline = min(scored, key=lambda p: p["depth_value"]) if scored else None
    candidate = next((p for p in scored if p["id"] == row["pred_id"]), None)
    v1_match = next((p for p in preds if p["id"] == row["pred_id"]), None)
    row.update(conf_ge_030=row["confidence"] >= .30,
               v1_matched=candidate is not None or bool(v1_match and v1_match["gt_matched"]),
               v1_candidate=candidate is not None, v1_eligible_frame=key in eligible,
               v1_candidate_gt_id=candidate["matched_gt_id"] if candidate else None,
               v1_same_gt_match=bool(candidate and candidate["matched_gt_id"] == row["gt_id"]),
               v1_candidate_count=len(scored),
               v1_baseline_pred=baseline["id"] if actual_eligible else None,
               v1_full_pred=full["id"] if actual_eligible else None,
               v1_baseline_selected=bool(actual_eligible and candidate is baseline),
               v1_full_selected=bool(actual_eligible and candidate is full))
    row["v1_selection_role"] = ("BOTH" if row["v1_baseline_selected"] and row["v1_full_selected"]
                               else "BASELINE" if row["v1_baseline_selected"]
                               else "FULL" if row["v1_full_selected"] else "NEITHER")
    if key in stage1:
        stored = stage1[key]
        assert baseline["id"] == int(stored["baseline_candidate_id"])
        assert full["id"] == int(stored["full_candidate_id"])
        assert baseline["depth_value"] == float(stored["baseline_representative_depth_mm"])
        assert full["depth_value"] == float(stored["full_representative_depth_mm"])
    elif actual_eligible:
        assert tuple(baseline["bbox"]) == tuple(full["bbox"]), f"Unexpected V1 disagreement: {key}"


def distributions(rows):
    output = []
    n = len(rows)
    for threshold in (50, 100, 200):
        subset = [r for r in rows if r["err_3D"] > threshold]
        med = lambda key: float(np.median([r[key] for r in subset])) if subset else None
        output.append(dict(row_type="ERROR_THRESHOLD", threshold_mm=threshold, count=len(subset),
                           total=n, percent=100*len(subset)/n, median_confidence=med("confidence"),
                           median_iou=med("iou"), median_abs_delta_Z=med("abs_delta_Z"),
                           median_pixel_center_distance=med("pixel_center_distance"),
                           center_outside_count=sum(not r["center_in_gt_mask"] for r in subset),
                           center_outside_percent=100*sum(not r["center_in_gt_mask"] for r in subset)/len(subset)
                           if subset else None))
    errors = np.array(sorted((r["err_3D"] for r in rows), reverse=True))
    for k in (0, 1, 5, 10):
        output.append(dict(row_type="DIAGNOSTIC_RMSE_ONLY", removed_top_k=k, count=n-k, total=n,
                           diagnostic_rmse_mm=float(np.sqrt(np.mean(errors[k:]**2))),
                           removed_squared_error_percent=float(100*np.sum(errors[:k]**2)/np.sum(errors**2)),
                           note="Diagnostic only; does not replace paper performance or remove stored objects."))
    return output


def mask_overlay(base, mask, opacity=.25):
    a = np.array(base.convert("RGB"), copy=True)
    a[mask] = ((1-opacity)*a[mask] + opacity*np.array(GT)).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(a, contours, -1, GT, 2)
    return Image.fromarray(a)


def annotated(base, row, mask, depth_mode=False):
    image = mask_overlay(base, mask, .10 if depth_mode else .25)
    d = ImageDraw.Draw(image)
    d.rectangle(row["gt_bbox"], outline=GT, width=3)
    d.rectangle(row["pred_bbox"], outline=PRED, width=4)
    u, v, gu, gv = row["u_pred"], row["v_pred"], row["u_gt"], row["v_gt"]
    d.line((u-9, v, u+9, v), fill=PRED, width=3)
    d.line((u, v-9, u, v+9), fill=PRED, width=3)
    d.ellipse((gu-6, gv-6, gu+6, gv+6), outline=GT, width=3)
    if depth_mode:
        x1, y1, x2, y2 = row["window_xyxy_exclusive"]
        d.rectangle((x1, y1, x2-1, y2-1), outline=WINDOW, width=1)
    # Keep the fruit and sampled pixels unobstructed in enlarged crops.
    # Method names / colors are explained outside the data panels.
    return image


def fit(canvas, image, box, nearest=False):
    x, y, w, h = box
    tile = ImageOps.contain(image, (w, h), Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
    canvas.paste(tile, (x+(w-tile.width)//2, y+(h-tile.height)//2))


def crop_box(row, size):
    boxes = [row["pred_bbox"], row["gt_bbox"]]
    return (max(0, int(min(b[0] for b in boxes))-35),
            max(0, int(min(b[1] for b in boxes))-35),
            min(size[0], int(np.ceil(max(b[2] for b in boxes)))+35),
            min(size[1], int(np.ceil(max(b[3] for b in boxes)))+35))


def depth_colors(depth):
    vals = depth[depth > 0]
    low, high = map(float, np.percentile(vals, [2, 98]))
    scale = np.clip((depth.astype(float)-low)/max(high-low, 1), 0, 1)
    a = (colormaps["viridis"](scale)[:, :, :3]*255).astype(np.uint8)
    a[depth == 0] = (18, 22, 28)
    return Image.fromarray(a), low, high


def header(canvas, row, title):
    d = ImageDraw.Draw(canvas)
    d.text((60, 40), f"Rank {row['rank']:02d} | {title}", font=font(53, True), fill=INK)
    d.text((60, 112), f"Seq {row['sequence_id']}  |  Frame {row['frame_id']}  |  "
           f"Pred #{row['pred_id']} / GT {row['gt_id']}  |  3D error {row['err_3D']:.1f} mm",
           font=font(35), fill=INK)


def line_block(canvas, x, y, lines, size=32, step=51):
    d = ImageDraw.Draw(canvas)
    for line in lines:
        d.text((x, y), line, font=font(size), fill=INK)
        y += step


def side_statistics(canvas, row, x=2280, y=300):
    r = row
    line_block(canvas, x, y, [
        f"Confidence: {r['confidence']:.4f}",
        f"Matched IoU: {r['iou']:.4f}",
        "",
        f"Pred Z: {r['pred_Z']:g} mm",
        f"GT-mask Z: {r['gt_Z']:g} mm",
        f"Delta Z: {r['delta_Z']:+g} mm",
        "",
        f"Error X: {r['err_X']:+.1f} mm",
        f"Error Y: {r['err_Y']:+.1f} mm",
        f"Error Z: {r['err_Z']:+.1f} mm",
        "",
        f"Center offset: {r['pixel_center_distance']:.2f} px",
        f"Center in GT: {r['center_in_gt_mask']}",
        f"5x5 mask: {r['window_mask_pixels']}/{r['window_total_pixels']}",
        f"5x5 valid: {r['window_valid_pixels']}/{r['window_total_pixels']}",
        f"GT valid: {r['gt_mask_valid_pixels']}/{r['gt_mask_pixels']}",
        "",
        f"V1 candidate: {r['v1_candidate']}",
        f"V1 role: {r['v1_selection_role']}",
    ], size=32, step=51)


def save_png(canvas, path, rank):
    if path.exists():
        raise FileExistsError(path)
    # Rank 1 is a 4K review asset; numerical pixels remain unchanged in source.
    if rank == 1:
        canvas = canvas.resize((3840, 2304), Image.Resampling.LANCZOS)
    canvas.save(path, dpi=(300, 300))


def visualizations(row, rgb, depth, mask, folder):
    cb = crop_box(row, rgb.size)
    dv, low, high = depth_colors(depth)
    rgb_ann = annotated(rgb, row, mask)
    depth_ann = annotated(dv, row, mask, True)
    for name, title, display, is_depth in (
        ("rgb_overlay.png", "Localization: RGB / GT mask", rgb_ann, False),
        ("depth_visual.png", "Localization: depth / 5x5 sample", depth_ann, True),
    ):
        canvas = Image.new("RGB", (3000, 1800), BG)
        header(canvas, row, title)
        d = ImageDraw.Draw(canvas)
        d.text((70, 210), "Full frame", font=font(33, True), fill=INK)
        d.text((990, 210), "Predicted / matched GT region (enlarged)", font=font(33, True), fill=INK)
        fit(canvas, display, (60, 275, 850, 1420), nearest=is_depth)
        fit(canvas, display.crop(cb), (990, 280, 1210, 1140), nearest=is_depth)
        side_statistics(canvas, row)
        d.text((990, 1480), "Red: predicted bbox / cross = sampled center", font=font(32), fill=PRED)
        d.text((990, 1530), "Cyan: GT mask / bbox / circle = mask centroid", font=font(32), fill=GT)
        if is_depth:
            d.text((990, 1580), "Yellow: exact 5x5 sample window", font=font(32), fill=INK)
            strip = np.tile(np.linspace(0, 1, 700), (26, 1))
            bar = Image.fromarray((colormaps["viridis"](strip)[:, :, :3]*255).astype(np.uint8))
            canvas.paste(bar, (990, 1640))
            d.text((990, 1680), f"{low:.0f} mm", font=font(27), fill=INK)
            d.text((1530, 1680), f"{high:.0f} mm", font=font(27), fill=INK)
            d.text((1770, 1640), "Black = invalid", font=font(29), fill=INK)
        else:
            d.text((990, 1610), "GT reference: all-mask centroid + valid mask median", font=font(30), fill=INK)
        d.text((60, 1752), "Diagnostic visualization only; source depth, masks and evaluation remain unchanged.",
               font=font(27), fill=INK)
        save_png(canvas, folder/name, row["rank"])

    canvas = Image.new("RGB", (3000, 1800), BG)
    header(canvas, row, "Crop / mask / depth evidence")
    d = ImageDraw.Draw(canvas)
    for x, label, image, nearest in ((60, "Original RGB crop", rgb.crop(cb), False),
                                    (1040, "Matched GT mask + geometry", rgb_ann.crop(cb), False),
                                    (2020, "Depth + exact 5x5 location", depth_ann.crop(cb), True)):
        d.text((x, 205), label, font=font(33, True), fill=INK)
        fit(canvas, image, (x, 260, 920, 720), nearest)
    d.text((1040, 985), "Red: prediction / cross; cyan: GT / centroid", font=font(27), fill=INK)
    d.text((2020, 985), "Yellow: 5x5 window; black: invalid depth", font=font(27), fill=INK)
    d.text((60, 1020), "5x5 cleaned depth values (mm)", font=font(34, True), fill=INK)
    grid = row["window_depth_grid_cleaned"]
    maskgrid = row["window_gt_mask_grid"]
    xs, ys, _, _ = row["window_xyxy_exclusive"]
    for iy, values in enumerate(grid):
        for ix, value in enumerate(values):
            box = (65+105*ix, 1085+105*iy, 165+105*ix, 1185+105*iy)
            fill = (190, 232, 245) if maskgrid[iy][ix] else (224, 228, 234)
            d.rectangle(box, fill=fill, outline=(175, 181, 190), width=2)
            if xs+ix == row["u_pred"] and ys+iy == row["v_pred"]:
                d.rectangle(box, outline=PRED, width=6)
            d.text((box[0]+13, box[1]+34), str(value), font=font(29, True), fill=INK)
    d.text((60, 1640), "Cyan cell = inside matched GT mask", font=font(29), fill=GT)
    d.text((60, 1690), "Red border = predicted bbox center", font=font(29), fill=PRED)
    r = row
    line_block(canvas, 710, 1080, [
        f"5x5 valid depth: {r['window_valid_pixels']}/{r['window_total_pixels']} pixels",
        f"5x5 matched-mask coverage: {r['window_mask_pixels']}/{r['window_total_pixels']}",
        f"Min / Q1 / Median / Q3 / Max (mm):",
        " / ".join(f"{r['bbox_depth_'+k]:g}" for k in ("min", "q1", "median", "q3", "max")),
        "",
        f"GT mask: {r['gt_mask_pixels']} pixels",
        f"GT valid depth: {r['gt_mask_valid_pixels']} ({100*r['gt_mask_valid_ratio']:.1f}%)",
        f"GT depth min / Q1 / Median / Q3 / Max:",
        " / ".join(f"{r['mask_depth_'+k]:g}" for k in ("min", "q1", "median", "q3", "max")),
    ], size=30, step=55)
    line_block(canvas, 1850, 1080, [
        f"Pred Z / GT Z: {r['pred_Z']:g} / {r['gt_Z']:g} mm",
        f"Delta Z: {r['delta_Z']:+g} mm;  3D error: {r['err_3D']:.1f} mm",
        f"Center displacement: {r['pixel_center_distance']:.2f} px",
        f"Center inside matched GT mask: {r['center_in_gt_mask']}",
        f"V1 candidate / role: {r['v1_candidate']} / {r['v1_selection_role']}",
        "",
        "Flags (not causal labels):",
        *row["flags"],
    ], size=29, step=44)
    save_png(canvas, folder/"crop_analysis.png", row["rank"])


def report(top, dist, actual, meta, max_depth_rows, same, protected_count):
    r = top[0]
    threshold_rows = [d for d in dist if d["row_type"] == "ERROR_THRESHOLD"]
    rmse_rows = [d for d in dist if d["row_type"] == "DIAGNOSTIC_RMSE_ONLY"]
    depthnorm, centernorm = r["depth_difference_vector_norm"], r["center_offset_vector_norm"]
    lines = [
        "# Localization Outlier Audit", "", "## Reproduction", "",
        f"- Result: PASS; objects **{actual['objects']}**, median **{actual['median']:.1f} mm**, "
        f"RMSE **{actual['RMSE']:.1f} mm**, maximum **{actual['max']:.1f} mm**.",
        f"- Localization input: {meta['input_frames']} frames / {meta['gt_objects']} GT objects, sequences 400–409.",
        "- confidence >= 0.05 → localization.match_by_iou(IoU >= 0.5), original greedy pair ordering.",
        "- Prediction: localization.depth_from_bbox_center(), integer bbox center, clipped 5×5, valid (>0) median.",
        "- Reference: localization.depth_from_mask(), centroid of ALL mask pixels + valid (>0) mask-depth median.",
        "- data_loader.clean_depth(): 65535 → 0; zeros excluded. No rescale, inferred correction, or outlier removal.",
        f"- Evaluated reference fallback count: {meta['reference_fallback_count']} (all evaluated references are mask-based).",
        f"- Intrinsics from {ROOT/loc_config.CAM_PATH}: {CAM}.",
        "- Paper statistics, ordering and threshold groups use core err_3D rounded to 0.1 mm per object; raw precision is additionally stored.",
        "- V1 crosscheck uses the separate authoritative 364-frame set and confidence 0.30; 0.30 matching is rerun from the full CSV in original order.",
        "- V1 baseline uses the original single min(depth) and Full uses priority.select_top1; GT-mask depth is NOT used for V1 selection.",
        "", "## Maximum-error object", "",
        f"- Seq / Frame: **{r['sequence_id']} / {r['frame_id']}**; pred **{r['pred_id']}**, GT **{r['gt_id']}**.",
        f"- Confidence {r['confidence']:.8f}; IoU {r['iou']:.6f}.",
        f"- Pred bbox {r['pred_bbox']}; GT bbox {r['gt_bbox']}.",
        f"- Pred center ({r['u_pred']}, {r['v_pred']}); GT mask centroid ({r['u_gt']:.6f}, {r['v_gt']:.6f}); offset {r['pixel_center_distance']:.6f} px.",
        f"- Pred Z **{r['pred_Z']:g} mm**, GT Z **{r['gt_Z']:g} mm**, delta Z **{r['delta_Z']:+g} mm**.",
        f"- Signed X/Y/Z errors: {r['err_X']:+.1f} / {r['err_Y']:+.1f} / {r['err_Z']:+.1f} mm; 3D error {r['err_3D']:.1f} mm.",
        f"- 5×5 window [x1,y1,x2,y2), exclusive upper bounds: {r['window_xyxy_exclusive']}.",
        f"- Valid values (row-major order, invalid excluded): {r['window_valid_raw_values']}.",
        f"- Center inside GT mask: **{r['center_in_gt_mask']}**; 5×5 GT coverage **{r['window_mask_pixels']}/{r['window_total_pixels']}**.",
        f"- Of the {r['window_valid_pixels']} valid 5×5 samples, **{r['window_valid_inside_mask_pixels']}** are inside the matched GT mask.",
        f"- GT mask valid depth pixels: {r['gt_mask_valid_pixels']}/{r['gt_mask_pixels']} ({100*r['gt_mask_valid_ratio']:.2f}%).",
        f"- Same object as global maximum absolute bbox/mask depth difference (274 mm): **{'YES' if same else 'NO'}**.",
        f"- V1 conf >=0.30: {r['conf_ge_030']}; candidate: {r['v1_candidate']}; eligible frame: {r['v1_eligible_frame']}; "
        f"baseline selected: {r['v1_baseline_selected']}; Full selected: {r['v1_full_selected']}.",
        "", "### Observations, not an automatic causal verdict", "",
        f"- 최대 사례의 5×5 표본은 matched GT mask와 {r['window_mask_pixels']}/{r['window_total_pixels']} 픽셀만 겹친다. "
        f"bbox center의 mask 내부 여부는 {r['center_in_gt_mask']}이며, 표본 median은 GT-mask median보다 {abs(r['delta_Z']):g} mm "
        f"{'가깝다' if r['delta_Z'] < 0 else '멀다'}. 이는 같은 GT 과실의 mask 내부 깊이를 표본화하지 못했는지 확인할 직접적인 근거다.",
        f"- 순수 좌표 차이와 depth 차이를 분리하면 depth 항 벡터 norm {depthnorm:.3f} mm, "
        f"GT depth에서 center 이동 항 norm {centernorm:.3f} mm이다. 두 벡터의 합이 실제 XYZ 오차와 정확히 일치한다 "
        "(벡터 norm끼리는 단순 합산하지 않음).",
        "- 위 depth 항에도 off-axis projection에 따른 X/Y 오차가 포함된다. 따라서 큰 X/Y 오차를 단순 pixel-center offset으로만 설명하지 않는다.",
        "- RGB/mask/depth 이미지를 함께 검토해야 표본이 잎·다른 과실·배경 중 무엇에 놓였는지 판단할 수 있다. "
        "IoU, confidence, edge 등의 flag는 원인 확정이나 제외 규칙이 아니다.",
        "- GT-mask median도 동일 센서의 reference proxy이며 독립적인 외부 3D 정답이 아니다.",
        "", "## Top 10", "",
        "|Rank|Seq|Frame|Pred / GT|3D mm|Delta Z mm|Conf|IoU|Center in mask|5×5 mask|V1 role|",
        "|---:|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for t in top:
        lines.append(f"|{t['rank']}|{t['sequence_id']}|{t['frame_id']}|{t['pred_id']} / {t['gt_id']}|"
                     f"{t['err_3D']:.1f}|{t['delta_Z']:+g}|{t['confidence']:.4f}|{t['iou']:.4f}|"
                     f"{t['center_in_gt_mask']}|{t['window_mask_pixels']}/{t['window_total_pixels']}|{t['v1_selection_role']}|")
    lines += ["", "## Outlier distribution", "",
              "|Strict threshold|Objects|% of 2,959|Median confidence|Median IoU|Median abs delta Z mm|Median center px|Center outside %|",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d in threshold_rows:
        lines.append(f"|>{d['threshold_mm']} mm|{d['count']}|{d['percent']:.3f}|{d['median_confidence']:.4f}|"
                     f"{d['median_iou']:.4f}|{d['median_abs_delta_Z']:.2f}|{d['median_pixel_center_distance']:.2f}|"
                     f"{d['center_outside_percent']:.2f}|")
    lines += ["", "## RMSE influence — diagnostic only", "",
              "다음 수치는 상위 오차의 RMSE 기여를 보여주는 진단값일 뿐, 논문 성능을 대체하지 않으며 원래 2,959개 결과는 유지한다.", "",
              "|Top k excluded only for this calculation|Remaining n|Diagnostic RMSE mm|Excluded share of squared error %|",
              "|---:|---:|---:|---:|"]
    for d in rmse_rows:
        lines.append(f"|{d['removed_top_k']}|{d['count']}|{d['diagnostic_rmse_mm']:.4f}|{d['removed_squared_error_percent']:.3f}|")
    lines += ["", "## Diagnostic flag definitions", ""]
    lines.extend(f"- {k}: {v}." for k, v in FLAG_RULES.items())
    lines += ["", "## Depth-difference maximum identity", ""]
    lines.extend(f"- Seq{m['sequence_id']} / {m['frame_id']} / pred{m['pred_id']} / GT{m['gt_id']}: "
                 f"abs delta Z {m['abs_delta_Z']:g} mm, 3D {m['err_3D']:.1f} mm." for m in max_depth_rows)
    lines += ["", "## Files and preservation", "",
              "- [Top 10 CSV](top10_localization_outliers.csv)",
              "- [Distribution and diagnostic RMSE CSV](outlier_distribution.csv)",
              "- [Max case JSON](max_error_case.json)",
              "- [Max RGB](rank01_max_error/rgb_overlay.png), [max depth](rank01_max_error/depth_visual.png), "
              "[max crop analysis](rank01_max_error/crop_analysis.png)",
              "- Rank02–rank10 each contain the same three PNG filenames.",
              "- Depth colormap: frame-wise valid-depth percentiles 2–98; display clipping only, invalid pixels black.",
              "- Full images and enlarged crops are spatially consistent; every 5×5 raw grid is exported in CSV and drawn in crop analysis.",
              f"- SHA-256 unchanged check: {protected_count} existing source/result/input files.",
              "- Source modified: NO. Existing results modified: NO. Dataset modified: NO. No Git operation.", ""]
    return "\n".join(lines)


def print_final(actual, top, dist, same):
    r = top[0]
    print("\n=== LOCALIZATION OUTLIER AUDIT ===")
    print("Reproduction:")
    for k, unit in (("objects", ""), ("median", " mm"), ("RMSE", " mm"), ("max", " mm")):
        print(f"{k:<13}: {actual[k]} / expected {EXPECTED[k]}{unit}")
    print("Result       : PASS")
    print("\nMax error:")
    for label, value in (
        ("Seq / Frame", f"{r['sequence_id']} / {r['frame_id']}"),
        ("Pred / GT", f"{r['pred_id']} / {r['gt_id']}"),
        ("confidence", r["confidence"]), ("IoU", f"{r['iou']:.9f}"),
        ("pred_Z", f"{r['pred_Z']:g} mm"), ("gt_Z", f"{r['gt_Z']:g} mm"),
        ("Delta Z", f"{r['delta_Z']:+g} mm"),
        ("X / Y / Z / 3D error", f"{r['err_X']:+.1f} / {r['err_Y']:+.1f} / {r['err_Z']:+.1f} / {r['err_3D']:.1f} mm"),
        ("pixel center distance", f"{r['pixel_center_distance']:.6f} px"),
        ("bbox center in GT mask", r["center_in_gt_mask"]),
        ("5x5 GT-mask coverage", f"{r['window_mask_pixels']}/{r['window_total_pixels']}"),
        ("5x5 valid depth raw values", r["window_valid_raw_values"]),
        ("Conf >= 0.30", r["conf_ge_030"]), ("V1 eligible candidate", r["v1_candidate"]),
        ("V1 eligible frame", r["v1_eligible_frame"]),
        ("V1 baseline selected", r["v1_baseline_selected"]), ("V1 Full selected", r["v1_full_selected"]),
    ):
        print(f"{label}: {value}")
    print("\nDepth difference max 274 mm same object:")
    print("YES" if same else "NO")
    print("\nOutlier counts:")
    for d in dist:
        if d["row_type"] == "ERROR_THRESHOLD":
            print(f"> {d['threshold_mm']} mm: {d['count']}/{d['total']} ({d['percent']:.3f}%)")
    print("\nSource modified: NO\nExisting results modified: NO")
    print("Output:", OUT)
    print("=============================")


def main():
    global CAM
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing audit: {OUT}")
    expected_weights = dict(proximity=.35, visibility=.30, confidence=.20, center=.15)
    assert WEIGHTS["visibility"] == expected_weights
    assert v1_config.CONF_THRESHOLD == .30 and v1_config.IOU_THRESHOLD == .5
    assert loc_config.IOU_THRESHOLD == .5
    protected = protect_existing()
    det = load_detections_csv(str(ROOT/loc_config.DETECTION_CSV))
    CAM = load_cam_params(str(ROOT/loc_config.CAM_PATH))
    specs = specs_for(det)
    for paths in specs.values():
        for p in paths:
            protected[p] = digest(p)
    frame_csv = csv_read(ROOT/"gt_coverage_frame_summary.csv")
    eligible = {(r["sequence"], r["frame_id"]) for r in frame_csv}
    assert len(frame_csv) == len(eligible) == 364
    historical = csv_read(ROOT/"gt_count_3544_vs_3542_by_frame.csv")
    loc_set = {(r["sequence"], r["frame_id"]) for r in historical if r["old_frame_included"] == "True"}
    assert set(specs) == loc_set, "Localization input frame IDs differ from existing input audit"
    assert eligible <= loc_set
    stage1 = {(r["sequence_id"], r["frame_id"]): r for r in csv_read(
        ROOT/"results/ppt_v1_examples_stage1/v1_disagreement_all.csv")}
    assert len(stage1) == 103
    core, actual, meta = reproduce(det, specs)
    print("Reproduction:", actual, flush=True)
    if actual != EXPECTED:
        print("Result: FAIL — stopping before outlier analysis.")
        for key in EXPECTED:
            if actual[key] != EXPECTED[key]:
                print(f"Mismatch {key}: actual={actual[key]}, expected={EXPECTED[key]}")
        print("Input frames / GT objects:", meta["input_frames"], meta["gt_objects"])
        check_protected(protected)
        return 1
    print("Reproduction PASS. Tracing matched objects...", flush=True)
    rows, fallback = trace_all(det, specs, core)
    meta["reference_fallback_count"] = fallback
    ordered = sorted(rows, key=lambda r: r["err_3D"], reverse=True)  # Stable original order for equal errors.
    top = [dict(r, rank=i) for i, r in enumerate(ordered[:10], 1)]
    max_abs = max(r["abs_delta_Z"] for r in rows)
    max_depth_rows = [r for r in rows if r["abs_delta_Z"] == max_abs]
    identity = lambda r: (r["sequence_id"], r["frame_id"], r["pred_id"], r["gt_id"])
    same = identity(top[0]) in {identity(r) for r in max_depth_rows}
    if max_abs != 274:
        print(f"Observed maximum abs depth difference differs from historical 274 mm: {max_abs}", flush=True)
    prepared = []
    for row in top:
        key = row["sequence_id"], row["frame_id"]
        v1_crosscheck(row, det, specs[key], eligible, stage1)
        rp = rgb_path(*key, det[key])
        protected[rp] = digest(rp)
        row["rgb_path"] = str(rp)
        with Image.open(rp) as source:
            assert source.mode == "RGB", f"Unexpected RGB mode: {source.mode}"
            rgb = source.copy()
        depth, gts = frame_data(specs[key])
        assert depth.shape == (rgb.height, rgb.width)
        assert rgb.size == (row["image_width"], row["image_height"])
        gt = next(g for g in gts if g["id"] == row["gt_id"])
        prepared.append((row, rgb, depth, np.asarray(gt["instance_mask"], dtype=bool)))
    dist = distributions(rows)
    check_protected(protected)
    OUT.mkdir(parents=True, exist_ok=True)
    for row, rgb, depth, mask in prepared:
        folder = OUT/("rank01_max_error" if row["rank"] == 1 else f"rank{row['rank']:02d}")
        folder.mkdir()
        visualizations(row, rgb, depth, mask, folder)
        print(f"Rank {row['rank']:02d} exported: seq{row['sequence_id']}/{row['frame_id']} "
              f"pred{row['pred_id']} GT{row['gt_id']} | {row['err_3D']:.1f} mm | V1 {row['v1_selection_role']}",
              flush=True)
    check_protected(protected)
    csv_write(OUT/"top10_localization_outliers.csv", top)
    csv_write(OUT/"outlier_distribution.csv", dist)
    bundle = dict(reproduction=actual, input_metadata=meta, camera_intrinsics=CAM, max_error_case=top[0],
                  maximum_absolute_depth_difference_mm=max_abs, max_depth_difference_same_object=same,
                  max_depth_difference_cases=max_depth_rows, diagnostic_flags=FLAG_RULES,
                  diagnostic_rmse=[d for d in dist if d["row_type"] == "DIAGNOSTIC_RMSE_ONLY"],
                  localization_input_frame_ids=[dict(sequence_id=s, frame_id=f) for s, f in specs],
                  v1_eligible_frame_ids=[dict(sequence_id=s, frame_id=f) for s, f in sorted(eligible)],
                  input_sha256={str(p): v for p, v in protected.items()},
                  source_modified=False, existing_results_modified=False, dataset_modified=False)
    with (OUT/"max_error_case.json").open("x", encoding="utf-8") as stream:
        json.dump(bundle, stream, indent=2, ensure_ascii=False, default=native, allow_nan=False)
    with (OUT/"LOCALIZATION_OUTLIER_AUDIT.md").open("x", encoding="utf-8") as stream:
        stream.write(report(top, dist, actual, meta, max_depth_rows, same, len(protected)))
    assert len(csv_read(OUT/"top10_localization_outliers.csv")) == 10
    for path in OUT.rglob("*.png"):
        with Image.open(path) as im:
            im.verify()
    assert len(list(OUT.rglob("*.png"))) == 30
    check_protected(protected)
    print_final(actual, top, dist, same)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
