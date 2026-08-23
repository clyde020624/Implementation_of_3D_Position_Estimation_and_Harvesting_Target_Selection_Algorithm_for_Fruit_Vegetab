"""Create the GT-mask-depth selection demo for sequence 405.

This standalone script leaves all existing research/demo files untouched and
writes only ``demo_405_mask_depth_selection.png``.
"""

import glob
import os

import cv2
from PIL import Image, ImageDraw, ImageFont

from data_loader import clean_depth, load_bupst20_annotation, load_depth
from load_detections import load_detections_csv
from mask_depth_reanalysis import (
    FULL_WEIGHTS,
    MODE,
    prepare_candidates,
    select_top1_precomputed,
)
from priority import score_visibility


SEQUENCE = "405"
FRAME_ID = "1600938547284129"

RGB_PATH = FRAME_ID + ".tiff"
DATA_DIR = "dataset_bulk"
DETECTION_CSV = "eval_detections_fixed.csv"
OUTPUT_PATH = "demo_405_mask_depth_selection.png"

CONF_THRESHOLD = 0.30
EXPECTED_NEAREST_ID = 2
EXPECTED_FULL_ID = 0

NEAREST_COLOR = (215, 48, 39)
FULL_COLOR = (31, 119, 180)


def get_font(size):
    for path in (
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def load_rgb(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read RGB image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def draw_candidate(draw, image_size, bbox, color, title, depth, visibility):
    width, height = image_size
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))

    line_width = 5
    draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

    label = (
        f"{title}\n"
        f"Depth {depth:.0f} mm\n"
        f"Visibility {visibility:.4f}"
    )
    label_font = get_font(22)
    bounds = draw.multiline_textbbox(
        (0, 0), label, font=label_font, spacing=3
    )
    label_width = bounds[2] - bounds[0] + 14
    label_height = bounds[3] - bounds[1] + 10

    label_x = max(0, min(width - label_width, x1))
    label_y = y1 - label_height - 4
    if label_y < 0:
        label_y = min(height - label_height, y2 + 4)

    draw.rectangle(
        (
            label_x,
            label_y,
            label_x + label_width,
            label_y + label_height,
        ),
        fill=color,
    )
    draw.multiline_text(
        (label_x + 7, label_y + 5),
        label,
        fill="white",
        font=label_font,
        spacing=3,
    )


def main():
    detections = load_detections_csv(DETECTION_CSV)
    info = detections.get((SEQUENCE, FRAME_ID))
    if info is None:
        raise RuntimeError(
            f"No detections for sequence={SEQUENCE}, frame={FRAME_ID}"
        )

    depth_paths = glob.glob(
        os.path.join(
            DATA_DIR,
            "depth",
            SEQUENCE,
            FRAME_ID + ".tif*",
        )
    )
    annotation_path = os.path.join(
        DATA_DIR,
        "annotations",
        SEQUENCE,
        FRAME_ID + ".pkl",
    )
    if not depth_paths or not os.path.exists(annotation_path):
        raise FileNotFoundError("Depth or annotation file is missing")

    predictions = [
        dict(obj)
        for obj in info["objects"]
        if obj["confidence"] >= CONF_THRESHOLD
    ]
    depth = clean_depth(load_depth(depth_paths[0]))
    ground_truth = load_bupst20_annotation(annotation_path)
    candidates = prepare_candidates(predictions, ground_truth, depth)

    image_size = info["img_size"] or (720, 1280)
    full, _ = select_top1_precomputed(
        candidates,
        "mask_depth",
        FULL_WEIGHTS,
        image_size,
        MODE,
    )
    minimum_depth = min(candidate["mask_depth"] for candidate in candidates)
    nearest_set = [
        candidate
        for candidate in candidates
        if candidate["mask_depth"] == minimum_depth
    ]

    if len(nearest_set) != 1:
        raise RuntimeError(
            f"Expected singleton nearest set, found {len(nearest_set)}"
        )
    nearest = nearest_set[0]
    if nearest["id"] != EXPECTED_NEAREST_ID:
        raise RuntimeError(
            f"Nearest ID mismatch: {nearest['id']} != {EXPECTED_NEAREST_ID}"
        )
    if full is None or full["id"] != EXPECTED_FULL_ID:
        actual = None if full is None else full["id"]
        raise RuntimeError(f"Full ID mismatch: {actual} != {EXPECTED_FULL_ID}")

    nearest_visibility = float(score_visibility(nearest))
    full_visibility = float(score_visibility(full))
    delta_depth = float(full["mask_depth"] - nearest["mask_depth"])
    delta_visibility = full_visibility - nearest_visibility

    if abs(delta_depth - 6.0) > 1e-9:
        raise RuntimeError(f"Delta depth mismatch: {delta_depth}")
    if abs(delta_visibility - 0.11142112277322969) > 1e-9:
        raise RuntimeError(f"Delta visibility mismatch: {delta_visibility}")

    image = load_rgb(RGB_PATH)
    if image.size != image_size:
        raise RuntimeError(
            f"RGB size {image.size} does not match detection size {image_size}"
        )
    draw = ImageDraw.Draw(image)

    draw_candidate(
        draw,
        image.size,
        nearest["bbox"],
        NEAREST_COLOR,
        "Nearest",
        nearest["mask_depth"],
        nearest_visibility,
    )
    draw_candidate(
        draw,
        image.size,
        full["bbox"],
        FULL_COLOR,
        "Full",
        full["mask_depth"],
        full_visibility,
    )

    image.save(OUTPUT_PATH)

    print("=" * 76)
    print("GT-mask depth selection demo verification")
    print("=" * 76)
    print(
        f"Nearest prediction ID={nearest['id']} | bbox={nearest['bbox']}"
    )
    print(
        f"  Depth={nearest['mask_depth']:.1f} mm | "
        f"Visibility={nearest_visibility:.4f}"
    )
    print(f"Full prediction ID={full['id']} | bbox={full['bbox']}")
    print(
        f"  Depth={full['mask_depth']:.1f} mm | "
        f"Visibility={full_visibility:.4f}"
    )
    print(f"Delta Depth      = {delta_depth:+.1f} mm")
    print(f"Delta Visibility = {delta_visibility:+.4f}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
