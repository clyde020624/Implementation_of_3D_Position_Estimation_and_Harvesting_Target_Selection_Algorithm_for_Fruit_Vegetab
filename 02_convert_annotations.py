"""BUP-ST20 annotation pkl을 YOLO Detection bbox 라벨로 순차 변환한다.

주의:
- 공식 배포처에서 받은 신뢰할 수 있는 pickle 파일에만 사용한다.
- instance_mask는 YOLO Detection 학습에 저장하지 않는다.
- bbox 형식은 공식 문서의 [x, y, width, height]를 사용한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SEMANTIC_NAMES = {
    0: "red",
    1: "yellow",
    2: "green",
    3: "mixed_red",
    4: "mixed_yellow",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--classes",
        choices=["one", "five"],
        default="one",
        help="one: 모든 파프리카를 class 0, five: 원래 5개 semantic class 유지",
    )
    parser.add_argument("--image-width", type=int, default=720)
    parser.add_argument("--image-height", type=int, default=1280)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sequence_and_frame(member_name: str) -> tuple[str, str] | None:
    parts = PurePosixPath(member_name).parts
    lowered = [part.lower() for part in parts]
    if "annotations" in lowered:
        index = lowered.index("annotations")
        remaining = parts[index + 1 :]
    else:
        remaining = parts[-2:]

    if len(remaining) < 2:
        return None
    sequence_id = remaining[-2]
    frame_id = Path(remaining[-1]).stem
    return sequence_id, frame_id


def iter_instances(annotation: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(annotation, dict):
        for instance_id, record in annotation.items():
            if isinstance(record, dict):
                yield str(instance_id), record
    elif isinstance(annotation, list):
        for index, record in enumerate(annotation):
            if isinstance(record, dict):
                yield str(record.get("instanceid", index)), record


def bbox_to_yolo(
    bbox: Any, image_width: int, image_height: int
) -> tuple[float, float, float, float, tuple[float, float, float, float]] | None:
    if bbox is None or len(bbox) != 4:
        return None

    x, y, width, height = (float(value) for value in bbox)
    x1 = min(max(x, 0.0), float(image_width))
    y1 = min(max(y, 0.0), float(image_height))
    x2 = min(max(x + width, 0.0), float(image_width))
    y2 = min(max(y + height, 0.0), float(image_height))

    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        return None

    center_x = (x1 + x2) / 2.0 / image_width
    center_y = (y1 + y2) / 2.0 / image_height
    normalized_width = clipped_width / image_width
    normalized_height = clipped_height / image_height
    return (
        center_x,
        center_y,
        normalized_width,
        normalized_height,
        (x1, y1, x2, y2),
    )


def main() -> None:
    args = parse_args()
    archive_path = args.archive.resolve()
    dataset_root = args.dataset_root.resolve()
    labels_root = dataset_root / "labels"
    metadata_root = dataset_root / "metadata"
    labels_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    frames_csv = metadata_root / "annotation_frames.csv"
    objects_csv = metadata_root / "annotation_objects.csv"

    converted_frames = 0
    empty_frames = 0
    invalid_objects = 0
    converted_objects = 0

    with (
        frames_csv.open("w", newline="", encoding="utf-8-sig") as frame_file,
        objects_csv.open("w", newline="", encoding="utf-8-sig") as object_file,
        tarfile.open(archive_path, mode="r|gz") as archive,
    ):
        frame_writer = csv.DictWriter(
            frame_file,
            fieldnames=["sequence_id", "frame_id", "object_count", "status"],
        )
        object_writer = csv.DictWriter(
            object_file,
            fieldnames=[
                "sequence_id",
                "frame_id",
                "instance_id",
                "source_semlabel",
                "source_class_name",
                "yolo_class_id",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )
        frame_writer.writeheader()
        object_writer.writeheader()

        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".pkl"):
                continue

            identity = sequence_and_frame(member.name)
            if identity is None:
                continue
            sequence_id, frame_id = identity

            source = archive.extractfile(member)
            if source is None:
                continue
            with source:
                # Pickle은 임의 코드를 실행할 수 있으므로 공식 데이터에만 사용한다.
                annotation = pickle.load(source)

            label_lines: list[str] = []
            object_rows: list[dict[str, Any]] = []
            for instance_id, record in iter_instances(annotation):
                converted = bbox_to_yolo(
                    record.get("bbox"), args.image_width, args.image_height
                )
                if converted is None:
                    invalid_objects += 1
                    continue

                semlabel = int(record.get("semlabel", 0))
                if semlabel not in SEMANTIC_NAMES:
                    invalid_objects += 1
                    continue
                class_id = 0 if args.classes == "one" else semlabel
                center_x, center_y, width, height, xyxy = converted
                x1, y1, x2, y2 = xyxy

                label_lines.append(
                    f"{class_id} {center_x:.8f} {center_y:.8f} "
                    f"{width:.8f} {height:.8f}"
                )
                object_rows.append(
                    {
                        "sequence_id": sequence_id,
                        "frame_id": frame_id,
                        "instance_id": instance_id,
                        "source_semlabel": semlabel,
                        "source_class_name": SEMANTIC_NAMES[semlabel],
                        "yolo_class_id": class_id,
                        "x1": f"{x1:.3f}",
                        "y1": f"{y1:.3f}",
                        "x2": f"{x2:.3f}",
                        "y2": f"{y2:.3f}",
                    }
                )

            label_path = labels_root / sequence_id / f"{frame_id}.txt"
            if label_lines:
                label_path.parent.mkdir(parents=True, exist_ok=True)
                if args.overwrite or not label_path.exists():
                    label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                converted_frames += 1
                converted_objects += len(label_lines)
                status = "converted"
                for row in object_rows:
                    object_writer.writerow(row)
            else:
                empty_frames += 1
                status = "empty_or_invalid"

            frame_writer.writerow(
                {
                    "sequence_id": sequence_id,
                    "frame_id": frame_id,
                    "object_count": len(label_lines),
                    "status": status,
                }
            )
            if (converted_frames + empty_frames) % 500 == 0:
                print(
                    f"처리 {converted_frames + empty_frames:,}프레임 / "
                    f"유효 객체 {converted_objects:,}개"
                )

    config = {
        "classes": args.classes,
        "image_width": args.image_width,
        "image_height": args.image_height,
        "semantic_names": SEMANTIC_NAMES,
        "annotation_archive": str(archive_path),
    }
    (metadata_root / "conversion_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"변환 프레임: {converted_frames:,}")
    print(f"빈/무효 프레임: {empty_frames:,}")
    print(f"변환 객체: {converted_objects:,}")
    print(f"무효 객체: {invalid_objects:,}")
    print(f"YOLO 라벨: {labels_root}")


if __name__ == "__main__":
    main()

