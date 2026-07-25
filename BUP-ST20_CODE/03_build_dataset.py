"""이미지와 YOLO 라벨을 연결해 공식 sequence split 목록과 data.yaml을 만듦."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


SPLIT_RANGES = {
    "train": range(100, 227),
    "val": range(300, 372),
    "eval": range(400, 476),
}
IMAGE_SUFFIXES = (".tiff", ".tif", ".png", ".jpg", ".jpeg")
CLASS_NAMES = {
    "one": {0: "pepper"},
    "five": {
        0: "red",
        1: "yellow",
        2: "green",
        3: "mixed_red",
        4: "mixed_yellow",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--classes", choices=["one", "five"], default="one")
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="라벨이 없는 이미지를 진짜 배경(negative)으로 확신할 때만 사용",
    )
    return parser.parse_args()


def find_image(images_root: Path, sequence_id: str, frame_id: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = images_root / sequence_id / f"{frame_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def class_ids_in_label(label_path: Path) -> set[int]:
    class_ids: set[int] = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields:
            class_ids.add(int(fields[0]))
    return class_ids


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    images_root = root / "images"
    labels_root = root / "labels"
    splits_root = root / "splits"
    splits_root.mkdir(parents=True, exist_ok=True)

    if not images_root.exists():
        raise FileNotFoundError(f"RGB 이미지 폴더가 없습니다: {images_root}")
    if not labels_root.exists() and not args.include_unlabeled:
        raise FileNotFoundError(f"라벨 폴더가 없습니다: {labels_root}")

    allowed_class_ids = set(CLASS_NAMES[args.classes])
    counts: dict[str, int] = {}

    for split_name, sequence_range in SPLIT_RANGES.items():
        image_paths: list[Path] = []

        if args.include_unlabeled:
            for sequence_number in sequence_range:
                sequence_dir = images_root / str(sequence_number)
                if not sequence_dir.exists():
                    continue
                image_paths.extend(
                    path
                    for path in sequence_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                )
        else:
            for sequence_number in sequence_range:
                sequence_id = str(sequence_number)
                sequence_labels = labels_root / sequence_id
                if not sequence_labels.exists():
                    continue

                for label_path in sequence_labels.glob("*.txt"):
                    if label_path.stat().st_size == 0:
                        continue
                    unknown = class_ids_in_label(label_path) - allowed_class_ids
                    if unknown:
                        raise ValueError(
                            f"{label_path}에 선택한 class 구성과 맞지 않는 ID가 있습니다: "
                            f"{sorted(unknown)}"
                        )
                    image_path = find_image(images_root, sequence_id, label_path.stem)
                    if image_path is None:
                        raise FileNotFoundError(
                            f"라벨과 짝이 되는 이미지가 없습니다: {label_path}"
                        )
                    image_paths.append(image_path)

        image_paths = sorted(set(image_paths))
        list_path = splits_root / f"{split_name}.txt"
        list_path.write_text(
            "".join(f"{path.resolve().as_posix()}\n" for path in image_paths),
            encoding="utf-8",
        )
        counts[split_name] = len(image_paths)
        print(f"{split_name}: {len(image_paths):,}장 → {list_path}")

    if counts.get("train", 0) == 0 or counts.get("val", 0) == 0:
        raise RuntimeError("train 또는 val 목록이 비어 있습니다. 경로와 변환 결과를 확인하세요.")

    yaml_data = {
        "path": root.as_posix(),
        "train": "splits/train.txt",
        "val": "splits/val.txt",
        "test": "splits/eval.txt",
        "names": CLASS_NAMES[args.classes],
    }
    yaml_path = root / f"bupst20_{args.classes}_class.yaml"
    yaml_path.write_text(
        yaml.safe_dump(yaml_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"dataset YAML 생성: {yaml_path}")


if __name__ == "__main__":
    main()

