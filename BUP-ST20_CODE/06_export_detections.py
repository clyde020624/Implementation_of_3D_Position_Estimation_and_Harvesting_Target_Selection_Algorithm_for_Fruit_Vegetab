"""eval 이미지의 모든 검출 후보를 알고리즘 팀 전달용 CSV로 저장."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


FIELDNAMES = [
    "sequence_id",
    "frame_id",
    "image_path",
    "candidate_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "box_width",
    "box_height",
    "image_width",
    "image_height",
]
FRAME_FIELDNAMES = [
    "sequence_id",
    "frame_id",
    "image_path",
    "detection_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("eval_detections.csv"))
    parser.add_argument(
        "--frames-output",
        type=Path,
        help="생략하면 eval_detections_frames.csv처럼 자동 생성",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="후보를 너무 일찍 버리지 않도록 낮게 저장하고 후속 알고리즘에서 필터링",
    )
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="한 번에 GPU로 전달할 이미지 수(RTX 3080 10GB 권장값: 8)",
    )
    return parser.parse_args()


def sequence_and_frame(image_path: Path) -> tuple[str, str]:
    return image_path.parent.name, image_path.stem


def predict_in_batches(model: YOLO, image_paths: list[str], args: argparse.Namespace):
    """경로 목록 전체가 한 배치로 적재되지 않도록 명시적으로 나눠 추론한다."""
    for start in range(0, len(image_paths), args.batch):
        batch_paths = image_paths[start : start + args.batch]
        batch_results = model.predict(
            source=batch_paths,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            batch=len(batch_paths),
            stream=False,
            verbose=False,
        )
        yield from batch_results


def main() -> None:
    args = parse_args()
    if args.batch <= 0:
        raise ValueError(f"--batch는 1 이상의 정수여야 합니다: {args.batch}")

    weights = args.weights.resolve()
    image_list = args.image_list.resolve()
    if not weights.exists():
        raise FileNotFoundError(weights)
    if not image_list.exists():
        raise FileNotFoundError(image_list)

    image_paths = [
        line.strip()
        for line in image_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not image_paths:
        raise ValueError(f"이미지 목록이 비었습니다: {image_list}")

    model = YOLO(str(weights))
    results = predict_in_batches(model, image_paths, args)

    output = args.output.resolve()
    frames_output = (
        args.frames_output.resolve()
        if args.frames_output
        else output.with_name(f"{output.stem}_frames.csv")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frames_output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    detection_count = 0

    with (
        output.open("w", newline="", encoding="utf-8-sig") as csv_file,
        frames_output.open("w", newline="", encoding="utf-8-sig") as frames_file,
    ):
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        frames_writer = csv.DictWriter(frames_file, fieldnames=FRAME_FIELDNAMES)
        writer.writeheader()
        frames_writer.writeheader()

        for result in results:
            image_path = Path(result.path).resolve()
            sequence_id, frame_id = sequence_and_frame(image_path)
            image_height, image_width = result.orig_shape
            boxes = result.boxes
            current_detection_count = 0

            if boxes is not None:
                xyxy_values = boxes.xyxy.cpu().tolist()
                confidence_values = boxes.conf.cpu().tolist()
                class_values = boxes.cls.cpu().tolist()

                for candidate_id, (xyxy, confidence, class_value) in enumerate(
                    zip(xyxy_values, confidence_values, class_values)
                ):
                    x1, y1, x2, y2 = (float(value) for value in xyxy)
                    class_id = int(class_value)
                    writer.writerow(
                        {
                            "sequence_id": sequence_id,
                            "frame_id": frame_id,
                            "image_path": image_path.as_posix(),
                            "candidate_id": candidate_id,
                            "class_id": class_id,
                            "class_name": model.names[class_id],
                            "confidence": f"{float(confidence):.8f}",
                            "x1": f"{x1:.3f}",
                            "y1": f"{y1:.3f}",
                            "x2": f"{x2:.3f}",
                            "y2": f"{y2:.3f}",
                            "center_x": f"{(x1 + x2) / 2.0:.3f}",
                            "center_y": f"{(y1 + y2) / 2.0:.3f}",
                            "box_width": f"{x2 - x1:.3f}",
                            "box_height": f"{y2 - y1:.3f}",
                            "image_width": image_width,
                            "image_height": image_height,
                        }
                    )
                    detection_count += 1
                    current_detection_count += 1

            frames_writer.writerow(
                {
                    "sequence_id": sequence_id,
                    "frame_id": frame_id,
                    "image_path": image_path.as_posix(),
                    "detection_count": current_detection_count,
                }
            )

            frame_count += 1
            if frame_count % 500 == 0:
                print(f"{frame_count:,}프레임 / {detection_count:,}개 후보 저장")

    print(f"완료: {frame_count:,}프레임, {detection_count:,}개 후보")
    print(f"알고리즘 팀 전달 파일: {output}")
    print(f"0개 검출 프레임까지 포함한 목록: {frames_output}")


if __name__ == "__main__":
    main()