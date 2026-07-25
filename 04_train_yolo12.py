"""YOLO12 Detection 모델을 BUP-ST20으로 학습한다."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, help="03번에서 만든 data.yaml")
    parser.add_argument("--model", default="yolo12n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=float,
        default=-1,
        help="-1이면 GPU 메모리에 맞춰 자동 결정",
    )
    parser.add_argument("--device", default="0", help="GPU 0은 0, CPU는 cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("runs/bupst20"))
    parser.add_argument("--name", default="yolo12n_100e")
    parser.add_argument("--resume", type=Path, help="중단된 runs/.../weights/last.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.resume:
        checkpoint = args.resume.resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        print(f"중단 지점부터 재개: {checkpoint}")
        YOLO(str(checkpoint)).train(resume=True)
        return

    if args.data is None:
        raise ValueError("새 학습에는 --data 경로가 필요합니다.")
    data_path = args.data.resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        seed=42,
        deterministic=True,
        cache=False,
        plots=True,
        save=True,
    )

    expected_best = args.project.resolve() / args.name / "weights" / "best.pt"
    print(f"학습 완료. best.pt 예상 위치: {expected_best}")


if __name__ == "__main__":
    # Windows multiprocessing 오류 방지를 위해 반드시 이 블록 안에서 실행한다.
    main()

