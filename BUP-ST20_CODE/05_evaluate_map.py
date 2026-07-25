"""학습된 best.pt의 val/eval mAP를 계산하고 JSON으로 저장."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="test가 BUP-ST20 공식 eval split",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("map_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    data = args.data.resolve()
    if not weights.exists():
        raise FileNotFoundError(weights)
    if not data.exists():
        raise FileNotFoundError(data)

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=True,
        save_json=True,
    )

    result = {
        "weights": str(weights),
        "data": str(data),
        "split": args.split,
        "mAP50_95": float(metrics.box.map),
        "mAP50": float(metrics.box.map50),
        "mAP75": float(metrics.box.map75),
        "precision_mean": float(metrics.box.mp),
        "recall_mean": float(metrics.box.mr),
        "mAP50_95_per_class": [float(value) for value in metrics.box.maps],
        "class_names": {str(key): value for key, value in model.names.items()},
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"mAP50-95: {result['mAP50_95']:.4f}")
    print(f"mAP50:    {result['mAP50']:.4f}")
    print(f"mAP75:    {result['mAP75']:.4f}")
    print(f"Precision: {result['precision_mean']:.4f}")
    print(f"Recall:    {result['recall_mean']:.4f}")
    print(f"저장: {output}")


if __name__ == "__main__":
    main()

