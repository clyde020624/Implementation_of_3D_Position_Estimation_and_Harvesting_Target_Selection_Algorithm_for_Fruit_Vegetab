"""학습 전에 Python, PyTorch, CUDA, Ultralytics 설치 상태를 확인한다."""

from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"OS: {platform.platform()}")

    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU 개수: {torch.cuda.device_count()}")
            for index in range(torch.cuda.device_count()):
                print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
    except ImportError:
        print("PyTorch: 설치되지 않음")

    try:
        import ultralytics

        print(f"Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("Ultralytics: 설치되지 않음")


if __name__ == "__main__":
    main()

