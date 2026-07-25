r"""분할된 BUP-ST20 RGB/Depth tar.gz를 합친 파일 없이 순차 해제.

예:
python 01_extract_archive_parts.py ^
  --parts-dir D:\BUP-ST20_DOWNLOAD ^
  --pattern "bupst20_rgb.tar.gz.*" ^
  --modality rgb ^
  --dataset-root D:\BUPST20_YOLO
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath


IMAGE_SUFFIXES = {".tif", ".tiff"}


class SplitFileReader(io.RawIOBase):
    """여러 분할 파일을 하나의 연속된 읽기 스트림처럼 제공."""

    def __init__(self, paths: list[Path]) -> None:
        super().__init__()
        if not paths:
            raise ValueError("분할 파일이 없습니다.")
        self.paths = paths
        self.index = 0
        self.current = paths[0].open("rb")

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        if self.index >= len(self.paths):
            return 0

        view = memoryview(buffer)
        total = 0

        while total < len(view):
            count = self.current.readinto(view[total:])
            if count:
                total += count
                continue

            self.current.close()
            self.index += 1
            if self.index >= len(self.paths):
                break
            self.current = self.paths[self.index].open("rb")

        return total

    def close(self) -> None:
        if not self.closed and not self.current.closed:
            self.current.close()
        super().close()


def relative_after_modality(member_name: str, modality: str) -> Path | None:
    """archive/.../rgb/100/frame.tiff에서 100/frame.tiff를 얻는다."""
    parts = PurePosixPath(member_name).parts
    lowered = [part.lower() for part in parts]
    if modality not in lowered:
        return None

    modality_index = lowered.index(modality)
    relative_parts = parts[modality_index + 1 :]
    if len(relative_parts) < 2:
        return None
    return Path(*relative_parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument(
        "--pattern",
        required=True,
        help='예: "bupst20_rgb.tar.gz.*" 또는 "bupst20_depth.tar.gz.*"',
    )
    parser.add_argument("--modality", choices=["rgb", "depth"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    part_paths = sorted(args.parts_dir.glob(args.pattern))
    if not part_paths:
        raise FileNotFoundError(
            f"{args.parts_dir}에서 {args.pattern!r}에 해당하는 파일을 찾지 못했습니다."
        )

    output_name = "images" if args.modality == "rgb" else "depth"
    output_root = args.dataset_root.resolve() / output_name
    output_root.mkdir(parents=True, exist_ok=True)

    print("읽을 분할 파일:")
    for path in part_paths:
        print(f"  {path.name}")
    print(f"출력 폴더: {output_root}")

    extracted = 0
    skipped = 0
    with SplitFileReader(part_paths) as raw:
        with io.BufferedReader(raw, buffer_size=1024 * 1024) as combined:
            with tarfile.open(fileobj=combined, mode="r|gz") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    relative = relative_after_modality(member.name, args.modality)
                    if relative is None or relative.suffix.lower() not in IMAGE_SUFFIXES:
                        continue

                    # 상대경로 이탈(path traversal)을 차단한다.
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"안전하지 않은 archive 경로: {member.name}")

                    target = output_root / relative
                    if target.exists() and not args.overwrite:
                        skipped += 1
                        continue

                    source = archive.extractfile(member)
                    if source is None:
                        continue

                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(target.suffix + ".part")
                    with source, temporary.open("wb") as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
                    os.replace(temporary, target)

                    extracted += 1
                    if extracted % 500 == 0:
                        print(f"{extracted:,}개 해제 완료")

    print(f"완료: 새로 해제 {extracted:,}개, 기존 파일 건너뜀 {skipped:,}개")


if __name__ == "__main__":
    main()
