# -*- coding: utf-8 -*-
"""
====================================================================
 extract_bulk.py - 논문용 데이터 대량 추출
====================================================================

일관성 분석을 통계적으로 의미있게 하려면 프레임이 많아야 합니다.
이 스크립트는 지정한 시퀀스의 depth와 pkl을 짝 맞춰 대량 추출합니다.

[전략]
  - 여러 시퀀스에서 조금씩(X) → 한 시퀀스에서 많이(O)
  - 같은 파프리카를 여러 프레임에서 봐야 일관성 측정 가능
  - 시퀀스 2~3개면 시퀀스 간 비교도 가능

[설정] 아래 TARGET_SEQUENCES, MAX_PER_SEQ 를 조절
====================================================================
"""

import io
import os
import re
import tarfile
import zipfile


# ★ 실제 경로로 ★
ANN_ARCHIVE = r"C:\Users\82109\Downloads\bupst20_annotations (1).tar.gz"
DEPTH_ZIP   = r"C:\Users\82109\Downloads\bupst20_depth.tar.gz.zip"

OUT_DIR = "dataset_bulk"          # 추출 폴더
TARGET_SEQUENCES = ["100", "101", "102"]   # 뽑을 시퀀스 (원하는 만큼)
MAX_PER_SEQ = 40                   # 시퀀스당 최대 프레임 수


# ====================================================================
# zip 안 분할 파일을 이어 읽는 스트림
# ====================================================================
class ZipSplitReader(io.RawIOBase):
    def __init__(self, zip_path, member_names):
        super().__init__()
        self.zf = zipfile.ZipFile(zip_path)
        self.names = member_names
        self.index = 0
        self.current = self.zf.open(self.names[0])

    def readable(self):
        return True

    def readinto(self, buffer):
        view = memoryview(buffer)
        total = 0
        while total < len(view):
            n = self.current.readinto(view[total:])
            if n:
                total += n
                continue
            self.current.close()
            self.index += 1
            if self.index >= len(self.names):
                break
            self.current = self.zf.open(self.names[self.index])
        return total

    def close(self):
        try:
            if not self.current.closed:
                self.current.close()
            self.zf.close()
        except Exception:
            pass
        super().close()


def seq_and_ts(member_name):
    """'.../depth/100/1600936832187439.tiff' → ('100','1600936832187439')"""
    parts = member_name.replace("\\", "/").split("/")
    if len(parts) < 2:
        return None, None
    ts = os.path.splitext(parts[-1])[0]
    seq = parts[-2]
    return seq, ts


# ====================================================================
# 1) annotation(pkl) 추출 — 어떤 프레임이 있는지 먼저 파악
# ====================================================================
def extract_annotations():
    print("=" * 62)
    print(" [1/2] annotation(pkl) 추출")
    print("=" * 62)

    found = {s: [] for s in TARGET_SEQUENCES}
    mode = "r:gz" if ANN_ARCHIVE.endswith((".gz", ".tgz")) else "r"

    with tarfile.open(ANN_ARCHIVE, mode) as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".pkl"):
                continue
            seq, ts = seq_and_ts(member.name)
            if seq not in found:
                continue
            if len(found[seq]) >= MAX_PER_SEQ:
                # 목표 시퀀스를 모두 채웠으면 종료
                if all(len(v) >= MAX_PER_SEQ for v in found.values()):
                    break
                continue

            out_dir = os.path.join(OUT_DIR, "annotations", seq)
            os.makedirs(out_dir, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with src, open(os.path.join(out_dir, ts + ".pkl"), "wb") as dst:
                dst.write(src.read())
            found[seq].append(ts)

            total = sum(len(v) for v in found.values())
            if total % 20 == 0:
                print(f"   추출 {total}개...")

    for s, lst in found.items():
        print(f"   시퀀스 {s}: pkl {len(lst)}개")
    return found


# ====================================================================
# 2) depth 추출 — pkl과 같은 프레임만
# ====================================================================
def extract_depths(wanted):
    """wanted: {시퀀스: [타임스탬프,...]} — 이 프레임의 depth만 꺼냄"""
    print("\n" + "=" * 62)
    print(" [2/2] depth(tiff) 추출 — pkl과 같은 프레임만")
    print("=" * 62)

    want_set = {(s, ts) for s, lst in wanted.items() for ts in lst}
    got = 0
    need = len(want_set)

    with zipfile.ZipFile(DEPTH_ZIP) as z:
        parts = sorted(n for n in z.namelist() if not n.endswith("/"))
    reader = ZipSplitReader(DEPTH_ZIP, parts)

    try:
        buffered = io.BufferedReader(reader, buffer_size=1024 * 1024)
        tar = tarfile.open(fileobj=buffered, mode="r|gz")
        try:
            for member in tar:
                if not member.isfile():
                    continue
                if not member.name.lower().endswith((".tiff", ".tif")):
                    continue
                seq, ts = seq_and_ts(member.name)
                if (seq, ts) not in want_set:
                    continue

                out_dir = os.path.join(OUT_DIR, "depth", seq)
                os.makedirs(out_dir, exist_ok=True)
                src = tar.extractfile(member)
                if src is None:
                    continue
                with src, open(os.path.join(out_dir, ts + ".tiff"), "wb") as dst:
                    dst.write(src.read())
                got += 1
                if got % 20 == 0:
                    print(f"   추출 {got}/{need}...")
                if got >= need:
                    print("\n   (필요한 depth를 모두 얻어 중단)")
                    break
        finally:
            try: tar.close()
            except Exception: pass
            try: buffered.close()
            except Exception: pass
    finally:
        try: reader.close()
        except Exception: pass

    print(f"   depth 총 {got}개 추출")
    return got


if __name__ == "__main__":
    print(f"목표: 시퀀스 {TARGET_SEQUENCES}, 각 최대 {MAX_PER_SEQ}프레임\n")
    found = extract_annotations()
    n = extract_depths(found)

    print("\n" + "=" * 62)
    print(" 완료")
    print("=" * 62)
    print(f" 저장 위치: {OUT_DIR}/")
    print(f"   annotations/<시퀀스>/<타임스탬프>.pkl")
    print(f"   depth/<시퀀스>/<타임스탬프>.tiff")
    print("\n 이제 run_bulk.py 로 대량 분석을 실행하세요.")
