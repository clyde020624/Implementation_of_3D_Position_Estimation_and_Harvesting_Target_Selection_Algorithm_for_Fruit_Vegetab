# -*- coding: utf-8 -*-
"""
====================================================================
 list_sequences.py - 데이터셋에 어떤 시퀀스가 있는지 확인
====================================================================

extract_bulk.py로 추출하기 전에, annotation 아카이브를 훑어서
  - 어떤 시퀀스 번호가 존재하는지
  - 각 시퀀스에 프레임이 몇 개인지
를 확인합니다. (파일을 꺼내지 않고 목록만 읽으므로 빠름)

BUP-ST20 공식 split (민혁 03번 스크립트 기준):
  train = 100~226,  val = 300~371,  eval = 400~475

실행: python list_sequences.py
====================================================================
"""

import os
import tarfile
from collections import Counter
from pathlib import PurePosixPath


# ★ 실제 경로로 ★
ANN_ARCHIVE = r"C:\Users\82109\Downloads\bupst20_annotations (1).tar.gz"


def seq_of(member_name):
    """'.../annotations/400/16009368xxxxx.pkl' → '400'"""
    parts = PurePosixPath(member_name).parts
    if len(parts) < 2:
        return None
    return parts[-2]


def main():
    if not os.path.exists(ANN_ARCHIVE):
        print(f"❌ 파일 없음: {ANN_ARCHIVE}")
        return

    mode = "r:gz" if ANN_ARCHIVE.endswith((".gz", ".tgz")) else "r"
    counter = Counter()

    print("아카이브를 훑는 중... (파일을 꺼내지 않고 목록만 읽습니다)\n")
    with tarfile.open(ANN_ARCHIVE, mode) as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".pkl"):
                continue
            s = seq_of(member.name)
            if s:
                counter[s] += 1

    if not counter:
        print("시퀀스를 찾지 못했습니다. 경로/형식을 확인하세요.")
        return

    # 숫자 시퀀스만 정렬
    def key(s):
        return int(s) if s.isdigit() else 10**9

    print("=" * 56)
    print(f" 발견된 시퀀스: {len(counter)}개 "
          f"(총 프레임 {sum(counter.values()):,}개)")
    print("=" * 56)

    groups = {"train(100~226)": [], "val(300~371)": [],
              "eval(400~475)": [], "기타": []}
    for s in sorted(counter, key=key):
        n = counter[s]
        if s.isdigit():
            v = int(s)
            if 100 <= v <= 226:
                groups["train(100~226)"].append((s, n))
            elif 300 <= v <= 371:
                groups["val(300~371)"].append((s, n))
            elif 400 <= v <= 475:
                groups["eval(400~475)"].append((s, n))
            else:
                groups["기타"].append((s, n))
        else:
            groups["기타"].append((s, n))

    for name, items in groups.items():
        if not items:
            continue
        total = sum(n for _, n in items)
        print(f"\n[{name}]  시퀀스 {len(items)}개 / 프레임 {total:,}개")
        line = "   "
        for s, n in items:
            piece = f"{s}({n})  "
            if len(line) + len(piece) > 74:
                print(line); line = "   "
            line += piece
        if line.strip():
            print(line)

    # eval 추천
    ev = groups["eval(400~475)"]
    if ev:
        top = sorted(ev, key=lambda t: -t[1])[:3]
        print("\n" + "=" * 56)
        print(" 추천: 프레임이 많은 eval 시퀀스 3개")
        print("=" * 56)
        print(f'   TARGET_SEQUENCES = {[s for s, _ in top]}')
        for s, n in top:
            print(f"     시퀀스 {s}: {n}프레임")
    else:
        print("\n⚠️ eval(400~475) 시퀀스를 찾지 못했습니다.")
        print("   민혁님 CSV의 sequence_id와 맞는지 확인이 필요합니다.")


if __name__ == "__main__":
    main()
