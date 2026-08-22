# -*- coding: utf-8 -*-
"""
====================================================================
 check_pkl_structure.py - eval 시퀀스 pkl 구조 진단
====================================================================
IndexError 원인을 찾기 위해 pkl 안의 bbox 형태를 확인합니다.
train(100번대)과 eval(400번대) 구조가 다를 수 있습니다.

실행: python check_pkl_structure.py
====================================================================
"""

import os
import glob
import pickle
import numpy as np

DATA_DIR = "dataset_bulk"


def inspect(pkl_path, label):
    print(f"\n{'='*60}")
    print(f" [{label}] {os.path.basename(pkl_path)}")
    print('='*60)
    with open(pkl_path, "rb") as f:
        ann = pickle.load(f)

    print(f"  타입: {type(ann).__name__}, 객체 수: {len(ann)}")
    if not ann:
        print("  ⚠️ 비어 있음 (객체 0개)")
        return

    # 객체 몇 개만 상세히
    for i, (oid, rec) in enumerate(list(ann.items())[:3]):
        print(f"\n  [객체 {i+1}] id={oid} ({type(oid).__name__})")
        if not isinstance(rec, dict):
            print(f"    ⚠️ dict 아님: {type(rec)}")
            continue
        print(f"    키: {list(rec.keys())}")
        bbox = rec.get("bbox")
        print(f"    bbox 값 : {bbox}")
        print(f"    bbox 타입: {type(bbox).__name__}", end="")
        if hasattr(bbox, "shape"):
            print(f", shape={bbox.shape}")
        elif bbox is not None:
            print(f", 길이={len(bbox)}")
        else:
            print(" (None!)")
        for k in ("area", "semantic_label"):
            if k in rec:
                print(f"    {k}: {rec[k]}")
        m = rec.get("instance_mask")
        if m is not None:
            print(f"    instance_mask: shape={getattr(m,'shape',None)}, "
                  f"True={np.count_nonzero(m):,}")


def scan_all(seq_dir, label, max_files=200):
    """시퀀스 전체를 훑어 bbox 길이가 4가 아닌 것 찾기"""
    files = sorted(glob.glob(os.path.join(seq_dir, "*.pkl")))[:max_files]
    bad = []
    empty = 0
    lengths = {}
    for p in files:
        try:
            with open(p, "rb") as f:
                ann = pickle.load(f)
        except Exception as e:
            bad.append((p, f"읽기실패: {e}"))
            continue
        if len(ann) == 0:
            empty += 1
            continue
        for oid, rec in ann.items():
            b = rec.get("bbox") if isinstance(rec, dict) else None
            n = (len(b) if b is not None and hasattr(b, "__len__") else -1)
            lengths[n] = lengths.get(n, 0) + 1
            if n != 4:
                bad.append((os.path.basename(p), f"id={oid}, bbox={b}"))

    print(f"\n{'='*60}")
    print(f" [{label}] 전체 스캔: {len(files)}개 파일")
    print('='*60)
    print(f"  객체 0개인 파일: {empty}개")
    print(f"  bbox 길이 분포 : {lengths}   (정상은 4)")
    if bad:
        print(f"  ⚠️ 문제 객체 {len(bad)}건 — 앞 5건:")
        for item in bad[:5]:
            print(f"     {item}")
    else:
        print("  ✅ 모든 bbox가 정상(길이 4)")


if __name__ == "__main__":
    ann_root = os.path.join(DATA_DIR, "annotations")
    if not os.path.isdir(ann_root):
        print(f"'{ann_root}' 없음")
    else:
        seqs = sorted(os.listdir(ann_root))
        print(f"발견된 시퀀스: {seqs}")
        for seq in seqs:
            seq_dir = os.path.join(ann_root, seq)
            files = sorted(glob.glob(os.path.join(seq_dir, "*.pkl")))
            if not files:
                print(f"\n[시퀀스 {seq}] pkl 없음")
                continue
            inspect(files[0], f"시퀀스 {seq} 첫 파일")
            scan_all(seq_dir, f"시퀀스 {seq}")
