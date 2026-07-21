# -*- coding: utf-8 -*-
"""
====================================================================
 7단계: 평가 코드 (Top-1 Accuracy)
====================================================================

Priority Score 알고리즘이 고른 1순위(Top-1)가
사람이 정한 정답 1순위와 얼마나 자주 일치하는지 측정합니다.

이것이 이 연구의 '핵심 성과 지표'입니다.
  Top-1 Accuracy = (알고리즘과 사람이 일치한 이미지 수) / (전체 이미지 수)

또한 가중치(weights)를 바꿔가며 어떤 조합이 사람 판단과
가장 잘 맞는지 찾는 실험도 여기서 합니다. (논문의 실험 결과 파트)

실행: python evaluate.py
====================================================================
"""

import numpy as np
from priority import select_top1, WEIGHTS   # 우리가 만든 알고리즘 재사용


# ====================================================================
# 1) Top-1 Accuracy 계산
# --------------------------------------------------------------------
# dataset: 이미지별 정보 리스트. 각 원소는 이런 형태:
#   {
#     "image_id": "img_001",
#     "objects": [...],           # 그 이미지의 검출 객체들
#     "depth_map": np.array,      # 그 이미지의 depth 맵
#     "img_size": (W, H),
#     "human_top1_id": 3,         # 사람이 고른 정답 1순위 객체 id
#   }
# ====================================================================
def evaluate_top1(dataset, weights, mode):
    correct = 0
    total = 0
    wrong_cases = []   # 틀린 경우 기록 (실패 케이스 분석용)

    for sample in dataset:
        pred_top1, scored = select_top1(
            sample["objects"], sample["depth_map"],
            weights, sample["img_size"], mode
        )
        if pred_top1 is None:
            continue   # 검출 객체가 없는 이미지는 건너뜀

        total += 1
        if pred_top1["id"] == sample["human_top1_id"]:
            correct += 1
        else:
            wrong_cases.append({
                "image_id": sample["image_id"],
                "예측": pred_top1["id"],
                "정답": sample["human_top1_id"],
            })

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, wrong_cases


# ====================================================================
# 2) 가중치 실험: 여러 조합을 돌려 최적을 찾음 (논문 실험 파트)
# ====================================================================
def weight_experiment(dataset, mode, weight_candidates):
    print("=" * 62)
    print(f" 가중치 실험 (MODE = '{mode}')")
    print("=" * 62)
    results = []
    for name, w in weight_candidates.items():
        acc, wrong = evaluate_top1(dataset, w, mode)
        results.append((name, acc, len(wrong)))
        print(f"  [{name}]  Top-1 Accuracy = {acc:.1%}  (틀림 {len(wrong)}건)")
        print(f"      가중치: {w}")

    # 가장 높은 정확도 조합 찾기
    best = max(results, key=lambda r: r[1])
    print("-" * 62)
    print(f"  >>> 최적 조합: [{best[0]}]  정확도 {best[1]:.1%} <<<")
    return results


# ====================================================================
# 3) 더미 데이터셋으로 평가 시연
#    (실제로는 data_loader.py로 불러온 여러 이미지가 들어감)
# ====================================================================
def make_dummy_dataset(n_images=10, mode="ripeness"):
    """평가 코드가 잘 도는지 보여주기 위한 가짜 데이터셋 생성"""
    np.random.seed(0)
    dataset = []
    for i in range(n_images):
        depth = np.random.randint(50, 200, (480, 640))
        objects = [
            {"id": 1, "bbox": [100, 50, 180, 140],
             "confidence": round(np.random.uniform(0.7, 0.95), 2),
             "class": np.random.choice(["ripe", "unripe"]),
             "mask_area": np.random.randint(3000, 7000)},
            {"id": 2, "bbox": [300, 200, 360, 270],
             "confidence": round(np.random.uniform(0.7, 0.95), 2),
             "class": np.random.choice(["ripe", "unripe"]),
             "mask_area": np.random.randint(2000, 6000)},
            {"id": 3, "bbox": [380, 100, 460, 190],
             "confidence": round(np.random.uniform(0.7, 0.95), 2),
             "class": np.random.choice(["ripe", "unripe"]),
             "mask_area": np.random.randint(3000, 7000)},
        ]
        # 가짜 '사람 정답' (실제로는 사람이 직접 라벨링)
        human_top1 = int(np.random.choice([1, 2, 3]))
        dataset.append({
            "image_id": f"img_{i:03d}",
            "objects": objects,
            "depth_map": depth,
            "img_size": (640, 480),
            "human_top1_id": human_top1,
        })
    return dataset


if __name__ == "__main__":
    MODE = "ripeness"   # priority.py와 동일하게 맞추기

    # 더미 데이터셋 생성 (실제로는 data_loader로 불러옴)
    dataset = make_dummy_dataset(n_images=10, mode=MODE)

    # --- (A) 기본 가중치로 한 번 평가 ---
    acc, wrong = evaluate_top1(dataset, WEIGHTS[MODE], MODE)
    print("=" * 62)
    print(f" 기본 평가 결과 (MODE = '{MODE}')")
    print("=" * 62)
    print(f"  Top-1 Accuracy = {acc:.1%}")
    print(f"  틀린 케이스 {len(wrong)}건:")
    for w in wrong:
        print(f"    {w['image_id']}: 예측 {w['예측']} vs 정답 {w['정답']}")
    print()

    # --- (B) 여러 가중치 조합 실험 (논문 실험 파트) ---
    candidates = {
        "거리중시":   {"proximity": 0.5, "confidence": 0.2, "center": 0.1, "ripeness": 0.2},
        "균형":       {"proximity": 0.4, "confidence": 0.2, "center": 0.2, "ripeness": 0.2},
        "성숙도중시": {"proximity": 0.2, "confidence": 0.2, "center": 0.2, "ripeness": 0.4},
    }
    weight_experiment(dataset, MODE, candidates)

    # [참고] 더미는 정답이 랜덤이라 정확도가 낮게 나오는 게 정상입니다.
    #        실제 데이터에서는 사람 판단에 규칙성이 있어 훨씬 높게 나옵니다.
