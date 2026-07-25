# BUP-ST20 → YOLO12 Detection 연구 코드

이 프로젝트 작업.

1. BUP-ST20 RGB 분할 압축 해제
2. BUP-ST20 `pkl` annotation의 bbox를 YOLO Detection 라벨로 변환
3. 공식 sequence 단위 train/valid/eval 분할 생성
4. YOLO12 Detection 학습
5. mAP50, mAP50-95 평가
6. Priority Score 알고리즘에 넘길 모든 검출 후보를 CSV로 저장

세그멘테이션 학습은 포함하지 않음. annotation 안의 `instance_mask`는
pickle을 읽는 순간 메모리에는 올라오지만 파일로 풀지 않고 바로 버림.

## 0. 중요한 원칙

- 기본 설정은 파프리카 한 클래스(`pepper`) 검출.
- `train=100~226`, `valid=300~371`, `eval=400~475` sequence를 유지.
- 프레임을 무작위로 섞어 나누면 비슷한 연속 프레임이 train/eval에 동시에
  들어가므로 성능이 과대평가될 수 있음.
- 공식 데이터에서 `pkl`이 없거나 비어 있으면 “정답 없는 이미지”일 수
  있으므로 기본 설정에서는 학습 목록에서 제외.
- eval mAP는 모델과 설정을 확정한 뒤 마지막에 확인.

## 1. 준비

Windows PowerShell 또는 VS Code 터미널에서 프로젝트 폴더로 이동.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python 00_check_environment.py
```

`CUDA 사용 가능: True`와 RTX GPU 이름이 나오면 GPU 학습 준비가 된 것.

## 2. 권장 폴더

```text
D:\BUP_ST20_DOWNLOAD\
  bupst20_rgb.tar.gz.aa
  ...
  bupst20_rgb.tar.gz.ag
  bupst20_annotations.tar.gz

D:\BUPST20_YOLO\
  images\
  labels\
  metadata\
  splits\
  bupst20_one_class.yaml
```

다운로드 파일명에 약간 차이가 있으면 아래 명령의 `--pattern`과
`--archive`만 실제 파일명으로 바꿈.

## 3. RGB 해제

7개 RGB 조각을 합친 27GB짜리 중간 파일을 만들지 않고 곧바로 `images`에
해제.

```powershell
python 01_extract_archive_parts.py `
  --parts-dir "D:\BUP_ST20_DOWNLOAD" `
  --pattern "bupst20_rgb.tar.gz.*" `
  --modality rgb `
  --dataset-root "D:\BUPST20_YOLO"
```

Depth도 필요할 때만 아래처럼 실행. YOLO Detection 학습 자체에는
Depth가 필요하지 않음.

```powershell
python 01_extract_archive_parts.py `
  --parts-dir "D:\BUP_ST20_DOWNLOAD" `
  --pattern "bupst20_depth.tar.gz.*" `
  --modality depth `
  --dataset-root "D:\BUPST20_YOLO"
```

중간에 중단되면 같은 명령을 다시 실행해도 이미 완료된 파일은 건너뜀.

## 4. bbox 라벨 변환

기본 이미지 크기는 BUP-ST20의 세로형 RGB인 `720 × 1280
(width × height)`로 설정했습니다. 실제 이미지 한 장의 크기가 다르면
반드시 인수를 수정.

한 클래스 검출:

```powershell
python 02_convert_annotations.py `
  --archive "D:\BUP_ST20_DOWNLOAD\bupst20_annotations.tar.gz" `
  --dataset-root "D:\BUPST20_YOLO" `
  --classes one `
  --image-width 720 `
  --image-height 1280
```

색상/숙도 5개 클래스를 유지하려면 `--classes five`를 사용. 연구의
핵심이 “파프리카 후보 검출 후 수확 우선순위 선정”이라면 먼저 한 클래스를
권장.

이 스크립트가 만드는 라벨 한 줄:

```text
class_id x_center y_center width height
```

네 좌표는 모두 이미지 크기로 나눈 `0~1` 값.

## 5. 공식 split 및 YAML 생성

```powershell
python 03_build_dataset.py `
  --dataset-root "D:\BUPST20_YOLO" `
  --classes one
```

출력:

- `splits/train.txt`
- `splits/val.txt`
- `splits/eval.txt`
- `bupst20_one_class.yaml`

## 6. 먼저 1 epoch 시험

전체 100 epoch 전에 경로와 메모리 문제가 없는지 먼저 확인.

```powershell
python 04_train_yolo12.py `
  --data "D:\BUPST20_YOLO\bupst20_one_class.yaml" `
  --epochs 1 `
  --batch -1 `
  --device 0 `
  --name smoke_test
```

문제가 없다면 본 학습:

```powershell
python 04_train_yolo12.py `
  --data "D:\BUPST20_YOLO\bupst20_one_class.yaml" `
  --model yolo12n.pt `
  --epochs 100 `
  --imgsz 640 `
  --batch -1 `
  --device 0 `
  --name yolo12n_100e
```

RTX 3080에서 메모리 오류가 발생하면 자동 batch 대신 `--batch 8`, 그래도
부족하면 `--batch 4`를 사용. 모델 크기를 키우기 전에 `yolo12n`으로
전체 파이프라인을 먼저 완성하는 것이 좋음.

중단된 학습 재개:

```powershell
python 04_train_yolo12.py `
  --resume "runs\bupst20\yolo12n_100e\weights\last.pt"
```

주요 결과:

- `weights/best.pt`: validation 성능이 가장 좋았던 모델
- `weights/last.pt`: 마지막 epoch 모델 및 재개용 상태
- `results.csv`: epoch별 loss와 지표
- `results.png`: 학습 곡선
- PR curve, confusion matrix 등

## 7. 공식 eval에서 mAP 측정

```powershell
python 05_evaluate_map.py `
  --weights "runs\bupst20\yolo12n_100e\weights\best.pt" `
  --data "D:\BUPST20_YOLO\bupst20_one_class.yaml" `
  --split test `
  --output "runs\bupst20\yolo12n_100e\eval_metrics.json"
```

보고할 핵심 수치:

- `mAP50`: IoU 0.5에서의 평균 정밀도
- `mAP50-95`: IoU 0.5부터 0.95까지 평균한 더 엄격한 지표
- Precision
- Recall

`--split test`가 YAML의 `test`, 즉 BUP-ST20 공식 eval sequence를 뜻함.

## 8. 검출 CSV 전달

```powershell
python 06_export_detections.py `
  --weights "runs\bupst20\yolo12n_100e\weights\best.pt" `
  --image-list "D:\BUPST20_YOLO\splits\eval.txt" `
  --output "runs\bupst20\yolo12n_100e\eval_detections.csv" `
  --conf 0.05
```

CSV에는 각 후보의 다음 정보가 들어감.

- `sequence_id`, `frame_id`
- bbox `x1, y1, x2, y2`
- `confidence`
- bbox 중심과 크기
- 원본 이미지 크기

같이 생성되는 `eval_detections_frames.csv`에는 검출이 0개인 프레임도
기록. Top-1 Accuracy를 계산할 때 검출 실패 프레임을 조용히 제외하면
결과가 과대평가될 수 있으므로, 이 파일도 전달.

이 bbox로 Depth 대표값과 중심근접도를 계산.
단순 최근거리 방식과 Priority Score 방식은 반드시 **동일한 CSV의 동일한
후보 집합**을 사용해야 공정하게 비교할 수 있음.

`--conf 0.05`는 후보를 미리 너무 많이 버리지 않기 위한 저장 기준.
논문에서 실제 최종 임계값을 정했다면 두 비교 방식에 똑같이 적용하고,
사용한 값을 기록.

## 9. GitHub에 올릴 것

올릴 것:

- 이 프로젝트의 `.py`, `.md`, `requirements.txt`, `.gitignore`
- 최종 실험 설정과 작은 JSON/CSV 결과
- 필요한 경우 용량이 허용되는 `best.pt`(또는 GitHub Release)

올리지 말 것:

- BUP-ST20 원본 RGB/Depth/annotation
- `images`, `labels`, 전체 `runs` 폴더
- 분할 압축 파일