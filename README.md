# 🍅 Implementation of 3D Position Estimation and Harvesting Target Selection Algorithm for Fruit Vegetables

> **주제 : 과채류의 3차원 위치 추정 및 수확 대상 선택 알고리즘 구현**

---

## 📌 프로젝트 소개 (Project Overview)
본 프로젝트는 스마트 팜 환경 내 자율주행 수확 로봇을 위한 핵심 제어 및 비전 알고리즘 연구입니다.  
RGB-D (Depth) 카메라와 객체 인식 AI 모델(YOLO 등)을 융합하여 과채류의 **3차원 공간 좌표(X, Y, Z)를 정밀하게 추정**하고, 로봇 팔(Manipulator)이 최적의 경로로 안전하게 수확할 수 있도록 **익음 정도(Mature Level) 및 장애물 간섭을 고려한 수확 대상 선택 알고리즘**을 구현하였습니다.

---

## 🛠️ 주요 기능 및 기술 스택 (Key Features & Tech Stack)
* **Computer Vision & 3D Estimation** : 
  * RGB-D Sensor Fusion (Depth Camera)
  * Object Detection & Segmentation (YOLO, OpenCV)
  * Point Cloud Data (PCD) Processing
* **Robot Control & Path Planning** :
  * 3D Coordinate Transformation (Camera Frame to Robot Base Frame)
  * MoveIt! / Robot Arm Trajectory Planning
  * ROS 2 (Robot Operating System) Integration
* **Decision Making Algorithm** :
  * Harvesting Priority Target Selection (과채류 숙도 판별 및 수확 우선순위 알고리즘 구현)

---

## 📂 디렉토리 구조 (Directory Structure)
```text
├── vision/            # 3D 위치 추정 및 객체 인식 (YOLO, Depth) 관련 소스코드
├── planning/          # 매니퓰레이터 경로 계획 및 좌표 변환 코드
├── simulation/        # Gazebo 또는 Webots 시뮬레이션 환경 설정 파일
└── docs/              # 프로젝트 설계 문서 및 실험 데이터
