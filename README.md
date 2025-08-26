# Deep Motion **Reduction**  
*(Based on 12dmodel/deep_motion_mag, MIT License)*

---

## 📌 프로젝트 개요

본 프로젝트는 **모션 감소(Motion Reduction)** 관점에서 DeepMag(ECCV'18) 구조를 재구성하여,  
영상에서 불필요한 미세 움직임(진동·노이즈)을 줄이고 **안정화 효과**를 얻는 것을 목표로 합니다.  

- **원본 DeepMag**: 영상 내 움직임을 증폭(α > 1)  
- **본 구현**: 움직임을 감쇠(α < 1) → 불필요한 모션 억제  

연구 참여 기간 동안 발표한 슬라이드에서 모델 구조, 데이터 합성, Temporal 필터링 및 결과를 정리했습니다.  
## Architecture
![아키텍처](slide/slide1.png)

---
## ⚙️ 환경 설정 (Docker 권장)

본 프로젝트는 **TensorFlow 1.15.5 GPU** 환경을 기준으로 작성되었습니다.  
Docker + NVIDIA GPU 환경에서 재현하는 것을 권장합니다.  

### 1. Docker 실행
```bash
docker run --gpus all -it --name deepmag-tf \
  -v /home/shchoi/deepmag:/workspace \
  tensorflow/tensorflow:1.15.5-gpu-py3 bash
```

> - 컨테이너 이름: `deepmag-tf`  
> - 호스트 `/home/shchoi/deepmag` 폴더가 `/workspace`로 마운트됨  

### 2. 컨테이너 관리
```bash
# 재접속
docker start -ai deepmag-tf

# 실행 중 컨테이너에 새 셸 붙기
docker exec -it deepmag-tf bash

# 중지
docker stop deepmag-tf
```

---

## 🖥️ 사용 소프트웨어 & 수정 방법

- **프레임워크**: TensorFlow 1.15.5 (GPU)  
- **언어/툴**: Python3, OpenCV, NumPy, Matplotlib, YAML  
- **변경 포인트**  
  - `models/manipulator.py`: `diff` 텐서 정규화 + 게이팅 추가  
  - `run_temporal.py`: warmup/clip/band 옵션 추가 → 안정적 감쇠  
  - `train.py`: motion 규제 항(loss term) 추가 → ghosting 억제  
  - `configs/motion_reduction.yaml`: 감쇠용 기본 설정  

---

## ▶️ 실행 방법

### 입력/출력 폴더 준비
```bash
mkdir -p demo/input demo/out demo/results
# demo/input 에 테스트 영상 또는 프레임 시퀀스를 넣습니다.
```

### 추론 (모션 감소 기본 α=0.5)
```bash
python run_temporal.py \
  --input_dir demo/input \
  --output_dir demo/out \
  --alpha 0.5 \
  --diff_reg 0.1 \
  --gate 0.8 \
  --band low \
  --warmup 5 \
  --clip_min -0.02 --clip_max 0.02
```

- `alpha < 1`: 모션 감쇠 정도  
- `diff_reg`: diff 크기 규제 → ghosting 억제  
- `gate`: 전체 게인 (0~1)  
- `band`: `low|mid|high` 주파수 대역 (기본 `low`)  
- `warmup`: 초기 프레임 안정화 구간  
- `clip_min/max`: 과도치 클리핑  

### 학습 (선택)
```bash
python train.py --config configs/motion_reduction.yaml --seed 42
```

---

## 📊 결과

- **정성적**: 결과 영상/GIF (before vs after) → `demo/results/` 폴더 참조  
- **정량적** (예시)  
  - 정지 구간: PSNR/SSIM ↑, Ghosting Index ↓  
  - 움직임 구간: LPIPS 유지/개선  

---

## 📜 변경 기록 (Changelog)

| 날짜 | 파일/모듈 | 변경 내용 | 이유/의도 | 커밋 |
|---|---|---|---|---|
| 2025-08-26 | 초기 | 레포 생성, Docker 실행법 추가 | 재현성 확보 | `init` |
| 2025-08-27 | manipulator.py | diff 정규화+게이팅 | 과감쇠 시 ghosting 억제 | `abc1234` |
| 2025-08-27 | run_temporal.py | warmup/clip/band 옵션 | temporal 안정화 | `def5678` |
| 2025-08-27 | train.py | motion 규제 항 추가 | 정지 구간 보호 | `ghi9abc` |

---

## 📖 라이선스 & 출처

- License: **MIT License** (레포 `LICENSE` 파일 참조)  
- Based on: [12dmodel/deep_motion_mag](https://github.com/12dmodel/deep_motion_mag) (MIT)  
- 연구 배경/결과: 연구 참여 발표 슬라이드 (Week1~4)  

---
