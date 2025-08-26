# Deep Motion **Smoothing**  
*(Based on 12dmodel/deep_motion_mag, MIT License)*

https://github.com/12dmodel/deep_motion_mag


## 📌 프로젝트 개요

본 프로젝트는 **모션 스무딩(Motion Smoothing)** 관점에서 DeepMag(ECCV'18) 구조를 재구성하여,  
영상에서 불필요한 (진동·노이즈)을 줄이고 **안정화 효과**를 얻는 것을 목표로 합니다.  

- **원본 DeepMag**: 움직임을 **증폭 (α > 1)**  
- **본 구현**: 움직임을 **감쇠 (α < 1)** → **Motion Smoothing**  


## 아키텍쳐
![아키텍처](slide/slide1.png)

## ⚙️ 실행 방법 (How to Run)

### 1. Docker 환경 실행
```bash
docker run --gpus all -it --name deepmag-tf \
  -v /home/shchoi/deepmag:/workspace \
  tensorflow/tensorflow:1.15.5-gpu-py3 bash
```

> 컨테이너 이름: `deepmag-tf`, 작업 디렉토리: `/workspace`

### 2. 입력/출력 폴더 준비
```bash
mkdir -p demo/input demo/out demo/results
# demo/input 에 테스트 영상 또는 프레임 시퀀스를 넣습니다.
```

### 3. 추론 (Motion Smoothing, α=0.5 기본)
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

**주요 옵션**
- `alpha < 1`: 움직임 감쇠 정도 (스무딩 강도)  
- `diff_reg`: diff 크기 규제 → ghosting 억제  
- `gate`: 전체 게인 (0~1)  
- `band`: `low|mid|high` 주파수 대역 (기본 `low`)  
- `warmup`: 초기 프레임 안정화 구간  
- `clip_min/max`: 과도치 클리핑  

### 4. 학습 (선택)
```bash
python train.py --config configs/motion_smoothing.yaml --seed 42
```

---

## 🖥️ 코드 변경점 (What Changed)

| 모듈 | 원본 코드 | 변경 코드 | 목적 |
|------|-----------|-----------|------|
| `models/manipulator.py` | diff 직접 스케일링 | **diff 정규화 + 게이팅 추가** | 과도한 감쇠 시 ghosting 억제 |
| `run_temporal.py` | 단순 IIR 필터링 | **warmup/clip/band 옵션 추가** | Temporal 안정화 |
| `train.py` | 단순 L1 reconstruction loss | **motion 규제 항 추가** | 정지 구간에서 모션 억제 강화 |
| `configs/` | 기본 증폭 세팅 | **motion_smoothing.yaml 추가** | 감쇠 전용 설정 제공 |

## 코드 수정 – Motion Suppression Loss

기존 DeepMag은 모션 증폭만 수행합니다:
```python
self.loss = L1_loss(self.output, frameAmp) + texture_loss + shape_loss
```

**Motion Suppression**을 위해, 출력이 첫 번째 프레임(`frameA`)에 더 가까워지도록 추가 loss를 삽입합니다:

```python
with tf.variable_scope('ynet_3frames/encoder', reuse=True):
    texture_c, shape_c = self._encoder(frameC)
    self.loss += train_config["texture_loss_weight"] * L1_loss(texture_c, self.texture_a)
    self.loss += train_config["shape_loss_weight"] * L1_loss(shape_c, self.shape_b)

# Motion suppression loss
motion_suppression_loss = L1_loss(self.output, frameA) * 0.5
self.loss += motion_suppression_loss
```

### 효과
- 모션 증폭 (α > 1): 움직임을 크게 강조  
- 모션 감소 (α < 1 + suppression loss): 움직임 감소 및 부드러운 영상  
- 반대방향 증폭 (`enc_a - enc_b`): 또 다른 모션 감소 접근 방식

## 코드 수정 – Reverse Amplification (역증폭)

기존 Manipulator는 두 프레임의 차이를 `enc_b - enc_a`로 계산합니다:

```python
# 기존 방식
diff = enc_b - enc_a
diff = (alpha - 1) * diff
return enc_b + diff
```

이를 **반대 방향으로 증폭**하여 모션을 억제하는 방식으로 변경할 수 있습니다:

```python
# 수정된 방식 (역증폭)
diff = enc_a - enc_b
diff = (alpha - 1) * diff
return enc_a + diff
```

### 효과
- 기존: 두 번째 프레임(`enc_b`) 중심으로 움직임을 증폭  
- 수정: 첫 번째 프레임(`enc_a`) 중심으로 움직임을 반대로 증폭 → 실제로는 **motion suppression** 효과  
- Dynamic mode에서 특히 효과적 (frame 간 interpolation이 아닌 억제된 motion 표현)

## 코드 수정 – Temporal 모드에서 gaussian/Sharpening 적용

Temporal 모드에서 필터링된 feature `y`에 **Sharpening/gaussian**을 적용하여 ghosting을 조절하고 경계 선명도를 높일 수 있습니다.



### 1) Sharpening 함수 정의 (클래스 밖)
```python
from scipy.ndimage import gaussian_filter

#  Sharpening 함수 (Unsharp Masking)
def sharpen_feature(x, alpha=0.8, sigma=1.0):
    blurred = gaussian_filter(x, sigma=sigma)
    return x + alpha * (x - blurred)
```


### 2) MagNet3Frames.run_temporal 내부 
`y` 계산 직후, `y_state.insert(0, y)` 전에 Sharpening/gaussian을 적용합니다.

```python
y = np.zeros_like(x)
for i in range(len(x_state)):
    y += x_state[i] * filter_b[i]
for i in range(len(y_state)):
    y -= y_state[i] * filter_a[i]

# ✅ Sharpening 적용 (둘중 하나 선택)
y = sharpen_feature(y, alpha=0.8, sigma=1.0)
# ✅ gaussian 적용
y = gaussian_filter(y, sigma=sigma)

# update y state
y_state.insert(0, y)
if len(y_state) > len(filter_a):
    y_state = y_state[:len(filter_a)]

out_amp = self.sess.run(self.output_image,
                        feed_dict={self.out_texture_enc: texture_enc,
                                   self.filtered_enc: y,
                                   self.ref_shape_enc: x,
                                   self.amplification_factor: [amplification_factor]})
```

---

### 효과
- gaussian을 사용할 경우 ghosting 일부 억제
- Feature edge를 부드럽게 하여 ghosting 없앰



## 📊 결과 (Results)

### 정성적 결과
- Before: 증폭 구조 그대로 사용 시, 감쇠 과정에서 ghosting 발생  
- After: diff 정규화 + 게이팅 적용 → **잔상 감소**, **안정화 개선**

예시 결과:
| Before (No Smoothing) | After (Motion Smoothing) |
|-----------------------|--------------------------|
| ![Before](images/before.png) | ![After](images/after.png) |

### 정량적 결과 (예시)
- **정지 구간**: PSNR ↑, SSIM ↑, Ghosting Index ↓  
- **움직임 구간**: LPIPS 유지/개선  




## 📖 라이선스 & 출처

- License: **MIT License** (레포 `LICENSE` 참조)  
- Based on: [12dmodel/deep_motion_mag](https://github.com/12dmodel/deep_motion_mag) (MIT)  
- 연구 배경/결과: 연구 참여 발표 슬라이드 (Week1~4)  

---
