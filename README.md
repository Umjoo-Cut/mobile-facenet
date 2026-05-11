# Update 2026/5/11

# MobileFaceNet Driver Authentication System

라즈베리파이 기반 얼굴 인증 및 음주 측정 연동 시스템 프로젝트입니다.


## 📌 프로젝트 상세

운전자 얼굴 인증 후,
고정형 음주 측정기를 이용하여
실제 운전자가 측정을 진행했는지 확인하는 시스템입니다.

### 주요 목표
- 얼굴 인증 기반 운전자 식별
- 타인의 대리 음주 측정 방지
- Raspberry Pi 기반 실시간 동작
- 향후 MQ-3 및 STM32 연동 예정

---

# 📷 Demo

## 실시간 얼굴 인증 화면

> 위 사람이 등록된 사람이라고 가정

### Pass 일 때

![demo](assets/pass.gif)

### Fail 일 때

![demo](assets/fail.gif)

# 🧠 시스템 구조

```bash
Camera
 ↓
MediaPipe FaceMesh
(Face Alignment)

 ↓
MobileFaceNet
(Face Embedding)

 ↓
Cosine Similarity

 ↓
PASS / FAIL
```

# 🔧 기술 스택

## AI / Vision
- MobileFaceNet (ONNX)
- MediaPipe FaceMesh
- OpenCV
- ONNX Runtime

## 하드웨어
- Raspberry Pi 5
- USB Camera
- MQ-3 (계획중)
- STM32 (계획중)

## 언어
- Python

# 📂 프로젝트 구조
```bash
mobile-facenet/
├── main.py
├── models/
│   └── w600k_mbf.onnx
├── register/
│   └── owner/
├── embeddings/
├── requirements.txt
└── README.md
```

# ⚙️ 기능

## ✅ 구현 완료 사항
- 실시간 얼굴 인증
- MediaPipe FaceMesh 기반 얼굴 정렬
- 얼굴 임베딩 생성
- 코사인 유사도 비교
- PASS / FAIL 인증 처리
- 라즈베리파이 이식 후 실행 완료

## 🚧 앞으로의 계획
- 고정형 음주 측정기 위치 감지
- MQ-3 음주 센서 연동
- STM32 시동 제어
- 다중 얼굴 차단 기능
- 운전자 잠금 상태 시스템

# 🧩 인증 과정
1. 운전자 얼굴 인증 PASS
2. 입 위치를 고정된 측정기에 가까이 이동
3. 몇 초 동안 위치 유지
4. MQ-3 센서 값 감지
5. 최종 PASS

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/your-repo/mobile-facenet.git
cd mobile-facenet
```

## 가상 환경 생성

```bash
python -m venv venv
```

## 가상 환경 실행

### Windows
```bash
venv\Scripts\activate
```

### Linux / Raspberry Pi
```bash
source venv/bin/activate
```

### 패키지 설치
```bash
pip install -r requirements.txt
```

## ▶️ 실행
```bash
python main.py
```

## 📝 참고 사항
- 안정적인 얼굴 정렬을 위해 MediaPipe FaceMesh를 사용했습니다.
- 라즈베리파이 환경에서의 경량 추론을 위해 MobileFaceNet을 선택했습니다.
- 얼굴 정렬 적용 후 얼굴 인증 성능이 크게 향상되었습니다.

## 👥 팀 역할

- 얼굴 인증: MobileFaceNet
- 얼굴 정렬: MediaPipe FaceMesh
- 라즈베리파이 연동
- 음주 감지 시스템


