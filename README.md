# 🛡️ AI 기반 낙상 감지 스마트 벨트 (Fall Guard)

- **문제:** 고령자 낙상은 바로 발견되기 어렵고, 어디를 다쳤는지 몰라 응급 대응이 늦어짐
- **한 일:** MPU-6050 벨트 센서 + XGBoost 실시간 낙상 감지 모델, 논문 기반 부상 부위 위험도 로직, Firebase 보호자 알림 구현
- **결과:** 민감도 92.79% / AUC-ROC 99.50% 달성, WON+PBL 챌린지 2026 3등 수상

> WON+PBL 챌린지 2026 **3등 수상** 🥉 | 팀명: acu5

MPU-6050 센서와 XGBoost AI로 낙상을 실시간 감지하고,  
낙상 방향에 따라 **어디가 다쳤는지** 의학 논문 기반으로 자동 계산해  
보호자/의료진에게 즉시 전달하는 스마트 벨트 시스템입니다.

---

## 📌 핵심 성과

| 지표 | 결과 |
|---|---|
| 민감도 (Recall) | **92.79%** |
| AUC-ROC | **99.50%** |
| 학습 데이터 | 공개 데이터셋 226,059개(KFall + SisFall) + 직접 측정 350개 |
| 수상 | WON+PBL 챌린지 2026 3등 🥉 |

---

## 🏗️ 시스템 구조

```
MPU-6050 센서 (100Hz)
    ↓ 저역통과 필터 (alpha=0.3)
슬라이딩 윈도우 1초 / Feature 57개 추출
    ↓
XGBoost AI 낙상 판단 (확률 90% 이상)
    ↓
낙상 방향 분류 (앞/뒤/옆)
    ↓
논문 기반 부상 위험도 계산
    ↓
Logic A (즉시) / Logic B (15초 버튼 대기)
    ↓
Firebase Firestore → Fall Guard 대시보드 (카카오맵)
```

---

## 🛠️ 기술 스택

### 하드웨어
- Raspberry Pi 5 4GB
- MPU-6050 (GY-521) — 6축 IMU 센서
- NEO-6M GPS + Arduino Uno — 실시간 위치
- 피에조 부저 (GPIO 27) / 푸시 버튼 (GPIO 17)

### 소프트웨어
- Python 3.13 / NumPy / Pandas
- XGBoost / Scikit-learn
- Firebase Firestore
- lgpio / pyserial / pynmea2

---

## 📂 파일 구조

```
fall-guard-belt/
├── firestore_detection.py     # 메인 실시간 감지 코드 (약 970줄)
├── data_collection.py         # 직접 데이터 수집 코드
├── retrain_xgboost.py         # 재학습 코드
├── requirements.txt           # 패키지 목록
└── .gitignore                 # 비공개·대용량 파일 제외 규칙
```

아래 파일은 `.gitignore`로 제외되어 레포에 포함되지 않습니다. 실행하려면 직접 준비해야 합니다.

| 파일 | 설명 | 제외 이유 |
|---|---|---|
| `firebase_key.json` | Firebase 서비스 계정 키 | 🔒 보안 |
| `xgb_model.pkl` | XGBoost 낙상 감지 모델 | 📦 용량 |
| `xgb_direction_model.pkl` | 낙상 방향 분류 모델 | 📦 용량 |
| `xgb_feature_cols.pkl` | Feature 컬럼 목록 | 📦 용량 |
| `*.csv` | 학습/수집 데이터셋 | 📦 용량 |

---

## 🧰 실행 준비

- `firebase_key.json`, `xgb_model.pkl`, `xgb_direction_model.pkl`, `xgb_feature_cols.pkl`을 `firestore_detection.py`와 **같은 폴더**에 둡니다.
- `retrain_xgboost.py`의 결과물은 `XGBoost_Model` 폴더에 저장됩니다. 재학습 후에는 생성된 pkl 파일들을 `firestore_detection.py`가 있는 폴더로 복사하세요.
- 재학습에는 `combined_features.csv`와 `custom_data_*.csv`가 필요합니다.

---

## ⚙️ 설치 및 실행

```bash
# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# 실행
python3 firestore_detection.py
```

---

## 🤖 AI 모델

### 모델 비교

| 모델 | 정확도 | 민감도 | AUC-ROC |
|---|---|---|---|
| Random Forest | 98.40% | 86.15% | 99.16% |
| **XGBoost ★** | **98.61%** | **90.53%** | **99.38%** |
| XGBoost (재학습 후) | 98.40% | **92.79%** | **99.50%** |

### Feature 중요도 TOP 3

| 순위 | Feature | 중요도 |
|---|---|---|
| 1위 | gyr_resultant_max | **35.11%** |
| 2위 | AccY_mean | 11.12% |
| 3위 | GyrX_peak | 7.54% |

### 직접 측정 데이터를 합쳐 재학습

공개 데이터셋(실험실 환경) 1차 학습 후,  
실제 벨트 착용 데이터 350개를 직접 수집해 재학습

---

## 🩺 논문 기반 부상 위험도 로직

낙상 방향별로 어디가 위험한지 의학 논문 5편 기반으로 자동 계산합니다.

| 논문 | 핵심 수치 |
|---|---|
| Nevitt & Cummings (1993) JAGS | 고령 여성 대상. 고관절 골절은 옆으로 넘어지거나 곧장 주저앉듯 넘어질 때(OR 3.3)와 고관절 부위 충격(OR 32.5), 손목 골절은 뒤로 넘어질 때(OR 2.2)에 많았음 |
| Berry & Miller (2008) PMC2793090 | 뒤 낙상 시 엉덩이 연부조직 충격 흡수 |
| Greenspan et al. (1994) JAMA | 65세 이상 낙상자 대상. 낙상 방향(OR 5.7), 골밀도, 낙상 시 위치에너지, 체질량지수가 고관절 골절의 독립 위험요인 |
| PMC2562433 | 원문 확인 필요 |
| PMC3624001 | 선형+회전 가속도 복합 → 뇌진탕 위험 |

**판단 기준:**
- 방향별 기본 위험 레벨 (논문 결과를 참고해 직접 설정)
- 나이 보정: 75세↑ +2단계 / 65~74세 +1단계 (고령일수록 골밀도가 낮아지는 점을 반영한 자체 규칙)
- 충격 보정: 5g↑ +1단계 / 1g↓ -1단계

**위험 등급:** 낮음 / 보통 / 높음 / 매우높음 (4단계)

---

## 🔧 오탐 방지 Logic A/B

```python
if acc_max >= 2.5 or gyr_max >= 200:
    buzz("urgent")       # Logic A: 즉시 부저 3번 + Firebase 전송
else:
    buzz("short")        # Logic B: 부저 1번
    cancelled = wait_for_button(15)   # 15초 버튼 대기
    is_confirmed = not cancelled      # 버튼 누르면 오탐 취소
```

---

## 🔥 Firebase 데이터 구조

```
patients/{patient_id}
  → name, age, medical_history, guardian_contact

fall_events/{patient_id}/events/{event_id}
  → fall_probability, fall_direction, acc_max_g, gyr_max_dps
  → location, reason, main_injury, first_aid_guide

injury_risk/{patient_id}/events/{event_id}
  → 고관절_골절("매우높음"), 고관절_골절_pct(100)
  → 위험_등급 (낮음/보통/높음/매우높음)

alerts/{event_id}
  → patient_id, direction, risk_grade, confirmed
```

---

## 🌐 Fall Guard 모니터링 대시보드

- React + Firebase Firestore 실시간 연동
- 카카오맵 API — 낙상 위치 실시간 표시
- 위험(80%↑) / 주의(50~79%) / 정상 3단계 분류
- CSV 내보내기 / 긴급 알림 팝업

---

## ⚠️ 한계

- 직접 측정 데이터는 일상 동작만 포함합니다. 낙상 데이터는 안전상 매트 위에서만 수집할 수 있었습니다.
- 실시간 감지에는 저역통과 필터를 쓰지만, 직접 측정한 학습 데이터에는 적용하지 않았습니다.
- 부상 위험도는 논문 결과를 참고한 자체 규칙이며 임상 검증을 거치지 않았습니다.
- 배터리 잔량은 시뮬레이션 값입니다.

---

## 🔮 향후 계획

- [ ] 낙상 데이터 직접 수집 후 재학습
- [ ] FCM 푸시 알림 (보호자 핸드폰)
- [ ] 경량화: ESP32/STM32 전용 MCU 교체
- [ ] AI 경량화: XGBoost → TensorFlow Lite
- [ ] LTE 모듈 추가

---

## 👤 개발자

| 이름 | 역할 |
|---|---|
| **이상훈 (팀장)** | AI 모델, 데이터 전처리, 실시간 감지 코드, 하드웨어 세팅, 부상 위험도 로직 |
| 정영진 | 웹 대시보드 (Fall Guard), Firebase 연동, 카카오맵 |
| 이승호 | 하드웨어 제작, PPT |

---

## 📄 노션 포트폴리오

- 🔗 [요약](https://app.notion.com/p/3efab46034fc8166a583c7a371cb55be)
- 🔗 [상세 기록](https://app.notion.com/p/38aab46034fc8168ad4add3e5747b013)
