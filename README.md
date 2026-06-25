# 🛡️ AI 기반 낙상 감지 스마트 벨트 (Fall Guard)

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
| 학습 데이터 | 226,059개 (KFall + SisFall + 직접 측정) |
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
├── firestore_detection.py     # 메인 실시간 감지 코드 (약 850줄)
├── data_collection.py         # 직접 데이터 수집 코드
├── retrain_xgboost.py         # 재학습 코드
├── firebase_key.json          # Firebase 인증 (비공개)
├── xgb_model.pkl              # XGBoost 학습된 모델
├── xgb_direction_model.pkl    # 방향 분류 모델
└── xgb_feature_cols.pkl       # Feature 컬럼 목록
```

---

## ⚙️ 설치 및 실행

```bash
# 패키지 설치
pip install firebase-admin numpy pandas scikit-learn smbus2 pyserial pynmea2 lgpio --break-system-packages

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

### 전이학습 방식 재학습

공개 데이터셋(실험실 환경) 1차 학습 후,  
실제 벨트 착용 데이터 350개를 직접 수집해 재학습 → 민감도 **+2.26%** 향상

---

## 🩺 논문 기반 부상 위험도 로직

낙상 방향별로 어디가 위험한지 의학 논문 5편 기반으로 자동 계산합니다.

| 논문 | 핵심 수치 |
|---|---|
| Nevitt & Cummings (1993) JAGS | 옆 낙상 → 고관절 **6배** / 뒤 낙상 → 손목 **OR 2.2** |
| Berry & Miller (2008) PMC2793090 | 뒤 낙상 시 엉덩이 연부조직 충격 흡수 |
| Greenspan et al. (1994) JAMA | 옆 낙상 → 고관절 **OR 5.7** |
| PMC2562433 | 뒤 낙상 → 고관절 충격 **OR 12.6** |
| PMC3624001 | 선형+회전 가속도 복합 → 뇌진탕 위험 |

**판단 기준:**
- 방향별 기본 위험 레벨 (논문 OR값 기반)
- 나이 보정: 75세↑ +2단계 / 65~74세 +1단계 (WHO 기준)
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

## 🔮 향후 계획

- [ ] 낙상 데이터 직접 수집 후 재재학습
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

🔗 [상세 포트폴리오 보기](https://www.notion.so/373ab46034fc81f0888fe5d81d2c1c60)
