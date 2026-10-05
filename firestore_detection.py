"""
낙상 감지 벨트 - 실시간 감지 코드 (Firestore + GPS 버전)
=========================================================
시뮬레이션 모드: 데이터셋으로 실시간 테스트
실제 센서 모드: MPU-6050 + NEO-6M GPS + 라즈베리파이5

폴더 구조:
  XGBoost_Model/
  ├── firestore_detection.py  ← 이 파일
  ├── firebase_key.json
  ├── xgb_model.pkl
  ├── xgb_direction_model.pkl
  ├── xgb_feature_cols.pkl
  └── combined_features.csv   (시뮬레이션 모드 시 필요)

실행: python3 firestore_detection.py
필요: pip install firebase-admin numpy pandas scikit-learn smbus2 pyserial pynmea2 --break-system-packages
"""

import os, time, pickle, threading
try:
    import lgpio
    _chip = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_input(_chip, 17, lgpio.SET_PULL_UP)  # 17번 핀 버튼
    lgpio.gpio_claim_output(_chip, 27, 0)                  # 27번 핀 피에조 부저
    GPIO_AVAILABLE = True
    print("  GPIO 버튼 + 피에조 부저 초기화 완료 ✅")
except Exception as e:
    GPIO_AVAILABLE = False
    _chip = None
    print(f"  GPIO 없음 → 버튼/부저 비활성화")
import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# ──────────────────────────────────────────────────────────
# ★ 모드 설정 ★
# True  = 시뮬레이션 모드 (데이터셋으로 테스트)
# False = 실제 센서 모드  (MPU-6050 + NEO-6M GPS)
# ──────────────────────────────────────────────────────────
SIMULATION_MODE = False

# ──────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
KEY_FILE      = os.path.join(BASE_DIR, "firebase_key.json")
DATA_FILE     = os.path.join(BASE_DIR, "combined_features.csv")

RF_MODEL      = os.path.join(BASE_DIR, "rf_model.pkl")
XGB_MODEL     = os.path.join(BASE_DIR, "xgb_model.pkl")

DIRECTION_MAP  = {0:"앞으로 낙상", 1:"뒤로 낙상", 2:"옆으로 낙상"}
WINDOW_SIZE    = 100
SAMPLING_RATE  = 100
FALL_THRESHOLD   = 0.90   # 낙상 확률 90% 이상 (오탐 방지)
COOLDOWN_SEC     = 3.0    # 낙상 감지 후 3초간 재감지 방지

# 즉시 알림 기준 (Logic A)
IMMEDIATE_ACC    = 2.5    # 2.5g 이상 → 즉시 알림 (노인 기준)
IMMEDIATE_GYR    = 200.0  # 200°/s 이상 → 즉시 알림 (노인 기준)

# 버튼 대기 시간 (Logic B)
BUTTON_WAIT_SEC  = 15     # 15초 내 버튼 누르면 오탐 취소
DEMO_MODE        = False  # 발표 시연용: True → 오탐 취소 기능 비활성화 / 평상시 False
BUTTON_PIN       = 17     # GPIO 17번 핀 (버튼)
BUZZER_PIN       = 27     # GPIO 27번 핀 (피에조 부저)

# 배터리 설정 (소프트웨어 시뮬레이션)
BATTERY_LEVEL    = 100    # 시작 시 100%
BATTERY_INTERVAL = 60     # 1분마다 1% 감소
BATTERY_WARN     = 20     # 20% 이하 경고
BATTERY_CRITICAL = 5      # 5% 이하 보호자 알림

# GPS 설정 (NEO-6M → 아두이노 우노 → USB)
# 아두이노 연결 시 ttyUSB0 또는 ttyACM0 자동 감지
GPS_PORTS    = ["/dev/ttyUSB0", "/dev/ttyACM0", "/dev/ttyUSB1"]
GPS_BAUDRATE = 9600

# 환자 ID는 실행 시 입력받음 (메인에서 설정)
PATIENT_ID   = None
PATIENT_INFO = None

# 모델 자동 감지
if os.path.exists(XGB_MODEL):
    MODEL_FILE = XGB_MODEL
    DIR_FILE   = os.path.join(BASE_DIR, "xgb_direction_model.pkl")
    FEAT_FILE  = os.path.join(BASE_DIR, "xgb_feature_cols.pkl")
    MODEL_NAME = "XGBoost"
else:
    MODEL_FILE = RF_MODEL
    DIR_FILE   = os.path.join(BASE_DIR, "rf_direction_model.pkl")
    FEAT_FILE  = os.path.join(BASE_DIR, "rf_feature_cols.pkl")
    MODEL_NAME = "Random Forest"


# ──────────────────────────────────────────────────────────
# GPS 클래스 (NEO-6M 실시간 위치 읽기)
# ──────────────────────────────────────────────────────────
class GPSReader:
    def __init__(self):
        self.latitude  = None
        self.longitude = None
        self.is_valid  = False
        self._thread   = None
        self._running  = False

    def start(self):
        try:
            import serial
            import pynmea2
            import os

            # 포트 자동 감지
            found_port = None
            for port in GPS_PORTS:
                if os.path.exists(port):
                    found_port = port
                    break

            if not found_port:
                print(f"  ❌ 아두이노 USB 포트를 찾을 수 없음")
                print(f"  → 아두이노 USB 연결 확인 후 재시작")
                self._running = False
                return

            print(f"  GPS 포트 ({found_port}) 연결 시도 중...")
            self._serial = serial.Serial(found_port, GPS_BAUDRATE, timeout=2)
            print(f"  GPS 포트 열림 ✅")

            # 실제 GPS 데이터 수신 확인 (최대 10초 대기)
            print("  GPS 센서 데이터 확인 중... (최대 10초)")
            gps_detected = False
            for i in range(20):
                line = self._serial.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("$GP") or line.startswith("$GN"):
                    gps_detected = True
                    print(f"  GPS 데이터 수신됨: {line[:30]}...")
                    break

            if gps_detected:
                self._running = True
                self._thread  = threading.Thread(target=self._read_loop, daemon=True)
                self._thread.start()
                print("  ✅ GPS 센서 감지됨! 실외에서 위성 신호 수신 시 위치 저장")
                print("  ⚠️  실내에서는 GPS 신호 없음 (실외로 나가면 자동으로 잡힘)")
            else:
                self._serial.close()
                self._running = False
                print("  ❌ GPS 센서 없음 (포트는 열렸지만 데이터 없음)")
                print("  → GPS 없이 실행 (위치 정보 미포함)")

        except serial.SerialException as e:
            print(f"  ❌ GPS 포트 연결 실패: {found_port} 포트를 열 수 없음")
            print(f"  → GPS 센서가 연결되지 않았거나 포트가 다름")
            print(f"  → GPS 없이 실행 (위치 정보 미포함)")
            self._running = False
        except ImportError:
            print(f"  ❌ pyserial 미설치 → pip install pyserial --break-system-packages")
            self._running = False
        except Exception as e:
            print(f"  ❌ GPS 오류: {e}")
            print(f"  → GPS 없이 실행 (위치 정보 미포함)")
            self._running = False

    def _read_loop(self):
        import pynmea2
        while self._running:
            try:
                line = self._serial.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("$GPRMC") or line.startswith("$GNRMC"):
                    msg = pynmea2.parse(line)
                    if msg.status == "A":   # A = 유효한 GPS 신호
                        self.latitude  = float(msg.latitude)
                        self.longitude = float(msg.longitude)
                        self.is_valid  = True
            except Exception:
                pass

    def get_location(self):
        if self.is_valid and self.latitude and self.longitude:
            return {
                "latitude":  round(self.latitude, 6),
                "longitude": round(self.longitude, 6),
                "gps_valid": True
            }
        else:
            return {
                "latitude":  None,
                "longitude": None,
                "gps_valid": False,
                "note":      "GPS 신호 없음 (실내 또는 신호 미수신)"
            }

    def stop(self):
        self._running = False
        if hasattr(self, "_serial"):
            self._serial.close()


# ──────────────────────────────────────────────────────────
# 저역통과 필터 (노이즈 제거)
# ──────────────────────────────────────────────────────────
class LowPassFilter:
    def __init__(self, alpha=0.3):
        self.alpha  = alpha
        self.prev   = None

    def filter(self, value):
        if self.prev is None:
            self.prev = value
        self.prev = self.alpha * value + (1 - self.alpha) * self.prev
        return self.prev

    def reset(self):
        self.prev = None

# 6축 필터 (AccX/Y/Z + GyrX/Y/Z)
filters = [LowPassFilter(alpha=0.3) for _ in range(6)]

def apply_filter(raw_data):
    return [filters[i].filter(raw_data[i]) for i in range(6)]

# ──────────────────────────────────────────────────────────
# 버튼 대기 함수
# ──────────────────────────────────────────────────────────
def buzz(pattern="short"):
    """
    피에조 부저 소리 출력
    pattern:
      short  → 짧게 1번 (정상 알림)
      long   → 길게 1번 (낙상 감지)
      urgent → 3번 빠르게 (긴급 상황)
    """
    if not GPIO_AVAILABLE:
        return
    try:
        if pattern == "short":
            lgpio.gpio_write(_chip, BUZZER_PIN, 1)
            time.sleep(0.2)
            lgpio.gpio_write(_chip, BUZZER_PIN, 0)
        elif pattern == "long":
            lgpio.gpio_write(_chip, BUZZER_PIN, 1)
            time.sleep(1.0)
            lgpio.gpio_write(_chip, BUZZER_PIN, 0)
        elif pattern == "urgent":
            for _ in range(3):
                lgpio.gpio_write(_chip, BUZZER_PIN, 1)
                time.sleep(0.3)
                lgpio.gpio_write(_chip, BUZZER_PIN, 0)
                time.sleep(0.2)
    except:
        pass

def update_battery(db, patient_id, level):
    """배터리 잔량 터미널 출력만 (Firebase 전송 안 함 → 사용량 절약)"""
    if level <= 5:
        print(f"\n  🔴 배터리 위급! {level}%")
    elif level <= 20:
        print(f"\n  🟡 배터리 부족! {level}%")

def wait_for_button(seconds):
    """
    seconds 동안 버튼 입력 대기
    버튼 누르면 True (오탐), 시간 초과면 False (진짜 낙상)
    GPIO 없으면 즉시 False 리턴 → 바로 전송
    """
    if not GPIO_AVAILABLE:
        print("  ⚠️  GPIO 없음 → 즉시 전송")
        return False

    start = time.time()
    while time.time() - start < seconds:
        remaining = seconds - (time.time() - start)
        print(f"  ⏳ 취소 버튼 대기 중... {remaining:.0f}초 남음", end="\r")
        if lgpio.gpio_read(_chip, BUTTON_PIN) == 0:  # 버튼 눌림
            print()
            return True
        time.sleep(0.1)
    print()
    return False

def get_main_injury(risk):
    """
    주요 부상 부위 1~2개 추출
    위험 레벨(매우높음>높음>보통>낮음) 기준으로 정렬
    """
    LEVEL_ORDER = {"낮음": 0, "보통": 1, "높음": 2, "매우높음": 3}
    items = {k: v for k, v in risk.items() if k != "위험_등급" and "_pct" not in k}
    sorted_items = sorted(
        items.items(),
        key=lambda x: LEVEL_ORDER.get(x[1], 0),
        reverse=True
    )
    main   = sorted_items[0] if len(sorted_items) > 0 else None
    second = sorted_items[1] if len(sorted_items) > 1 else None
    return main, second

def get_first_aid_guide(direction, risk, medical_history="없음"):
    """
    즉각 조치 가이드 자동 생성
    부상 위험도 레벨 + 기저질환 + 위험 등급 연동
    """
    main, second = get_main_injury(risk)
    grade = risk.get("위험_등급", "낮음")
    guide = []

    # 기저질환 연동
    if medical_history and medical_history != "없음":
        if "골다공증" in medical_history:
            guide.append("⚠️ 골다공증 환자 → 골절 위험 매우 높음, 절대 이동 금지")
        if "혈액희석제" in medical_history or "와파린" in medical_history:
            guide.append("⚠️ 혈액희석제 복용 → 내출혈 위험, 즉시 병원 이송")
        if "당뇨" in medical_history:
            guide.append("⚠️ 당뇨 환자 → 상처 회복 지연, 상처 즉시 확인")

    # 주요 부상 부위별 조치 (위험도 레벨 반영)
    if main:
        injury_name  = main[0]
        injury_level = main[1]

        if "고관절" in injury_name:
            if injury_level in ["매우높음", "높음"]:
                guide.append("🚨 고관절 골절 의심 → 절대 이동 금지! 즉시 119 요청")
            else:
                guide.append("📋 고관절 부상 의심 → 이동 주의, 병원 방문 권장")

        elif "뇌진탕" in injury_name:
            if injury_level in ["매우높음", "높음"]:
                guide.append("🚨 뇌진탕 의심 → 즉시 의식 확인, 움직이지 말고 119 요청")
            else:
                guide.append("📋 두부 충격 가능 → 의식 확인, 두통/구토 시 즉시 병원")

        elif "손목" in injury_name:
            if injury_level in ["매우높음", "높음"]:
                guide.append("🚨 손목 골절 의심 → 손목 고정 후 즉시 병원 이송")
            else:
                guide.append("📋 손목 충격 → 손목 고정 후 병원 방문")

        elif "허리" in injury_name:
            if injury_level in ["매우높음", "높음"]:
                guide.append("🚨 척추 부상 의심 → 절대 이동 금지! 등 받쳐주고 119 요청")
            else:
                guide.append("📋 허리 부상 의심 → 등 받쳐주고 움직임 최소화")

        elif "어깨" in injury_name:
            if injury_level in ["매우높음", "높음"]:
                guide.append("🚨 어깨 골절 의심 → 팔 고정 후 즉시 병원 이송")
            else:
                guide.append("📋 어깨 부상 의심 → 팔 고정 후 병원 방문")

    # 2순위 부상 추가 안내
    if second and second[1] in ["매우높음", "높음"]:
        guide.append(f"⚠️ {second[0].replace('_', ' ')} 추가 주의 필요")

    # 위험 등급별 최종 조치 (매우높음 포함)
    if grade == "매우높음":
        guide.append("🔴 즉시 119 신고! 움직이지 마세요!")
    elif grade == "높음":
        guide.append("🔴 즉시 119 신고 또는 보호자 긴급 연락!")
    elif grade == "보통":
        guide.append("🟡 보호자 즉시 연락 후 병원 이송 권장")
    else:
        guide.append("🟢 환자 상태 확인 후 필요시 병원 방문")

    return guide

def get_fall_reason(acc_max, gyr_max, direction, prob):
    """판단 근거 텍스트 생성 (Explainability)"""
    dir_str = {0:"앞으로", 1:"뒤로", 2:"옆으로"}.get(direction, "")
    reason = (
        f"가속도가 순간적으로 {acc_max:.2f}g를 초과하고 "
        f"자이로가 {gyr_max:.1f}°/s로 급격히 변화하여 "
        f"{dir_str} 낙상으로 판단됨 (확률 {prob*100:.1f}%)"
    )
    return reason

# ──────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────
def to_python_type(obj):
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_python_type(i) for i in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    return obj


def calculate_injury_risk(direction, acc_max, gyr_max, age=65):
    """
    낙상 방향별 부상 위험도 계산 (논문 기반 레벨 시스템)
    ─────────────────────────────────────────────────────
    근거 논문:
    1. Nevitt & Cummings (1993) J Am Geriatr Soc 41:1226-1234
       - 고령 여성 대상. 고관절 골절은 옆으로 넘어지거나 곧장 주저앉듯
         넘어질 때, 고관절 부위 충격 시 많았음
       - 손목 골절은 뒤로 넘어질 때 많았음

    2. Berry & Miller (2008) Curr Osteoporos Rep (PMC2793090)
       - 고관절 골절은 옆으로 넘어질 때 더 많이 연관됨
       - 엉덩이로 곧장 떨어질 때는 뒤쪽 연부조직이 충격을 일부 흡수해
         골절 위험이 낮게 관찰됨(리뷰)

    3. Greenspan et al. (1994) JAMA 271:128-133
       - 65세 이상 낙상자 대상. 낙상 방향, 골밀도, 낙상 시 위치에너지,
         체질량지수가 고관절 골절의 독립 위험요인

    4. Lo & Ashton-Miller (2008) J Biomech (PMC2562433)
       - 낙하 전략에 따른 충격력 감소 모의실험 연구
       - 충격력과 골절 한계의 비율로 위험을 평가하는 방식 참고

    5. Rowson & Duma (2013) Ann Biomed Eng (PMC3624001)
       - 머리 선형·회전 가속도를 함께 쓰는 아이디어 참고
       - 임계값은 직접 설정, 허리 센서 기준 간접 추정 적용

    ─────────────────────────────────────────────────────
    위험 레벨 정의:
      매우높음: 즉각 의료 개입 필요 (나이+충격 복합 등)
      높음:     빠른 의료 확인 필요
      보통:     병원 방문 권장
      낮음:     모니터링

    판단 보정 요소:
    ① 나이 보정 (자체 규칙: 고령일수록 골밀도가 낮아지는 점 반영)
       - 75세 이상 → 레벨 2단계 상향 (골밀도 현저 저하)
       - 65~74세   → 레벨 1단계 상향 (골밀도 저하 시작)
       - 65세 미만 → 보정 없음

    ② 충격 세기 보정 (Greenspan 1994 개념 적용)
       - 5g 이상   → 레벨 1단계 상향 (고속 낙상)
       - 1g ~ 5g   → 보정 없음 (일반 낙상)
       - 1g 미만   → 레벨 1단계 하향 (매우 경미한 충격)

    ③ 뇌진탕 위험 보정 (Rowson & Duma 2013 선형+회전 복합 아이디어, 임계값은 직접 설정)
       - 4g 이상 AND 400°/s 이상 → 레벨 1단계 상향 (강한 충격+급격한 회전)
       - 그 외 → 추가 보정 없음 (②의 충격 보정만 적용, 이중 하향 방지)
    ─────────────────────────────────────────────────────
    """

    # 레벨 정의 (숫자가 클수록 위험)
    LEVELS = ["낮음", "보통", "높음", "매우높음"]

    def get_level(base_level, age, acc, gyr, injury_type="fracture"):
        """
        base_level: 0~3 (낮음~매우높음)
        논문 결과를 참고해 직접 설정한 기본값 + 나이/충격 보정

        보정 원칙:
        - 나이는 항상 상향 보정 (골밀도 저하)
        - 충격 5g 이상만 상향, 1g 미만만 하향
        - 1g~5g 사이는 보정 없음 (일반 낙상)
        - 뇌진탕은 강한 충격+회전 시에만 상향
        """
        level = base_level

        # ① 나이 보정 (자체 규칙)
        if age >= 75:
            level += 2   # 75세 이상: 골밀도 현저 저하
        elif age >= 65:
            level += 1   # 65~74세: 골밀도 저하 시작

        # ② 충격 세기 보정
        if acc >= 5.0:
            level += 1   # 고속 낙상 → 추가 위험
        elif acc < 1.0:
            level -= 1   # 매우 경미 (1g 미만) → 하향
        # 1g~5g 사이: 보정 없음

        # ③ 뇌진탕 전용 보정 (Rowson & Duma 2013 선형+회전 복합 아이디어, 임계값은 직접 설정)
        # ※ ②에서 이미 acc<1.0이면 -1 적용됨 → 여기서는 상향만 추가 검토
        if injury_type == "brain":
            if acc >= 4.0 and gyr >= 400:
                level += 1   # 강한 충격 + 급격한 회전 → 뇌진탕 고위험
            # acc<1.0 하향은 ②에서 이미 처리됨 → 중복 적용 방지

        # 범위 제한 (0~3)
        level = max(0, min(3, level))
        return LEVELS[level]

    risk = {}

    if direction == 0:
        # ── 앞으로 낙상 ──────────────────────────────────────
        # 논문 결과를 참고해 직접 설정한 기본 레벨 → 보통(1)
        # 65세 이상이면 자동으로 높음
        # 75세 이상이면 자동으로 매우높음
        risk["손목_골절"]   = get_level(1, age, acc_max, gyr_max)
        # 논문 결과를 참고해 직접 설정한 기본 레벨 → 보통(1)

        risk["뇌진탕"]      = get_level(0, age, acc_max, gyr_max, "brain")
        # 앞으로 낙상 시 머리 충격 낮음 → 기본 낮음(0)

        risk["허리_부상"]   = get_level(1, age, acc_max, gyr_max)
        # 척추 전방 압박 가능 → 기본 보통(1)

        risk["고관절_골절"] = get_level(0, age, acc_max, gyr_max)
        # 앞으로 낙상 고관절 위험 낮음 → 기본 낮음(0)

        risk["어깨_부상"]   = get_level(0, age, acc_max, gyr_max)
        # 팔로 짚는 경우 → 기본 낮음(0)

    elif direction == 1:
        # ── 뒤로 낙상 ──────────────────────────────────────
        risk["고관절_골절"] = get_level(2, age, acc_max, gyr_max)
        # 논문 결과를 참고해 직접 설정한 기본 레벨 (연부조직 흡수 반영) → 높음(2)

        risk["허리_부상"]   = get_level(2, age, acc_max, gyr_max)
        # 뒤로 넘어질 때 척추 압박 위험 높음 → 기본 높음(2)

        risk["뇌진탕"]      = get_level(1, age, acc_max, gyr_max, "brain")
        # 후두부 충격 가능 → 기본 보통(1)

        risk["손목_골절"]   = get_level(1, age, acc_max, gyr_max)
        # 논문 결과를 참고해 직접 설정한 기본 레벨 → 보통(1)

        risk["어깨_부상"]   = get_level(0, age, acc_max, gyr_max)
        # 뒤로 낙상 시 어깨 위험 낮음 → 기본 낮음(0)

    elif direction == 2:
        # ── 옆으로 낙상 ──────────────────────────────────────
        risk["고관절_골절"] = get_level(3, age, acc_max, gyr_max)
        # 논문 결과를 참고해 직접 설정한 기본 레벨 → 매우높음(3)

        risk["어깨_부상"]   = get_level(2, age, acc_max, gyr_max)
        # 논문 결과를 참고해 직접 설정한 기본 레벨 → 높음(2)

        risk["뇌진탕"]      = get_level(1, age, acc_max, gyr_max, "brain")
        # 측두부 충격 가능 → 기본 보통(1)

        risk["허리_부상"]   = get_level(1, age, acc_max, gyr_max)
        # 척추 측면 압박 → 기본 보통(1)

        risk["손목_골절"]   = get_level(1, age, acc_max, gyr_max)
        # 옆으로 짚는 경우 가능 → 기본 보통(1)

    # 레벨 → 숫자 변환 (대시보드 그래프용)
    LEVEL_TO_PCT = {"낮음": 25, "보통": 50, "높음": 75, "매우높음": 100}

    # 숫자값 추가 (퍼센트로 표시용)
    risk_pct = {}
    for k, v in risk.items():
        risk_pct[f"{k}_pct"] = LEVEL_TO_PCT.get(v, 0)

    risk.update(risk_pct)

    # 위험 등급 결정 (최고 레벨 기준)
    level_order = {"낮음": 0, "보통": 1, "높음": 2, "매우높음": 3}
    max_level_val = max(level_order[v] for k, v in risk.items() if "_pct" not in k and k != "위험_등급")
    if max_level_val >= 3:
        risk["위험_등급"] = "매우높음"  # 즉시 119
    elif max_level_val >= 2:
        risk["위험_등급"] = "높음"      # 즉시 보호자 연락
    elif max_level_val >= 1:
        risk["위험_등급"] = "보통"      # 병원 방문 권장
    else:
        risk["위험_등급"] = "낮음"      # 모니터링

    return risk


def save_to_firestore(db, fall_data, injury_risk):
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    event_id    = f"fall_{timestamp}"
    fall_data   = to_python_type(fall_data)
    injury_risk = to_python_type(injury_risk)

    # 부모 문서에 기본 필드 추가 (빈 문서로 보여서 실수로 삭제 방지)
    db.collection("fall_events").document(PATIENT_ID).set({
        "patient_id": PATIENT_ID,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, merge=True)

    db.collection("fall_events") \
      .document(PATIENT_ID) \
      .collection("events") \
      .document(event_id) \
      .set(fall_data)

    # 부모 문서에 기본 필드 추가
    db.collection("injury_risk").document(PATIENT_ID).set({
        "patient_id": PATIENT_ID,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, merge=True)

    db.collection("injury_risk") \
      .document(PATIENT_ID) \
      .collection("events") \
      .document(event_id) \
      .set(injury_risk)

    db.collection("alerts") \
      .document(event_id) \
      .set({
          "patient_id": PATIENT_ID,
          "event_id":   event_id,
          "timestamp":  fall_data["timestamp"],
          "location":   fall_data["location"],
          "direction":  fall_data["fall_direction"],
          "risk_grade": injury_risk["위험_등급"],
          "confirmed":  False
      })

    return event_id


def extract_features_realtime(window):
    acc = window[:, :3]
    gyr = window[:, 3:]
    f   = {}
    for i, ax in enumerate(["AccX","AccY","AccZ"]):
        c = acc[:, i]
        f[f"{ax}_mean"]  = float(np.mean(c))
        f[f"{ax}_std"]   = float(np.std(c))
        f[f"{ax}_max"]   = float(np.max(c))
        f[f"{ax}_min"]   = float(np.min(c))
        f[f"{ax}_range"] = float(np.ptp(c))
        f[f"{ax}_peak"]  = float(np.max(np.abs(c)))
    for i, ax in enumerate(["GyrX","GyrY","GyrZ"]):
        c = gyr[:, i]
        f[f"{ax}_mean"]  = float(np.mean(c))
        f[f"{ax}_std"]   = float(np.std(c))
        f[f"{ax}_max"]   = float(np.max(c))
        f[f"{ax}_min"]   = float(np.min(c))
        f[f"{ax}_range"] = float(np.ptp(c))
        f[f"{ax}_peak"]  = float(np.max(np.abs(c)))
    f["acc_SMA"]    = float(np.mean(np.sum(np.abs(acc), axis=1)))
    f["gyr_SMA"]    = float(np.mean(np.sum(np.abs(gyr), axis=1)))
    f["acc_energy"] = float(np.sum(acc**2) / len(acc))
    f["gyr_energy"] = float(np.sum(gyr**2) / len(gyr))
    acc_res = np.sqrt(np.sum(acc**2, axis=1))
    f["acc_resultant_mean"] = float(np.mean(acc_res))
    f["acc_resultant_max"]  = float(np.max(acc_res))
    f["acc_resultant_std"]  = float(np.std(acc_res))
    gyr_res = np.sqrt(np.sum(gyr**2, axis=1))
    f["gyr_resultant_mean"] = float(np.mean(gyr_res))
    f["gyr_resultant_max"]  = float(np.max(gyr_res))
    peak_idx = int(np.argmax(acc_res))
    f["peak_position_ratio"] = float(peak_idx / len(acc_res))
    before = float(np.mean(acc_res[:peak_idx+1])) if peak_idx > 0 else 0.0
    after  = float(np.mean(acc_res[peak_idx:]))
    f["peak_before_mean"] = before
    f["peak_after_mean"]  = after
    f["peak_ratio"]       = float(before / after) if after > 0 else 0.0
    f["impact_AccX"] = float(acc[peak_idx, 0])
    f["impact_AccY"] = float(acc[peak_idx, 1])
    f["impact_AccZ"] = float(acc[peak_idx, 2])
    f["impact_GyrX"] = float(gyr[peak_idx, 0])
    f["impact_GyrY"] = float(gyr[peak_idx, 1])
    f["impact_GyrZ"] = float(gyr[peak_idx, 2])
    half = len(acc_res) // 2
    f["acc_std_first_half"]  = float(np.std(acc_res[:half]))
    f["acc_std_second_half"] = float(np.std(acc_res[half:]))
    return f


# ──────────────────────────────────────────────────────────
# 시뮬레이션 모드
# ──────────────────────────────────────────────────────────
def run_simulation(db, model, dir_model, feat_cols):
    print("\n[시뮬레이션 모드] 데이터셋으로 실시간 테스트 중...")
    print("Ctrl+C 로 종료\n")

    df         = pd.read_csv(DATA_FILE)
    samples    = df.sample(20).reset_index(drop=True)
    fall_count = 0

    for i, (idx, row) in enumerate(samples.iterrows()):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {i+1:02d}번째 윈도우 분석 중...", end=" ")

        X         = np.array([row[c] for c in feat_cols]).reshape(1, -1)
        pred      = model.predict(X)[0]
        prob      = model.predict_proba(X)[0][1]
        direction = int(dir_model.predict(X)[0]) if pred == 1 else -1
        match     = "✅" if int(pred) == int(row["is_fall"]) else "❌"

        if pred == 1:
            fall_count += 1
            dir_str  = DIRECTION_MAP.get(direction, "알 수 없음")
            acc_max  = float(row["acc_resultant_max"])
            gyr_max  = float(row["gyr_resultant_max"])
            risk     = calculate_injury_risk(direction, acc_max, gyr_max, PATIENT_INFO["age"])

            print(f"🚨 낙상 감지! ({prob*100:.1f}%) {dir_str} {match}")
            print(f"         충격: {acc_max:.2f}g | 회전: {gyr_max:.1f}°/s | 위험등급: {risk['위험_등급']}")
            for k, v in risk.items():
                if k != "위험_등급":
                    print(f"         {k}: {v}%")

            fall_data = {
                "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "patient_id":       PATIENT_ID,
                "is_fall":          1,
                "fall_probability": round(prob * 100, 1),
                "fall_direction":   dir_str,
                "acc_max_g":        round(acc_max, 3),
                "gyr_max_dps":      round(gyr_max, 1),
                "location": {
                    "latitude":  37.5665,
                    "longitude": 126.9780,
                    "gps_valid": False,
                    "note":      "시뮬레이션 모드 (고정 위치)"
                },
                "model_used": MODEL_NAME
            }
            event_id = save_to_firestore(db, fall_data, risk)
            print(f"         Firestore 전송 완료: {event_id} ✅")
        else:
            print(f"정상 (ADL) ({prob*100:.1f}%) {match}")

        time.sleep(1)

    print(f"\n{'='*50}")
    print(f"시뮬레이션 완료!")
    print(f"총 20개 중 낙상 {fall_count}개 감지")
    print(f"{'='*50}")


# ──────────────────────────────────────────────────────────
# 실제 센서 모드 (MPU-6050 + NEO-6M GPS)
# ──────────────────────────────────────────────────────────
def run_real_sensor(db, model, dir_model, feat_cols, gps):
    try:
        import smbus2
    except ImportError:
        print("smbus2 없음! pip install smbus2 --break-system-packages")
        return

    bus     = smbus2.SMBus(1)
    MPU6050 = 0x68
    bus.write_byte_data(MPU6050, 0x6B, 0)
    bus.write_byte_data(MPU6050, 0x1C, 0x18)  # ±16g
    bus.write_byte_data(MPU6050, 0x1B, 0x18)  # ±2000°/s

    print("\n[실제 센서 모드] MPU-6050 + NEO-6M GPS 실행 중!")
    print("Ctrl+C 로 종료\n")

    def read_raw(addr):
        high = bus.read_byte_data(MPU6050, addr)
        low  = bus.read_byte_data(MPU6050, addr + 1)
        val  = (high << 8) | low
        return val - 65536 if val > 32767 else val

    ACC_SCALE       = 16.0 / 32768.0
    GYR_SCALE       = 2000.0 / 32768.0
    buffer          = []
    fall_count      = 0
    battery_level   = BATTERY_LEVEL
    battery_timer   = time.time()

    # 배터리: Firebase 전송 없이 터미널 출력만 (사용량 절약)
    print(f"  🔋 배터리: {battery_level}%")

    try:
        while True:
            ax = read_raw(0x3B) * ACC_SCALE
            ay = read_raw(0x3D) * ACC_SCALE
            az = read_raw(0x3F) * ACC_SCALE
            gx = read_raw(0x43) * GYR_SCALE
            gy = read_raw(0x45) * GYR_SCALE
            gz = read_raw(0x47) * GYR_SCALE

            # 저역통과 필터 적용 (노이즈 제거)
            filtered = apply_filter([ax, ay, az, gx, gy, gz])
            buffer.append(filtered)

            if len(buffer) >= WINDOW_SIZE:
                window = np.array(buffer[-WINDOW_SIZE:])
                feats  = extract_features_realtime(window)
                X      = np.array([feats[c] for c in feat_cols]).reshape(1, -1)
                pred   = model.predict(X)[0]
                prob   = model.predict_proba(X)[0][1]
                now    = datetime.now().strftime("%H:%M:%S")

                acc_max = float(np.max(np.sqrt(np.sum(window[:, :3]**2, axis=1))))
                gyr_max = float(np.max(np.sqrt(np.sum(window[:, 3:]**2, axis=1))))

                # 낙상 판단: 확률 90% 이상 (FALL_THRESHOLD, 오탐 방지 우선)
                if pred == 1 and prob >= FALL_THRESHOLD:

                    is_confirmed = False  # 초기화
                    direction = int(dir_model.predict(X)[0])
                    dir_str   = DIRECTION_MAP.get(direction, "알 수 없음")
                    age       = PATIENT_INFO["age"] if PATIENT_INFO else 65
                    risk      = calculate_injury_risk(direction, acc_max, gyr_max, age)
                    location  = gps.get_location()
                    reason    = get_fall_reason(acc_max, gyr_max, direction, prob)

                    # 주요 부상 추출 + 조치 가이드 + 기저질환
                    main_injury, second_injury = get_main_injury(risk)
                    medical_history = PATIENT_INFO.get("medical_history", "없음") if PATIENT_INFO else "없음"
                    first_aid = get_first_aid_guide(direction, risk, medical_history)

                    print(f"\n[{now}] 🚨 낙상 감지! ({prob*100:.1f}%)")
                    print(f"  방향:     {dir_str}")
                    print(f"  충격:     {acc_max:.2f}g | 회전: {gyr_max:.1f}°/s")
                    print(f"  위험등급: {risk['위험_등급']}")
                    print(f"  근거:     {reason}")
                    if main_injury:
                        print(f"  주요부상:  {main_injury[0]} → {main_injury[1]}")
                    if second_injury:
                        print(f"  2차부상:   {second_injury[0]} → {second_injury[1]}")
                    print(f"  조치가이드:")
                    for g in first_aid:
                        print(f"    {g}")

                    # ── Logic A: 심각한 낙상 → 즉시 알림 ──
                    if acc_max >= IMMEDIATE_ACC or gyr_max >= IMMEDIATE_GYR:
                        print(f"  ⚡ [Logic A] 심각한 낙상 → 즉시 알림!")
                        buzz("urgent")
                        is_confirmed = True

                    # ── Logic B: 경미한 낙상 → 버튼 대기 (DEMO_MODE 시 스킵) ──
                    elif DEMO_MODE:
                        print(f"  ⚡ [시연 모드] 오탐 취소 기능 비활성화 → 즉시 전송!")
                        buzz("urgent")
                        is_confirmed = True
                    else:
                        print(f"  ⏳ [Logic B] 경미한 낙상 → 15초 내 버튼 누르면 취소!")
                        buzz("short")
                        cancelled = wait_for_button(BUTTON_WAIT_SEC)

                        if cancelled:
                            print(f"  ✅ 버튼 눌림 → 오탐으로 처리! Firebase에 기록")
                            db.collection("false_alarms").document().set({
                                "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "patient_id": PATIENT_ID,
                                "acc_max_g":  round(acc_max, 3),
                                "gyr_max_dps": round(gyr_max, 1),
                                "reason":     reason
                            })
                            buffer = []
                            for lpf in filters:
                                lpf.reset()
                            time.sleep(COOLDOWN_SEC)
                            continue
                        else:
                            print(f"  🚨 응답 없음 → 진짜 낙상으로 처리!")
                            is_confirmed = True

                    # ── 낙상 확정 → Firebase 전송 ──
                    if is_confirmed:
                        fall_count += 1
                        if location["gps_valid"]:
                            print(f"  위치:     위도 {location['latitude']}, 경도 {location['longitude']} ✅")
                        else:
                            print(f"  위치:     GPS 신호 없음 (실내)")

                        fall_data = {
                            "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "patient_id":       PATIENT_ID,
                            "is_fall":          1,
                            "fall_probability": round(prob * 100, 1),
                            "fall_direction":   dir_str,
                            "acc_max_g":        round(acc_max, 3),
                            "gyr_max_dps":      round(gyr_max, 1),
                            "location":         location,
                            "model_used":       MODEL_NAME,
                            "reason":           reason,
                            "immediate":        acc_max >= IMMEDIATE_ACC or gyr_max >= IMMEDIATE_GYR,
                            "main_injury":      f"{main_injury[0]} ({main_injury[1]})" if main_injury else "없음",
                            "second_injury":    f"{second_injury[0]} ({second_injury[1]})" if second_injury else "없음",
                            "first_aid_guide":  first_aid,
                            "medical_history":  medical_history,
                        }
                        event_id = save_to_firestore(db, fall_data, risk)
                        print(f"  Firestore 전송 완료: {event_id} ✅")
                        buffer = []
                        for lpf in filters:
                            lpf.reset()
                        time.sleep(COOLDOWN_SEC)
                else:
                    print(f"[{now}] 정상 ({prob*100:.1f}%) 🔋{battery_level}%", end="\r")

                # 배터리 1분마다 1% 감소
                if time.time() - battery_timer >= BATTERY_INTERVAL:
                    battery_level = max(0, battery_level - 1)
                    battery_timer = time.time()
                    update_battery(db, PATIENT_ID, battery_level)

            time.sleep(1.0 / SAMPLING_RATE)

    except KeyboardInterrupt:
        print(f"\n\n종료! 총 낙상 감지: {fall_count}회")
        gps.stop()


# ──────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────
print()
print("┌─────────────────────────────────────────────────────┐")
print(f"│   낙상 감지 벨트 실시간 감지  [{MODEL_NAME}]")
print(f"│   모드: {'시뮬레이션' if SIMULATION_MODE else '실제 센서 (MPU-6050 + NEO-6M GPS)'}")
print("└─────────────────────────────────────────────────────┘")

# Firebase 연결
print("\n[1] Firebase 연결 중...")
try:
    cred = credentials.Certificate(KEY_FILE)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("  연결 성공! ✅")
except Exception as e:
    print(f"  연결 실패: {e}")
    exit()

# 모델 로딩
print("\n[2] 모델 로딩 중...")
with open(MODEL_FILE, "rb") as f: model     = pickle.load(f)
with open(DIR_FILE,   "rb") as f: dir_model = pickle.load(f)
with open(FEAT_FILE,  "rb") as f: feat_cols = pickle.load(f)
print("  로딩 완료! ✅")

# 환자 ID 입력
print()
print("=" * 50)
PATIENT_ID = input("  환자 ID 입력 (예: patient_001): ").strip()
if not PATIENT_ID:
    PATIENT_ID = "patient_001"

# Firebase에서 환자 정보 읽어오기
print(f"  환자 정보 조회 중...")
patient_doc = db.collection("patients").document(PATIENT_ID).get()

if patient_doc.exists:
    PATIENT_INFO = patient_doc.to_dict()
    print(f"  환자: {PATIENT_INFO.get('name', '이름없음')} ({PATIENT_INFO.get('age', 0)}세) ✅")
else:
    # Firebase에 없으면 기본값으로 새로 생성
    print(f"  신규 환자 등록")
    name    = input("  이름: ").strip() or "홍길동"
    age     = int(input("  나이: ").strip() or "70")
    gender  = input("  성별 (M/F): ").strip() or "M"
    height  = int(input("  키 (cm): ").strip() or "170")
    weight  = int(input("  몸무게 (kg): ").strip() or "65")
    contact = input("  보호자 연락처: ").strip() or "010-1234-5678"
    medical = input("  기저질환 (없으면 엔터): ").strip() or "없음"
    blood   = input("  혈액형 (A/B/O/AB): ").strip() or "A"

    PATIENT_INFO = {
        "name":             name,
        "age":              age,
        "gender":           gender,
        "height_cm":        height,
        "weight_kg":        weight,
        "guardian_contact": contact,
        "medical_history":  medical,
        "blood_type":       blood,
    }
    db.collection("patients").document(PATIENT_ID).set(PATIENT_INFO)
    print(f"  Firebase에 저장 완료 ✅")
print("=" * 50)

# GPS 초기화
gps = GPSReader()
if not SIMULATION_MODE:
    print("\n[3] GPS 초기화 중...")
    gps.start()

# 실행
if SIMULATION_MODE:
    run_simulation(db, model, dir_model, feat_cols)
else:
    run_real_sensor(db, model, dir_model, feat_cols, gps)
