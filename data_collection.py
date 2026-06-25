"""
직접 측정 데이터 수집 코드
============================
낙상 감지 스마트 벨트 - My_Custom 데이터 수집
MPU-6050 센서로 직접 데이터 수집 후 XGBoost 재학습 가능

실행: python3 data_collection.py
필요: pip install smbus2 numpy pandas --break-system-packages

측정 방법:
  ADL → 30초씩 수집 (걷기/앉기/서있기)
  낙상 → 5초씩 수집 (앞/뒤/옆)
  매트 위에서 안전하게 측정!
"""

import smbus2
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime

# ──────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────
SAMPLING_RATE = 100   # Hz (KFall과 동일)
WINDOW_SIZE   = 100   # 1초 윈도우
SAVE_DIR      = "/home/ttjjdd9340/custom_data"
SUBJECT_ID    = "custom_001"

# 동작별 설정 (label, is_fall, direction, duration_sec)
ACTIONS = {
    "1": ("walking",     0, -1, 30, "걷기",         "일직선으로 자연스럽게 걷기 반복"),
    "2": ("sitting",     0, -1, 30, "앉기/일어서기", "의자에 천천히 앉았다 일어서기 반복"),
    "3": ("standing",    0, -1, 20, "서있기",        "제자리에 자연스럽게 서있기"),
    "4": ("fall_front",  1,  0,  5, "앞으로 낙상",  "매트에 무릎 먼저 닿게 앞으로 넘어지기"),
    "5": ("fall_back",   1,  1,  5, "뒤로 낙상",    "매트에 엉덩이 먼저 닿게 뒤로 넘어지기"),
    "6": ("fall_side",   1,  2,  5, "옆으로 낙상",  "매트에 엉덩이 옆으로 닿게 옆으로 넘어지기"),
}

# ──────────────────────────────────────────────────────────
# MPU-6050 초기화
# ──────────────────────────────────────────────────────────
bus     = smbus2.SMBus(1)
MPU6050 = 0x68
bus.write_byte_data(MPU6050, 0x6B, 0)
bus.write_byte_data(MPU6050, 0x1C, 0x18)  # ±16g (낙상 충격 측정)
bus.write_byte_data(MPU6050, 0x1B, 0x18)  # ±2000°/s (빠른 회전 측정)

ACC_SCALE = 16.0 / 32768.0
GYR_SCALE = 2000.0 / 32768.0

def read_raw(addr):
    high = bus.read_byte_data(MPU6050, addr)
    low  = bus.read_byte_data(MPU6050, addr + 1)
    val  = (high << 8) | low
    return val - 65536 if val > 32767 else val

def read_sensor():
    return [
        read_raw(0x3B) * ACC_SCALE,  # AccX
        read_raw(0x3D) * ACC_SCALE,  # AccY
        read_raw(0x3F) * ACC_SCALE,  # AccZ
        read_raw(0x43) * GYR_SCALE,  # GyrX
        read_raw(0x45) * GYR_SCALE,  # GyrY
        read_raw(0x47) * GYR_SCALE,  # GyrZ
    ]

# ──────────────────────────────────────────────────────────
# Feature 추출 (combined_features.csv 와 동일한 형식)
# ──────────────────────────────────────────────────────────
def extract_features(window, action, is_fall, direction):
    acc = window[:, :3]
    gyr = window[:, 3:]
    f = {
        "subject":        SUBJECT_ID,
        "action":         action,
        "is_fall":        is_fall,
        "fall_direction": direction,
        "source":         "Custom"
    }

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
# 데이터 수집
# ──────────────────────────────────────────────────────────
def collect_data(action, is_fall, direction, duration_sec):
    print(f"\n  3초 후 시작합니다. 준비하세요!")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print(f"  수집 시작! ({duration_sec}초)")

    buffer  = []
    windows = []
    start   = time.time()
    elapsed = 0

    while elapsed < duration_sec:
        data = read_sensor()
        buffer.append(data)

        if len(buffer) >= WINDOW_SIZE:
            window = np.array(buffer[-WINDOW_SIZE:])
            feats  = extract_features(window, action, is_fall, direction)
            windows.append(feats)
            buffer = []

        elapsed = time.time() - start
        remaining = duration_sec - elapsed
        print(f"  남은 시간: {remaining:.1f}초 | 수집: {len(windows)}개", end="\r")
        time.sleep(1.0 / SAMPLING_RATE)

    print(f"\n  완료! ({len(windows)}개 윈도우 수집)")
    return windows

# ──────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────
os.makedirs(SAVE_DIR, exist_ok=True)
all_data = []
counts   = {k: 0 for k in ACTIONS}

print()
print("=" * 55)
print("  낙상 감지 벨트 - 직접 측정 데이터 수집")
print("=" * 55)
print()
print("  벨트를 허리 뒤 중앙에 착용하세요!")
print("  낙상 측정 시 반드시 매트 위에서 하세요!")
print()

while True:
    print("\n" + "─" * 55)
    print("  [동작 선택]")
    for k, (action, is_fall, direction, duration, label, guide) in ACTIONS.items():
        tag = "낙상" if is_fall else "ADL"
        print(f"  {k}. [{tag}] {label} ({duration}초) - {guide}")
    print("  s. 저장 후 종료")
    print("─" * 55)

    print("\n  현재 수집 현황:")
    total_fall = 0
    total_adl  = 0
    for k, (action, is_fall, direction, duration, label, guide) in ACTIONS.items():
        print(f"  {label}: {counts[k]}회", end="  ")
        if is_fall:
            total_fall += counts[k]
        else:
            total_adl += counts[k]
    print(f"\n  총 ADL: {total_adl}회 | 총 낙상: {total_fall}회")

    choice = input("\n  선택 (1~6 / s): ").strip().lower()

    if choice == "s":
        break
    elif choice in ACTIONS:
        action, is_fall, direction, duration, label, guide = ACTIONS[choice]

        print(f"\n  [{label}] 준비!")
        if is_fall:
            print(f"  ⚠️  매트 위에서 안전하게 {guide}")
        else:
            print(f"  ✅  {guide}")

        data = collect_data(action, is_fall, direction, duration)
        all_data.extend(data)
        counts[choice] += 1
    else:
        print("  잘못된 입력이에요!")

# ──────────────────────────────────────────────────────────
# 저장
# ──────────────────────────────────────────────────────────
if all_data:
    df       = pd.DataFrame(all_data)
    filename = os.path.join(SAVE_DIR, f"custom_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(filename, index=False)

    fall_count = df['is_fall'].sum()
    adl_count  = (df['is_fall']==0).sum()

    print(f"\n{'='*55}")
    print(f"저장 완료!")
    print(f"  파일:     {filename}")
    print(f"  총 데이터: {len(df)}개")
    print(f"  낙상:     {fall_count}개 ({fall_count/len(df)*100:.1f}%)")
    print(f"  ADL:      {adl_count}개 ({adl_count/len(df)*100:.1f}%)")
    print(f"{'='*55}")
    print(f"\n다음 단계:")
    print(f"  1. 파일을 노트북으로 가져오기")
    print(f"  2. combined_features.csv 에 합치기")
    print(f"  3. XGBoost 재학습")
else:
    print("\n저장할 데이터가 없어요!")
