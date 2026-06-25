"""
XGBoost 재학습 코드
====================
기존 combined_features.csv + custom_data.csv 합쳐서 재학습

실행: python retrain_xgboost.py
필요: pip install scikit-learn xgboost numpy pandas
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# ──────────────────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
COMBINED_CSV    = os.path.join(BASE_DIR, "combined_features.csv")
CUSTOM_CSV      = os.path.join(BASE_DIR, "custom_data_20260520_174846.csv")
SAVE_DIR        = os.path.join(BASE_DIR, "XGBoost_Model")

os.makedirs(SAVE_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────
# 데이터 로드 및 합치기
# ──────────────────────────────────────────────────────────
print("=" * 55)
print("  XGBoost 재학습")
print("=" * 55)

print("\n[1] 데이터 로딩 중...")
df_orig   = pd.read_csv(COMBINED_CSV)
df_custom = pd.read_csv(CUSTOM_CSV)

print(f"  기존 데이터:  {len(df_orig):,}개")
print(f"  직접 측정:    {len(df_custom):,}개")

# 합치기
df = pd.concat([df_orig, df_custom], ignore_index=True)
df = df.fillna(0)

print(f"  통합 데이터:  {len(df):,}개")
print(f"  낙상:         {df['is_fall'].sum():,}개 ({df['is_fall'].mean()*100:.1f}%)")
print(f"  ADL:          {(df['is_fall']==0).sum():,}개 ({(df['is_fall']==0).mean()*100:.1f}%)")

# ──────────────────────────────────────────────────────────
# Feature 컬럼 선택
# ──────────────────────────────────────────────────────────
exclude_cols = ["subject", "action", "is_fall", "fall_direction", "source"]
feat_cols    = [c for c in df.columns if c not in exclude_cols]

# 숫자형 컬럼만 사용
feat_cols = [c for c in feat_cols if df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

X = df[feat_cols].values
y = df["is_fall"].values
y_dir = df["fall_direction"].values
groups = df["subject"].values

print(f"\n  Feature 수: {len(feat_cols)}개")

# ──────────────────────────────────────────────────────────
# 피험자 기준 학습/테스트 분리
# ──────────────────────────────────────────────────────────
print("\n[2] 피험자 기준 데이터 분리 중...")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
y_dir_train, y_dir_test = y_dir[train_idx], y_dir[test_idx]

print(f"  학습: {len(X_train):,}개 / 테스트: {len(X_test):,}개")

# ──────────────────────────────────────────────────────────
# 클래스 불균형 처리
# ──────────────────────────────────────────────────────────
n_adl  = (y_train == 0).sum()
n_fall = (y_train == 1).sum()
scale  = round(n_adl / n_fall, 1)
print(f"  클래스 가중치: {scale}:1")

# ──────────────────────────────────────────────────────────
# 낙상 감지 모델 학습
# ──────────────────────────────────────────────────────────
print("\n[3] XGBoost 낙상 감지 모델 학습 중...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale,
    random_state=42,
    eval_metric="logloss",
    verbosity=0
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred) * 100
sen  = recall_score(y_test, y_pred) * 100
pre  = precision_score(y_test, y_pred) * 100
f1   = f1_score(y_test, y_pred) * 100
auc  = roc_auc_score(y_test, y_prob) * 100

print(f"\n  [낙상 감지 결과]")
print(f"  정확도:   {acc:.2f}%")
print(f"  민감도:   {sen:.2f}%")
print(f"  정밀도:   {pre:.2f}%")
print(f"  F1 Score: {f1:.2f}%")
print(f"  AUC-ROC:  {auc:.2f}%")

# ──────────────────────────────────────────────────────────
# 방향 분류 모델 학습
# ──────────────────────────────────────────────────────────
print("\n[4] XGBoost 방향 분류 모델 학습 중...")

fall_train = y_train == 1
fall_test  = y_test == 1

dir_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric="mlogloss",
    verbosity=0
)
dir_model.fit(X_train[fall_train], y_dir_train[fall_train])

dir_pred = dir_model.predict(X_test[fall_test])
dir_acc  = accuracy_score(y_dir_test[fall_test], dir_pred) * 100

print(f"  방향 분류 정확도: {dir_acc:.2f}%")

# ──────────────────────────────────────────────────────────
# 모델 저장
# ──────────────────────────────────────────────────────────
print("\n[5] 모델 저장 중...")

with open(os.path.join(SAVE_DIR, "xgb_model.pkl"), "wb") as f:
    pickle.dump(model, f)
with open(os.path.join(SAVE_DIR, "xgb_direction_model.pkl"), "wb") as f:
    pickle.dump(dir_model, f)
with open(os.path.join(SAVE_DIR, "xgb_feature_cols.pkl"), "wb") as f:
    pickle.dump(feat_cols, f)

print(f"  저장 완료!")
print(f"  위치: {SAVE_DIR}")

print(f"\n{'='*55}")
print(f"재학습 완료!")
print(f"  낙상 감지 정확도: {acc:.2f}%")
print(f"  낙상 감지 민감도: {sen:.2f}%")
print(f"  방향 분류 정확도: {dir_acc:.2f}%")
print(f"{'='*55}")
print(f"\n다음 단계:")
print(f"  1. XGBoost_Model 폴더의 pkl 파일들 라즈베리파이에 복사")
print(f"  2. 실제 테스트!")
