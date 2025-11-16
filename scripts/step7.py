# scripts/step7_monitor_mae.py
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error

from HCM_temp_forcast.data import load_from_parquet
from HCM_temp_forcast.prepare import (
    basic_clean,
    time_split_train_test,
    feature_engineer_timeseries,
)

# ==== CONFIG (giống tune.py càng nhiều càng tốt) ====
DATA_PATH  = "data/weather.parquet"
TRAIN_FRAC = 0.10
GAP_LMAX   = 30
HORIZONS   = (1, 2, 3, 4, 5)
DATE_COL   = "datetime"
TARGET_COL = "target_temp_tplus"

N_ESTIMATORS = 300
MAX_DEPTH    = None
RANDOM_STATE = 42
K_BEST       = 64   # hoặc None -> lấy hết
# ================================================


def build_rf_pipeline(n_features: int) -> Pipeline:
    """
    Pipeline đơn giản: SelectKBest -> RandomForest
    (không cần StandardScaler cho RF).
    """
    k = K_BEST if (K_BEST is not None and K_BEST > 0 and K_BEST <= n_features) else n_features

    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("select", SelectKBest(score_func=f_regression, k=k)),
            ("model", rf),
        ]
    )
    return pipe


def main():
    # 1) Load + clean
    df_raw = load_from_parquet(DATA_PATH)
    df = basic_clean(df_raw, drop_text_cols=True, drop_na_core=True)

    # 2) Split theo thời gian
    train_df, test_df, _ = time_split_train_test(
        df,
        train_frac=TRAIN_FRAC,
        gap=GAP_LMAX,
        datetime_col=DATE_COL,
    )

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # 3) Train + evaluate cho từng horizon
    results_per_h: dict[int, pd.DataFrame] = {}

    for h in HORIZONS:
        print(f"\n==== Horizon H={h} ====")

        # FE trên train/test, GIỮ datetime để theo dõi theo thời gian
        train_fe = feature_engineer_timeseries(
            train_df,
            horizon=h,
            make_target=True,
            drop_original_astro=True,
            drop_datetime_cols=False,  # giữ datetime
        )
        test_fe = feature_engineer_timeseries(
            test_df,
            horizon=h,
            make_target=True,
            drop_original_astro=True,
            drop_datetime_cols=False,  # giữ datetime
        )

        # Bỏ những hàng thiếu target nếu có
        train_fe = train_fe.dropna(subset=[TARGET_COL])
        test_fe  = test_fe.dropna(subset=[TARGET_COL])

        # Lấy base_date (ngày t – tức ngày mình dùng để dự báo t+h)
        base_dates_test = pd.to_datetime(test_fe[DATE_COL]).dt.date

        # Chọn feature cols (loại datetime, sunrise, sunset, target)
        drop_cols = {TARGET_COL, DATE_COL, "sunrise", "sunset"}
        feat_cols = [c for c in train_fe.columns if c not in drop_cols]

        X_tr, y_tr = train_fe[feat_cols], train_fe[TARGET_COL]
        X_te, y_te = test_fe[feat_cols], test_fe[TARGET_COL]

        print(f"[H{h}] #features = {len(feat_cols)}, "
              f"train={X_tr.shape}, test={X_te.shape}")

        # 4) Build RF pipeline + train
        pipe = build_rf_pipeline(n_features=X_tr.shape[1])
        pipe.fit(X_tr, y_tr)

        # 5) Predict trên test, tính lỗi theo thời gian
        y_pred = pipe.predict(X_te)
        abs_err = np.abs(y_pred - y_te.values)
        mae_overall = mean_absolute_error(y_te, y_pred)
        print(f"[H{h}] Test MAE overall = {mae_overall:.4f}")

        df_eval = pd.DataFrame(
            {
                "base_date": base_dates_test,
                "y_true": y_tr.dtype.type(y_te.values) if hasattr(y_tr, "dtype") else y_te.values,
                "y_pred": y_pred,
                "abs_err": abs_err,
            }
        ).sort_values("base_date")

        # Rolling MAE (ví dụ 30 ngày)
        df_eval["mae_roll_30"] = (
            df_eval["abs_err"].rolling(window=90, min_periods=10).mean()
        )

        results_per_h[h] = df_eval

    # 6) Vẽ MAE theo thời gian cho 5 horizons
    plt.figure(figsize=(12, 6))

    for h in HORIZONS:
        df_eval = results_per_h[h]
        plt.plot(
            df_eval["base_date"],
            df_eval["mae_roll_30"],
            label=f"H{h} - rolling MAE (30d)",
        )

    plt.title("Rolling MAE (30 ngày) trên tập test theo thời gian")
    plt.xlabel("Ngày cơ sở (t)")
    plt.ylabel("MAE (°C)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True, alpha=0.3)

    plt.show()


if __name__ == "__main__":
    main()
