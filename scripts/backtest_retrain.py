# scripts/backtest_retrain_multi.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from HCM_temp_forcast.data import load_from_parquet
from HCM_temp_forcast.prepare import basic_clean, feature_engineer_timeseries

# ================== CONFIG ==================
DATA_PATH   = "data/weather.parquet"
DATE_COL    = "datetime"
TARGET_COL  = "target_temp_tplus"

# Chỉ backtest horizon 1 (t+1) làm đại diện
HORIZONS    = (1,)

# Số tháng dùng làm warm-up ban đầu
INIT_TRAIN_MONTHS = 12

RANDOM_STATE = 42
# ============================================


@dataclass
class BacktestResult:
    horizon: int
    policy: str           # never_retrain / retrain_1m / retrain_3m
    granularity: str      # luôn là "month"
    period: str           # "YYYY-MM"
    n_train: int
    n_test: int
    mae: float
    rmse: float
    r2: float


def build_rf_model(seed: int = RANDOM_STATE) -> RandomForestRegressor:
    """
    RandomForest cố định để so sánh giữa các policy,
    không làm hyper-parameter tuning ở đây.
    """
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed,
    )


def prepare_fe_for_horizon(
    df: pd.DataFrame,
    h: int
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    Chạy feature engineering cho toàn bộ lịch sử với horizon h.
    Giữ lại cột datetime để group theo tháng.
    """
    fe = feature_engineer_timeseries(
        df,
        horizon=h,
        make_target=True,
        drop_original_astro=True,
        drop_datetime_cols=False,  # GIỮ datetime
    )

    if DATE_COL not in fe.columns:
        raise KeyError(f"'{DATE_COL}' không có trong FE dataframe cho H={h}")

    dt = pd.to_datetime(fe[DATE_COL])
    y = fe[TARGET_COL]
    feat_cols = [c for c in fe.columns if c not in (TARGET_COL, DATE_COL)]
    X = fe[feat_cols]
    return dt, X, y


# ---------- UTILS CHO THÁNG ----------

def _month_code(dt: pd.Series) -> np.ndarray:
    """Mã hoá YYYYMM thành số nguyên để sort / index."""
    return dt.dt.year.values * 100 + dt.dt.month.values


def _month_label(code: int) -> str:
    year = code // 100
    mon  = code % 100
    return f"{year}-{mon:02d}"


# ---------- CÁC POLICY THEO THÁNG ----------

def backtest_never_retrain_monthly(
    dt: pd.Series,
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int
) -> List[BacktestResult]:
    """
    Policy 1: train một lần trên INIT_TRAIN_MONTHS tháng đầu tiên,
    sau đó giữ nguyên model để dự báo tất cả các tháng còn lại.
    """
    ym = _month_code(dt)
    uniq = np.unique(ym)
    uniq.sort()

    if len(uniq) <= INIT_TRAIN_MONTHS:
        raise ValueError("Không đủ tháng để backtest never_retrain.")

    # Warm-up: train trên INIT_TRAIN_MONTHS tháng đầu
    train_codes = uniq[:INIT_TRAIN_MONTHS]
    train_mask = np.isin(ym, train_codes)
    X_train, y_train = X[train_mask], y[train_mask]

    model = build_rf_model()
    model.fit(X_train, y_train)

    results: List[BacktestResult] = []

    # Test trên từng tháng sau warm-up
    for i in range(INIT_TRAIN_MONTHS, len(uniq)):
        test_code = uniq[i]
        test_mask = (ym == test_code)

        if test_mask.sum() == 0:
            continue

        X_test, y_test = X[test_mask], y[test_mask]
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2   = r2_score(y_test, y_pred)

        results.append(
            BacktestResult(
                horizon=horizon,
                policy="never_retrain",
                granularity="month",
                period=_month_label(test_code),
                n_train=int(train_mask.sum()),
                n_test=int(test_mask.sum()),
                mae=float(mae),
                rmse=float(rmse),
                r2=float(r2),
            )
        )
    return results


def backtest_monthly_retrain_expanding(
    dt: pd.Series,
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    retrain_every_months: int,
    policy_name: str,
) -> List[BacktestResult]:
    """
    Policy retrain định kỳ (expanding):
      - Mỗi lần train lại dùng TOÀN BỘ dữ liệu từ đầu tới cuối tháng trước.
      - retrain_every_months = 1: retrain mỗi tháng.
      - retrain_every_months = 3: 3 tháng mới retrain 1 lần.
    Trong cùng 1 block (VD: 3 tháng), các tháng dùng chung 1 model.
    """
    ym = _month_code(dt)
    uniq = np.unique(ym)
    uniq.sort()

    if len(uniq) <= INIT_TRAIN_MONTHS:
        raise ValueError("Không đủ tháng để backtest policy retrain.")

    results: List[BacktestResult] = []
    model = None
    last_train_end_index = None

    # Chỉ bắt đầu backtest sau khi đã có INIT_TRAIN_MONTHS tháng dữ liệu
    for i in range(INIT_TRAIN_MONTHS, len(uniq)):
        # Nếu là thời điểm cần retrain (tính từ sau warm-up)
        if (i - INIT_TRAIN_MONTHS) % retrain_every_months == 0 or model is None:
            # Train trên tất cả tháng trước tháng test hiện tại
            train_codes = uniq[:i]              # 0..i-1
            train_mask = np.isin(ym, train_codes)

            if train_mask.sum() == 0:
                continue

            X_train, y_train = X[train_mask], y[train_mask]
            model = build_rf_model()
            model.fit(X_train, y_train)
            last_train_end_index = i

        test_code = uniq[i]
        test_mask = (ym == test_code)
        if test_mask.sum() == 0:
            continue

        X_test, y_test = X[test_mask], y[test_mask]
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2   = r2_score(y_test, y_pred)

        n_train = int(np.isin(ym, uniq[:last_train_end_index]).sum())

        results.append(
            BacktestResult(
                horizon=horizon,
                policy=policy_name,
                granularity="month",
                period=_month_label(test_code),
                n_train=n_train,
                n_test=int(test_mask.sum()),
                mae=float(mae),
                rmse=float(rmse),
                r2=float(r2),
            )
        )
    return results


# ---------- VẼ MAE GOM THEO 3 THÁNG (QUÝ) ----------

def plot_mae_quarterly(df_res: pd.DataFrame, horizon: int):
    """
    Gom MAE theo quý (3 tháng) để đồ thị thoáng hơn.
    Mỗi policy: trung bình MAE của 3 tháng trong cùng 1 quý.
    """
    sub = df_res[(df_res["horizon"] == horizon) &
                 (df_res["granularity"] == "month")].copy()
    if sub.empty:
        print(f"[WARN] Không có kết quả cho H{horizon} / month")
        return

    # period: "YYYY-MM" -> datetime -> period quý
    sub["period_dt"] = pd.to_datetime(sub["period"] + "-01")
    sub["quarter"] = sub["period_dt"].dt.to_period("Q").astype(str)  # vd: "2021Q3"

    # Tính MAE trung bình mỗi quý cho từng policy
    agg = (
        sub.groupby(["policy", "quarter"], as_index=False)
           .agg({"mae": "mean"})
           .sort_values("quarter")
    )

    plt.figure(figsize=(10, 4))
    for policy, grp in agg.groupby("policy"):
        plt.plot(grp["quarter"], grp["mae"], marker="o", label=policy)

    plt.title(f"MAE theo quý (3 tháng) – Horizon H{horizon} (so sánh policy retrain)")
    plt.xlabel("Quý (YYYYQn, theo ngày cơ sở t)")
    plt.ylabel("MAE (°C)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


# ---------- MAIN ----------

def main():
    print("==> Load parquet & basic_clean")
    df_raw = load_from_parquet(DATA_PATH)
    df = basic_clean(df_raw, drop_text_cols=True, drop_na_core=True)
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    all_results: List[BacktestResult] = []

    for h in HORIZONS:  # hiện tại chỉ có H=1
        print(f"\n==== Horizon H{h} (t+{h}) ====")
        dt, X, y = prepare_fe_for_horizon(df, h)
        print(f"FE H{h}: X={X.shape}, y={y.shape}, range={dt.min().date()}..{dt.max().date()}")

        # Policy 1: không retrain
        all_results.extend(
            backtest_never_retrain_monthly(dt, X, y, horizon=h)
        )

        # Policy 2: retrain mỗi tháng (expanding)
        all_results.extend(
            backtest_monthly_retrain_expanding(
                dt, X, y,
                horizon=h,
                retrain_every_months=1,
                policy_name="retrain_1m",
            )
        )

        # Policy 3: retrain 3 tháng 1 lần (expanding)
        all_results.extend(
            backtest_monthly_retrain_expanding(
                dt, X, y,
                horizon=h,
                retrain_every_months=3,
                policy_name="retrain_3m",
            )
        )

    df_res = pd.DataFrame([r.__dict__ for r in all_results])
    print("\n=== Backtest summary (head) ===")
    print(df_res.head())

    out_csv = "backtest_retrain_policies_H1.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"[OK] Saved backtest results -> {out_csv}")

    # Vẽ MAE theo QUÝ cho H1
    plot_mae_quarterly(df_res, horizon=1)
    plt.show()


if __name__ == "__main__":
    main()
