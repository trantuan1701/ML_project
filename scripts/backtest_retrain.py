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
HORIZONS    = (1, 2, 3, 4, 5)

INIT_TRAIN_YEARS    = 5    # số năm đầu dùng để warm-up cho policy theo năm
INIT_TRAIN_MONTHS   = 12   # số tháng đầu dùng để warm-up cho policy monthly_expanding
ROLLING_WINDOW_MONS = 12   # cửa sổ 12 tháng cho monthly_rolling_12m

RANDOM_STATE = 42
# ============================================


@dataclass
class BacktestResult:
    horizon: int
    policy: str           # never_retrain / yearly_expanding / monthly_expanding / monthly_rolling_12m
    granularity: str      # "year" hoặc "month"
    period: str           # "2022" hoặc "2022-05"
    n_train: int
    n_test: int
    mae: float
    rmse: float
    r2: float


def build_rf_model(seed: int = RANDOM_STATE) -> RandomForestRegressor:
    """RandomForest cố định để so sánh policy, không tối ưu hyper-param ở đây."""
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed,
    )


def prepare_fe_for_horizon(df: pd.DataFrame, h: int) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    Chạy FE full history cho 1 horizon.
    Giữ lại cột datetime để chia theo năm / tháng.
    """
    fe = feature_engineer_timeseries(
        df,
        horizon=h,
        make_target=True,
        drop_original_astro=True,
        drop_datetime_cols=False,   # GIỮ datetime
    )

    if DATE_COL not in fe.columns:
        raise KeyError(f"'{DATE_COL}' không có trong FE dataframe cho H={h}")

    dt = pd.to_datetime(fe[DATE_COL])
    y = fe[TARGET_COL]
    feat_cols = [c for c in fe.columns if c not in (TARGET_COL, DATE_COL)]
    X = fe[feat_cols]

    return dt, X, y


# ---------- POLICY THEO NĂM ----------

def backtest_never_retrain_yearly(dt, X, y, horizon) -> List[BacktestResult]:
    years = dt.dt.year
    min_year = int(years.min())
    max_year = int(years.max())
    init_end_year = min_year + INIT_TRAIN_YEARS - 1

    if init_end_year >= max_year:
        raise ValueError("Không đủ năm để backtest yearly.")

    # Train 1 lần trên INIT_TRAIN_YEARS đầu
    train_mask = years <= init_end_year
    X_train, y_train = X[train_mask], y[train_mask]
    model = build_rf_model()
    model.fit(X_train, y_train)

    results: List[BacktestResult] = []
    for test_year in range(init_end_year + 1, max_year + 1):
        test_mask = years == test_year
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
                granularity="year",
                period=str(test_year),
                n_train=int(train_mask.sum()),
                n_test=int(test_mask.sum()),
                mae=float(mae),
                rmse=float(rmse),
                r2=float(r2),
            )
        )
    return results


def backtest_yearly_expanding(dt, X, y, horizon) -> List[BacktestResult]:
    years = dt.dt.year
    min_year = int(years.min())
    max_year = int(years.max())
    init_end_year = min_year + INIT_TRAIN_YEARS - 1

    if init_end_year >= max_year:
        raise ValueError("Không đủ năm để backtest yearly.")

    results: List[BacktestResult] = []
    for test_year in range(init_end_year + 1, max_year + 1):
        train_mask = years <= (test_year - 1)
        test_mask = years == test_year
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test   = X[test_mask], y[test_mask]

        model = build_rf_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2   = r2_score(y_test, y_pred)

        results.append(
            BacktestResult(
                horizon=horizon,
                policy="yearly_expanding",
                granularity="year",
                period=str(test_year),
                n_train=int(train_mask.sum()),
                n_test=int(test_mask.sum()),
                mae=float(mae),
                rmse=float(rmse),
                r2=float(r2),
            )
        )
    return results


# ---------- POLICY THEO THÁNG ----------

def _month_code(dt: pd.Series) -> np.ndarray:
    """Mã hoá YYYYMM thành số nguyên để sort dễ."""
    return dt.dt.year.values * 100 + dt.dt.month.values


def _month_label(code: int) -> str:
    year = code // 100
    mon  = code % 100
    return f"{year}-{mon:02d}"


def backtest_monthly_expanding(dt, X, y, horizon) -> List[BacktestResult]:
    ym = _month_code(dt)
    uniq = np.unique(ym)
    uniq.sort()

    if len(uniq) <= INIT_TRAIN_MONTHS:
        raise ValueError("Không đủ tháng để backtest monthly_expanding.")

    results: List[BacktestResult] = []

    # dùng INIT_TRAIN_MONTHS tháng đầu làm warm-up
    for i in range(INIT_TRAIN_MONTHS, len(uniq)):
        test_code = uniq[i]
        train_codes = uniq[:i]     # tất cả tháng trước đó

        train_mask = np.isin(ym, train_codes)
        test_mask  = (ym == test_code)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test   = X[test_mask], y[test_mask]

        model = build_rf_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2   = r2_score(y_test, y_pred)

        results.append(
            BacktestResult(
                horizon=horizon,
                policy="monthly_expanding",
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


def backtest_monthly_rolling(dt, X, y, horizon,
                             window_months: int = ROLLING_WINDOW_MONS) -> List[BacktestResult]:
    ym = _month_code(dt)
    uniq = np.unique(ym)
    uniq.sort()

    if len(uniq) <= window_months:
        raise ValueError("Không đủ tháng để backtest monthly_rolling.")

    results: List[BacktestResult] = []

    # bắt đầu từ khi có đủ window_months tháng lịch sử
    for i in range(window_months, len(uniq)):
        test_code = uniq[i]
        train_codes = uniq[i - window_months:i]   # window_months tháng ngay trước test

        train_mask = np.isin(ym, train_codes)
        test_mask  = (ym == test_code)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test   = X[test_mask], y[test_mask]

        model = build_rf_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2   = r2_score(y_test, y_pred)

        results.append(
            BacktestResult(
                horizon=horizon,
                policy=f"monthly_rolling_{window_months}m",
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


# ---------- UTILS VẼ BIỂU ĐỒ ----------

def plot_mae(df_res: pd.DataFrame, horizon: int, granularity: str):
    sub = df_res[(df_res["horizon"] == horizon) &
                 (df_res["granularity"] == granularity)].copy()
    if sub.empty:
        print(f"[WARN] Không có kết quả cho H{horizon} / {granularity}")
        return

    # sort theo period (chuỗi "YYYY" hoặc "YYYY-MM")
    sub = sub.sort_values("period")

    plt.figure(figsize=(9, 4))
    for policy, grp in sub.groupby("policy"):
        plt.plot(grp["period"], grp["mae"], marker="o", label=policy)

    plt.title(f"MAE theo {granularity} – Horizon H{horizon}")
    plt.xlabel("Thời gian (năm / tháng theo ngày cơ sở t)")
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

    for h in HORIZONS:
        print(f"\n==== Horizon H{h} ====")
        dt, X, y = prepare_fe_for_horizon(df, h)
        print(f"FE H{h}: X={X.shape}, y={y.shape}, years={dt.dt.year.min()}..{dt.dt.year.max()}")

        # YEARLY policies
        all_results.extend(backtest_never_retrain_yearly(dt, X, y, horizon=h))
        all_results.extend(backtest_yearly_expanding(dt, X, y, horizon=h))

        # MONTHLY policies
        all_results.extend(backtest_monthly_expanding(dt, X, y, horizon=h))
        all_results.extend(backtest_monthly_rolling(dt, X, y, horizon=h))

    df_res = pd.DataFrame([r.__dict__ for r in all_results])
    print("\n=== Backtest summary (head) ===")
    print(df_res.head())

    out_csv = "backtest_multi_policy_results.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"[OK] Saved backtest results -> {out_csv}")

    # Vẽ: ví dụ chỉ cần plot H1
    plot_mae(df_res, horizon=1, granularity="year")
    plot_mae(df_res, horizon=1, granularity="month")

    plt.show()


if __name__ == "__main__":
    main()
