# htf/tune.py
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd
import optuna

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# Dùng DynamicScaler & danh sách model từ htf.model
from .model import DynamicScaler, build_model_candidates


# =========================
# 1) Walk-forward CV (MAE)
# =========================
def _cv_mae_for_estimator(
    X: pd.DataFrame,
    y: pd.Series,
    estimator,
    n_splits: int = 3,
    gap: int = 0,
) -> float:
    """
    TimeSeriesSplit CV; pipeline: DynamicScaler -> model.
    KHÔNG impute: nếu X có NaN, các estimator không hỗ trợ NaN sẽ lỗi.
    """
    tss = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    maes = []
    for tr_idx, va_idx in tss.split(X):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        pipe = Pipeline([
            ("scale", DynamicScaler()),
            ("model", clone(estimator)),
        ])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)
        maes.append(mean_absolute_error(y_va, pred))
    return float(np.mean(maes)) if maes else np.inf


# =========================================
# 2) Build estimator theo trial (search space)
# =========================================
def _build_estimator_from_trial(model_name: str, trial: optuna.Trial, random_state: int = 42):
    base = build_model_candidates(random_state=random_state).get(model_name)
    if base is None:
        raise ValueError(f"Model '{model_name}' chưa có trong build_model_candidates().")

    est = clone(base)
    params = {}

    if model_name == "linreg":
        pass

    elif model_name == "ridge":
        params["alpha"] = trial.suggest_float("alpha", 1e-3, 10.0, log=True)

    elif model_name == "lasso":
        params["alpha"] = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        params["max_iter"] = trial.suggest_int("max_iter", 5000, 30000, step=5000)

    elif model_name == "rf":
        params["n_estimators"]      = trial.suggest_int("n_estimators", 200, 1200, step=100)
        params["max_depth"]         = trial.suggest_int("max_depth", 3, 20)
        params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
        params["min_samples_leaf"]  = trial.suggest_int("min_samples_leaf", 1, 10)

    elif model_name == "gbr":
        params["n_estimators"]      = trial.suggest_int("n_estimators", 200, 1500, step=100)
        params["learning_rate"]     = trial.suggest_float("learning_rate", 0.005, 0.2, log=True)
        params["max_depth"]         = trial.suggest_int("max_depth", 2, 8)
        params["subsample"]         = trial.suggest_float("subsample", 0.5, 1.0)
        params["min_samples_leaf"]  = trial.suggest_int("min_samples_leaf", 1, 20)

    elif model_name == "xgb":
        params["n_estimators"]      = trial.suggest_int("n_estimators", 300, 2000, step=100)
        params["learning_rate"]     = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        params["max_depth"]         = trial.suggest_int("max_depth", 3, 12)
        params["subsample"]         = trial.suggest_float("subsample", 0.5, 1.0)
        params["colsample_bytree"]  = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        params["reg_alpha"]         = trial.suggest_float("reg_alpha", 1e-8, 1e-1, log=True)
        params["reg_lambda"]        = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)

    elif model_name == "lgbm":
        params["n_estimators"]      = trial.suggest_int("n_estimators", 500, 3000, step=100)
        params["learning_rate"]     = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True)
        params["num_leaves"]        = trial.suggest_int("num_leaves", 15, 255)
        params["max_depth"]         = trial.suggest_int("max_depth", -1, 16)
        params["subsample"]         = trial.suggest_float("subsample", 0.5, 1.0)
        params["colsample_bytree"]  = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        params["reg_alpha"]         = trial.suggest_float("reg_alpha", 1e-8, 1e-1, log=True)
        params["reg_lambda"]        = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)

    # set params an toàn
    valid = est.get_params()
    for k, v in params.items():
        if k in valid:
            est.set_params(**{k: v})
    return est


# =======================================================
# 3) Objective Optuna: MAE (walk-forward CV trên train)
# =======================================================
def _make_objective_for_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    random_state: int = 42,
    n_splits: int = 3,
    gap: int = 0,
):
    def objective(trial: optuna.Trial) -> float:
        est = _build_estimator_from_trial(model_name, trial, random_state=random_state)
        mae = _cv_mae_for_estimator(X_train, y_train, est, n_splits=n_splits, gap=gap)
        trial.set_user_attr("cv_mae", mae)
        return mae
    return objective


# ===================================================
# 4) Tune CHO 1 horizon (chạy TẤT CẢ model)
# ===================================================
def tune_all_models_for_horizon(
    datasets_for_h: dict,
    *,
    random_state: int = 42,
    n_trials: int = 50,
    n_splits: int = 3,
    gap: int = 0,
    storage: Optional[str] = None,          # ví dụ "sqlite:///optuna.db"
    study_prefix: str = "tune",
) -> Dict[str, Dict]:
    """
    Input (cho 1 horizon):
      datasets_for_h = {'train': (X_train, y_train), 'test': (X_test, y_test), 'feature_cols': [...]}
    Output:
      {
        model_name: {'best_value': cv_mae, 'best_params': {...}, 'study_name': ...},
        ...
        '_best_overall': {'model': name, 'cv_mae': value}
      }
    """
    X_train, y_train = datasets_for_h["train"]
    results: Dict[str, Dict] = {}

    for model_name in build_model_candidates(random_state).keys():
        # Sampler RIÊNG cho mỗi study; group=False để tránh lỗi multi-studies
        sampler = optuna.samplers.TPESampler(
            consider_prior=True,
            multivariate=True,
            group=False,
            seed=random_state,
        )
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10)

        study_name = f"{study_prefix}-{model_name}"
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )

        obj = _make_objective_for_model(
            X_train, y_train,
            model_name=model_name,
            random_state=random_state,
            n_splits=n_splits,
            gap=gap,
        )
        study.optimize(obj, n_trials=n_trials)

        results[model_name] = {
            "best_value": float(study.best_value),
            "best_params": dict(study.best_params),
            "study_name": study_name,
        }

    best_model = min(results.items(), key=lambda kv: kv[1]["best_value"])[0]
    results["_best_overall"] = {"model": best_model, "cv_mae": results[best_model]["best_value"]}
    return results


# ===================================================
# 5) Tune CHO NHIỀU horizon
# ===================================================
def tune_all_horizons(
    datasets: Dict[int, dict],
    *,
    random_state: int = 42,
    n_trials: int = 50,
    n_splits: int = 3,
    gap: int = 0,
    storage: Optional[str] = None,
    study_prefix: str = "tune",
) -> Dict[int, Dict]:
    """
    datasets: {h: {'train': (X,y), 'test': (X,y), 'feature_cols': [...]}}
    """
    out: Dict[int, Dict] = {}
    for h, parts in datasets.items():
        res = tune_all_models_for_horizon(
            parts,
            random_state=random_state,
            n_trials=n_trials,
            n_splits=n_splits,
            gap=gap,
            storage=storage,
            study_prefix=f"{study_prefix}-H{h}",
        )
        out[h] = res
    return out
import os
import time
import json
from datetime import date, datetime, timedelta
from typing import Optional, List

import pandas as pd
import requests

# ================== CONFIG ==================
DATA_PATH = "data/weather.parquet"
LOCATION = "Hanoi"
UNIT_GROUP = "metric"          # metric / us
INCLUDE = "days"               # 'days' or 'hours'
YEARS_BACK_IF_BOOTSTRAP = 10   # nếu chưa có file parquet, fetch 10 năm gần nhất
MAX_RETRIES = 3
TIMEOUT_SEC = 60

# Chọn cột cần lấy để tiết kiệm băng thông/quota
ELEMENTS_DAILY = (
    "datetime,temp,humidity,dew,cloudcover,windspeed,"
    "sealevelpressure,precip,preciptype,precipprob,"
    "solarradiation,solarenergy,uvindex,sunrise,sunset,"
    "conditions,icon,description,stations"
)

BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
API_KEY = os.getenv("VC_API_KEY")  # <-- đặt VC_API_KEY trong env
# ============================================


def _request_interval(location: str,
                      start: str,
                      end: str,
                      include: str = "days",
                      elements: Optional[str] = None,
                      unit_group: str = "metric",
                      api_key: Optional[str] = None,
                      content: str = "json",
                      retries: int = MAX_RETRIES,
                      timeout: int = TIMEOUT_SEC) -> pd.DataFrame:
    """Gọi 1 khoảng thời gian [start, end] và trả về DataFrame đã chuẩn hoá."""
    if not api_key:
        raise RuntimeError("Missing API key. Set VC_API_KEY environment variable.")

    params = {
        "unitGroup": unit_group,
        "include": include,
        "key": api_key,
        "contentType": content,
        "timezone": "auto",
    }
    if elements:
        params["elements"] = elements

    url = f"{BASE_URL}/{location}/{start}/{end}"

    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                if content == "json":
                    js = r.json()
                    key = "days" if include == "days" else "hours"
                    df = pd.json_normalize(js.get(key, []))
                elif content == "csv":
                    from io import StringIO
                    df = pd.read_csv(StringIO(r.text))
                else:
                    import io
                    df = pd.read_excel(io.BytesIO(r.content))

                # chuẩn hoá cột
                df.columns = [str(c).strip().lower() for c in df.columns]
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                return df
            elif r.status_code in (429, 500, 502, 503):
                # backoff
                sleep_s = 2 ** i
                print(f"[Retryable {r.status_code}] {r.text[:120]}... -> sleep {sleep_s}s & retry")
                time.sleep(sleep_s)
                continue
            else:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_err = e
            sleep_s = 2 ** i
            print(f"[Exception] {e} -> sleep {sleep_s}s & retry")
            time.sleep(sleep_s)

    raise RuntimeError(f"Max retries exceeded. Last error: {last_err}")


def _year_chunks(start_date: date, end_date: date) -> List[tuple]:
    """Chia [start_date, end_date] thành các đoạn theo năm (YYYY-01-01 .. YYYY-12-31)."""
    chunks = []
    cur = date(start_date.year, 1, 1)
    # bắt đầu tại đầu năm chứa start_date
    if start_date > cur:
        cur = start_date

    while cur <= end_date:
        end_of_year = date(cur.year, 12, 31)
        seg_end = min(end_of_year, end_date)
        chunks.append((cur, seg_end))
        cur = seg_end + timedelta(days=1)
    return chunks


def _load_existing_parquet(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    # normalize
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "datetime" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def _bootstrap_start_date(today: date, years_back: int) -> date:
    try:
        return date(today.year - years_back, today.month, today.day)
    except Exception:
        # fallback nếu ngày không hợp lệ (29/2,…): quay về 1/1
        return date(today.year - years_back, 1, 1)


def incremental_fetch(location: str,
                      parquet_path: str,
                      include: str = "days",
                      elements: Optional[str] = None,
                      unit_group: str = "metric",
                      api_key: Optional[str] = None):
    """Main incremental flow."""
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

    existing = _load_existing_parquet(parquet_path)
    today = date.today()

    if existing is None or existing.empty or "datetime" not in existing.columns:
        # bootstrap
        start = _bootstrap_start_date(today, YEARS_BACK_IF_BOOTSTRAP)
        print(f"[Bootstrap] No parquet found → fetching from {start} to {today}")
    else:
        last_dt = pd.to_datetime(existing["datetime"], errors="coerce").max()
        if pd.isna(last_dt):
            start = _bootstrap_start_date(today, YEARS_BACK_IF_BOOTSTRAP)
            print(f"[Bootstrap] Invalid last datetime → fetching from {start} to {today}")
        else:
            start = (last_dt.date() + timedelta(days=1))
            if start > today:
                print("[Info] Already up to date. Nothing to fetch.")
                return
            print(f"[Incremental] Fetching from {start} to {today}")

    # fetch theo từng năm
    chunks = _year_chunks(start, today)
    fetched = []
    for s, e in chunks:
        print(f"  - {s} → {e}")
        df = _request_interval(
            location=location,
            start=s.isoformat(),
            end=e.isoformat(),
            include=include,
            elements=elements,
            unit_group=unit_group,
            api_key=api_key,
            content="json",
        )
        if not df.empty:
            fetched.append(df)

    if not fetched:
        print("[Info] No new data returned.")
        return

    new_df = pd.concat(fetched, ignore_index=True)
    # Chuẩn hoá datetime
    if "datetime" in new_df.columns:
        new_df["datetime"] = pd.to_datetime(new_df["datetime"], errors="coerce")

    # Gộp với existing
    if existing is not None and not existing.empty:
        all_df = pd.concat([existing, new_df], ignore_index=True)
    else:
        all_df = new_df

    # Khử trùng lặp & sort
    if "datetime" not in all_df.columns:
        raise ValueError("API response missing 'datetime' column.")
    before = len(all_df)
    all_df = (
        all_df
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    after = len(all_df)
    print(f"[Save] Rows before dedup={before}, after={after}")

    # Lưu parquet
    all_df.to_parquet(parquet_path, index=False)
    print(f"[OK] Saved → {parquet_path}. Range: {all_df['datetime'].min().date()} .. {all_df['datetime'].max().date()}")


def main():
    print("== Visual Crossing incremental fetch ==")
    print(f"DATA_PATH  : {DATA_PATH}")
    print(f"LOCATION   : {LOCATION}")
    print(f"INCLUDE    : {INCLUDE}")
    print(f"UNIT_GROUP : {UNIT_GROUP}")

    elements = ELEMENTS_DAILY if INCLUDE == "days" else None  # tự chọn elements theo include
    incremental_fetch(
        location=LOCATION,
        parquet_path=DATA_PATH,
        include=INCLUDE,
        elements=elements,
        unit_group=UNIT_GROUP,
        api_key=API_KEY,
    )


if __name__ == "__main__":
    main()