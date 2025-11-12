from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Iterable, Optional, Tuple

from .prepare import feature_engineer_timeseries
from HCM_temp_forcast.data import load_from_parquet  # <-- dùng loader parquet

# ---------- nạp artifacts đã lưu ----------
def load_pipelines_from_dir(artifact_dir: str) -> Dict[int, dict]:
    """
    Tìm các file như: artifacts/pipeline_H{h}_{model}.joblib
    Trả về: {h: {"pipeline": sklearn.Pipeline, "feature_cols": [...], "model_name": str}}
    """
    out: Dict[int, dict] = {}
    for path in glob.glob(os.path.join(artifact_dir, "pipeline_H*_*.joblib")):
        obj = joblib.load(path)  # {"pipeline": pipe, "feature_cols": [...]}
        fname = os.path.basename(path)
        # parse H
        # ví dụ: pipeline_H3_xgb.joblib -> h=3
        try:
            h_str = fname.split("_")[1]  # "H3"
            h = int(h_str[1:])
        except Exception:
            continue
        model_name = os.path.splitext(fname)[0].split("_", 2)[-1]  # phần sau H{h}_
        out[h] = {
            "pipeline": obj["pipeline"],
            "feature_cols": obj.get("feature_cols"),
            "model_name": model_name
        }
    if not out:
        raise FileNotFoundError(f"Không tìm thấy pipeline trong '{artifact_dir}'.")
    return out

# ---------- suy luận cho 1 ngày t ----------
def forecast_next_5_days(
    df_recent: pd.DataFrame,
    *,
    current_date,                      # ngày t (str | datetime)
    artifact_dir: str = "artifacts",   # nơi bạn đã lưu pipeline_H{h}_*.joblib
    horizons: Iterable[int] = (1,2,3,4,5),
    lags: Iterable[int] = (1,2,3,7,14),
    roll_windows: Iterable[int] = (7,14),
    # ---- ground truth options ----
    df_truth: Optional[pd.DataFrame] = None,
    date_col: str = "datetime",
    truth_col: str = "temp",
) -> pd.DataFrame:
    """
    Trả về DataFrame: [horizon_days, target_date, pred_temp, (opt) gt_temp, abs_error]
    - FE lại từ df_recent (đến đúng 'current_date')
    - Nạp pipeline đã lưu cho từng horizon và predict.
    """
    current_date = pd.to_datetime(current_date)
    hist = df_recent[df_recent[date_col] <= current_date].copy().sort_values(date_col)

    # đủ dữ liệu cho lag/rolling?
    drop_n = max(max(lags) if lags else 0, max(roll_windows) if roll_windows else 0)
    if len(hist) < drop_n + 1:
        raise ValueError(f"Không đủ lịch sử ({len(hist)} ngày). Cần >= {drop_n+1} ngày.")

    # nạp pipelines đã lưu
    bank = load_pipelines_from_dir(artifact_dir)

    preds = []
    for h in horizons:
        if h not in bank:
            raise KeyError(f"Thiếu artifact cho horizon H{h} trong '{artifact_dir}'.")

        pipe = bank[h]["pipeline"]
        feat_cols = bank[h]["feature_cols"]

        # FE như lúc train (make_target=False)
        fe = feature_engineer_timeseries(
            hist, horizon=h, lags=lags, roll_windows=roll_windows, make_target=False
        )

        if feat_cols is None:
            # fallback: dùng toàn bộ cột numeric nếu không có feature_cols (ít gặp)
            feat_cols = fe.select_dtypes(include=[np.number]).columns.tolist()

        missing = [c for c in feat_cols if c not in fe.columns]
        if missing:
            raise ValueError(f"(H{h}) Thiếu features ở inference: {missing}")

        X = fe[feat_cols]
        # pipeline đã gồm scaler (DynamicScaler) + model → gọi thẳng predict
        y_hat = float(pipe.predict(X.tail(1))[0])
        target_date = current_date + pd.Timedelta(days=h)

        gt_val = np.nan
        if df_truth is not None and {date_col, truth_col} <= set(df_truth.columns):
            m = df_truth.loc[df_truth[date_col] == target_date, truth_col]
            if not m.empty:
                gt_val = float(m.values[0])

        preds.append({
            "horizon_days": h,
            "target_date": target_date,
            "pred_temp": y_hat,
            "gt_temp": gt_val,
        })

    out = pd.DataFrame(preds).sort_values("horizon_days").reset_index(drop=True)
    if "gt_temp" in out.columns:
        out["abs_error"] = (out["pred_temp"] - out["gt_temp"]).abs()
    return out

if __name__ == "__main__":
    from .prepare import basic_clean

    # ---- Config nhanh ----
    DATA_PATH    = "data/weather.parquet"  # <-- đọc từ parquet
    ARTIFACT_DIR = "artifacts"
    CURRENT_DATE = "2025-01-20"
    HORIZONS     = (1, 2, 3, 4, 5)
    LAGS         = (1, 2, 3, 7, 14)
    ROLLS        = (7, 14)
    DATE_COL     = "datetime"
    TRUTH_COL    = "temp"

    # ---- Load parquet & chuẩn hoá ----
    print("==> Load recent data (parquet)")
    df_recent = load_from_parquet(DATA_PATH)
    print("Raw shape:", df_recent.shape)

    print("==> basic_clean")
    df_recent = basic_clean(df_recent, drop_text_cols=True)
    print("Cleaned shape:", df_recent.shape)

    # Ngày hiện tại để dự báo
    print("==> Forecast next 5 days")
    out = forecast_next_5_days(
        df_recent,
        current_date=CURRENT_DATE,
        artifact_dir=ARTIFACT_DIR,
        horizons=HORIZONS,
        lags=LAGS,
        roll_windows=ROLLS,
        df_truth=df_recent,         
        date_col=DATE_COL,
        truth_col=TRUTH_COL,
    )

    cols_print = ["horizon_days", "target_date", "pred_temp"]
    if "gt_temp" in out.columns:
        cols_print += ["gt_temp", "abs_error"]
    print("\n=== RESULT ===")
    print(out[cols_print].round(3).to_string(index=False))