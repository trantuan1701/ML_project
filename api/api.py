# api/api.py
from __future__ import annotations
import os
from datetime import date
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from HCM_temp_forcast.infer import forecast_next_5_days
from HCM_temp_forcast.prepare import basic_clean
from HCM_temp_forcast.data import load_from_parquet

try:
    from scripts.fetch import incremental_fetch_via_xlsx
except Exception:
    incremental_fetch_via_xlsx = None

# ================== CONFIG ==================
DATA_PATH    = "data/weather.parquet"
ARTIFACT_DIR = "artifacts_113"
HORIZONS     = (1, 2, 3, 4, 5)
DATE_COL     = "datetime"
TRUTH_COL    = "temp"
DECIMALS     = 1
# ============================================

# Đường dẫn tuyệt đối tới thư mục api/ và api/static/
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

_df_clean_cache: Optional[pd.DataFrame] = None


def _load_clean_data() -> pd.DataFrame:
    """Load + clean dữ liệu, có cache giống bản Gradio."""
    global _df_clean_cache
    if _df_clean_cache is not None:
        return _df_clean_cache
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Không thấy file {DATA_PATH}. Hãy fetch dữ liệu trước.")
    df_raw = load_from_parquet(DATA_PATH)
    df = basic_clean(df_raw, drop_text_cols=True, drop_na_core=True)
    if DATE_COL in df.columns:
        df = df.sort_values(DATE_COL).reset_index(drop=True)
    _df_clean_cache = df
    return df


def _default_current_date() -> date:
    """Ngày gốc mặc định = max date trong dữ liệu, fallback = hôm nay."""
    df = _load_clean_data()
    last_dt = pd.to_datetime(df[DATE_COL]).max().date() if not df.empty else date.today()
    return last_dt


# ============== Pydantic models ==============
class PredictRequest(BaseModel):
    current_date: str


class DayForecast(BaseModel):
    horizon_days: int
    target_date: str
    pred_temp: float
    gt_temp: Optional[float] = None
    abs_error: Optional[float] = None


class PredictResponse(BaseModel):
    current_date: str
    min_date: str
    max_date: str
    items: List[DayForecast]
    fetched_note: Optional[str] = None


# =============== FastAPI app =================
app = FastAPI(title="HCM Temperature Forecast API")

# Serve frontend static files (HTML/CSS/JS trong api/static)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    """Trang HTML chính: api/static/index.html"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(
            status_code=404,
            detail="Không tìm thấy api/static/index.html. Hãy tạo UI frontend trong thư mục api/static."
        )
    return FileResponse(index_path)


@app.get("/default-date")
def default_date():
    """Lấy ngày gốc mặc định (giống _default_current_date_str trong app Gradio)."""
    return {"current_date": _default_current_date().isoformat()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    /predict:
    - Gọi incremental_fetch_via_xlsx() nếu có
    - Load dữ liệu, validate current_date
    - Gọi forecast_next_5_days
    - Trả JSON cho frontend
    """
    global _df_clean_cache

    # 1) Thử fetch incremental nếu có
    fetched_note: Optional[str] = None
    if incremental_fetch_via_xlsx is not None:
        try:
            incremental_fetch_via_xlsx()
            _df_clean_cache = None  # reset cache để load lại data mới
            # fetched_note = "Đã fetch dữ liệu mới."
        except Exception as e:
            fetched_note = f"Không fetch được dữ liệu mới ({e}). Dùng dữ liệu hiện có."

    # 2) Load dữ liệu + validate
    try:
        df = _load_clean_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if df.empty:
        raise HTTPException(status_code=500, detail="Dữ liệu rỗng sau khi load/clean.")

    dt_min = pd.to_datetime(df[DATE_COL]).min().date()
    dt_max = pd.to_datetime(df[DATE_COL]).max().date()

    try:
        cur = pd.to_datetime(req.current_date).date()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Ngày không hợp lệ. Hãy dùng định dạng YYYY-MM-DD."
        )

    if cur < dt_min:
        raise HTTPException(
            status_code=400,
            detail=f"Ngày gốc {cur} < min date trong dữ liệu ({dt_min}).",
        )
    if cur > dt_max:
        raise HTTPException(
            status_code=400,
            detail=f"Ngày gốc {cur} > max date trong dữ liệu ({dt_max}). Hãy fetch dữ liệu mới trước.",
        )

    # 3) Gọi forecast_next_5_days
    try:
        out = forecast_next_5_days(
            df_recent=df,
            current_date=cur,
            artifact_dir=ARTIFACT_DIR,
            horizons=HORIZONS,
            df_truth=df,
            date_col=DATE_COL,
            truth_col=TRUTH_COL,
        )
    except KeyError as ke:
        raise HTTPException(status_code=500, detail=f"Thiếu pipeline/feature: {ke}")
    except ValueError as ve:
        # Đây chính là chỗ log ra "Lỗi giá trị: (H1) Thiếu features..."
        raise HTTPException(status_code=500, detail=f"Lỗi giá trị: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi bất ngờ: {e}")

    cols = ["horizon_days", "target_date", "pred_temp"]
    if "gt_temp" in out.columns:
        cols += ["gt_temp", "abs_error"]

    items: List[DayForecast] = []
    for _, r in out[cols].iterrows():
        target_date = pd.to_datetime(r["target_date"]).date()
        pred_temp = round(float(r["pred_temp"]), DECIMALS)

        gt_temp_val = None
        abs_error_val = None
        if "gt_temp" in r and pd.notna(r["gt_temp"]):
            gt_temp_val = round(float(r["gt_temp"]), DECIMALS)
        if "abs_error" in r and pd.notna(r["abs_error"]):
            abs_error_val = round(float(r["abs_error"]), DECIMALS)

        items.append(
            DayForecast(
                horizon_days=int(r["horizon_days"]),
                target_date=target_date.isoformat(),
                pred_temp=pred_temp,
                gt_temp=gt_temp_val,
                abs_error=abs_error_val,
            )
        )

    return PredictResponse(
        current_date=cur.isoformat(),
        min_date=dt_min.isoformat(),
        max_date=dt_max.isoformat(),
        items=items,
        fetched_note=fetched_note,
    )

# uvicorn api.api:app --reload --port 8000
