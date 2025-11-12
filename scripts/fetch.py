# scripts/fetch.py
from __future__ import annotations
import os
import sys
import io
import json
import urllib.parse
import urllib.request
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd

# ===================== CONFIG =====================
DATA_PATH    = Path("data/weather.parquet")
EXPORTS_DIR  = Path("exports")                # nơi lưu bản XLSX tải về (để audit)
LOCATION     = "10.82, 106.67"                # có thể dùng 'Hochiminh' hoặc 'lat,lon'
UNIT_GROUP   = "metric"                       # 'metric' | 'us'
INCLUDE      = "days"                         # 'days' | 'hours'
CONTENT_TYPE = "xlsx"                         # tải về XLSX
BOOTSTRAP_YEARS = 10                          # nếu chưa có parquet -> lấy 10 năm
FORCE_OVERWRITE_DAYS = 0                      # >0: backfill N ngày để ghi đè (khi muốn cập nhật/cải chính)
LOCK_SCHEMA_TO_OLD = True                     # True: ép schema new theo file cũ (tên + thứ tự cột)
BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
# ==================================================


# ----------------- Helpers -----------------

def _load_api_key() -> str:
    """Đọc VC_API_KEY (tự load .env nếu có)."""
    try:
        from dotenv import load_dotenv  # type: ignore
        for p in [Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"]:
            if p.exists():
                load_dotenv(p)
    except Exception:
        pass
    key = os.getenv("VC_API_KEY") or os.getenv("VISUAL_CROSSING_KEY")
    if not key:
        raise RuntimeError("Missing API key. Đặt VC_API_KEY trong .env hoặc export vào môi trường.")
    return key


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    return d


def _ensure_datetime_cols(df: pd.DataFrame, cols=("datetime", "sunrise", "sunset")) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")  # ép về datetime64[ns]
    return d


def _coalesce_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Đồng nhất áp suất: ưu tiên 'sealevelpressure'; nếu thiếu, dùng 'pressure'."""
    d = df.copy()
    has_sea = "sealevelpressure" in d.columns
    has_p   = "pressure" in d.columns
    if (not has_sea) and has_p:
        d["sealevelpressure"] = d["pressure"]
    elif has_sea and has_p:
        mask = d["sealevelpressure"].isna() & d["pressure"].notna()
        if mask.any():
            d.loc[mask, "sealevelpressure"] = d.loc[mask, "pressure"]
    return d


def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Parquet không nhận list/dict trong object; ép về string."""
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        s = out[c]
        if s.apply(lambda x: isinstance(x, (list, dict))).any():
            out[c] = s.map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
    return out


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = _normalize_cols(df)
    df = _ensure_datetime_cols(df, cols=("datetime", "sunrise", "sunset"))
    return df


def _save_parquet(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    out = _ensure_datetime_cols(out, cols=("datetime", "sunrise", "sunset"))
    out = _sanitize_for_parquet(out)
    out.to_parquet(path, index=False)
    return out


def _merge_keep_new(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old.empty:
        return new.copy()
    cols = sorted(set(old.columns) | set(new.columns))
    old2 = old.reindex(columns=cols)
    new2 = new.reindex(columns=cols)
    both = pd.concat([old2, new2], axis=0, ignore_index=True)
    both = both.drop_duplicates(subset=["datetime"], keep="last")
    return both.sort_values("datetime").reset_index(drop=True)


def _align_to_schema(new_df: pd.DataFrame, schema_cols: List[str]) -> pd.DataFrame:
    """Ép new_df có đúng cột & thứ tự như schema cũ."""
    d = new_df.copy()
    # Áp suất trước
    d = _coalesce_pressure(d)
    # Thêm cột thiếu
    for c in schema_cols:
        if c not in d.columns:
            d[c] = pd.NA
    # Chỉ giữ và reorder theo schema cũ
    d = d.reindex(columns=schema_cols)
    return d


def _bootstrap_span(days: int) -> Tuple[date, date]:
    today = date.today()
    return today - timedelta(days=days), today


def _fmt(d: date | datetime | str | None) -> str:
    if d is None:
        return "None"
    if isinstance(d, str):
        return d
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    return d.isoformat()


def _year_chunks(a: date, b: date):
    cur = a
    while cur <= b:
        year_end = date(cur.year, 12, 31)
        seg_end = min(year_end, b)
        yield cur, seg_end
        cur = seg_end + timedelta(days=1)


# ----------------- Fetch (XLSX) -----------------

def build_url(location: str, start_date: str, end_date: str, api_key: str,
              unit_group: str, content_type: str, include: str) -> str:
    # Clean "lat, lon" -> "lat,lon"
    location_clean = ",".join([p.strip() for p in location.split(",")]) if "," in location else location
    query_parameters = urllib.parse.urlencode({
        "key": api_key,
        "unitGroup": unit_group,
        "contentType": content_type,
        "include": include,
    })
    return f"{BASE_URL}{urllib.parse.quote_plus(location_clean)}/{start_date}/{end_date}?{query_parameters}"


def fetch_xlsx_to_df(url: str, save_copy: Optional[Path] = None) -> pd.DataFrame:
    """Tải XLSX bằng urllib; trả về DataFrame. Có thể lưu bản sao để audit."""
    resp = urllib.request.urlopen(url)
    data = resp.read()

    if save_copy is not None:
        save_copy.parent.mkdir(parents=True, exist_ok=True)
        with open(save_copy, "wb") as f:
            f.write(data)

    bio = io.BytesIO(data)
    df = pd.read_excel(bio)
    return df


# ----------------- Main -----------------

def incremental_fetch_via_xlsx():
    api_key = _load_api_key()

    # 1) Load parquet hiện có
    df_old = _load_existing(DATA_PATH)

    # 2) Xác định khoảng thời gian cần lấy
    if df_old.empty:
        start_d, end_d = _bootstrap_span(days=BOOTSTRAP_YEARS * 365)
        mode = f"[Bootstrap] last {BOOTSTRAP_YEARS} years"
        schema_cols = None
    else:
        last_dt = pd.to_datetime(df_old["datetime"].max()).date()
        if FORCE_OVERWRITE_DAYS and FORCE_OVERWRITE_DAYS > 0:
            start_d = max(date(1970, 1, 1), last_dt - timedelta(days=FORCE_OVERWRITE_DAYS - 1))
            mode = f"[Overwrite] backfill {FORCE_OVERWRITE_DAYS} days (from {start_d})"
        else:
            start_d = last_dt + timedelta(days=1)
            mode = "[Incremental] last_date+1 → today"
        end_d = date.today()
        schema_cols = list(df_old.columns) if LOCK_SCHEMA_TO_OLD else None

    print("== Visual Crossing XLSX incremental fetch ==")
    print(f"DATA_PATH  : {DATA_PATH}")
    print(f"LOCATION   : {LOCATION}")
    print(f"UNIT_GROUP : {UNIT_GROUP}")
    print(mode)
    print(f"  - {_fmt(start_d)} → {_fmt(end_d)}")

    if start_d > end_d:
        print("Up-to-date. Không có gì để fetch.")
        return

    # 3) Theo năm để an toàn payload
    new_frames: List[pd.DataFrame] = []
    for s, e in _year_chunks(start_d, end_d):
        url = build_url(
            LOCATION, s.isoformat(), e.isoformat(),
            api_key=api_key, unit_group=UNIT_GROUP,
            content_type=CONTENT_TYPE, include=INCLUDE
        )
        print(f"  -> {s} .. {e}")
        try:
            save_copy = EXPORTS_DIR / f"vc_{s}_{e}.xlsx"
            df_chunk = fetch_xlsx_to_df(url, save_copy=save_copy)
        except urllib.error.HTTPError as he:
            print(f"    HTTP error: {he.code}")
            try:
                print(he.read().decode())
            except Exception:
                pass
            continue
        except urllib.error.URLError as ue:
            print(f"    URL error: {ue.reason}")
            continue
        except Exception as ex:
            print(f"    Unexpected error: {ex}")
            continue

        # 4) Chuẩn hóa chunk
        df_chunk = _normalize_cols(df_chunk)
        df_chunk = _ensure_datetime_cols(df_chunk, cols=("datetime", "sunrise", "sunset"))
        df_chunk = _coalesce_pressure(df_chunk)

        if schema_cols is not None:
            df_chunk = _align_to_schema(df_chunk, schema_cols)

        new_frames.append(df_chunk)
        print(f"     +{len(df_chunk)} rows")

    if not new_frames:
        print("No new rows fetched.")
        return

    # 5) Hợp nhất vào data cũ (giữ bản mới)
    df_new = pd.concat(new_frames, axis=0, ignore_index=True)
    df_new = _ensure_datetime_cols(df_new, cols=("datetime", "sunrise", "sunset"))
    merged = _merge_keep_new(df_old, df_new)

    # Nếu khóa schema, đảm bảo lần cuối đúng thứ tự cột cũ
    if (not df_old.empty) and LOCK_SCHEMA_TO_OLD:
        merged = merged.reindex(columns=df_old.columns)

    # 6) Lưu parquet
    merged = _save_parquet(merged, DATA_PATH)
    print(f"Done. Total rows={len(merged)}")
    if "datetime" in merged.columns:
        print(f"Date range: {merged['datetime'].min()} → {merged['datetime'].max()}")


if __name__ == "__main__":
    try:
        incremental_fetch_via_xlsx()
    except Exception as e:
        print(str(e))
        sys.exit(1)
