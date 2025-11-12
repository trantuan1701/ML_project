# htf/prepare.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, List, Optional, Tuple, Dict
from HCM_temp_forcast.data import load_from_parquet

# ====== cấu hình mặc định ======
TIME_COLS_DEFAULT = ("datetime", "sunrise", "sunset")
TEXT_COLS_DEFAULT = ["conditions", "icon", "description"]
DROP_ALWAYS = ["stations", "severerisk"]

# Các cột số cốt lõi dùng nhiều trong FE (để drop NaN nếu muốn “sạch tuyệt đối”)
CORE_NUMERIC_COLS = [
    "temp", "humidity", "dew", "cloudcover", "windspeed",
    "sealevelpressure", "precip", "precipcover",
    "solarradiation", "solarenergy", "uvindex"
]

# ====== utils cơ bản ======
def _coerce_time_cols(df: pd.DataFrame, time_cols: Iterable[str] = TIME_COLS_DEFAULT) -> pd.DataFrame:
    df = df.copy()
    for c in time_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def _normalize_preciptype(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s.startswith("[") and s.endswith("]"):
        s = s.strip("[]").replace("'", "").replace('"', "")
    return s

def _add_has_rain(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "preciptype" in df.columns:
        df["preciptype_norm"] = df["preciptype"].map(_normalize_preciptype)
        has_rain = df["preciptype_norm"].fillna("").str.contains(r"\brain\b")
    else:
        has_rain = pd.Series(False, index=df.index)

    if "precipprob" in df.columns:
        prob_ok = (pd.to_numeric(df["precipprob"], errors="coerce").fillna(0) == 100)
    else:
        prob_ok = pd.Series(True, index=df.index)

    df["has_rain"] = (has_rain & prob_ok).astype(int)
    if "preciptype_norm" in df.columns:
        df.drop(columns=["preciptype_norm"], inplace=True)
    return df

def _drop_columns_if_exist(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    existing = [c for c in cols if c in df.columns]
    if existing:
        df.drop(columns=existing, inplace=True)
    return df

def _drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        df.drop(columns=constant_cols, inplace=True)
    return df

# ====== clean ======
def basic_clean(
    data: pd.DataFrame,
    *,
    drop_text_cols: bool = True,
    extra_time_cols: List[str] | None = None,
    drop_na_core: bool = True,  # <- thêm: đảm bảo numeric core không còn NaN
) -> pd.DataFrame:
    """
    - Ép kiểu cột thời gian
    - Tạo has_rain, sau đó bỏ preciptype/precipprob
    - Bỏ cột rác cố định (stations, severerisk) và (tuỳ chọn) text cols
    - Bỏ các cột chỉ có 1 giá trị
    - (tuỳ chọn) Drop các hàng thiếu ở CORE_NUMERIC_COLS để đảm bảo sạch NaN trước FE
    """
    time_cols = list(TIME_COLS_DEFAULT) + (extra_time_cols or [])
    df = _coerce_time_cols(data, time_cols=time_cols)
    df = _add_has_rain(df)
    df = _drop_columns_if_exist(df, ["preciptype", "precipprob"])
    df = _drop_columns_if_exist(df, DROP_ALWAYS)

    if drop_text_cols:
        df = _drop_columns_if_exist(df, TEXT_COLS_DEFAULT)

    df = _drop_constant_columns(df)

    if drop_na_core:
        keep_cols = [c for c in CORE_NUMERIC_COLS if c in df.columns]
        # Không drop theo sunrise/sunset ở đây để không mất nhiều dữ liệu;
        # FE sẽ tự xử lý daylight nếu đủ hai cột.
        if keep_cols:
            df = df.dropna(subset=keep_cols)

    # sắp xếp lại theo thời gian để an toàn
    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)
    return df

# ====== split ======
def time_split_train_test(
    df: pd.DataFrame,
    *,
    datetime_col: str = "datetime",
    train_frac: float = 0.8,
    gap: int = 0,
    test_start_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Chia theo thời gian: [TRAIN] --gap--> [TEST]
    - Nếu truyền test_start_date, ưu tiên dùng mốc đó với searchsorted (ổn định hơn idxmax).
    """
    data = df.sort_values(datetime_col).reset_index(drop=True)
    n = len(data)
    if n == 0:
        return data.copy(), data.copy(), gap

    if test_start_date is not None:
        ts = pd.to_datetime(test_start_date)
        # vị trí đầu tiên có datetime >= ts
        arr = pd.to_datetime(data[datetime_col]).values
        i_test_start = int(np.searchsorted(arr, np.datetime64(ts), side="left"))
        i_test_start = max(0, min(i_test_start, n))
        i_train_end = i_test_start - gap - 1
    else:
        assert 0 < train_frac < 1, "train_frac phải nằm trong (0,1)"
        i_train_end_raw = int(n * train_frac) - 1
        i_test_start = i_train_end_raw + 1 + gap
        i_train_end  = i_train_end_raw

    # chốt chỉ số an toàn
    i_test_start = max(0, min(i_test_start, n))
    i_train_end  = max(-1, min(i_train_end, n - 1))

    train = data.iloc[:i_train_end + 1].copy()
    test  = data.iloc[i_test_start:].copy()
    return train, test, gap

# ====== FE ======
def feature_engineer_timeseries(
    df: pd.DataFrame,
    *,
    lags: Iterable[int] = (1, 2, 3, 7, 14),
    roll_windows: Iterable[int] = (7, 14),
    horizon: int = 1,
    drop_original_astro: bool = True,
    drop_datetime_cols: bool = True,
    make_target: bool = True
) -> pd.DataFrame:
    d = df.copy()

    # 0) Chuẩn dtype thời gian
    for c in ["datetime", "sunrise", "sunset"]:
        if c in d.columns and not pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = pd.to_datetime(d[c], errors="coerce")

    # Bảo đảm loại moonphase / winddir nếu còn
    for c in ["moonphase", "winddir"]:
        if c in d.columns:
            d.drop(columns=[c], inplace=True)

    # 1) Calendar / cyclical
    if "datetime" not in d.columns:
        raise ValueError("Thiếu cột 'datetime' trước FE.")
    d["year"]      = d["datetime"].dt.year
    d["month"]     = d["datetime"].dt.month
    d["day"]       = d["datetime"].dt.day
    d["dayofweek"] = d["datetime"].dt.dayofweek
    d["dayofyear"] = d["datetime"].dt.dayofyear
    d["sin_doy"]   = np.sin(2 * np.pi * d["dayofyear"] / 365.25)
    d["cos_doy"]   = np.cos(2 * np.pi * d["dayofyear"] / 365.25)

    # 2) Astronomical / daylight
    if {"sunrise", "sunset"}.issubset(d.columns):
        d["daylength_min"] = (d["sunset"] - d["sunrise"]).dt.total_seconds() / 60.0
        d["sunrise_min"]   = d["sunrise"].dt.hour * 60 + d["sunrise"].dt.minute
        d["sunset_min"]    = d["sunset"].dt.hour  * 60 + d["sunset"].dt.minute

    # 3) Lags & Rolling (past-only)
    lag_vars = [
        "temp", "humidity", "dew", "cloudcover", "windspeed",
        "sealevelpressure", "precip", "precipcover",
        "solarradiation", "solarenergy", "uvindex", "has_rain"
    ]
    lag_vars = [c for c in lag_vars if c in d.columns]

    for k in lags:
        for col in lag_vars:
            d[f"{col}_lag{k}"] = d[col].shift(k)

    for w in roll_windows:
        if "temp" in d.columns:
            d[f"temp_roll{w}_mean"] = d["temp"].shift(1).rolling(w, min_periods=w).mean()
            d[f"temp_roll{w}_std"]  = d["temp"].shift(1).rolling(w, min_periods=w).std()
        if "has_rain" in d.columns:
            d[f"has_rain_roll{w}_sum"] = d["has_rain"].shift(1).rolling(w, min_periods=w).sum()
        for v in ["humidity", "cloudcover", "windspeed"]:
            if v in d.columns:
                d[f"{v}_roll{w}_mean"] = d[v].shift(1).rolling(w, min_periods=w).mean()

    # 4) Drop astro gốc / datetime nếu muốn
    if drop_original_astro:
        for c in ["sunrise", "sunset"]:
            if c in d.columns:
                d.drop(columns=[c], inplace=True)
    if drop_datetime_cols:
        dt_cols = [c for c in d.columns if pd.api.types.is_datetime64_any_dtype(d[c])]
        if dt_cols:
            d.drop(columns=dt_cols, inplace=True)

    # 5) Cắt phần thiếu do lag/rolling
    drop_n = max(max(lags) if lags else 0, max(roll_windows) if roll_windows else 0)
    if drop_n > 0:
        d = d.iloc[drop_n:].reset_index(drop=True)

    # 6) Target cho Direct
    if make_target and "temp" in d.columns:
        d["target_temp_tplus"] = d["temp"].shift(-horizon)
        d = d.iloc[:-horizon].reset_index(drop=True)

    return d

# ====== ghép khung feature train/test ======
def select_feature_frame(
    train_fe: pd.DataFrame,
    test_fe: pd.DataFrame,
    target_col: str,
) -> Tuple[List[str], Tuple[pd.DataFrame,pd.Series], Tuple[pd.DataFrame,pd.Series]]:
    feat_cols = sorted([c for c in train_fe.columns if c != target_col])
    X_tr, y_tr = train_fe[feat_cols], train_fe[target_col]
    X_te = test_fe.reindex(columns=feat_cols)
    y_te = test_fe[target_col]
    X_te, y_te = X_te.dropna(), y_te.loc[X_te.index]
    return feat_cols, (X_tr, y_tr), (X_te, y_te)

def make_direct_datasets_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    horizons: Iterable[int] = (1,2,3,4,5),
    lags: Iterable[int] = (1,2,3,7,14),
    roll_windows: Iterable[int] = (7,14),
) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    y_col = "target_temp_tplus"
    for h in horizons:
        tr = feature_engineer_timeseries(train_df, lags=lags, roll_windows=roll_windows, horizon=h, make_target=True)
        te = feature_engineer_timeseries(test_df,  lags=lags, roll_windows=roll_windows, horizon=h, make_target=True)
        feat_cols, (X_tr, y_tr), (X_te, y_te) = select_feature_frame(tr, te, y_col)
        out[h] = {"train": (X_tr, y_tr), "test": (X_te, y_te), "feature_cols": feat_cols}
    return out

# ====== audit / assert ======
def audit_missing(df: pd.DataFrame, top_k: int = 20) -> Dict[str, Dict[str, int]]:
    nan_counts = df.isna().sum().sort_values(ascending=False)
    num_df = df.select_dtypes(include=[np.number])
    inf_counts = pd.Series(dtype=int)
    if not num_df.empty:
        inf_counts = np.isinf(num_df).sum().sort_values(ascending=False)
    report = {
        "nan": nan_counts[nan_counts > 0].head(top_k).to_dict(),
        "inf": inf_counts[inf_counts > 0].head(top_k).to_dict(),
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
    }
    return report

def assert_no_nan_numeric(df: pd.DataFrame, exclude: List[str] | None = None):
    ex = set(exclude or [])
    num_cols = [c for c in df.columns if (c not in ex and pd.api.types.is_numeric_dtype(df[c]))]
    if not num_cols:
        return
    num_df = df[num_cols]
    has_nan = num_df.isna().any().any()
    has_inf = np.isinf(num_df).any().any()
    if has_nan or has_inf:
        rep = audit_missing(num_df)
        raise ValueError(f"[ASSERT] basic_clean còn NaN/Inf ở numeric columns: {rep}")

_DATA_PATH = "data/weather.parquet"   
_TRAIN_FRAC  = 0.80
_GAP         = 14
_HORIZONS    = (1, 2, 3, 4, 5)
_LAGS        = (1, 2, 3, 7, 14)
_ROLLS       = (7, 14)

def _inspect_time_index(df: pd.DataFrame, dt_col: str = "datetime") -> None:
    """In thông tin phạm vi thời gian, ngày cuối cùng, số ngày thiếu / trùng."""
    if dt_col not in df.columns:
        print(f"[WARN] Không thấy cột thời gian '{dt_col}'.")
        return

    d = df.copy()
    d[dt_col] = pd.to_datetime(d[dt_col], errors="coerce")
    d = d.dropna(subset=[dt_col]).sort_values(dt_col).reset_index(drop=True)

    if d.empty:
        print("[INFO] DataFrame rỗng sau khi chuẩn hoá thời gian.")
        return

    min_dt = d[dt_col].min().normalize()
    max_dt = d[dt_col].max().normalize()
    n_rows = len(d)

    # kiểm tra trùng lặp timestamp
    dup_cnt = int(d.duplicated(subset=[dt_col]).sum())

    # kiểm tra thiếu ngày (giả định dữ liệu daily)
    # tạo dải ngày liên tục từ min_dt -> max_dt
    full_range = pd.date_range(min_dt, max_dt, freq="D")
    missing = full_range.difference(d[dt_col].dt.normalize().unique())
    missing_cnt = len(missing)

    print("=== TIME INDEX INSPECT ===")
    print(f"Rows        : {n_rows}")
    print(f"Start date  : {min_dt.date()}")
    print(f"Last date   : {max_dt.date()}  <-- ngày cuối cùng hiện có")
    print(f"Duplicates  : {dup_cnt}")
    print(f"Missing days: {missing_cnt}")
    if missing_cnt > 0:
        # in tối đa 10 ngày thiếu đầu tiên để gợi ý
        head_list = [ts.date() for ts in missing[:10]]
        more = f" (+{missing_cnt-10} more)" if missing_cnt > 10 else ""
        print(f"  e.g. missing: {head_list}{more}")
    print("==========================")


def _summ_na_rate(df: pd.DataFrame) -> pd.Series:
    return (df.isna().sum() / len(df)).sort_values(ascending=False)

def inspect_new_vs_old_columns(
    df_raw: pd.DataFrame,
    *,
    dt_col: str = "datetime",
    last_k_days: int = 14,
    baseline_days: int = 60,   # cửa sổ “cũ” để so sánh
    show_top: int = 20
):
    d0 = df_raw.copy()
    d0[dt_col] = pd.to_datetime(d0[dt_col], errors="coerce")
    d0 = d0.sort_values(dt_col).reset_index(drop=True)

    last_dt = pd.to_datetime(d0[dt_col].max()).normalize()
    recent_cut = last_dt - pd.Timedelta(days=last_k_days-1)
    base_end  = recent_cut - pd.Timedelta(days=1)
    base_start= base_end - pd.Timedelta(days=baseline_days-1)

    recent = d0[d0[dt_col] >= recent_cut]
    base   = d0[(d0[dt_col] >= base_start) & (d0[dt_col] <= base_end)]

    print(f"[NEW/OLD SPLIT] last_dt={last_dt.date()} | recent: >= {recent_cut.date()} (n={len(recent)}) | "
          f"baseline: {base_start.date()}..{base_end.date()} (n={len(base)})")

    # 1) Tỷ lệ NaN theo cột
    na_recent = _summ_na_rate(recent)
    na_base   = _summ_na_rate(base).reindex(na_recent.index).fillna(0)
    delta = (na_recent - na_base).sort_values(ascending=False)

    print("\n== Cột thiếu nhiều hơn ở NGÀY MỚI (delta NaN rate) ==")
    df_delta = pd.DataFrame({
        "na_recent_rate": na_recent,
        "na_base_rate": na_base,
        "delta_recent_minus_base": delta
    }).sort_values("delta_recent_minus_base", ascending=False)
    print(df_delta.head(show_top).round(3).to_string())

    # 2) Những cột mà baseline hầu như đầy đủ (<5% NaN) nhưng recent trống nhiều (>50% NaN)
    problematic = df_delta[(df_delta["na_base_rate"] <= 0.05) & (df_delta["na_recent_rate"] >= 0.50)]
    if not problematic.empty:
        print("\n== Cột baseline gần như đầy đủ nhưng recent thiếu nhiều (gợi ý nguyên nhân rụng hàng) ==")
        print(problematic.sort_values("na_recent_rate", ascending=False).round(3).to_string())
    else:
        print("\n(no columns where baseline ~full but recent is very sparse)")

    # 3) Thống kê nhanh min/max ở baseline vs recent cho các cột số
    num_cols = [c for c in d0.columns if pd.api.types.is_numeric_dtype(d0[c])]
    if num_cols:
        def _minmax(df): 
            return pd.DataFrame({"min": df[num_cols].min(), "max": df[num_cols].max()})
        print("\n== Min/Max numeric: baseline vs recent (để phát hiện cột toàn 0/NaN) ==")
        mm_base = _minmax(base); mm_recent = _minmax(recent)
        mm = mm_base.join(mm_recent, lsuffix="_base", rsuffix="_recent")
        print(mm.head(20).to_string())

    # 4) In danh sách cột “core” có NaN trong recent
    core = [c for c in CORE_NUMERIC_COLS if c in d0.columns]
    if core:
        core_recent_nan = recent[core].isna().mean().sort_values(ascending=False)
        print("\n== CORE columns NaN rate in recent ==")
        print(core_recent_nan.to_frame("recent_nan_rate").round(3).to_string())

def main():
    print("==> Load data (parquet)")
    df_raw = load_from_parquet(_DATA_PATH)  # dùng module HCM_temp_forcast.data
    print(f"Raw shape: {df_raw.shape}")

    # Kiểm tra phạm vi thời gian & ngày cuối cùng TRƯỚC khi clean
    _inspect_time_index(df_raw, dt_col="datetime")

    print("==> basic_clean (drop_na_core=True)")
    df = basic_clean(df_raw, drop_text_cols=True, drop_na_core=True)
    print(f"After clean: {df.shape}")

    # Kiểm tra phạm vi thời gian & ngày cuối cùng SAU khi clean
    _inspect_time_index(df, dt_col="datetime")

    # Báo cáo & assert sạch NaN numeric
    rep = audit_missing(df, top_k=20)
    print("Missing report (post-clean):", rep)
    assert_no_nan_numeric(df, exclude=["datetime", "sunrise", "sunset"])

    print("==> time_split_train_test (train+GAP+test)")
    train_df, test_df, gap_used = time_split_train_test(
        df, datetime_col="datetime", train_frac=_TRAIN_FRAC, gap=_GAP
    )
    print(f"Train: {train_df.shape}, Test: {test_df.shape}, GAP={gap_used}")
    # In ngày cuối train và ngày đầu test để bạn đối chiếu
    if not train_df.empty:
        print("  Train last date:", pd.to_datetime(train_df["datetime"]).max().date())
    if not test_df.empty:
        print("  Test first date:", pd.to_datetime(test_df["datetime"]).min().date())

    print("==> make_direct_datasets_train_test")
    datasets = make_direct_datasets_train_test(
        train_df, test_df,
        horizons=_HORIZONS, lags=_LAGS, roll_windows=_ROLLS
    )

    for h in sorted(datasets):
        Xtr, ytr = datasets[h]["train"]
        Xte, yte = datasets[h]["test"]
        nan_tr = int(Xtr.isna().sum().sum() + ytr.isna().sum())
        nan_te = int(Xte.isna().sum().sum() + yte.isna().sum())
        print(f"[H{h}] X_tr: {Xtr.shape}, y_tr: {ytr.shape}, X_te: {Xte.shape}, y_te: {yte.shape}, "
              f"NaN(tr/te)={nan_tr}/{nan_te}, #features={len(datasets[h]['feature_cols'])}")

    print("OK! File prepare.py test xong — sẵn sàng cho bước tune/train.")

    print("==> Load data (parquet)")
    df_raw = load_from_parquet(_DATA_PATH)
    print(f"Raw shape: {df_raw.shape}")
    print("Raw last date:", pd.to_datetime(df_raw["datetime"]).max().date())

    print("==> basic_clean (drop_na_core=True)")
    df = basic_clean(df_raw, drop_text_cols=True, drop_na_core=True)
    print(f"After clean: {df.shape}")
    print("Clean last date:", pd.to_datetime(df["datetime"]).max().date())

    print("==> Column diagnostics: recent vs baseline")
    inspect_new_vs_old_columns(
        df_raw,
        dt_col="datetime",
        last_k_days=14,    # cửa sổ “ngày mới” bạn quan tâm
        baseline_days=60,  # so với 60 ngày trước đó
        show_top=20
    )


if __name__ == "__main__":
    main()
