# HCM_temp_forcast/prepare.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, List, Optional, Tuple, Dict
from HCM_temp_forcast.data import load_from_parquet

# ====== cấu hình mặc định ======
TIME_COLS_DEFAULT = ("datetime", "sunrise", "sunset")
TEXT_COLS_DEFAULT = ["conditions", "icon", "description"]
DROP_ALWAYS = ["stations", "severerisk", "name", "solarenergy"]

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
    lags: Iterable[int] = (1, 2, 3, 7, 14),   # giữ để tương thích, không dùng trực tiếp
    roll_windows: Iterable[int] = (7, 14),    # giữ để tương thích, không dùng trực tiếp
    horizon: int = 1,
    drop_original_astro: bool = True,
    drop_datetime_cols: bool = True,
    make_target: bool = True
) -> pd.DataFrame:
    """
    Feature engineering cho daily weather time series, theo spec:

    - Lag & rolling riêng cho từng biến (temp, tempmax, tempmin, humidity, dew, pressure, precip, v.v.)
    - Winddir (deg) -> rad, tạo u/v, wind_proficiency.
    - Temporal, domain, interaction features như đã thiết kế.
    - Rolling windows là trailing, KHÔNG shift(1), vì target là temp_{t+1}.

    Đã tối ưu để tránh DataFrame fragmentation: tất cả cột mới được gom vào dict
    `new_cols` rồi concat một lần.
    """
    d = df.copy()

    # 0) Chuẩn dtype thời gian
    for c in ["datetime", "sunrise", "sunset"]:
        if c in d.columns and not pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = pd.to_datetime(d[c], errors="coerce")

    if "datetime" not in d.columns:
        raise ValueError("Thiếu cột 'datetime' trước FE.")

    # Dict chứa tất cả feature mới
    new_cols: Dict[str, pd.Series] = {}

    # 1) Temporal / calendar features
    month = d["datetime"].dt.month
    doy   = d["datetime"].dt.dayofyear
    dow   = d["datetime"].dt.dayofweek

    new_cols["feature_Month"]      = month
    new_cols["feature_DayOfYear"]  = doy
    new_cols["feature_DayOfWeek"]  = dow
    new_cols["feature_IsWeekend"]  = dow.isin([5, 6]).astype(int)

    new_cols["feature_Month_sin"]     = np.sin(2 * np.pi * month / 12.0)
    new_cols["feature_Month_cos"]     = np.cos(2 * np.pi * month / 12.0)
    new_cols["feature_DayOfYear_sin"] = np.sin(2 * np.pi * doy / 366.0)
    new_cols["feature_DayOfYear_cos"] = np.cos(2 * np.pi * doy / 366.0)

    if "has_rain" in d.columns:
        new_cols["feature_Has_Rain"] = d["has_rain"]

    # Daylight duration (giờ)
    if {"sunrise", "sunset"}.issubset(d.columns):
        dur_hours = (d["sunset"] - d["sunrise"]).dt.total_seconds() / 3600.0
        new_cols["feature_Daylight_Duration"] = dur_hours

    # Mùa mưa: 1 nếu tháng 5..11
    new_cols["feature_Is_RainySeason"] = month.between(5, 11).astype(int)

    # 2) Wind direction & components
    if "winddir" in d.columns:
        winddir_deg = pd.to_numeric(d["winddir"], errors="coerce")
        winddir_rad = np.deg2rad(winddir_deg)
    else:
        winddir_rad = pd.Series(np.nan, index=d.index)

    new_cols["winddir_rad"] = winddir_rad

    if "windspeed" in d.columns:
        ws = pd.to_numeric(d["windspeed"], errors="coerce")
        new_cols["wind_u"] = ws * np.sin(winddir_rad)   # Đông - Tây
        new_cols["wind_v"] = ws * np.cos(winddir_rad)   # Bắc - Nam
        new_cols["wind_proficiency"] = np.sin(winddir_rad - np.pi / 2.0) * ws
    else:
        ws = None  # đề phòng dùng ws phía sau

    # 2.5) EWMA (exponentially weighted mean) – chọn lọc
    if "temp" in d.columns:
        temp_num = pd.to_numeric(d["temp"], errors="coerce")
        new_cols["temp_ewm7"] = (
            temp_num.ewm(span=7, min_periods=7, adjust=False).mean()
        )
        new_cols["temp_ewm30"] = (
            temp_num.ewm(span=30, min_periods=30, adjust=False).mean()
        )

    if "humidity" in d.columns:
        hum_num = pd.to_numeric(d["humidity"], errors="coerce")
        new_cols["humidity_ewm7"] = (
            hum_num.ewm(span=7, min_periods=7, adjust=False).mean()
        )

    if "precip" in d.columns:
        pr_num = pd.to_numeric(d["precip"], errors="coerce")
        new_cols["precip_ewm7"] = (
            pr_num.ewm(span=7, min_periods=7, adjust=False).mean()
        )


    # 3) Lag specifications (variable-specific)
    lag_spec: Dict[str, List[int]] = {
        # Temp, TempMax, TempMin
        "temp":             [1, 2, 3, 7],
        # "tempmax":          [1, 2, 3, 7],
        # "tempmin":          [1, 2, 3, 7],
        # Humidity, Dew
        "humidity":         [1, 2, 3],
        # "dew":              [1, 2, 3],
        # Sealevel Pressure
        "sealevelpressure": [1],
        # precip
        "precip":           [1, 2, 3],
        # cloudcover
        "cloudcover":       [1, 2, 3],
        # solarradiation
        "solarradiation":   [1, 2, 3],
        # solarenergy
        # "solarenergy":      [1, 2, 3],
        # uvindex
        # "uvindex":          [1, 2, 3],
        # windspeed
        "windspeed":        [1, 2, 5],
        # windgust
        "windgust":         [1, 2, 5],
    }

    def _get_series_for(col: str) -> Optional[pd.Series]:
        if col in d.columns:
            return pd.to_numeric(d[col], errors="coerce")
        if col in new_cols:
            return pd.to_numeric(new_cols[col], errors="coerce")
        return None

    for col, ks in lag_spec.items():
        s = _get_series_for(col)
        if s is None:
            continue
        for k in ks:
            new_cols[f"{col}_lag{k}"] = s.shift(k)

    # 4) Rolling statistics (trailing, bao gồm cả ngày t)
    mean_spec: Dict[str, List[int]] = {
        "temp":             [3, 7, 14, 30],
        # "tempmax":          [3, 7, 14, 30],
        # "tempmin":          [3, 7, 14, 30],
        "humidity":         [3, 7],
        # "dew":              [3, 7],
        "sealevelpressure": [3, 7],
        "precip":           [2, 3, 5, 7],
        "cloudcover":       [2, 7, 14],
        "solarradiation":   [2, 7, 14],
        # "solarenergy":      [14, 30],
        # "uvindex":          [7, 14, 30],
        "windspeed":        [3, 7, 14],
        "windgust":         [3, 7, 14],
        "wind_u":           [7],
        "wind_v":           [7],
    }

    std_spec: Dict[str, List[int]] = {
        "temp":             [7, 14],
        # "tempmax":          [7, 14],
        # "tempmin":          [7, 14],
        "humidity":         [7],
        # "dew":              [7],
        "sealevelpressure": [7, 30],
        "precip":           [3, 5, 7],
        "cloudcover":       [3, 7],
        "solarradiation":   [3, 7, 14],
        # "solarenergy":      [7, 14],
        # "uvindex":          [7, 14],
    }

    max_spec: Dict[str, List[int]] = {
        "windgust":         [3, 7, 14],
    }

    def _roll_trailing(col: str, window: int, stat: str) -> Optional[pd.Series]:
        s = _get_series_for(col)
        if s is None:
            return None
        r = s.rolling(window, min_periods=window)
        if stat == "mean":
            return r.mean()
        if stat == "std":
            return r.std()
        if stat == "max":
            return r.max()
        raise ValueError(f"Unsupported stat={stat}")

    for col, ws_list in mean_spec.items():
        for w in ws_list:
            res = _roll_trailing(col, w, "mean")
            if res is not None:
                new_cols[f"{col}_roll{w}_mean"] = res

    for col, ws_list in std_spec.items():
        for w in ws_list:
            res = _roll_trailing(col, w, "std")
            if res is not None:
                new_cols[f"{col}_roll{w}_std"] = res

    for col, ws_list in max_spec.items():
        for w in ws_list:
            res = _roll_trailing(col, w, "max")
            if res is not None:
                new_cols[f"{col}_roll{w}_max"] = res

    # 5) Domain & interaction features

    # Num_Rain7Day
    if "has_rain" in d.columns:
        new_cols["Num_Rain7Day"] = (
            pd.to_numeric(d["has_rain"], errors="coerce")
            .rolling(7, min_periods=7)
            .sum()
        )

    # domain_temp_range = tempmax - tempmin
    if {"tempmax", "tempmin"}.issubset(d.columns):
        new_cols["domain_temp_range"] = (
            pd.to_numeric(d["tempmax"], errors="coerce")
            - pd.to_numeric(d["tempmin"], errors="coerce")
        )

    # domain_dew_point_spread = temp - dew
    if {"temp", "dew"}.issubset(d.columns):
        new_cols["domain_dew_point_spread"] = (
            pd.to_numeric(d["temp"], errors="coerce")
            - pd.to_numeric(d["dew"], errors="coerce")
        )

    # domain_feelslike_diff = feelslike - temp
    if {"feelslike", "temp"}.issubset(d.columns):
        new_cols["domain_feelslike_diff"] = (
            pd.to_numeric(d["feelslike"], errors="coerce")
            - pd.to_numeric(d["temp"], errors="coerce")
        )

    # domain_pressure_change_1d = sealevelpressure - sealevelpressure_lag1
    if "sealevelpressure" in d.columns and "sealevelpressure_lag1" in new_cols:
        sp  = pd.to_numeric(d["sealevelpressure"], errors="coerce")
        sp1 = pd.to_numeric(new_cols["sealevelpressure_lag1"], errors="coerce")
        new_cols["domain_pressure_change_1d"] = sp - sp1

    # domain_wind_power = windspeed^2
    if "windspeed" in d.columns:
        ws_num = pd.to_numeric(d["windspeed"], errors="coerce")
        new_cols["domain_wind_power"] = ws_num ** 2

    # domain_wind_gust_ratio = windgust / (windspeed + 1e-6)
    if {"windgust", "windspeed"}.issubset(d.columns):
        gust = pd.to_numeric(d["windgust"], errors="coerce")
        ws_num = pd.to_numeric(d["windspeed"], errors="coerce")
        new_cols["domain_wind_gust_ratio"] = gust / (ws_num + 1e-6)

    # domain_cloud_radiation_interaction = solarradiation / (cloudcover + 1e-6)
    if {"solarradiation", "cloudcover"}.issubset(d.columns):
        rad = pd.to_numeric(d["solarradiation"], errors="coerce")
        cc  = pd.to_numeric(d["cloudcover"], errors="coerce")
        new_cols["domain_cloud_radiation_interaction"] = rad / (cc + 1e-6)

    # inter_temp_x_humidity = temp_lag1 * humidity_lag1
    if "temp_lag1" in new_cols and "humidity_lag1" in new_cols:
        t1 = pd.to_numeric(new_cols["temp_lag1"], errors="coerce")
        h1 = pd.to_numeric(new_cols["humidity_lag1"], errors="coerce")
        new_cols["inter_temp_x_humidity"] = t1 * h1

    # inter_humidity_x_dew = (2 - humidity) * dew
    if {"humidity", "dew"}.issubset(d.columns):
        hum = pd.to_numeric(d["humidity"], errors="coerce")
        dew_ = pd.to_numeric(d["dew"], errors="coerce")
        new_cols["inter_humidity_x_dew"] = (2 - hum) * dew_

    # ratio_humidity_pressure = lag_humidity_1 / (lag_sealevelpressure_1 + 1e-6)
    if "humidity_lag1" in new_cols and "sealevelpressure_lag1" in new_cols:
        h1 = pd.to_numeric(new_cols["humidity_lag1"], errors="coerce")
        p1 = pd.to_numeric(new_cols["sealevelpressure_lag1"], errors="coerce")
        new_cols["ratio_humidity_pressure"] = h1 / (p1 + 1e-6)

    # 6) Ghép tất cả feature mới vào DataFrame một lần
    if new_cols:
        features_df = pd.DataFrame(new_cols, index=d.index)
        d = pd.concat([d, features_df], axis=1)
        # copy() để defragment như warning gợi ý
        d = d.copy()

    # 7) Dọn cột astro / winddir nếu cần
    if "moonphase" in d.columns:
        d.drop(columns=["moonphase"], inplace=True)

    if drop_original_astro:
        for c in ["sunrise", "sunset"]:
            if c in d.columns:
                d.drop(columns=[c], inplace=True)

    if "winddir" in d.columns:
        d.drop(columns=["winddir"], inplace=True)

    # 8) Drop datetime-type columns nếu cần
    if drop_datetime_cols:
        dt_cols = [c for c in d.columns if pd.api.types.is_datetime64_any_dtype(d[c])]
        if dt_cols:
            d.drop(columns=dt_cols, inplace=True)

    # 9) Cắt phần đầu bị thiếu do lag/rolling
    all_lags = [k for ks in lag_spec.values() for k in ks]
    all_rolls = (
        [w for ws_ in mean_spec.values() for w in ws_] +
        [w for ws_ in std_spec.values()  for w in ws_] +
        [w for ws_ in max_spec.values()  for w in ws_]
    )
    max_lag  = max(all_lags) if all_lags else 0
    max_roll = max(all_rolls) if all_rolls else 0
    drop_n   = max(max_lag, max_roll)    # -> 30

    if drop_n > 0:
        d = d.iloc[drop_n:].reset_index(drop=True)

    # 10) Target cho Direct: temp_{t+horizon}
    if make_target and "temp" in d.columns:
        temp_series = pd.to_numeric(d["temp"], errors="coerce")
        d["target_temp_tplus"] = temp_series.shift(-horizon)
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
    horizons: Iterable[int] = (1, 2, 3, 4, 5),
) -> Dict[int, Dict]:
    """
    Tạo dataset cho Direct Multi-horizon:
    - Với mỗi horizon h, chạy feature_engineer_timeseries riêng trên train/test
      (FE sử dụng cùng spec lags/rolling cố định).
    - Target mặc định: 'target_temp_tplus'.
    """
    out: Dict[int, Dict] = {}
    y_col = "target_temp_tplus"

    for h in horizons:
        tr = feature_engineer_timeseries(
            train_df,
            horizon=h,
            make_target=True
        )
        te = feature_engineer_timeseries(
            test_df,
            horizon=h,
            make_target=True
        )
        feat_cols, (X_tr, y_tr), (X_te, y_te) = select_feature_frame(tr, te, y_col)
        out[h] = {
            "train": (X_tr, y_tr),
            "test": (X_te, y_te),
            "feature_cols": feat_cols,
        }
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


def main():
    print("==> Load data (parquet)")
    df_raw = load_from_parquet(_DATA_PATH)
    print(f"Raw shape: {df_raw.shape}")

    # Kiểm tra phạm vi thời gian & ngày cuối cùng TRƯỚC khi clean
    _inspect_time_index(df_raw, dt_col="datetime")

    print("\n==> basic_clean (drop_na_core=True)")
    df = basic_clean(df_raw, drop_text_cols=True, drop_na_core=True)
    print(f"After clean: {df.shape}")

    # Kiểm tra phạm vi thời gian & ngày cuối cùng SAU khi clean
    _inspect_time_index(df, dt_col="datetime")

    # Báo cáo & assert sạch NaN numeric
    rep = audit_missing(df, top_k=10)
    print("\nMissing report (post-clean, top 10):", rep)
    assert_no_nan_numeric(df, exclude=["datetime", "sunrise", "sunset"])

    print("\n==> time_split_train_test (train+GAP+test)")
    train_df, test_df, gap_used = time_split_train_test(
        df,
        datetime_col="datetime",
        train_frac=_TRAIN_FRAC,
        gap=_GAP
    )
    print(f"Train: {train_df.shape}, Test: {test_df.shape}, GAP={gap_used}")
    if not train_df.empty:
        print("  Train last date:",
              pd.to_datetime(train_df["datetime"]).max().date())
    if not test_df.empty:
        print("  Test first date:",
              pd.to_datetime(test_df["datetime"]).min().date())

    print("\n==> Feature engineering per horizon")
    y_col = "target_temp_tplus"

    for h in _HORIZONS:
        print(f"\n---- Horizon H={h} ----")
        # 1) Chạy FE riêng trên train/test
        train_fe = feature_engineer_timeseries(
            train_df,
            horizon=h,
            make_target=True,
            drop_original_astro=True,
            drop_datetime_cols=True,
        )
        test_fe = feature_engineer_timeseries(
            test_df,
            horizon=h,
            make_target=True,
            drop_original_astro=True,
            drop_datetime_cols=True,
        )

        print(f"[H{h}] train_fe shape: {train_fe.shape}")
        print(f"[H{h}] test_fe  shape: {test_fe.shape}")

        # 2) Kiểm tra NaN/Inf sau FE
        rep_tr = audit_missing(train_fe, top_k=5)
        rep_te = audit_missing(test_fe, top_k=5)
        print(f"[H{h}] Missing train (top 5): {rep_tr['nan']}")
        print(f"[H{h}] Missing test  (top 5): {rep_te['nan']}")

        # 3) Build X/y để xem #features thực tế
        feat_cols, (X_tr, y_tr), (X_te, y_te) = select_feature_frame(
            train_fe,
            test_fe,
            target_col=y_col,
        )
        print(
            f"[H{h}] X_tr: {X_tr.shape}, y_tr: {y_tr.shape}, "
            f"X_te: {X_te.shape}, y_te: {y_te.shape}, "
            f"#features={len(feat_cols)}"
        )

    print("\nOK! Feature engineering đã chạy xong cho tất cả horizons.")


if __name__ == "__main__":
    main()
