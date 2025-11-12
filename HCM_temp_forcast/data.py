from typing import Optional, List, Dict
import pandas as pd

def load_from_parquet(
    path: str,
    datetime_candidates: List[str] = ("datetime", "date", "datetime_ts", "time", "timestamp"),
    rename_map: Optional[Dict[str, str]] = None,
    lower_strip_cols: bool = True,
) -> pd.DataFrame:
    """
    Load weather data từ một file .parquet
    - Chuẩn hoá tên cột (lower/strip) nếu cần
    - Áp dụng rename_map nếu cung cấp
    - Đảm bảo tồn tại cột thời gian chuẩn tên 'datetime'
      (nếu chưa có, sẽ tìm trong datetime_candidates và chuẩn hoá sang datetime64[ns])
    """
    # read parquet (cần pyarrow hoặc fastparquet)
    df = pd.read_parquet(path)

    # normalize column names
    if lower_strip_cols:
        df.columns = [str(c).strip().lower() for c in df.columns]

    # optional explicit rename
    if rename_map:
        lowered_map = {str(k).lower(): v for k, v in rename_map.items()}
        df = df.rename(columns=lowered_map)

    # standardize a datetime column named 'datetime'
    if "datetime" in df.columns:
        # nếu đã có nhưng chưa phải dtype datetime -> cố gắng parse
        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=False)
    else:
        dt_col = None
        for c in datetime_candidates:
            if c in df.columns:
                dt_col = c
                break
        if dt_col is None:
            raise ValueError(
                f"Không tìm thấy cột thời gian trong {datetime_candidates}. "
                f"Columns={list(df.columns)[:12]}..."
            )
        df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce", utc=False)

    return df
