# app.py
from __future__ import annotations
import os
import pandas as pd
import gradio as gr
from datetime import date

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

_df_clean_cache = None


def _load_clean_data() -> pd.DataFrame:
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


def _default_current_date_str() -> str:
    df = _load_clean_data()
    last_dt = pd.to_datetime(df[DATE_COL]).max().date() if not df.empty else date.today()
    return last_dt.isoformat()


def _render_cards_html(
    df_cards: pd.DataFrame,
    data_range: tuple[date, date],
    current_date: str,
    fetched_note: str = ""
) -> str:
    css = """
    <style>
    .wx-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px;margin-top:12px}
    @media(max-width:1100px){.wx-grid{grid-template-columns:repeat(3,minmax(0,1fr))}}
    @media(max-width:700px){.wx-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
    .wx-card{
      background:#ffffff;
      border:1px solid #e7eaf0;
      padding:16px 18px;
      border-radius:16px;
      box-shadow:0 2px 10px rgba(0,0,0,.04);
    }
    .wx-title{font-weight:700; font-size:18px; margin-bottom:6px; color:#111827}
    .wx-temp{font-size:30px; font-weight:800; margin:4px 0 10px 0; color:#0f172a}
    .wx-row{font-size:13px; color:#334155; display:flex; justify-content:space-between}
    .wx-badge{
      display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px;
      background:#f1f5f9; border:1px solid #e2e8f0; color:#0f172a
    }
    .wx-head{display:flex; align-items:center; justify-content:space-between}
    .wx-note{font-size:13px; color:#475569; margin-top:6px}
    .wx-hbar{display:flex; gap:8px; align-items:center; color:#475569; font-size:13px}
    .wx-hbar b{color:#0f172a}
    .wx-fetch{font-size:12px; color:#16a34a; margin-top:4px}
    .wx-fetch .err{color:#dc2626}
    </style>
    """
    dr_min, dr_max = data_range
    fetch_html = f'<div class="wx-fetch">{fetched_note}</div>' if fetched_note else ""
    head = f"""
      <div class="wx-head">
        <div>
          <div class="wx-title">Dự báo nhiệt độ 5 ngày tiếp theo</div>
          <div class="wx-hbar">
            <span>Ngày gốc (t): <b>{current_date}</b></span>
          </div>
          {fetch_html}
        </div>
        <span class="wx-badge">t+1 → t+5</span>
      </div>
    """

    cards_html = []
    for _, r in df_cards.iterrows():
        tdate = pd.to_datetime(r["target_date"]).date()
        temp  = f'{float(r["pred_temp"]):.{DECIMALS}f}°C'
        gt    = r.get("gt_temp", float("nan"))
        err   = r.get("abs_error", float("nan"))

        gt_line = ""
        if pd.notna(gt):
            gt_line = f"""
              <div class="wx-row"><span>Thực tế</span><span><b>{float(gt):.{DECIMALS}f}°C</b></span></div>
              <div class="wx-row"><span>Sai số tuyệt đối</span><span>{float(err):.{DECIMALS}f}°C</span></div>
            """

        card = f"""
          <div class="wx-card">
            <div class="wx-row"><span>t+{int(r['horizon_days'])}</span><span>{tdate}</span></div>
            <div class="wx-temp">{temp}</div>
            {gt_line}
          </div>
        """
        cards_html.append(card)

    grid = f'<div class="wx-grid">{"".join(cards_html)}</div>'
    return css + head + grid


def predict(current_date_str: str):
    global _df_clean_cache
    fetched_note = ""

    # thử fetch incremental nếu có
    if incremental_fetch_via_xlsx is not None:
        try:
            incremental_fetch_via_xlsx()
            fetched_note = ""  # có thể thêm message nếu muốn
            _df_clean_cache = None
        except Exception as e:
            fetched_note = f'<span class="err">Không fetch được dữ liệu mới ({e}). Dùng dữ liệu hiện có.</span>'

    try:
        df = _load_clean_data()
        dt_min = pd.to_datetime(df[DATE_COL]).min().date() if not df.empty else None
        dt_max = pd.to_datetime(df[DATE_COL]).max().date() if not df.empty else None

        # validate date
        try:
            cur = pd.to_datetime(current_date_str).date()
        except Exception:
            return gr.update(value="<b>❌ Ngày không hợp lệ.</b>")

        if dt_min is not None and cur < dt_min:
            return gr.update(value=f"<b>❌</b> Ngày gốc {cur} < min date trong dữ liệu ({dt_min}).")
        if dt_max is not None and cur > dt_max:
            return gr.update(value=f"<b>❌</b> Ngày gốc {cur} > max date trong dữ liệu ({dt_max}). Hãy fetch dữ liệu mới trước.")

        # FE mới đã cố định spec lags/rolling bên trong infer,
        # nên ở đây chỉ cần truyền df, horizons, không cần LAGS/ROLLS nữa.
        out = forecast_next_5_days(
            df_recent=df,
            current_date=cur,
            artifact_dir=ARTIFACT_DIR,
            horizons=HORIZONS,
            df_truth=df,
            date_col=DATE_COL,
            truth_col=TRUTH_COL,
        )

        cols = ["horizon_days", "target_date", "pred_temp"]
        if "gt_temp" in out.columns:
            cols += ["gt_temp", "abs_error"]
        disp = out[cols].copy()

        html = _render_cards_html(disp, (dt_min, dt_max), cur, fetched_note=fetched_note)
        return gr.update(value=html)

    except FileNotFoundError as fe:
        return gr.update(value=f"<b>❌ {fe}</b>")
    except KeyError as ke:
        return gr.update(value=f"<b>❌ Thiếu pipeline/feature:</b> {ke}")
    except ValueError as ve:
        return gr.update(value=f"<b>❌ Lỗi:</b> {ve}")
    except Exception as e:
        return gr.update(value=f"<b>❌ Lỗi bất ngờ:</b> {e}")


# ----------------- Gradio UI -----------------
theme = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="slate",
).set(
    body_background_fill="#ffffff",
    block_background_fill="#ffffff",
)

with gr.Blocks(title="Dự báo nhiệt độ thành phố Hồ Chí Minh", theme=theme) as demo:
    gr.Markdown("## 🌤️ Dự báo nhiệt độ thành phố Hồ Chí Minh — 5 ngày tiếp theo")

    current_date_in = gr.Textbox(
        label="Ngày gốc (YYYY-MM-DD)",
        value=_default_current_date_str(),
        info="Chỉ nhập ngày gốc (t). App sẽ dự báo t+1..t+5.",
    )
    btn = gr.Button("🚀 Dự báo", variant="primary")

    cards = gr.HTML()

    btn.click(fn=predict, inputs=current_date_in, outputs=cards)

if __name__ == "__main__":
    demo.launch()
