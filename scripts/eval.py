# scripts/evaluate_artifacts_113.py
from __future__ import annotations
import os
import glob
import json
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from HCM_temp_forcast.data import load_from_parquet
from HCM_temp_forcast.prepare import (
    basic_clean,
    time_split_train_test,
    make_direct_datasets_train_test,
)

# ==== CẤU HÌNH PHẢI KHỚP VỚI tune.py ====
DATA_PATH   = "data/weather.parquet"
TRAIN_FRAC  = 0.80
GAP_LMAX    = 30
HORIZONS    = (1, 2, 3, 4, 5)

ARTIFACT_DIR   = "artifacts_113"
EVAL_CSV       = "eval_artifacts_113.csv"
EVAL_JSON      = "eval_artifacts_113.json"
# ========================================


def _load_artifacts(artifact_dir: str) -> Dict[int, List[dict]]:
    """
    Đọc tất cả file joblib trong artifact_dir dạng:
      pipeline_H{h}_{model}.joblib

    Trả về:
      {h: [ {"model_name": str, "pipeline": Pipeline, "feature_cols": [...]} , ... ]}
    (thực tế mỗi horizon thường chỉ có 1 model, nhưng hàm này support nhiều).
    """
    bank: Dict[int, List[dict]] = {}
    pattern = os.path.join(artifact_dir, "pipeline_H*_*.joblib")

    for path in glob.glob(pattern):
        obj = joblib.load(path)
        fname = os.path.basename(path)  # vd: pipeline_H3_gbrt.joblib

        parts = fname.split("_")
        # parts: ["pipeline", "H3", "{model}.joblib" hoặc ["pipeline","H3","xgboost.joblib",...]]
        try:
            h_str = parts[1]  # "H3"
            h = int(h_str[1:])
        except Exception:
            print(f"[WARN] Không parse được horizon từ '{fname}', bỏ qua.")
            continue

        # model_name: gộp các phần còn lại trừ đuôi .joblib
        model_name = "_".join(parts[2:])
        if model_name.lower().endswith(".joblib"):
            model_name = model_name[:-7]

        rec = {
            "model_name": model_name,
            "pipeline": obj["pipeline"],
            "feature_cols": obj.get("feature_cols"),
            "path": path,
        }
        bank.setdefault(h, []).append(rec)

    if not bank:
        raise FileNotFoundError(f"Không tìm thấy artifact trong '{artifact_dir}'")
    return bank


def main():
    # 1) Load & clean dữ liệu như tune.py
    print("==> Load parquet & basic_clean")
    df_raw = load_from_parquet(DATA_PATH)
    df = basic_clean(df_raw, drop_text_cols=True, drop_na_core=True)
    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)

    # 2) Split train/test + build direct datasets
    print("==> time_split_train_test")
    train_df, test_df, _ = time_split_train_test(
        df,
        train_frac=TRAIN_FRAC,
        gap=GAP_LMAX,
        datetime_col="datetime",
    )
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    print("==> make_direct_datasets_train_test")
    datasets = make_direct_datasets_train_test(
        train_df,
        test_df,
        horizons=HORIZONS,
    )

    # 3) Nạp artifacts
    print(f"==> Load artifacts from {ARTIFACT_DIR}")
    bank = _load_artifacts(ARTIFACT_DIR)

    results = []

    # 4) Evaluate
    for h in sorted(HORIZONS):
        if h not in bank:
            print(f"[WARN] Không có artifact cho H{h}, bỏ qua.")
            continue

        X_train, y_train = datasets[h]["train"]
        X_test,  y_test  = datasets[h]["test"]

        for rec in bank[h]:
            model_name  = rec["model_name"]
            pipe        = rec["pipeline"]
            feat_cols   = rec["feature_cols"]

            # Reindex để chắc chắn thứ tự cột giống lúc train
            if feat_cols is not None:
                Xtr = X_train.reindex(columns=feat_cols)
                Xte = X_test.reindex(columns=feat_cols)
            else:
                # fallback: dùng toàn bộ cột numeric hiện có
                num_cols = X_train.select_dtypes(include=[np.number]).columns
                Xtr = X_train[num_cols]
                Xte = X_test[num_cols]

            # Bỏ hàng NA (nếu có) cho an toàn
            tr_valid = Xtr.dropna()
            ytr_valid = y_train.loc[tr_valid.index]

            te_valid = Xte.dropna()
            yte_valid = y_test.loc[te_valid.index]

            # --- Train split metrics (dùng model đã fit sẵn, KHÔNG fit lại) ---
            ytr_pred = pipe.predict(tr_valid)
            mae_tr   = mean_absolute_error(ytr_valid, ytr_pred)
            rmse_tr  = mean_squared_error(ytr_valid, ytr_pred) ** 0.5
            r2_tr    = r2_score(ytr_valid, ytr_pred)

            results.append(
                {
                    "horizon": h,
                    "model": model_name,
                    "split": "train",
                    "n_samples": int(len(ytr_valid)),
                    "mae": float(mae_tr),
                    "rmse": float(rmse_tr),
                    "r2": float(r2_tr),
                }
            )

            # --- Test split metrics ---
            yte_pred = pipe.predict(te_valid)
            mae_te   = mean_absolute_error(yte_valid, yte_pred)
            rmse_te  = mean_squared_error(yte_valid, yte_pred) ** 0.5
            r2_te    = r2_score(yte_valid, yte_pred)

            results.append(
                {
                    "horizon": h,
                    "model": model_name,
                    "split": "test",
                    "n_samples": int(len(yte_valid)),
                    "mae": float(mae_te),
                    "rmse": float(rmse_te),
                    "r2": float(r2_te),
                }
            )

            print(
                f"[H{h}][{model_name}] "
                f"TRAIN n={len(ytr_valid)} MAE={mae_tr:.4f} RMSE={rmse_tr:.4f} R2={r2_tr:.4f} | "
                f"TEST n={len(yte_valid)} MAE={mae_te:.4f} RMSE={rmse_te:.4f} R2={r2_te:.4f}"
            )

    df_eval = pd.DataFrame(results).sort_values(["horizon", "model", "split"])
    print("\n=== Summary ===")
    print(df_eval.to_string(index=False))

    # 5) Lưu CSV + JSON
    df_eval.to_csv(EVAL_CSV, index=False)
    with open(EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "artifacts_dir": ARTIFACT_DIR,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n[OK] Saved evaluation -> {EVAL_CSV}, {EVAL_JSON}")


if __name__ == "__main__":
    main()
