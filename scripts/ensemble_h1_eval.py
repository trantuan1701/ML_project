# scripts/ensemble_h1_eval.py
from __future__ import annotations
import json
import os

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

from HCM_temp_forcast.data import load_from_parquet
from HCM_temp_forcast.prepare import (
    basic_clean,
    time_split_train_test,
    make_direct_datasets_train_test,
)
from HCM_temp_forcast.model import build_model_candidates

# ==== PHẢI KHỚP VỚI tune.py ====
DATA_PATH   = "data/weather.parquet"
TRAIN_FRAC  = 0.80
GAP_LMAX    = 30
HORIZONS    = (1, 2, 3, 4, 5)
TUNE_JSON   = "tune_results_113.json"

# chỉ chơi với H=1
TARGET_H    = 1
MODEL_NAMES = ["linreg", "ridge", "lasso", "rf", "gbr", "xgb", "lgbm"]
# =================================


def _build_final_pipeline(best_name: str, est, k_best: int, n_features: int) -> Pipeline:
    """
    Giống tune.py:
      - Linear models: StandardScaler -> SelectKBest -> model
      - Tree/boosting: SelectKBest -> model
    """
    if k_best is None or k_best <= 0 or k_best > n_features:
        k_best = n_features

    linear_models = {"linreg", "ridge", "lasso"}
    steps = []

    if best_name in linear_models:
        steps.append(("scale", StandardScaler()))

    steps.append(("select", SelectKBest(score_func=f_regression, k=k_best)))
    steps.append(("model", est))

    return Pipeline(steps)


def main():
    # 1) Load + clean + split + FE như đúng bài
    print("==> Load parquet & basic_clean")
    df_raw = load_from_parquet(DATA_PATH)
    df = basic_clean(df_raw, drop_text_cols=True, drop_na_core=True)
    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)

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

    X_train, y_train = datasets[TARGET_H]["train"]
    X_test,  y_test  = datasets[TARGET_H]["test"]
    n_features = X_train.shape[1]
    print(f"[H{TARGET_H}] X_train={X_train.shape}, X_test={X_test.shape}")

    # 2) Load best_params từ tune_results_113.json
    print(f"==> Load tuning results from {TUNE_JSON}")
    with open(TUNE_JSON, "r", encoding="utf-8") as f:
        tune_results = json.load(f)

    h_cfg = tune_results[str(TARGET_H)]

    base_models = build_model_candidates(random_state=42)

    all_preds_test = {}
    all_preds_train = {}
    rows = []

    # 3) Train lại từng model với best_params
    for mname in MODEL_NAMES:
        cfg = h_cfg[mname]
        best_params = cfg["best_params"]
        cv_mae = cfg["best_value"]

        # tách k_best ra
        k_best = best_params.get("k_best", n_features)

        # clone estimator gốc và set hyperparam
        est = clone(base_models[mname])
        valid_keys = est.get_params().keys()
        est.set_params(**{k: v for k, v in best_params.items() if k in valid_keys})

        pipe = _build_final_pipeline(
            best_name=mname,
            est=est,
            k_best=k_best,
            n_features=n_features,
        )

        print(f"\n[TRAIN] Model={mname}, k_best={k_best}, cv_mae={cv_mae:.4f}")
        pipe.fit(X_train, y_train)

        # dự đoán train & test
        yhat_tr = pipe.predict(X_train)
        yhat_te = pipe.predict(X_test)

        all_preds_train[mname] = yhat_tr
        all_preds_test[mname] = yhat_te

        # metrics train
        mae_tr  = mean_absolute_error(y_train, yhat_tr)
        rmse_tr = mean_squared_error(y_train, yhat_tr) ** 0.5
        r2_tr   = r2_score(y_train, yhat_tr)

        rows.append({
            "horizon": TARGET_H,
            "model": mname,
            "split": "train",
            "n": int(len(y_train)),
            "mae": mae_tr,
            "rmse": rmse_tr,
            "r2": r2_tr,
        })

        # metrics test
        mae_te  = mean_absolute_error(y_test, yhat_te)
        rmse_te = mean_squared_error(y_test, yhat_te) ** 0.5
        r2_te   = r2_score(y_test, yhat_te)

        rows.append({
            "horizon": TARGET_H,
            "model": mname,
            "split": "test",
            "n": int(len(y_test)),
            "mae": mae_te,
            "rmse": rmse_te,
            "r2": r2_te,
        })

        print(
            f"[{mname}] TRAIN: MAE={mae_tr:.4f}, RMSE={rmse_tr:.4f}, R2={r2_tr:.4f} | "
            f"TEST: MAE={mae_te:.4f}, RMSE={rmse_te:.4f}, R2={r2_te:.4f}"
        )

    # 4) Ensemble đơn giản (trung bình dự đoán)
    def _add_ensemble(name: str, members: list[str]):
        yhat_tr = np.mean([all_preds_train[m] for m in members], axis=0)
        yhat_te = np.mean([all_preds_test[m] for m in members], axis=0)

        mae_tr  = mean_absolute_error(y_train, yhat_tr)
        rmse_tr = mean_squared_error(y_train, yhat_tr) ** 0.5
        r2_tr   = r2_score(y_train, yhat_tr)

        mae_te  = mean_absolute_error(y_test, yhat_te)
        rmse_te = mean_squared_error(y_test, yhat_te) ** 0.5
        r2_te   = r2_score(y_test, yhat_te)

        rows.extend([
            {
                "horizon": TARGET_H,
                "model": name,
                "split": "train",
                "n": int(len(y_train)),
                "mae": mae_tr,
                "rmse": rmse_tr,
                "r2": r2_tr,
            },
            {
                "horizon": TARGET_H,
                "model": name,
                "split": "test",
                "n": int(len(y_test)),
                "mae": mae_te,
                "rmse": rmse_te,
                "r2": r2_te,
            },
        ])

        print(
            f"\n[ENSEMBLE {name}] "
            f"TRAIN: MAE={mae_tr:.4f}, RMSE={rmse_tr:.4f}, R2={r2_tr:.4f} | "
            f"TEST: MAE={mae_te:.4f}, RMSE={rmse_te:.4f}, R2={r2_te:.4f}"
        )

    # ensemble tất cả
    _add_ensemble("ens_all_mean", MODEL_NAMES)

    # ensemble chỉ các model tree/boosting
    tree_models = ["rf", "gbr", "xgb", "lgbm"]
    _add_ensemble("ens_tree_mean", tree_models)

    # 5) In bảng tổng hợp
    df_res = pd.DataFrame(rows).sort_values(["split", "mae"])
    print("\n=== SUMMARY (sorted by split, MAE) ===")
    print(df_res.to_string(index=False))

    # optionally lưu ra file
    out_path = "ensemble_h1_eval.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\n[OK] Saved metrics -> {out_path}")


if __name__ == "__main__":
    main()
