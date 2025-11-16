# scripts/tune.py
import os, json
from datetime import datetime

import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# dùng parquet
DATA_PATH = "data/weather.parquet"
TRAIN_FRAC = 0.80
GAP_LMAX   = 30
HORIZONS   = (1, 2, 3, 4, 5)
CV_SPLITS  = 3
N_TRIALS   = 50
OPTUNA_STORAGE = "sqlite:///optuna.db"
STUDY_PREFIX   = "hcm-daily-final-v12-114-features"
SAVE_JSON      = "tune_results_114.json"
RETRAIN_BEST   = True
ARTIFACT_DIR   = "artifacts_114"

from HCM_temp_forcast.data import load_from_parquet
from HCM_temp_forcast.prepare import basic_clean, time_split_train_test, make_direct_datasets_train_test
from HCM_temp_forcast.tune import tune_all_horizons
from HCM_temp_forcast.model import build_model_candidates


def _build_final_pipeline(best_name: str, est, k_best: int, n_features: int) -> Pipeline:
    """
    Dùng cùng style với tune.py:
      - Linear models: StandardScaler -> SelectKBest -> model
      - Tree/boosting models: SelectKBest -> model
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
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # 1) Load parquet + clean
    df = load_from_parquet(DATA_PATH)
    df = basic_clean(df, drop_text_cols=True, drop_na_core=True)

    # 2) Split (GAP = khoảng trống thời gian giữa train/test)
    train_df, test_df, _ = time_split_train_test(
        df,
        train_frac=TRAIN_FRAC,
        gap=GAP_LMAX,
        datetime_col="datetime",
    )

    # 3) Datasets cho direct-forecast (FE đã cố định spec lags/rolling bên trong)
    datasets = make_direct_datasets_train_test(
        train_df,
        test_df,
        horizons=HORIZONS,
    )

    # 4) Tune (CV gap = GAP_LMAX + max(H) - 1) – tránh leakage giữa fold
    cv_gap = GAP_LMAX + max(HORIZONS) - 1
    results = tune_all_horizons(
        datasets,
        random_state=42,
        n_trials=N_TRIALS,
        n_splits=CV_SPLITS,
        gap=cv_gap,
        storage=OPTUNA_STORAGE,
        study_prefix=STUDY_PREFIX,
    )

    with open(SAVE_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved tuning results -> {SAVE_JSON} @ {datetime.now().isoformat()}")

    # 5) (optional) retrain best + test + save artifact
    if RETRAIN_BEST:
        for h in sorted(datasets):
            X_train, y_train = datasets[h]["train"]
            X_test,  y_test  = datasets[h]["test"]

            best_name   = results[h]["_best_overall"]["model"]
            best_params = results[h][best_name]["best_params"]

            # tách k_best ra khỏi best_params
            k_best = best_params.get("k_best", X_train.shape[1])

            # build estimator với hyperparam model
            est = clone(build_model_candidates(42)[best_name])
            valid_params = est.get_params()
            est.set_params(**{k: v for k, v in best_params.items() if k in valid_params})

            # pipeline cuối cùng: giống logic trong tune.py
            pipe = _build_final_pipeline(
                best_name=best_name,
                est=est,
                k_best=k_best,
                n_features=X_train.shape[1],
            )

            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            mae  = mean_absolute_error(y_test, pred)
            rmse = mean_squared_error(y_test, pred) ** 0.5
            r2   = r2_score(y_test, pred)

            print(
                f"[H{h}] BEST={best_name}  k_best={k_best}  "
                f"TEST -> MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}"
            )

            # save artifact (pipeline + feature_cols)
            try:
                import joblib
                path = os.path.join(ARTIFACT_DIR, f"pipeline_H{h}_{best_name}.joblib")
                joblib.dump(
                    {
                        "pipeline": pipe,
                        "feature_cols": datasets[h]["feature_cols"],
                    },
                    path,
                )
                print(f"    saved: {path}")
            except Exception as e:
                print(f"    (skip save) {e}")


if __name__ == "__main__":
    main()
 