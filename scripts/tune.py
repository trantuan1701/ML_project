# scripts/tune.py
import os, json
from datetime import datetime
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

# dùng parquet
DATA_PATH = "data/weather.parquet"
TRAIN_FRAC = 0.80
GAP_LMAX   = 14
HORIZONS   = (1,2,3,4,5)
CV_SPLITS  = 3
N_TRIALS   = 50
OPTUNA_STORAGE = "sqlite:///optuna.db"
STUDY_PREFIX   = "hcm-daily"
SAVE_JSON      = "tune_results.json"
RETRAIN_BEST   = True
ARTIFACT_DIR   = "artifacts"

from HCM_temp_forcast.data import load_from_parquet
from HCM_temp_forcast.prepare import basic_clean, time_split_train_test, make_direct_datasets_train_test
from HCM_temp_forcast.tune import tune_all_horizons
from HCM_temp_forcast.model import DynamicScaler, build_model_candidates

def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # 1) Load parquet + clean
    df = load_from_parquet(DATA_PATH)
    df = basic_clean(df, drop_text_cols=True, drop_na_core=True)

    # 2) Split (GAP = L_max)
    train_df, test_df, _ = time_split_train_test(df, train_frac=0.80, gap=GAP_LMAX)

    # 3) Datasets
    datasets = make_direct_datasets_train_test(
        train_df, test_df,
        horizons=HORIZONS,
        lags=(1,2,3,7,GAP_LMAX),
        roll_windows=(7,GAP_LMAX),
    )

    # 4) Tune (CV gap = L_max + max(H) - 1)
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
            best_name = results[h]["_best_overall"]["model"]
            best_params = results[h][best_name]["best_params"]

            est = clone(build_model_candidates(42)[best_name])
            est.set_params(**{k: v for k, v in best_params.items() if k in est.get_params()})

            pipe = Pipeline([("scale", DynamicScaler()), ("model", est)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            mae  = mean_absolute_error(y_test, pred)
            rmse = mean_squared_error(y_test, pred) ** 0.5
            print(f"[H{h}] TEST -> MAE={mae:.4f}  RMSE={rmse:.4f}")

            try:
                import joblib
                path = os.path.join(ARTIFACT_DIR, f"pipeline_H{h}_{best_name}.joblib")
                joblib.dump({"pipeline": pipe, "feature_cols": datasets[h]["feature_cols"]}, path)
                print(f"    saved: {path}")
            except Exception as e:
                print(f"    (skip save) {e}")

if __name__ == "__main__":
    main()
