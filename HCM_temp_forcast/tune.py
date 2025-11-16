# HCM_temp_forcast/tune.py
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd
import optuna

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

from .model import build_model_candidates


# =========================
# 1) Pipeline helper
# =========================
def _build_pipeline_with_kbest(
    estimator,
    model_name: str,
    k_best: int,
) -> Pipeline:
    """
    - Linear models: StandardScaler -> SelectKBest -> model
    - Tree/boosting models: SelectKBest -> model (không scale)
    """
    linear_models = {"linreg", "ridge", "lasso"}

    steps = []
    if model_name in linear_models:
        steps.append(("scale", StandardScaler()))
        steps.append(("select", SelectKBest(score_func=f_regression, k=k_best)))
        steps.append(("model", clone(estimator)))
    else:
        # rf / gbr / xgb / lgbm: không cần scale
        steps.append(("select", SelectKBest(score_func=f_regression, k=k_best)))
        steps.append(("model", clone(estimator)))

    return Pipeline(steps)


# =========================
# 2) Walk-forward CV (MAE)
# =========================
def _cv_mae_for_estimator(
    X: pd.DataFrame,
    y: pd.Series,
    estimator,
    model_name: str,
    k_best: int,
    n_splits: int = 3,
    gap: int = 0,
) -> float:
    """
    TimeSeriesSplit CV; pipeline: (scale nếu cần) -> SelectKBest(k_best) -> model.
    KHÔNG impute: nếu X có NaN, các estimator không hỗ trợ NaN sẽ lỗi.
    """
    tss = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    maes = []
    for tr_idx, va_idx in tss.split(X):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        pipe = _build_pipeline_with_kbest(
            estimator=estimator,
            model_name=model_name,
            k_best=k_best,
        )
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)
        maes.append(mean_absolute_error(y_va, pred))

    return float(np.mean(maes)) if maes else np.inf


# =========================================
# 3) Build estimator theo trial (search space)
# =========================================
def _build_estimator_from_trial(
    model_name: str,
    trial: optuna.Trial,
    random_state: int = 42,
):
    base = build_model_candidates(random_state=random_state).get(model_name)
    if base is None:
        raise ValueError(f"Model '{model_name}' chưa có trong build_model_candidates().")

    est = clone(base)
    params = {}

    if model_name == "linreg":
        # không có hyperparam quan trọng
        pass

    elif model_name == "ridge":
        params["alpha"] = trial.suggest_float("alpha", 1e-3, 10.0, log=True)

    elif model_name == "lasso":
        params["alpha"] = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        params["max_iter"] = trial.suggest_int("max_iter", 5000, 30000, step=5000)

    elif model_name == "rf":
        params["n_estimators"]      = trial.suggest_int("n_estimators", 200, 1200, step=100)
        params["max_depth"]         = trial.suggest_int("max_depth", 3, 20)
        params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
        params["min_samples_leaf"]  = trial.suggest_int("min_samples_leaf", 1, 10)

    elif model_name == "gbr":
        params["n_estimators"]      = trial.suggest_int("n_estimators", 200, 1500, step=100)
        params["learning_rate"]     = trial.suggest_float("learning_rate", 0.005, 0.2, log=True)
        params["max_depth"]         = trial.suggest_int("max_depth", 2, 8)
        params["subsample"]         = trial.suggest_float("subsample", 0.5, 1.0)
        params["min_samples_leaf"]  = trial.suggest_int("min_samples_leaf", 1, 20)

    elif model_name == "xgb":
        params["n_estimators"]      = trial.suggest_int("n_estimators", 300, 2000, step=100)
        params["learning_rate"]     = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
        params["max_depth"]         = trial.suggest_int("max_depth", 3, 12)
        params["subsample"]         = trial.suggest_float("subsample", 0.5, 1.0)
        params["colsample_bytree"]  = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        params["reg_alpha"]         = trial.suggest_float("reg_alpha", 1e-8, 1e-1, log=True)
        params["reg_lambda"]        = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)

    elif model_name == "lgbm":
        params["n_estimators"]      = trial.suggest_int("n_estimators", 500, 3000, step=100)
        params["learning_rate"]     = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True)
        params["num_leaves"]        = trial.suggest_int("num_leaves", 15, 255)
        params["max_depth"]         = trial.suggest_int("max_depth", -1, 16)
        params["subsample"]         = trial.suggest_float("subsample", 0.5, 1.0)
        params["colsample_bytree"]  = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        params["reg_alpha"]         = trial.suggest_float("reg_alpha", 1e-8, 1e-1, log=True)
        params["reg_lambda"]        = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)

    valid = est.get_params()
    for k, v in params.items():
        if k in valid:
            est.set_params(**{k: v})
    return est


# =======================================================
# 4) Objective Optuna: MAE (walk-forward CV trên train)
# =======================================================
def _make_objective_for_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    random_state: int = 42,
    n_splits: int = 3,
    gap: int = 0,
):
    # Số feature hiện có
    n_features = X_train.shape[1]

    # Các tỉ lệ k tương đối (10%, 15%, 20%, 25%, 33%, 50%, 75%, 100%)
    ratio_grid = [0.10, 0.15, 0.20, 0.25, 0.33, 0.50, 0.75, 1.0]
    candidate_ks = {
        max(8, min(n_features, int(round(n_features * r))))
        for r in ratio_grid
    }
    candidate_ks.add(n_features)  # chắc chắn có full feature
    candidate_ks = sorted(candidate_ks)

    # Ví dụ: n_features=165 -> [16, 25, 33, 41, 54, 82, 124, 165]

    def objective(trial: optuna.Trial) -> float:
        # chọn k từ tập rải đều
        k_best = trial.suggest_categorical("k_best", candidate_ks)

        # build estimator với hyperparams riêng
        est = _build_estimator_from_trial(model_name, trial, random_state=random_state)

        mae = _cv_mae_for_estimator(
            X_train,
            y_train,
            estimator=est,
            model_name=model_name,
            k_best=k_best,
            n_splits=n_splits,
            gap=gap,
        )
        trial.set_user_attr("cv_mae", mae)
        return mae

    return objective


# ===================================================
# 5) Tune CHO 1 horizon (chạy TẤT CẢ model)
# ===================================================
def tune_all_models_for_horizon(
    datasets_for_h: dict,
    *,
    random_state: int = 42,
    n_trials: int = 50,
    n_splits: int = 3,
    gap: int = 0,
    storage: Optional[str] = None,          # ví dụ "sqlite:///optuna.db"
    study_prefix: str = "tune",
) -> Dict[str, Dict]:
    """
    Input (cho 1 horizon):
      datasets_for_h = {'train': (X_train, y_train), 'test': (X_test, y_test), 'feature_cols': [...]}
    Output:
      {
        model_name: {'best_value': cv_mae, 'best_params': {...}, 'study_name': ...},
        ...
        '_best_overall': {'model': name, 'cv_mae': value}
      }
    """
    X_train, y_train = datasets_for_h["train"]
    results: Dict[str, Dict] = {}

    for model_name in build_model_candidates(random_state).keys():
        sampler = optuna.samplers.TPESampler(
            consider_prior=True,
            multivariate=True,
            group=False,
            seed=random_state,
        )
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10)

        study_name = f"{study_prefix}-{model_name}"
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )

        obj = _make_objective_for_model(
            X_train, y_train,
            model_name=model_name,
            random_state=random_state,
            n_splits=n_splits,
            gap=gap,
        )
        study.optimize(obj, n_trials=n_trials)

        results[model_name] = {
            "best_value": float(study.best_value),
            "best_params": dict(study.best_params),
            "study_name": study_name,
        }

    best_model = min(results.items(), key=lambda kv: kv[1]["best_value"])[0]
    results["_best_overall"] = {
        "model": best_model,
        "cv_mae": results[best_model]["best_value"]
    }
    return results


# ===================================================
# 6) Tune CHO NHIỀU horizon
# ===================================================
def tune_all_horizons(
    datasets: Dict[int, dict],
    *,
    random_state: int = 42,
    n_trials: int = 50,
    n_splits: int = 3,
    gap: int = 0,
    storage: Optional[str] = None,
    study_prefix: str = "tune",
) -> Dict[int, Dict]:
    """
    datasets: {h: {'train': (X,y), 'test': (X,y), 'feature_cols': [...]}}
    """
    out: Dict[int, Dict] = {}
    for h, parts in datasets.items():
        res = tune_all_models_for_horizon(
            parts,
            random_state=random_state,
            n_trials=n_trials,
            n_splits=n_splits,
            gap=gap,
            storage=storage,
            study_prefix=f"{study_prefix}-H{h}",
        )
        out[h] = res
    return out
