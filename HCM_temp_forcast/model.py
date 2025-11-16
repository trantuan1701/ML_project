# HCM_temp_forcast/model.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Iterable, List, Optional

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None


# ========= 1) Ứng viên mô hình =========
def build_model_candidates(random_state: int = 42) -> Dict[str, object]:
    models = {
        "linreg": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=random_state) if "random_state" in Ridge().get_params() else Ridge(alpha=1.0),
        "lasso": Lasso(alpha=1e-3, max_iter=10000, random_state=random_state) if "random_state" in Lasso().get_params() else Lasso(alpha=1e-3, max_iter=10000),
        "rf": RandomForestRegressor(n_estimators=500, max_depth=None, random_state=random_state, n_jobs=-1),
        "gbr": GradientBoostingRegressor(random_state=random_state),
    }
    if XGBRegressor is not None:
        models["xgb"] = XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, n_jobs=-1, tree_method="hist"
        )
    if LGBMRegressor is not None:
        models["lgbm"] = LGBMRegressor(
            n_estimators=1200, num_leaves=63, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, n_jobs=-1
        )
    return models


# ========= 2) Helper: nhận diện mô hình cây & build pipeline =========
def _is_tree_model(model) -> bool:
    tree_like = [RandomForestRegressor, GradientBoostingRegressor]
    if XGBRegressor is not None:
        tree_like.append(XGBRegressor)
    if LGBMRegressor is not None:
        tree_like.append(LGBMRegressor)
    return isinstance(model, tuple(tree_like))


def _make_pipeline_for_model(
    model,
    *,
    k_features: Optional[int] = None,
) -> Pipeline:
    """
    Pipeline chung cho tất cả model:
      - Linear / non-tree: StandardScaler -> SelectKBest -> model
      - Tree-based:        SelectKBest -> model
    k_features:
      - None  -> k="all"
      - int   -> dùng k đó cho SelectKBest
    """
    steps = []

    # Linear / distance-based model: scale trước
    if not _is_tree_model(model):
        steps.append(("scale", StandardScaler()))

    # Feature selection
    k = "all" if k_features is None else int(k_features)
    steps.append(("select", SelectKBest(score_func=f_regression, k=k)))

    # Model
    steps.append(("model", clone(model)))

    return Pipeline(steps)


# ========= 3) Train  =========
def time_series_cv_mae(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    n_splits: int = 3,
    gap: int = 0,
    *,
    k_features: Optional[int] = None,
) -> float:
    """
    Walk-forward CV (TimeSeriesSplit) với MAE.
    Dùng pipeline: (scale nếu cần) -> SelectKBest(k) -> model.
    """
    tss = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    maes = []
    for tr_idx, va_idx in tss.split(X):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        pipe = _make_pipeline_for_model(model, k_features=k_features)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)
        maes.append(mean_absolute_error(y_va, pred))
    return float(np.mean(maes)) if maes else np.inf


def train_direct_models_train_test(
    datasets: dict,
    *,
    random_state: int = 42,
    verbose: bool = True,
    cv_splits: int = 3,
    cv_gap: int = 0,
    k_features: Optional[int] = None,
):
    """
    datasets[h] phải có:
      - 'train': (X_train, y_train)
      - 'test' : (X_test,  y_test)

    k_features:
      - None  -> SelectKBest(k="all") (không giảm chiều)
      - int   -> chọn top-k feature theo f_regression
    """
    results = {}
    for h, parts in datasets.items():
        X_train, y_train = parts["train"]
        X_test,  y_test  = parts["test"]

        # chọn model bằng CV trên train
        best_name, best_cv, best_model = None, np.inf, None
        for name, model in build_model_candidates(random_state).items():
            cv_mae = time_series_cv_mae(
                X_train, y_train,
                model,
                n_splits=cv_splits,
                gap=cv_gap,
                k_features=k_features,
            )
            if verbose:
                print(f"[H{h}] {name:6s}  CV MAE={cv_mae:.4f}")
            if cv_mae < best_cv:
                best_name, best_cv, best_model = name, cv_mae, model

        if verbose:
            print(f"[H{h}] -> Best by CV: {best_name} (CV MAE={best_cv:.4f})")

        # refit toàn bộ train, chấm test
        final_pipe = _make_pipeline_for_model(best_model, k_features=k_features)
        final_pipe.fit(X_train, y_train)
        pred_test = final_pipe.predict(X_test)
        mae_test  = float(mean_absolute_error(y_test, pred_test))
        rmse_test = float(np.sqrt(mean_squared_error(y_test, pred_test)))

        if verbose:
            print(f"[H{h}] Test MAE={mae_test:.4f}  RMSE={rmse_test:.4f}")
            print("-" * 60)

        results[h] = {
            "best_name": best_name,
            "cv_MAE": float(best_cv),
            "test_MAE": mae_test,
            "test_RMSE": rmse_test,
            "pipeline": final_pipe,
            "feature_cols": parts["feature_cols"],
        }
    return results
