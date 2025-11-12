# htf/model.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Iterable, List, Optional

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
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


# ========= 1) DynamicScaler: chọn cột cần scale tự động =========
class DynamicScaler(BaseEstimator, TransformerMixin):
    """
    - Fit: phát hiện cột nhị phân & sin/cos, chỉ scale các cột số còn lại.
    - Transform: chuẩn hoá các cột đó, giữ nguyên thứ tự & dtype DataFrame.
    """
    def __init__(self):
        self.scale_cols_: List[str] = []
        self.pass_cols_: List[str] = []
        self._scaler: Optional[StandardScaler] = None

    def _detect_binary_cols(self, X: pd.DataFrame) -> List[str]:
        bin_cols = []
        for c in X.select_dtypes(include=[np.number]).columns:
            vals = pd.Series(X[c].dropna().unique())
            if len(vals) > 0 and set(vals.astype(float)) <= {0.0, 1.0}:
                bin_cols.append(c)
        return bin_cols

    def _detect_cyc_cols(self, X: pd.DataFrame) -> List[str]:
        return [c for c in X.columns if c.startswith("sin_") or c.startswith("cos_")]

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        bin_cols = self._detect_binary_cols(X)
        cyc_cols = self._detect_cyc_cols(X)
        self.pass_cols_  = sorted(set(bin_cols + cyc_cols))
        self.scale_cols_ = sorted(list(set(num_cols) - set(self.pass_cols_)))
        self._scaler = StandardScaler()
        if self.scale_cols_:
            self._scaler.fit(X[self.scale_cols_].astype("float64"))
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        if self.scale_cols_:
            scaled = self._scaler.transform(X[self.scale_cols_].astype("float64"))
            for j, col in enumerate(self.scale_cols_):
                X[col] = scaled[:, j]
        # đảm bảo trả về đúng DataFrame với thứ tự cột ổn định
        return X[X.columns.sort_values()]


# ========= 2) Ứng viên mô hình =========
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



# ========= 3) Train KHÔNG CÓ VAL (chọn model bằng walk-forward CV trên TRAIN) =========
def time_series_cv_mae(X: pd.DataFrame, y: pd.Series, model, n_splits: int = 3, gap: int = 0) -> float:
    tss = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    maes = []
    for tr_idx, va_idx in tss.split(X):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        pipe = Pipeline([("scale", DynamicScaler()), ("model", clone(model))])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)
        maes.append(mean_absolute_error(y_va, pred))
    return float(np.mean(maes)) if maes else np.inf

def train_direct_models_train_test(datasets: dict, random_state=42, verbose=True, cv_splits=3, cv_gap=0):
    """
    datasets[h] phải có:
      - 'train': (X_train, y_train)
      - 'test' : (X_test,  y_test)
    """
    results = {}
    for h, parts in datasets.items():
        X_train, y_train = parts["train"]
        X_test,  y_test  = parts["test"]

        # chọn model bằng CV trên train
        best_name, best_cv, best_model = None, np.inf, None
        for name, model in build_model_candidates(random_state).items():
            cv_mae = time_series_cv_mae(X_train, y_train, model, n_splits=cv_splits, gap=cv_gap)
            if verbose:
                print(f"[H{h}] {name:6s}  CV MAE={cv_mae:.4f}")
            if cv_mae < best_cv:
                best_name, best_cv, best_model = name, cv_mae, model

        if verbose:
            print(f"[H{h}] -> Best by CV: {best_name} (CV MAE={best_cv:.4f})")

        # refit toàn bộ train, chấm test
        final_pipe = Pipeline([("scale", DynamicScaler()), ("model", clone(best_model))])
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
