"""Árbol de decisión, XGBoost y LightGBM con variables de calendario, rezagos y
búsqueda de hiperparámetros con validación cruzada temporal (TimeSeriesSplit)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from utils.features import MIN_HIST, calendar_features, lag_row


def _tabular(make, grid):
    def run(y, idx, H, country=None):
        n = len(y)
        cal = calendar_features(idx.append(pd.date_range(idx[-1] + pd.Timedelta(days=1), periods=H)), country).values
        ly = np.log1p(np.maximum(y, 0))
        X = np.array([lag_row(ly[:i]) + list(cal[i]) for i in range(MIN_HIST, n)])
        search = RandomizedSearchCV(make(), grid, n_iter=8, cv=TimeSeriesSplit(3), random_state=42,
                                    scoring="neg_root_mean_squared_error", n_jobs=-1)
        model = search.fit(X, ly[MIN_HIST:]).best_estimator_
        hist, out = list(ly), []
        for k in range(H):  # pronóstico recursivo
            q = max(0.0, float(model.predict(np.array([lag_row(hist) + list(cal[n + k])]))[0]))
            hist.append(q)
            out.append(np.expm1(q))
        return np.array(out)
    return run


def _dt():
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(random_state=42)


def _xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(random_state=42, n_jobs=1)


def _lgbm():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(random_state=42, n_jobs=1, verbose=-1)


decision_tree = _tabular(_dt, {"max_depth": [4, 6, 8, 10, 12], "min_samples_leaf": [3, 5, 10, 20],
                               "ccp_alpha": [0, 1e-4, 1e-3]})
xgboost = _tabular(_xgb, {"n_estimators": [300, 600], "max_depth": [3, 4, 6], "learning_rate": [0.03, 0.05, 0.1],
                          "subsample": [0.7, 0.9], "colsample_bytree": [0.7, 0.9],
                          "min_child_weight": [1, 3, 5], "reg_lambda": [1, 5]})
lightgbm = _tabular(_lgbm, {"n_estimators": [300, 600], "num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.05, 0.1],
                            "subsample": [0.7, 0.9], "subsample_freq": [1], "colsample_bytree": [0.7, 0.9],
                            "min_child_samples": [10, 20, 40], "reg_lambda": [0, 1, 5]})
