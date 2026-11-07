import numpy as np


def metrics(a, p):
    """MAE, RMSE, RMSLE y MAPE (%) entre valores reales y pronosticados."""
    a, p = np.maximum(0, np.asarray(a, float)), np.maximum(0, np.asarray(p, float))
    nz = a != 0
    return {"MAE": float(np.mean(np.abs(a - p))),
            "RMSE": float(np.sqrt(np.mean((a - p) ** 2))),
            "RMSLE": float(np.sqrt(np.mean((np.log1p(a) - np.log1p(p)) ** 2))),
            "MAPE": float(np.mean(np.abs((a[nz] - p[nz]) / a[nz])) * 100) if nz.any() else None}
