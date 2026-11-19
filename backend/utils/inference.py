import time
 
import numpy as np
import pandas as pd
 
from models.registry import MODELS, available, get_model
from . import progress
from .metrics import metrics
 
 
def run_forecast(series, model_ids, H, mode="backtest", country=None):
    """backtest: entrena sin los últimos H días y compara con lo real.
    future: entrena con todo y proyecta H días hacia adelante."""
    back = mode == "backtest"
    train = series.iloc[:-H] if back else series
    if len(train) < 120:
        raise ValueError("Se necesitan al menos 120 días de entrenamiento; reduce el horizonte.")
    y, idx = train.values.astype(float), train.index
    dates = series.index[-H:] if back else pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=H)
    ids = [m for m in model_ids if m in MODELS]
    out = []
    for i, mid in enumerate(ids):
        item = {"id": mid, "name": MODELS[mid]["name"], "pred": None, "metrics": None, "error": None}
        progress.set_span(i / len(ids), (i + 1) / len(ids), f"Entrenando {MODELS[mid]['name']}…")
        t0 = time.time()
        try:
            if not available(mid):
                raise RuntimeError(f"Librería no instalada ({MODELS[mid]['needs']}).")
            p = np.nan_to_num(np.maximum(0, np.asarray(get_model(mid)(y, idx, H, country), float)))
            item["pred"] = [round(float(v), 2) for v in p]
            if back:
                item["metrics"] = metrics(series.values[-H:], p)
        except Exception as e:
            item["error"] = f"{type(e).__name__}: {e}"
        item["seconds"] = round(time.time() - t0, 1)
        progress.report(1.0)
        out.append(item)
    return {"mode": mode, "horizon": H, "dates": [str(d.date()) for d in dates], "results": out}