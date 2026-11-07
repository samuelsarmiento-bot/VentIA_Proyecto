"""Modelos estadísticos: baseline, SARIMAX con selección por AIC y Prophet con feriados."""
import numpy as np
import pandas as pd

from utils.features import calendar_features

# (orden, orden estacional semanal) candidatos; se elige el de menor AIC
CANDIDATOS = [((1, 1, 1), (0, 1, 1, 7)), ((2, 1, 1), (0, 1, 1, 7)),
              ((1, 1, 2), (1, 1, 1, 7)), ((2, 1, 2), (0, 1, 1, 7))]


def naive(y, idx, H, country=None):
    return np.array([y[-7 + i % 7] for i in range(H)])


def arima(y, idx, H, country=None):
    """SARIMAX sobre log1p con exógenas (feriados, quincena, armónicos anuales)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    fut = pd.date_range(idx[-1] + pd.Timedelta(days=1), periods=H)
    ex = calendar_features(idx.append(fut), country)[["is_holiday", "is_payday", "doy_sin", "doy_cos"]].values
    ex = ex[:, ex[:len(y)].std(0) > 0]  # quita columnas constantes (p. ej. sin feriados)
    ly, best = np.log1p(np.maximum(y, 0)), None
    for order, sorder in CANDIDATOS:
        try:
            r = SARIMAX(ly, exog=ex[:len(y)], order=order, seasonal_order=sorder,
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=60)
            if best is None or r.aic < best.aic:
                best = r
        except Exception:
            continue
    if best is None:
        raise RuntimeError("Ningún SARIMAX convergió.")
    return np.expm1(np.asarray(best.forecast(H, exog=ex[len(y):])))


def prophet(y, idx, H, country=None):
    """Prophet con estacionalidad mensual, regresor de quincena y feriados del país."""
    from prophet import Prophet
    payday = lambda d: ((d.dt.day == 15) | d.dt.is_month_end).astype(float)
    df = pd.DataFrame({"ds": idx, "y": y})
    df["payday"] = payday(df["ds"])
    m = Prophet(seasonality_mode="multiplicative" if y.min() > 0 else "additive",
                changepoint_prior_scale=0.1, weekly_seasonality=True,
                yearly_seasonality=len(y) >= 400, daily_seasonality=False)
    m.add_seasonality("mensual", period=30.5, fourier_order=5)
    m.add_regressor("payday")
    if country and country != "Ninguno":
        m.add_country_holidays(country_name=country)
    m.fit(df)
    f = m.make_future_dataframe(H)
    f["payday"] = payday(f["ds"])
    return m.predict(f)["yhat"].values[-H:]
