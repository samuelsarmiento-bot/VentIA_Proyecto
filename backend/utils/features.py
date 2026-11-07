"""Ingeniería de variables: calendario, feriados y rezagos."""
import numpy as np
import pandas as pd

COV_COLS = ["dow_sin", "dow_cos", "mon_sin", "mon_cos", "doy_sin", "doy_cos",
            "is_weekend", "is_payday", "is_holiday"]
LAGS = (1, 2, 3, 7, 14, 21, 28)
MIN_HIST = 28


def holiday_dates(idx, country):
    if not country or country == "Ninguno":
        return []
    try:
        import holidays
        years = range(idx.min().year, idx.max().year + 1)
        return list(pd.to_datetime(list(holidays.country_holidays(country, years=years).keys())))
    except Exception:
        return []


def calendar_features(idx, country=None) -> pd.DataFrame:
    """Covariables conocidas de antemano (valen también para fechas futuras)."""
    idx = pd.DatetimeIndex(idx)
    c = pd.DataFrame(index=idx)
    c["dow_sin"], c["dow_cos"] = np.sin(2 * np.pi * idx.dayofweek / 7), np.cos(2 * np.pi * idx.dayofweek / 7)
    c["mon_sin"], c["mon_cos"] = np.sin(2 * np.pi * idx.month / 12), np.cos(2 * np.pi * idx.month / 12)
    c["doy_sin"], c["doy_cos"] = np.sin(2 * np.pi * idx.dayofyear / 365.25), np.cos(2 * np.pi * idx.dayofyear / 365.25)
    c["is_weekend"] = (idx.dayofweek >= 5).astype(float)
    c["is_payday"] = ((idx.day == 15) | idx.is_month_end).astype(float)
    c["is_holiday"] = idx.isin(holiday_dates(idx, country)).astype(float)
    c["dow"], c["dom"], c["month"] = idx.dayofweek, idx.day, idx.month
    c["woy"] = idx.isocalendar().week.to_numpy().astype(float)
    return c


def lag_row(h):
    """Rezagos y estadísticos móviles de la historia (en escala log1p)."""
    h = np.asarray(h)
    r = [h[-l] for l in LAGS]
    r += [h[-7:].mean(), h[-14:].mean(), h[-28:].mean(), h[-7:].std(), h[-28:].std(),
          h[-7:].mean() - h[-28:].mean(), h[-1] - h[-8]]
    return r
