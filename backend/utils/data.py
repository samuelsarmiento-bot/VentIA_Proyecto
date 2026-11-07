"""Lectura de datos: CSV/ZIP -> serie diaria agregada."""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def read_df(raw: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                raise ValueError("El ZIP no contiene archivos CSV.")
            raw = z.read(next((n for n in csvs if Path(n).name == "train.csv"), csvs[0]))
    return pd.read_csv(io.BytesIO(raw))


def to_series(df: pd.DataFrame) -> pd.Series:
    """Suma las ventas por día (varias tiendas/productos -> una serie) y rellena días faltantes con 0."""
    cols = {c.lower(): c for c in df.columns}
    dcol = cols.get("date") or next((c for c in df.columns if "fecha" in c.lower()), df.columns[0])
    num = [c for c in df.select_dtypes("number").columns if c.lower() not in {"id", "store_nbr", "onpromotion"}]
    if not num:
        raise ValueError("No se encontró una columna numérica para pronosticar.")
    vcol = cols["sales"] if cols.get("sales") in num else num[-1]
    d = pd.DataFrame({"date": pd.to_datetime(df[dcol], errors="coerce"), "v": df[vcol]}).dropna(subset=["date"])
    s = d.groupby("date")["v"].sum().sort_index().asfreq("D").fillna(0)
    if len(s) < 120:
        raise ValueError("La serie necesita al menos 120 días de datos.")
    return s.astype(float)


def demo_series() -> pd.Series:
    t = np.arange(1095)
    r = np.random.default_rng(42)
    v = (1000 + 0.8 * t + 150 * np.sin(2 * np.pi * t / 7) + 200 * np.sin(2 * np.pi * t / 365.25)
         + 120 * ((pd.date_range("2015-01-01", periods=len(t)).day == 15)) + r.normal(0, 40, len(t)))
    return pd.Series(np.maximum(v, 0), index=pd.date_range("2015-01-01", periods=len(t)))


def load_default():
    """Primer ZIP/CSV válido de datasets/; si no hay, datos sintéticos de ejemplo."""
    folder = Path(__file__).resolve().parents[2] / "datasets"
    for p in sorted(folder.glob("*.zip")) + sorted(folder.glob("*.csv")):
        try:
            return to_series(read_df(p.read_bytes(), p.name)), p.name
        except Exception:
            continue
    return demo_series(), "Datos de ejemplo (sintéticos)"


def summary(s: pd.Series, name: str) -> dict:
    return {"name": name, "start": str(s.index[0].date()), "end": str(s.index[-1].date()),
            "days": len(s), "total": float(s.sum()),
            "dates": [str(d.date()) for d in s.index], "values": [round(float(v), 2) for v in s.values]}
