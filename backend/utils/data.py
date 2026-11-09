"""Lectura de datos: CSV / Excel / ZIP -> serie diaria agregada.
 
Incluye detección de columnas (fecha y ventas), limpieza de números con símbolos
y validaciones que devuelven mensajes claros para el usuario.
"""
import io
import re
import warnings
import zipfile
from pathlib import Path
 
import numpy as np
import pandas as pd
 
MIN_DIAS = 120
NOMBRES_FECHA = ("date", "fecha", "dia", "día", "day", "ds", "periodo")
NOMBRES_VENTAS = ("sales", "ventas", "venta", "total", "valor", "monto", "ingresos", "revenue", "cantidad", "units")
CODIGOS = {"id", "store", "store_nbr", "item", "onpromotion"}  # columnas numéricas que son códigos, no ventas
 
 
# --------------------------------------------------------------------------- lectura
def _leer_csv(raw: bytes) -> pd.DataFrame:
    lineas = raw[:4000].decode("utf-8", errors="ignore").splitlines()
    primera = lineas[0] if lineas else ""
    sep = max([",", ";", "\t", "|"], key=primera.count)  # separador más frecuente en el encabezado
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise ValueError("El archivo está vacío.")
        except pd.errors.ParserError:
            raise ValueError("No pude leer el archivo: parece un CSV mal formado. "
                             "Ábrelo en Excel y guárdalo de nuevo como CSV.")
    raise ValueError("No pude leer el archivo por su codificación de texto. Guárdalo de nuevo como CSV UTF-8.")
 
 
def read_df(raw: bytes, filename: str) -> pd.DataFrame:
    nombre = (filename or "").lower()
    if nombre.endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not csvs:
                    raise ValueError("El ZIP no contiene archivos CSV.")
                raw = z.read(next((n for n in csvs if Path(n).name == "train.csv"), csvs[0]))
        except zipfile.BadZipFile:
            raise ValueError("El archivo ZIP está dañado o no es un ZIP válido.")
        df = _leer_csv(raw)
    elif nombre.endswith(".xlsx"):
        try:
            df = pd.read_excel(io.BytesIO(raw))
        except Exception:
            raise ValueError("No pude leer el Excel. Comprueba que no esté dañado ni protegido con contraseña.")
    elif nombre.endswith((".csv", ".txt")):
        df = _leer_csv(raw)
    else:
        raise ValueError("Formato no admitido. Sube un archivo CSV, Excel (.xlsx) o ZIP con un CSV.")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    if df.empty or len(df.columns) < 2:
        raise ValueError("El archivo necesita al menos dos columnas (una de fechas y una de ventas) y algunas filas.")
    return df
 
 
# --------------------------------------------------------------------------- conversión de tipos
def _es_numerica(col: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)
 
 
def _a_numero(x) -> float:
    """Convierte textos como '$1.250.000', '1,250.50' o '12,5' en número (NaN si no se puede)."""
    t = re.sub(r"(?i)\b(cop|usd|eur|mxn)\b|[$€\s]", "", str(x))
    if re.search(r"[A-Za-z/:]", t):
        return np.nan
    t = re.sub(r"[^\d,.\-]", "", t)
    if not re.search(r"\d", t) or "-" in t[1:]:
        return np.nan
    neg, t = t.startswith("-"), t.replace("-", "")
    if "," in t and "." in t:  # el último símbolo es el decimal
        t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
    elif "," in t:
        p = t.split(",")
        t = p[0] + "." + p[1] if (len(p) == 2 and len(p[1]) != 3) else t.replace(",", "")
    elif "." in t:
        p = t.split(".")
        if len(p) > 2 or (len(p[1]) == 3 and p[0] not in ("", "0")):  # '1.250' se lee como mil doscientos cincuenta
            t = t.replace(".", "")
    try:
        v = float(t)
    except ValueError:
        return np.nan
    return -v if neg else v
 
 
def _a_numeros(col: pd.Series) -> pd.Series:
    return col.astype(float) if _es_numerica(col) else col.map(_a_numero).astype(float)
 
 
_PATRON_DMY = re.compile(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})")
 
 
def _dayfirst(col: pd.Series) -> bool:
    """Decide si '03/04/2024' es día/mes (lo usual en Latinoamérica) o mes/día."""
    ms = [_PATRON_DMY.match(v) for v in col.dropna().astype(str).head(500)]
    ms = [m for m in ms if m]
    if not ms:
        return False  # formato ISO (2024-04-03) u otro: no importa
    if max(int(m.group(1)) for m in ms) > 12:
        return True
    if max(int(m.group(2)) for m in ms) > 12:
        return False
    return True
 
 
def _a_fecha(col: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(col):
        return col
    if _es_numerica(col) or pd.api.types.is_bool_dtype(col):
        return pd.Series(pd.NaT, index=col.index)
    dayfirst = _dayfirst(col)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = pd.to_datetime(col, errors="coerce", dayfirst=dayfirst)
        if r.isna().mean() > 0.2:  # formatos mezclados: se interpreta cada valor por separado
            r2 = pd.to_datetime(col, errors="coerce", dayfirst=dayfirst, format="mixed")
            if r2.isna().sum() < r.isna().sum():
                r = r2
    return r
 
 
def _es_fecha(col: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(col):
        return True
    if _es_numerica(col) or pd.api.types.is_bool_dtype(col):
        return False
    muestra = col.dropna().head(300)
    return len(muestra) > 0 and _a_fecha(muestra).notna().mean() >= 0.8
 
 
def _es_numero(col: pd.Series) -> bool:
    if _es_numerica(col):
        return True
    muestra = col.dropna().head(300)
    return len(muestra) > 0 and _a_numeros(muestra).notna().mean() >= 0.8
 
 
# --------------------------------------------------------------------------- detección de columnas
def detectar_columnas(df: pd.DataFrame) -> dict:
    """Propone qué columnas son la fecha y las ventas, y lista las alternativas posibles."""
    fechas = [c for c in df.columns if _es_fecha(df[c])]
    numeros = [c for c in df.columns if c not in fechas and _es_numero(df[c])]
 
    sug_fecha = next((c for c in fechas if c.lower() in NOMBRES_FECHA), fechas[0] if fechas else None)
 
    def puntaje(c):  # menor = mejor candidata a ventas
        n = c.lower()
        if n in NOMBRES_VENTAS:
            return 0
        if any(k in n for k in NOMBRES_VENTAS):
            return 1
        return 3 if n in CODIGOS or n.endswith("id") or n.startswith("id") else 2
 
    sug_valor = None
    if numeros:
        mejor = min(puntaje(c) for c in numeros)
        sug_valor = [c for c in numeros if puntaje(c) == mejor][-1]  # si hay empate, la última columna
    return {"date_candidates": fechas, "value_candidates": numeros,
            "suggest_date": sug_fecha, "suggest_value": sug_valor}
 
 
def vista_previa(df: pd.DataFrame, nombre: str) -> dict:
    info = detectar_columnas(df)
    info.update(name=nombre, rows=int(len(df)), columns=[str(c) for c in df.columns],
                sample=df.head(5).astype(str).to_dict("records"))
    return info
 
 
# --------------------------------------------------------------------------- serie diaria
def to_series(df: pd.DataFrame, date_col=None, value_col=None, relleno="cero"):
    """Devuelve (serie diaria, lista de avisos). Si algo está mal lanza ValueError con un mensaje claro.
 
    Suma las ventas por día (varias tiendas/productos -> una serie). Los días sin datos se rellenan
    con 0 (relleno='cero') o interpolando entre los días vecinos (relleno='interpolar').
    """
    avisos = []
    sug = detectar_columnas(df)
    date_col = date_col or sug["suggest_date"]
    value_col = value_col or sug["suggest_value"]
    if date_col is None:
        raise ValueError("No encontré una columna de fechas. El archivo debe tener una columna con la fecha de cada venta.")
    if value_col is None:
        raise ValueError("No encontré una columna numérica con las ventas. Revisa que los valores sean números.")
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError("Una de las columnas elegidas no existe en el archivo.")
    if date_col == value_col:
        raise ValueError("La columna de fechas y la de ventas no pueden ser la misma.")
 
    fechas = _a_fecha(df[date_col])
    ilegibles = int(fechas.isna().sum())
    if ilegibles / len(df) > 0.2:
        raise ValueError(f"La columna «{date_col}» no parece tener fechas: {ilegibles} de {len(df)} valores no se pudieron "
                         "leer como fecha. Revisa que hayas elegido la columna correcta.")
    if ilegibles:
        avisos.append(f"Se ignoraron {ilegibles} filas con fecha ilegible.")
 
    valores = _a_numeros(df[value_col])
    if valores[fechas.notna()].notna().mean() < 0.8:
        raise ValueError(f"La columna «{value_col}» no parece tener números: la mayoría de sus valores no se pudieron "
                         "leer como cantidad. Revisa que hayas elegido la columna correcta.")
 
    d = pd.DataFrame({"date": fechas.dt.normalize(), "v": valores}).dropna(subset=["date"])
    sin_valor = int(d["v"].isna().sum())
    if sin_valor:
        avisos.append(f"{sin_valor} filas no tenían un valor numérico y se tomaron como 0.")
        d["v"] = d["v"].fillna(0)
    negativos = int((d["v"] < 0).sum())
    if negativos:
        avisos.append(f"{negativos} filas tenían ventas negativas y se tomaron como 0.")
        d["v"] = d["v"].clip(lower=0)
 
    s = d.groupby("date")["v"].sum().sort_index()
    pasos = s.index.to_series().diff().dropna().dt.days
    if len(pasos) and pasos.median() >= 5:
        raise ValueError("Tus fechas parecen semanales o mensuales. La página trabaja con datos diarios: "
                         "necesita un registro (o varios que se sumen) por cada día.")
 
    completo = s.asfreq("D")
    faltan = int(completo.isna().sum())
    if faltan:
        if relleno == "interpolar":
            completo = completo.interpolate(limit_direction="both")
            avisos.append(f"Faltaban {faltan} días en el calendario; se estimaron a partir de los días vecinos.")
        else:
            completo = completo.fillna(0)
            avisos.append(f"Faltaban {faltan} días en el calendario; se rellenaron con ventas = 0.")
    if len(completo) < MIN_DIAS:
        raise ValueError(f"Tu archivo cubre solo {len(completo)} días y se necesitan al menos {MIN_DIAS}. "
                         "Sube un historial más largo.")
    if len(completo) < 365:
        avisos.append("Tienes menos de un año de datos: las predicciones pueden ser poco confiables.")
    return completo.astype(float), avisos
 
 
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
            return to_series(read_df(p.read_bytes(), p.name))[0], p.name
        except Exception:
            continue
    return demo_series(), "Datos de ejemplo (sintéticos)"
 
 
def summary(s: pd.Series, name: str, avisos=None) -> dict:
    return {"name": name, "start": str(s.index[0].date()), "end": str(s.index[-1].date()),
            "days": len(s), "total": float(s.sum()), "warnings": avisos or [],
            "dates": [str(d.date()) for d in s.index], "values": [round(float(v), 2) for v in s.values]}
