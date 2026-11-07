# VentIA — Proyección de ventas con modelos estadísticos, de árboles y redes neuronales

Backend **FastAPI** + interfaz **Streamlit**, organizados en `backend/` y `frontend/`.

```
VentIA/
├── backend/                    # API (FastAPI)
│   ├── main.py                 # Endpoints: /models, /dataset, /forecast
│   ├── models/
│   │   ├── statistical_models.py   # Naive, ARIMA (SARIMAX), Prophet
│   │   ├── tree_models.py          # Árbol, XGBoost, LightGBM
│   │   ├── pytorch_arch.py         # GRU+atención, CNN-BiGRU
│   │   ├── trainer_pt.py           # Entrenamiento de las redes
│   │   └── registry.py             # Catálogo de modelos
│   ├── utils/                  # data, features, metrics, inference
│   └── uploads/                # Archivos subidos desde la interfaz
├── frontend/
│   └── app.py                  # Interfaz (Streamlit)
├── datasets/                   # Deja aquí tu ZIP/CSV (ej. store-sales-time-series-forecasting.zip)
├── notebooks/                  # Experimento preliminar
├── doc/
├── requirements.txt
└── README.md
```

## Instalación

Usa **Python 3.11 o 3.12** (Prophet y PyTorch pueden no instalarse en versiones más nuevas).

```
# Crear el entorno virtual
> python -m venv .venv

# Activar en Windows
> .\.venv\Scripts\activate

# Activar en Linux o Mac
> source .venv/bin/activate

# Instalar dependencias
> pip install -r requirements.txt
```

## Ejecución

En **dos terminales**, con el entorno activado (el `.venv` se crea en la carpeta raíz `VentIA`):

```
# Terminal 1: backend
> cd backend
> uvicorn main:app --reload --port 8000

# Terminal 2: frontend (desde la carpeta raíz VentIA)
> streamlit run frontend/app.py
```

Abre http://localhost:8501. La documentación de la API está en http://localhost:8000/docs.

## Uso

1. **Datos:** al iniciar usa `datasets/` (primer ZIP/CSV válido) o datos sintéticos de ejemplo. También puedes subir un CSV/ZIP desde el panel izquierdo (columna de fecha y una numérica; si hay `sales` se usa esa; varias filas por fecha se suman).
2. **Modelos:** marca los que quieras comparar.
3. **Configuración:** *Validar* entrena sin los últimos N días y calcula MAE, RMSE, RMSLE y MAPE; *Proyectar* entrena con todo y pronostica N días. Puedes activar feriados de Colombia, Ecuador o EE. UU.

## Modelos

Todos trabajan sobre `log1p` de la serie y calculan sus variables de calendario (día de la semana, mes, día del año, fin de semana, quincena, feriados).

| Modelo | Qué incluye |
|---|---|
| Naive estacional | Repite la última semana (línea base) |
| ARIMA | 4 SARIMA candidatos con estacionalidad semanal; se elige el de menor AIC; exógenas: feriados, quincena, armónicos anuales |
| Prophet | Modo multiplicativo, estacionalidad semanal, anual y mensual, regresor de quincena y feriados del país |
| Árbol / XGBoost / LightGBM | Rezagos (1–28 d), medias y desviaciones móviles, calendario; hiperparámetros por `RandomizedSearchCV` con `TimeSeriesSplit`; pronóstico recursivo |
| GRU + atención | GRU de 2 capas, atención temporal, covariables conocidas del día a predecir, AdamW, pérdida Huber, early stopping y ensamble de 3 semillas |
| Híbrido CNN-BiGRU | Convoluciones multiescala (kernels 3/5/7) → BiGRU de 2 capas → atención, mismo entrenamiento y ensamble |

> Los modelos complejos tardan más (ARIMA, boosting con búsqueda y las redes pueden tomar varios minutos con ~1700 días). Para pruebas rápidas empieza con Naive, Árbol y LightGBM.

## Problemas frecuentes

- **El frontend dice que no conecta con el backend:** inicia primero la terminal 1 (backend).
- **`ModuleNotFoundError: models`/`utils`:** el backend se inicia desde dentro de la carpeta `backend/` (`cd backend`).
- **Falla la instalación de prophet/torch:** usa Python 3.11/3.12. Un modelo sin su librería aparece deshabilitado y el resto sigue funcionando.
- **Cambios que no se ven:** guarda los archivos y reinicia (el backend con `--reload` lo hace solo).
