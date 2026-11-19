import logging
import threading
import time
import uuid
from pathlib import Path
 
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
 
from models.registry import MODELS, available
from utils import progress
from utils.data import demo_series, load_default, read_df, summary, to_series, vista_previa
from utils.inference import run_forecast
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
app = FastAPI(title="VentIA")
UPLOADS = Path(__file__).parent / "uploads"
UPLOADS.mkdir(exist_ok=True)
STATE = {}
STATE["s"], STATE["name"] = load_default()
STATE["warnings"] = []
JOBS = {}  # análisis en segundo plano: id -> estado, progreso y resultado
 
 
@app.get("/models")
def list_models():
    return [{"id": k, "name": v["name"], "family": v["family"], "desc": v["desc"], "available": available(k),
             "badge": v.get("badge", ""), "speed": v.get("speed", ""), "plain": v.get("plain", v["desc"]),
             "default": v.get("default", False)} for k, v in MODELS.items()]
 
 
@app.get("/dataset")
def get_dataset():
    return summary(STATE["s"], STATE["name"], STATE["warnings"])
 
 
@app.post("/dataset/preview")
async def preview_dataset(file: UploadFile = File(...)):
    """Paso 1 de la carga: lee el archivo y propone qué columnas son la fecha y las ventas."""
    raw = await file.read()
    nombre = file.filename or "archivo"
    try:
        df = read_df(raw, nombre)
        info = vista_previa(df, nombre)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"No se pudo leer el archivo: {e}")
    STATE["pending"] = {"df": df, "name": nombre, "raw": raw}
    return info
 
 
class ConfirmReq(BaseModel):
    date_col: str
    value_col: str
    fill: str = "cero"  # 'cero' | 'interpolar'
 
 
@app.post("/dataset/confirm")
def confirm_dataset(req: ConfirmReq):
    """Paso 2 de la carga: construye la serie con las columnas que eligió el usuario."""
    pend = STATE.get("pending")
    if not pend:
        raise HTTPException(400, "Primero sube un archivo.")
    try:
        series, avisos = to_series(pend["df"], req.date_col, req.value_col, req.fill)
    except ValueError as e:
        raise HTTPException(400, str(e))
    (UPLOADS / Path(pend["name"]).name).write_bytes(pend["raw"])
    STATE["s"], STATE["name"], STATE["warnings"] = series, pend["name"], avisos
    logger.info("Dataset cargado: %s (%d días)", pend["name"], len(series))
    return summary(series, pend["name"], avisos)
 
 
@app.post("/dataset/example")
def use_example():
    """Vuelve a los datos de ejemplo (sintéticos) para probar la página sin subir nada."""
    STATE["s"], STATE["name"], STATE["warnings"] = demo_series(), "Datos de ejemplo (sintéticos)", []
    STATE.pop("pending", None)
    return summary(STATE["s"], STATE["name"], [])
 
 
@app.post("/dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Carga directa con detección automática de columnas (se mantiene por compatibilidad)."""
    raw = await file.read()
    nombre = file.filename or "archivo"
    try:
        series, avisos = to_series(read_df(raw, nombre))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"No se pudo leer el archivo: {e}")
    (UPLOADS / Path(nombre).name).write_bytes(raw)
    STATE["s"], STATE["name"], STATE["warnings"] = series, nombre, avisos
    logger.info("Dataset cargado: %s (%d días)", nombre, len(series))
    return summary(series, nombre, avisos)
 
 
class ForecastReq(BaseModel):
    models: list[str]
    horizon: int = 30
    mode: str = "backtest"    # 'backtest' | 'future'
    country: str = "Ninguno"  # 'Ninguno' | 'CO' | 'EC' | 'US'
    quality: str = "rapida"   # 'rapida' | 'completa' (afecta a las redes neuronales)
 
 
def _check(req: ForecastReq):
    if not req.models:
        raise HTTPException(400, "Elige al menos un modelo.")
    if not 1 <= req.horizon <= 365:
        raise HTTPException(400, "El horizonte debe estar entre 1 y 365 días.")
 
 
@app.post("/forecast")
def forecast(req: ForecastReq):
    """Versión directa (espera a que termine). La página usa /forecast/start para mostrar el progreso."""
    _check(req)
    try:
        return run_forecast(STATE["s"], req.models, req.horizon, req.mode, req.country)
    except ValueError as e:
        raise HTTPException(400, str(e))
 
 
def _run_job(job, series, req):
    progress.attach(job, quick=req.quality != "completa")
    try:
        job["result"] = run_forecast(series, req.models, req.horizon, req.mode, req.country)
        job["progress"], job["message"], job["status"] = 1.0, "Listo", "done"
    except ValueError as e:
        job["error"], job["status"] = str(e), "error"
    except Exception as e:
        logger.exception("Falló el análisis")
        job["error"], job["status"] = f"Ocurrió un error inesperado: {e}", "error"
    finally:
        progress.detach()
 
 
@app.post("/forecast/start")
def forecast_start(req: ForecastReq):
    """Inicia el análisis en segundo plano y devuelve un id para consultar el progreso."""
    _check(req)
    if any(j["status"] == "running" for j in JOBS.values()):
        raise HTTPException(409, "Ya hay un análisis en curso. Espera a que termine.")
    job_id = uuid.uuid4().hex[:12]
    job = {"status": "running", "progress": 0.0, "message": "Preparando los datos…",
           "result": None, "error": None, "started": time.time()}
    JOBS[job_id] = job
    for old in list(JOBS)[:-5]:  # se conservan solo los últimos 5
        JOBS.pop(old, None)
    threading.Thread(target=_run_job, args=(job, STATE["s"], req), daemon=True).start()
    return {"job_id": job_id}
 
 
@app.get("/forecast/status/{job_id}")
def forecast_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "No encontré ese análisis (¿se reinició el backend?). Vuelve a ejecutarlo.")
    return {"status": job["status"], "progress": job["progress"], "message": job["message"],
            "seconds": round(time.time() - job["started"], 1), "result": job["result"], "error": job["error"]}