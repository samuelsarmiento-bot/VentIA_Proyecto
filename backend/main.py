import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from models.registry import MODELS, available
from utils.data import load_default, read_df, summary, to_series
from utils.inference import run_forecast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VentIA")
UPLOADS = Path(__file__).parent / "uploads"
UPLOADS.mkdir(exist_ok=True)
STATE = {}
STATE["s"], STATE["name"] = load_default()


@app.get("/models")
def list_models():
    return [{"id": k, "name": v["name"], "family": v["family"], "desc": v["desc"], "available": available(k)}
            for k, v in MODELS.items()]


@app.get("/dataset")
def get_dataset():
    return summary(STATE["s"], STATE["name"])


@app.post("/dataset")
async def upload_dataset(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        series = to_series(read_df(raw, file.filename or ""))
    except Exception as e:
        raise HTTPException(400, f"No se pudo leer el archivo: {e}")
    (UPLOADS / Path(file.filename).name).write_bytes(raw)
    STATE["s"], STATE["name"] = series, file.filename
    logger.info("Dataset cargado: %s (%d días)", file.filename, len(series))
    return summary(series, file.filename)


class ForecastReq(BaseModel):
    models: list[str]
    horizon: int = 30
    mode: str = "backtest"   # 'backtest' | 'future'
    country: str = "Ninguno"  # 'Ninguno' | 'CO' | 'EC' | 'US'


@app.post("/forecast")
def forecast(req: ForecastReq):
    if not 1 <= req.horizon <= 365:
        raise HTTPException(400, "El horizonte debe estar entre 1 y 365 días.")
    try:
        return run_forecast(STATE["s"], req.models, req.horizon, req.mode, req.country)
    except ValueError as e:
        raise HTTPException(400, str(e))
