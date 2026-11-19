"""Progreso y opciones del análisis en curso (una por hilo), para que la página muestre una barra real."""
import threading
 
_local = threading.local()
 
 
def attach(job: dict, quick: bool = False):
    _local.job, _local.span, _local.quick = job, (0.0, 1.0), quick
 
 
def detach():
    _local.job = None
 
 
def quick() -> bool:
    """True si el usuario pidió el modo rápido para las redes neuronales."""
    return bool(getattr(_local, "quick", False))
 
 
def set_span(a: float, b: float, msg: str = None):
    """Define qué tramo de la barra (de a a b, entre 0 y 1) ocupa el modelo que va a empezar."""
    _local.span = (a, b)
    report(0.0, msg)
 
 
def report(frac: float, msg: str = None):
    """frac = avance (0 a 1) dentro del tramo actual."""
    job = getattr(_local, "job", None)
    if job is None:
        return
    a, b = getattr(_local, "span", (0.0, 1.0))
    frac = max(0.0, min(1.0, frac))
    job["progress"] = max(job.get("progress", 0.0), min(1.0, a + (b - a) * frac))
    if msg:
        job["message"] = msg