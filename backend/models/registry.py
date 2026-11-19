"""Catálogo de modelos. Las librerías pesadas se importan solo al usar cada modelo."""
import importlib
import importlib.util
 
 
def _lazy(module, func):
    def run(y, idx, H, country=None):
        return getattr(importlib.import_module(f"models.{module}"), func)(y, idx, H, country)
    return run
 
 
# Los textos "plain", "badge", "speed" y "default" se muestran tal cual en la página.
MODELS = {
    "naive": dict(
        name="Naive (referencia)", family="Referencia", needs=None,
        fn=_lazy("statistical_models", "naive"), desc="Repite la última semana (línea base).",
        badge="Referencia", speed="Instantáneo", default=True,
        plain="Predice que cada día se venderá lo mismo que el mismo día de la última semana. No sirve para "
              "decidir: es la vara de medir. Si otro modelo no le gana, no está aportando nada."),
    "lightgbm": dict(
        name="LightGBM", family="Árboles de decisión", needs="lightgbm",
        fn=_lazy("tree_models", "lightgbm"), desc="Boosting por hojas ajustado con TimeSeriesSplit.",
        badge="Rápido", speed="Segundos", default=True,
        plain="Aprende de las ventas pasadas y del calendario (día de la semana, mes, quincena) con muchos "
              "árboles de decisión. No es una red neuronal, pero es muy rápido y suele dar buenos resultados."),
    "gru": dict(
        name="GRU + atención", family="Redes neuronales recurrentes (RNN)", needs="torch",
        fn=_lazy("trainer_pt", "gru"), desc="GRU de 2 capas, atención temporal, covariables, ensamble de semillas.",
        badge="Recomendado", speed="Varios minutos", default=True,
        plain="Red neuronal recurrente: lee tu historial día por día y va guardando en su memoria los patrones "
              "(la semana, la época del año). Con 'atención' se fija más en los días que más importan. "
              "Fue la más precisa en nuestras pruebas."),
    "lstm": dict(
        name="LSTM + atención", family="Redes neuronales recurrentes (RNN)", needs="torch",
        fn=_lazy("trainer_pt", "lstm"), desc="LSTM de 2 capas, atención temporal, covariables, ensamble de semillas.",
        badge="Red neuronal", speed="Varios minutos", default=False,
        plain="Otra red recurrente, con una memoria más elaborada que decide qué recordar y qué olvidar. "
              "Sirve para comparar dos tipos de RNN sobre tus mismos datos."),
}
 
 
def available(mid):
    need = MODELS[mid]["needs"]
    return need is None or importlib.util.find_spec(need) is not None
 
 
def get_model(mid):
    return MODELS[mid]["fn"]