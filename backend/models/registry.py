"""Catálogo de modelos. Las librerías pesadas se importan solo al usar cada modelo."""
import importlib
import importlib.util


def _lazy(module, func):
    def run(y, idx, H, country=None):
        return getattr(importlib.import_module(f"models.{module}"), func)(y, idx, H, country)
    return run


MODELS = {
    "naive":    dict(name="Naive estacional (7 d)", family="Referencia", needs=None,
                     fn=_lazy("statistical_models", "naive"), desc="Repite la última semana (línea base)."),
    "arima":    dict(name="ARIMA (SARIMAX)", family="Estadísticos", needs="statsmodels",
                     fn=_lazy("statistical_models", "arima"),
                     desc="Selección por AIC entre 4 SARIMA, log1p y exógenas (feriados, quincena, anual)."),
    "prophet":  dict(name="Prophet", family="Estadísticos", needs="prophet",
                     fn=_lazy("statistical_models", "prophet"),
                     desc="Multiplicativo, estacionalidad mensual, regresor de quincena y feriados."),
    "tree":     dict(name="Árbol de decisión", family="Árboles y boosting", needs="sklearn",
                     fn=_lazy("tree_models", "decision_tree"),
                     desc="Rezagos + calendario, búsqueda de hiperparámetros con validación temporal."),
    "xgboost":  dict(name="XGBoost", family="Árboles y boosting", needs="xgboost",
                     fn=_lazy("tree_models", "xgboost"), desc="Boosting ajustado con TimeSeriesSplit."),
    "lightgbm": dict(name="LightGBM", family="Árboles y boosting", needs="lightgbm",
                     fn=_lazy("tree_models", "lightgbm"), desc="Boosting por hojas ajustado con TimeSeriesSplit."),
    "gru":      dict(name="GRU + atención", family="Deep learning", needs="torch",
                     fn=_lazy("trainer_pt", "gru"),
                     desc="GRU de 2 capas, atención temporal, covariables, ensamble de 3 semillas."),
    "cnn_gru":  dict(name="Híbrido CNN-BiGRU", family="Deep learning", needs="torch",
                     fn=_lazy("trainer_pt", "cnn_gru"),
                     desc="CNN multiescala + BiGRU de 2 capas + atención, ensamble de 3 semillas."),
}


def available(mid):
    need = MODELS[mid]["needs"]
    return need is None or importlib.util.find_spec(need) is not None


def get_model(mid):
    return MODELS[mid]["fn"]
