import pandas as pd
import numpy as np

from models.modelo import Modelo

CLASSES = ["1_local", "X_empate", "2_visitante"]


class EnsembleModelo(Modelo):
    """
    Ensemble de modelos con pesos adaptativos (soft voting ponderado).

    Los pesos se inicializan iguales y se pueden ajustar llamando a
    compute_weights() sobre un conjunto de validación, asignando más
    influencia a los sub-modelos con mayor AUC-PR.
    """

    def __init__(self, modelos: list[Modelo]):
        self._modelos = modelos
        self._pesos: np.ndarray | None = None

    @property
    def nombre(self) -> str:
        nombres = " + ".join(m.nombre for m in self._modelos)
        return f"Ensemble({nombres})"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        for modelo in self._modelos:
            modelo.fit(X, y)

    def compute_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """
        Calcula y fija pesos proporcionales al AUC-PR de cada sub-modelo
        sobre el conjunto de validación proporcionado.
        Los sub-modelos deben estar ya entrenados (fit llamado previamente).
        """
        from validation.metrics import auc_pr_macro
        scores = np.array([
            auc_pr_macro(y_val, m.predict_proba(X_val))
            for m in self._modelos
        ])
        # Normalizar: pesos suman 1
        self._pesos = scores / scores.sum()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = np.stack([m.predict_proba(X) for m in self._modelos], axis=0)
        if self._pesos is not None:
            w = self._pesos[:, np.newaxis, np.newaxis]
            return (probas * w).sum(axis=0)
        return probas.mean(axis=0)

    def importancia(self) -> pd.Series:
        importancias = [m.importancia() for m in self._modelos]
        combined = pd.concat(importancias, axis=1).mean(axis=1)
        return combined.sort_values(ascending=False)
