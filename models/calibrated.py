from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from models.ensemble import CLASSES


class CalibratedEnsemble:
    """
    Ensemble con pesos adaptativos + calibración post-hoc.

    Proceso (temporalmente correcto, sin data leakage):
      1. Entrena los sub-modelos en X_base / y_base
      2. Calcula pesos por AUC-PR de cada sub-modelo sobre X_cal
      3. Genera probabilidades ponderadas del ensemble sobre X_cal
      4. Ajusta una IsotonicRegression por clase (OvR) para calibrar la escala
      5. En predict_proba: aplica pesos → calibración → renormalización

    La IsotonicRegression es monotónica: no invierte el ranking, solo
    corrige la escala de las probabilidades (mejora directamente AUC-PR).
    """

    def __init__(self, ensemble):
        self._ensemble = ensemble
        self._calibrators: list[IsotonicRegression] = []

    @property
    def nombre(self) -> str:
        return f"Calibrated({self._ensemble.nombre})"

    def fit(
        self,
        X_base: pd.DataFrame, y_base: pd.Series,
        X_cal:  pd.DataFrame, y_cal:  pd.Series,
    ) -> None:
        # 1. Entrenar sub-modelos
        self._ensemble.fit(X_base, y_base)

        # 2. Calcular pesos adaptativos según AUC-PR en cal
        self._ensemble.compute_weights(X_cal, y_cal)

        # 3. Probabilidades ponderadas sobre cal (usadas para calibrar la escala)
        probas_cal = self._ensemble.predict_proba(X_cal)

        # 4. Calibrador isotónico por clase
        self._calibrators = []
        for i, clase in enumerate(CLASSES):
            y_bin = (y_cal == clase).astype(float).values
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(probas_cal[:, i], y_bin)
            self._calibrators.append(cal)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas_raw = self._ensemble.predict_proba(X)
        calibrated = np.column_stack([
            cal.predict(probas_raw[:, i])
            for i, cal in enumerate(self._calibrators)
        ])
        totals = calibrated.sum(axis=1, keepdims=True)
        totals = np.where(totals == 0, 1.0, totals)
        return calibrated / totals

    def importancia(self) -> pd.Series:
        return self._ensemble.importancia()
