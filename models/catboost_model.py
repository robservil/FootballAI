import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

from models.modelo import Modelo

CLASSES = ["1_local", "X_empate", "2_visitante"]


class CatBoostModelo(Modelo):

    def __init__(self, params: dict | None = None):
        defaults = dict(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
        )
        if params:
            defaults.update(params)
        self._model = CatBoostClassifier(**defaults)
        self._feature_names: list[str] = []

    @property
    def nombre(self) -> str:
        return "CatBoost"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = X.columns.tolist()
        self._model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # CatBoost devuelve las clases en orden alfabético — reordenamos a CLASSES
        cb_classes = list(self._model.classes_)
        probas = self._model.predict_proba(X)
        order = [cb_classes.index(c) for c in CLASSES]
        return probas[:, order]

    def importancia(self) -> pd.Series:
        return (
            pd.Series(self._model.get_feature_importance(), index=self._feature_names)
            .sort_values(ascending=False)
        )
