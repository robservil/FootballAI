import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

from models.modelo import Modelo

CLASSES = ["1_local", "X_empate", "2_visitante"]


class LGBMModelo(Modelo):

    def __init__(self, params: dict | None = None):
        defaults = dict(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
        )
        if params:
            defaults.update(params)
        self._model = LGBMClassifier(**defaults)
        self._feature_names: list[str] = []

    @property
    def nombre(self) -> str:
        return "LightGBM"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = X.columns.tolist()
        self._model.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    def importancia(self) -> pd.Series:
        return (
            pd.Series(self._model.feature_importances_, index=self._feature_names)
            .sort_values(ascending=False)
        )
