import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from models.modelo import Modelo

CLASSES = ["1_local", "X_empate", "2_visitante"]


class XGBoostModelo(Modelo):

    def __init__(self, params: dict | None = None):
        defaults = dict(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0,
        )
        if params:
            defaults.update(params)
        self._model = XGBClassifier(**defaults)
        self._feature_names: list[str] = []
        self._le = LabelEncoder()

    @property
    def nombre(self) -> str:
        return "XGBoost"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = X.columns.tolist()
        y_enc = self._le.fit_transform(y)
        self._model.fit(X, y_enc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # le.classes_ está en orden alfabético: ["1_local","2_visitante","X_empate"]
        # Reordenamos a CLASSES = ["1_local","X_empate","2_visitante"]
        probas = self._model.predict_proba(X)
        xgb_classes = list(self._le.classes_)
        order = [xgb_classes.index(c) for c in CLASSES]
        return probas[:, order]

    def importancia(self) -> pd.Series:
        return (
            pd.Series(self._model.feature_importances_, index=self._feature_names)
            .sort_values(ascending=False)
        )
