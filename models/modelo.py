from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Modelo(ABC):
    """
    Contrato base para cualquier modelo predictivo.

    Responsabilidades:
    - Aprender de los datos con fit().
    - Predecir probabilidades con predict_proba().
    - Informar del peso que dio a cada feature con importancia().
    """

    @property
    @abstractmethod
    def nombre(self) -> str:
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Entrena el modelo con los datos proporcionados."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Devuelve probabilidades de shape (n_partidos, 3).
        Una columna por clase en el orden de CLASSES:
        [1_local, X_empate, 2_visitante]
        """
        pass

    @abstractmethod
    def importancia(self) -> pd.Series:
        """
        Devuelve el peso que el modelo dio a cada feature.
        Índice: nombre de la feature. Valor: importancia.
        Ordenado de mayor a menor.
        """
        pass