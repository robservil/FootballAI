from abc import ABC, abstractmethod
import pandas as pd


class DataSource(ABC):
    """
    Contrato mínimo para cualquier fuente de datos.
    Solo sabe cargar datos y decir su nombre.
    """

    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def source_name(self) -> str:
        pass