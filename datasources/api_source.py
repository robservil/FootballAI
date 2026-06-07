from abc import abstractmethod
from pathlib import Path
import pandas as pd

from datasources.source import DataSource


class APISource(DataSource):
    """
    Contrato para fuentes que obtienen datos via API.
    Llama a la API, persiste en CSV y carga desde caché.
    """

    @property
    @abstractmethod
    def cache_path(self) -> Path:
        pass

    @abstractmethod
    def _call_api(self) -> pd.DataFrame:
        pass

    def download(self) -> None:
        df = self._call_api()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.cache_path, index=False)
        print(f"[{self.source_name}] Datos guardados en {self.cache_path}")

    def load(self) -> pd.DataFrame:
        if not self.cache_path.exists():
            print(f"[{self.source_name}] Caché no encontrado. Descargando...")
            self.download()
        return pd.read_csv(self.cache_path, parse_dates=["fecha"])