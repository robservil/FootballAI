from abc import abstractmethod
from pathlib import Path
import pandas as pd

from datasources.source import DataSource


class ScrapingSource(DataSource):
    """
    Contrato para fuentes que obtienen datos via web scraping.
    Ejecuta el scraping, persiste en CSV y carga desde caché.
    """

    @property
    @abstractmethod
    def cache_path(self) -> Path:
        pass

    @abstractmethod
    def _scrape(self) -> pd.DataFrame:
        pass

    def download(self) -> None:
        df = self._scrape()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.cache_path, index=False)
        print(f"[{self.source_name}] Datos guardados en {self.cache_path}")

    def load(self) -> pd.DataFrame:
        if not self.cache_path.exists():
            print(f"[{self.source_name}] Caché no encontrado. Ejecutando scraping...")
            self.download()
        return pd.read_csv(self.cache_path, parse_dates=["fecha"])