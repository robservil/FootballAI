# datasources/csv_source.py
from pathlib import Path
import pandas as pd

from datasources.source import DataSource


class CSVSource(DataSource):
    """
    Fuente genérica para cualquier fichero CSV local.
    Si no se especifica ruta usa train.csv por defecto.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    @property
    def source_name(self) -> str:
        return self.path.stem      # nombre del fichero sin extensión

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"No se encuentra {self.path}."
            )
        return pd.read_csv(self.path)