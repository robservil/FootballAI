from __future__ import annotations
import pandas as pd
from typing import Iterator


def splits(
    df: pd.DataFrame,
    season_col: str = "anio_fin_temporada",
    min_train_seasons: int = 5,
    window: int | None = None,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, int]]:
    """
    Walk-forward temporal splits. Yields (train_df, val_df, val_season).

    window=None → ventana expansiva: entrena en TODAS las temporadas anteriores.
    window=N    → ventana rodante: entrena solo en las últimas N temporadas.

    Nunca hay datos futuros en el entrenamiento (sin data leakage temporal).
    """
    seasons = sorted(df[season_col].unique())
    for i in range(min_train_seasons, len(seasons)):
        val_season = seasons[i]
        if window is None:
            train_df = df[df[season_col] < val_season]
        else:
            start = max(0, i - window)
            train_seasons = set(seasons[start:i])
            train_df = df[df[season_col].isin(train_seasons)]
        val_df = df[df[season_col] == val_season]
        yield train_df, val_df, val_season
