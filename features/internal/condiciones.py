# ─────────────────────────────────────────────────────────────
# CONDICIONES — ¿qué rodea a este partido concreto?
# ─────────────────────────────────────────────────────────────
# home/away_dias_descanso → días desde el último partido del equipo
# ─────────────────────────────────────────────────────────────

import pandas as pd


def add_condiciones(df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    # Ordenar por (division, team, fecha) SIN agrupar por temporada
    # para capturar el descanso real entre el último partido de una
    # temporada y el primero de la siguiente
    cond_df = (
        team_df
        .sort_values(["division", "team", "fecha"])
        .copy()
    )

    # diff() sobre fecha dentro de cada (division, team)
    # El primer partido de cada equipo no tiene anterior → rellenamos con 7 días
    cond_df["dias_descanso"] = (
        cond_df.groupby(["division", "team"])["fecha"]
        .diff()
        .dt.days
        .fillna(7)
    )

    home = (
        cond_df[cond_df["is_home"] == 1][["id", "dias_descanso"]]
        .rename(columns={"dias_descanso": "home_dias_descanso"})
    )
    away = (
        cond_df[cond_df["is_home"] == 0][["id", "dias_descanso"]]
        .rename(columns={"dias_descanso": "away_dias_descanso"})
    )

    df = df.merge(home, on="id").merge(away, on="id")
    return df
