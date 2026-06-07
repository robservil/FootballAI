# ─────────────────────────────────────────────────────────────
# CALIDAD — ¿cómo de bueno es este equipo en la temporada?
# ─────────────────────────────────────────────────────────────
# home/away_pts_acum       → puntos acumulados ANTES del partido
# home/away_gd_acum        → diferencia de goles acumulada ANTES del partido
# home/away_wins_acum      → victorias acumuladas ANTES del partido
# home/away_pct_wins       → % victorias ANTES del partido (0 si 0 partidos)
# home/away_posicion       → posición en tabla ANTES del partido (1 = líder)
# home_pts_acum_local      → puntos del local solo en sus partidos en casa
# away_pts_acum_visit      → puntos del visitante solo en sus partidos fuera
# jornada                  → número de partido en la temporada
# ─────────────────────────────────────────────────────────────

import pandas as pd


def pts(df: pd.DataFrame) -> pd.DataFrame:
    df["home_pts"] = df["resultado"].map({
        "1_local": 3,
        "X_empate": 1,
        "2_visitante": 0
    })
    df["away_pts"] = df["resultado"].map({
        "1_local": 0,
        "X_empate": 1,
        "2_visitante": 3
    })
    return df


def add_calidad(df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["division", "anio_fin_temporada", "team"]

    team_df = team_df.copy()

    # ── Acumulados globales (shift garantiza "antes del partido") ─────────────
    team_df["pts_acum"] = (
        team_df.groupby(group_keys)["points"]
        .transform(lambda x: x.cumsum().shift(fill_value=0))
    )
    team_df["gd_acum"] = (
        team_df.groupby(group_keys)["gd"]
        .transform(lambda x: x.cumsum().shift(fill_value=0))
    )
    team_df["wins_acum"] = (
        team_df.groupby(group_keys)["win"]
        .transform(lambda x: x.cumsum().shift(fill_value=0))
    )

    # ── Jornada: nº de partido del equipo dentro de la temporada ─────────────
    team_df["jornada"] = (
        team_df.groupby(group_keys).cumcount() + 1
    )

    # ── % victorias ──────────────────────────────────────────────────────────
    team_df["partidos_jugados"] = team_df["jornada"] - 1
    team_df["pct_wins"] = (
        team_df["wins_acum"]
        / team_df["partidos_jugados"].replace(0, float("nan"))
    ).fillna(0.0)

    # ── Posición en tabla (rank por pts+gd dentro de la jornada) ─────────────
    team_df["rank_score"] = team_df["pts_acum"] * 1000 + team_df["gd_acum"]
    team_df["posicion"] = (
        team_df.groupby(["division", "anio_fin_temporada", "jornada"])["rank_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # ── Puntos acumulados solo como local / solo como visitante ───────────────
    # Calculados únicamente sobre partidos del mismo tipo → shift dentro
    # del subconjunto home/away evita ver el partido actual
    home_mask = team_df["is_home"] == 1
    away_mask = team_df["is_home"] == 0

    team_df.loc[home_mask, "pts_acum_local"] = (
        team_df[home_mask]
        .groupby(group_keys)["points"]
        .transform(lambda x: x.cumsum().shift(fill_value=0))
    )
    team_df.loc[away_mask, "pts_acum_visit"] = (
        team_df[away_mask]
        .groupby(group_keys)["points"]
        .transform(lambda x: x.cumsum().shift(fill_value=0))
    )

    # ── Merge al df de partidos ───────────────────────────────────────────────
    acum_cols = ["pts_acum", "gd_acum", "wins_acum", "pct_wins", "posicion"]

    home = (
        team_df[home_mask][["id", "jornada"] + acum_cols + ["pts_acum_local"]]
        .rename(columns={c: f"home_{c}" for c in acum_cols + ["pts_acum_local"]})
    )
    away = (
        team_df[away_mask][["id"] + acum_cols + ["pts_acum_visit"]]
        .rename(columns={c: f"away_{c}" for c in acum_cols + ["pts_acum_visit"]})
    )

    df = df.merge(home, on="id").merge(away, on="id")
    return df
