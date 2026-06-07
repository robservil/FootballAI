# ─────────────────────────────────────────────────────────────
# ESTADO — ¿cómo llega este equipo al partido?
# ─────────────────────────────────────────────────────────────
# Forma CROSS-SEASON: el rolling no se resetea al inicio de temporada.
# Un equipo que terminó mayo con 5 victorias seguidas lleva esa racha
# al primer partido de agosto.
#
# home/away_pts_N{n}        → puntos en los últimos N partidos (cualquier)
# home/away_gf_avg_N{n}     → media goles anotados en los últimos N
# home/away_gc_avg_N{n}     → media goles encajados en los últimos N
# home/away_pts_N{n}_home   → puntos del LOCAL en sus últimos N partidos
#                              jugados EN CASA (contexto específico)
# away/away_pts_N{n}_away   → puntos del VISITANTE en sus últimos N
#                              partidos jugados FUERA (contexto específico)
# home/away_racha_victorias → victorias consecutivas activas (cross-season)
# home/away_racha_sin_perder→ partidos sin perder consecutivos (cross-season)
# home/away_pts_ponderados  → EWM de puntos (más reciente pesa más)
#
# Cold start: cuando un equipo no tiene historial suficiente (ascendido,
# primera temporada en el dataset), se rellena con la mediana del resto
# de equipos de la misma liga-temporada. Nunca se pone 0 arbitrariamente.
# ─────────────────────────────────────────────────────────────

import pandas as pd


def _racha_consec(series: pd.Series) -> pd.Series:
    """
    Para cada posición devuelve cuántos 1s consecutivos hay
    inmediatamente ANTES de esa posición (anti-leakage vía shift).
    """
    shifted = series.shift(fill_value=0).astype(int)
    group_id = (shifted == 0).cumsum()
    return (shifted.groupby(group_id).cumsum() * shifted).astype(int)


def _rolling_split(
    team_df: pd.DataFrame,
    roll_keys: list[str],
    is_home_val: int,
    col: str,
    n: int,
    agg: str = "mean",
) -> pd.Series:
    """
    Calcula rolling(n) SOLO para partidos en casa (is_home_val=1)
    o SOLO fuera (is_home_val=0), en orden cronológico.
    Devuelve una Serie indexada por 'id'.
    """
    sub = (
        team_df[team_df["is_home"] == is_home_val]
        .sort_values(["division", "team", "fecha"])
        .copy()
    )
    fn = (lambda x: x.shift(1).rolling(n, min_periods=1).sum()) if agg == "sum" \
        else (lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    sub["_val"] = sub.groupby(roll_keys)[col].transform(fn)
    return sub.set_index("id")["_val"]


def _fill_cold_start(team_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Rellena NaN con la mediana de liga-temporada (prior para equipos
    sin historial). Segunda pasada con 0 si toda la temporada está vacía.
    """
    for col in cols:
        if col not in team_df.columns:
            continue
        if team_df[col].isna().any():
            med = team_df.groupby(["division", "anio_fin_temporada"])[col].transform("median")
            team_df[col] = team_df[col].fillna(med).fillna(0)
    return team_df


def add_estado(df: pd.DataFrame, team_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    # Agrupar cross-season: sin anio_fin_temporada
    roll_keys = ["division", "team"]

    team_df = (
        team_df.copy()
        .sort_values(["division", "team", "fecha"])
        .reset_index(drop=True)
    )

    # ── Forma general (últimos N, cualquier partido) ──────────────────────────
    team_df[f"pts_N{n}"] = (
        team_df.groupby(roll_keys)["points"]
        .transform(lambda x: x.shift(1).rolling(n, min_periods=1).sum())
    )
    team_df[f"gf_avg_N{n}"] = (
        team_df.groupby(roll_keys)["goals_for"]
        .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    )
    team_df[f"gc_avg_N{n}"] = (
        team_df.groupby(roll_keys)["goals_against"]
        .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    )

    # ── Forma específica de contexto (local/visitante por separado) ───────────
    home_pts_split  = _rolling_split(team_df, roll_keys, 1, "points",       n, "sum")
    home_gf_split   = _rolling_split(team_df, roll_keys, 1, "goals_for",    n, "mean")
    home_gc_split   = _rolling_split(team_df, roll_keys, 1, "goals_against", n, "mean")
    away_pts_split  = _rolling_split(team_df, roll_keys, 0, "points",       n, "sum")
    away_gf_split   = _rolling_split(team_df, roll_keys, 0, "goals_for",    n, "mean")
    away_gc_split   = _rolling_split(team_df, roll_keys, 0, "goals_against", n, "mean")

    team_df = team_df.join(
        pd.DataFrame({
            f"pts_N{n}_home":       home_pts_split,
            f"gf_avg_N{n}_home":    home_gf_split,
            f"gc_avg_N{n}_home":    home_gc_split,
            f"pts_N{n}_away":       away_pts_split,
            f"gf_avg_N{n}_away":    away_gf_split,
            f"gc_avg_N{n}_away":    away_gc_split,
        }),
        on="id",
    )

    # Fallback: si no hay historial en ese contexto, usar forma general
    for split_col, gen_col in [
        (f"pts_N{n}_home",    f"pts_N{n}"),
        (f"gf_avg_N{n}_home", f"gf_avg_N{n}"),
        (f"gc_avg_N{n}_home", f"gc_avg_N{n}"),
        (f"pts_N{n}_away",    f"pts_N{n}"),
        (f"gf_avg_N{n}_away", f"gf_avg_N{n}"),
        (f"gc_avg_N{n}_away", f"gc_avg_N{n}"),
    ]:
        team_df[split_col] = team_df[split_col].fillna(team_df[gen_col])

    # ── Rachas (cross-season) ─────────────────────────────────────────────────
    team_df["racha_victorias"] = (
        team_df.groupby(roll_keys)["win"]
        .transform(_racha_consec)
    )
    team_df["no_loss"] = (team_df["loss"] == 0).astype(int)
    team_df["racha_sin_perder"] = (
        team_df.groupby(roll_keys)["no_loss"]
        .transform(_racha_consec)
    )

    # ── Puntos ponderados EWM (cross-season) ──────────────────────────────────
    team_df["pts_ponderados"] = (
        team_df.groupby(roll_keys)["points"]
        .transform(lambda x: x.shift(fill_value=0).ewm(span=n, min_periods=1).mean())
    )

    # ── Cold start: rellenar NaN con mediana de liga-temporada ────────────────
    form_cols = [
        f"pts_N{n}", f"gf_avg_N{n}", f"gc_avg_N{n}",
        f"pts_N{n}_home", f"gf_avg_N{n}_home", f"gc_avg_N{n}_home",
        f"pts_N{n}_away", f"gf_avg_N{n}_away", f"gc_avg_N{n}_away",
        "racha_victorias", "racha_sin_perder", "pts_ponderados",
    ]
    team_df = _fill_cold_start(team_df, form_cols)

    # ── Merge al df de partidos ───────────────────────────────────────────────
    estado_base = [
        f"pts_N{n}", f"gf_avg_N{n}", f"gc_avg_N{n}",
        "racha_victorias", "racha_sin_perder", "pts_ponderados",
    ]
    split_home = [f"pts_N{n}_home", f"gf_avg_N{n}_home", f"gc_avg_N{n}_home"]
    split_away = [f"pts_N{n}_away", f"gf_avg_N{n}_away", f"gc_avg_N{n}_away"]

    home = (
        team_df[team_df["is_home"] == 1][["id"] + estado_base + split_home]
        .rename(columns={c: f"home_{c}" for c in estado_base + split_home})
    )
    away = (
        team_df[team_df["is_home"] == 0][["id"] + estado_base + split_away]
        .rename(columns={c: f"away_{c}" for c in estado_base + split_away})
    )

    df = df.merge(home, on="id").merge(away, on="id")
    return df
