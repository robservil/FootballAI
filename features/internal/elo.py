# ─────────────────────────────────────────────────────────────
# ELO — dos sistemas de rating por equipo
#
# ELO GLOBAL (home_elo / away_elo):
#   Rating único que se actualiza con TODOS los partidos.
#   Captura la calidad general del equipo.
#
# ELO SPLIT (home_elo_local / away_elo_visit):
#   - home_elo_local: rating del LOCAL basado solo en sus partidos EN CASA
#   - away_elo_visit: rating del VISITANTE basado solo en sus partidos FUERA
#   Captura si un equipo es especialmente fuerte/débil según dónde juega.
#
# Parámetros estándar FiveThirtyEight para fútbol de clubes:
#   K=20, H=65 (ventaja local), R0=1500 (rating inicial)
#
# Anti-leakage: el rating se REGISTRA antes de actualizarlo.
# El ELO persiste entre temporadas.
# ─────────────────────────────────────────────────────────────

import pandas as pd


def add_elo(
    df: pd.DataFrame,
    k: float = 20.0,
    home_advantage: float = 65.0,
    initial_rating: float = 1500.0,
) -> pd.DataFrame:
    df = df.sort_values("fecha").copy()

    # ELO global (todos los partidos)
    global_ratings: dict[str, float] = {}
    # ELO split: solo partidos en casa / solo partidos fuera
    home_ratings:   dict[str, float] = {}
    away_ratings:   dict[str, float] = {}

    home_elos:       list[float] = []
    away_elos:       list[float] = []
    home_elos_local: list[float] = []
    away_elos_visit: list[float] = []

    for row in df.itertuples(index=False):
        home = row.local
        away = row.visitante

        r_home   = global_ratings.get(home, initial_rating)
        r_away   = global_ratings.get(away, initial_rating)
        r_home_h = home_ratings.get(home,   initial_rating)
        r_away_a = away_ratings.get(away,   initial_rating)

        # ── Registrar ANTES de actualizar (anti-leakage) ──────────────────────
        home_elos.append(r_home)
        away_elos.append(r_away)
        home_elos_local.append(r_home_h)
        away_elos_visit.append(r_away_a)

        # ── Resultado real ────────────────────────────────────────────────────
        if row.resultado == "1_local":
            s_home = 1.0
        elif row.resultado == "X_empate":
            s_home = 0.5
        else:
            s_home = 0.0

        # ── Actualizar ELO global ─────────────────────────────────────────────
        e_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home - home_advantage) / 400.0))
        delta  = k * (s_home - e_home)
        global_ratings[home] = r_home + delta
        global_ratings[away] = r_away - delta

        # ── Actualizar ELO split ──────────────────────────────────────────────
        e_home_split = 1.0 / (1.0 + 10.0 ** ((r_away_a - r_home_h - home_advantage) / 400.0))
        delta_split  = k * (s_home - e_home_split)
        home_ratings[home] = r_home_h + delta_split
        away_ratings[away] = r_away_a - delta_split

    df["home_elo"]        = home_elos
    df["away_elo"]        = away_elos
    df["home_elo_local"]  = home_elos_local
    df["away_elo_visit"]  = away_elos_visit

    return df
