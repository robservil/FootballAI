# ─────────────────────────────────────────────────────────────
# DIFERENCIAS — siempre calculadas al final, sobre columnas ya existentes
# ─────────────────────────────────────────────────────────────
# De calidad:
#   diff_pts_acum      → home_pts_acum  - away_pts_acum
#   diff_gd_acum       → home_gd_acum   - away_gd_acum
#   diff_wins_acum     → home_wins_acum - away_wins_acum
#   diff_pct_wins      → home_pct_wins  - away_pct_wins
#   diff_posicion      → home_posicion  - away_posicion
#   ventaja_local      → home_pts_acum_local - away_pts_acum_visit
# De estado:
#   diff_pts_N{n}          → home - away
#   diff_gf_avg_N{n}       → home - away
#   diff_gc_avg_N{n}       → home - away
#   diff_racha_victorias   → home - away
#   diff_racha_sin_perder  → home - away
#   diff_pts_ponderados    → home - away
# De condiciones:
#   diff_dias_descanso → home - away
# ─────────────────────────────────────────────────────────────

import pandas as pd


def add_diferencias(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    df = df.copy()

    # calidad
    df["diff_pts_acum"]   = df["home_pts_acum"]   - df["away_pts_acum"]
    df["diff_gd_acum"]    = df["home_gd_acum"]    - df["away_gd_acum"]
    df["diff_wins_acum"]  = df["home_wins_acum"]   - df["away_wins_acum"]
    df["diff_pct_wins"]   = df["home_pct_wins"]    - df["away_pct_wins"]
    df["diff_posicion"]   = df["home_posicion"]    - df["away_posicion"]
    # puntos como local vs puntos como visitante: captura la ventaja de campo real
    df["ventaja_local"]   = df["home_pts_acum_local"] - df["away_pts_acum_visit"]

    # estado
    df[f"diff_pts_N{n}"]          = df[f"home_pts_N{n}"]          - df[f"away_pts_N{n}"]
    df[f"diff_gf_avg_N{n}"]       = df[f"home_gf_avg_N{n}"]       - df[f"away_gf_avg_N{n}"]
    df[f"diff_gc_avg_N{n}"]       = df[f"home_gc_avg_N{n}"]       - df[f"away_gc_avg_N{n}"]
    df["diff_racha_victorias"]    = df["home_racha_victorias"]     - df["away_racha_victorias"]
    df["diff_racha_sin_perder"]   = df["home_racha_sin_perder"]    - df["away_racha_sin_perder"]
    df["diff_pts_ponderados"]     = df["home_pts_ponderados"]      - df["away_pts_ponderados"]

    # condiciones
    df["diff_dias_descanso"] = df["home_dias_descanso"] - df["away_dias_descanso"]

    # elo global
    df["diff_elo"]       = df["home_elo"]       - df["away_elo"]
    # elo split (local vs visitante en sus contextos respectivos)
    df["diff_elo_split"] = df["home_elo_local"]  - df["away_elo_visit"]

    # h2h
    df["diff_h2h_win"]   = df["h2h_win_local"]  - df["h2h_win_visit"]

    # forma específica de contexto (local en casa vs visitante fuera)
    df[f"diff_pts_N{n}_split"]    = df[f"home_pts_N{n}_home"]    - df[f"away_pts_N{n}_away"]
    df[f"diff_gf_avg_N{n}_split"] = df[f"home_gf_avg_N{n}_home"] - df[f"away_gf_avg_N{n}_away"]
    df[f"diff_gc_avg_N{n}_split"] = df[f"home_gc_avg_N{n}_home"] - df[f"away_gc_avg_N{n}_away"]

    return df
