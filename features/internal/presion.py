# ─────────────────────────────────────────────────────────────
# PRESIÓN DE TEMPORADA — ¿qué hay en juego en este partido?
# ─────────────────────────────────────────────────────────────
# jornada_ratio          → avance de la temporada (0 = inicio, 1 = fin)
# home/away_zona_descenso → 1 si el equipo está en zona de descenso
# home/away_zona_champions→ 1 si el equipo está en zona Champions (top 4)
# diff_zona_descenso     → diferencia de presión por descenso
# diff_zona_champions    → diferencia de presión por Champions
#
# Requiere: home_posicion, away_posicion, jornada (de add_calidad)
# La liga de 20 equipos tiene 3 plazas de descenso (posición ≥ 18)
# y 4 de Champions (posición ≤ 4).
# ─────────────────────────────────────────────────────────────

import pandas as pd


def add_presion(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tamaño dinámico de la liga por temporada (para calcular zona de descenso)
    n_equipos = df.groupby("anio_fin_temporada")["local"].transform("nunique")
    zona_descenso_min = n_equipos - 2        # posición ≥ n-2 → descenso (3 plazas)

    # Avance de temporada: cuántos partidos ha jugado el local vs máximo posible
    max_jornada = df.groupby("anio_fin_temporada")["jornada"].transform("max")
    df["jornada_ratio"] = df["jornada"] / max_jornada

    # Zonas de clasificación
    df["home_zona_descenso"]  = (df["home_posicion"] >= zona_descenso_min).astype(int)
    df["away_zona_descenso"]  = (df["away_posicion"] >= zona_descenso_min).astype(int)
    df["home_zona_champions"] = (df["home_posicion"] <= 4).astype(int)
    df["away_zona_champions"] = (df["away_posicion"] <= 4).astype(int)

    # Diferencias de presión
    df["diff_zona_descenso"]  = df["home_zona_descenso"]  - df["away_zona_descenso"]
    df["diff_zona_champions"] = df["home_zona_champions"] - df["away_zona_champions"]

    return df
