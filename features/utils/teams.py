import pandas as pd

def create_teams_df(df: pd.DataFrame) -> pd.DataFrame:

    # --------------------------------------------------
    # LOCAL
    # --------------------------------------------------

    home_df = df[
        [
            "id",
            "division",
            "anio_fin_temporada",
            "fecha",
            "local",
            "visitante",
            "goles_local",
            "goles_visitante",
            "resultado"
        ]
    ].copy()

    # renombrar columnas
    home_df = home_df.rename(columns={
        "local": "team",
        "visitante": "opponent",
        "goles_local": "goals_for",
        "goles_visitante": "goals_against"
    })

    # flag local
    home_df["is_home"] = 1

    # resultado
    home_df["win"] = (
        home_df["resultado"] == "1_local"
    ).astype(int)

    home_df["draw"] = (
        home_df["resultado"] == "X_empate"
    ).astype(int)

    home_df["loss"] = (
        home_df["resultado"] == "2_visitante"
    ).astype(int)

    # puntos
    home_df["points"] = (
        home_df["win"] * 3 +
        home_df["draw"]
    )

    # --------------------------------------------------
    # VISITANTE
    # --------------------------------------------------

    away_df = df[
        [
            "id",
            "division",
            "anio_fin_temporada",
            "fecha",
            "visitante",
            "local",
            "goles_visitante",
            "goles_local",
            "resultado"
        ]
    ].copy()

    # renombrar columnas
    away_df = away_df.rename(columns={
        "visitante": "team",
        "local": "opponent",
        "goles_visitante": "goals_for",
        "goles_local": "goals_against"
    })

    # flag visitante
    away_df["is_home"] = 0

    # resultado
    away_df["win"] = (
        away_df["resultado"] == "2_visitante"
    ).astype(int)

    away_df["draw"] = (
        away_df["resultado"] == "X_empate"
    ).astype(int)

    away_df["loss"] = (
        away_df["resultado"] == "1_local"
    ).astype(int)

    # puntos
    away_df["points"] = (
        away_df["win"] * 3 +
        away_df["draw"]
    )

    # --------------------------------------------------
    # UNIR
    # --------------------------------------------------

    teams_df = pd.concat(
        [home_df, away_df],
        ignore_index=True
    )

    # --------------------------------------------------
    # GOAL DIFFERENCE POR PARTIDO
    # --------------------------------------------------

    teams_df["gd"] = teams_df["goals_for"] - teams_df["goals_against"]

    # --------------------------------------------------
    # ORDENAR — ascendente por fecha para que cumsum sea correcto
    # --------------------------------------------------

    teams_df = teams_df.sort_values(
        ["division", "anio_fin_temporada", "team", "fecha"],
        ascending=True
    ).reset_index(drop=True)

    return teams_df