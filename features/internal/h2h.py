# ─────────────────────────────────────────────────────────────
# HEAD-TO-HEAD — historial directo entre los dos equipos
# ─────────────────────────────────────────────────────────────
# h2h_n              → nº de encuentros previos entre estos equipos
# h2h_win_local      → % victorias del equipo LOCAL en h2h anteriores
# h2h_win_visit      → % victorias del equipo VISITANTE en h2h anteriores
# h2h_draw_rate      → % empates en h2h anteriores
# h2h_gf_avg_local   → media de goles del LOCAL en h2h anteriores
# h2h_gf_avg_visit   → media de goles del VISITANTE en h2h anteriores
#
# Anti-leakage: las estadísticas se registran ANTES de actualizar con
# el resultado del partido actual.
# Implementación O(n): usa un dict incremental en lugar de filtrar el df.
# ─────────────────────────────────────────────────────────────

import pandas as pd


def add_h2h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("fecha").copy()

    # Para cada par canónico (ordenado alfabéticamente) guardamos:
    # [wins_A, draws, wins_B, gf_A, gf_B, n]
    h2h: dict[tuple, list] = {}

    h2h_n_vals:     list[int]   = []
    win_local_vals: list[float] = []
    win_visit_vals: list[float] = []
    draw_vals:      list[float] = []
    gf_local_vals:  list[float] = []
    gf_visit_vals:  list[float] = []

    for row in df.itertuples(index=False):
        home = row.local
        away = row.visitante

        if home < away:
            key = (home, away)
            home_is_first = True
        else:
            key = (away, home)
            home_is_first = False

        if key not in h2h:
            h2h[key] = [0, 0, 0, 0.0, 0.0, 0]

        stats = h2h[key]
        n = stats[5]

        # ── Registrar ANTES de actualizar (anti-leakage) ─────────────────────
        if n == 0:
            h2h_n_vals.append(0)
            win_local_vals.append(0.333)
            win_visit_vals.append(0.333)
            draw_vals.append(0.333)
            gf_local_vals.append(0.0)
            gf_visit_vals.append(0.0)
        else:
            wins_A, draws, wins_B, gf_A, gf_B, _ = stats
            if home_is_first:
                win_local_vals.append(wins_A / n)
                win_visit_vals.append(wins_B / n)
                gf_local_vals.append(gf_A / n)
                gf_visit_vals.append(gf_B / n)
            else:
                win_local_vals.append(wins_B / n)
                win_visit_vals.append(wins_A / n)
                gf_local_vals.append(gf_B / n)
                gf_visit_vals.append(gf_A / n)
            h2h_n_vals.append(n)
            draw_vals.append(draws / n)

        # ── Actualizar DESPUÉS de registrar ──────────────────────────────────
        resultado      = row.resultado
        goles_local    = row.goles_local
        goles_visitante = row.goles_visitante

        if home_is_first:
            if resultado == "1_local":
                stats[0] += 1          # wins_A (local = A ganó)
            elif resultado == "X_empate":
                stats[1] += 1
            else:
                stats[2] += 1          # wins_B (visitante = B ganó)
            stats[3] += goles_local    # gf_A = goles del local (= A)
            stats[4] += goles_visitante
        else:
            if resultado == "2_visitante":
                stats[0] += 1          # wins_A (visitante = A ganó)
            elif resultado == "X_empate":
                stats[1] += 1
            else:
                stats[2] += 1          # wins_B (local = B ganó)
            stats[3] += goles_visitante  # gf_A = goles del visitante (= A)
            stats[4] += goles_local
        stats[5] += 1

    df["h2h_n"]          = h2h_n_vals
    df["h2h_win_local"]  = win_local_vals
    df["h2h_win_visit"]  = win_visit_vals
    df["h2h_draw_rate"]  = draw_vals
    df["h2h_gf_avg_local"]  = gf_local_vals
    df["h2h_gf_avg_visit"]  = gf_visit_vals

    return df
