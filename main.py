"""
Flujo principal del proyecto.
Ejecutar desde la raíz del proyecto: python main.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from datasources.csv_source import CSVSource
from features.utils.teams import create_teams_df
from features.internal.calidad import pts, add_calidad
from features.internal.estado import add_estado
from features.internal.condiciones import add_condiciones
from features.internal.elo import add_elo
from features.internal.h2h import add_h2h
from features.internal.presion import add_presion
from features.derived.diferencias import add_diferencias
from features.selection import permutation_importance, select_by_importance

from models.lgbm import LGBMModelo
from models.catboost_model import CatBoostModelo
from models.xgboost_model import XGBoostModelo
from models.ensemble import EnsembleModelo
from models.calibrated import CalibratedEnsemble
from models.tuning import tune

import validation.walk_forward as wf
import validation.metrics as metricas

# ─────────────────────────────────────────────────────────────────────────────
N_FORM = 5
LIGAS  = ["ESP_LaLiga", "ENG_Premier"]
TARGET = "resultado"

FEATURE_COLS = [
    "home_pts_acum",       "away_pts_acum",
    "home_gd_acum",        "away_gd_acum",
    "home_wins_acum",      "away_wins_acum",
    "home_pct_wins",       "away_pct_wins",
    "home_posicion",       "away_posicion",
    "home_pts_acum_local", "away_pts_acum_visit",
    "jornada",
    "home_dias_descanso",  "away_dias_descanso",
    f"home_pts_N{N_FORM}",      f"away_pts_N{N_FORM}",
    f"home_gf_avg_N{N_FORM}",   f"away_gf_avg_N{N_FORM}",
    f"home_gc_avg_N{N_FORM}",   f"away_gc_avg_N{N_FORM}",
    "home_racha_victorias",      "away_racha_victorias",
    "home_racha_sin_perder",     "away_racha_sin_perder",
    "home_pts_ponderados",       "away_pts_ponderados",
    "diff_pts_acum",    "diff_gd_acum",   "diff_wins_acum",
    "diff_pct_wins",    "diff_posicion",  "ventaja_local",
    f"diff_pts_N{N_FORM}",
    f"diff_gf_avg_N{N_FORM}",
    f"diff_gc_avg_N{N_FORM}",
    "diff_racha_victorias",  "diff_racha_sin_perder",
    "diff_pts_ponderados",   "diff_dias_descanso",
    "home_elo", "away_elo", "diff_elo",
    # elo split (rendimiento específico local/visitante)
    "home_elo_local", "away_elo_visit", "diff_elo_split",
    # head-to-head
    "h2h_n", "h2h_win_local", "h2h_win_visit", "h2h_draw_rate",
    "h2h_gf_avg_local", "h2h_gf_avg_visit", "diff_h2h_win",
    # presión de temporada
    "jornada_ratio",
    "home_zona_descenso", "away_zona_descenso",
    "home_zona_champions", "away_zona_champions",
    "diff_zona_descenso", "diff_zona_champions",
    # forma específica de contexto (local en casa / visitante fuera)
    f"home_pts_N{N_FORM}_home",     f"away_pts_N{N_FORM}_away",
    f"home_gf_avg_N{N_FORM}_home",  f"away_gf_avg_N{N_FORM}_away",
    f"home_gc_avg_N{N_FORM}_home",  f"away_gc_avg_N{N_FORM}_away",
    f"diff_pts_N{N_FORM}_split",
    f"diff_gf_avg_N{N_FORM}_split",
    f"diff_gc_avg_N{N_FORM}_split",
]


def nuevo_ensemble(params: dict | None = None) -> EnsembleModelo:
    if params:
        return EnsembleModelo([
            LGBMModelo(params["lgbm"]),
            CatBoostModelo(params["catboost"]),
            XGBoostModelo(params["xgboost"]),
        ])
    return EnsembleModelo([LGBMModelo(), CatBoostModelo(), XGBoostModelo()])


def build_features(df_liga: pd.DataFrame) -> pd.DataFrame:
    team_df = create_teams_df(df_liga)
    df = pts(df_liga.copy())
    df = add_elo(df)                       # ELO global + split
    df = add_h2h(df)                       # head-to-head cross-temporada
    df = add_calidad(df, team_df)
    df = add_estado(df, team_df, n=N_FORM) # forma cross-season + split casa/fuera
    df = add_condiciones(df, team_df)
    df = add_presion(df)                   # necesita posicion y jornada de calidad
    df = add_diferencias(df, n=N_FORM)

    # Protección cold start global: cualquier NaN numérico residual
    # recibe la mediana de su columna (equipos ascendidos sin historial)
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def wf_calibrado(
    df_liga: pd.DataFrame,
    feature_cols: list[str],
    best_params: dict,
    window: int | None,
    label: str,
) -> tuple[float, float, object, pd.DataFrame, pd.Series]:
    """
    Walk-forward con calibración y modelos afinados.
    Devuelve (media_aucpr, media_f1, ultimo_modelo, ultima_Xval, ultima_yval).
    """
    resultados: list[dict] = []
    last_modelo = None
    last_Xval: pd.DataFrame | None = None
    last_yval: pd.Series | None    = None

    for train_df, val_df, temporada in wf.splits(df_liga, min_train_seasons=5, window=window):
        seasons_tr = sorted(train_df["anio_fin_temporada"].unique())

        if len(seasons_tr) >= 2:
            # Última temporada de train = calibración; el resto = base
            cal_season = seasons_tr[-1]
            base_df = train_df[train_df["anio_fin_temporada"] != cal_season]
            cal_df  = train_df[train_df["anio_fin_temporada"] == cal_season]

            ensemble = nuevo_ensemble(best_params)
            modelo = CalibratedEnsemble(ensemble)
            modelo.fit(
                base_df[feature_cols], base_df[TARGET],
                cal_df[feature_cols],  cal_df[TARGET],
            )
        else:
            modelo = nuevo_ensemble(best_params)
            modelo.fit(train_df[feature_cols], train_df[TARGET])

        X_va = val_df[feature_cols]
        y_va = val_df[TARGET]
        probas = modelo.predict_proba(X_va)

        r = metricas.evaluar(y_va, probas, label=str(temporada))
        r["temporada"] = temporada
        resultados.append(r)

        last_modelo, last_Xval, last_yval = modelo, X_va, y_va

    media_aucpr = float(np.mean([r["auc_pr"] for r in resultados]))
    media_f1    = float(np.mean([r["f1"]     for r in resultados]))
    n = len(resultados)
    print(f"\n  Media {label} ({n} folds) → AUC-PR: {media_aucpr:.4f}  |  F1: {media_f1:.4f}")
    return media_aucpr, media_f1, last_modelo, last_Xval, last_yval


# ═════════════════════════════════════════════════════════════════════════════
# FASE 1 — Desarrollo  (train.csv, sin datos de test)
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("  FASE 1 — Desarrollo  (train.csv exclusivamente)")
print("=" * 72)

df_train_raw = CSVSource(path=Path("data/raw/train.csv")).load()
df_train_raw["fecha"] = pd.to_datetime(df_train_raw["fecha"])

features_por_liga:  dict[str, list[str]] = {}
params_por_liga:    dict[str, dict]      = {}
ventana_por_liga:   dict[str, int | None]= {}

for liga in LIGAS:
    print(f"\n{'─' * 72}")
    print(f"  Liga: {liga}")
    print(f"{'─' * 72}")

    df_liga = df_train_raw[df_train_raw["division"] == liga].copy()
    df_liga = build_features(df_liga)
    n_temp  = df_liga["anio_fin_temporada"].nunique()
    print(f"  {len(df_liga)} partidos  |  {n_temp} temporadas")

    # ── [1/4] Optuna — afinar hiperparámetros ─────────────────────────────────
    print("\n  [1/4] Optuna — búsqueda de hiperparámetros (30 trials, 3 folds)")
    best_params = tune(df_liga, FEATURE_COLS, n_trials=30, n_folds=3)
    params_por_liga[liga] = best_params

    # ── [2/4] Walk-forward expandida vs rodante (ambas calibradas) ────────────
    print("\n  [2/4] Comparativa de estrategia de ventana (calibrado + afinado)")

    print("\n  -- Ventana expansiva --")
    aucpr_exp, f1_exp, mdl_exp, Xval_exp, yval_exp = wf_calibrado(
        df_liga, FEATURE_COLS, best_params, window=None, label="expansiva"
    )

    print("\n  -- Ventana rodante (7 temporadas) --")
    aucpr_rol, f1_rol, mdl_rol, Xval_rol, yval_rol = wf_calibrado(
        df_liga, FEATURE_COLS, best_params, window=7, label="rodante-7"
    )

    if aucpr_exp >= aucpr_rol:
        best_window = None
        last_modelo, last_Xval, last_yval = mdl_exp, Xval_exp, yval_exp
        print(f"\n  Ventana seleccionada: EXPANSIVA  (AUC-PR {aucpr_exp:.4f} vs {aucpr_rol:.4f})")
    else:
        best_window = 7
        last_modelo, last_Xval, last_yval = mdl_rol, Xval_rol, yval_rol
        print(f"\n  Ventana seleccionada: RODANTE-7  (AUC-PR {aucpr_rol:.4f} vs {aucpr_exp:.4f})")

    ventana_por_liga[liga] = best_window

    # ── [3/4] Selección de features — permutation importance ─────────────────
    print("\n  [3/4] Seleccion de features — permutation importance (ultimo fold)")
    importancias = permutation_importance(last_modelo, last_Xval, last_yval, n_repeats=5)
    features_sel = select_by_importance(importancias, umbral=0.0)

    if not features_sel:
        print("  Aviso: ninguna feature supera umbral; se conservan todas.")
        features_sel = FEATURE_COLS[:]

    print(f"\n  Features seleccionadas: {len(features_sel)}/{len(FEATURE_COLS)}")
    print("  Importancia por permutacion (AUC-PR drop):")
    for feat, imp in importancias.items():
        marca = "[+]" if imp > 0 else "[-]"
        print(f"    {marca} {feat:<35} {imp:+.5f}")

    features_por_liga[liga] = features_sel

    # ── [4/4] Walk-forward final (features seleccionadas) ─────────────────────
    win_label = "expansiva" if best_window is None else f"rodante-{best_window}"
    print(f"\n  [4/4] Walk-forward final — {len(features_sel)} features  |  ventana {win_label}")
    aucpr_sel, f1_sel, _, _, _ = wf_calibrado(
        df_liga, features_sel, best_params, window=best_window, label="final"
    )
    best_aucpr = aucpr_exp if best_window is None else aucpr_rol
    print(f"  Delta features  AUC-PR: {aucpr_sel - best_aucpr:+.4f}  |  F1: {f1_sel - (f1_exp if best_window is None else f1_rol):+.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# FASE 2 — Modelo final + submission
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  FASE 2 — Modelo final + submission")
print("=" * 72)

df_test_raw = CSVSource(path=Path("data/raw/test_con_resultados.csv")).load()
df_test_raw["fecha"] = pd.to_datetime(df_test_raw["fecha"])

df_train_raw["_split"] = "train"
df_test_raw["_split"]  = "test"
df_combined = pd.concat([df_train_raw, df_test_raw], ignore_index=True)

all_preds: list[pd.DataFrame] = []

for liga in LIGAS:
    feats       = features_por_liga[liga]
    best_params = params_por_liga[liga]
    print(f"\n  {liga} — {len(feats)} features")

    df_comb_liga = df_combined[df_combined["division"] == liga].copy()
    df_comb_liga = build_features(df_comb_liga)

    train_part = df_comb_liga[df_comb_liga["_split"] == "train"]
    test_part  = df_comb_liga[df_comb_liga["_split"] == "test"]

    # Calibración: usa última temporada de train como conjunto de calibración
    seasons_tr = sorted(train_part["anio_fin_temporada"].unique())
    cal_season  = seasons_tr[-1]
    base_part   = train_part[train_part["anio_fin_temporada"] != cal_season]
    cal_part    = train_part[train_part["anio_fin_temporada"] == cal_season]

    ensemble     = nuevo_ensemble(best_params)
    modelo_final = CalibratedEnsemble(ensemble)
    modelo_final.fit(
        base_part[feats], base_part[TARGET],
        cal_part[feats],  cal_part[TARGET],
    )
    print(f"  Entrenado con {len(train_part)} partidos "
          f"(base: {len(base_part)}, cal: {len(cal_part)})")

    probas = modelo_final.predict_proba(test_part[feats])

    print("  Metricas sobre test (informativas):")
    metricas.evaluar(test_part[TARGET], probas)

    preds_df = pd.DataFrame({
        "id":               test_part["id"].values,
        "1_local_prob":     probas[:, 0],
        "X_empate_prob":    probas[:, 1],
        "2_visitante_prob": probas[:, 2],
    })
    all_preds.append(preds_df)

submission = pd.concat(all_preds).sort_values("id").reset_index(drop=True)
out_path = Path("data/processed/submission.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(out_path, index=False)
print(f"\n  Submission guardado -> {out_path}  ({len(submission)} partidos)")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  RESUMEN FINAL")
print("=" * 72)
for liga in LIGAS:
    win = ventana_por_liga[liga]
    w_str = "expansiva" if win is None else f"rodante-{win}"
    print(f"  {liga}: {len(features_por_liga[liga])}/{len(FEATURE_COLS)} features  |  ventana {w_str}")
print(f"  Submission: {out_path}")
