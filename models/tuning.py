"""
Búsqueda de hiperparámetros con Optuna.

Optimiza el ensemble (LightGBM + CatBoost + XGBoost) maximizando AUC-PR macro
sobre los últimos N folds del walk-forward temporal (sin data leakage).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from models.ensemble import CLASSES
from validation.metrics import auc_pr_macro
import validation.walk_forward as wf

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _predict_ensemble(lgbm_p: dict, cat_p: dict, xgb_p: dict,
                      X_tr: pd.DataFrame, y_tr: pd.Series,
                      X_va: pd.DataFrame) -> np.ndarray:
    le = LabelEncoder()
    y_enc = le.fit_transform(y_tr)

    lgbm = LGBMClassifier(**lgbm_p).fit(X_tr, y_tr)
    cat  = CatBoostClassifier(**cat_p).fit(X_tr, y_tr)
    xgb  = XGBClassifier(**xgb_p).fit(X_tr, y_enc)

    lgbm_ord = [list(lgbm.classes_).index(c) for c in CLASSES]
    cat_ord  = [list(cat.classes_).index(c)  for c in CLASSES]
    xgb_ord  = [list(le.classes_).index(c)   for c in CLASSES]

    return (
        lgbm.predict_proba(X_va)[:, lgbm_ord] +
        cat.predict_proba(X_va)[:, cat_ord]   +
        xgb.predict_proba(X_va)[:, xgb_ord]
    ) / 3.0


def _objective(trial: optuna.Trial, folds_data: list) -> float:
    lgbm_p = dict(
        n_estimators     = trial.suggest_int("lgbm_n_est",   100, 600, step=50),
        learning_rate    = trial.suggest_float("lgbm_lr",    0.01, 0.2, log=True),
        num_leaves       = trial.suggest_int("lgbm_leaves",  15, 127),
        min_child_samples= trial.suggest_int("lgbm_min",     5, 50),
        subsample        = trial.suggest_float("lgbm_sub",   0.6, 1.0),
        colsample_bytree = trial.suggest_float("lgbm_col",   0.6, 1.0),
        random_state=42, verbose=-1,
    )
    cat_p = dict(
        iterations    = trial.suggest_int("cat_iter",  100, 600, step=50),
        learning_rate = trial.suggest_float("cat_lr",  0.01, 0.2, log=True),
        depth         = trial.suggest_int("cat_depth", 4, 10),
        random_seed=42, verbose=0,
    )
    xgb_p = dict(
        n_estimators     = trial.suggest_int("xgb_n_est",  100, 600, step=50),
        learning_rate    = trial.suggest_float("xgb_lr",   0.01, 0.2, log=True),
        max_depth        = trial.suggest_int("xgb_depth",  3, 9),
        subsample        = trial.suggest_float("xgb_sub",  0.6, 1.0),
        colsample_bytree = trial.suggest_float("xgb_col",  0.6, 1.0),
        random_state=42, eval_metric="mlogloss", verbosity=0,
    )

    scores = []
    for X_tr, y_tr, X_va, y_va in folds_data:
        probas = _predict_ensemble(lgbm_p, cat_p, xgb_p, X_tr, y_tr, X_va)
        scores.append(auc_pr_macro(y_va, probas))

    return float(np.mean(scores))


def tune(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str = "resultado",
    n_trials: int = 30,
    n_folds: int = 3,
) -> dict[str, dict]:
    """
    Devuelve un dict con los mejores hiperparámetros por modelo:
    {"lgbm": {...}, "catboost": {...}, "xgboost": {...}}

    Usa los últimos n_folds del walk-forward (expanding) como objetivo.
    """
    all_folds = list(wf.splits(df, min_train_seasons=5))
    target_folds = [
        (
            f[0][feature_cols],
            f[0][target],
            f[1][feature_cols],
            f[1][target],
        )
        for f in all_folds[-n_folds:]
    ]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: _objective(trial, target_folds),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    print(f"  Mejor AUC-PR (Optuna): {study.best_value:.4f}")

    return {
        "lgbm": dict(
            n_estimators      = best["lgbm_n_est"],
            learning_rate     = best["lgbm_lr"],
            num_leaves        = best["lgbm_leaves"],
            min_child_samples = best["lgbm_min"],
            subsample         = best["lgbm_sub"],
            colsample_bytree  = best["lgbm_col"],
            random_state=42, verbose=-1,
        ),
        "catboost": dict(
            iterations    = best["cat_iter"],
            learning_rate = best["cat_lr"],
            depth         = best["cat_depth"],
            random_seed=42, verbose=0,
        ),
        "xgboost": dict(
            n_estimators     = best["xgb_n_est"],
            learning_rate    = best["xgb_lr"],
            max_depth        = best["xgb_depth"],
            subsample        = best["xgb_sub"],
            colsample_bytree = best["xgb_col"],
            random_state=42, eval_metric="mlogloss", verbosity=0,
        ),
    }
