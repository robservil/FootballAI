from __future__ import annotations
import numpy as np
import pandas as pd

from validation.metrics import auc_pr_macro


def permutation_importance(
    modelo,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.Series:
    """
    Importancia por permutación basada en AUC-PR macro.

    Para cada feature: aleatoriza sus valores n_repeats veces y mide la caída
    media en AUC-PR macro respecto a la línea base. Un valor positivo indica
    que la feature contribuye al modelo; negativo indica que introduce ruido.

    No requiere reentrenamiento — es una técnica de caja negra aplicable
    a cualquier modelo que implemente predict_proba().
    """
    rng = np.random.default_rng(random_state)

    probas_base = modelo.predict_proba(X)
    score_base  = auc_pr_macro(y, probas_base)

    importancias: dict[str, float] = {}
    for col in X.columns:
        drops: list[float] = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)
            score_perm = auc_pr_macro(y, modelo.predict_proba(X_perm))
            drops.append(score_base - score_perm)
        importancias[col] = float(np.mean(drops))

    return pd.Series(importancias).sort_values(ascending=False)


def select_by_importance(
    importancias: pd.Series,
    umbral: float = 0.0,
) -> list[str]:
    """
    Devuelve features con importancia estrictamente mayor que umbral.
    Umbral 0.0 elimina las features que no aportan o perjudican el AUC-PR.
    """
    return importancias[importancias > umbral].index.tolist()
