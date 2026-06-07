import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from lightgbm import LGBMClassifier

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


def select_by_importance(importancias, umbral=0.0):
    """Devuelve features con importancia estrictamente mayor que umbral."""
    return importancias[importancias > umbral].index.tolist()


def rfe_selection(X, y, n_features_to_select=None, step=1):
    """
    Selección recursiva de variables (RFE) usando LightGBM como estimador base.

    Elimina en cada paso la variable menos importante según feature_importances_
    del modelo, hasta quedarse con n_features_to_select (por defecto la mitad).
    Devuelve la lista de features seleccionadas y el objeto RFE ajustado.
    """
    if n_features_to_select is None:
        n_features_to_select = max(1, len(X.columns) // 2)

    estimador = LGBMClassifier(n_estimators=100, learning_rate=0.05,
                               num_leaves=31, random_state=42, verbose=-1)
    selector = RFE(estimator=estimador, n_features_to_select=n_features_to_select,
                   step=step)
    selector.fit(X, y)

    features_sel = X.columns[selector.support_].tolist()
    ranking = pd.Series(selector.ranking_, index=X.columns).sort_values()
    return features_sel, ranking
