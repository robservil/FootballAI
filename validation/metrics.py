from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import label_binarize

CLASSES = ["1_local", "X_empate", "2_visitante"]


def auc_pr_macro(y_true: pd.Series, probas: np.ndarray) -> float:
    """
    Average Precision macro (AUC-PR macro): métrica de la competición de probabilidades.

    Calcula la AP de cada clase bajo esquema One-vs-Rest y promedia (macro).
    Equivalente a la métrica de Kaggle para la sub-competición de probabilidades.
    """
    y_bin = label_binarize(y_true, classes=CLASSES)
    return float(average_precision_score(y_bin, probas, average="macro"))


def f1_macro(y_true: pd.Series, probas: np.ndarray) -> float:
    """
    F1 macro: métrica de la sub-competición de clasificación.

    Toma la clase con mayor probabilidad como predicción.
    """
    pred = pd.Series([CLASSES[i] for i in probas.argmax(axis=1)], index=y_true.index)
    return float(f1_score(y_true, pred, average="macro"))


def evaluar(y_true: pd.Series, probas: np.ndarray, label: str = "") -> dict[str, float]:
    """Imprime y devuelve AUC-PR macro y F1 macro para un fold."""
    aucpr = auc_pr_macro(y_true, probas)
    f1    = f1_macro(y_true, probas)
    tag   = f"[{label}] " if label else ""
    print(f"    {tag}AUC-PR: {aucpr:.4f}  |  F1: {f1:.4f}")
    return {"auc_pr": aucpr, "f1": f1}
