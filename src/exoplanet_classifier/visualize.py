"""Plotting helpers for evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from .evaluate import get_score_values


def save_confusion_matrix(cm: np.ndarray, model_name: str, figures_dir: str) -> Path:
    """Save a confusion matrix figure."""
    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"confusion_matrix_{model_name}.png"
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot(values_format="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_roc_curve(best_estimator: Any, X_test: pd.DataFrame, y_test: pd.Series, figures_dir: str) -> Path | None:
    """Save ROC curve if a continuous score is available."""
    scores = get_score_values(best_estimator, X_test)
    if scores is None or len(set(y_test)) != 2:
        return None
    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "roc_curve.png"
    RocCurveDisplay.from_predictions(y_test, scores)
    plt.title("ROC Curve: Best Model")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_precision_recall_curve(best_estimator: Any, X_test: pd.DataFrame, y_test: pd.Series, figures_dir: str) -> Path | None:
    """Save precision-recall curve if a continuous score is available."""
    scores = get_score_values(best_estimator, X_test)
    if scores is None or len(set(y_test)) != 2:
        return None
    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "precision_recall_curve.png"
    PrecisionRecallDisplay.from_predictions(y_test, scores)
    plt.title("Precision-Recall Curve: Best Model")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_permutation_importance(
    estimator: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figures_dir: str,
    random_state: int,
) -> Path | None:
    """Save permutation importance for a fitted pipeline if feasible."""
    if X_test.empty:
        return None
    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "permutation_importance.png"
    result = permutation_importance(
        estimator,
        X_test,
        y_test,
        n_repeats=5,
        random_state=random_state,
        scoring="f1_macro",
    )
    importances = pd.Series(result.importances_mean, index=X_test.columns).sort_values()
    importances.plot(kind="barh", figsize=(8, 6))
    plt.xlabel("Mean decrease in macro F1")
    plt.title("Permutation Importance: Best Model")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path
