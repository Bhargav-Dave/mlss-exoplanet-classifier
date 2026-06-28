"""Evaluation metrics and report table helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_score_values(estimator: Any, X: pd.DataFrame) -> np.ndarray | None:
    """Return continuous scores for ROC/PR metrics when available."""
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)
        if probabilities.shape[1] >= 2:
            return probabilities[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        return np.asarray(scores)
    return None


def evaluate_estimator(name: str, estimator: Any, X_test: pd.DataFrame, y_test: pd.Series, cv_score: float | None = None) -> dict[str, Any]:
    """Compute reusable classification metrics for one fitted estimator."""
    y_pred = estimator.predict(X_test)
    scores = get_score_values(estimator, X_test)

    metrics: dict[str, Any] = {
        "model": name,
        "cv_f1_macro": cv_score,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "roc_auc": None,
        "pr_auc": None,
    }
    if scores is not None and len(set(y_test)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_test, scores)
        metrics["pr_auc"] = average_precision_score(y_test, scores)
    return metrics


def comparison_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a compact model comparison table."""
    columns = [
        "model",
        "cv_f1_macro",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc",
        "pr_auc",
    ]
    table = pd.DataFrame([{column: row.get(column) for column in columns} for row in rows])
    numeric_columns = [column for column in columns if column != "model"]
    for column in numeric_columns:
        if column in table:
            table[column] = table[column].astype(float).round(4)
    return table.sort_values("f1_macro", ascending=False).reset_index(drop=True)


def write_comparison_outputs(table: pd.DataFrame, reports_dir: str) -> None:
    """Write CSV and Markdown model-comparison artifacts."""
    from pathlib import Path

    output_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "model_comparison.csv", index=False)
    (output_dir / "model_comparison.md").write_text(_to_markdown(table), encoding="utf-8")


def _to_markdown(table: pd.DataFrame) -> str:
    """Render a small DataFrame as GitHub-flavored Markdown without extra deps."""
    headers = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        values = ["" if pd.isna(row[column]) else str(row[column]) for column in table.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"
