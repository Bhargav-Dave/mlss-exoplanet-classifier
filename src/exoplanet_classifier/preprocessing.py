"""Leakage-safe preprocessing components."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_numeric_preprocessor(feature_columns: list[str], scale: bool) -> ColumnTransformer:
    """Create a ColumnTransformer for median imputation and optional scaling."""
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps)
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, feature_columns)],
        remainder="drop",
    )
