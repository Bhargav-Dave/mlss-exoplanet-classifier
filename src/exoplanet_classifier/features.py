"""Feature constants and validation helpers."""

from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "koi_period",
    "koi_ror",
    "koi_srho",
    "koi_prad",
    "koi_sma",
    "koi_teq",
    "koi_insol",
    "koi_dor",
    "koi_count",
    "koi_steff",
    "koi_slogg",
    "koi_smet",
    "koi_srad",
    "koi_smass",
    "koi_eccen",
]


def validate_feature_columns(df: pd.DataFrame, feature_columns: list[str]) -> None:
    """Raise an error if any expected feature is missing."""
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
