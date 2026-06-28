"""Data loading, labeling, cleaning, and splitting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import resolve_repo_path


def expected_raw_paths(config: dict[str, Any], repo_root: str | Path | None = None) -> dict[str, Path]:
    """Return expected raw input paths from config."""
    data_cfg = config["data"]
    return {
        "cumulative": resolve_repo_path(data_cfg["cumulative_path"], repo_root),
        "habitable": resolve_repo_path(data_cfg["habitable_path"], repo_root),
        "non_habitable": resolve_repo_path(data_cfg["non_habitable_path"], repo_root),
    }


def check_raw_files(config: dict[str, Any], repo_root: str | Path | None = None) -> None:
    """Raise a helpful error if any expected raw file is missing."""
    paths = expected_raw_paths(config, repo_root)
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        message = "\n".join(missing)
        raise FileNotFoundError(
            "Required raw data files are missing. Place the three Excel files under data/raw/ "
            "using the paths defined in configs/default.yaml.\nMissing files:\n"
            f"{message}"
        )


def load_raw_data(config: dict[str, Any], repo_root: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the cumulative KOI table and two label-list files."""
    check_raw_files(config, repo_root)
    paths = expected_raw_paths(config, repo_root)
    sheet_name = config["data"].get("cumulative_sheet_name", 0)
    cumulative = pd.read_excel(paths["cumulative"], sheet_name=sheet_name)
    habitable = pd.read_excel(paths["habitable"])
    non_habitable = pd.read_excel(paths["non_habitable"])
    return cumulative, habitable, non_habitable


def select_feature_table(cumulative: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Keep the identifier column and configured feature columns."""
    id_column = config["data"]["id_column"]
    columns = [id_column] + list(config["features"])
    missing = [column for column in columns if column not in cumulative.columns]
    if missing:
        raise ValueError(f"Cumulative data is missing required columns: {missing}")
    return cumulative.loc[:, columns].copy()


def construct_label_table(habitable: pd.DataFrame, non_habitable: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Construct the binary target from the two supplied label-list files."""
    id_column = config["data"]["id_column"]
    target_column = config["data"]["target_column"]
    for name, frame in [("habitable", habitable), ("non_habitable", non_habitable)]:
        if id_column not in frame.columns:
            raise ValueError(f"{name} label data is missing id column: {id_column}")

    habitable_labels = habitable.loc[:, [id_column]].copy()
    habitable_labels[target_column] = 1
    non_habitable_labels = non_habitable.loc[:, [id_column]].copy()
    non_habitable_labels[target_column] = 0
    return pd.concat([habitable_labels, non_habitable_labels], ignore_index=True)


def merge_features_and_labels(feature_table: pd.DataFrame, label_table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Merge labels onto selected KOI features by the configured identifier."""
    id_column = config["data"]["id_column"]
    return pd.merge(feature_table, label_table, on=id_column, how="right")


def build_modeling_table(
    cumulative: pd.DataFrame,
    habitable: pd.DataFrame,
    non_habitable: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build the labeled modeling table from raw DataFrames."""
    feature_table = select_feature_table(cumulative, config)
    labels = construct_label_table(habitable, non_habitable, config)
    return merge_features_and_labels(feature_table, labels, config)


def remove_iqr_outliers(df: pd.DataFrame, column: str, multiplier: float) -> pd.DataFrame:
    """Remove rows outside configurable IQR bounds for one numeric column."""
    if column not in df.columns:
        raise ValueError(f"Cannot remove outliers; column is missing: {column}")
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return df.loc[(df[column] >= lower) & (df[column] <= upper)].copy()


def apply_outlier_rules(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Apply IQR outlier rules in config order."""
    output = df.copy()
    rules = config.get("preprocessing", {}).get("outlier_rules", {})
    for column, multiplier in rules.items():
        output = remove_iqr_outliers(output, column, float(multiplier))
    return output


def split_features_target(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Split a modeling table into X and y."""
    features = list(config["features"])
    target_column = config["data"]["target_column"]
    missing = [column for column in features + [target_column] if column not in df.columns]
    if missing:
        raise ValueError(f"Modeling data is missing required columns: {missing}")
    return df.loc[:, features].copy(), df.loc[:, target_column].copy()


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train-test split using configured settings."""
    split_cfg = config["split"]
    return train_test_split(
        X,
        y,
        test_size=float(split_cfg["test_size"]),
        stratify=y,
        random_state=int(split_cfg["random_seed"]),
    )


def load_prepared_data(
    config: dict[str, Any],
    repo_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Load raw files, build labels, remove configured outliers, and split."""
    cumulative, habitable, non_habitable = load_raw_data(config, repo_root)
    modeling = build_modeling_table(cumulative, habitable, non_habitable, config)
    cleaned = apply_outlier_rules(modeling, config)
    X, y = split_features_target(cleaned, config)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, config)
    return X_train, X_test, y_train, y_test, cleaned
