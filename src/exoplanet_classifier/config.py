"""Configuration loading and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config file did not contain a mapping: {path}")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate the small set of config keys required by the workflow."""
    required_top_level = ["data", "features", "preprocessing", "split", "model_selection", "models", "outputs"]
    missing = [key for key in required_top_level if key not in config]
    if missing:
        raise ValueError(f"Config is missing required top-level keys: {missing}")

    data_keys = ["cumulative_path", "habitable_path", "non_habitable_path", "id_column", "target_column"]
    missing_data = [key for key in data_keys if key not in config["data"]]
    if missing_data:
        raise ValueError(f"Config data section is missing keys: {missing_data}")

    features = config.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("Config features must be a non-empty list.")


def resolve_repo_path(path_value: str | Path, repo_root: str | Path | None = None) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    root = Path.cwd() if repo_root is None else Path(repo_root)
    return root / path
