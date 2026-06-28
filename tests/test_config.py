from pathlib import Path

from exoplanet_classifier.config import load_config
from exoplanet_classifier.features import FEATURE_COLUMNS


def test_load_default_config():
    config = load_config(Path("configs/default.yaml"))

    assert config["data"]["id_column"] == "kepoi_name"
    assert config["data"]["target_column"] == "habitable?"
    assert config["split"]["random_seed"] == 42


def test_feature_list_matches_project_features():
    config = load_config(Path("configs/default.yaml"))

    assert config["features"] == FEATURE_COLUMNS
    assert len(config["features"]) == 15
