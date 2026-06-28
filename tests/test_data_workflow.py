import pandas as pd

from exoplanet_classifier.data import (
    apply_outlier_rules,
    build_modeling_table,
    make_train_test_split,
    split_features_target,
)
from exoplanet_classifier.features import FEATURE_COLUMNS


def _config():
    return {
        "data": {"id_column": "kepoi_name", "target_column": "habitable?"},
        "features": FEATURE_COLUMNS,
        "preprocessing": {"outlier_rules": {"koi_smass": 1.5, "koi_prad": 2.0}},
        "split": {"test_size": 0.25, "random_seed": 42},
    }


def _tiny_cumulative():
    rows = []
    for index in range(20):
        row = {"kepoi_name": f"K{index:05d}.01"}
        for feature in FEATURE_COLUMNS:
            row[feature] = float(index + 1)
        row["koi_smass"] = 1.0
        row["koi_prad"] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_label_merge_and_split_workflow():
    config = _config()
    cumulative = _tiny_cumulative()
    habitable = pd.DataFrame({"kepoi_name": [f"K{index:05d}.01" for index in range(4)]})
    non_habitable = pd.DataFrame({"kepoi_name": [f"K{index:05d}.01" for index in range(4, 20)]})

    modeling = build_modeling_table(cumulative, habitable, non_habitable, config)
    cleaned = apply_outlier_rules(modeling, config)
    X, y = split_features_target(cleaned, config)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, config)

    assert modeling.shape == (20, 17)
    assert list(X.columns) == FEATURE_COLUMNS
    assert set(y.unique()) == {0, 1}
    assert len(X_train) == 15
    assert len(X_test) == 5
    assert y_train.sum() > 0
    assert y_test.sum() > 0
