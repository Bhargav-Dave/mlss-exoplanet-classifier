import numpy as np
import pandas as pd

from exoplanet_classifier.features import FEATURE_COLUMNS
from exoplanet_classifier.models import available_model_specs, make_grid_search
from exoplanet_classifier.preprocessing import make_numeric_preprocessor


def _config():
    return {
        "features": FEATURE_COLUMNS,
        "split": {"random_seed": 42},
        "model_selection": {"cv_folds": 2, "scoring": "f1_macro", "n_jobs": 1},
        "models": {
            "dummy": {"enabled": True, "strategy": "most_frequent"},
            "logistic_elasticnet": {
                "enabled": True,
                "param_grid": {"classifier__C": [1.0], "classifier__l1_ratio": [0.5]},
            },
            "svm_rbf": {"enabled": False, "param_grid": {}},
            "random_forest": {"enabled": False, "param_grid": {}},
            "gradient_boosting": {"enabled": False, "param_grid": {}},
            "xgboost": {"enabled": False, "param_grid": {}},
        },
    }


def _synthetic_xy():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(24, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    X.iloc[0, 0] = np.nan
    y = pd.Series([0, 1] * 12)
    return X, y


def test_preprocessing_pipeline_smoke():
    X, _ = _synthetic_xy()
    preprocessor = make_numeric_preprocessor(FEATURE_COLUMNS, scale=True)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape == (24, len(FEATURE_COLUMNS))
    assert not np.isnan(transformed).any()


def test_model_registry_smoke():
    specs, skipped = available_model_specs(_config(), model_names=["dummy", "logistic_elasticnet"], smoke_test=True)

    assert "dummy" in specs
    assert "logistic_elasticnet" in specs
    assert skipped == []


def test_tiny_model_training_smoke():
    X, y = _synthetic_xy()
    config = _config()
    specs, _ = available_model_specs(config, model_names=["logistic_elasticnet"], smoke_test=True)
    search = make_grid_search(specs["logistic_elasticnet"], config)

    search.fit(X, y)

    assert search.best_estimator_ is not None
    assert "classifier__C" in search.best_params_
