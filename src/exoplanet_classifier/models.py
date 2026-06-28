"""Model registry and bounded model-selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .preprocessing import make_numeric_preprocessor


@dataclass(frozen=True)
class ModelSpec:
    """A configured estimator and bounded search grid."""

    name: str
    pipeline: Pipeline
    param_grid: dict[str, list[Any]]


def _xgboost_classifier(random_state: int):
    """Return an XGBoost classifier if xgboost is installed, otherwise None."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=1,
    )


def available_model_specs(
    config: dict[str, Any],
    model_names: list[str] | None = None,
    smoke_test: bool = False,
) -> tuple[dict[str, ModelSpec], list[str]]:
    """Build enabled model specs and return skipped model messages."""
    features = list(config["features"])
    random_state = int(config["split"]["random_seed"])
    models_cfg = config["models"]
    requested = set(model_names or models_cfg.keys())
    specs: dict[str, ModelSpec] = {}
    skipped: list[str] = []

    def enabled(name: str) -> bool:
        return name in requested and bool(models_cfg.get(name, {}).get("enabled", False))

    def grid(name: str) -> dict[str, list[Any]]:
        param_grid = dict(models_cfg.get(name, {}).get("param_grid", {}))
        if smoke_test:
            return {key: values[:1] for key, values in param_grid.items()}
        return param_grid

    if enabled("dummy"):
        strategy = models_cfg["dummy"].get("strategy", "most_frequent")
        specs["dummy"] = ModelSpec(
            name="dummy",
            pipeline=Pipeline(
                [
                    ("preprocess", make_numeric_preprocessor(features, scale=False)),
                    ("classifier", DummyClassifier(strategy=strategy)),
                ]
            ),
            param_grid={},
        )

    if enabled("logistic_elasticnet"):
        specs["logistic_elasticnet"] = ModelSpec(
            name="logistic_elasticnet",
            pipeline=Pipeline(
                [
                    ("preprocess", make_numeric_preprocessor(features, scale=True)),
                    (
                        "classifier",
                        LogisticRegression(
                            penalty="elasticnet",
                            solver="saga",
                            max_iter=5000,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            param_grid=grid("logistic_elasticnet"),
        )

    if enabled("logistic_l1"):
        specs["logistic_l1"] = ModelSpec(
            name="logistic_l1",
            pipeline=Pipeline(
                [
                    ("preprocess", make_numeric_preprocessor(features, scale=True)),
                    (
                        "classifier",
                        LogisticRegression(
                            penalty="l1",
                            solver="liblinear",
                            max_iter=2500,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            param_grid=grid("logistic_l1"),
        )

    if enabled("logistic_l2"):
        specs["logistic_l2"] = ModelSpec(
            name="logistic_l2",
            pipeline=Pipeline(
                [
                    ("preprocess", make_numeric_preprocessor(features, scale=True)),
                    (
                        "classifier",
                        LogisticRegression(
                            penalty="l2",
                            solver="liblinear",
                            max_iter=2500,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            param_grid=grid("logistic_l2"),
        )

    if enabled("svm_rbf"):
        specs["svm_rbf"] = ModelSpec(
            name="svm_rbf",
            pipeline=Pipeline(
                [
                    ("preprocess", make_numeric_preprocessor(features, scale=True)),
                    ("classifier", SVC(kernel="rbf", probability=True, random_state=random_state)),
                ]
            ),
            param_grid=grid("svm_rbf"),
        )

    if enabled("svm_linear"):
        specs["svm_linear"] = ModelSpec(
            name="svm_linear",
            pipeline=Pipeline(
                [
                    ("preprocess", make_numeric_preprocessor(features, scale=True)),
                    ("classifier", SVC(kernel="linear", probability=True, random_state=random_state)),
                ]
            ),
            param_grid=grid("svm_linear"),
        )

    if enabled("svm_poly"):
        specs["svm_poly"] = ModelSpec(
            name="svm_poly",
            pipeline=Pipeline(
                [
                    ("preprocess", make_numeric_preprocessor(features, scale=True)),
                    ("classifier", SVC(kernel="poly", probability=True, random_state=random_state)),
                ]
            ),
            param_grid=grid("svm_poly"),
        )

    if enabled("random_forest"):
        specs["random_forest"] = ModelSpec(
            name="random_forest",
            pipeline=Pipeline(
                [
                    ("preprocess", make_numeric_preprocessor(features, scale=False)),
                    ("classifier", RandomForestClassifier(random_state=random_state, n_jobs=-1)),
                ]
            ),
            param_grid=grid("random_forest"),
        )

    if enabled("gradient_boosting"):
        specs["gradient_boosting"] = ModelSpec(
            name="gradient_boosting",
            pipeline=Pipeline(
                [
                    ("preprocess", make_numeric_preprocessor(features, scale=False)),
                    ("classifier", GradientBoostingClassifier(random_state=random_state)),
                ]
            ),
            param_grid=grid("gradient_boosting"),
        )

    if "xgboost" in requested and bool(models_cfg.get("xgboost", {}).get("enabled", False)):
        xgb = _xgboost_classifier(random_state)
        if xgb is None:
            skipped.append("xgboost skipped because the xgboost package is not installed.")
        else:
            specs["xgboost"] = ModelSpec(
                name="xgboost",
                pipeline=Pipeline(
                    [
                        ("preprocess", make_numeric_preprocessor(features, scale=False)),
                        ("classifier", xgb),
                    ]
                ),
                param_grid=grid("xgboost"),
            )

    unknown = sorted(requested - set(models_cfg.keys()))
    for name in unknown:
        skipped.append(f"{name} skipped because it is not defined in config.")

    return specs, skipped


def make_grid_search(spec: ModelSpec, config: dict[str, Any]) -> GridSearchCV:
    """Create a bounded GridSearchCV for one model spec."""
    selection = config["model_selection"]
    cv = StratifiedKFold(
        n_splits=int(selection["cv_folds"]),
        shuffle=True,
        random_state=int(config["split"]["random_seed"]),
    )
    return GridSearchCV(
        estimator=spec.pipeline,
        param_grid=spec.param_grid,
        scoring=selection["scoring"],
        cv=cv,
        refit=True,
        n_jobs=int(selection.get("n_jobs", 1)),
        return_train_score=False,
    )
