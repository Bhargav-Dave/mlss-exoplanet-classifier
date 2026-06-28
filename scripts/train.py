"""Train and evaluate configured exoplanet habitability classifiers."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from exoplanet_classifier.config import load_config
from exoplanet_classifier.data import load_prepared_data
from exoplanet_classifier.evaluate import comparison_table, evaluate_estimator, write_comparison_outputs
from exoplanet_classifier.models import available_model_specs, make_grid_search
from exoplanet_classifier.utils import ensure_directory
from exoplanet_classifier.visualize import (
    save_confusion_matrix,
    save_permutation_importance,
    save_precision_recall_curve,
    save_roc_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train exoplanet habitability classifiers.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--model",
        default="all",
        choices=[
            "all",
            "dummy",
            "logistic_elasticnet",
            "logistic_l1",
            "logistic_l2",
            "svm_rbf",
            "svm_linear",
            "svm_poly",
            "random_forest",
            "gradient_boosting",
            "xgboost",
        ],
        help="Model to train. Use all for all enabled models.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Use one hyperparameter value per grid.")
    return parser.parse_args()


def _smoke_config(config: dict) -> dict:
    """Lighten CV while preserving the same workflow."""
    updated = copy.deepcopy(config)
    updated["model_selection"]["cv_folds"] = min(3, int(config["model_selection"]["cv_folds"]))
    return updated


def main() -> int:
    args = parse_args()
    config = load_config(REPO_ROOT / args.config)
    if args.smoke_test:
        config = _smoke_config(config)

    try:
        X_train, X_test, y_train, y_test, cleaned = load_prepared_data(config, REPO_ROOT)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    model_names = None if args.model == "all" else [args.model]
    specs, skipped = available_model_specs(config, model_names=model_names, smoke_test=args.smoke_test)
    for message in skipped:
        print(message)

    if not specs:
        print("No models selected or available to train.", file=sys.stderr)
        return 1

    reports_dir = ensure_directory(REPO_ROOT / config["outputs"]["reports_dir"])
    figures_dir = ensure_directory(REPO_ROOT / config["outputs"]["figures_dir"])

    print(f"Prepared dataset rows after configured cleaning: {len(cleaned)}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Models to train: {', '.join(specs.keys())}")

    evaluations = []
    fitted = {}
    for name, spec in specs.items():
        print(f"Training {name}...")
        search = make_grid_search(spec, config)
        search.fit(X_train, y_train)
        metrics = evaluate_estimator(name, search.best_estimator_, X_test, y_test, cv_score=search.best_score_)
        evaluations.append(metrics)
        fitted[name] = search.best_estimator_
        save_confusion_matrix(metrics["confusion_matrix"], name, figures_dir)
        report_path = reports_dir / f"classification_report_{name}.txt"
        report_path.write_text(metrics["classification_report"], encoding="utf-8")
        print(f"{name}: CV macro F1={search.best_score_:.4f}, test macro F1={metrics['f1_macro']:.4f}")

    table = comparison_table(evaluations)
    write_comparison_outputs(table, reports_dir)

    best_name = table.iloc[0]["model"]
    best_estimator = fitted[best_name]
    save_roc_curve(best_estimator, X_test, y_test, figures_dir)
    save_precision_recall_curve(best_estimator, X_test, y_test, figures_dir)
    save_permutation_importance(
        best_estimator,
        X_test,
        y_test,
        figures_dir,
        random_state=int(config["split"]["random_seed"]),
    )

    print(f"Best model by test macro F1: {best_name}")
    print(f"Wrote model comparison outputs to {reports_dir}")
    print(f"Wrote figures to {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
