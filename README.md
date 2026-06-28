# MLSS Exoplanet Habitability Classifier

This repository contains a cleaned, reproducible version of a graduate-level MLSS tabular machine learning course project. The project predicts a binary exoplanet habitability label from selected Kepler Object of Interest (KOI) features.

The original course notebook is preserved unchanged at:

```text
notebooks/mlss-dacss756-final-project-bdave_v10.ipynb
```

## Problem Statement

The task is binary classification. Given 15 numeric KOI features, the workflow predicts the target column `habitable?`, where:

- `1` comes from a supplied habitable planet list.
- `0` comes from a supplied non-habitable confirmed planet list.

Important scientific caveat: this project predicts a derived supervised label from provided lists. It does not prove biological habitability, detect life, or directly measure whether a planet can support life.

## Expected Raw Data

Raw data is intentionally not committed. Place these files under `data/raw/`:

```text
data/raw/cumulative-exoplanets.xlsx
data/raw/habitable_planets_detailed_list.xlsx
data/raw/non_habitable_planets_confirmed_detailed_list.xlsx
```

The original notebook describes the data as Kepler Object of Interest / KOI cumulative exoplanet data plus separate habitable and non-habitable planet lists.

## Setup on Windows PowerShell

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run Tests

The tests use synthetic data and do not require the real raw Excel files:

```powershell
python -m pytest
```

## Run Training

Smoke-test run using the same workflow with lighter hyperparameter grids:

```powershell
python scripts/train.py --config configs/default.yaml --model all --smoke-test
```

Full bounded run:

```powershell
python scripts/train.py --config configs/default.yaml --model all
```

Train one model:

```powershell
python scripts/train.py --config configs/default.yaml --model logistic_elasticnet
python scripts/train.py --config configs/default.yaml --model svm_rbf
python scripts/train.py --config configs/default.yaml --model random_forest
python scripts/train.py --config configs/default.yaml --model gradient_boosting
```

If `xgboost` is not installed, the XGBoost model is skipped gracefully.

## Outputs

When training is run, outputs are written to:

```text
reports/model_comparison.csv
reports/model_comparison.md
reports/classification_report_<model>.txt
figures/confusion_matrix_<model>.png
figures/roc_curve.png
figures/precision_recall_curve.png
figures/permutation_importance.png
```

Generated zips, raw data, model binaries, and local experiment outputs are ignored by Git.

## Project Structure

```text
configs/                         YAML configuration
data/raw/                        local raw data, ignored except .gitkeep
docs/                            data provenance and scientific caveats
figures/                         generated figures, plus .gitkeep
models/                          optional local model artifacts, ignored except .gitkeep
notebooks/                       preserved original course notebook
reports/current_project_explanation/
                                  early explanatory Markdown/LaTeX report sources
scripts/train.py                 training CLI
src/exoplanet_classifier/        reusable Python package
tests/                           lightweight pytest tests using synthetic data
```

## Current Limitations

- Raw data is not included in the repository.
- Exact provenance/version of the local Excel files is unknown unless documented separately.
- The target label is derived from supplied lists rather than direct biological habitability measurements.
- The positive class is small and imbalanced.
- The original visible notebook test set contains only 22 habitable-class cases.
- Outlier filtering may affect rare cases and should be audited carefully.
- This version focuses on reproducibility and classical ML, not deep learning or complex experiment tracking.

## Future Work

- Add a final enhanced report after the reproducible pipeline has been run.
- Add a documented data acquisition or provenance workflow if licensing allows.
- Add richer model diagnostics and uncertainty discussion.
- Add experiment tracking later only if it remains lightweight and useful.
- Consider model persistence under `models/`, but keep generated model binaries out of Git.
