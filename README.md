# MLSS Exoplanet Habitability Classifier

A cleaned, reproducible version of my graduate MLSS tabular machine learning course project. The project predicts a binary exoplanet habitability label from selected Kepler Object of Interest (KOI) features, while keeping the original notebook preserved as a historical course artifact.

The main upgrade in this repository is not just adding more models. The goal was to turn a notebook-only project into a structured, testable, reproducible machine learning workflow with clear documentation, scientific caveats, generated evaluation artifacts, and Overleaf-ready reports.

## Project Snapshot

| Item | Current State |
|---|---|
| Task | Binary classification of `habitable?` |
| Data framing | KOI cumulative exoplanet data plus supplied habitable/non-habitable lists |
| Features | 15 numeric KOI/orbital/stellar features |
| Main metric | Macro F1 |
| Split | 80/20 stratified split, `random_state=42` |
| Best enhanced test macro F1 | `0.9648`, tied by Logistic Elastic Net and SVM RBF |
| Best enhanced CV macro F1 | `0.9559`, XGBoost |
| Tests | `7 passed` |
| Raw data | Local-only, ignored by Git |

## Important Scientific Caveat

This project predicts a derived supervised label from supplied lists. It does **not** prove biological habitability, detect life, or directly measure whether a planet can support life.

The model should be read as learning the distinction between the provided habitable and non-habitable label lists. If those lists were created using variables similar to the predictors, the model may partly reproduce the label construction logic. That caveat is central to the scientific interpretation of the results.

## Original Notebook

The original course notebook is preserved unchanged at:

```text
notebooks/mlss-dacss756-final-project-bdave_v10.ipynb
```

The original notebook workflow:

- loads a cumulative KOI-style dataset and two label-list files
- keeps 15 numeric KOI features plus `kepoi_name`
- constructs `habitable?`
- merges labels by `kepoi_name`
- removes outliers from `koi_smass` and `koi_prad`
- uses a stratified train-test split
- compares Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and SVM variants

## Dataset Files Expected

Raw data is intentionally not committed. To run training locally, place these files under `data/raw/`:

```text
data/raw/cumulative-exoplanets.xlsx
data/raw/habitable_planets_detailed_list.xlsx
data/raw/non_habitable_planets_confirmed_detailed_list.xlsx
```

The original notebook describes the data as Kepler Object of Interest / KOI cumulative exoplanet data plus separate habitable and non-habitable planet lists. Exact local Excel provenance/version remains unknown unless documented separately from the files.

## Outcomes

The enhanced pipeline was run successfully after installing dependencies and placing the raw Excel files locally under ignored `data/raw/`.

Validation summary:

- Dependencies installed with `python -m pip install -r requirements.txt`
- Python version: `3.10.5`
- Core imports passed: `numpy`, `pandas`, `sklearn`, `yaml`, `matplotlib`, `pytest`
- XGBoost was available and ran successfully
- Tests passed: `7 passed in 21.13s`
- Smoke training run succeeded
- Full bounded training run succeeded
- No model grids were changed during validation

### Enhanced Pipeline Results

Newly computed enhanced results:

| Model | CV Macro F1 | Test Macro F1 | Accuracy | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Elastic Net | 0.9327 | **0.9648** | 0.9929 | 0.9990 | 0.9809 |
| SVM RBF | 0.9356 | **0.9648** | 0.9929 | 0.9992 | 0.9876 |
| Gradient Boosting | 0.9372 | 0.9540 | 0.9906 | 0.9547 | 0.9527 |
| Random Forest | 0.9557 | 0.9540 | 0.9906 | 0.9967 | 0.9578 |
| XGBoost | **0.9559** | 0.9413 | 0.9882 | 0.9979 | 0.9582 |
| DummyClassifier | 0.4868 | 0.4867 | 0.9482 | 0.5000 | 0.0518 |

What this means:

- Logistic Elastic Net and SVM RBF tied for best enhanced test macro F1 at `0.9648`.
- XGBoost had the best enhanced cross-validated macro F1 at `0.9559`, with Random Forest essentially tied at `0.9557`.
- The DummyClassifier baseline was much worse, which confirms the trained models are doing more than majority-class prediction.
- PR-AUC is important here because the positive class is rare.
- Test performance should be interpreted cautiously because the visible test set contains only 22 habitable-class examples.

### Rare-Class Behavior

The class-specific reports show why macro metrics matter:

- Logistic Elastic Net and SVM RBF both achieved class `1` precision `0.91`, recall `0.95`, and F1 `0.93`.
- Random Forest and Gradient Boosting both achieved class `1` precision `0.88`, recall `0.95`, and F1 `0.91`.
- XGBoost achieved class `1` precision `0.87`, recall `0.91`, and F1 `0.89`.
- DummyClassifier predicted only the majority class and had class `1` precision/recall/F1 of `0.00`.

### Historical Notebook vs Enhanced Pipeline

The enhanced test macro F1 values reproduce the historical notebook ranking closely:

| Model | Historical Test Macro F1 | Enhanced Test Macro F1 |
|---|---:|---:|
| Logistic Regression Elastic Net | 0.9648 | 0.9648 |
| SVM RBF | 0.9648 | 0.9648 |
| Random Forest | 0.9540 | 0.9540 |
| Gradient Boosting | 0.9540 | 0.9540 |
| XGBoost | 0.9413 | 0.9413 |

The enhanced CV values are generally lower than the historical notebook values. That is not a problem by itself: the enhanced workflow uses leakage-safer preprocessing inside pipelines, shuffled `StratifiedKFold`, and smaller bounded grids.

## Generated Artifacts

After training, the project generated:

```text
reports/model_comparison.csv
reports/model_comparison.md
reports/classification_report_<model>.txt
figures/confusion_matrix_<model>.png
figures/roc_curve.png
figures/precision_recall_curve.png
figures/permutation_importance.png
```

The enhanced Overleaf report package is available under:

```text
reports/enhanced_project_overleaf/
```

The generated Overleaf zip is intentionally ignored by Git:

```text
reports/enhanced_project_overleaf/mlss-exoplanet-classifier-enhanced-report.zip
```

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

Smoke-test run:

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
python scripts/train.py --config configs/default.yaml --model xgboost
```

If `xgboost` is not installed, the training registry skips it gracefully.

## Project Structure

```text
configs/                         YAML configuration
data/raw/                        local raw data, ignored except .gitkeep
docs/                            data provenance and scientific caveats
figures/                         generated evaluation figures
models/                          optional local model artifacts, ignored except .gitkeep
notebooks/                       preserved original course notebook
reports/                         generated results and report packages
reports/current_project_explanation/
                                  early explanatory LaTeX report
reports/enhanced_project_overleaf/
                                  final enhanced Overleaf report source
scripts/train.py                 training CLI
src/exoplanet_classifier/        reusable Python package
tests/                           lightweight pytest tests using synthetic data
```

## Current Limitations

- Raw data is not included in the repository.
- Exact provenance/version of the local Excel files is unknown unless documented separately.
- The target label is derived from supplied lists rather than direct biological habitability measurements.
- The positive class is small and imbalanced.
- The visible test set contains only 22 habitable-class cases.
- Outlier filtering may affect rare cases and should be audited carefully.
- There is no CI pipeline yet.
- There is no model card yet.
- This version focuses on reproducibility and classical tabular ML, not deep learning or heavy experiment tracking.

## Future Work

- Add a verified data acquisition script if licensing and source access permit.
- Add a model card and data card.
- Add CI for tests.
- Add threshold tuning and probability calibration.
- Run repeated cross-validation or bootstrap stability checks.
- Add uncertainty-aware analysis if measurement-error information is available.
- Add lightweight experiment tracking later only if it improves the workflow.
