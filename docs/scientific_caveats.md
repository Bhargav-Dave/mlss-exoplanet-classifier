# Scientific Caveats

This project should be interpreted as a supervised tabular machine learning classifier for a derived label, not as a definitive scientific habitability detector.

## Derived Target

The target column `habitable?` is built from two supplied lists:

- rows in the habitable list are assigned `1`
- rows in the non-habitable list are assigned `0`

The model learns to reproduce this label-list distinction. It does not directly measure biological habitability, life, atmospheric composition, surface water, or biosignatures.

## Possible Target Leakage

If the supplied labels were created using variables similar to the predictors, such as equilibrium temperature, insolation, planet radius, or orbital distance, then the model may learn the label-generation rules rather than independent evidence of habitability.

## Class Imbalance

The visible notebook output shows that the processed data is approximately 94.86% class `0` and 5.14% class `1`. Macro F1 is therefore more appropriate than accuracy alone, but the small positive class still creates uncertainty.

## Small Positive Test Set

The visible notebook classification reports show only 22 habitable-class cases in the test set. A small number of changed predictions can substantially change positive-class precision, recall, and F1.

## Measurement Uncertainty

Exoplanet and stellar properties can have measurement uncertainty. This project currently uses point estimates and does not propagate uncertainty through the classifier.

## Discovery and Selection Bias

Exoplanet catalogs are shaped by detection methods, telescope sensitivity, follow-up practices, and catalog inclusion criteria. The training data may not represent all exoplanets or all potentially habitable planets.

## Outlier Removal

The original notebook removes outliers from `koi_smass` and `koi_prad` using IQR rules. Outlier removal can disproportionately affect rare or scientifically interesting cases, especially when the positive class is small. Future work should audit class balance before and after filtering.
